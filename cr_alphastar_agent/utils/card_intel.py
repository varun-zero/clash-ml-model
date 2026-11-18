"""Card intelligence derived from the game's own data tables.

``data/logic.json`` contains the full, shipped card database (effects, units,
buildings, spells and level scaling).  This module turns it into a normalised
card catalogue plus a *data-driven* counter evaluator.

Nothing here is hand-labelled "X counters Y": every score comes from the
actual stats (elixir cost, hitpoints, damage, hit speed, range, splash radius,
air/ground targeting, number of summoned units).  The result is a ranking with
explanations, used by the live bridge to prefer the right answer to the enemy
push it can see.
"""
from __future__ import annotations

import json
import math
import os
import numpy as np
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOGIC_PATH = os.path.join(_PROJECT_ROOT, "data", "logic.json")

#: Engine range units are 1/1000 of a tile (see crsim/engine.py).
_RANGE_PER_TILE = 1000.0
#: Engine conversion: Speed u -> tiles/s = 0.0125 * u (crsim/engine.py:53).
SPEED_TILE_S_PER_UNIT = 0.0125
ARENA_WIDTH_TILES = 18
ARENA_HEIGHT_TILES = 32
#: Default character level used by the simulator.
DEFAULT_LEVEL = 11


@dataclass(frozen=True)
class CardStats:
    """Normalised stats for one playable card."""

    name: str
    cost: float
    hp: float
    damage: float
    hit_speed_s: float
    range_tiles: float
    count: int
    hits_air: bool
    hits_ground: bool
    flying: bool
    splash_tiles: float
    building: bool
    spell: bool
    instant_damage: float
    area_radius_tiles: float
    rarity: str = ""
    icon: str = ""

    @property
    def dps(self) -> float:
        """Total damage per second for the whole summoned group."""
        if self.hit_speed_s <= 0 or self.damage <= 0:
            return 0.0
        return self.damage / self.hit_speed_s * max(1, self.count)

    @property
    def effective_hp(self) -> float:
        return self.hp * max(1, self.count)

    @property
    def is_swarm(self) -> bool:
        return max(1, self.count) >= 3

    @property
    def is_tank(self) -> bool:
        return self.effective_hp >= 1200 and not self.spell

    @property
    def is_support(self) -> bool:
        return (not self.spell and not self.building and self.dps > 80
                and self.effective_hp < 1200)

    @property
    def is_win_condition(self) -> bool:
        return self.building is False and self.range_tiles <= 1.5 and self.speed_hint > 0

    # Set by the loader from the unit's speed field; kept out of the frozen
    # dataclass constructor so equality/hash stay stable.
    speed_hint: float = 0.0
    projectile_speed_tiles_s: float = 0.0


@dataclass
class CounterScore:
    """Why one card is (or is not) a good answer to another."""

    our_card: str
    enemy_card: str
    score: float
    reasons: List[str] = field(default_factory=list)

    @property
    def good(self) -> bool:
        return self.score >= 10.0

    def explain(self) -> str:
        detail = ", ".join(self.reasons) if self.reasons else "neutral matchup"
        return f"{self.our_card} vs {self.enemy_card}: {self.score:+.0f} ({detail})"


@lru_cache(maxsize=1)
def load_logic() -> dict:
    """Load and cache the shipped logic tables."""
    with open(_LOGIC_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _num(value, default=0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value) -> bool:
    return value is True or value == "true" or value == 1


class CardIntel:
    """Normalised card catalogue and counter evaluator."""

    def __init__(self, logic: Optional[dict] = None) -> None:
        self.logic = logic if logic is not None else load_logic()
        self.cards: Dict[str, CardStats] = {}
        self._index_aliases: Dict[str, str] = {}
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        tables = self.logic.get("tables", {})
        units = self.logic.get("characters", {})
        buildings = self.logic.get("buildings", {})

        rows: List[Tuple[dict, bool, bool]] = []
        for table_name, is_building in (("spells_characters", False),
                                        ("spells_buildings", True),
                                        ("spells_other", False)):
            for row in tables.get(table_name, []) or []:
                rows.append((row, is_building, table_name == "spells_other"))

        for row, is_building, is_other in rows:
            name = row.get("Name")
            if not name or _bool(row.get("NotInUse")):
                continue
            summon = row.get("SummonCharacter") or row.get("Stats")
            unit = buildings.get(summon, {}) if is_building else units.get(summon, {})
            if is_other:
                unit = buildings.get(row.get("Stats"), {}) or unit

            dmg = _num(unit.get("Damage"))
            if dmg <= 0:
                dmg = _num(row.get("InstantDamage"))

            # Spell damage is not stored on the card row.  It lives either in
            # the referenced projectile, in an area effect, or in the card's
            # own action script.  Resolve all three so spell matchups get a
            # real damage number instead of a silent zero.
            area_radius = _num(row.get("Radius")) / _RANGE_PER_TILE
            projectile_speed = 0.0
            projectile_name = row.get("Projectile")
            if projectile_name:
                projectile = self.logic.get("projectiles", {}).get(projectile_name, {}) or {}
                # Engine conversion: projectile Speed -> tiles/s = 0.01 * u
                # (crsim/engine.py PROJECTILE_TILE_S_PER_UNIT).
                projectile_speed = _num(projectile.get("Speed")) * 0.01
                if dmg <= 0:
                    dmg = _num(projectile.get("Damage"))
                if dmg > 0:
                    # Waves hit the same area in sequence; MultipleProjectiles
                    # spreads damage across *different* targets, so it must
                    # NOT inflate the per-unit damage used for a 1v1 verdict.
                    waves = max(1, int(_num(row.get("ProjectileWaves"), 1) or 1))
                    dmg *= waves
                if area_radius <= 0:
                    area_radius = _num(projectile.get("Radius")) / _RANGE_PER_TILE
            if dmg <= 0:
                area_effect = row.get("AreaEffectObject")
                effect = (self.logic.get("area_effects", {}).get(area_effect, {}) or {})
                damage_obj = effect.get("Damage")
                if isinstance(damage_obj, dict):
                    dmg = _num(damage_obj.get("BaseDamage"))
                else:
                    dmg = _num(damage_obj)
                if area_radius <= 0:
                    area_radius = _num(effect.get("Radius")) / _RANGE_PER_TILE

            hit_speed_ms = _num(unit.get("HitSpeed"), 1000.0)
            if hit_speed_ms <= 0:
                hit_speed_ms = 1000.0
            hit_range = _num(unit.get("Range")) / _RANGE_PER_TILE
            splash = _num(unit.get("AreaDamageRadius")) / _RANGE_PER_TILE
            if splash <= 0 and area_radius > 0:
                splash = area_radius

            count = int(_num(row.get("SummonNumber"), 1) or 1)
            if count <= 0:
                count = 1

            stats = CardStats(
                name=str(name),
                cost=_num(row.get("ManaCost")),
                hp=_num(unit.get("Hitpoints")),
                damage=dmg,
                hit_speed_s=max(0.1, hit_speed_ms / 1000.0),
                range_tiles=hit_range,
                count=count,
                hits_air=_bool(unit.get("AttacksAir")) or _bool(row.get("CanTargetAir")),
                hits_ground=_bool(unit.get("AttacksGround")) or not _bool(unit.get("AttacksAir")),
                # The shipped data marks air units with FlyingHeight > 0,
                # not with a boolean "Flying" flag.
                flying=(_num(unit.get("FlyingHeight")) > 0
                        or _bool(unit.get("Flying"))
                        or _bool(unit.get("IsFlying"))),
                splash_tiles=max(0.0, splash),
                building=is_building,
                spell=is_other or (dmg > 0 and _num(unit.get("Hitpoints")) <= 0),
                instant_damage=dmg if is_other else _num(row.get("InstantDamage")),
                area_radius_tiles=area_radius,
                rarity=str(row.get("Rarity") or ""),
                icon=str(row.get("IconFile") or ""),
                speed_hint=_num(unit.get("Speed")) * SPEED_TILE_S_PER_UNIT,
                projectile_speed_tiles_s=projectile_speed,
            )
            self.cards[stats.name] = stats
            self._index_aliases[self._key(stats.name)] = stats.name
            if stats.icon:
                self._index_aliases[self._key(stats.icon)] = stats.name

        # Display names used by the live bridge and the config presets.
        from cr_alphastar_agent.config import DECK_NAME_ALIASES
        for display, engine in DECK_NAME_ALIASES.items():
            if engine in self.cards:
                self._index_aliases[self._key(display)] = engine
        for display, engine in {
            "MiniPEKKA": "MiniPekka", "Mini P.E.K.K.A": "MiniPekka",
            "HogRider": "HogRider", "Hog 2.6": "HogRider",
            "Fireball": "Fireball", "Arrows": "Arrows", "Zap": "Zap",
            "Log": "Log", "Barbarian Barrel": "BarbLog",
            "Baby Dragon": "BabyDragon", "Mega Minion": "MegaMinion",
            "Goblin Barrel": "GoblinBarrel", "Inferno Tower": "InfernoTower",
        }.items():
            if engine in self.cards:
                self._index_aliases[self._key(display)] = engine

    @staticmethod
    def _key(name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    # ------------------------------------------------------------------ query
    def get(self, name: str) -> Optional[CardStats]:
        """Resolve a card by engine name, display name, or icon name."""
        if name in self.cards:
            return self.cards[name]
        return self.cards.get(self._index_aliases.get(self._key(name), ""))

    def __len__(self) -> int:
        return len(self.cards)

    def names(self) -> List[str]:
        return sorted(self.cards)

    # ---------------------------------------------------------------- counter
    def counter_score(self, our_card: str, enemy_card: str) -> CounterScore:
        """Score ``our_card`` as an answer to ``enemy_card``.

        Positive means a favourable interaction.  Every term is derived from
        the two cards' shipped stats; the reasons make the decision auditable.
        """
        ours = self.get(our_card)
        theirs = self.get(enemy_card)
        if ours is None or theirs is None:
            return CounterScore(our_card, enemy_card, 0.0, ["unknown card"])

        score = 0.0
        why: List[str] = []

        # 1. Air/ground targeting is a hard constraint.
        if theirs.flying and not ours.hits_air and not ours.spell:
            return CounterScore(our_card, enemy_card, -100.0,
                                ["cannot hit air"])
        if theirs.flying and ours.hits_air:
            score += 18
            why.append("hits air")

        # 2. Spells: lethal damage and value clusters.
        if ours.spell:
            per_unit = max(ours.damage, ours.instant_damage)
            if per_unit >= theirs.hp > 0:
                score += 35
                why.append(f"one-shots ({per_unit:.0f} >= {theirs.hp:.0f} hp)")
            if theirs.is_swarm and ours.area_radius_tiles >= 1.5:
                score += 25
                why.append(f"splash clears {theirs.count} units")
            if theirs.is_support and per_unit >= theirs.hp > 0:
                score += 15
                why.append("removes high-value support")
            if theirs.effective_hp > 1500 and per_unit < theirs.hp:
                score -= 20
                why.append("spell cannot kill tank")
            return CounterScore(our_card, enemy_card, score, why)

        # 3. Direct combat: compare time-to-kill both ways.
        our_dps = ours.dps
        their_dps = theirs.dps
        if our_dps > 0 and theirs.effective_hp > 0:
            ttk_ours = theirs.effective_hp / our_dps
            if their_dps > 0 and ours.effective_hp > 0:
                ttk_theirs = ours.effective_hp / their_dps
                if ttk_ours < ttk_theirs * 0.8:
                    score += 30
                    why.append(f"kills first ({ttk_ours:.1f}s vs {ttk_theirs:.1f}s)")
                elif ttk_ours > ttk_theirs * 1.25:
                    score -= 25
                    why.append(f"dies first ({ttk_ours:.1f}s vs {ttk_theirs:.1f}s)")
        # 4. Splash vs swarms.
        if theirs.is_swarm and ours.splash_tiles >= 1.5:
            score += 28
            why.append(f"splash hits {theirs.count} targets")
        # 5. Range/kiting.
        if ours.range_tiles > theirs.range_tiles + 1.0 and not theirs.flying:
            score += 12
            why.append(f"out-ranges ({ours.range_tiles:.1f} vs {theirs.range_tiles:.1f})")
        # 6. Cheaper answer to an expensive threat.
        if theirs.cost >= ours.cost + 2 and score > 0:
            score += 8
            why.append(f"positive {theirs.cost - ours.cost:.0f} elixir trade")
        # 7. Buildings pull win conditions.
        if ours.building and theirs.cost >= 4 and theirs.range_tiles <= 1.5:
            score += 30
            why.append("building pulls the win condition")
        # 8. Tanks should be answered by high DPS, not by other tanks.
        if theirs.is_tank:
            if ours.is_support and our_dps >= 100:
                score += 20
                why.append(f"high DPS ({our_dps:.0f}) into tank")
            elif ours.is_tank and our_dps < 100:
                score -= 12
                why.append("low-DPS tank vs tank")

        return CounterScore(our_card, enemy_card, float(score), why)

    def candidates_by_kinematics(self, speed_tiles_s: float,
                                 is_air: Optional[bool] = None,
                                 tolerance: float = 0.30) -> List[Tuple[str, float]]:
        """Zero-shot unit typing from observed movement speed.

        ``data/logic.json`` stores each unit's raw ``Speed``.  The simulator
        converts it with ``SPEED_TILE_S_PER_UNIT = 0.0125`` (crsim/engine.py),
        giving distinct speed tiers: Slow ~0.56, Medium ~0.75, Fast ~1.13 and
        Very Fast ~1.5 tiles/s.  Matching the observed velocity against those
        tiers identifies air swarms versus slow tanks even when no sprite
        template has been learned yet.
        """
        speed_tiles_s = max(0.0, float(speed_tiles_s))
        results: List[Tuple[str, float]] = []
        for name, card in self.cards.items():
            if card.spell or card.building or card.speed_hint <= 0:
                continue
            if is_air is not None and card.flying != is_air:
                continue
            delta = abs(card.speed_hint - speed_tiles_s)
            if delta > tolerance:
                continue
            confidence = max(0.0, 1.0 - delta / max(tolerance, 1e-6))
            results.append((name, round(confidence, 3)))
        results.sort(key=lambda item: -item[1])
        return results

    def coarse_role(self, speed_tiles_s: float,
                    is_air: Optional[bool] = None) -> Tuple[str, float]:
        """Coarse but immediate role label for an unseen unit."""
        candidates = self.candidates_by_kinematics(speed_tiles_s, is_air)
        if not candidates:
            return ("unknown", 0.0)
        name, confidence = candidates[0]
        card = self.cards.get(name)
        if card is None:
            return ("unknown", 0.0)
        if card.is_swarm:
            role = "air_swarm" if card.flying else "ground_swarm"
        elif card.is_tank:
            role = "air_tank" if card.flying else "ground_tank"
        elif card.is_support:
            role = "air_support" if card.flying else "ground_support"
        else:
            role = "air_unit" if card.flying else "ground_unit"
        return role, confidence

    def evaluate_play_quality(self, card_name: str, tile_x: float, tile_y: float,
                              enemies: Sequence[Dict[str, float]],
                              under_threat: bool = False,
                              affordable_defense: bool = False) -> Tuple[float, str]:
        """Score how useful a verified play was, for training credit.

        This is deliberately conservative: a play is only punished when the
        frame gives clear evidence it accomplished nothing.

        * spells must actually cover an enemy — an Arrows on empty grass is
          the classic "useless late spell";
        * win conditions played while our tower is under threat and a cheap
          defender is in hand are bad tempo;
        * troops aimed more than 6 tiles from any enemy on an empty board are
          treated as neutral (they may be a legitimate slow roll).
        """
        card = self.get(card_name)
        if card is None:
            return 1.0, "unknown card - neutral"

        if card.spell or card.instant_damage > 0:
            radius = max(0.9, card.area_radius_tiles or 2.0)
            hits = [e for e in enemies
                    if (float(e.get("x", 0)) - tile_x) ** 2
                    + (float(e.get("y", 0)) - tile_y) ** 2 <= radius ** 2]
            if not hits:
                return 0.05, "spell hit nothing (wasted elixir)"
            quality = min(1.0, 0.55 + 0.2 * len(hits))
            return quality, f"spell hit {len(hits)} target(s)"

        if card.cost >= 4 and under_threat and affordable_defense:
            return 0.35, "expensive play while under threat with cheap defence"

        return 1.0, "no negative evidence"

    def spell_lead_target(self, card_name: str, enemy: Dict[str, float],
                          king_pos: Tuple[float, float] = (9.0, 3.0),
                          dispatch_s: float = 0.35) -> Tuple[float, float]:
        """Kinematic interception point for a spell.

        ``enemy`` needs ``x``, ``y`` and optionally ``vx``/``vy`` in tiles per
        second (from :class:`BoardTracker`).  The lead accounts for the ADB
        dispatch latency plus projectile flight, and is clamped so a fast
        unit cannot pull the target absurdly far.
        """
        x = float(enemy.get("x", 0.0))
        y = float(enemy.get("y", 0.0))
        vx = float(enemy.get("vx", 0.0))
        vy = float(enemy.get("vy", 0.0))
        card = self.get(card_name)
        speed = float(card.projectile_speed_tiles_s) if card is not None else 0.0
        if speed <= 0:
            speed = 12.0    # instant area effects (Zap)
        distance = float(np.hypot(x - king_pos[0], y - king_pos[1]))
        flight = min(0.4, distance / speed)
        lead = min(0.85, dispatch_s + flight)
        return (
            float(np.clip(x + vx * lead, 0, ARENA_WIDTH_TILES - 1)),
            float(np.clip(y + vy * lead, 0, ARENA_HEIGHT_TILES - 1)),
        )

    def best_counters(self, our_hand: Sequence[str],
                      enemy_cards: Sequence[str]) -> List[CounterScore]:
        """Rank each of our hand cards by its best answer to the enemy push."""
        results: List[CounterScore] = []
        for ours in our_hand:
            best: Optional[CounterScore] = None
            for enemy in enemy_cards:
                candidate = self.counter_score(ours, enemy)
                if best is None or candidate.score > best.score:
                    best = candidate
            if best is not None:
                results.append(best)
        results.sort(key=lambda r: -r.score)
        return results

    def recommend(self, our_hand: Sequence[str], enemy_cards: Sequence[str],
                  affordable: Optional[Sequence[bool]] = None) -> Optional[CounterScore]:
        """Best affordable answer, or ``None`` when nothing is a counter.

        ``affordable`` is indexed by the **hand slot**, so ranking must keep
        the original slot alongside each score.
        """
        scored = [
            (slot, self.counter_score(card, enemy))
            for slot, card in enumerate(our_hand)
            for enemy in enemy_cards
        ]
        scored.sort(key=lambda pair: -pair[1].score)
        for slot, item in scored:
            if affordable is not None and slot < len(affordable) and not affordable[slot]:
                continue
            if item.good:
                return item
        return None


@lru_cache(maxsize=1)
def get_card_intel() -> CardIntel:
    return CardIntel()
