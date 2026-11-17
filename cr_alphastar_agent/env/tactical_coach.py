"""Tactical coach bot: the behaviour-cloning teacher for the current deck.

The previous scripted bots had three problems that showed up directly in
live matches:

1. **Spell misuse** - Fireball was thrown at the opponent king tower (which
   activates it) and Arrows/Fireball were wasted on empty tiles.
2. **No defence** - the bot kept pushing on the other lane while the
   opponent was destroying its towers.
3. **Repetitive placement** - every troop went to one of three or four fixed
   tiles, so the cloned policy learned exactly that.

This bot fixes all three:

* spells are only cast when ``tactics.evaluate_spell_cast`` says the cast has
  value (never in our own half, never on the enemy king tower, only on real
  targets);
* when a threat is on our half the bot *always* answers with a defensive
  placement and never pushes;
* troop placements are sampled from a spread of logical anchors with jitter
  instead of a handful of fixed tiles.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from cr_alphastar_agent.config import ARENA_HEIGHT, ARENA_WIDTH, CARDS_CATALOG
from cr_alphastar_agent.env.tactics import (
    DEFENSIVE_ANCHORS,
    PLACEMENT_ANCHORS,
    SUPPORT_CARD_IDS,
    TANK_CARD_IDS,
    UnitView,
    best_spell_target,
    card_cost,
    card_hp,
    is_spell,
    nearest_legal_tile,
    perspective,
    sample_placement,
    threat_pressure,
    threats_on_our_side,
    tile_index,
)


def _entities_to_units(obs: Dict[str, np.ndarray]) -> List[UnitView]:
    """Convert the observation entities into tactical unit views."""
    entities = obs["entities"]
    entity_mask = obs["entity_mask"]
    out: List[UnitView] = []
    for i in range(len(entity_mask)):
        if entity_mask[i]:
            continue
        e = entities[i]
        cid = int(round(float(e[0]) * 16.0))
        side = float(e[1])
        hp_ratio = float(e[4])
        out.append(UnitView(
            card_id=cid,
            x=float(e[2]) * ARENA_WIDTH,
            y=float(e[3]) * ARENA_HEIGHT,
            hp=card_hp(cid) * max(0.0, min(1.0, hp_ratio)),
            is_enemy=side < 0.0,
            is_air=float(e[5]) > 0.5,
        ))
    return out


def _affordable_slots(obs: Dict[str, np.ndarray]) -> List[int]:
    card_mask = obs["card_mask"]
    return [s for s in range(4) if bool(card_mask[s])]


class TacticalCoachBot:
    """Defence-first, spell-disciplined, placement-varied bot.

    The elixir thresholds decide how often the bot plays.  The default
    (``push_elixir=9``, ``build_elixir=5``, ``build_probability=0.45``) makes
    the bot play only ~6 % of the decisions, which a cloned policy inherits
    verbatim - it then waits out the whole match.  The ``cycle`` variant
    (see ``TEACHERS``) lowers the thresholds so the agent keeps the pressure
    and cycles its cards.
    """

    def __init__(self, player_id: int = 2, seed: int = 0,
                 push_elixir: float = 9.0,
                 build_elixir: float = 5.0,
                 build_probability: float = 0.45,
                 max_wait_streak: int = 6,
                 prefer_spell: bool = False):
        self.player_id = player_id
        self.team = 0 if int(player_id) <= 1 else 1
        self.rng = random.Random(seed + player_id * 7919)
        self.step_count = 0
        self.push_elixir = float(push_elixir)
        self.build_elixir = float(build_elixir)
        self.build_probability = float(build_probability)
        self.max_wait_streak = int(max_wait_streak)
        # Spell-happy variant: casts at any target, not only when the
        # opponent is broke.
        self.prefer_spell = bool(prefer_spell)
        self._wait_streak = 0

    # ------------------------------------------------------------------ util
    def _hand_card(self, obs: Dict[str, np.ndarray], slot: int) -> int:
        hand = obs["hand_cards"]
        idx = slot if slot < len(hand) else 0
        return int(hand[idx])

    def _spell_slots(self, obs: Dict[str, np.ndarray], slots: List[int]) -> List[int]:
        return [s for s in slots if is_spell(self._hand_card(obs, s))]

    def _troop_slots(self, obs: Dict[str, np.ndarray], slots: List[int]) -> List[int]:
        return [s for s in slots if not is_spell(self._hand_card(obs, s))]

    def _pick_slot(self, slots: List[int], prefer: Optional[set] = None) -> int:
        if prefer:
            preferred = [s for s in slots if s in prefer]
            if preferred:
                return int(self.rng.choice(preferred))
        return int(self.rng.choice(slots))

    def _legal_tile(self, obs: Dict[str, np.ndarray], slot: int,
                    desired_x: float, desired_y: float,
                    min_y: Optional[int] = None) -> Optional[int]:
        """Snap the desired tile onto the closest tile legal for this card.

        Buildings block deployment zones and the engine refuses those
        placements, so a purely geometric anchor is not always playable.
        Without this snap the teacher would emit actions the network cannot
        imitate and behaviour cloning would train against a masked target.

        ``min_y`` restricts the snap to rows at or above the given tile row.
        It is used for spells: snapping a spell onto the closest unmasked
        tile would otherwise happily drop a Fireball *in front of our own
        tower* once the target has already crossed the bridge.
        """
        card_tiles = obs.get("card_tile_mask")
        if card_tiles is None:
            return tile_index(desired_x, desired_y)
        mask = np.asarray(card_tiles[slot], dtype=bool)
        if min_y is not None:
            rows = np.arange(mask.size) // ARENA_WIDTH
            mask = np.logical_and(mask, rows >= int(min_y))
        if not mask.any():
            return None
        return nearest_legal_tile(mask, desired_x, desired_y)

    def _spell_min_row(self) -> int:
        """First canonical row that is *not* in our own half."""
        return int(perspective(self.team).own_half_max_y) + 1

    # ------------------------------------------------------------------ act
    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        self.step_count += 1
        scalars = obs["scalars"]
        own_elixir = float(scalars[0]) * 10.0
        opp_elixir = float(scalars[1]) * 10.0
        slots = _affordable_slots(obs)
        if not slots:
            return np.array([4, 0], dtype=np.int64)

        units = _entities_to_units(obs)
        threats = threats_on_our_side(units, self.team)
        pressure = threat_pressure(units, self.team)

        # ------------------------------------------------------------------
        # 1. DEFENCE FIRST - never push while the opponent is on our half.
        # ------------------------------------------------------------------
        if threats:
            # 1a. If a spell can clear the threat for value, use it.
            spell_slots = self._spell_slots(obs, slots)
            for slot in spell_slots:
                cid = self._hand_card(obs, slot)
                decision = best_spell_target(cid, units, team=self.team)
                if decision is not None and decision.target is not None:
                    tx, ty = decision.target
                    tile = self._legal_tile(obs, slot, tx, ty, min_y=self._spell_min_row())
                    if tile is not None:
                        return np.array([slot, tile], dtype=np.int64)

            # 1b. Otherwise drop a defender behind the threatened tower.
            troop_slots = self._troop_slots(obs, slots)
            if troop_slots:
                threat = max(threats, key=lambda u: (16.0 - u.y))
                # Anchor the answer behind the threatened princess tower.
                lane_x = 3.0 if threat.x < 9.0 else 15.0
                dx, dy = sample_placement(
                    self.rng,
                    defensive=True,
                    lane_x=lane_x,
                    jitter=1.2,
                )
                # A tank in front buys time; a support unit behind survives.
                prefer_tank = {s for s in troop_slots
                               if self._hand_card(obs, s) in TANK_CARD_IDS}
                prefer_support = {s for s in troop_slots
                                  if self._hand_card(obs, s) in SUPPORT_CARD_IDS}
                chosen = self._pick_slot(troop_slots, prefer_tank or prefer_support)
                tile = self._legal_tile(obs, chosen, dx, dy)
                if tile is not None:
                    return np.array([chosen, tile], dtype=np.int64)

        # ------------------------------------------------------------------
        # 2. SPELL VALUE - only cast a spell when it hits something worth it.
        # ------------------------------------------------------------------
        spell_slots = self._spell_slots(obs, slots)
        if spell_slots and (self.prefer_spell or opp_elixir < 6.0):
            for slot in spell_slots:
                cid = self._hand_card(obs, slot)
                decision = best_spell_target(cid, units, team=self.team)
                if decision is not None and decision.target is not None:
                    tx, ty = decision.target
                    tile = self._legal_tile(obs, slot, tx, ty, min_y=self._spell_min_row())
                    if tile is not None:
                        return np.array([slot, tile], dtype=np.int64)

        # ------------------------------------------------------------------
        # 3. PUSH - varied logical anchors instead of three fixed tiles.
        # ------------------------------------------------------------------
        troop_slots = self._troop_slots(obs, slots)
        if troop_slots:
            if own_elixir >= self.push_elixir:
                # Enough elixir: commit a push at the bridge.
                lane_x = 3.0 if self.rng.random() < 0.5 else 15.0
                dx, dy = sample_placement(self.rng, lane_x=lane_x, jitter=1.3)
                chosen = self._pick_slot(troop_slots)
                tile = self._legal_tile(obs, chosen, dx, dy)
                if tile is not None:
                    self._wait_streak = 0
                    return np.array([chosen, tile], dtype=np.int64)

            if own_elixir >= self.build_elixir and self.rng.random() < self.build_probability:
                # Medium elixir: build behind the king / in a flank.
                dx, dy = sample_placement(self.rng, jitter=1.6)
                chosen = self._pick_slot(troop_slots)
                tile = self._legal_tile(obs, chosen, dx, dy)
                if tile is not None:
                    self._wait_streak = 0
                    return np.array([chosen, tile], dtype=np.int64)

        # 4. Bank elixir - but never longer than `max_wait_streak` in a row,
        # otherwise the cloned policy learns to sit out the whole match.
        self._wait_streak += 1
        if self._wait_streak >= self.max_wait_streak and troop_slots:
            # Cycle the cheapest affordable card so the hand keeps moving.
            cheapest = min(troop_slots,
                           key=lambda s: card_cost(self._hand_card(obs, s)))
            dx, dy = sample_placement(self.rng, jitter=1.8)
            tile = self._legal_tile(obs, cheapest, dx, dy)
            if tile is not None:
                self._wait_streak = 0
                return np.array([cheapest, tile], dtype=np.int64)
        return np.array([4, 0], dtype=np.int64)


__all__ = ["TacticalCoachBot", "TEACHERS"]


def _make_cycle_bot(player_id: int = 2, seed: int = 0) -> TacticalCoachBot:
    """Cycle-balanced teacher.

    The default coach plays only ~6 % of the decisions (it waits for 9
    elixir before it commits to a push), and a cloned policy inherits that
    passivity - it waits out the whole match and never pressures.  This
    variant plays at a moderate elixir level and never waits more than a
    couple of decisions in a row.
    """
    return TacticalCoachBot(
        player_id=player_id, seed=seed,
        push_elixir=6.5,
        build_elixir=4.0,
        build_probability=0.85,
        max_wait_streak=2,
    )


# Named teachers for behaviour cloning.
TEACHERS = {
    "coach": TacticalCoachBot,
    "cycle": _make_cycle_bot,
}


# ---------------------------------------------------------------------------
# Opponent variants
# ---------------------------------------------------------------------------
# A league of three fixed bots gives the learner no learning signal: the
# win rate oscillates around 60 % with no trend because there is nothing new
# to beat.  These variants differ in *behaviour*, not just in deck, so the
# learner has to face genuinely different pressure.

def _make_aggro_bot(player_id: int = 2, seed: int = 0) -> TacticalCoachBot:
    """Aggressive opponent: pushes early and keeps the pressure on."""
    return TacticalCoachBot(
        player_id=player_id, seed=seed,
        push_elixir=6.0, build_elixir=3.5, build_probability=0.90,
        max_wait_streak=2,
    )


def _make_defensive_bot(player_id: int = 2, seed: int = 0) -> TacticalCoachBot:
    """Defensive opponent: hoards elixir and only answers real threats."""
    return TacticalCoachBot(
        player_id=player_id, seed=seed,
        push_elixir=10.0, build_elixir=6.0, build_probability=0.20,
        max_wait_streak=8,
    )


def _make_spell_bot(player_id: int = 2, seed: int = 0) -> TacticalCoachBot:
    """Spell-happy opponent: casts spells whenever it finds a target."""
    return TacticalCoachBot(
        player_id=player_id, seed=seed,
        push_elixir=7.0, build_elixir=4.5, build_probability=0.55,
        max_wait_streak=3,
        prefer_spell=True,
    )


# Named opponents used by the league.
OPPONENT_VARIANTS = {
    "coach": TacticalCoachBot,
    "coach_aggro": _make_aggro_bot,
    "coach_defensive": _make_defensive_bot,
    "coach_spell": _make_spell_bot,
}
