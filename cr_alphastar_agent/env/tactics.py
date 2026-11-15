"""Shared tactical rules for the ClashBot training stack.

Everything here is expressed on the canonical 18 x 32 grid used by
``cr_alphastar_agent.env.crsim_replica_env``:

* ``x`` runs 0 (left) .. 17 (right)
* ``y`` runs 0 (our king tower) .. 31 (enemy king tower)
* the river occupies row 16 (engine rows 15.4 .. 16.6)
* our princess towers sit at (3.5, 6.5) and (14.5, 6.5), our king at (9, 3)
* the enemy towers are mirrored at (3.5, 25.5), (14.5, 25.5) and (9, 29)

**Perspective matters.**  Player 2 observes a Y-mirrored arena, so the
canonical frame of seat 1 and seat 2 are not the same picture.  Every helper
that depends on geometry therefore takes the observing ``team`` (0 = blue /
bottom, 1 = red / top) and resolves the anchors through ``perspective()``.

The rules exist because the previous agent exhibited three concrete failures
in live matches:

1. it cast Fireball / Arrows right in front of its own tower,
2. it kept pushing while the opponent was chewing on its towers,
3. it always placed troops on the same three or four tiles.

The helpers below encode the counter-measures so the scripted teacher, the
behaviour-cloning data and the RL reward all agree on what "good" means.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cr_alphastar_agent.config import (
    ARENA_HEIGHT, ARENA_WIDTH, CARDS_CATALOG, SPELL_CARD_IDS,
    WIN_CONDITION_IDS, DEFENSIVE_CARD_IDS
)

# Engine frame constants (mirrored from crsim.engine).
RIVER_Y = 16.0
RIVER_HALF = 0.6

# Cards that are worth protecting / sniping with a spell (support units).
SUPPORT_CARD_IDS = frozenset({2, 4, 12, 15})  # Archer(s), Musketeer, BabyDragon, MegaMinion
TANK_CARD_IDS = frozenset({1, 3, 5, 10, 11})  # Knight, Giant, MiniPekka, Valkyrie, HogRider


@dataclass(frozen=True)
class Perspective:
    """Geometry of the canonical frame for one seat.

    ``own_half_max_y``  - highest canonical row that still belongs to us
    ``enemy_half_min_y`` - lowest canonical row that belongs to the opponent
    """

    team: int
    own_half_max_y: float
    enemy_half_min_y: float
    own_king: Tuple[float, float]
    enemy_king: Tuple[float, float]
    own_princess: Tuple[Tuple[float, float], Tuple[float, float]]
    enemy_princess: Tuple[Tuple[float, float], Tuple[float, float]]

    def is_own_half(self, y: float) -> bool:
        return y <= self.own_half_max_y

    def is_enemy_half(self, y: float) -> bool:
        return y >= self.enemy_half_min_y


# Player 1 sees the engine frame directly; player 2 sees it mirrored
# (canonical_y = 31 - engine_y), which shifts every anchor by one row.
_PERSPECTIVES = {
    0: Perspective(
        team=0,
        own_half_max_y=RIVER_Y - RIVER_HALF,      # 15.4
        enemy_half_min_y=RIVER_Y + RIVER_HALF,     # 16.6
        own_king=(9.0, 3.0),
        enemy_king=(9.0, 29.0),
        own_princess=((3.5, 6.5), (14.5, 6.5)),
        enemy_princess=((3.5, 25.5), (14.5, 25.5)),
    ),
    1: Perspective(
        team=1,
        own_half_max_y=ARENA_HEIGHT - 1 - (RIVER_Y + RIVER_HALF),  # 14.4
        enemy_half_min_y=ARENA_HEIGHT - 1 - (RIVER_Y - RIVER_HALF),  # 15.6
        own_king=(9.0, 2.0),
        enemy_king=(9.0, 28.0),
        own_princess=((3.5, 5.5), (14.5, 5.5)),
        enemy_princess=((3.5, 24.5), (14.5, 24.5)),
    ),
}

# Legacy aliases (player-1 canonical frame).
FRIENDLY_KING = _PERSPECTIVES[0].own_king
ENEMY_KING = _PERSPECTIVES[0].enemy_king
FRIENDLY_PRINCESS = _PERSPECTIVES[0].own_princess
ENEMY_PRINCESS = _PERSPECTIVES[0].enemy_princess


def perspective(team: int) -> Perspective:
    """Return the canonical-frame geometry for the given seat."""
    return _PERSPECTIVES[1 if int(team) else 0]


# Spell metadata per policy card id:
#   (cost, radius in tiles, damage, minimum cluster size, support hit is value)
SPELL_STATS: Dict[int, Tuple[float, float, float, int, bool]] = {
    8: (4.0, 2.5, 570.0, 2, True),    # Fireball
    9: (3.0, 4.0, 300.0, 2, False),   # Arrows
    14: (3.0, 1.0, 95.0, 1, False),   # Goblin Barrel (spawn spell)
    16: (2.0, 2.5, 160.0, 2, False),  # Zap
}


def is_spell(card_id: int) -> bool:
    return int(card_id) in SPELL_CARD_IDS


def is_win_condition(card_id: int) -> bool:
    return int(card_id) in WIN_CONDITION_IDS


def card_cost(card_id: int) -> float:
    return float(CARDS_CATALOG.get(int(card_id), {}).get("cost", 3))


def card_hp(card_id: int) -> float:
    return float(CARDS_CATALOG.get(int(card_id), {}).get("hp", 0.0))


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


@dataclass
class UnitView:
    """Minimal view of a unit on the board (canonical coordinates)."""

    card_id: int
    x: float
    y: float
    hp: float
    is_enemy: bool
    is_air: bool = False


@dataclass
class SpellDecision:
    """Result of evaluating a spell cast."""

    allowed: bool
    reason: str
    target: Optional[Tuple[float, float]] = None
    hits: int = 0
    kills_support: bool = False


def touches_enemy_king(x: float, y: float, team: int = 0, radius: float = 3.2) -> bool:
    """Is this tile close enough to the enemy king tower to activate it?"""
    king = perspective(team).enemy_king
    return _dist(x, y, king[0], king[1]) <= radius


def evaluate_spell_cast(
    card_id: int,
    cast_x: float,
    cast_y: float,
    units: Sequence[UnitView],
    *,
    team: int = 0,
    allow_king_tower: bool = False,
) -> SpellDecision:
    """Decide whether a spell cast is a *good* spell cast.

    Rules (all derived from real Clash Royale practice):

    * never cast a spell in our own half - it can never hit anything useful
      and it exposes us to a counter-push;
    * never cast at the enemy king tower - it activates it for a cheap trade
      and it is rarely worth 4 elixir unless it kills a lot;
    * the spell has to hit something: either at least two enemy units, or a
      single support unit whose hit count is lethal.
    """
    cid = int(card_id)
    if not is_spell(cid):
        return SpellDecision(False, "not-a-spell")

    persp = perspective(team)
    cost, radius, damage, min_cluster, support_is_value = SPELL_STATS.get(
        cid, (3.0, 3.0, 300.0, 2, False)
    )

    # UNCONDITIONALLY forbid spells on tile_y <= 11 (ZERO EXCEPTIONS)
    if cast_y <= 11.5 or persp.is_own_half(cast_y):
        return SpellDecision(False, "own-half")

    if not allow_king_tower and touches_enemy_king(cast_x, cast_y, team):
        return SpellDecision(False, "enemy-king-tower")

    hits = [u for u in units if u.is_enemy and _dist(cast_x, cast_y, u.x, u.y) <= radius]
    if not hits:
        return SpellDecision(False, "no-hits")

    # A support unit (Musketeer, Wizard, ...) is the highest-value target:
    # removing it stops the push that follows the tank.
    kills_support = any(
        h.card_id in SUPPORT_CARD_IDS and (damage >= h.hp or support_is_value)
        for h in hits
    )
    lethal_single = len(hits) == 1 and damage >= hits[0].hp
    value_cluster = len(hits) >= min_cluster

    if not (kills_support or lethal_single or value_cluster):
        return SpellDecision(False, "low-value", hits=len(hits))

    return SpellDecision(
        True,
        "value",
        target=(cast_x, cast_y),
        hits=len(hits),
        kills_support=kills_support,
    )


def best_spell_target(
    card_id: int,
    units: Sequence[UnitView],
    *,
    team: int = 0,
    allow_king_tower: bool = False,
) -> Optional[SpellDecision]:
    """Pick the best spell target among the enemy units on the board."""
    cid = int(card_id)
    if not is_spell(cid):
        return None
    _, radius, damage = SPELL_STATS.get(cid, (3.0, 3.0, 300.0, 2, False))[:3]

    best: Optional[SpellDecision] = None
    best_score = -1.0
    for u in units:
        if not u.is_enemy:
            continue
        decision = evaluate_spell_cast(
            cid, u.x, u.y, units, team=team, allow_king_tower=allow_king_tower
        )
        if not decision.allowed:
            continue
        hits = [h for h in units if h.is_enemy and _dist(u.x, u.y, h.x, h.y) <= radius]
        score = float(len(hits))
        if any(h.card_id in SUPPORT_CARD_IDS for h in hits):
            score += 2.0
        score += sum(damage >= h.hp for h in hits)
        if score > best_score:
            best_score = score
            best = decision
    return best


def threats_on_our_side(units: Sequence[UnitView], team: int = 0) -> List[UnitView]:
    """Enemy units that have crossed onto our half."""
    persp = perspective(team)
    return [u for u in units if u.is_enemy and persp.is_own_half(u.y) and u.hp > 0]


def threat_pressure(units: Sequence[UnitView], team: int = 0) -> float:
    """0..1 urgency of the incoming threat (deeper + closer to a tower)."""
    persp = perspective(team)
    threats = threats_on_our_side(units, team)
    if not threats:
        return 0.0
    pressure = 0.0
    for u in threats:
        depth = max(0.0, (persp.own_half_max_y - u.y) / max(1.0, persp.own_half_max_y))
        tower_dist = min(_dist(u.x, u.y, t[0], t[1]) for t in persp.own_princess)
        proximity = max(0.0, 1.0 - tower_dist / 8.0)
        pressure += 0.6 * depth + 0.4 * proximity
    return float(min(1.0, pressure / max(1, len(threats)) * 1.6))


# ---------------------------------------------------------------------------
# Placement anchors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlacementAnchor:
    x: float
    y: float
    tag: str
    weight: float = 1.0


# A spread of *logical* anchors.  The old bot used three tiles only, which is
# why the learned policy kept deploying on the same three squares.
PLACEMENT_ANCHORS: Tuple[PlacementAnchor, ...] = (
    PlacementAnchor(3.0, 14.0, "bridge_left", 1.0),
    PlacementAnchor(6.0, 14.5, "bridge_left_inner", 0.8),
    PlacementAnchor(9.0, 15.0, "bridge_center", 0.6),
    PlacementAnchor(12.0, 14.5, "bridge_right_inner", 0.8),
    PlacementAnchor(15.0, 14.0, "bridge_right", 1.0),
    PlacementAnchor(4.0, 12.5, "left_support", 0.8),
    PlacementAnchor(14.0, 12.5, "right_support", 0.8),
    PlacementAnchor(9.0, 12.0, "center_push", 0.9),
    PlacementAnchor(9.0, 8.0, "center_hold", 0.6),
    PlacementAnchor(3.0, 9.0, "left_flank", 0.8),
    PlacementAnchor(15.0, 9.0, "right_flank", 0.8),
    PlacementAnchor(7.5, 2.5, "back_king_left", 0.8),
    PlacementAnchor(10.5, 2.5, "back_king_right", 0.8),
    PlacementAnchor(9.0, 2.0, "back_king_center", 0.6),
    PlacementAnchor(2.5, 4.5, "behind_left_princess", 0.7),
    PlacementAnchor(15.5, 4.5, "behind_right_princess", 0.7),
)

DEFENSIVE_ANCHORS: Tuple[PlacementAnchor, ...] = (
    PlacementAnchor(8.5, 11.0, "defend_center_pull_high", 1.2),
    PlacementAnchor(9.0, 10.0, "defend_center_killzone", 1.2),
    PlacementAnchor(4.0, 11.0, "defend_left_infront", 1.0),
    PlacementAnchor(14.0, 11.0, "defend_right_infront", 1.0),
    PlacementAnchor(8.0, 12.0, "defend_center_intercept_left", 1.0),
    PlacementAnchor(10.0, 12.0, "defend_center_intercept_right", 1.0),
    PlacementAnchor(3.5, 10.0, "defend_left_bridge_infront", 0.9),
    PlacementAnchor(14.5, 10.0, "defend_right_bridge_infront", 0.9),
)


def _weighted_choice(
    anchors: Sequence[PlacementAnchor], rng: random.Random
) -> PlacementAnchor:
    weights = [max(1e-3, a.weight) for a in anchors]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for a, w in zip(anchors, weights):
        acc += w
        if r <= acc:
            return a
    return anchors[-1]


def sample_placement(
    rng: random.Random,
    *,
    defensive: bool = False,
    lane_x: Optional[float] = None,
    jitter: float = 1.4,
) -> Tuple[int, int]:
    """Sample a *logical but varied* tile.

    Returns a canonical ``(x, y)`` tile index pair.
    """
    anchors = DEFENSIVE_ANCHORS if defensive else PLACEMENT_ANCHORS
    anchor = _weighted_choice(anchors, rng)
    x, y = anchor.x, anchor.y

    if lane_x is not None and not defensive:
        # Pull the anchor towards the lane we actually want to push.
        x = 0.5 * x + 0.5 * lane_x

    x += rng.uniform(-jitter, jitter)
    y += rng.uniform(-jitter, jitter)

    xi = int(max(0, min(ARENA_WIDTH - 1, round(x))))
    if defensive:
        yi = int(max(9, min(13, round(y))))
    else:
        yi = int(max(0, min(ARENA_HEIGHT - 1, round(y))))
    return xi, yi


def tile_index(x: float, y: float) -> int:
    """Canonical tile index for a (possibly fractional) tile coordinate."""
    xi = int(max(0, min(ARENA_WIDTH - 1, round(x))))
    yi = int(max(0, min(ARENA_HEIGHT - 1, round(y))))
    return yi * ARENA_WIDTH + xi


def tile_to_xy(tile: int) -> Tuple[float, float]:
    return float(tile % ARENA_WIDTH), float(tile // ARENA_WIDTH)


def nearest_legal_tile(legal_mask, desired_x: float, desired_y: float) -> Optional[int]:
    """Snap a desired tile onto the closest *legal* tile.

    ``legal_mask`` is a boolean array over the 576 canonical tiles (the
    per-card mask carried in the observation).  Buildings block deployment
    zones, so a purely geometric anchor is not always playable; without this
    snap the teacher produces actions that the network cannot imitate (and
    behaviour cloning would train against a masked target).
    """
    import numpy as np

    mask = np.asarray(legal_mask)
    if mask.ndim != 1 or mask.size != ARENA_WIDTH * ARENA_HEIGHT:
        raise ValueError("legal_mask must be a flat 576-entry boolean array")
    allowed = np.flatnonzero(mask)
    if allowed.size == 0:
        return None
    xs = allowed % ARENA_WIDTH
    ys = allowed // ARENA_WIDTH
    d2 = (xs - float(desired_x)) ** 2 + (ys - float(desired_y)) ** 2
    return int(allowed[int(np.argmin(d2))])


__all__ = [
    "DEFENSIVE_ANCHORS",
    "ENEMY_KING",
    "ENEMY_PRINCESS",
    "FRIENDLY_KING",
    "FRIENDLY_PRINCESS",
    "Perspective",
    "PlacementAnchor",
    "PLACEMENT_ANCHORS",
    "RIVER_HALF",
    "RIVER_Y",
    "SPELL_STATS",
    "SpellDecision",
    "SUPPORT_CARD_IDS",
    "TANK_CARD_IDS",
    "UnitView",
    "best_spell_target",
    "card_cost",
    "card_hp",
    "evaluate_spell_cast",
    "is_spell",
    "nearest_legal_tile",
    "perspective",
    "sample_placement",
    "threat_pressure",
    "threats_on_our_side",
    "tile_index",
    "tile_to_xy",
    "touches_enemy_king",
]
