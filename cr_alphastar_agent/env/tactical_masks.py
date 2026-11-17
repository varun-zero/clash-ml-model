"""Single source of truth for the tactical action masks.

Historically the simulator (``crsim_replica_env``) and the live MuMu bridge
(``deploy_mumu``) each implemented their own version of the same rules.
They drifted apart, which meant the behaviour-cloning teacher produced
targets the live agent could never execute (and vice versa).

Every rule now lives here exactly once:

* **Spells** (``Fireball``, ``Arrows``, ``Zap``, ``Log`` and friends)
  - the card slot is masked out entirely unless at least one live enemy sits
    at ``tile_y >= 12``;
  - the spell has **zero** unmasked tiles with ``tile_y <= 11``;
  - beyond that, the only unmasked tiles are inside a 2-tile radius around a
    live enemy, using a ``-1`` tile y-lead so the spell lands where the push
    is going, not where it was.
* **Win conditions** (``Giant``, ``HogRider``) are masked out while we are
  under threat (``enemy_y <= 15``) *and* an affordable defensive card is in
  hand.
* **Same-lane troop defense**: while under threat, troop defenders may only
  be placed in the lane of the threat (``x in [threat_x-2 .. threat_x+2]``)
  and only in front of our towers (``y in [8 .. 12]``).
* **Empty-board elixir bank**: with no enemy on the board and less than
  ``ELIXIR_BANK_THRESHOLD`` elixir, all four card slots are masked and the
  agent must bank (slot 4 = Wait stays legal).

Both the simulator and the live bridge call :func:`build_tactical_masks`, so
the two observation contracts cannot drift any more.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cr_alphastar_agent.config import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    CARDS_CATALOG,
    DEFENSIVE_CARD_IDS,
    SPELL_CARD_IDS,
    TOTAL_TILES,
    WIN_CONDITION_IDS,
)

# --------------------------------------------------------------------------
# Tunables - every number the masks depend on lives in this block.
# --------------------------------------------------------------------------

#: A spell may only be aimed at rows at or above this tile row.
SPELL_MIN_TILE_Y = 12
#: Rows at or below this index may never be unmasked for a spell.
SPELL_BLOCKED_TILE_Y = 11
#: Radius (in tiles) of the circle of tiles a spell may be aimed into.
SPELL_RADIUS = 2
#: Y-lead (in tiles) applied to the enemy position before aiming a spell.
SPELL_LEAD_Y = 1
#: An enemy at or below this canonical row means "we are under threat".
THREAT_TILE_Y = 15.0
#: Half-width of the lane a defender must be placed in.
DEFENSE_LANE_RADIUS = 2
#: Vertical band defenders are allowed to be placed in.
DEFENSE_MIN_TILE_Y = 8
DEFENSE_MAX_TILE_Y = 12
#: Bank elixir below this value when the board is empty.
ELIXIR_BANK_THRESHOLD = 8.0
#: Policy card id of the Cannon (a building that pulls to the centre).
CANNON_CARD_ID = 13
#: Policy card id of the Hog Rider (bridge-only win condition).
HOG_RIDER_CARD_ID = 11

#: Tiles that are always legal for a troop when we are not under threat.
FRIENDLY_DEPLOY_MAX_TILE_Y = 15


@dataclass
class EnemyUnit:
    """A live enemy unit in canonical tile coordinates."""

    card_id: int
    x: float
    y: float
    hp: float = 1.0
    is_air: bool = False


@dataclass
class MaskBundle:
    """The result of :func:`build_tactical_masks`."""

    card_mask: np.ndarray            # [5] bool  - slots 0..3 + slot 4 (Wait)
    card_tile_mask: np.ndarray       # [5, 576] bool
    tile_mask: np.ndarray            # [576] bool - union of slots 0..3
    threats: List[EnemyUnit] = field(default_factory=list)
    under_threat: bool = False

    @property
    def affordable_slots(self) -> List[int]:
        return [s for s in range(4) if self.card_mask[s]]


def _card_cost(card_id: int) -> float:
    return float(CARDS_CATALOG.get(int(card_id), {}).get("cost", 3))


def _spell_tiles(enemies: Sequence[EnemyUnit]) -> np.ndarray:
    """Tiles a spell may be aimed at (2-tile radius around every live enemy)."""
    tiles = np.zeros(TOTAL_TILES, dtype=bool)
    for enemy in enemies:
        if enemy.y < SPELL_MIN_TILE_Y - 0.5:
            continue
        cx = int(round(enemy.x))
        cy = max(SPELL_MIN_TILE_Y, int(round(enemy.y)) - SPELL_LEAD_Y)
        for dx in range(-SPELL_RADIUS, SPELL_RADIUS + 1):
            for dy in range(-SPELL_RADIUS, SPELL_RADIUS + 1):
                tx = cx + dx
                ty = cy + dy
                if not (0 <= tx < ARENA_WIDTH):
                    continue
                if ty <= SPELL_BLOCKED_TILE_Y or ty >= ARENA_HEIGHT:
                    continue
                tiles[ty * ARENA_WIDTH + tx] = True
    return tiles


def _troop_tiles(card_id: int, threats: Sequence[EnemyUnit]) -> np.ndarray:
    """Tiles a troop (non-spell, non win-condition) may be deployed on."""
    tiles = np.zeros(TOTAL_TILES, dtype=bool)
    if not threats:
        for y in range(FRIENDLY_DEPLOY_MAX_TILE_Y):
            for x in range(ARENA_WIDTH):
                tiles[y * ARENA_WIDTH + x] = True
        return tiles

    # Under threat: stay in the lane of the threat, in front of our towers.
    threat_x = threats[0].x
    if card_id == CANNON_CARD_ID:
        # Buildings pull the push towards the centre.
        for y in range(DEFENSE_MIN_TILE_Y + 1, DEFENSE_MAX_TILE_Y + 1):
            for x in range(7, 11):
                tiles[y * ARENA_WIDTH + x] = True
        return tiles

    lane_min_x = int(max(1, threat_x - DEFENSE_LANE_RADIUS))
    lane_max_x = int(min(ARENA_WIDTH - 2, threat_x + DEFENSE_LANE_RADIUS))
    for y in range(DEFENSE_MIN_TILE_Y, DEFENSE_MAX_TILE_Y + 1):
        for x in range(lane_min_x, lane_max_x + 1):
            tiles[y * ARENA_WIDTH + x] = True
    return tiles


def build_tactical_masks(
    hand_cards: Sequence[int],
    elixir: float,
    enemies: Sequence[EnemyUnit],
    *,
    legal_tiles: Optional[Sequence[np.ndarray]] = None,
    enemy_king: Optional[Tuple[float, float]] = None,
    blocked_slots: Optional[Sequence[int]] = None,
) -> MaskBundle:
    """Build the per-card mask and the per-card tile mask.

    Parameters
    ----------
    hand_cards:
        The four policy card ids currently in hand (slots 0..3).
    elixir:
        Current elixir, 0..10.
    enemies:
        Live enemy units in canonical tile coordinates.
    legal_tiles:
        Optional per-slot engine legality (simulator only).  When given, the
        computed tiles are intersected with the engine's legal set.
    enemy_king:
        Optional ``(x, y)`` of the enemy king tower.  Tiles that would
        activate it are removed from spell masks.
    blocked_slots:
        Slot indices that must not be played right now (e.g. a rejected tap).
    """
    hand = [int(c) for c in list(hand_cards)[:4]]
    while len(hand) < 4:
        hand.append(1)

    enemies = list(enemies)
    threats = [e for e in enemies if e.y <= THREAT_TILE_Y]
    under_threat = bool(threats)

    card_mask = np.zeros(5, dtype=bool)
    card_tile_mask = np.zeros((5, TOTAL_TILES), dtype=bool)

    spells_have_target = any(e.y >= SPELL_MIN_TILE_Y - 0.5 for e in enemies)

    for slot in range(4):
        cid = hand[slot]
        if elixir < _card_cost(cid):
            continue

        if cid in SPELL_CARD_IDS:
            # A spell without a target is not a legal action at all.
            if not spells_have_target:
                continue
            tiles = _spell_tiles(enemies)
            if enemy_king is not None:
                kx, ky = enemy_king
                for ty in range(max(0, int(ky) - 4), min(ARENA_HEIGHT, int(ky) + 5)):
                    for tx in range(max(0, int(kx) - 4), min(ARENA_WIDTH, int(kx) + 5)):
                        if (tx - kx) ** 2 + (ty - ky) ** 2 <= 3.2 ** 2:
                            tiles[ty * ARENA_WIDTH + tx] = False
        elif cid == HOG_RIDER_CARD_ID:
            # Hog Rider: the two bridges only, never our own lane.
            tiles = np.zeros(TOTAL_TILES, dtype=bool)
            for x in list(range(2, 5)) + list(range(13, 16)):
                tiles[14 * ARENA_WIDTH + x] = True
        else:
            tiles = _troop_tiles(cid, threats)

        if legal_tiles is not None and slot < len(legal_tiles):
            tiles = np.logical_and(tiles, np.asarray(legal_tiles[slot], dtype=bool))

        card_tile_mask[slot] = tiles
        card_mask[slot] = bool(tiles.any())

    # Never waste a win condition on defence while a defender is affordable.
    if under_threat:
        has_affordable_defense = any(
            card_mask[s] and hand[s] in DEFENSIVE_CARD_IDS for s in range(4)
        )
        if has_affordable_defense:
            for s in range(4):
                if hand[s] in WIN_CONDITION_IDS:
                    card_mask[s] = False

    # Empty board: bank elixir instead of wasting it.
    if len(enemies) == 0 and elixir < ELIXIR_BANK_THRESHOLD:
        card_mask[:4] = False

    if blocked_slots:
        for s in blocked_slots:
            if 0 <= s < 4:
                card_mask[s] = False

    card_mask[4] = True                     # Wait is always legal
    card_tile_mask[4] = True               # ... and legal everywhere
    tile_mask = card_tile_mask[:4].any(axis=0)

    return MaskBundle(
        card_mask=card_mask,
        card_tile_mask=card_tile_mask,
        tile_mask=tile_mask,
        threats=threats,
        under_threat=under_threat,
    )


__all__ = [
    "CANNON_CARD_ID",
    "DEFENSE_LANE_RADIUS",
    "DEFENSE_MAX_TILE_Y",
    "DEFENSE_MIN_TILE_Y",
    "ELIXIR_BANK_THRESHOLD",
    "EnemyUnit",
    "HOG_RIDER_CARD_ID",
    "MaskBundle",
    "SPELL_BLOCKED_TILE_Y",
    "SPELL_LEAD_Y",
    "SPELL_MIN_TILE_Y",
    "SPELL_RADIUS",
    "THREAT_TILE_Y",
    "build_tactical_masks",
]
