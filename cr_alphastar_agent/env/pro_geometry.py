"""Pro-level placement geometry: kiting, King activation and spell cycles.

These are fixed geometric facts of the arena, not learned behaviour.  Keeping
them in one module makes them testable and stops the live bridge from
hard-coding tile numbers in its main loop.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

# Towers in canonical coordinates (y=0 is our King, y=31 the enemy King).
OUR_KING = (9.0, 3.0)
OUR_PRINCESS = ((3.5, 6.5), (14.5, 6.5))
KING_ACTIVATION_TILES = {
    "HogRider": [(8, 7), (9, 7)],
    "RoyalGiant": [(8, 7), (9, 7)],
    "GoblinBarrel": [(8, 4)],
    "Miner": [(8, 7), (9, 7)],
    "Balloon": [(9, 7)],
}
MELEE_THREAT_ROLES = frozenset({"ground_tank", "ground_support", "ground_unit"})


def kite_tiles(threat_x: float, threat_y: float) -> List[Tuple[int, int]]:
    """Cross-lane pull tiles for a melee attacker.

    A melee unit chasing a troop placed on the opposite side of the centre is
    dragged diagonally across the river into BOTH Princess Towers' range.
    The tile must still be on our half so the placement is legal.
    """
    if threat_x < 9.0:
        # Left-lane melee -> pull to the right-centre.
        return [(10, 11), (10, 10), (11, 11)]
    # Right-lane melee -> pull to the left-centre.
    return [(7, 11), (7, 10), (6, 11)]


def should_kite(role: str, speed: float, threat_y: float) -> bool:
    """Only slow melee ground units are worth kiting."""
    return role in MELEE_THREAT_ROLES and speed <= 0.85 and threat_y >= 8.0


def king_activation_tiles(threat_card: Optional[str]) -> List[Tuple[int, int]]:
    if not threat_card:
        return []
    return list(KING_ACTIVATION_TILES.get(threat_card, []))


#: Level-11 Princess Tower hitpoints (the level used by the simulator).
TOWER_MAX_HP = 2534.0


def lethal_spell_cycle(tower_hp_ratio: float, spell_damage: Sequence[float],
                       time_left_s: float, overtime: bool) -> bool:
    """True when one spell can finish the tower before time expires.

    ``tower_hp_ratio`` is 0..1 of a full Princess Tower; ``spell_damage`` is
    in absolute hitpoints from ``data/logic.json``.
    """
    if not overtime and time_left_s > 30.0:
        return False
    if tower_hp_ratio <= 0 or tower_hp_ratio > 0.35:
        return False
    remaining_hp = tower_hp_ratio * TOWER_MAX_HP
    return any(float(d) >= remaining_hp for d in spell_damage)


def opposite_bridge_x(lane_x: float) -> range:
    """Bridge tiles on the side opposite to ``lane_x``."""
    return range(12, 16) if lane_x < 9 else range(2, 6)


__all__ = [
    "KING_ACTIVATION_TILES", "OUR_KING", "OUR_PRINCESS",
    "king_activation_tiles", "kite_tiles", "lethal_spell_cycle",
    "opposite_bridge_x", "should_kite",
]
