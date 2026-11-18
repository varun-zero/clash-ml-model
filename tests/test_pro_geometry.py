"""Tests for pro-level placement geometry and kinematic spell leads."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.env.pro_geometry import (  # noqa: E402
    king_activation_tiles, kite_tiles, lethal_spell_cycle,
    opposite_bridge_x, should_kite,
)
from cr_alphastar_agent.utils.card_intel import get_card_intel  # noqa: E402


def test_kite_tiles_cross_the_centre():
    left = kite_tiles(3.0, 13.0)
    right = kite_tiles(15.0, 13.0)
    assert left and right
    assert all(x > 9 for x, _ in left)
    assert all(x < 9 for x, _ in right)
    assert all(8 <= y <= 14 for _, y in left + right)


def test_kiting_only_for_slow_melee():
    assert should_kite("ground_tank", 0.56, 13.0)
    assert should_kite("ground_support", 0.75, 13.0)
    assert not should_kite("ground_tank", 1.5, 13.0)   # too fast
    assert not should_kite("air_swarm", 1.125, 13.0)
    assert not should_kite("ground_tank", 0.56, 4.0)   # already in our backfield


def test_king_activation_coordinates():
    assert king_activation_tiles("HogRider") == [(8, 7), (9, 7)]
    assert king_activation_tiles("GoblinBarrel") == [(8, 4)]
    assert king_activation_tiles("Knight") == []


def test_lethal_spell_cycle_requires_low_tower_and_late_game():
    assert lethal_spell_cycle(0.20, [570, 300], 20.0, False)   # 507 HP left
    assert lethal_spell_cycle(0.20, [570, 300], 0.0, True)
    assert not lethal_spell_cycle(0.20, [570, 300], 90.0, False)
    assert not lethal_spell_cycle(0.9, [570, 300], 20.0, False)
    assert not lethal_spell_cycle(0.20, [10, 20], 20.0, False)


def test_opposite_bridge():
    assert list(opposite_bridge_x(3.0)) == [12, 13, 14, 15]
    assert list(opposite_bridge_x(15.0)) == [2, 3, 4, 5]


def test_spell_lead_intercepts_a_moving_unit():
    intel = get_card_intel()
    stationary = intel.spell_lead_target("Arrows", {"x": 4, "y": 20, "vx": 0, "vy": 0})
    moving = intel.spell_lead_target("Arrows", {"x": 4, "y": 20, "vx": 0, "vy": -1.5})
    assert stationary == (4.0, 20.0)
    assert moving[1] < stationary[1]          # lead toward its direction
    assert 0 <= moving[0] < 18 and 0 <= moving[1] < 32


def test_spell_lead_clamps_to_arena():
    intel = get_card_intel()
    x, y = intel.spell_lead_target("Fireball", {"x": 1, "y": 1, "vx": -20, "vy": -20})
    assert 0 <= x < 18 and 0 <= y < 32
