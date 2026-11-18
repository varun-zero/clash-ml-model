"""Tests for hidden opponent state, full-HP detection and play credit."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.env.arena_monitor import ArenaMonitor  # noqa: E402
from cr_alphastar_agent.env.opponent_tracker import OpponentTracker  # noqa: E402
from cr_alphastar_agent.utils.card_intel import get_card_intel  # noqa: E402


@pytest.fixture(scope="module")
def intel():
    return get_card_intel()


# ------------------------------------------------------------------ elixir
def test_elixir_starts_at_five_and_regenerates(intel):
    tracker = OpponentTracker(intel)
    tracker.tick(0, 5.0)
    assert tracker.elixir == 5.0
    tracker.tick(2.8, 5.0)
    assert tracker.elixir == pytest.approx(6.0, abs=0.05)


def test_elixir_rates_by_phase(intel):
    for phase, expected in ((1, 2.8), (2, 1.4), (3, 0.93)):
        tracker = OpponentTracker(intel)
        tracker.tick(0, 5.0)
        tracker.tick(expected, 5.0, speed=phase)
        assert tracker.elixir == pytest.approx(6.0, abs=0.05)


def test_playing_a_card_subtracts_its_real_cost(intel):
    tracker = OpponentTracker(intel)
    tracker.tick(0, 5.0)
    tracker.observe_play("Giant", 0, 3)
    # Giant costs 5 in the shipped data.
    assert tracker.elixir == pytest.approx(0.0, abs=0.05)
    assert tracker.seen == ["Giant"]


def test_idle_resync_to_ten(intel):
    tracker = OpponentTracker(intel)
    tracker.tick(0, 5.0)
    tracker.observe_play("Giant", 0, 3)
    tracker.tick(10, 5.0)   # long inactivity
    assert tracker.elixir == 10.0


def test_cycle_lockout_after_four_plays(intel):
    tracker = OpponentTracker(intel)
    tracker.tick(0, 5.0)
    tracker.observe_play("Cannon", 0, 13)
    assert tracker.is_locked("Cannon") is True
    for i, name in enumerate(["Knight", "Arrows", "Fireball", "Giant"], start=1):
        tracker.observe_play(name, i * 2, i)
    # After four other plays the counter is back in cycle.
    assert tracker.is_locked("Cannon") is False


def test_punish_window_requires_low_elixir_and_no_counter(intel):
    tracker = OpponentTracker(intel)
    tracker.tick(0, 5.0)
    tracker.observe_play("Giant", 0, 3)         # broke
    tracker.observe_play("Cannon", 1, 13)       # their counter used
    for name in ["Knight", "Arrows"]:
        tracker.observe_play(name, 2, 1)
    state = tracker.snapshot(my_elixir=9.0, now=3.0)
    assert state.counter_locked is True
    assert state.opposite_lane_punish is True


# -------------------------------------------------------------- full HP/blobs
def test_background_subtraction_finds_a_full_hp_unit():
    monitor = ArenaMonitor(alpha=1.0)
    arena = np.full((400, 600, 3), 70, dtype=np.uint8)
    monitor.suppress(arena, [])
    moved = arena.copy()
    cv2.rectangle(moved, (280, 180), (330, 250), (180, 40, 40), -1)
    blobs = monitor.detect(moved, {"top": 0.05, "bottom": 0.78,
                                   "left": 0.03, "right": 0.97}, (900, 1600))
    assert blobs
    centre = min(blobs, key=lambda b: abs(b.px - 305) + abs(b.py - 215))
    assert centre.area >= 90


def test_active_unit_boxes_are_excluded_from_background():
    monitor = ArenaMonitor(alpha=0.5)
    arena = np.full((400, 600, 3), 70, dtype=np.uint8)
    monitor.suppress(arena, [(100, 100, 50, 50)])
    altered = arena.copy()
    altered[100:150, 100:150] = 255
    monitor.suppress(altered, [(100, 100, 50, 50)])
    assert np.mean(np.abs(monitor.background[100:150, 100:150] - 70)) < 2


# --------------------------------------------------------------- zero-shot
def test_kinematic_typing_distinguishes_swarm_from_tank(intel):
    fast_air, c_fast = intel.coarse_role(1.125, True)
    slow_ground, c_slow = intel.coarse_role(0.562, False)
    assert fast_air == "air_swarm" and c_fast > 0.8
    assert slow_ground == "ground_tank" and c_slow > 0.8


def test_kinematic_candidates_respect_air_filter(intel):
    air = intel.candidates_by_kinematics(1.125, is_air=True)
    assert air
    assert all(intel.get(name).flying for name, _ in air)


# ------------------------------------------------------------- play credit
def test_wasted_spell_gets_near_zero_weight(intel):
    quality, reason = intel.evaluate_play_quality("Arrows", 4, 17, [])
    assert quality < 0.1
    assert "hit nothing" in reason


def test_useful_spell_gets_full_credit(intel):
    quality, reason = intel.evaluate_play_quality(
        "Arrows", 4, 17, [{"x": 4, "y": 17}, {"x": 5, "y": 18}])
    assert quality > 0.8
    assert "hit 2" in reason


def test_overcommit_under_threat_is_penalised(intel):
    quality, reason = intel.evaluate_play_quality(
        "Giant", 2, 13, [], under_threat=True, affordable_defense=True)
    assert quality < 0.5
