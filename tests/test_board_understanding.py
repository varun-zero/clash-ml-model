"""Tests for live board tracking and data-driven card counters."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.env.board_tracker import BoardTracker  # noqa: E402
from cr_alphastar_agent.utils.card_intel import CardIntel, get_card_intel  # noqa: E402


@pytest.fixture(scope="module")
def intel():
    return get_card_intel()


def test_card_database_is_loaded_from_the_shipped_logic_tables(intel):
    assert len(intel) >= 130
    expected = ["Knight", "MiniPekka", "Arrows", "Fireball", "Giant",
                "HogRider", "Cannon", "Minions", "Zap"]
    for name in expected:
        assert intel.get(name) is not None, name


def test_display_name_aliases_resolve(intel):
    for name in ["Archers", "Mini PEKKA", "Mini P.E.K.K.A", "MiniPekka",
                 "Hog Rider", "Baby Dragon", "Mega Minion"]:
        card = intel.get(name)
        assert card is not None, name
        assert card.cost > 0


def test_spell_damage_is_extracted(intel):
    arrows = intel.get("Arrows")
    fireball = intel.get("Fireball")
    zap = intel.get("Zap")
    assert arrows.damage > 0
    assert fireball.damage > 0
    assert zap.damage > 0
    assert arrows.area_radius_tiles > 0


def test_counter_scores_match_game_sense(intel):
    assert intel.counter_score("Arrows", "Minions").score > 40       # swarms
    assert intel.counter_score("Fireball", "Archers").score > 20     # support
    assert intel.counter_score("MiniPekka", "Giant").score > 10      # tank killer
    assert intel.counter_score("Cannon", "HogRider").score > 20      # building pull
    assert intel.counter_score("Knight", "MiniPekka").score < 0      # bad trade
    assert intel.counter_score("Arrows", "Giant").score < 0          # no spell value
    assert intel.counter_score("Knight", "Balloon").score <= -100    # cannot hit air


def test_best_counters_are_ranked(intel):
    ranked = intel.best_counters(
        ["Knight", "Arrows", "Musketeer", "Giant"],
        ["Minions", "Giant"],
    )
    assert len(ranked) == 4
    assert ranked[0].our_card == "Arrows"
    assert ranked[0].score >= ranked[-1].score


def test_recommend_respects_affordability(intel):
    rec = intel.recommend(
        ["Knight", "Arrows", "Musketeer", "Giant"],
        ["Minions"],
        affordable=[True, False, True, True],
    )
    assert rec is not None
    assert rec.our_card != "Arrows"


def test_counter_explanations_are_auditable(intel):
    score = intel.counter_score("Arrows", "Minions")
    assert score.reasons
    assert "one-shots" in score.explain()
    assert "splash" in score.explain()


# ------------------------------------------------------------------ tracking
def test_tracker_estimates_velocity_and_destination():
    tracker = BoardTracker()
    snapshot = tracker.update([{"x": 3.0, "y": 20.0, "hp": 1.0}], now=0.0)
    assert len(snapshot.enemy_tracks) == 1

    # Same unit moves 1 tile up in 0.5 s -> vy = -2 tiles/s.
    snapshot = tracker.update([{"x": 3.0, "y": 19.0, "hp": 1.0}], now=0.5)
    track = snapshot.enemy_tracks[0]
    assert track.vy < -0.5
    assert track.pushing_toward_our_half
    dx, dy = track.destination
    assert dy < track.y
    assert 0 <= dx < 18 and 0 <= dy < 32


def test_tracker_detects_a_push_and_the_endangered_tower():
    tracker = BoardTracker()
    # Two units walking down the left lane toward our princess tower.
    for tick in range(6):
        snapshot = tracker.update([
            {"x": 3.0, "y": 18.0 - 0.4 * tick, "hp": 1.0},
            {"x": 4.0, "y": 18.5 - 0.4 * tick, "hp": 1.0},
        ], now=tick * 0.5)
    assert snapshot.push_detected
    assert snapshot.push_size >= 2

    # Once the same push reaches our half the towers must light up as threatened.
    for tick in range(6, 14):
        snapshot = tracker.update([
            {"x": 3.0, "y": 14.0 - 0.6 * (tick - 6), "hp": 1.0},
            {"x": 4.0, "y": 14.5 - 0.6 * (tick - 6), "hp": 1.0},
        ], now=tick * 0.5)
    assert snapshot.under_threat
    assert snapshot.most_dangerous_tower in ("princess_left", "princess_right")
    assert snapshot.defense_target is not None
    tx, ty = snapshot.defense_target
    assert 0 <= tx < 18 and 8 <= ty <= 14


def test_tracker_uses_tower_hp_when_available():
    tracker = BoardTracker()
    tracker.set_tower_hp("princess_right", 0.3)
    snapshot = tracker.update([
        {"x": 15.0, "y": 12.0, "hp": 1.0},
    ], now=0.0)
    right = snapshot.towers["princess_right"]
    assert right.hp_ratio == pytest.approx(0.3)
    assert right.hp_known
    assert snapshot.most_dangerous_tower == "princess_right"


def test_tracker_expires_stale_tracks():
    tracker = BoardTracker()
    tracker.update([{"x": 3.0, "y": 18.0}], now=0.0)
    snapshot = tracker.update([], now=10.0)
    assert snapshot.enemy_tracks == []
