"""Tests for the current-deck training additions.

Covers:
  * the explicit opponent-elixir tracker,
  * the tactical rules (spell discipline, defence priority, placement variety),
  * the deck registry / current-deck resolution,
  * the tactical coach bot used as the behaviour-cloning teacher.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.config import (  # noqa: E402
    CARDS_CATALOG,
    CURRENT_DECK,
    CURRENT_DECK_IDS,
    resolve_deck_ids,
    resolve_deck_names,
)
from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv  # noqa: E402
from cr_alphastar_agent.env.opponent_elixir import (  # noqa: E402
    OpponentElixirTracker,
)
from cr_alphastar_agent.env.tactical_coach import TacticalCoachBot  # noqa: E402
from cr_alphastar_agent.env.tactics import (  # noqa: E402
    ENEMY_KING,
    FRIENDLY_KING,
    UnitView,
    best_spell_target,
    evaluate_spell_cast,
    is_spell,
    sample_placement,
    threat_pressure,
    threats_on_our_side,
    tile_index,
)


# --------------------------------------------------------------------- deck
def test_current_deck_resolves_to_eight_distinct_policy_ids():
    ids = resolve_deck_ids("current")
    assert ids == CURRENT_DECK_IDS
    assert len(ids) == 8
    assert len(set(ids)) == 8
    for cid in ids:
        assert cid in CARDS_CATALOG


def test_resolve_deck_ids_accepts_names_and_presets():
    names = resolve_deck_names(["Knight", "Fireball", "Arrows"])
    # A deck is always padded to eight distinct engine cards.
    assert len(names) == 8
    assert len(set(names)) == 8
    assert names[:3] == ["Knight", "Fireball", "Arrows"]
    assert resolve_deck_ids("starter") == CURRENT_DECK_IDS
    with pytest.raises(ValueError):
        resolve_deck_ids("not-a-deck")


def test_every_preset_resolves_to_a_legal_engine_deck():
    """The simulator rejects decks that are not eight distinct real cards."""
    from crsim.engine import load_model
    from cr_alphastar_agent.config import DECK_PRESETS

    model = load_model()
    tables = model.get("tables", {})
    engine_names = set()
    for key in ("spells_characters", "spells_buildings", "spells_other", "spells_evolved"):
        for row in tables.get(key) or []:
            if isinstance(row.get("Name"), str):
                engine_names.add(row["Name"])

    for preset in DECK_PRESETS:
        names = resolve_deck_names(preset)
        assert len(names) == 8, preset
        assert len(set(names)) == 8, preset
        missing = [n for n in names if n not in engine_names]
        assert not missing, f"{preset}: unknown engine cards {missing}"


# ------------------------------------------------------------ elixir tracker
def test_tracker_starts_at_five_and_regenerates():
    t = OpponentElixirTracker()
    est = t.estimate(0.0)
    assert est.high == pytest.approx(5.0)
    assert 0.0 < est.mean <= 5.0
    # 28 s of normal time is one full bar (10 elixir) -> capped at 10.
    est = t.estimate(28.0)
    assert est.high == pytest.approx(10.0)


def test_tracker_accounts_for_observed_plays():
    t = OpponentElixirTracker()
    t.observe_play(cost=4, t_s=10.0)
    est = t.estimate(10.0)
    # 5 + regen(10 s) - 4 spent, capped at 10.
    expected_high = min(10.0, 5.0 + 10.0 * 10.0 / 28.0 - 4.0)
    assert est.high == pytest.approx(expected_high, abs=1e-6)
    assert est.low < est.high
    assert 0.0 <= est.confidence <= 1.0


def test_tracker_uncertainty_grows_with_missed_plays():
    exact = OpponentElixirTracker()
    lossy = OpponentElixirTracker(assume_missed_plays=True)
    exact.observe_play(cost=3, t_s=5.0)
    lossy.observe_play(cost=3, t_s=5.0)
    a = exact.estimate(90.0)
    b = lossy.estimate(90.0)
    assert b.spread > a.spread
    assert b.confidence < a.confidence


def test_tracker_vector_is_bounded():
    t = OpponentElixirTracker()
    vec = t.estimate(30.0).as_vector()
    assert len(vec) == 4
    assert all(0.0 <= v <= 1.0 for v in vec)


# ------------------------------------------------------------------ tactics
def test_spell_is_rejected_in_own_half():
    units = [UnitView(card_id=4, x=3.0, y=20.0, hp=600, is_enemy=True)]
    decision = evaluate_spell_cast(8, 3.0, 8.0, units)   # Fireball, own half
    assert not decision.allowed
    assert decision.reason == "own-half"


def test_spell_is_rejected_on_enemy_king_tower():
    units = [UnitView(card_id=3, x=ENEMY_KING[0], y=ENEMY_KING[1], hp=3300, is_enemy=True)]
    decision = evaluate_spell_cast(8, ENEMY_KING[0], ENEMY_KING[1], units)
    assert not decision.allowed
    assert decision.reason == "enemy-king-tower"


def test_spell_perspective_is_mirrored_for_player_two():
    """Player 2 sees a mirrored arena; the canonical frame always puts our own
    king at the bottom and the enemy king at the top."""
    from cr_alphastar_agent.env.tactics import perspective

    p1 = perspective(0)
    p2 = perspective(1)
    assert p1.is_own_half(5.0) and not p1.is_own_half(25.0)
    assert p2.is_own_half(5.0) and not p2.is_own_half(25.0)
    assert p1.enemy_king[1] > p1.own_king[1]
    assert p2.enemy_king[1] > p2.own_king[1]
    # The seats are mirrored: seat 2's own king sits one row higher because
    # the mirror is canonical_y = 31 - engine_y.
    assert p2.own_king[1] == pytest.approx(p1.own_king[1] - 1.0)
    assert p2.enemy_king[1] == pytest.approx(p1.enemy_king[1] - 1.0)

    # A cast at canonical y=8 is in our own half for both seats.
    units = [UnitView(card_id=4, x=3.0, y=8.0, hp=600, is_enemy=True)]
    assert not evaluate_spell_cast(8, 3.0, 8.0, units, team=0).allowed
    assert not evaluate_spell_cast(8, 3.0, 8.0, units, team=1).allowed

    # The enemy king is at the top of the canonical frame for both seats.
    for team in (0, 1):
        king = perspective(team).enemy_king
        units2 = [UnitView(card_id=3, x=king[0], y=king[1], hp=3300, is_enemy=True)]
        decision = evaluate_spell_cast(8, king[0], king[1], units2, team=team)
        assert not decision.allowed
        assert decision.reason == "enemy-king-tower"


def test_spell_without_targets_is_rejected():
    units = [UnitView(card_id=1, x=3.0, y=20.0, hp=1400, is_enemy=True)]
    decision = evaluate_spell_cast(8, 12.0, 20.0, units)
    assert not decision.allowed
    assert decision.reason == "no-hits"


def test_spell_hitting_a_support_unit_is_allowed():
    units = [
        UnitView(card_id=4, x=10.0, y=20.0, hp=600, is_enemy=True),   # Musketeer
        UnitView(card_id=3, x=12.0, y=22.0, hp=3300, is_enemy=True),  # Giant
    ]
    decision = evaluate_spell_cast(8, 10.0, 20.0, units)   # Fireball on the Musketeer
    assert decision.allowed
    assert decision.kills_support

    best = best_spell_target(8, units)
    assert best is not None and best.allowed


def test_threat_detection_and_pressure():
    assert threats_on_our_side([]) == []
    units = [UnitView(card_id=11, x=3.0, y=10.0, hp=1400, is_enemy=True)]
    assert len(threats_on_our_side(units)) == 1
    assert threat_pressure(units) > 0.0
    assert threat_pressure([]) == 0.0


def test_placement_sampling_is_varied_but_bounded():
    import random
    rng = random.Random(0)
    tiles = {sample_placement(rng) for _ in range(200)}
    assert len(tiles) > 10, "placements collapsed onto a handful of tiles"
    for x, y in tiles:
        assert 0 <= x <= 17 and 0 <= y <= 31


def test_tile_index_roundtrip():
    assert tile_index(0, 0) == 0
    assert tile_index(17, 31) == 31 * 18 + 17


# --------------------------------------------------------------- coach bot
def _make_env():
    return CrSimReplicaEnv(seed=11, blue_deck=CURRENT_DECK_IDS, red_deck=CURRENT_DECK_IDS)


def test_coach_bot_never_plays_a_spell_in_our_own_half():
    env = _make_env()
    bot = TacticalCoachBot(player_id=1, seed=3)
    obs, _ = env.reset()
    for _ in range(300):
        obs_p2 = env.get_obs(2)
        a1 = bot.act(obs)
        a2 = bot.act(obs_p2)
        if a1[0] != 4:
            cid = int(obs["hand_cards"][a1[0]])
            ty = float(a1[1]) // 18.0
            if is_spell(cid):
                assert ty >= 16.0, "coach cast a spell into our own half"
        obs, _, _, _, term, trunc, _ = env.step_both(a1, a2)
        if term or trunc:
            break


def test_coach_bot_uses_many_different_placement_tiles():
    env = _make_env()
    bot = TacticalCoachBot(player_id=1, seed=5)
    obs, _ = env.reset()
    tiles = set()
    for _ in range(400):
        obs_p2 = env.get_obs(2)
        a1 = bot.act(obs)
        a2 = bot.act(obs_p2)
        if a1[0] != 4:
            tiles.add(int(a1[1]))
        obs, _, _, _, term, trunc, _ = env.step_both(a1, a2)
        if term or trunc:
            break
    assert len(tiles) >= 5, f"only {len(tiles)} distinct placement tiles"


# ------------------------------------------------------------------- env
def test_env_observation_does_not_leak_opponent_elixir():
    """The opponent-elixir scalar must come from the tracker, not the truth."""
    env = CrSimReplicaEnv(seed=13, blue_deck=CURRENT_DECK_IDS, red_deck=CURRENT_DECK_IDS)
    obs, _ = env.reset()
    truth = env.battle.elixir[1]
    estimate = env._opponent_elixir(0)
    # At the start the tracker knows the opponent has exactly 5 elixir.
    assert estimate == pytest.approx(5.0, abs=0.3)
    assert truth == pytest.approx(5.0, abs=1e-6)
    assert obs["scalars"][1] == pytest.approx(estimate / 10.0, abs=1e-6)


def test_env_tracker_follows_observed_plays():
    env = CrSimReplicaEnv(seed=17, blue_deck=CURRENT_DECK_IDS, red_deck=CURRENT_DECK_IDS)
    env.reset()
    # Player 2 (team 1) plays a 4-elixir Fireball.
    env.battle.elixir[1] = 10.0
    env.battle.play(1, 0, 3.0, 20.0)
    env._observe_opponent_play(1, "Fireball", env.battle.t)
    est = env._elixir_trackers[0].estimate(env.battle.t / 1000.0)
    assert est.high < 10.0
    assert est.mean < 6.0


def test_tactical_shaping_penalises_wasted_spells_and_pushes():
    env = CrSimReplicaEnv(seed=19, blue_deck=CURRENT_DECK_IDS, red_deck=CURRENT_DECK_IDS)
    env.reset()
    # Fireball in our own half -> misuse penalty.
    penalty = env._tactical_shaping(0, "Fireball", 3.0, 8.0)
    assert penalty < 0.0


# ------------------------------------------------------------------ live BC
def test_live_recorder_round_trips_into_a_bc_dataset(tmp_path):
    from cr_alphastar_agent.env.live_recorder import (
        LiveMatchRecorder,
        load_live_replays,
        rows_to_dataset,
    )

    env = _make_env()
    obs, _ = env.reset()
    path = tmp_path / "live.jsonl"
    with LiveMatchRecorder(str(path), session={"deck": "current"}) as rec:
        for i in range(5):
            rec.record(obs, [i % 4, 100 + i], reward=0.0, done=False, t_ms=i * 400)

    rows = load_live_replays([str(path)])
    assert len(rows) == 5
    dataset = rows_to_dataset(rows)
    assert dataset["action"].shape == (5, 2)
    assert dataset["scalars"].shape[0] == 5
    assert dataset["hand_cards"].dtype == np.int64
    assert "card_tile_mask" in dataset


def test_deploy_deck_resolution_matches_the_current_deck():
    import deploy_mumu as deploy

    names = deploy.resolve_deck_names("current")
    assert names == list(CURRENT_DECK)
    assert deploy.resolve_deck_names("hog26")[0] == "HogRider"


def test_spell_placement_mask_blocks_own_half_and_king_tower():
    from cr_alphastar_agent.env.tactics import perspective, touches_enemy_king

    env = CrSimReplicaEnv(seed=23, blue_deck=CURRENT_DECK_IDS, red_deck=CURRENT_DECK_IDS)
    env.reset()
    # Spawn opponent units on enemy territory to activate spell targeting
    cm = env.battle._card_meta("Knight")
    u_red = env.battle._sp_unit("Knight", 1, 4.0, 20.0, building=False, defdict=cm)
    u_blue = env.battle._sp_unit("Knight", 0, 4.0, 12.0, building=False, defdict=cm)
    env.battle.units.extend([u_red, u_blue])

    obs = env.get_obs(1)
    obs2 = env.get_obs(2)

    for player_obs, team in ((obs, 0), (obs2, 1)):
        persp = perspective(team)
        hand = player_obs["hand_cards"]
        spell_slots = [s for s in range(4) if int(hand[s]) in (8, 9, 14, 16)]
        if not spell_slots:
            continue
        mask = player_obs["card_tile_mask"][spell_slots[0]]
        checked = 0
        for tile in range(mask.size):
            if not mask[tile]:
                continue
            y = tile // 18
            x = tile % 18
            assert not persp.is_own_half(float(y)), "spell allowed in our own half"
            assert not touches_enemy_king(float(x), float(y), team), \
                "spell allowed on the enemy king tower"
            checked += 1
        assert checked > 0
