"""Tests for the competitive v2 stack.

These are deliberately small: a tiny transformer, short rollouts and preset
decks.  They cover the parts most likely to regress: public-observation
invariance, legality masks, belief updates, replay parsing, PPO plumbing and
paired-evaluation statistics.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from crsim.engine import ARENA_H, ARENA_W, Battle, MAX_ELIXIR, NORMAL_MS

from cr_alphastar_agent.competitive.bc import build_model_from_config
from cr_alphastar_agent.competitive.belief import (
    ElixirCounter, OpponentBelief, regen_between,
)
from cr_alphastar_agent.competitive.cards import CardVocab, default_vocab
from cr_alphastar_agent.competitive.env import CompetitiveEnv, CompetitiveEnvConfig
from cr_alphastar_agent.competitive.evaluation import wilson_ci
from cr_alphastar_agent.competitive.league import League, ScriptedOpponent
from cr_alphastar_agent.competitive.model import CompetitiveNet, patch_index
from cr_alphastar_agent.competitive.obs import (
    Action, N_CELLS, build_observation, cell_center, cell_patch, cell_xy,
    decode_action, engine_xy, legal_cell_mask, view_card_mask,
)
from cr_alphastar_agent.competitive.policy import (
    ActionSampler, evaluate_actions, obs_to_tensors,
)
from cr_alphastar_agent.competitive.ppo import (
    PPOConfig, PPOUpdater, RolloutBuffer, compute_gae, normalize_advantages,
)
from cr_alphastar_agent.competitive.replay import (
    build_examples_for_team, examples_to_arrays, parse_hf_payload,
    parse_local_replay,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def vocab() -> CardVocab:
    return default_vocab()


@pytest.fixture(scope="module")
def decks(vocab):
    pool = vocab.playable_deck_pool()
    assert len(pool) >= 2, "expected at least two playable preset decks"
    return pool


def _tiny_model(vocab) -> CompetitiveNet:
    return CompetitiveNet(
        vocab.n_cards, d_model=32, layers=1, heads=2,
        dim_feedforward=64, dropout=0.0, d_card=16,
    )


# --------------------------------------------------------------------- vocab
def test_vocab_is_deterministic_and_alias_aware(vocab):
    assert vocab.n_cards > 100
    assert vocab.id("Knight") == vocab.id("knight") == vocab.id("KNIGHT")
    assert vocab.id("NotACard") == 1  # unknown, not crash
    knight = vocab.cards if False else vocab.info("Knight")
    assert knight.cost > 0 and knight.is_unit
    assert vocab.info("Fireball").is_spell
    assert vocab.id("TheLog") == vocab.id("Log")


def test_preset_decks_are_engine_complete(vocab, decks):
    for deck in decks:
        assert len(deck) == 8 and len(set(deck)) == 8
        assert all(vocab.card_meta(c) for c in deck)
        assert len(vocab.deck_ids(deck)) == 8


# -------------------------------------------------------------------- belief
def test_elixir_counter_regen_and_rebase():
    c = ElixirCounter()
    c.advance(28_000.0)
    assert c.est == pytest.approx(MAX_ELIXIR, abs=1e-6)
    c.observe_play(28_000.0, 3.0)
    assert c.est == pytest.approx(7.0, abs=1e-6)
    # A play the estimate cannot afford re-bases it to exactly the cost.
    c.observe_play(28_000.0, 9.0)
    assert c.rebases == 1
    assert c.est == pytest.approx(0.0, abs=1e-6)
    assert regen_between(0.0, 28_000.0) == pytest.approx(10.0)
    assert regen_between(120_000.0, 148_000.0) == pytest.approx(20.0)
    assert regen_between(300_000.0, 320_000.0) == 0.0


def test_belief_hand_distribution_and_updates(vocab, decks):
    belief = OpponentBelief(vocab, deck_prior=decks[:3], n_particles=32, seed=3)
    hand = belief.hand_probs()
    assert hand.shape == (vocab.n_cards,)
    assert hand.sum() == pytest.approx(4.0, abs=1e-4)
    assert belief.deck_probs().sum() == pytest.approx(8.0, abs=1e-4)
    belief.observe_play(decks[0][0], 1_000.0)
    summary = belief.summary()
    assert summary["plays"] == 1
    assert decks[0][0] in summary["revealed"]
    assert 0.0 <= summary["ess_frac"] <= 1.0
    vec = belief.belief_vector()
    assert vec.shape == (3 * vocab.n_cards + 6,)
    assert np.isfinite(vec).all()


def test_belief_recovers_from_total_particle_depletion(vocab, decks):
    belief = OpponentBelief(vocab, deck_prior=decks[:1], n_particles=16, seed=5)
    # Simulate a filter whose every hypothesis has already been rejected.
    for p in belief.particles:
        p.weight = 0.0
    belief.observe_play(decks[1][0], 500.0)
    assert belief.particles
    assert sum(p.weight for p in belief.particles) == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= belief.summary()["ess_frac"] <= 1.0
    assert np.isfinite(belief.belief_vector()).all()

    # A long out-of-prior sequence must not raise ZeroDivisionError.
    for i, card in enumerate(decks[1] * 4):
        belief.observe_play(card, float(1_000 + 700 * i))
    summary = belief.summary()
    assert summary["plays"] == 33
    assert 0.0 <= summary["ess_frac"] <= 1.0
    assert np.isfinite(belief.hand_probs()).all()


# ---------------------------------------------------------------- observation
def test_policy_observation_hides_opponent_state(vocab, decks):
    battle = Battle([decks[0], decks[1]], seed=5)
    for _ in range(25):
        battle.step([])
    beliefs = {0: OpponentBelief(vocab, deck_prior=decks, seed=1)}
    a = build_observation(battle, 0, beliefs[0], vocab=vocab)
    hidden = (battle.elixir[1], list(battle.hand[1]), list(battle.cycle[1]))
    battle.elixir[1] = 0.0
    battle.hand[1] = list(reversed(battle.hand[1]))
    battle.cycle[1] = list(reversed(battle.cycle[1]))
    b = build_observation(battle, 0, beliefs[0], vocab=vocab)
    battle.elixir[1], battle.hand[1], battle.cycle[1] = hidden
    for key, va in a.policy_arrays().items():
        vb = b.policy_arrays()[key]
        assert np.array_equal(np.asarray(va), np.asarray(vb)), f"leak via {key}"
    # Privileged labels do change, proving the mutation was real.
    assert a.aux_opp_elixir != b.aux_opp_elixir or a.aux_opp_next != b.aux_opp_next


def test_legal_masks_match_engine(vocab, decks):
    battle = Battle([decks[0], decks[1]], seed=7)
    for _ in range(15):
        battle.step([])
    rng = np.random.default_rng(0)
    for slot, card in enumerate(battle.hand[0]):
        engine_mask = legal_cell_mask(battle, 0, card)
        for cell in rng.integers(0, N_CELLS, size=40):
            vx, vy = cell_center(int(cell))
            ex, ey = engine_xy(vx, vy, 0)
            assert bool(engine_mask[cell]) == battle.legal_action(0, slot, ex, ey), (
                f"slot {slot} cell {cell} card {card}"
            )
    card_mask = view_card_mask(battle, 0)
    assert card_mask.shape == (4,)
    for slot, affordable in enumerate(card_mask):
        if affordable:
            action = Action(gate=1, hand_slot=slot, cell=int(np.argmax(
                legal_cell_mask(battle, 0, battle.hand[0][slot]).astype(np.int8))))
            assert decode_action(action, battle, 0) is not None


# --------------------------------------------------------------------- model
def test_patch_index_matches_cell_geometry():
    xy = torch.as_tensor(cell_xy() / np.array([ARENA_W, ARENA_H], dtype=np.float32))
    assert torch.equal(patch_index(xy), torch.as_tensor(cell_patch()))
    # Every one of the 144 patches must be reachable; the old x18/y32 mapping
    # collapsed the right and bottom halves onto the last row/column.
    assert len(set(cell_patch().tolist())) == 144


def test_model_forward_shapes(vocab, decks):
    model = _tiny_model(vocab).eval()
    battle = Battle([decks[0], decks[1]], seed=9)
    for _ in range(10):
        battle.step([])
    belief = OpponentBelief(vocab, deck_prior=decks, n_particles=16, seed=2)
    obs = build_observation(battle, 0, belief, vocab=vocab)
    tensors = obs_to_tensors([obs, obs], torch.device("cpu"))
    out = model(tensors["unit_ids"], tensors["unit_feats"], tensors["unit_mask"],
                tensors["global_feats"], tensors["hand_ids"], tensors["hand_mask"],
                tensors["deck_ids"], tensors["deck_mask"], tensors["next_id"],
                tensors["past_own"], tensors["past_opp"], tensors["belief"],
                cell_card=tensors["hand_ids"][:, 0])
    assert out["gate"].shape == (2,)
    assert out["value"].shape == (2, 1)
    assert out["cell"].shape == (2, N_CELLS)
    assert torch.isfinite(out["gate"]).all() and torch.isfinite(out["cell"]).all()

    sampler = ActionSampler(model, torch.device("cpu"), seed=1)
    actions, meta = sampler.sample([obs, obs])
    assert len(actions) == 2
    assert torch.isfinite(meta["logprob"]).all()
    assert torch.isfinite(meta["entropy"]).all()
    for a in actions:
        assert 0 <= a.hand_slot < 4
        assert 0 <= a.cell < N_CELLS


def test_match_loo_advantage_uses_other_match_returns():
    cfg = PPOConfig(advantage="match_loo")
    n = 4
    batch = {
        "reward": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "done": torch.tensor([False, True, False, True]),
        "old_value": torch.zeros(n),
        "phi": torch.zeros(n),
        "env_id": torch.tensor([0, 0, 1, 1]),
        "match_id": torch.tensor([10, 10, 11, 11]),
        "group_id": torch.tensor([1, 1, 1, 1]),
    }
    compute_gae(batch, cfg)
    # Match 10 won (R=1), match 11 drew (R=0): LOO is +1 / -1.
    assert torch.allclose(batch["advantages"], torch.tensor([1.0, 1.0, -1.0, -1.0]))
    assert torch.allclose(batch["returns"], torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_truncated_match_keeps_bootstrapped_gae():
    cfg = PPOConfig()
    n = 2
    batch = {
        "reward": torch.zeros(n), "done": torch.tensor([False, False]),
        "old_value": torch.tensor([0.0, 0.0]), "phi": torch.zeros(n),
        "bootstrap_value": torch.tensor([0.0, 2.0]),
        "bootstrap_phi": torch.zeros(n),
        "env_id": torch.tensor([0, 0]), "match_id": torch.tensor([1, 1]),
        "group_id": torch.tensor([1, 1]),
    }
    compute_gae(batch, cfg)
    # With gamma=1 and zero reward the truncated return is exactly V(s_T).
    assert batch["returns"][-1].item() == pytest.approx(cfg.gamma * 2.0)


def test_ppo_update_runs_and_kl_is_finite(vocab, decks):
    torch.manual_seed(0)
    model = _tiny_model(vocab)
    prior = type(model)(**model.config()).eval()
    prior.load_state_dict(model.state_dict())
    for p in prior.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = PPOConfig(epochs=1, minibatch=8, kl_leash="sum")
    updater = PPOUpdater(model, prior, opt, cfg, torch.device("cpu"))
    sampler = ActionSampler(model, torch.device("cpu"), seed=2)

    battle = Battle([decks[0], decks[1]], seed=11)
    belief = OpponentBelief(vocab, deck_prior=decks, n_particles=16, seed=4)
    buffer = RolloutBuffer()
    for step in range(12):
        battle.step([])
        obs = build_observation(battle, 0, belief, vocab=vocab)
        actions, meta = sampler.sample(obs)
        buffer.add(obs, actions[0], {k: v.reshape(1, *v.shape[1:]) for k, v in meta.items()},
                   reward=0.0, done=(step == 11), env_id=0, match_id=1, group_id=1)
    buffer.finalize_wait_targets()
    batch = buffer.build_batch()
    compute_gae(batch, cfg)
    normalize_advantages(batch)
    metrics = updater.update(batch, beta=0.1)
    for key in ("pg", "value", "kl", "entropy", "total"):
        assert math.isfinite(metrics[key]), key
    assert metrics["kl"] >= 0.0


# ----------------------------------------------------------------------- env
def test_env_decision_cadence_and_opponent(vocab, decks):
    env = CompetitiveEnv(
        decks, CompetitiveEnvConfig(decision_ticks=5, n_particles=16),
        seed=3, vocab=vocab,
    )
    obs = env.reset(decks[0], decks[1], seed=3,
                    opponent=ScriptedOpponent("tactical", seed=3))
    assert obs.global_feats.shape[0] == 37
    assert obs.cell_masks.shape == (4, N_CELLS)
    for _ in range(25):
        action = ScriptedOpponent("tactical", seed=5).act(obs, env)
        obs, reward, done, info = env.step(action)
        assert info["t_ms"] == env.decisions * 500.0
        if done:
            break
    assert env.decisions >= 1
    assert env.invalid_actions == 0


def test_env_episode_can_finish(vocab, decks):
    env = CompetitiveEnv(decks, CompetitiveEnvConfig(n_particles=16), seed=8, vocab=vocab)
    summary = env.run_episode(
        lambda obs, e: Action(gate=0), decks[0], decks[1],
        seed=8, opponent_name="wait",
    )
    assert summary["decisions"] >= 1
    assert "rows" in summary


# -------------------------------------------------------------------- replay
def test_parse_local_replay_and_build_examples(vocab):
    path = PROJECT_ROOT / "replays" / "replay_20261108_125502_Custom_Deck_vs_Giant_Beatdown.json"
    if not path.exists():
        pytest.skip("sample replay not present")
    record = parse_local_replay(path, vocab)
    assert record is not None
    assert len(record.decks[0]) == 8 and len(record.decks[1]) == 8
    assert record.events
    examples = build_examples_for_team(record, 1, vocab=vocab, wait_prob=1.0, seed=1)
    assert examples, "expected at least one BC example for the red seat"
    arrays = examples_to_arrays(examples, vocab=vocab)
    assert arrays["gate"].shape[0] == len(examples)
    assert arrays["cell_mask"].shape == (len(examples), N_CELLS)
    assert np.isfinite(arrays["belief"]).all()


def test_parse_hf_payload(vocab, decks):
    payload = {
        "battle": {
            "team": {"players": [{"deck": [{"card_key": c} for c in decks[0]]}]},
            "opponent": {"players": [{"deck": [{"card_key": c} for c in decks[1]]}]},
            "team_crowns": 1,
        },
        "events": [
            {
                "kind": "play_card", "side": "team", "card_key": decks[0][0],
                "source_fields": {"data_x": 3500, "data_y": 12000, "data_t": 100, "data_i": 2},
            }
        ],
    }
    record = parse_hf_payload(payload, tag="unit", vocab=vocab)
    assert record is not None
    assert record.replay_id == "unit"
    assert record.events[0].t_ms == pytest.approx(5000.0)
    assert record.events[0].x == pytest.approx(3.5)


# --------------------------------------------------------------- evaluation
def test_wilson_interval():
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4044, abs=1e-3)
    assert hi == pytest.approx(0.5956, abs=1e-3)
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.31


# ------------------------------------------------------------------- league
def test_league_samples_and_builds_scripted_opponent(vocab, decks, tmp_path):
    league = League(
        {"scripted_ratio": 1.0, "snapshot_ratio": 0.0, "self_ratio": 0.0},
        vocab, decks, checkpoint_dir=tmp_path, seed=1,
    )
    matchup = league.sample_matchup(step=0)
    assert matchup.opponent_kind == "scripted"
    assert len(matchup.learner_deck) == 8 and len(matchup.opponent_deck) == 8
    opponent = league.make_opponent(matchup)
    assert isinstance(opponent, ScriptedOpponent)
    assert league.sample_matchup(step=1).group_id != matchup.group_id
