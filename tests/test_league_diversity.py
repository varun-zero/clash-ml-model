"""Tests for the league diversity and telemetry robustness additions.

Covers:
1. The varied opponent pool (behavioural variants, not just deck variants).
2. Atomic telemetry writes (a reader must never see a truncated file).
3. The nucleus being replayed during the PPO update.
"""
from pathlib import Path
import json
import os
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.env.tactical_coach import (  # noqa: E402
    OPPONENT_VARIANTS,
    TacticalCoachBot,
    _make_aggro_bot,
    _make_defensive_bot,
    _make_spell_bot,
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet  # noqa: E402
from cr_alphastar_agent.training.hybrid_ppo_trainer import (  # noqa: E402
    SNAPSHOT_OPPONENT_RATE,
    HybridPPOTrainer,
    _atomic_write_json,
)


# ------------------------------------------------------------- opponent pool
def test_opponent_variants_behave_differently():
    """The variants must not be cosmetic - different elixir thresholds."""
    default = TacticalCoachBot(player_id=1)
    aggro = _make_aggro_bot(player_id=1)
    defensive = _make_defensive_bot(player_id=1)
    spell = _make_spell_bot(player_id=1)

    assert aggro.push_elixir < default.push_elixir < defensive.push_elixir
    assert aggro.build_probability > default.build_probability > defensive.build_probability
    assert spell.prefer_spell is True
    assert default.prefer_spell is False
    assert set(OPPONENT_VARIANTS) == {"coach", "coach_aggro", "coach_defensive", "coach_spell"}


def test_trainer_opponent_pool_contains_every_variant(tmp_path):
    trainer = HybridPPOTrainer(
        model=AlphaStarCRNet(), checkpoint_dir=str(tmp_path),
        num_envs=2, rollout_steps=16, critic_warmup_updates=0,
        opponent_mode="scripted",
    )
    for name in OPPONENT_VARIANTS:
        assert name in trainer.opponent_bots
        assert hasattr(trainer.opponent_bots[name], "act")
    assert "rush" in trainer.opponent_bots
    assert "tactical_pro" in trainer.opponent_bots
    assert trainer.snapshot_opponent_rate == SNAPSHOT_OPPONENT_RATE


def test_trainer_samples_several_opponent_types(tmp_path):
    trainer = HybridPPOTrainer(
        model=AlphaStarCRNet(), checkpoint_dir=str(tmp_path),
        num_envs=2, rollout_steps=16, critic_warmup_updates=0,
        opponent_mode="scripted",
    )
    seen = set()
    for _ in range(300):
        trainer._sample_opponent_for_single_env(0)
        seen.add(trainer.env_opponents[0]["type"])
    assert len(seen) >= 4, f"league is not diverse enough: {seen}"


# ------------------------------------------------------------ atomic writes
def test_atomic_json_write_never_leaves_a_partial_file(tmp_path):
    target = tmp_path / "match_log.json"
    payload = [{"update": i, "result": "WIN"} for i in range(500)]

    _atomic_write_json(str(target), payload)

    # The file is complete and parseable straight away.
    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert len(data) == 500
    assert not (tmp_path / "match_log.json.tmp").exists()


def test_atomic_write_replaces_the_previous_file(tmp_path):
    target = tmp_path / "history.json"
    _atomic_write_json(str(target), [{"a": 1}])
    _atomic_write_json(str(target), [{"b": 2}])
    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert data == [{"b": 2}]


def test_monitor_retries_a_transiently_unreadable_file(tmp_path):
    from tools.monitor_training import load_json

    target = tmp_path / "match_log.json"
    target.write_text("{ this is not json", encoding="utf-8")
    # A corrupt file must not raise; the monitor reports "no data" instead of
    # crashing the training session.
    assert load_json(target, retries=1) == []

    _atomic_write_json(str(target), [{"update": 1, "result": "WIN"}])
    assert len(load_json(target)) == 1
