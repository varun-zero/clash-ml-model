"""Tests for the policy-improvement upgrades.

Covers:
1. Progressive KL target expansion (the leash must stop acting as a bungee
   cord once the critic has settled).
2. Top-p (nucleus) truncation of the tile distribution, including PPO
   log-probability consistency between the sampling and evaluation paths.
3. Low-temperature exploitation (temperature sharpens both heads).
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.model.alphastar_cr_net import (  # noqa: E402
    TOP_P_NUCLEUS,
    AlphaStarCRNet,
)
from cr_alphastar_agent.training.hybrid_ppo_trainer import (  # noqa: E402
    BC_KL_BETA_CAP,
    BC_KL_BETA_DECAY,
    KL_RAMP_END_UPDATE,
    KL_RAMP_START_UPDATE,
    KL_TARGET_MAX,
    HybridPPOTrainer,
)


# ------------------------------------------------------- 1. progressive KL ramp
def _trainer(tmp_path, **kwargs) -> HybridPPOTrainer:
    model = AlphaStarCRNet()
    return HybridPPOTrainer(
        model=model,
        checkpoint_dir=str(tmp_path),
        num_envs=2,
        rollout_steps=16,
        critic_warmup_updates=0,
        **kwargs,
    )


def test_kl_target_is_tight_before_the_ramp(tmp_path):
    trainer = _trainer(tmp_path)
    for update in (0, 10, 25, KL_RAMP_START_UPDATE):
        trainer.update_count = update
        assert trainer._current_kl_target() == pytest.approx(0.02), update


def test_kl_target_expands_linearly_to_the_maximum(tmp_path):
    trainer = _trainer(tmp_path)
    span = KL_RAMP_END_UPDATE - KL_RAMP_START_UPDATE

    trainer.update_count = KL_RAMP_START_UPDATE + span // 2
    mid = trainer._current_kl_target()
    expected = 0.02 + 0.5 * (KL_TARGET_MAX - 0.02)
    assert mid == pytest.approx(expected)

    trainer.update_count = KL_RAMP_END_UPDATE
    assert trainer._current_kl_target() == pytest.approx(KL_TARGET_MAX)

    # Never exceeds the maximum, even far past the ramp.
    trainer.update_count = KL_RAMP_END_UPDATE + 500
    assert trainer._current_kl_target() == pytest.approx(KL_TARGET_MAX)


def test_beta_cap_decays_per_update(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.update_count = 0
    first = trainer._current_beta_cap()
    assert first == pytest.approx(BC_KL_BETA_CAP)

    trainer.update_count = 100
    second = trainer._current_beta_cap()
    assert second == pytest.approx(BC_KL_BETA_CAP * (BC_KL_BETA_DECAY ** 100))
    assert second < first


def test_leash_is_relaxed_at_update_200(tmp_path):
    """Resuming at update 200 must already allow the policy to outgrow the
    teacher - that was the 60 % ceiling."""
    trainer = _trainer(tmp_path)
    trainer.update_count = 200
    assert trainer._current_kl_target() > 0.15
    assert trainer._current_beta_cap() < 10.0


# --------------------------------------------------------------- 2. top-p nucleus
def test_top_p_default_is_eighty_five_percent():
    assert TOP_P_NUCLEUS == 0.85
    assert AlphaStarCRNet().top_p == pytest.approx(0.85)


def test_top_p_filter_keeps_the_nucleus_and_drops_the_tail():
    torch.manual_seed(0)
    logits = torch.randn(3, 64)
    valid = torch.ones(3, 64, dtype=torch.bool)
    valid[1, 40:] = False

    keep = AlphaStarCRNet._top_p_filter(logits, valid, 0.85)

    for row in range(3):
        neg = torch.finfo(logits.dtype).min
        probs = torch.softmax(
            torch.where(valid[row], logits[row], torch.full_like(logits[row], neg)), dim=-1)
        mass = probs[keep[row]].sum().item()
        assert mass >= 0.85 - 1e-5, f"row {row} nucleus only covers {mass}"
        assert mass <= 1.0 + 1e-6

        # The kept set must be the highest-probability prefix.
        order = torch.argsort(probs, descending=True).tolist()
        positions = [i for i, idx in enumerate(order) if bool(keep[row][idx])]
        assert positions == list(range(len(positions)))

    # Illegal tiles can never be in the nucleus.
    assert keep[1, 40:].sum() == 0


def test_top_p_never_removes_every_legal_tile():
    logits = torch.zeros(1, 10)
    valid = torch.ones(1, 10, dtype=torch.bool)
    keep = AlphaStarCRNet._top_p_filter(logits, valid, 0.85)
    assert keep.any(dim=-1).all()


def _fake_obs(batch: int = 4) -> dict:
    return {
        "scalars": torch.randn(batch, 4),
        "hand_cards": torch.tensor([[0, 1, 2, 3, 4]] * batch),
        "entities": torch.randn(batch, 40, 7),
        "entity_mask": torch.ones(batch, 40, dtype=torch.bool),
        "arena_grid": torch.randn(batch, 6, 32, 18),
        "card_mask": torch.ones(batch, 5, dtype=torch.bool),
        "tile_mask": torch.ones(batch, 576, dtype=torch.bool),
    }


def test_top_p_keeps_ppo_log_probs_consistent():
    """The behaviour policy and the evaluated policy must agree, otherwise the
    PPO importance ratio is corrupted."""
    torch.manual_seed(1)
    model = AlphaStarCRNet()
    model.eval()
    with torch.no_grad():
        model.tile_head[-1].weight.normal_(0, 0.5)

    obs = _fake_obs()
    torch.manual_seed(7)
    action, logprob, _, _ = model.get_action_and_value(obs)
    _, logprob2, _, _ = model.get_action_and_value(obs, action=action)
    assert torch.allclose(logprob, logprob2, atol=1e-4)


def test_top_p_removes_the_low_probability_tail_from_the_support():
    torch.manual_seed(5)
    model = AlphaStarCRNet()
    model.eval()
    with torch.no_grad():
        model.tile_head[-1].weight.normal_(0, 0.02)   # nearly uniform head
    obs = _fake_obs(batch=2)

    with torch.no_grad():
        core = model.forward_core(
            obs["scalars"], obs["hand_cards"], obs["entities"],
            obs["entity_mask"], obs["arena_grid"])
        emb = model.chosen_action_emb(torch.zeros(2, dtype=torch.long))
        logits = model.tile_head(torch.cat([core, emb], dim=-1)).float()

    # Without truncation every tile is reachable; with top-p only the
    # high-probability nucleus survives.
    mask = torch.ones(2, 576, dtype=torch.bool)
    keep = AlphaStarCRNet._top_p_filter(logits, mask, 0.85)
    assert keep.sum(dim=-1).max() < 576
    assert keep.sum(dim=-1).min() >= 1


# ------------------------------------------------------------- 3. temperature
def test_temperature_sharpens_both_heads():
    torch.manual_seed(3)
    model = AlphaStarCRNet()
    model.eval()
    obs = _fake_obs()

    with torch.no_grad():
        _, cold, _, _ = model.get_action_and_value(obs, temperature=0.35)
        _, hot, _, _ = model.get_action_and_value(obs, temperature=1.0)

    assert cold.mean() > hot.mean(), "temperature did not sharpen the policy"


def test_temperature_does_not_change_the_deterministic_action():
    torch.manual_seed(11)
    model = AlphaStarCRNet()
    model.eval()
    obs = _fake_obs()
    with torch.no_grad():
        a_hot, _, _, _ = model.get_action_and_value(obs, deterministic=True)
        a_cold, _, _, _ = model.get_action_and_value(obs, deterministic=True, temperature=0.35)
    assert torch.equal(a_hot[:, 0], a_cold[:, 0])
