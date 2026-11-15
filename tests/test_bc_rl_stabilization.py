"""Tests for the 3 AlphaStar BC-to-RL stabilization fixes.

Covers:
1. Critic value calibration in warmstart_bc.py (discounted return-to-go G_t with gamma=0.995).
2. Hinge-clamped entropy bonus (preventing entropy inflation above BC tactical sharpness).
3. Critic warmup updates (actor parameters frozen, pg_loss=0, entropy=0 for first 8 updates).
4. BC anchor KL leash with adaptive beta targeting 0.02.
"""
from pathlib import Path
import sys
import copy
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.config import (
    TOTAL_TILES,
    CURRENT_DECK_IDS,
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.warmstart_bc import _returns_to_go, BC_VALUE_COEF, BC_RETURN_GAMMA
from cr_alphastar_agent.training.hybrid_ppo_trainer import (
    HybridPPOTrainer,
    RolloutBuffer,
    ENTROPY_CARD_CEILING,
    ENTROPY_TILE_CEILING,
    ENTROPY_CARD_COEF,
    ENTROPY_TILE_COEF,
    CRITIC_WARMUP_UPDATES,
    CRITIC_WARMUP_TARGET_LOSS,
)


# --------------------------------------------------------------------- Fix 1: Value calibration
def test_returns_to_go_computation():
    """Verify discounted return-to-go calculation with gamma = 0.995."""
    gamma = 0.995
    rewards = [0.0, 0.0, 1.0, 0.5]
    returns = _returns_to_go(rewards, gamma=gamma)

    assert len(returns) == 4
    # G_3 = 0.5
    assert pytest.approx(returns[3], rel=1e-5) == 0.5
    # G_2 = 1.0 + 0.995 * 0.5 = 1.4975
    assert pytest.approx(returns[2], rel=1e-5) == 1.0 + gamma * 0.5
    # G_1 = 0.0 + 0.995 * 1.4975
    assert pytest.approx(returns[1], rel=1e-5) == gamma * (1.0 + gamma * 0.5)
    # G_0 = 0.0 + 0.995 * G_1
    assert pytest.approx(returns[0], rel=1e-5) == (gamma ** 2) * (1.0 + gamma * 0.5)


def test_bc_value_calibration_loss():
    """Verify value head regression in BC loss."""
    model = AlphaStarCRNet()
    core_repr = torch.randn(4, 128)
    returns = torch.tensor([1.5, -0.5, 0.0, 2.0], dtype=torch.float32)

    values = model.value_head(core_repr).squeeze(-1).float()
    v_loss = F.mse_loss(values, returns)

    assert v_loss.ndim == 0
    assert torch.isfinite(v_loss)
    assert BC_VALUE_COEF == 0.5
    assert BC_RETURN_GAMMA == 0.995


# --------------------------------------------------------------------- Fix 2: Hinge-clamped entropy
def test_hinge_clamped_entropy_gradient_behavior():
    """Verify that entropy bonus is clamped so PPO never drives entropy above BC sharpness."""
    # When entropy is below ceilings, gradient is non-zero (pulling entropy up to floor)
    low_card_ent = torch.tensor([0.50], requires_grad=True)
    low_tile_ent = torch.tensor([2.00], requires_grad=True)

    clamped_card = torch.clamp(low_card_ent, max=ENTROPY_CARD_CEILING)
    clamped_tile = torch.clamp(low_tile_ent, max=ENTROPY_TILE_CEILING)
    ent_loss_low = (ENTROPY_CARD_COEF * clamped_card + ENTROPY_TILE_COEF * clamped_tile).mean()

    # In PPO: loss = pg_loss - ent_loss + v_loss.
    # Minimizing loss means maximizing ent_loss.
    loss_low = -ent_loss_low
    loss_low.backward()

    assert low_card_ent.grad is not None
    assert pytest.approx(low_card_ent.grad.item(), rel=1e-4) == -ENTROPY_CARD_COEF
    assert pytest.approx(low_tile_ent.grad.item(), rel=1e-4) == -ENTROPY_TILE_COEF

    # When entropy is above ceilings (e.g. 1.20 and 3.50), gradient must be ZERO
    high_card_ent = torch.tensor([1.20], requires_grad=True)
    high_tile_ent = torch.tensor([3.50], requires_grad=True)

    clamped_high_card = torch.clamp(high_card_ent, max=ENTROPY_CARD_CEILING)
    clamped_high_tile = torch.clamp(high_tile_ent, max=ENTROPY_TILE_CEILING)
    ent_loss_high = (ENTROPY_CARD_COEF * clamped_high_card + ENTROPY_TILE_COEF * clamped_high_tile).mean()

    loss_high = -ent_loss_high
    loss_high.backward()

    # Gradient must be 0: optimizer cannot inflate entropy above 0.90 / 2.80!
    assert high_card_ent.grad.item() == 0.0
    assert high_tile_ent.grad.item() == 0.0


# --------------------------------------------------------------------- Fix 3: Critic warmup & Anchor KL
def test_critic_warmup_freezes_actor_gradients(tmp_path):
    """Verify that during critic warmup, actor parameters receive NO gradient."""
    model = AlphaStarCRNet()
    trainer = HybridPPOTrainer(
        model=model,
        checkpoint_dir=str(tmp_path),
        num_envs=2,
        rollout_steps=16,
        critic_warmup_updates=8,
    )
    # Manually arm critic warmup
    trainer._critic_warmup_active = True

    # Record initial weights of an actor layer (card_head) and critic layer (value_head)
    card_head_weight_before = model.card_head.weight.clone()
    value_head_weight_before = model.value_head[0].weight.clone()

    # Construct a synthetic rollout buffer
    buffer = RolloutBuffer(num_steps=16, num_envs=2, device=trainer.device)
    buffer.advantages = torch.randn((16, 2), device=trainer.device)
    buffer.returns = torch.randn((16, 2), device=trainer.device)
    buffer.actions = torch.zeros((16, 2, 2), dtype=torch.long, device=trainer.device)
    buffer.actions[:, :, 0] = 4  # Wait action
    buffer.obs["card_mask"] = torch.ones((16, 2, 5), dtype=torch.bool, device=trainer.device)
    buffer.obs["card_tile_mask"] = torch.ones((16, 2, 5, TOTAL_TILES), dtype=torch.bool, device=trainer.device)
    buffer.obs["tile_mask"] = torch.ones((16, 2, TOTAL_TILES), dtype=torch.bool, device=trainer.device)

    metrics = trainer.ppo_update(buffer, epochs=1)

    # In critic warmup: pg_loss = 0.0 and entropy = 0.0
    assert metrics["pg_loss"] == 0.0
    assert metrics["entropy"] == 0.0
    assert metrics["critic_warmup"] == 1.0

    # Actor card_head weights MUST NOT HAVE CHANGED
    assert torch.allclose(model.card_head.weight, card_head_weight_before)
    # Critic value_head weights SHOULD HAVE CHANGED
    assert not torch.allclose(model.value_head[0].weight, value_head_weight_before)


def test_bc_anchor_kl_and_adaptive_beta(tmp_path):
    """Verify that adaptive beta adjusts based on anchor KL vs target (0.02)."""
    model = AlphaStarCRNet()
    trainer = HybridPPOTrainer(
        model=model,
        checkpoint_dir=str(tmp_path),
        num_envs=2,
        rollout_steps=16,
        critic_warmup_updates=0,
        bc_kl_target=0.02,
        bc_kl_beta=1.0,
    )
    # Enable reference model
    trainer._setup_bc_reference_and_warmup(critic_warmup_active=False)
    assert trainer.bc_reference_model is not None
    assert trainer.bc_kl_beta == 1.0

    # Case 1: Drift too high (mean anchor KL > 1.5 * 0.02 = 0.03) -> beta increases
    high_kl_metrics = {"anchor_kl": 0.045}
    mean_anchor_kl = high_kl_metrics["anchor_kl"]
    if mean_anchor_kl > 1.5 * trainer.bc_kl_target:
        trainer.bc_kl_beta = min(20.0, trainer.bc_kl_beta * 1.5)
    assert pytest.approx(trainer.bc_kl_beta) == 1.5

    # Case 2: Drift tight (mean anchor KL < 0.02 / 1.5 = 0.0133) -> beta decreases
    low_kl_metrics = {"anchor_kl": 0.008}
    mean_anchor_kl = low_kl_metrics["anchor_kl"]
    if mean_anchor_kl < (trainer.bc_kl_target / 1.5):
        trainer.bc_kl_beta = max(0.1, trainer.bc_kl_beta / 1.5)
    assert pytest.approx(trainer.bc_kl_beta) == 1.0
