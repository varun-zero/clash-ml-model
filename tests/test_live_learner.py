"""Tests for the guarded post-match learning loop."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.config import (  # noqa: E402
    ENTITY_FEATURES, SPATIAL_CHANNELS, TOTAL_TILES,
)
from cr_alphastar_agent.env.live_recorder import LiveMatchRecorder  # noqa: E402
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet  # noqa: E402
from cr_alphastar_agent.training.live_learner import (  # noqa: E402
    MIN_DECISIONS, LiveLearner,
)


def make_checkpoint(path: Path) -> None:
    model = AlphaStarCRNet()
    torch.save({
        "model_state_dict": model.state_dict(),
        "global_step": 100,
        "update_count": 1,
    }, path)


def make_rows(count: int, weight: float = 1.0) -> list:
    """Synthetic replay rows with the exact observation shapes."""
    rows = []
    rng = np.random.default_rng(7)
    for i in range(count):
        obs = {
            "scalars": rng.random(7).astype(np.float32),
            "hand_cards": np.array([1, 2, 5, 9, 8], dtype=np.int64),
            "entities": np.zeros((40, ENTITY_FEATURES), dtype=np.float32),
            "entity_mask": np.ones(40, dtype=bool),
            "arena_grid": np.zeros((SPATIAL_CHANNELS, 32, 18), dtype=np.float32),
            "card_mask": np.array([True, True, True, True, True]),
            "tile_mask": np.ones(TOTAL_TILES, dtype=bool),
            "card_tile_mask": np.ones((5, TOTAL_TILES), dtype=bool),
        }
        # keep at least one tower unmasked like the real adapter does
        obs["entity_mask"][0] = False
        rows.append({
            "t_ms": i * 400,
            "obs": obs,
            "action": [i % 4, (i * 37) % TOTAL_TILES],
            "weight": weight,
            "verified": weight > 0,
            "session": {"match_index": 1},
        })
    return rows


@pytest.fixture
def learner(tmp_path):
    ckpt = tmp_path / "latest.pt"
    make_checkpoint(ckpt)
    return LiveLearner(str(ckpt), device="cpu",
                       log_path=str(tmp_path / "learning.jsonl"))


def test_recorder_handles_cuda_tensors(tmp_path):
    """Live observations are built on the CUDA device in this environment."""
    path = tmp_path / "replay.jsonl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs = {
        "scalars": torch.randn(1, 7, device=device),
        "hand_cards": torch.tensor([[1, 2, 3, 4, 5]], device=device),
    }
    recorder = LiveMatchRecorder(str(path))
    recorder.record(obs, [0, 100], weight=1.0, verified=True)
    recorder.close()
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["verified"] is True
    assert row["weight"] == 1.0
    assert len(row["obs"]["scalars"][0]) == 7


def test_legacy_checkpoint_is_padded_to_the_new_observation_shape():
    from cr_alphastar_agent.model.alphastar_cr_net import adapt_legacy_state_dict
    model = AlphaStarCRNet()
    state = model.state_dict()
    old = dict(state)
    old["scalar_fc.0.weight"] = old["scalar_fc.0.weight"][:, :164].clone()
    old["entity_proj.weight"] = old["entity_proj.weight"][:, :7].clone()
    adapted = adapt_legacy_state_dict(old)
    AlphaStarCRNet().load_state_dict(adapted)
    assert adapted["scalar_fc.0.weight"].shape[1] == 167
    assert adapted["entity_proj.weight"].shape[1] == 9


def test_no_update_for_unverified_outcome(learner):
    result = learner.after_match(make_rows(40), "UNVERIFIED", deployments=20)
    assert result.updated is False
    assert "no outcome label" in result.reason


def test_no_update_with_too_few_verified_decisions(learner):
    result = learner.after_match(make_rows(5), "WIN", deployments=2)
    assert result.updated is False
    assert result.samples == 5
    assert "need" in result.reason


def test_zero_weight_rows_are_ignored(learner):
    rows = make_rows(20, weight=1.0) + make_rows(30, weight=0.0)
    result = learner.after_match(rows, "LOSS", deployments=20)
    assert result.samples == 20


def test_guarded_update_writes_checkpoint_and_log(learner, tmp_path):
    result = learner.after_match(make_rows(40), "WIN", deployments=20)
    assert result.updated is True
    assert result.samples == 40
    assert Path(result.checkpoint).exists()
    assert 0.0 <= result.mean_kl <= 0.02
    assert np.isfinite(result.mean_kl)

    log = tmp_path / "learning.jsonl"
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["updated"] is True
    assert records[-1]["match_result"] == "WIN"
    assert records[-1]["samples"] == 40


def test_checkpoint_keeps_trainer_compatible_keys(learner):
    learner.after_match(make_rows(40), "WIN", deployments=20)
    payload = torch.load(learner.checkpoint_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in payload
    assert payload["global_step"] == 140  # 100 + 40 verified decisions
    assert payload["update_count"] == 1
    assert payload["live_learning_updates"] == 1
    assert "optimizer_state_dict" not in payload  # never fabricate a trainer state
