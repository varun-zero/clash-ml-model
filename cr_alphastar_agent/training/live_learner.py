"""Conservative, guarded learning from verified live matches.

A live match gives us 200-400 real observations but only **one** outcome
label.  Naively running PPO on that would destroy a trained policy.  This
module therefore does *reinforced behaviour cloning*: it reuses the actions
that were actually verified on screen and weights them by the match outcome,
with a strict KL leash, a validation check and an automatic rollback.

Design rules (all enforced in code):

* only decisions whose deployment was **verified** are used as targets;
* ``UNVERIFIED`` matches produce no gradient at all;
* at least ``MIN_DECISIONS`` verified actions are required;
* the learning rate is tiny and only one epoch is run;
* the update is rejected if the mean KL from the old policy exceeds
  ``MAX_KL`` on a held-out slice, or if validation loss regresses;
* the previous checkpoint is backed up atomically before every write.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from cr_alphastar_agent.config import TOTAL_TILES
from cr_alphastar_agent.model.alphastar_cr_net import (
    AlphaStarCRNet, adapt_legacy_state_dict,
)

MIN_DECISIONS = 12
LEARNING_RATE = 1e-5
MAX_KL = 0.02
MAX_VAL_REGRESSION = 0.05
HOLDOUT_FRACTION = 0.25
MAX_EPOCHS = 1
GRAD_CLIP = 0.5


@dataclass
class LearnResult:
    """What happened during the post-match update."""

    updated: bool
    reason: str
    samples: int = 0
    loss_before: float = 0.0
    loss_after: float = 0.0
    mean_kl: float = 0.0
    checkpoint: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "updated": self.updated,
            "reason": self.reason,
            "samples": self.samples,
            "loss_before": round(self.loss_before, 6),
            "loss_after": round(self.loss_after, 6),
            "mean_kl": round(self.mean_kl, 6),
            "checkpoint": self.checkpoint,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def _tensor(obs: Dict[str, Any], key: str, device: torch.device,
            dtype: torch.dtype, expected_ndim: int) -> torch.Tensor:
    """Return one observation with the network's unbatched shape.

    The live bridge records tensors that already carry a leading batch
    dimension of 1 (``scalars`` is ``[1, 4]``, ``entities`` is ``[1, 40, 7]``,
    ...).  ``torch.stack`` would add another one, so strip that singleton
    here.  Older replays that already store unbatched arrays still work.
    """
    value = obs.get(key)
    if value is None:
        raise KeyError(key)
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    while tensor.ndim > expected_ndim and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != expected_ndim:
        raise ValueError(f"{key}: expected {expected_ndim} dims, got {tuple(tensor.shape)}")
    return tensor


def _to_batch(rows: Sequence[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
    """Stack the replay rows into the network's fixed-shape observation dict."""
    first = rows[0]["obs"]

    shapes = {
        "scalars": 1, "hand_cards": 1, "entities": 2, "entity_mask": 1,
        "arena_grid": 3, "card_mask": 1, "tile_mask": 1, "card_tile_mask": 2,
    }
    dtypes = {
        "scalars": torch.float32, "hand_cards": torch.int64,
        "entities": torch.float32, "entity_mask": torch.bool,
        "arena_grid": torch.float32, "card_mask": torch.bool,
        "tile_mask": torch.bool, "card_tile_mask": torch.bool,
    }

    def stack(key: str) -> torch.Tensor:
        return torch.stack([
            _tensor(row["obs"], key, device, dtypes[key], shapes[key])
            for row in rows
        ])

    batch = {key: stack(key) for key in shapes}
    actions = torch.tensor([list(row["action"]) for row in rows],
                           dtype=torch.int64, device=device)
    weights = torch.tensor([float(row.get("weight", 1.0)) for row in rows],
                           dtype=torch.float32, device=device)
    return {"obs": batch, "actions": actions, "weights": weights}


def _mean_kl(ref_probs: torch.Tensor, new_probs: torch.Tensor) -> float:
    """KL(reference || new), ignoring masked (zero-probability) slots.

    ``log(0)`` times zero is NaN, so masked entries must be excluded before
    the product is formed.
    """
    keep = ref_probs > 0
    if not bool(keep.any()):
        return 0.0
    r = ref_probs[keep]
    n = new_probs[keep].clamp_min(1e-8)
    return float((r * (r.log() - n.log())).sum(dim=-1).mean())


def _policy_stats(model, batch: Dict[str, torch.Tensor]):
    """Card and tile log-probabilities for the recorded actions."""
    obs = batch["obs"]
    core = model.forward_core(obs["scalars"], obs["hand_cards"], obs["entities"],
                              obs["entity_mask"], obs["arena_grid"])
    card_logits = model.card_head(core).float()
    card_mask = obs["card_mask"].bool()
    card_dist = torch.distributions.Categorical(
        logits=card_logits.masked_fill(~card_mask, -1e9)
    )

    actions = batch["actions"]
    slots = actions[:, 0]
    tiles = actions[:, 1]
    card_logp = card_dist.log_prob(slots)

    emb = model.chosen_action_emb(slots)
    tile_logits = model.tile_head(torch.cat([core, emb], dim=-1)).float()
    legal = obs["card_tile_mask"][torch.arange(len(slots), device=slots.device), slots]
    tile_logits = tile_logits.masked_fill(~legal, -1e9)
    tile_dist = torch.distributions.Categorical(logits=tile_logits)
    tile_logp = tile_dist.log_prob(tiles)

    play = (slots != 4).float()
    weighted = batch["weights"] * (card_logp + play * tile_logp)
    return {
        "card_logp": card_logp,
        "tile_logp": tile_logp,
        "loss": -weighted.mean(),
        "card_probs": card_dist.probs,
    }


class LiveLearner:
    """Applies one guarded fine-tuning step after a match."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None,
                 log_path: Optional[str] = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.log_path = log_path or os.path.join(
            os.path.dirname(os.path.abspath(checkpoint_path)), "live_learning_log.jsonl"
        )
        self.model = AlphaStarCRNet().to(self.device)
        self._load_weights(self.model)
        self.model.train()

    def _load_weights(self, model) -> None:
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(self.checkpoint_path)
        payload = torch.load(self.checkpoint_path, map_location=self.device,
                             weights_only=False)
        state = payload.get("model_state_dict", payload)
        model.load_state_dict(adapt_legacy_state_dict(state))

    # -------------------------------------------------------------- public API
    def after_match(self, rows: Sequence[Dict[str, Any]], result: str,
                    deployments: int) -> LearnResult:
        """Run a guarded update using the just-finished match."""
        if result not in ("WIN", "LOSS"):
            return LearnResult(False, f"no outcome label ({result})")

        usable = [r for r in rows if r.get("weight", 0.0) > 0]
        if len(usable) < MIN_DECISIONS:
            return LearnResult(False, f"only {len(usable)} verified decisions "
                                      f"(need {MIN_DECISIONS})", samples=len(usable))

        batch_all = _to_batch(usable, self.device)
        n = len(usable)
        n_val = max(1, int(n * HOLDOUT_FRACTION))

        def slice_batch(start: int, end: int) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for key, value in batch_all.items():
                if isinstance(value, dict):
                    result[key] = {k: v[start:end] for k, v in value.items()}
                else:
                    result[key] = value[start:end]
            return result

        val = slice_batch(0, n_val)
        train = slice_batch(n_val, n)
        if len(train["actions"]) < MIN_DECISIONS // 2:
            return LearnResult(False, "not enough training rows after holdout", samples=n)

        reference = None
        try:
            from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
            reference = AlphaStarCRNet().to(self.device)
            self._load_weights(reference)
            reference.eval()
        except Exception as exc:  # pragma: no cover - defensive
            return LearnResult(False, f"could not build reference policy: {exc}", samples=n)

        with torch.no_grad():
            before = _policy_stats(self.model, val)
            ref = _policy_stats(reference, val)
            loss_before = float(before["loss"])
            kl = _mean_kl(ref["card_probs"], before["card_probs"])

        if kl > MAX_KL:
            return LearnResult(False, f"policy already drifted (KL={kl:.4f} > {MAX_KL})",
                               samples=n, loss_before=loss_before, mean_kl=kl)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE,
                                      weight_decay=1e-4)
        last_loss = loss_before
        for _ in range(MAX_EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            stats = _policy_stats(self.model, train)
            stats["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            optimizer.step()
            last_loss = float(stats["loss"])

        with torch.no_grad():
            after = _policy_stats(self.model, val)
            after_ref = _policy_stats(reference, val)
            loss_after = float(after["loss"])
            new_kl = _mean_kl(ref["card_probs"], after["card_probs"])

        if loss_after > loss_before * (1.0 + MAX_VAL_REGRESSION):
            self._load_weights(self.model)   # rollback in memory
            outcome = LearnResult(False,
                                  f"validation regressed {loss_before:.4f} -> {loss_after:.4f}",
                                  samples=n, loss_before=loss_before,
                                  loss_after=loss_after, mean_kl=new_kl)
            self._log(outcome, result)
            return outcome
        if new_kl > MAX_KL:
            self._load_weights(self.model)
            outcome = LearnResult(False, f"update exceeded KL leash ({new_kl:.4f})",
                                  samples=n, loss_before=loss_before,
                                  loss_after=loss_after, mean_kl=new_kl)
            self._log(outcome, result)
            return outcome

        saved = self._save(n)
        outcome = LearnResult(True, f"guarded BC update after {result}", samples=n,
                              loss_before=loss_before, loss_after=loss_after,
                              mean_kl=new_kl, checkpoint=saved)
        self._log(outcome, result)
        return outcome

    # ----------------------------------------------------------------- helpers
    def _save(self, samples: int = 0) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_path)), exist_ok=True)
        backup = self.checkpoint_path + ".pre_live.bak"
        if os.path.exists(self.checkpoint_path):
            shutil.copy2(self.checkpoint_path, backup)
        payload: Dict[str, Any] = {}
        if os.path.exists(self.checkpoint_path):
            try:
                payload = torch.load(self.checkpoint_path, map_location="cpu",
                                     weights_only=False) or {}
            except Exception:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload["model_state_dict"] = {
            k: v.detach().cpu() for k, v in self.model.state_dict().items()
        }
        payload["global_step"] = int(payload.get("global_step", 0)) + int(samples)
        payload["live_learning_updates"] = int(payload.get("live_learning_updates", 0)) + 1
        payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        tmp = self.checkpoint_path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.checkpoint_path)
        return self.checkpoint_path

    def _log(self, outcome: LearnResult, result: str) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            record = outcome.as_dict()
            record["match_result"] = result
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception:
            pass


__all__ = ["LearnResult", "LiveLearner", "MIN_DECISIONS"]
