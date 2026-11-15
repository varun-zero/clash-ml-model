"""PPO with GAE, optional match-level leave-one-out advantage, potential shaping,
KL leash to a frozen prior and adaptive beta -- the competitive v2 learner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .model import CompetitiveNet
from .obs import Observation, collate_observations
from .policy import (
    bernoulli_kl, categorical_kl, evaluate_actions, policy_distributions,
)

OBS_KEYS = (
    "unit_ids", "unit_feats", "unit_mask", "global_feats", "hand_ids",
    "hand_mask", "deck_ids", "deck_mask", "next_id", "past_own", "past_opp",
    "belief", "card_mask", "wait_mask",
)


@dataclass
class PPOConfig:
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    value_ce_coef: float = 0.25
    entropy_coef: float = 0.01
    aux_coef: float = 0.05
    wait_coef: float = 0.02
    kl_coef: float = 0.5
    kl_target: float = 0.02
    kl_adapt: float = 1.5
    kl_min: float = 0.005
    kl_max: float = 5.0
    kl_leash: str = "sum"          # "sum" | "max"
    advantage: str = "gae"         # "gae" | "match_loo"
    shaping: str = "potential"     # "potential" | "none"
    max_grad_norm: float = 1.0
    epochs: int = 4
    minibatch: int = 64
    normalize_advantage: bool = True
    critic_warmup_updates: int = 0


class RolloutBuffer:
    def __init__(self):
        self.rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.rows)

    def add(
        self,
        obs: Observation,
        action,
        meta: dict[str, torch.Tensor],
        *,
        reward: float,
        done: bool,
        env_id: int,
        match_id: int,
        group_id: int,
        wait_target: int = -1,
    ) -> None:
        policy = obs.policy_arrays()
        cell_mask = obs.cell_masks[action.hand_slot] if 0 <= action.hand_slot < obs.cell_masks.shape[0] else np.zeros(obs.cell_masks.shape[-1], dtype=bool)
        row = {
            **{k: np.asarray(v) for k, v in policy.items()},
            "card_mask": np.asarray(obs.card_mask),
            "gate": int(action.gate),
            "card": int(action.hand_slot),
            "cell": int(action.cell),
            "wait": int(action.wait_slot),
            "cell_mask": cell_mask,
            "wait_mask": obs.wait_mask,
            "old_logprob": float(meta["logprob"].reshape(-1)[0]),
            "old_value": float(meta["value"].reshape(-1)[0]),
            "reward": float(reward),
            "done": bool(done),
            "phi": float(obs.phi),
            "aux_opp_elixir": float(obs.aux_opp_elixir),
            "aux_opp_hand": np.asarray(obs.aux_opp_hand, dtype=np.float32),
            "aux_opp_next": int(obs.aux_opp_next),
            "env_id": int(env_id),
            "match_id": int(match_id),
            "group_id": int(group_id),
            "wait_target": int(wait_target),
            "bootstrap_value": 0.0,
            "bootstrap_phi": 0.0,
        }
        self.rows.append(row)

    def set_bootstrap(self, values: dict[int, tuple[float, float]]) -> None:
        """Attach V(s_T) / phi(s_T) to each env's final non-terminal row.

        Without this, a rollout that ends mid-match bootstraps with zero and
        biases every value/advantage near the 256-step horizon.
        """
        last: dict[int, int] = {}
        for i, row in enumerate(self.rows):
            last[int(row["env_id"])] = i
        for env_id, (value, phi) in values.items():
            i = last.get(int(env_id))
            if i is None or bool(self.rows[i]["done"]):
                continue
            self.rows[i]["bootstrap_value"] = float(value)
            self.rows[i]["bootstrap_phi"] = float(phi)

    # ------------------------------------------------------------------ batch
    def finalize_wait_targets(self, horizon: int = 12) -> None:
        """For wait decisions, point at the next card the learner actually plays.

        This is an auxiliary supervised label (never an input).  If the next
        play is already in hand, waiting for it is not meaningful and the
        target stays -1.
        """
        by_env: dict[int, list[int]] = {}
        for i, row in enumerate(self.rows):
            by_env.setdefault(row["env_id"], []).append(i)
        for idxs in by_env.values():
            for pos, i in enumerate(idxs):
                row = self.rows[i]
                if row["gate"] != 0:
                    continue
                # Only look inside the same match.
                for j in idxs[pos + 1: pos + 1 + horizon]:
                    nxt = self.rows[j]
                    if nxt["match_id"] != row["match_id"]:
                        break
                    if nxt["gate"] != 1:
                        continue
                    card_id = int(nxt["hand_ids"][nxt["card"]])
                    slot = int(np.where(row["deck_ids"] == card_id)[0][0]) if np.any(row["deck_ids"] == card_id) else -1
                    in_hand = bool(np.any(row["hand_mask"] & (row["hand_ids"] == card_id)))
                    row["wait_target"] = -1 if in_hand else slot
                    break

    def build_batch(self) -> dict[str, torch.Tensor]:
        rows = self.rows
        n = len(rows)
        batch: dict[str, torch.Tensor] = {}
        for k in OBS_KEYS:
            batch[k] = torch.as_tensor(np.stack([r[k] for r in rows]))
        batch["actions"] = {
            "gate": torch.as_tensor([r["gate"] for r in rows], dtype=torch.long),
            "card": torch.as_tensor([r["card"] for r in rows], dtype=torch.long),
            "cell": torch.as_tensor([r["cell"] for r in rows], dtype=torch.long),
            "wait": torch.as_tensor([r["wait"] for r in rows], dtype=torch.long),
            "cell_mask": torch.as_tensor(np.stack([r["cell_mask"] for r in rows]), dtype=torch.bool),
            "wait_mask": torch.as_tensor(np.stack([r["wait_mask"] for r in rows]), dtype=torch.bool),
        }
        batch["old_logprob"] = torch.as_tensor([r["old_logprob"] for r in rows], dtype=torch.float32)
        batch["old_value"] = torch.as_tensor([r["old_value"] for r in rows], dtype=torch.float32)
        batch["reward"] = torch.as_tensor([r["reward"] for r in rows], dtype=torch.float32)
        batch["done"] = torch.as_tensor([r["done"] for r in rows], dtype=torch.bool)
        batch["phi"] = torch.as_tensor([r["phi"] for r in rows], dtype=torch.float32)
        batch["aux_opp_elixir"] = torch.as_tensor([r["aux_opp_elixir"] for r in rows], dtype=torch.float32)
        batch["aux_opp_hand"] = torch.as_tensor(np.stack([r["aux_opp_hand"] for r in rows]), dtype=torch.float32)
        batch["aux_opp_next"] = torch.as_tensor([r["aux_opp_next"] for r in rows], dtype=torch.long)
        batch["wait_target"] = torch.as_tensor([r["wait_target"] for r in rows], dtype=torch.long)
        batch["bootstrap_value"] = torch.as_tensor([r["bootstrap_value"] for r in rows], dtype=torch.float32)
        batch["bootstrap_phi"] = torch.as_tensor([r["bootstrap_phi"] for r in rows], dtype=torch.float32)
        batch["env_id"] = torch.as_tensor([r["env_id"] for r in rows], dtype=torch.long)
        batch["match_id"] = torch.as_tensor([r["match_id"] for r in rows], dtype=torch.long)
        batch["group_id"] = torch.as_tensor([r["group_id"] for r in rows], dtype=torch.long)
        batch["returns"] = torch.zeros(n, dtype=torch.float32)
        batch["advantages"] = torch.zeros(n, dtype=torch.float32)
        return batch


def compute_gae(batch: dict[str, torch.Tensor], cfg: PPOConfig) -> None:
    n = int(batch["reward"].shape[0])
    env_ids = batch["env_id"].tolist()
    rewards = batch["reward"].tolist()
    values = batch["old_value"].tolist()
    dones = batch["done"].tolist()
    phis = batch["phi"].tolist()
    bv = batch.get("bootstrap_value")
    bp = batch.get("bootstrap_phi")
    bootstrap_values = (torch.zeros(n) if bv is None else bv).tolist()
    bootstrap_phis = (torch.zeros(n) if bp is None else bp).tolist()
    advantages = [0.0] * n
    returns = [0.0] * n

    starts: dict[int, list[int]] = {}
    for i, env in enumerate(env_ids):
        starts.setdefault(env, []).append(i)

    for idxs in starts.values():
        last_adv = 0.0
        for pos in range(len(idxs) - 1, -1, -1):
            i = idxs[pos]
            terminal = bool(dones[i])
            if terminal:
                next_v = 0.0
                next_phi = 0.0
                not_done = 0.0
            else:
                if pos + 1 < len(idxs):
                    ni = idxs[pos + 1]
                    next_v = values[ni]
                    next_phi = phis[ni]
                else:
                    # Rollout boundary: bootstrap from V(s_T) instead of 0.
                    next_v = bootstrap_values[i]
                    next_phi = bootstrap_phis[i]
                not_done = 1.0
            r = rewards[i]
            if cfg.shaping == "potential":
                r = r + cfg.gamma * not_done * next_phi - phis[i]
            delta = r + cfg.gamma * not_done * next_v - values[i]
            last_adv = delta + cfg.gamma * cfg.gae_lambda * not_done * last_adv
            advantages[i] = last_adv
            returns[i] = last_adv + values[i]

    if cfg.advantage == "match_loo":
        # ClashAI-style sparse group baseline: A_i = R_i - mean_{j != i} R_j,
        # assigned to every decision row of match i.
        by_match: dict[int, list[int]] = {}
        for i in range(n):
            by_match.setdefault(int(batch["match_id"][i]), []).append(i)
        by_group: dict[int, list[tuple[int, float]]] = {}
        completed: set[int] = set()
        for mid, idxs in by_match.items():
            terminal_idx = idxs[-1]
            if not bool(batch["done"][terminal_idx]):
                # The rollout ended mid-match; its return is not terminal and
                # must not contribute a spurious 0.0 to the group baseline.
                continue
            completed.add(mid)
            R = rewards[terminal_idx]
            gid = int(batch["group_id"][terminal_idx])
            by_group.setdefault(gid, []).append((mid, R))
        match_adv: dict[int, float] = {}
        match_ret: dict[int, float] = {}
        for gid, members in by_group.items():
            for mid, R in members:
                others = [r for m, r in members if m != mid]
                match_adv[mid] = 0.0 if not others else R - float(np.mean(others))
                match_ret[mid] = R
        for mid, idxs in by_match.items():
            if mid not in completed:
                continue  # keep the GAE estimate for truncated matches
            for i in idxs:
                advantages[i] = match_adv.get(mid, 0.0)
                returns[i] = match_ret.get(mid, 0.0)

    batch["advantages"] = torch.as_tensor(advantages, dtype=torch.float32)
    batch["returns"] = torch.as_tensor(returns, dtype=torch.float32)


def normalize_advantages(batch: dict[str, torch.Tensor]) -> None:
    a = batch["advantages"]
    if a.numel() > 1:
        batch["advantages"] = (a - a.mean()) / (a.std(unbiased=False) + 1e-8)


class PPOUpdater:
    def __init__(self, model: CompetitiveNet, prior: CompetitiveNet, optimizer: torch.optim.Optimizer,
                 cfg: PPOConfig, device: torch.device):
        self.model = model
        self.prior = prior
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = device

    def _mb(self, batch: dict[str, torch.Tensor], idx: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        obs = {k: batch[k][idx].to(self.device) for k in OBS_KEYS}
        actions = {k: batch["actions"][k][idx].to(self.device) for k in
                   ("gate", "card", "cell", "wait", "cell_mask", "wait_mask")}
        return obs, actions

    def update(self, batch: dict[str, torch.Tensor], beta: float) -> dict[str, float]:
        cfg = self.cfg
        n = int(batch["old_logprob"].shape[0])
        indices = torch.randperm(n)
        metrics: dict[str, list[float]] = {k: [] for k in (
            "pg", "value", "entropy", "aux", "kl", "kl_gate", "kl_card", "kl_cell",
            "kl_wait", "ratio", "clipfrac", "wait", "total",
        )}
        for epoch in range(cfg.epochs):
            for start in range(0, n, cfg.minibatch):
                idx = indices[start:start + cfg.minibatch]
                obs, actions = self._mb(batch, idx)
                out = evaluate_actions(self.model, obs, actions, self.device)
                old_logprob = batch["old_logprob"][idx].to(self.device)
                adv = batch["advantages"][idx].to(self.device)
                ret = batch["returns"][idx].to(self.device)
                ratio = torch.exp(out["logprob"] - old_logprob)
                pg1 = ratio * adv
                pg2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv
                pg_loss = -torch.min(pg1, pg2).mean()

                v = out["value"]
                value_loss = 0.5 * F.mse_loss(v, ret)
                bin_target = (ret.round().clamp(-3, 3) + 3).long()
                value_ce = F.cross_entropy(out["value_logits"], bin_target)
                value_loss = value_loss + cfg.value_ce_coef * value_ce

                aux_elixir = F.mse_loss(out["aux_elixir"], batch["aux_opp_elixir"][idx].to(self.device))
                aux_hand = F.binary_cross_entropy_with_logits(
                    out["aux_hand"], batch["aux_opp_hand"][idx].to(self.device),
                    pos_weight=torch.tensor(4.0, device=self.device),
                )
                aux_next = F.cross_entropy(out["aux_next"], batch["aux_opp_next"][idx].to(self.device))
                aux_loss = aux_elixir + aux_hand + 0.25 * aux_next
                wait_target = batch["wait_target"][idx].to(self.device)
                if bool((wait_target >= 0).any()):
                    wait_loss = F.cross_entropy(out["wait"], wait_target, ignore_index=-1)
                else:
                    # All-ignored cross entropy is NaN; skip the auxiliary head
                    # for minibatches with no valid wait target.
                    wait_loss = torch.zeros((), device=self.device)

                # KL leash to the frozen prior.
                with torch.no_grad():
                    prior_out = policy_distributions(self.prior, obs, actions["card"])
                kl_gate = bernoulli_kl(out["gate"], prior_out["gate"]).mean()
                play_rows = (actions["gate"] == 1).float()
                card_kl = categorical_kl(out["card"], prior_out["card"], out["card_mask"])
                cell_kl = categorical_kl(out["cell"], prior_out["cell"], actions["cell_mask"])
                kl_card = (card_kl * play_rows).sum() / play_rows.sum().clamp(min=1.0)
                kl_cell = (cell_kl * play_rows).sum() / play_rows.sum().clamp(min=1.0)
                wait_rows = (actions["gate"] == 0) & actions["wait_mask"].any(dim=-1)
                kl_wait = categorical_kl(out["wait"], prior_out["wait"], actions["wait_mask"])
                kl_wait = (kl_wait * wait_rows.float()).sum() / wait_rows.float().sum().clamp(min=1.0)
                if cfg.kl_leash == "max":
                    kl = torch.maximum(torch.maximum(kl_gate, kl_card), torch.maximum(kl_cell, kl_wait))
                else:
                    kl = kl_gate + kl_card + kl_cell + kl_wait

                entropy = out["entropy"].mean()
                total = (pg_loss + cfg.value_coef * value_loss + cfg.aux_coef * aux_loss
                         + cfg.wait_coef * wait_loss + beta * kl - cfg.entropy_coef * entropy)

                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip).float().mean()
                for name, val in (
                    ("pg", pg_loss), ("value", value_loss), ("entropy", entropy),
                    ("aux", aux_loss), ("kl", kl), ("kl_gate", kl_gate), ("kl_card", kl_card),
                    ("kl_cell", kl_cell), ("kl_wait", kl_wait), ("ratio", ratio.mean()),
                    ("clipfrac", clipfrac), ("wait", wait_loss), ("total", total),
                ):
                    metrics[name].append(float(val.detach().cpu()))
        return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}

    def adapt_beta(self, beta: float, measured_kl: float) -> float:
        cfg = self.cfg
        if measured_kl > cfg.kl_target * cfg.kl_adapt:
            beta *= cfg.kl_adapt
        elif measured_kl < cfg.kl_target / cfg.kl_adapt:
            beta /= cfg.kl_adapt
        return float(np.clip(beta, cfg.kl_min, cfg.kl_max))
