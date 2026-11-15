"""Behaviour-cloning bootstrap for the competitive v2 model."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .cards import CardVocab, default_vocab
from .model import CompetitiveNet
from .obs import GLOBAL_FEATURE_DIM, UNIT_FEATURE_DIM
from .policy import masked_categorical_logprob, policy_distributions
from .replay import CompetitiveDataset


@dataclass
class BCConfig:
    epochs: int = 4
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.0
    gate_coef: float = 1.0
    card_coef: float = 1.0
    cell_coef: float = 1.0
    wait_coef: float = 0.2
    value_coef: float = 0.2
    aux_coef: float = 0.02
    grad_clip: float = 1.0
    val_fraction: float = 0.1
    seed: int = 0


OBS_KEYS = (
    "unit_ids", "unit_feats", "unit_mask", "global_feats", "hand_ids",
    "hand_mask", "deck_ids", "deck_mask", "next_id", "past_own", "past_opp",
    "belief",
)


def _split_by_replay(dataset: CompetitiveDataset, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(dataset)
    ids = dataset.arrays.get("replay_id")
    if ids is None:
        perm = np.random.default_rng(seed).permutation(n)
        cut = int(n * (1.0 - val_fraction))
        return perm[:cut], perm[cut:]
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_fraction)) if len(unique) > 1 else 0
    val_ids = set(unique[:n_val].tolist())
    val_mask = np.isin(ids, list(val_ids))
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def _batch_tensors(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for k in OBS_KEYS:
        t = torch.as_tensor(batch[k], device=device)
        if k in ("unit_mask", "hand_mask", "deck_mask"):
            t = t.bool()
        elif k in ("unit_ids", "hand_ids", "deck_ids", "next_id", "past_own", "past_opp"):
            t = t.float()
        else:
            t = t.float()
        out[k] = t
    for k in ("card_mask", "cell_mask", "wait_mask", "cell_masks"):
        if k in batch:
            out[k] = torch.as_tensor(batch[k], device=device).bool()
    return out


def _losses(model: CompetitiveNet, tensors: dict[str, torch.Tensor], batch: dict[str, np.ndarray],
            device: torch.device, cfg: BCConfig) -> tuple[torch.Tensor, dict[str, float]]:
    gate = torch.as_tensor(batch["gate"], device=device).float()
    hand_slot = torch.as_tensor(batch["hand_slot"], device=device).long()
    cell = torch.as_tensor(batch["cell"], device=device).long()
    wait_slot = torch.as_tensor(batch["wait_slot"], device=device).long()
    cell_mask = torch.as_tensor(batch["cell_mask"], device=device).bool()
    wait_mask = tensors["wait_mask"]
    value_target = torch.as_tensor(batch["value_target"], device=device)

    card_ids = tensors["hand_ids"].gather(1, hand_slot.view(-1, 1).clamp(min=0)).squeeze(1)
    dist = policy_distributions(model, tensors, card_ids)
    gate_loss = F.binary_cross_entropy_with_logits(dist["gate"], gate)

    play = gate > 0.5
    if play.any():
        card_loss = -masked_categorical_logprob(
            dist["card"][play], dist["card_mask"][play], hand_slot[play]
        ).mean()
        cell_loss = -masked_categorical_logprob(
            dist["cell"][play], cell_mask[play], cell[play]
        ).mean()
    else:
        card_loss = torch.zeros((), device=device)
        cell_loss = torch.zeros((), device=device)

    wait_rows = (~play) & (wait_slot >= 0)
    if wait_rows.any():
        wait_loss = -masked_categorical_logprob(
            dist["wait"][wait_rows], wait_mask[wait_rows], wait_slot[wait_rows]
        ).mean()
    else:
        wait_loss = torch.zeros((), device=device)

    bin_target = (value_target.clamp(-3, 3) + 3).long()
    value_loss = F.cross_entropy(dist["value_logits"], bin_target)

    aux_elixir = F.mse_loss(dist["aux_elixir"], torch.as_tensor(batch["aux_opp_elixir"], device=device))
    aux_hand = F.binary_cross_entropy_with_logits(
        dist["aux_hand"], torch.as_tensor(batch["aux_opp_hand"], device=device),
        pos_weight=torch.tensor(4.0, device=device),
    )
    aux_next = F.cross_entropy(
        dist["aux_next"], torch.as_tensor(batch["aux_opp_next"], device=device).long()
    )
    aux_loss = aux_elixir + aux_hand + 0.25 * aux_next

    total = (cfg.gate_coef * gate_loss + cfg.card_coef * card_loss
             + cfg.cell_coef * cell_loss + cfg.wait_coef * wait_loss
             + cfg.value_coef * value_loss + cfg.aux_coef * aux_loss)

    with torch.no_grad():
        gate_acc = ((torch.sigmoid(dist["gate"]) >= 0.5).float() == gate).float().mean() if gate.numel() else torch.zeros(())
        card_pred = dist["card"].argmax(-1)
        card_acc = (card_pred[play] == hand_slot[play]).float().mean() if play.any() else torch.zeros(())
        cell_pred = dist["cell"].argmax(-1)
        cell_acc = (cell_pred[play] == cell[play]).float().mean() if play.any() else torch.zeros(())
        if wait_rows.any():
            wait_acc = (dist["wait"].argmax(-1)[wait_rows] == wait_slot[wait_rows]).float().mean()
        else:
            wait_acc = torch.zeros(())
    metrics = {
        "loss": float(total.detach().cpu()), "gate": float(gate_loss.detach().cpu()),
        "card": float(card_loss.detach().cpu()), "cell": float(cell_loss.detach().cpu()),
        "wait": float(wait_loss.detach().cpu()), "value": float(value_loss.detach().cpu()),
        "aux": float(aux_loss.detach().cpu()), "gate_acc": float(gate_acc.cpu()),
        "card_acc": float(card_acc.cpu()), "cell_acc": float(cell_acc.cpu()),
        "wait_acc": float(wait_acc.cpu()),
    }
    return total, metrics


def train_bc(
    model: CompetitiveNet,
    dataset: CompetitiveDataset,
    cfg: BCConfig,
    device: torch.device,
    *,
    log_fn=print,
) -> dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError("empty BC dataset")
    train_idx, val_idx = _split_by_replay(dataset, cfg.val_fraction, cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    history = []
    best_val = math.inf
    for epoch in range(cfg.epochs):
        model.train()
        rng.shuffle(train_idx)
        totals = {}
        for start in range(0, len(train_idx), cfg.batch_size):
            idx = train_idx[start:start + cfg.batch_size]
            batch = dataset.batch(idx)
            tensors = _batch_tensors(batch, device)
            loss, metrics = _losses(model, tensors, batch, device, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            for k, v in metrics.items():
                totals.setdefault(k, []).append(v)
        train_metrics = {k: float(np.mean(v)) for k, v in totals.items()}
        val_metrics = {}
        if len(val_idx):
            model.eval()
            vtotals = {}
            with torch.no_grad():
                for start in range(0, len(val_idx), cfg.batch_size):
                    idx = val_idx[start:start + cfg.batch_size]
                    batch = dataset.batch(idx)
                    tensors = _batch_tensors(batch, device)
                    _, metrics = _losses(model, tensors, batch, device, cfg)
                    for k, v in metrics.items():
                        vtotals.setdefault(k, []).append(v)
            val_metrics = {k: float(np.mean(v)) for k, v in vtotals.items()}
            best_val = min(best_val, val_metrics.get("loss", math.inf))
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        log_fn(f"[BC] epoch {epoch + 1}/{cfg.epochs} train {train_metrics} val {val_metrics}")
    return {"history": history, "best_val": best_val, "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}


def save_bc_checkpoint(path: str | Path, model: CompetitiveNet, vocab: CardVocab,
                       bc_metrics: dict[str, Any], extra: dict | None = None) -> None:
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "vocab_names": vocab.names,
        "stage": "bc",
        "bc_metrics": bc_metrics,
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def build_model_from_config(model_cfg: dict, n_cards: int) -> CompetitiveNet:
    return CompetitiveNet(
        n_cards=n_cards,
        d_model=model_cfg.get("d_model", 192),
        layers=model_cfg.get("layers", 4),
        heads=model_cfg.get("heads", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 512),
        dropout=model_cfg.get("dropout", 0.1),
        d_card=model_cfg.get("d_card", 64),
        unit_feat_dim=model_cfg.get("unit_feat_dim", UNIT_FEATURE_DIM),
        global_feat_dim=model_cfg.get("global_feat_dim", GLOBAL_FEATURE_DIM),
        belief_dim=model_cfg.get("belief_dim"),
        n_hand=model_cfg.get("n_hand", 4),
        n_deck=model_cfg.get("n_deck", 8),
        n_past=model_cfg.get("n_past", 4),
        n_fourier=model_cfg.get("n_fourier", 4),
        n_value_bins=model_cfg.get("n_value_bins", 7),
    )
