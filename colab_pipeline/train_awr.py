"""
Stage 2: Advantage-Weighted Regression (AWR) Training Pipeline for CR-Zero.
Offline Policy Improvement using Advantage-Weighted Maximum Likelihood.

In AWR, policy update optimizes:
    E_{(s, a) ~ D} [ exp(A(s, a) / beta) * log pi(a | s) ]
where:
    - Advantage A(s, a) is derived from forward 10s tower HP delta + elixir differential + outcome.
    - beta is the temperature parameter controlling greediness (default beta=0.5).
    - Advantage weights are clipped to [0.05, 10.0] * outcome_weight to avoid destructive gradients.
    - Actions leading to tower destruction or winning trades receive up to 10x gradient weight.
    - Blunders and overcommitments are suppressed down to near zero.

Can initialize from Stage 1 Behavior Cloning checkpoint (e.g., checkpoints/cr_zero_policy_best.pt).
Exports best checkpoint to .pt and .onnx for live inference.
"""
from typing import Dict, Tuple, Optional, List
import os
import sys
import glob
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.constants import (
    NUM_TOTAL_CARDS,
    STACKED_GRID_CHANNELS,
    TOTAL_TILES,
    ARENA_ROWS,
    ARENA_COLS,
)
from policy.model import DeckConditionedPolicy
from configs.observation_schema import Sc, check_schema, ALLY_TOWER_SLICE, ENEMY_TOWER_SLICE


class AWRTrajectoryDataset(Dataset):
    """
    Trajectory dataset with advantage weighting:
    Computes advantage A_t = (R_t - mean(R)) / (std(R) + 1e-4)
    Computes sample weight w_t = clip(exp(A_t / beta), 0.05, 10.0) * outcome_weight
    """
    def __init__(
        self,
        shard_paths: List[str],
        beta: float = 0.5,
        augment_horizontal_flip: bool = True,
        sample_range: Optional[Tuple[float, float]] = None,
    ):
        self.beta = beta
        self.augment = augment_horizontal_flip

        grids, scalars, hands, cmasks, tmasks, acards, atiles, raw_rewards, outcome_weights = [], [], [], [], [], [], [], [], []

        for p in shard_paths:
            if not os.path.exists(p):
                continue
            with np.load(p) as data:
                g = data["arena_grid"]
                s = data["scalars"] if "scalars" in data else data["scalar_features"]
                h = data["hand_ids"] if "hand_ids" in data else data["hand_cards"]
                cm = data["legal_card_mask"] if "legal_card_mask" in data else data["card_mask"]
                tm = data["legal_tile_mask"] if "legal_tile_mask" in data else data["tile_mask"]
                ac = data["action_card"] if "action_card" in data else data["card_action"]
                at = data["action_tile"] if "action_tile" in data else data["tile_action"]

                check_schema(data, p)

                # P0 Guard: Vanguard replay data is strictly ineligible for AWR / Advantage weighting.
                # Vanguard has no battlefield units, constant tower HP (1.0), fixed enemy elixir (0.5),
                # and no per-action state transitions or rewards.
                dataset_type = str(data.get("dataset_type", ""))
                has_board = bool(data.get("has_ground_truth_board", True)) if "has_ground_truth_board" in data else True
                is_vanguard_heuristic = (
                    "rewards" not in data
                    and s.ndim == 2
                    and s.shape[0] > 0
                    and np.all(s[:, Sc.ALLY_KING : Sc.ENEMY_RIGHT + 1] == 1.0)
                    and np.all(s[:, Sc.ENEMY_ELIXIR] == 0.5)
                )
                if dataset_type == "pretraining_vanguard" or not has_board or is_vanguard_heuristic:
                    raise ValueError(
                        f"{p}: Ineligible dataset source for AWR. Vanguard replay data lacks battlefield unit positions, "
                        "has constant tower HP, and has no per-action advantage rewards. "
                        "Vanguard shards are restricted to BC prior pretraining (data/pretraining/vanguard/)."
                    )

                outcome = int(data["match_outcome"]) if "match_outcome" in data else 0
                outcome_w = 1.5 if outcome > 0 else (0.5 if outcome < 0 else 1.0)

                # Compute or retrieve rewards
                if "rewards" in data:
                    r = data["rewards"].astype(np.float32)
                    if not np.all(np.isfinite(r)):
                        raise ValueError(f"{p}: Shard contains non-finite (NaN or Inf) values in 'rewards'.")
                else:
                    n_samples = len(ac)
                    if "timestamps" not in data or len(data["timestamps"]) != n_samples:
                        raise ValueError(
                            f"{p}: Missing valid 'timestamps' array matching sample length ({n_samples}). "
                            "Cannot compute forward advantage horizon without timestamps (unsafe fixed-step fallbacks are disabled)."
                        )
                    timestamps = data["timestamps"].astype(np.float32)
                    if not np.all(np.isfinite(timestamps)):
                        raise ValueError(f"{p}: Shard contains non-finite (NaN or Inf) timestamps.")
                    if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
                        raise ValueError(f"{p}: Timestamps in shard must be strictly non-decreasing.")

                    r = np.zeros(n_samples, dtype=np.float32)
                    match_end_t = float(timestamps[-1])
                    for idx in range(n_samples):
                        t_curr = float(timestamps[idx])
                        target_t = t_curr + 10.0
                        fut_idx = int(np.searchsorted(timestamps, target_t, side="left"))

                        # Handle incomplete horizon at match end
                        if target_t > match_end_t:
                            fut_idx = n_samples - 1
                            elapsed_window = max(0.001, match_end_t - t_curr)
                            horizon_scale = min(1.0, elapsed_window / 10.0)
                        else:
                            fut_idx = min(n_samples - 1, fut_idx)
                            horizon_scale = 1.0

                        # Ally king/princess HP vs Enemy king/princess HP
                        ally_hp_curr = s[idx, ALLY_TOWER_SLICE].sum()
                        ally_hp_fut = s[fut_idx, ALLY_TOWER_SLICE].sum()
                        enemy_hp_curr = s[idx, ENEMY_TOWER_SLICE].sum()
                        enemy_hp_fut = s[fut_idx, ENEMY_TOWER_SLICE].sum()

                        delta_tower = ((enemy_hp_curr - enemy_hp_fut) - (ally_hp_curr - ally_hp_fut)) * horizon_scale
                        elixir_diff = (s[idx, Sc.ALLY_ELIXIR] - s[idx, Sc.ENEMY_ELIXIR])
                        r[idx] = float(delta_tower + 0.15 * elixir_diff)

                if sample_range is not None:
                    s_frac, e_frac = sample_range
                    si = int(round(len(ac) * s_frac))
                    ei = int(round(len(ac) * e_frac))
                    g, s, h, cm, tm, ac, at, r = g[si:ei], s[si:ei], h[si:ei], cm[si:ei], tm[si:ei], ac[si:ei], at[si:ei], r[si:ei]

                if len(ac) > 0:
                    grids.append(g)
                    scalars.append(s)
                    hands.append(h)
                    cmasks.append(cm)
                    tmasks.append(tm)
                    acards.append(ac)
                    atiles.append(at)
                    raw_rewards.append(r)
                    outcome_weights.append(np.full(len(ac), fill_value=outcome_w, dtype=np.float32))

        if len(grids) == 0:
            raise ValueError(f"No samples loaded from shards: {shard_paths}")

        self.arena_grid = np.concatenate(grids, axis=0)
        self.scalars = np.concatenate(scalars, axis=0)
        self.hand_ids = np.concatenate(hands, axis=0)
        self.legal_card_mask = np.concatenate(cmasks, axis=0)
        self.legal_tile_mask = np.concatenate(tmasks, axis=0)
        self.action_card = np.concatenate(acards, axis=0)
        self.action_tile = np.concatenate(atiles, axis=0)

        all_rewards = np.concatenate(raw_rewards, axis=0)
        all_outcomes = np.concatenate(outcome_weights, axis=0)

        # Compute normalized advantages and AWR weights
        r_mean = float(np.mean(all_rewards))
        r_std = float(np.std(all_rewards)) + 1e-4
        advantages = (all_rewards - r_mean) / r_std

        # Exponentiated advantage weights: clip(exp(A / beta), 0.05, 10.0) * outcome_weight
        exp_adv = np.exp(np.clip(advantages / max(0.01, self.beta), -4.0, 4.0))
        self.weights = np.clip(exp_adv, 0.05, 10.0) * all_outcomes
        self.weights = (self.weights / max(1e-6, float(self.weights.mean()))).astype(np.float32)  # mean weight = 1
        self.length = len(self.action_card)
        print(f"[AWR Dataset] Loaded {self.length} frames. Weight stats: min={self.weights.min():.2f}, mean={self.weights.mean():.2f}, max={self.weights.max():.2f}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        grid = self.arena_grid[idx].copy()
        scalar = self.scalars[idx].copy()
        hand = self.hand_ids[idx].copy()
        cmask = self.legal_card_mask[idx].copy()
        tmask = self.legal_tile_mask[idx].copy()
        acard = int(self.action_card[idx])
        atile = int(self.action_tile[idx])
        weight = float(self.weights[idx])

        # Horizontal flip augmentation
        if self.augment and np.random.rand() > 0.5:
            grid = np.flip(grid, axis=2).copy()
            scalar[4], scalar[5] = scalar[5], scalar[4]
            scalar[7], scalar[8] = scalar[8], scalar[7]

            prev_tile = int(round(scalar[10] * 575.0))
            if prev_tile > 0:
                py = prev_tile // ARENA_COLS
                px = prev_tile % ARENA_COLS
                scalar[10] = (py * ARENA_COLS + ((ARENA_COLS - 1) - px)) / 575.0

            tile_y = atile // ARENA_COLS
            tile_x = atile % ARENA_COLS
            flipped_tile_x = (ARENA_COLS - 1) - tile_x
            atile = tile_y * ARENA_COLS + flipped_tile_x

            tmask_2d = tmask.reshape(ARENA_ROWS, ARENA_COLS)
            tmask = np.fliplr(tmask_2d).flatten()

        return (
            torch.from_numpy(grid).float(),
            torch.from_numpy(scalar).float(),
            torch.from_numpy(hand).long(),
            torch.from_numpy(cmask).bool(),
            torch.from_numpy(tmask).bool(),
            torch.tensor(acard, dtype=torch.long),
            torch.tensor(atile, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32),
        )


def export_to_onnx(model: DeckConditionedPolicy, export_path: str, device: torch.device):
    """Exports trained DeckConditionedPolicy to ONNX."""
    model.eval()
    dummy_grid = torch.randn(1, 32, 32, 18, device=device)
    dummy_scalar = torch.randn(1, 12, device=device)
    dummy_hand = torch.tensor([[1, 2, 20, 21]], dtype=torch.long, device=device)
    dummy_cmask = torch.ones(1, 5, dtype=torch.bool, device=device)
    dummy_tmask = torch.ones(1, 576, dtype=torch.bool, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    try:
        torch.onnx.export(
            model,
            (dummy_grid, dummy_scalar, dummy_hand, dummy_cmask, dummy_tmask, None),
            export_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["arena_grid", "scalars", "hand_card_ids", "legal_card_mask", "legal_tile_mask", "teacher_card_action"],
            output_names=["masked_card_logits", "masked_tile_logits"],
            dynamic_axes={
                "arena_grid": {0: "batch_size"},
                "scalars": {0: "batch_size"},
                "hand_card_ids": {0: "batch_size"},
                "legal_card_mask": {0: "batch_size"},
                "legal_tile_mask": {0: "batch_size"},
                "masked_card_logits": {0: "batch_size"},
                "masked_tile_logits": {0: "batch_size"},
            }
        )
        print(f"[Export] Saved AWR ONNX model ({os.path.getsize(export_path) / 1024:.1f} KB): {export_path}")
    except Exception as e:
        print(f"[Export] Notice: ONNX export skipped ({e})")


def train_awr_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    tile_loss_weight: float = 1.0,
) -> Dict[str, float]:
    model.train()
    card_crit = nn.CrossEntropyLoss(reduction="none")
    tile_crit = nn.CrossEntropyLoss(reduction="none")

    total_loss_accum = 0.0
    card_correct = 0
    total_samples = 0
    tile_within_2 = 0
    total_plays = 0

    for batch in loader:
        grid, scalar, hand, cmask, tmask, target_card, target_tile, sample_weights = [
            t.to(device) for t in batch
        ]
        B = grid.size(0)

        # Force target tile to be legal in mask
        played_mask = target_card > 0
        if played_mask.any():
            _pi = torch.where(played_mask)[0]
            tmask[_pi, target_tile[_pi]] = True

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            card_logits, tile_logits = model(
                arena_grid=grid,
                scalars=scalar,
                hand_card_ids=hand,
                legal_card_mask=cmask,
                legal_tile_mask=tmask,
                teacher_card_action=target_card,
            )

            # Advantage-Weighted Card Loss
            loss_c = (card_crit(card_logits, target_card) * sample_weights).mean()

            # Advantage-Weighted Tile Loss
            if played_mask.any():
                loss_t = (tile_crit(tile_logits[played_mask], target_tile[played_mask]) * sample_weights[played_mask]).mean()
            else:
                loss_t = torch.tensor(0.0, device=device)

            loss = loss_c + tile_loss_weight * loss_t

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss_accum += loss.item() * B
        preds_card = torch.argmax(card_logits, dim=-1)
        card_correct += (preds_card == target_card).sum().item()
        total_samples += B

        if played_mask.any():
            preds_tile = torch.argmax(tile_logits[played_mask], dim=-1)
            target_t = target_tile[played_mask]
            pred_y = preds_tile // ARENA_COLS
            pred_x = preds_tile % ARENA_COLS
            tgt_y = target_t // ARENA_COLS
            tgt_x = target_t % ARENA_COLS
            dist = torch.maximum(torch.abs(pred_y - tgt_y), torch.abs(pred_x - tgt_x))
            tile_within_2 += (dist <= 2).sum().item()
            total_plays += played_mask.sum().item()

    return {
        "loss": total_loss_accum / max(1, total_samples),
        "card_acc": card_correct / max(1, total_samples),
        "tile_acc": tile_within_2 / max(1, total_plays) if total_plays > 0 else 0.0,
    }


def validate_awr(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    card_crit = nn.CrossEntropyLoss()
    total_loss_accum = 0.0
    card_correct = 0
    total_samples = 0
    tile_within_2 = 0
    total_plays = 0

    with torch.no_grad():
        for batch in loader:
            grid, scalar, hand, cmask, tmask, target_card, target_tile, sample_weights = [
                t.to(device) for t in batch
            ]
            B = grid.size(0)
            played_mask = target_card > 0
            if played_mask.any():
                _pi = torch.where(played_mask)[0]
                tmask[_pi, target_tile[_pi]] = True

            card_logits, tile_logits = model(
                arena_grid=grid,
                scalars=scalar,
                hand_card_ids=hand,
                legal_card_mask=cmask,
                legal_tile_mask=tmask,
                teacher_card_action=target_card,
            )

            loss = card_crit(card_logits, target_card)
            total_loss_accum += loss.item() * B
            preds_c = torch.argmax(card_logits, dim=-1)
            card_correct += (preds_c == target_card).sum().item()
            total_samples += B

            if played_mask.any():
                preds_tile = torch.argmax(tile_logits[played_mask], dim=-1)
                tgt_t = target_tile[played_mask]
                dist = torch.maximum(
                    torch.abs((preds_tile // ARENA_COLS) - (tgt_t // ARENA_COLS)),
                    torch.abs((preds_tile % ARENA_COLS) - (tgt_t % ARENA_COLS)),
                )
                tile_within_2 += (dist <= 2).sum().item()
                total_plays += played_mask.sum().item()

    return {
        "val_loss": total_loss_accum / max(1, total_samples),
        "val_card_acc": card_correct / max(1, total_samples),
        "val_tile_acc": tile_within_2 / max(1, total_plays) if total_plays > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Advantage-Weighted Regression (AWR) for CR-Zero")
    parser.add_argument("--shards-dir", type=str, default="data/trajectories", help="Directory containing .npz shards")
    parser.add_argument("--init-checkpoint", type=str, default="checkpoints/cr_zero_policy_best.pt", help="Pretrained BC checkpoint")
    parser.add_argument("--epochs", type=int, default=15, help="Number of AWR epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.5, help="AWR temperature parameter (lower=greedier)")
    parser.add_argument("--val-split", type=float, default=0.20, help="Validation split fraction")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Output checkpoint directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AWR] Device: {device} | Shards: {args.shards_dir} | Beta: {args.beta}")

    shards = sorted(glob.glob(os.path.join(args.shards_dir, "*.npz")))
    if not shards:
        print(f"[AWR] Error: No .npz shards found in {args.shards_dir}!")
        sys.exit(1)

    print(f"[AWR] Found {len(shards)} trajectory shards: {[os.path.basename(s) for s in shards]}")

    # Train / Val Split (seeded shuffle: sorted-name split puts one data source entirely in val)
    rng = np.random.RandomState(42)
    shards = list(rng.permutation(shards))
    if len(shards) > 1:
        val_count = max(1, int(round(len(shards) * args.val_split)))
        train_shards = shards[:-val_count]
        val_shards = shards[-val_count:]
        train_dataset = AWRTrajectoryDataset(train_shards, beta=args.beta, augment_horizontal_flip=True)
        val_dataset = AWRTrajectoryDataset(val_shards, beta=args.beta, augment_horizontal_flip=False)
    else:
        train_dataset = AWRTrajectoryDataset(shards, beta=args.beta, augment_horizontal_flip=True, sample_range=(0.0, 1.0 - args.val_split))
        val_dataset = AWRTrajectoryDataset(shards, beta=args.beta, augment_horizontal_flip=False, sample_range=(1.0 - args.val_split, 1.0))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    # Model instantiation
    model = DeckConditionedPolicy().to(device)

    # Optional warm start from Stage 1 BC
    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        try:
            ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
            state_dict = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(f'Checkpoint/model mismatch: missing={missing}, unexpected={unexpected}')
            print(f"[AWR] Successfully warm-started weights from {args.init_checkpoint}")
        except Exception as e:
            print(f"[AWR] Could not load init checkpoint ({e}). Training from scratch.")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.checkpoint_dir, "cr_zero_policy_awr_best.pt")
    best_onnx_path = os.path.join(args.checkpoint_dir, "cr_zero_policy_awr_best.onnx")

    print("\n" + "=" * 70)
    print("Beginning Stage 2: Advantage-Weighted Regression Fine-Tuning")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_awr_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            tile_loss_weight=1.0,
        )
        val_metrics = validate_awr(model=model, loader=val_loader, device=device)
        scheduler.step()
        epoch_sec = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_sec:.1f}s] | "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Card Acc: {train_metrics['card_acc']*100:.1f}% | Train Tile Acc: {train_metrics['tile_acc']*100:.1f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val Card Acc: {val_metrics['val_card_acc']*100:.1f}%"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_card_acc": val_metrics["val_card_acc"],
                "beta": args.beta,
            }, best_ckpt_path)
            print(f"  -> Saved new best AWR checkpoint to {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")

    # Export best model to ONNX
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        export_to_onnx(model, best_onnx_path, device)

    print("\n[AWR] Stage 2 Advantage-Weighted Regression Training Complete!")


if __name__ == "__main__":
    main()

