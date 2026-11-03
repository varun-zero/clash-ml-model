"""
Behavior Cloning (Offline Imitation Learning) Training Script for CR-Zero.
Trains DeckConditionedPolicy on extracted gameplay shards (.npz).
Features:
- Global 135-card vocabulary with static feature projection.
- Outcome-weighted loss: winning match trajectories weighted higher than losing ones.
- Match-level train/validation splitting with zero intra-match sample leakage.
- AdamW + CosineAnnealingLR + Mixed Precision.
- Evaluation metrics: Top-1 Card Acc, Wait Acc, Tile-within-2-tiles.
- Checkpoint export to .pt and .onnx (model.eval() with teacher_card_action=None).
- Hard failure when real trajectory shards are missing (synthetic fallback moved to tests/).
"""
from typing import Dict, Tuple, Optional
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.constants import (
    NUM_TOTAL_CARDS,
    STACKED_GRID_CHANNELS,
    TOTAL_TILES,
    ARENA_ROWS,
    ARENA_COLS,
)
from policy.model import DeckConditionedPolicy
from colab_pipeline.dataset import ClashRoyaleDataset, create_dataloaders


def export_to_onnx(
    model: DeckConditionedPolicy,
    export_path: str,
    device: torch.device
):
    """
    Exports trained DeckConditionedPolicy to ONNX with dynamic batch axis.
    Always calls model.eval() and exports with teacher_card_action=None.
    """
    try:
        import onnx
    except ImportError:
        print(f"[Export] Notice: 'onnx' Python package is not installed. Skipping ONNX export. Install with 'pip install onnx'.")
        return

    model.eval()
    dummy_grid = torch.randn(1, 32, 32, 18, device=device)
    dummy_scalar = torch.randn(1, 12, device=device)
    dummy_hand = torch.tensor([[1, 2, 20, 21]], dtype=torch.long, device=device)
    dummy_cmask = torch.ones(1, 5, dtype=torch.bool, device=device)
    dummy_tmask = torch.ones(1, 576, dtype=torch.bool, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
    print(f"[Export] Exporting ONNX model to {export_path} (teacher_card_action=None)...")
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
        print(f"[Export] Saved ONNX model successfully ({os.path.getsize(export_path) / 1024:.1f} KB).")
    except Exception as e:
        print(f"[Export] Failed to export ONNX model: {e}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    tile_loss_weight: float = 1.0,
) -> Dict[str, float]:
    model.train()
    card_criterion_none = nn.CrossEntropyLoss(reduction="none")
    tile_criterion_none = nn.CrossEntropyLoss(reduction="none")

    total_loss_accum = 0.0
    card_loss_accum = 0.0
    tile_loss_accum = 0.0
    card_correct = 0
    wait_correct = 0
    total_waits = 0
    tile_within_2_count = 0
    total_samples = 0
    total_card_plays = 0

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

            # 1. Outcome-weighted Card Loss
            loss_card_per_sample = card_criterion_none(card_logits, target_card)
            loss_card = (loss_card_per_sample * sample_weights).mean()

            # 2. Outcome-weighted Tile Loss
            if played_mask.any():
                loss_tile_per_sample = tile_criterion_none(tile_logits[played_mask], target_tile[played_mask])
                weights_played = sample_weights[played_mask]
                loss_tile = (loss_tile_per_sample * weights_played).mean()
            else:
                loss_tile = torch.tensor(0.0, device=device)

            total_loss = loss_card + (tile_loss_weight * loss_tile)

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss_accum += total_loss.item() * B
        card_loss_accum += loss_card.item() * B
        if played_mask.any():
            tile_loss_accum += loss_tile.item() * played_mask.sum().item()

        # Metrics
        pred_cards = torch.argmax(card_logits, dim=-1)
        card_correct += (pred_cards == target_card).sum().item()

        wait_mask = target_card == 0
        if wait_mask.any():
            wait_correct += (pred_cards[wait_mask] == 0).sum().item()
            total_waits += wait_mask.sum().item()

        if played_mask.any():
            pred_tiles = torch.argmax(tile_logits[played_mask], dim=-1)
            targ_tiles = target_tile[played_mask]

            pred_y = pred_tiles // ARENA_COLS
            pred_x = pred_tiles % ARENA_COLS
            targ_y = targ_tiles // ARENA_COLS
            targ_x = targ_tiles % ARENA_COLS

            dist = torch.sqrt((pred_y.float() - targ_y.float()) ** 2 + (pred_x.float() - targ_x.float()) ** 2)
            tile_within_2_count += (dist <= 2.0).sum().item()
            total_card_plays += played_mask.sum().item()

        total_samples += B

    return {
        "loss": total_loss_accum / max(1, total_samples),
        "card_loss": card_loss_accum / max(1, total_samples),
        "tile_loss": tile_loss_accum / max(1, total_card_plays),
        "card_acc": card_correct / max(1, total_samples),
        "wait_acc": wait_correct / max(1, total_waits),
        "tile_within_2": tile_within_2_count / max(1, total_card_plays),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    card_criterion = nn.CrossEntropyLoss()
    tile_criterion = nn.CrossEntropyLoss()

    total_loss_accum = 0.0
    card_loss_accum = 0.0
    tile_loss_accum = 0.0
    card_correct = 0
    wait_correct = 0
    total_waits = 0
    tile_within_2_count = 0
    total_samples = 0
    total_card_plays = 0

    for batch in loader:
        grid, scalar, hand, cmask, tmask, target_card, target_tile, _ = [
            t.to(device) for t in batch
        ]
        B = grid.size(0)

        played_mask = target_card > 0
        if played_mask.any():
            _pi = torch.where(played_mask)[0]
            tmask[_pi, target_tile[_pi]] = True

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            card_logits, tile_logits = model(
                arena_grid=grid,
                scalars=scalar,
                hand_card_ids=hand,
                legal_card_mask=cmask,
                legal_tile_mask=tmask,
                teacher_card_action=target_card,
            )

            loss_card = card_criterion(card_logits, target_card)
            if played_mask.any():
                loss_tile = tile_criterion(tile_logits[played_mask], target_tile[played_mask])
            else:
                loss_tile = torch.tensor(0.0, device=device)

            total_loss = loss_card + loss_tile

        total_loss_accum += total_loss.item() * B
        card_loss_accum += loss_card.item() * B
        if played_mask.any():
            tile_loss_accum += loss_tile.item() * played_mask.sum().item()

        pred_cards = torch.argmax(card_logits, dim=-1)
        card_correct += (pred_cards == target_card).sum().item()

        wait_mask = target_card == 0
        if wait_mask.any():
            wait_correct += (pred_cards[wait_mask] == 0).sum().item()
            total_waits += wait_mask.sum().item()

        if played_mask.any():
            pred_tiles = torch.argmax(tile_logits[played_mask], dim=-1)
            targ_tiles = target_tile[played_mask]

            pred_y = pred_tiles // ARENA_COLS
            pred_x = pred_tiles % ARENA_COLS
            targ_y = targ_tiles // ARENA_COLS
            targ_x = targ_tiles % ARENA_COLS

            dist = torch.sqrt((pred_y.float() - targ_y.float()) ** 2 + (pred_x.float() - targ_x.float()) ** 2)
            tile_within_2_count += (dist <= 2.0).sum().item()
            total_card_plays += played_mask.sum().item()

        total_samples += B

    return {
        "val_loss": total_loss_accum / max(1, total_samples),
        "val_card_loss": card_loss_accum / max(1, total_samples),
        "val_tile_loss": tile_loss_accum / max(1, total_card_plays),
        "val_card_acc": card_correct / max(1, total_samples),
        "val_wait_acc": wait_correct / max(1, total_waits),
        "val_tile_within_2": tile_within_2_count / max(1, total_card_plays),
    }


def main():
    parser = argparse.ArgumentParser(description="Train CR-Zero Policy Network via Behavior Cloning")
    parser.add_argument("--data_dir", type=str, default="./data/trajectories", help="Path to .npz directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Path to save weights")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--tile_loss_weight", type=float, default=1.0, help="Weight for tile head CE loss")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory '{args.data_dir}' does not exist.")

    # Strict check: fail loudly if no real trajectory shards exist
    npz_files = [f for f in os.listdir(args.data_dir) if f.endswith(".npz")]
    if not npz_files:
        raise FileNotFoundError(
            f"No trajectory shards (.npz) found in {args.data_dir}.\n"
            f"Please extract real YouTube match VoDs using colab_pipeline/yt_action_extractor.py or "
            f"record live matches using python main_live.py --log-dir {args.data_dir}."
        )

    device = torch.device(args.device)
    print(f"[Training] Using compute device: {device} ({len(npz_files)} shard(s) available)")

    # DataLoaders with match-level splitting
    train_loader, val_loader = create_dataloaders(
        data_dir_or_files=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.15,
        num_workers=0 if os.name == "nt" else 2,
    )

    # Policy Model with 135 Global Card Vocabulary
    model = DeckConditionedPolicy(
        num_card_classes=NUM_TOTAL_CARDS,
        card_emb_dim=16,
        num_scalars=12,
        in_channels=STACKED_GRID_CHANNELS,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_pt_path = os.path.join(args.save_dir, "cr_zero_policy_best.pt")
    best_onnx_path = os.path.join(args.save_dir, "cr_zero_policy_best.onnx")

    print("\n" + "=" * 85)
    print(f"{'Epoch':<5} | {'Train Loss':<10} | {'Card Acc':<8} | {'Wait Acc':<8} | {'Tile<=2':<7} | {'Val Loss':<8} | {'Val Card':<8} | {'Val Wait':<8} | {'Val Tile<=2'}")
    print("=" * 85)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            tile_loss_weight=args.tile_loss_weight,
        )
        val_metrics = evaluate(model=model, loader=val_loader, device=device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"{epoch:02d}/{args.epochs:02d} | "
            f"{train_metrics['loss']:<10.4f} | "
            f"{train_metrics['card_acc']*100:<7.1f}% | "
            f"{train_metrics['wait_acc']*100:<7.1f}% | "
            f"{train_metrics['tile_within_2']*100:<6.1f}% | "
            f"{val_metrics['val_loss']:<8.4f} | "
            f"{val_metrics['val_card_acc']*100:<8.1f}% | "
            f"{val_metrics['val_wait_acc']*100:<8.1f}% | "
            f"{val_metrics['val_tile_within_2']*100:.1f}% ({elapsed:.1f}s)"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "num_card_classes": NUM_TOTAL_CARDS,
                },
                best_pt_path
            )
            print(f"  -> Saved new best PyTorch checkpoint to {best_pt_path}")

    print("\n[Training Complete] Loading best checkpoint for ONNX export...")
    if os.path.exists(best_pt_path):
        ckpt = torch.load(best_pt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        export_to_onnx(model, best_onnx_path, device=device)

    print(f"[Done] Checkpoints ready at {best_pt_path} and {best_onnx_path}\n")


if __name__ == "__main__":
    main()
