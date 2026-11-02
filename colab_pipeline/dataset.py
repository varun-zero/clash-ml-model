"""
PyTorch Dataset and DataLoader for Clash Royale Trajectory Shards (.npz).
Supports multi-shard loading, train/val splitting, and arena horizontal flip augmentation.
"""
from typing import List, Tuple, Optional
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from configs.constants import ARENA_COLS, ARENA_ROWS, TOTAL_TILES


class ClashRoyaleDataset(Dataset):
    """
    Loads one or more .npz shards containing:
      arena_grid: (N, 8, 32, 18) float32
      scalars: (N, 12) float32
      hand_ids: (N, 4) int64
      legal_card_mask: (N, 5) bool
      legal_tile_mask: (N, 576) bool
      action_card: (N,) int64
      action_tile: (N,) int64
    """
    def __init__(
        self,
        shard_paths: List[str],
        augment_horizontal_flip: bool = True,
    ):
        self.augment = augment_horizontal_flip

        grids, scalars, hands, cmasks, tmasks, acards, atiles = [], [], [], [], [], [], []

        for p in shard_paths:
            if not os.path.exists(p):
                continue
            with np.load(p) as data:
                grids.append(data["arena_grid"])
                scalars.append(data["scalars"])
                hands.append(data["hand_ids"])
                cmasks.append(data["legal_card_mask"])
                tmasks.append(data["legal_tile_mask"])
                acards.append(data["action_card"])
                atiles.append(data["action_tile"])

        if len(grids) == 0:
            raise ValueError(f"No valid shards loaded from {shard_paths}")

        self.arena_grid = np.concatenate(grids, axis=0)
        self.scalars = np.concatenate(scalars, axis=0)
        self.hand_ids = np.concatenate(hands, axis=0)
        self.legal_card_mask = np.concatenate(cmasks, axis=0)
        self.legal_tile_mask = np.concatenate(tmasks, axis=0)
        self.action_card = np.concatenate(acards, axis=0)
        self.action_tile = np.concatenate(atiles, axis=0)

        self.length = len(self.action_card)
        print(f"[Dataset] Loaded {self.length} samples from {len(shard_paths)} shards.")

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

        # Data augmentation: 50% chance horizontal flip
        if self.augment and np.random.rand() > 0.5:
            # 1. Flip spatial grid horizontally (axis=2 is cols)
            grid = np.flip(grid, axis=2).copy()

            # 2. Swap left and right princess tower HPs in scalars
            # Scalars indices:
            # 4: ally_left, 5: ally_right
            # 7: enemy_left, 8: enemy_right
            scalar[4], scalar[5] = scalar[5], scalar[4]
            scalar[7], scalar[8] = scalar[8], scalar[7]

            # 3. Flip action tile column
            tile_y = atile // ARENA_COLS
            tile_x = atile % ARENA_COLS
            flipped_tile_x = (ARENA_COLS - 1) - tile_x
            atile = tile_y * ARENA_COLS + flipped_tile_x

            # 4. Flip legal tile mask (576 -> 32, 18 -> flip horizontal -> 576)
            tmask_2d = tmask.reshape(ARENA_ROWS, ARENA_COLS)
            tmask = np.flip(tmask_2d, axis=1).flatten().copy()

        return (
            torch.from_numpy(grid).float(),
            torch.from_numpy(scalar).float(),
            torch.from_numpy(hand).long(),
            torch.from_numpy(cmask).bool(),
            torch.from_numpy(tmask).bool(),
            torch.tensor(acard, dtype=torch.long),
            torch.tensor(atile, dtype=torch.long),
        )


def create_dataloaders(
    data_dir_or_files: str,
    batch_size: int = 64,
    val_split: float = 0.15,
    num_workers: int = 2,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation DataLoaders from .npz files in directory.
    """
    if os.path.isdir(data_dir_or_files):
        shard_paths = sorted(glob.glob(os.path.join(data_dir_or_files, "*.npz")))
    else:
        shard_paths = [data_dir_or_files]

    if not shard_paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir_or_files}")

    np.random.seed(42)
    np.random.shuffle(shard_paths)

    num_val = max(1, int(round(len(shard_paths) * val_split))) if len(shard_paths) > 1 else 0
    val_shards = shard_paths[:num_val] if num_val > 0 else shard_paths
    train_shards = shard_paths[num_val:] if num_val > 0 else shard_paths

    train_dataset = ClashRoyaleDataset(train_shards, augment_horizontal_flip=augment)
    val_dataset = ClashRoyaleDataset(val_shards, augment_horizontal_flip=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader

