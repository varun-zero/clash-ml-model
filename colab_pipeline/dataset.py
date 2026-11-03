"""
PyTorch Dataset and DataLoader for Clash Royale Trajectory Shards (.npz).
Supports:
- Multi-match shard loading with match-level train/val splitting (no sample-level leakage).
- Temporal intra-match splitting when only a single match shard is available.
- Match outcome weighting: win (1.5) > draw (1.0) > loss (0.5).
- Arena horizontal flip data augmentation.
"""
from typing import List, Tuple, Optional, Union
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from configs.constants import ARENA_COLS, ARENA_ROWS, TOTAL_TILES
from configs.observation_schema import Sc, check_schema


class ClashRoyaleDataset(Dataset):
    """
    Loads one or more .npz shards containing:
      arena_grid: (N, 32, 32, 18) float32
      scalars: (N, 12) float32
      hand_ids: (N, 4) int64
      legal_card_mask: (N, 5) bool
      legal_tile_mask: (N, 576) bool
      action_card: (N,) int64
      action_tile: (N,) int64
      match_outcome: Optional int (+1 win, 0 draw, -1 loss)
    """
    def __init__(
        self,
        shard_paths: List[str],
        augment_horizontal_flip: bool = True,
        sample_range: Optional[Tuple[float, float]] = None,
    ):
        self.augment = augment_horizontal_flip

        grids, scalars, hands, cmasks, tmasks, acards, atiles, weights = [], [], [], [], [], [], [], []

        for p in shard_paths:
            if not os.path.exists(p):
                continue
            with np.load(p) as data:
                check_schema(data, p)
                g = data["arena_grid"]
                s = data["scalars"] if "scalars" in data else data["scalar_features"]
                h = data["hand_ids"] if "hand_ids" in data else data["hand_cards"]
                cm = data["legal_card_mask"] if "legal_card_mask" in data else data["card_mask"]
                tm = data["legal_tile_mask"] if "legal_tile_mask" in data else data["tile_mask"]
                ac = data["action_card"] if "action_card" in data else data["card_action"]
                at = data["action_tile"] if "action_tile" in data else data["tile_action"]

                # Outcome weighting: win > draw > loss
                outcome = int(data["match_outcome"]) if "match_outcome" in data else 0
                if outcome > 0:
                    outcome_w = 1.5
                elif outcome < 0:
                    outcome_w = 0.5
                else:
                    outcome_w = 1.0

                w_arr = np.full(len(ac), fill_value=outcome_w, dtype=np.float32)

                # Temporal intra-match slice if specified
                if sample_range is not None:
                    start_frac, end_frac = sample_range
                    start_idx = int(round(len(ac) * start_frac))
                    end_idx = int(round(len(ac) * end_frac))
                    g = g[start_idx:end_idx]
                    s = s[start_idx:end_idx]
                    h = h[start_idx:end_idx]
                    cm = cm[start_idx:end_idx]
                    tm = tm[start_idx:end_idx]
                    ac = ac[start_idx:end_idx]
                    at = at[start_idx:end_idx]
                    w_arr = w_arr[start_idx:end_idx]

                if len(ac) > 0:
                    grids.append(g)
                    scalars.append(s)
                    hands.append(h)
                    cmasks.append(cm)
                    tmasks.append(tm)
                    acards.append(ac)
                    atiles.append(at)
                    weights.append(w_arr)

        if len(grids) == 0:
            raise ValueError(f"No valid samples loaded from {shard_paths}")

        self.arena_grid = np.concatenate(grids, axis=0)
        self.scalars = np.concatenate(scalars, axis=0)
        self.hand_ids = np.concatenate(hands, axis=0)
        self.legal_card_mask = np.concatenate(cmasks, axis=0)
        self.legal_tile_mask = np.concatenate(tmasks, axis=0)
        self.action_card = np.concatenate(acards, axis=0)
        self.action_tile = np.concatenate(atiles, axis=0)
        self.weights = np.concatenate(weights, axis=0)

        self.length = len(self.action_card)
        print(f"[Dataset] Loaded {self.length} samples from {len(shard_paths)} shard(s).")

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

        # Data augmentation: 50% chance horizontal flip
        if self.augment and np.random.rand() > 0.5:
            # 1. Flip spatial grid horizontally (axis=2 is cols)
            grid = np.flip(grid, axis=2).copy()

            # 2. Swap left and right princess tower HPs in scalars
            # 4: ally_left, 5: ally_right; 7: enemy_left, 8: enemy_right
            scalar[Sc.ALLY_LEFT], scalar[Sc.ALLY_RIGHT] = scalar[Sc.ALLY_RIGHT], scalar[Sc.ALLY_LEFT]
            scalar[Sc.ENEMY_LEFT], scalar[Sc.ENEMY_RIGHT] = scalar[Sc.ENEMY_RIGHT], scalar[Sc.ENEMY_LEFT]

            # Mirror last action tile in scalars (index 10)
            prev_tile = int(round(scalar[10] * 575.0))
            if prev_tile > 0:
                py = prev_tile // ARENA_COLS
                px = prev_tile % ARENA_COLS
                scalar[10] = (py * ARENA_COLS + ((ARENA_COLS - 1) - px)) / 575.0

            # 3. Flip action tile column
            tile_y = atile // ARENA_COLS
            tile_x = atile % ARENA_COLS
            flipped_tile_x = (ARENA_COLS - 1) - tile_x
            atile = tile_y * ARENA_COLS + flipped_tile_x

            # 4. Flip legal tile mask
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
            torch.tensor(weight, dtype=torch.float32),
        )


def create_dataloaders(
    data_dir_or_files: Union[str, List[str]],
    batch_size: int = 64,
    val_split: float = 0.15,
    num_workers: int = 2,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation DataLoaders split by match (not by individual sample).
    Prevents leakage across consecutive frames of the same match.
    """
    if isinstance(data_dir_or_files, list):
        shard_paths = data_dir_or_files
    elif os.path.isdir(data_dir_or_files):
        shard_paths = sorted(glob.glob(os.path.join(data_dir_or_files, "*.npz")))
    else:
        shard_paths = [data_dir_or_files]

    if not shard_paths:
        raise FileNotFoundError(f"No .npz trajectory files found in {data_dir_or_files}")

    np.random.seed(42)
    shuffled_shards = list(shard_paths)
    np.random.shuffle(shuffled_shards)

    if len(shuffled_shards) > 1:
        # Split by match / shard
        num_val = max(1, int(round(len(shuffled_shards) * val_split)))
        val_shards = shuffled_shards[:num_val]
        train_shards = shuffled_shards[num_val:]
        print(f"[DataLoader] Split {len(shuffled_shards)} matches into {len(train_shards)} train and {len(val_shards)} val shards.")
        train_dataset = ClashRoyaleDataset(train_shards, augment_horizontal_flip=augment)
        val_dataset = ClashRoyaleDataset(val_shards, augment_horizontal_flip=False)
    else:
        # Single match: temporal intra-match split (first 85% train, last 15% val)
        print(f"[DataLoader] Single match shard detected. Splitting temporally (train: 0..{1.0-val_split:.2f}, val: {1.0-val_split:.2f}..1.0).")
        train_dataset = ClashRoyaleDataset(shuffled_shards, augment_horizontal_flip=augment, sample_range=(0.0, 1.0 - val_split))
        val_dataset = ClashRoyaleDataset(shuffled_shards, augment_horizontal_flip=False, sample_range=(1.0 - val_split, 1.0))

    effective_workers = 0 if os.name == "nt" else num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=True,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
