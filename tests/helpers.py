"""
Test helpers and synthetic data generators for CR-Zero test suites.
Keeps synthetic fallback logic isolated from production training scripts.
"""
from typing import Optional
import os
import numpy as np

from configs.constants import (
    NUM_TOTAL_CARDS,
    STACKED_GRID_CHANNELS,
    TOTAL_TILES,
    DEFAULT_UNLOCKED_CARD_IDS,
)


def generate_synthetic_shard(
    output_path: str,
    num_samples: int = 300,
    match_outcome: int = 1,
) -> str:
    """
    Generates a synthetic dataset shard for testing data loaders, model pipelines,
    and unit tests without requiring real YouTube VoD downloads.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    grids = np.random.randn(num_samples, STACKED_GRID_CHANNELS, 32, 18).astype(np.float32)
    scalars = np.random.rand(num_samples, 12).astype(np.float32)
    scalars[:, 1] = 0.5  # Constant in v1

    # Hand cards from starter vocabulary (global IDs)
    starter_ids = DEFAULT_UNLOCKED_CARD_IDS
    hands = np.random.choice(starter_ids, size=(num_samples, 4)).astype(np.int64)

    cmasks = np.ones((num_samples, 5), dtype=bool)
    tmasks = np.ones((num_samples, TOTAL_TILES), dtype=bool)

    cards = []
    tiles = []
    for i in range(num_samples):
        if i % 3 == 0:
            cards.append(0)  # Wait
            tiles.append(0)
        else:
            cards.append(int(np.random.randint(1, 5)))
            tiles.append(int(np.random.randint(16 * 18, 32 * 18)))

    np.savez_compressed(
        output_path,
        arena_grid=grids,
        scalars=scalars,
        hand_ids=hands,
        legal_card_mask=cmasks,
        legal_tile_mask=tmasks,
        action_card=np.array(cards, dtype=np.int64),
        action_tile=np.array(tiles, dtype=np.int64),
        timestamps=np.linspace(0, 180, num_samples, dtype=np.float32),
        match_outcome=match_outcome,
        ally_crowns=2 if match_outcome == 1 else 0,
        enemy_crowns=1 if match_outcome == 1 else 2,
    )
    return output_path

