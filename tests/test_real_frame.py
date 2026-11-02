"""
Test script running ClashRoyaleDetector and StateEncoder on sample gameplay frames.
Verifies detections, 32-channel stacked grids, (12,) scalars, and legality masks.
"""
from typing import List
import os
import sys
import glob
import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from perception.detector import ClashRoyaleDetector
from perception.state_encoder import StateEncoder, stack_spatial_frames
from configs.constants import DEFAULT_DECK_CLASSES, STACKED_GRID_CHANNELS, NUM_SCALARS


def test_real_frames():
    sample_pattern = os.path.join(REPO_ROOT, "samples", "*.png")
    image_paths = sorted(glob.glob(sample_pattern))

    print(f"[TestRealFrame] Searching for samples in {sample_pattern}...")
    if not image_paths:
        raise FileNotFoundError(f"No sample .png images found in {os.path.dirname(sample_pattern)}")

    print(f"[TestRealFrame] Found {len(image_paths)} sample frame(s). Initializing detector...")
    detector = ClashRoyaleDetector(card_classes=DEFAULT_DECK_CLASSES)
    encoder = StateEncoder()

    for idx, path in enumerate(image_paths):
        print("\n" + "=" * 70)
        print(f"Testing Frame [{idx + 1}/{len(image_paths)}]: {os.path.basename(path)}")
        print("=" * 70)

        frame = cv2.imread(path)
        assert frame is not None, f"Failed to load image from {path}"
        h, w = frame.shape[:2]
        print(f"Frame resolution: {w}x{h}")

        # 1. Parse Frame with Detector
        obs = detector.parse_frame(frame, time_left=120.0, est_enemy_elixir=0.5, last_action=(1, 300))

        # Detections
        dets = obs["detections"]
        print(f"Detections Count: {len(dets)}")
        for d in dets:
            print(f"  - [{d.category}] conf={d.confidence:.2f}, ally={d.is_ally}, box=({d.x1:.2f}, {d.y1:.2f}, {d.x2:.2f}, {d.y2:.2f})")

        # UI Readouts
        print(f"Detected Elixir: {obs['elixir']:.1f}/10")
        print(f"Hand Card IDs:   {obs['hand_ids']}")
        print(f"Tower HPs:       {[round(hp, 2) for hp in obs['tower_hps']]}")
        print(f"Selected Slot:   {obs['selected_slot']}")
        print(f"Overlay Active:  {obs['has_overlay']}")
        print(f"Clocks Detected: {obs['clock_coords']}")

        # 2. State Tensors & Legality Masks
        arena_grid = obs["arena_grid"]
        scalars = obs["scalars"]
        legal_card_mask = obs["legal_card_mask"]

        # Assertions
        assert arena_grid.shape == (8, 32, 18), f"Expected (8, 32, 18), got {arena_grid.shape}"
        assert scalars.shape == (NUM_SCALARS,), f"Expected ({NUM_SCALARS},), got {scalars.shape}"
        assert scalars[1] == 0.5, f"Expected est_enemy_elixir == 0.5, got {scalars[1]}"
        assert legal_card_mask.shape == (5,), f"Expected (5,), got {legal_card_mask.shape}"
        assert legal_card_mask[0] == True, "Wait slot must always be legal"

        # 3. Stack last 4 frames (32 channels)
        stacked_grid = stack_spatial_frames([arena_grid, arena_grid, arena_grid, arena_grid], target_frames=4)
        assert stacked_grid.shape == (STACKED_GRID_CHANNELS, 32, 18), f"Expected (32, 32, 18), got {stacked_grid.shape}"

        # 4. Legal Tile Masks for Cards in Hand
        print("\nCard Legality Masks:")
        for slot_idx, card_id in enumerate(obs["hand_ids"]):
            card_affordable = legal_card_mask[slot_idx + 1]
            tmask = encoder.get_legal_tile_mask(card_id)
            assert tmask.shape == (576,), f"Expected (576,), got {tmask.shape}"
            legal_tiles_count = int(tmask.sum())
            print(f"  Slot {slot_idx} (Card #{card_id}): Affordable={card_affordable}, Legal Tiles={legal_tiles_count}/576")

        print(f"\nScalars Vector (12 floats):")
        print(f"  {np.round(scalars, 3)}")

    print("\n[SUCCESS] All sample frames passed detection and encoding tests with exit code 0!")


if __name__ == "__main__":
    test_real_frames()
