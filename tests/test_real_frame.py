"""
Test script running ClashRoyaleDetector on real gameplay frames (samples/real_*.png).
Strict requirements:
1. Reads only samples/real_*.png.
2. Fails if --weights argument is missing or weights file does not exist.
3. Fails if detections == 0 (asserts real detection).
4. Verifies (12,) scalars and global-card legality masks.
"""
from typing import List, Optional
import os
import sys
import glob
import argparse
import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from perception.detector import ClashRoyaleDetector
from perception.state_encoder import StateEncoder, stack_spatial_frames
from configs.constants import (
    CARD_DATABASE,
    CARD_ID_TO_COST,
    CARD_ID_TO_IS_SPELL,
    CARD_ID_TO_NAME,
    NUM_SCALARS,
    STACKED_GRID_CHANNELS,
)


def run_test_real_frames(weights_path: Optional[str] = None):
    # Requirement: Fails if weights are missing or file does not exist
    if not weights_path:
        raise ValueError(
            "Test failed: --weights path is strictly required.\n"
            "Please provide path to trained YOLO unit detector weights: "
            "python tests/test_real_frame.py --weights <path_to_weights.pt>"
        )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Test failed: Specified --weights file not found on disk: {weights_path}")

    # Requirement: Reads ONLY samples/real_*.png
    sample_pattern = os.path.join(REPO_ROOT, "samples", "real_*.png")
    image_paths = sorted(glob.glob(sample_pattern))

    print(f"[TestRealFrame] Searching for real frames matching {sample_pattern}...")
    if not image_paths:
        raise FileNotFoundError(f"No real sample images found matching {sample_pattern}")

    print(f"[TestRealFrame] Found {len(image_paths)} real sample frame(s). Loading detector with weights: {weights_path}...")
    detector = ClashRoyaleDetector(weights=weights_path, require_weights=True)
    encoder = StateEncoder()

    for idx, path in enumerate(image_paths):
        print("\n" + "=" * 70)
        print(f"Testing Real Frame [{idx + 1}/{len(image_paths)}]: {os.path.basename(path)}")
        print("=" * 70)

        frame = cv2.imread(path)
        assert frame is not None, f"Failed to load image from {path}"
        h, w = frame.shape[:2]
        print(f"Frame resolution: {w}x{h}")

        # Parse Frame with Detector
        obs = detector.parse_frame(frame, time_left=120.0, est_enemy_elixir=0.5, last_action=(1, 300))

        # Requirement: Fails if detections == 0
        dets = obs["detections"]
        print(f"Detections Count: {len(dets)}")
        if len(dets) == 0:
            raise AssertionError(f"Test failed: Detections count on {os.path.basename(path)} was 0 (expected > 0).")

        for d in dets:
            print(f"  - [{d.category}] conf={d.confidence:.2f}, ally={d.is_ally}, box=({d.x1:.2f}, {d.y1:.2f}, {d.x2:.2f}, {d.y2:.2f})")

        # UI Readouts
        print(f"Screen State:    {obs['screen_state']}")
        print(f"Detected Elixir: {obs['elixir']:.1f}/10")
        print(f"Hand Card IDs:   {obs['hand_ids']}")
        print(f"Tower HPs:       {[round(hp, 2) for hp in obs['tower_hps']]}")
        print(f"Selected Slot:   {obs['selected_slot']}")
        print(f"Clocks Detected: {obs['clock_coords']}")

        # State Tensors & Legality Masks
        arena_grid = obs["arena_grid"]
        scalars = obs["scalars"]
        legal_card_mask = obs["legal_card_mask"]

        assert arena_grid.shape == (8, 32, 18), f"Expected (8, 32, 18), got {arena_grid.shape}"
        assert scalars.shape == (NUM_SCALARS,), f"Expected ({NUM_SCALARS},), got {scalars.shape}"
        assert scalars[1] == 0.5, f"Expected est_enemy_elixir == 0.5, got {scalars[1]}"
        assert legal_card_mask.shape == (5,), f"Expected (5,), got {legal_card_mask.shape}"
        assert legal_card_mask[0] == True, "Wait slot must always be legal"

        # 4-frame stacking (32 channels)
        stacked_grid = stack_spatial_frames([arena_grid, arena_grid, arena_grid, arena_grid], target_frames=4)
        assert stacked_grid.shape == (STACKED_GRID_CHANNELS, 32, 18), f"Expected (32, 32, 18), got {stacked_grid.shape}"

        # Legal Tile Masks for Cards in Hand
        print("\nCard Legality Masks (Global Vocabulary):")
        for slot_idx, card_id in enumerate(obs["hand_ids"]):
            card_affordable = legal_card_mask[slot_idx + 1]
            tmask = encoder.get_legal_tile_mask(card_id, tower_hps=obs["tower_hps"])
            assert tmask.shape == (576,), f"Expected (576,), got {tmask.shape}"
            legal_tiles_count = int(tmask.sum())
            card_name = CARD_ID_TO_NAME.get(card_id, f"Card #{card_id}")
            print(f"  Slot {slot_idx} ({card_name} [ID {card_id}]): Affordable={card_affordable}, Legal Tiles={legal_tiles_count}/576")

        print(f"\nScalars Vector (12 floats):")
        print(f"  {np.round(scalars, 3)}")

    print("\n[SUCCESS] All real sample frames passed detection and encoding tests!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CR-Zero detection on real gameplay frames")
    parser.add_argument("--weights", type=str, default=None, help="YOLO model path (.pt or .onnx)")
    args = parser.parse_args()

    run_test_real_frames(weights_path=args.weights)
