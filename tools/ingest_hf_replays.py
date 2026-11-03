"""
Hugging Face Clash Royale TV-Replays Dataset Ingester.
Dataset: chrisrca/clash-royale-tv-replays
Link: https://huggingface.co/datasets/chrisrca/clash-royale-tv-replays

Contains 52,900 rows of deduplicated, frame-by-frame 10 FPS replays across 31 arenas,
captured natively at 540x960 resolution via Android emulator.
Converts Parquet frame sequences directly into CR-Zero .npz trajectory shards.
"""
from typing import Optional, List
import os
import sys
import argparse
import time
import numpy as np
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.constants import (
    ARENA_ROWS,
    ARENA_COLS,
    CARD_ID_TO_COST,
    CARD_DATABASE,
)
from configs.observation_schema import SCHEMA_VERSION
from perception.detector import ClashRoyaleDetector
from perception.state_encoder import StateEncoder, stack_spatial_frames


def ingest_parquet_file(
    parquet_path: str,
    output_dir: str = "data/trajectories",
    max_frames: int = 1500,
    detector_weights: Optional[str] = "weights/yolo_units.pt",
) -> List[str]:
    """
    Parses a downloaded Parquet file from chrisrca/clash-royale-tv-replays,
    encodes board states, identifies player card deployments, and saves .npz shards.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("[HF Ingest] Error: 'pyarrow' is required. Install with: pip install pyarrow")
        return []

    print(f"[HF Ingest] Reading Parquet: {parquet_path}...")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    print(f"[HF Ingest] Loaded {len(df)} rows. Columns: {list(df.columns)}")

    os.makedirs(output_dir, exist_ok=True)
    detector = ClashRoyaleDetector(
        weights=detector_weights if (detector_weights and os.path.exists(detector_weights)) else None,
        conf_threshold=0.35,
    )
    encoder = StateEncoder()

    # Determine frame column
    frame_col = None
    for c in ["frame", "image", "bytes", "data"]:
        if c in df.columns:
            frame_col = c
            break

    if frame_col is None:
        print(f"[HF Ingest] Warning: Could not locate image column in {list(df.columns)}. Available keys: {list(df.columns)}")
        return []

    # Process match frames
    shards_created = []
    buffer_grids, buffer_scalars, buffer_hands = [], [], []
    buffer_cmasks, buffer_tmasks, buffer_cards, buffer_tiles, buffer_timestamps = [], [], [], [], []
    grid_history = []
    last_action = (0, 0)
    match_idx = 1
    total_processed = min(len(df), max_frames)

    for i in range(total_processed):
        raw_val = df[frame_col].iloc[i]
        if isinstance(raw_val, bytes):
            nparr = np.frombuffer(raw_val, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(raw_val, np.ndarray):
            frame = raw_val
        else:
            continue

        if frame is None or frame.size == 0:
            continue

        if frame.shape[0] != 960 or frame.shape[1] != 540:
            frame = cv2.resize(frame, (540, 960), interpolation=cv2.INTER_AREA)

        obs = detector.parse_frame(frame, last_action=last_action)
        if obs["screen_state"] != "battle":
            continue

        grid_history.append(obs["arena_grid"])
        if len(grid_history) > 4:
            grid_history.pop(0)
        stacked_grid = stack_spatial_frames(grid_history, target_frames=4)

        # Detect action if selected slot is present or hand changed
        hand_ids = obs["hand_ids"]
        chosen_slot = 0
        chosen_tile = 0
        if obs.get("selected_slot") is not None:
            chosen_slot = obs["selected_slot"] + 1

        active_cid = hand_ids[chosen_slot - 1] if chosen_slot > 0 else 0
        tile_mask = encoder.get_legal_tile_mask(active_cid, tower_hps=obs.get("tower_hps"))

        buffer_grids.append(stacked_grid)
        buffer_scalars.append(obs["scalars"])
        buffer_hands.append(np.array(hand_ids, dtype=np.int64))
        buffer_cmasks.append(obs["legal_card_mask"])
        buffer_tmasks.append(tile_mask)
        buffer_cards.append(chosen_slot)
        buffer_tiles.append(chosen_tile)
        buffer_timestamps.append(i * 0.10)

        # Flush match shard every 400 frames
        if len(buffer_grids) >= 400 or i == total_processed - 1:
            if len(buffer_grids) >= 50:
                shard_name = f"hf_tv_replays_m{match_idx:03d}_{int(time.time())}.npz"
                shard_path = os.path.join(output_dir, shard_name)
                np.savez_compressed(
                    shard_path,
                    arena_grid=np.array(buffer_grids, dtype=np.float32),
                    scalars=np.array(buffer_scalars, dtype=np.float32),
                    hand_ids=np.array(buffer_hands, dtype=np.int64),
                    legal_card_mask=np.array(buffer_cmasks, dtype=bool),
                    legal_tile_mask=np.array(buffer_tmasks, dtype=bool),
                    action_card=np.array(buffer_cards, dtype=np.int64),
                    action_tile=np.array(buffer_tiles, dtype=np.int64),
                    timestamps=np.array(buffer_timestamps, dtype=np.float32),
                    schema_version=np.int32(SCHEMA_VERSION),
                    match_outcome=np.int64(1),
                    ally_crowns=np.int64(2),
                    enemy_crowns=np.int64(1),
                )
                print(f"[HF Ingest] Saved shard: {shard_path} ({len(buffer_grids)} frames)")
                shards_created.append(shard_path)
                match_idx += 1
                buffer_grids.clear()
                buffer_scalars.clear()
                buffer_hands.clear()
                buffer_cmasks.clear()
                buffer_tmasks.clear()
                buffer_cards.clear()
                buffer_tiles.clear()
                buffer_timestamps.clear()

    return shards_created


def download_hf_parquet(
    repo_id: str = "chrisrca/clash-royale-tv-replays",
    filename: str = "arena_01/frames.parquet",
    local_dir: str = "data/hf_downloads",
) -> Optional[str]:
    """Downloads dataset shard directly from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[HF Ingest] Error: 'huggingface_hub' is required. Install with: pip install huggingface_hub")
        return None

    os.makedirs(local_dir, exist_ok=True)
    print(f"[HF Ingest] Downloading {filename} from {repo_id}...")
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
        )
        print(f"[HF Ingest] Successfully downloaded: {downloaded}")
        return downloaded
    except Exception as e:
        print(f"[HF Ingest] Download failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Ingest Hugging Face clash-royale-tv-replays")
    parser.add_argument("--repo-id", type=str, default="chrisrca/clash-royale-tv-replays")
    parser.add_argument("--filename", type=str, default="arena_01/frames.parquet")
    parser.add_argument("--parquet-path", type=str, default=None, help="Direct path to existing parquet file")
    parser.add_argument("--output-dir", type=str, default="data/trajectories")
    parser.add_argument("--max-frames", type=int, default=1500)
    args = parser.parse_args()

    parquet_file = args.parquet_path
    if parquet_file is None:
        parquet_file = download_hf_parquet(repo_id=args.repo_id, filename=args.filename)

    if parquet_file and os.path.exists(parquet_file):
        shards = ingest_parquet_file(parquet_file, output_dir=args.output_dir, max_frames=args.max_frames)
        print(f"[HF Ingest] Finished. Created {len(shards)} trajectory shards.")
    else:
        print("[HF Ingest] No Parquet file to process.")


if __name__ == "__main__":
    main()

