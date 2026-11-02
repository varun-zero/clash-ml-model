"""
Google Colab Execution Cell for CR-Zero Training Pipeline.
Copy and paste this script directly into a Google Colab T4 / A100 GPU notebook cell.
"""

# ==============================================================================
# CELL 1: Environment Setup & Dependencies
# ==============================================================================
COLAB_CELL_SETUP = """
# Step 1: Install system & python packages
!apt-get update -qq && apt-get install -y -qq ffmpeg git
!pip install -q yt-dlp ultralytics onnx onnxruntime-gpu opencv-python

# Step 2: Verify GPU availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")

# Step 3: Clone or link repository if not already in workspace
import os
import sys
if not os.path.exists("clash-ml-model"):
    !git clone https://github.com/varun-zero/clash-ml-model.git
    %cd clash-ml-model
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
"""

# ==============================================================================
# CELL 2: Scrape YouTube Gameplay & Reconstruct Trajectories
# ==============================================================================
COLAB_CELL_SCRAPE_AND_EXTRACT = """
import os
from perception.detector import ClashRoyaleDetector
from configs.constants import DEFAULT_DECK_CLASSES
from colab_pipeline.yt_action_extractor import YouTubeTrajectoryExtractor

# Step 1: Initialize detector with 2.6 Hog Cycle card classes (or pass --weights path)
detector = ClashRoyaleDetector(
    weights=None,  # Pass path to trained yolov8_cr_units.pt if available
    card_classes=DEFAULT_DECK_CLASSES,
)

# Step 2: Initialize retroactive trajectory extractor
extractor = YouTubeTrajectoryExtractor(
    detector=detector,
    lookback_frames=5,   # 500ms lookback before card leaves hand
    lookahead_frames=2,  # 200ms forward lookahead
    wait_ratio=0.5       # 1 Wait sample for every 2 Card plays (~33% waits in dataset)
)

# High-level Clash Royale tournament / top ladder replay URLs
YOUTUBE_VOD_URLS = [
    # (url, player_side: 'bottom' or 'top')
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "bottom"),
]

os.makedirs("./data/trajectories", exist_ok=True)

for i, (url, player_side) in enumerate(YOUTUBE_VOD_URLS):
    output_shard = f"./data/trajectories/shard_{i:03d}.npz"
    try:
        print(f"--- Extracting Trajectory from VoD {i+1}/{len(YOUTUBE_VOD_URLS)} (Side: {player_side}) ---")
        dataset = extractor.process_video(
            video_source=url,
            player_side=player_side,
            output_npz_path=output_shard,
            max_duration_seconds=300 # Extract first 5 minutes per VoD
        )
    except Exception as e:
        print(f"Failed to process {url}: {e}")
"""

# ==============================================================================
# CELL 3: Train DeckConditionedPolicy via Behavior Cloning & Export ONNX
# ==============================================================================
COLAB_CELL_TRAIN = """
# Run behavior cloning training with mixed precision and cosine annealing LR
!python colab_pipeline/train_bc.py \
    --data_dir ./data/trajectories \
    --save_dir ./checkpoints \
    --epochs 25 \
    --batch_size 64 \
    --lr 1e-3 \
    --synthetic

# Download trained checkpoints to your local Windows machine
from google.colab import files
files.download("./checkpoints/cr_zero_policy_best.pt")
files.download("./checkpoints/cr_zero_policy_best.onnx")
"""

if __name__ == "__main__":
    print("Google Colab Pipeline Code Templates:")
    print("=" * 60)
    print(COLAB_CELL_SETUP)
    print("=" * 60)
    print(COLAB_CELL_SCRAPE_AND_EXTRACT)
    print("=" * 60)
    print(COLAB_CELL_TRAIN)

