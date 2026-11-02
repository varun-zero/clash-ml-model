# Google Colab Training Pipeline Guide for CR-Zero

This guide details how to train **CR-Zero** using Google Colab's free T4 GPU (or A100 / V100).

---

## 1. Google Colab Notebook Setup

1. Open a new notebook at [Google Colab](https://colab.research.google.com).
2. Go to **Runtime -> Change runtime type** and select **T4 GPU** (or A100 GPU if available).
3. Paste the following setup block into the first notebook cell:

```python
# Cell 1: Environment & Dependency Installation
!apt-get update -qq && apt-get install -y -qq ffmpeg git
!pip install -q yt-dlp ultralytics onnx onnxruntime-gpu opencv-python

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Active GPU:      {torch.cuda.get_device_name(0)}")

# Clone repository or upload your files
!git clone https://github.com/varun-zero/clash-ml-model.git
%cd clash-ml-model
import sys, os
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
```

---

## 2. Scraping YouTube VoDs & Retroactive Trajectory Extraction

The extractor downloads YouTube tournament/top ladder match VoDs, samples frames at 10 FPS, detects deployment clocks and card replacements, and pairs pre-play states ($t - 0.5\text{s}$) with the executed action:

```python
# Cell 2: Extract Trajectories from YouTube Gameplay
import os
from perception.detector import ClashRoyaleDetector
from colab_pipeline.yt_action_extractor import YouTubeTrajectoryExtractor

# Add YouTube match URLs with player perspective ('bottom' or 'top')
YOUTUBE_URLS = [
    ("https://www.youtube.com/watch?v=VIDEO_ID_1", "bottom"),
    ("https://www.youtube.com/watch?v=VIDEO_ID_2", "top"),
]

# Initialize detector with global card vocabulary & optional weights
detector = ClashRoyaleDetector(
    weights="checkpoints/yolov8_cr_units.pt" if os.path.exists("checkpoints/yolov8_cr_units.pt") else None
)
extractor = YouTubeTrajectoryExtractor(
    detector=detector,
    lookback_frames=5,   # 500ms lookback before card leaves hand
    lookahead_frames=2,  # 200ms forward lookahead
    wait_ratio=0.5       # 1 Wait sample for every 2 Card plays (~33% waits in dataset)
)

os.makedirs("./data/trajectories", exist_ok=True)

for i, (url, player_side) in enumerate(YOUTUBE_URLS):
    out_shard = f"./data/trajectories/shard_{i:03d}.npz"
    print(f"Processing VoD [{i+1}/{len(YOUTUBE_URLS)}] (Side: {player_side}): {url}")
    dataset = extractor.process_video(
        video_source=url,
        player_side=player_side,
        output_npz_path=out_shard,
        max_duration_seconds=360  # 6 minutes per match
    )
```

---

## 3. Train Behavior Cloning Policy

Train the `DeckConditionedPolicy` using mixed precision (`fp16`), `AdamW`, and `CosineAnnealingLR`:

```python
# Cell 3: Training & Checkpoint Export
!python colab_pipeline/train_bc.py \
    --data_dir ./data/trajectories \
    --save_dir ./checkpoints \
    --epochs 30 \
    --batch_size 64 \
    --lr 1e-3

# Cell 4: Download Best Weights to Local Windows Machine
from google.colab import files
files.download("./checkpoints/cr_zero_policy_best.pt")
files.download("./checkpoints/cr_zero_policy_best.onnx")
```

Place the downloaded `cr_zero_policy_best.pt` inside your local Windows machine's `./checkpoints/` directory to run inference live with `main_live.py`.

