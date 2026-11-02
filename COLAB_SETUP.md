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
!git clone https://github.com/your-username/CR-Zero.git
%cd CR-Zero
import sys, os
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
```

---

## 2. Scraping YouTube VoDs & Retroactive Trajectory Extraction

The extractor downloads YouTube tournament/top ladder match VoDs, samples frames at 10 FPS, detects deployment clocks and card replacements, and pairs pre-play states ($t - 0.5\text{s}$) with the executed action:

```python
# Cell 2: Extract Trajectories from YouTube Gameplay
from colab_pipeline.yt_action_extractor import YouTubeTrajectoryExtractor

# Add YouTube match URLs (tournament gameplay, top 200 ladder matches)
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=VIDEO_ID_1",
    "https://www.youtube.com/watch?v=VIDEO_ID_2",
]

extractor = YouTubeTrajectoryExtractor(
    lookback_frames=5,   # 500ms lookback before card leaves hand
    lookahead_frames=2,  # 200ms forward lookahead
    wait_ratio=0.5       # 1 Wait sample for every 2 Card plays (prevents spamming)
)

os.makedirs("./data/trajectories", exist_ok=True)

for i, url in enumerate(YOUTUBE_URLS):
    out_shard = f"./data/trajectories/shard_{i:03d}.npz"
    print(f"Processing VoD [{i+1}/{len(YOUTUBE_URLS)}]: {url}")
    dataset = extractor.process_video(
        video_source=url,
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

