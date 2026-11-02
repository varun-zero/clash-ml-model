"""Generates and writes train_cr_zero.ipynb and colab_pipeline/cr_zero_training.ipynb."""
import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CR-Zero: Dual-Stage Deep RL & Imitation Learning Pipeline\n",
                "\n",
                "Trains the **Deck-Conditioned Policy Network** for the **CR-Zero Autonomous Clash Royale AI**.\n",
                "\n",
                "### Pipeline Highlights:\n",
                "- **Stage 1 (Behavior Cloning)**: Ingests 540x960 native match trajectories with outcome weighting.\n",
                "- **Stage 2 (Advantage-Weighted Regression - AWR)**: Rewards actions yielding positive elixir trades and tower HP deltas: $\\Delta \\text{TowerHP} + 0.15 \\times (E_{ally} - E_{enemy})$, exponentiated by $\\exp(A / \\beta)$.\n",
                "- **Multi-Source Ingestion**: Hugging Face TV Royale (`chrisrca/clash-royale-tv-replays`), Roboflow Universe YOLO (`clash-royale-bhjq1`), and curated 1v1 replays.\n",
                "- **Production Export**: Auto-exports PyTorch checkpoints (`.pt`) and FP16-ready ONNX models (`.onnx`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Environment & Dependency Setup\n",
                "import os\n",
                "import sys\n",
                "\n",
                "IS_COLAB = \"google.colab\" in sys.modules\n",
                "\n",
                "if IS_COLAB:\n",
                "    print(\"[*] Running inside Google Colab (GPU Runtime)\")\n",
                "    !apt-get update -qq && apt-get install -y -qq ffmpeg git\n",
                "    !pip install -q yt-dlp ultralytics onnx onnxruntime-gpu opencv-python huggingface_hub pyarrow roboflow\n",
                "    if not os.path.exists(\"clashroyale\"):\n",
                "        !git clone https://github.com/varun-zero/clash-ml-model.git clashroyale\n",
                "        %cd clashroyale\n",
                "else:\n",
                "    print(\"[*] Running inside Local Workspace / IDE\")\n",
                "\n",
                "repo_root = os.path.abspath(os.getcwd())\n",
                "if repo_root not in sys.path:\n",
                "    sys.path.insert(0, repo_root)\n",
                "\n",
                "import torch\n",
                "print(f\"PyTorch:         {torch.__version__}\")\n",
                "print(f\"CUDA Available:  {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"Active GPU:      {torch.cuda.get_device_name(0)}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Inspect Existing Trajectory Shards"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Verify Trajectory Dataset Shards\n",
                "import glob\n",
                "import numpy as np\n",
                "\n",
                "data_dir = os.path.join(\".\", \"data\", \"trajectories\")\n",
                "shards = sorted(glob.glob(os.path.join(data_dir, \"*.npz\")))\n",
                "print(f\"Found {len(shards)} trajectory shard(s) in {data_dir}:\")\n",
                "\n",
                "for s in shards:\n",
                "    data = np.load(s)\n",
                "    outcome = int(data[\"match_outcome\"]) if \"match_outcome\" in data else 0\n",
                "    outcome_str = \"WIN (+1)\" if outcome == 1 else (\"LOSS (-1)\" if outcome == -1 else \"DRAW (0)\")\n",
                "    n_frames = data[\"arena_grid\"].shape[0]\n",
                "    has_rew = \"rewards\" in data\n",
                "    print(f\"  -> {os.path.basename(s)}: {n_frames} frames | Outcome: {outcome_str} | Rewards: {has_rew}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Ingest External Datasets (Modular Options)\n",
                "Run any of the cells below to pull datasets from Hugging Face, Roboflow, or YouTube."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Ingest Hugging Face TV Royale Replays (540x960 Native Android)\n",
                "# Dataset: chrisrca/clash-royale-tv-replays\n",
                "!python tools/ingest_hf_replays.py --filename arena_01/frames.parquet --max-frames 1200\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Fine-Tune YOLO on Roboflow Clash Royale Dataset\n",
                "# Dataset: clash-royale-data-collection / clash-royale-bhjq1 (12,022 annotations, 92.3% mAP@50)\n",
                "# Set your free Roboflow API key (optional):\n",
                "# %env ROBOFLOW_API_KEY=your_key_here\n",
                "# !python tools/train_yolo_units.py --epochs 25 --weights-output weights/yolo_units.pt\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Download & Extract Curated 1v1 Replays (Giant Beatdown Archetype)\n",
                "!python tools/download_curated_replays.py --url https://www.youtube.com/watch?v=tBPiYbzDprE\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Stage 1: Behavior Cloning Training\n",
                "Trains the base policy network on gameplay trajectories with outcome-weighted cross-entropy."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Execute Stage 1 Behavior Cloning\n",
                "!python colab_pipeline/train_bc.py \\\n",
                "    --data_dir ./data/trajectories \\\n",
                "    --save_dir ./checkpoints \\\n",
                "    --epochs 10 \\\n",
                "    --batch_size 32 \\\n",
                "    --lr 1e-3\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Stage 2: Advantage-Weighted Regression (AWR / Offline RL)\n",
                "Fine-tunes the BC policy by weighting samples proportionally to their forward 10s tower damage and elixir trade advantage."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Execute Stage 2 AWR Fine-Tuning\n",
                "!python colab_pipeline/train_awr.py \\\n",
                "    --shards-dir data/trajectories \\\n",
                "    --init-checkpoint checkpoints/cr_zero_policy_best.pt \\\n",
                "    --epochs 15 \\\n",
                "    --batch-size 64 \\\n",
                "    --beta 0.5 \\\n",
                "    --lr 2e-4\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Verify Checkpoints & Export Model to Laptop"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 8: Verify Model Inference & Download Artifacts\n",
                "import torch\n",
                "from policy.model import DeckConditionedPolicy\n",
                "from configs.constants import NUM_TOTAL_CARDS, STACKED_GRID_CHANNELS\n",
                "\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "policy = DeckConditionedPolicy(num_card_classes=NUM_TOTAL_CARDS, in_channels=STACKED_GRID_CHANNELS).to(device)\n",
                "\n",
                "# Prefer AWR checkpoint if available, fallback to BC checkpoint\n",
                "awr_ckpt = \"./checkpoints/cr_zero_policy_awr_best.pt\"\n",
                "bc_ckpt = \"./checkpoints/cr_zero_policy_best.pt\"\n",
                "target_ckpt = awr_ckpt if os.path.exists(awr_ckpt) else bc_ckpt\n",
                "\n",
                "if os.path.exists(target_ckpt):\n",
                "    ckpt = torch.load(target_ckpt, map_location=device)\n",
                "    policy.load_state_dict(ckpt[\"model_state_dict\"])\n",
                "    policy.eval()\n",
                "    print(f\"[OK] Loaded weights from: {target_ckpt}\")\n",
                "    \n",
                "    # Forward inference pass\n",
                "    dummy_grid = torch.randn(1, 32, 32, 18, device=device)\n",
                "    dummy_scalars = torch.randn(1, 12, device=device)\n",
                "    dummy_hand = torch.tensor([[1, 2, 20, 21]], dtype=torch.long, device=device)\n",
                "    dummy_cmask = torch.tensor([[True, True, True, True, True]], dtype=torch.bool, device=device)\n",
                "    dummy_tmask = torch.ones(1, 576, dtype=torch.bool, device=device)\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        c_slot, t_idx, c_p, t_p = policy.sample_action(\n",
                "            dummy_grid, dummy_scalars, dummy_hand, dummy_cmask, dummy_tmask, temperature=0.7\n",
                "        )\n",
                "    print(f\"[OK] Forward inference verified: slot={c_slot}, tile={t_idx}\")\n",
                "    \n",
                "    if IS_COLAB:\n",
                "        from google.colab import files\n",
                "        files.download(target_ckpt)\n",
                "        onnx_file = target_ckpt.replace(\".pt\", \".onnx\")\n",
                "        if os.path.exists(onnx_file):\n",
                "            files.download(onnx_file)\n",
                "else:\n",
                "    print(\"[!] No checkpoint found. Run Cell 6 or Cell 7 first.\")\n"
            ]
        }
    ],
    "metadata": {
        "colab": {"toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("train_cr_zero.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

with open("colab_pipeline/cr_zero_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("[OK] Notebooks built successfully.")

