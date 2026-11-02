# CR-Zero: Real-Time Clash Royale AI Agent

**CR-Zero** is a modular, deep reinforcement learning and imitation learning AI agent for *Clash Royale*, engineered across two distinct operating environments:
1. **Google Colab Training Pipeline:** Scrapes YouTube gameplay VoDs via `yt-dlp`, detects units and UI via Ultralytics YOLO, reconstructs $(S_t, A_t)$ trajectories using **Retroactive Action Reconstruction** (matching deployment clocks with card hand transitions), and trains a **Deck-Conditioned Factored Policy Network** via Behavior Cloning.
2. **Windows 11 Local Runtime:** Captures a BlueStacks 5 window locked at $540 \times 960$ or $720 \times 1280$ using DirectX Desktop Duplication (`dxcam` with `mss` fallback) at 15–30 FPS, runs FP16 YOLO detection on an NVIDIA RTX 3050 GPU, encodes the arena into an $(8, 32, 18)$ spatial tensor + scalar vector, applies Elixir and Spell legality action masks, and dispatches non-blocking clicks via persistent ADB shell pipes.

---

## Architecture Overview

```
                      +------------------------------------------+
                      |       YouTube VoDs (1080p 60fps)         |
                      +------------------------------------------+
                                           |
                                [yt-dlp + Frame Sampler]
                                           |
                                           v (10 FPS)
+-------------------------+   +----------------------------------+
| BlueStacks 5 (540x960)  |   | YOLO Perception & Clock Detector |
+-------------------------+   +----------------------------------+
             |                                 |
     [dxcam DirectX]             [Retroactive Action Reconstruction]
             |                   (State at t-0.5s <-> Drop at t)
             v                                 |
+-------------------------+                    v
| Vision State Encoder    |   +----------------------------------+
|  - Spatial (8, 32, 18)  |   | Compressed Trajectories (.npz)   |
|  - Scalars (12,)        |   +----------------------------------+
|  - Action Masks (5, 576)|                    |
+-------------------------+                    v
             |                +----------------------------------+
             v                | Deck-Conditioned Policy Training |
+-------------------------+   |  - AdamW + CosineAnnealingLR     |
| DeckConditionedPolicy   |   |  - Mixed Precision (FP16)        |
|  - 135 Card Embeddings  |   |  - Multi-Head Factored Loss      |
|  - ResNet Spatial Conv  |   +----------------------------------+
|  - Card Head: 5 Logits  |                    |
|  - Tile Head: 576 Logits|                    | (Best Checkpoint)
+-------------------------+                    v
             |               <-- [cr_zero_policy_best.pt / .onnx]
             v
+-------------------------+
| Fast ADB Harness        |
|  - Persistent Shell Pipe|
|  - Two-Tap Non-blocking |
+-------------------------+
```

---

## Repository Structure

```
clashroyale/
├── configs/
│   ├── __init__.py
│   └── constants.py              # 32x18 grid specs, 9:16 viewport boxes, 135-card DB, BuildABot interop
├── perception/
│   ├── __init__.py
│   ├── detector.py               # YOLO & ONNX FP16 unit/UI detector, clock detector, elixir reader
│   └── state_encoder.py          # Encodes (8, 32, 18) grid, 12 scalars, and legality masks
├── policy/
│   ├── __init__.py
│   └── model.py                  # DeckConditionedPolicy (card embeddings + factored Card->Tile heads)
├── colab_pipeline/
│   ├── __init__.py
│   ├── yt_downloader.py          # yt-dlp downloader with automated resolution filtering
│   ├── yt_action_extractor.py    # Retroactive action reconstruction (t - 0.5s lookback)
│   ├── dataset.py                # Multi-shard PyTorch Dataset with horizontal flip augmentation
│   ├── train_bc.py               # Behavior cloning trainer (AdamW, Cosine LR, FP16, ONNX export)
│   └── colab_notebook_cell.py    # Copy-paste cells for Google Colab GPU environment
├── controller/
│   ├── __init__.py
│   ├── adb_harness.py            # Sub-ms persistent ADB shell pipe & threaded tap queue
│   └── screen_capture.py         # Windows 11 DirectX Desktop Duplication (dxcam) + mss fallback
├── main_live.py                  # 5-10 Hz real-time Windows live execution loop
├── requirements.txt              # Windows 11 (RTX 3050 CUDA 12.x) dependencies
├── COLAB_SETUP.md                # Complete Google Colab execution guide
└── README.md                     # Documentation
```

---

## 1. Perception & Tensor Specifications

### Spatial Arena Grid: `(8, 32, 18)` (`float32`)
The battlefield is mapped to 32 rows $\times$ 18 columns ($576$ tiles). Channel layout:
- **Ch 0:** `ally_ground_count`
- **Ch 1:** `ally_air_count`
- **Ch 2:** `ally_building`
- **Ch 3:** `enemy_ground_count`
- **Ch 4:** `enemy_air_count`
- **Ch 5:** `enemy_building`
- **Ch 6:** `spell_active`
- **Ch 7:** `deployable_zone_mask` ($1.0$ on valid ally deployment territory, $0.0$ on river/enemy zone)

### Scalars Vector: `(12,)` (`float32`, Normalized to $[0, 1]$)
- `[0]`: Ally Elixir ($0.0 \dots 1.0$)
- `[1]`: Estimated Enemy Elixir ($0.0 \dots 1.0$)
- `[2]`: Match Time Remaining ($0.0 \dots 1.0$, normalized from 300s)
- `[3..8]`: Tower HP ratios for:
  - `[3]`: Ally King Tower
  - `[4]`: Ally Left Princess Tower
  - `[5]`: Ally Right Princess Tower
  - `[6]`: Enemy King Tower
  - `[7]`: Enemy Left Princess Tower
  - `[8]`: Enemy Right Princess Tower
- `[9]`: Double Elixir Indicator ($1.0$ if active, else $0.0$)
- `[10]`: Overtime Indicator ($1.0$ if active, else $0.0$)
- `[11]`: Ally King Tower Activation status ($1.0$ if damaged or princess down, else $0.0$)

### Legality Action Masks
- **Legal Card Mask `(5,)`:** `[True, slot0_affordable, slot1_affordable, slot2_affordable, slot3_affordable]`
  Waiting (slot 0) is always legal. Hand slots are only legal if `card_cost <= current_elixir`.
- **Legal Tile Mask `(576,)`:**
  - Troops & Buildings: Restricted strictly to ally territory (rows 16..31) plus pocket territory if enemy princess towers are down.
  - Spells (Fireball, Arrows, Log, Rocket, Miner, Goblin Drill): All 576 tiles are legally deployable.

---

## 2. Policy Network: `DeckConditionedPolicy`

- **Generalization:** Uses an `nn.Embedding(135, 16)` layer for cards in hand, enabling the policy to generalize across **any Clash Royale deck** encountered in YouTube replays.
- **Residual Spatial CNN:** Processes the `(8, 32, 18)` grid with residual convolutional blocks down to $2560$ features.
- **Autoregressive Factored Decisions:**
  1. **Card Head:** Outputs 5 logits (`0=Wait, 1..4=Slots 0..3`). Illegal slots are masked with $-10^4$.
  2. **Tile Head:** Receives trunk features concatenated with the chosen card's embedding, outputting 576 spatial logits masked with the card's specific deployment mask.

$$\mathcal{L}_{\text{total}} = \text{CrossEntropy}(\hat{a}_{\text{card}}, a_{\text{card}}^*) + \mathbb{I}(a_{\text{card}}^* > 0) \cdot \text{CrossEntropy}(\hat{a}_{\text{tile}}, a_{\text{tile}}^*)$$

---

## 3. Windows 11 Local Setup & Live Execution

### Prerequisites
1. Windows 11 with an NVIDIA RTX 3050 (or higher) GPU.
2. BlueStacks 5:
   - Resolution locked to **540 x 960** (or **720 x 1280**) Portrait.
   - Android Debug Bridge (ADB) enabled in BlueStacks Settings -> Advanced -> **Android Debug Bridge** (`127.0.0.1:5555`).
3. ADB CLI installed in PATH (`adb devices` confirms `127.0.0.1:5555 device`).

### Installation
```powershell
# In Windows PowerShell / Command Prompt
cd c:\Users\varun\Projects\clashroyale

# Install PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install runtime dependencies
pip install -r requirements.txt
```

### Run Live Inference
```powershell
python main_live.py --checkpoint ./checkpoints/cr_zero_policy_best.pt --hz 5.0
```
Flags:
- `--checkpoint`: Path to trained PyTorch weights.
- `--device`: ADB device address (default: `127.0.0.1:5555`).
- `--hz`: Control frequency (default: 5.0 Hz).
- `--no-debug`: Disables the OpenCV visual HUD window for minimal latency.

---

## 4. Google Colab Training Pipeline

Refer to [COLAB_SETUP.md](file:///c:/Users/varun/Projects/clashroyale/COLAB_SETUP.md) for full instructions.

Quick summary:
```bash
# 1. Scrape YouTube gameplay and reconstruct trajectories:
python -c "
from colab_pipeline.yt_action_extractor import YouTubeTrajectoryExtractor
extractor = YouTubeTrajectoryExtractor()
extractor.process_video('https://www.youtube.com/watch?v=VIDEO_ID', output_npz_path='./data/trajectories/shard_001.npz')
"

# 2. Train Behavior Cloning policy:
python colab_pipeline/train_bc.py --data_dir ./data/trajectories --epochs 25 --batch_size 64
```
Outputs both `cr_zero_policy_best.pt` and `cr_zero_policy_best.onnx` into `./checkpoints/`.

