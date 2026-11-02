# CR-Zero: Real-Time Clash Royale AI Agent

**CR-Zero** is a modular deep imitation learning and reinforcement learning AI agent for *Clash Royale*, engineered across two environments:
1. **Google Colab Training Pipeline:** Downloads 720p YouTube gameplay replays via `yt-dlp`, detects units and UI via Ultralytics YOLO, reconstructs $(S_t, A_t)$ trajectories via **Retroactive Action Reconstruction** (matching deployment clocks with card hand transitions), and trains a **Deck-Conditioned Factored Policy Network** using Behavior Cloning.
2. **Windows 11 Local Runtime:** Captures the BlueStacks 5 window client area (locked to $540 \times 960$ or $720 \times 1280$) using DirectX Desktop Duplication (`dxcam` / `mss`) at 15–30 FPS, classifies screen state (`battle` / `menu` / `result`), encodes the arena into a 32-channel stacked tensor + 12-scalar vector, applies Elixir and Spell legality action masks, and dispatches verified taps via a safety-hardened ADB harness.

---

## Architecture Overview

```
                      +------------------------------------------+
                      |       YouTube VoDs (720p 60fps)          |
                      +------------------------------------------+
                                           |
                                [yt-dlp 720p Sampler]
                                           |
                                           v (10 FPS via POS_MSEC)
+-------------------------+   +----------------------------------+
| BlueStacks 5 (Client)   |   | YOLO Perception & Clock Detector |
+-------------------------+   +----------------------------------+
             |                                 |
     [Client Rect DXR]           [Retroactive Action Reconstruction]
             |                   (State at t-0.5s <-> Clock at t)
             v                                 |
+-------------------------+                    v
| Vision State Encoder    |   +----------------------------------+
|  - Stacked (32, 32, 18) |   | Match-Outcome Trajectories (.npz)|
|  - 12 Scalars Layout    |   +----------------------------------+
|  - Action Masks (5, 576)|                    |
+-------------------------+                    v
             |                +----------------------------------+
             v                | Deck-Conditioned Policy Training |
+-------------------------+   |  - AdamW + CosineAnnealingLR     |
| Policy / RuleBot Engine |   |  - Mixed Precision (FP16/AMP)    |
|  - 135 Card Embeddings  |   |  - Outcome-Weighted Losses       |
|  - Static Feature Sum   |   +----------------------------------+
|  - Factored Heads       |                    |
+-------------------------+                    v
             |               <-- [cr_zero_policy_best.pt / .onnx]
             v
+-------------------------+
| Safety ADB Harness      |
|  - Device Resolution    |
|  - 800ms Action Cooldown|
|  - F12 Panic Stop       |
+-------------------------+
```

---

## 1. Perception & Observation Specifications

### Spatial Arena Grid: `(32, 32, 18)` (`float32`)
The battlefield is mapped to 32 rows $\times$ 18 columns ($576$ tiles). Stacked across the last 4 consecutive frames (4 frames $\times$ 8 channels = 32 channels):
- **Ch 0:** `ally_ground_count`
- **Ch 1:** `ally_air_count`
- **Ch 2:** `ally_building`
- **Ch 3:** `enemy_ground_count`
- **Ch 4:** `enemy_air_count`
- **Ch 5:** `enemy_building`
- **Ch 6:** `spell_active`
- **Ch 7:** `deployable_zone_mask` ($1.0$ on valid ally deployment territory, $0.0$ on river/enemy zone; expands into enemy pocket if princess tower is destroyed)

### Scalars Vector: `(12,)` (`float32`, Normalized to $[0, 1]$)
- `[0]`: Ally Elixir ($0.0 \dots 1.0$)
- `[1]`: Estimated Enemy Elixir (constant $0.5$ in v1)
- `[2]`: Match Time Remaining ($\text{time\_left} / 300.0$)
- `[3]`: Ally King Tower HP ($0.0 \dots 1.0$)
- `[4]`: Ally Left Princess Tower HP ($0.0 \dots 1.0$, $0.0$ if destroyed)
- `[5]`: Ally Right Princess Tower HP ($0.0 \dots 1.0$, $0.0$ if destroyed)
- `[6]`: Enemy King Tower HP ($0.0 \dots 1.0$)
- `[7]`: Enemy Left Princess Tower HP ($0.0 \dots 1.0$, $0.0$ if destroyed)
- `[8]`: Enemy Right Princess Tower HP ($0.0 \dots 1.0$, $0.0$ if destroyed)
- `[9]`: Last action card slot normalized ($\text{slot} / 4.0$)
- `[10]`: Last action tile normalized ($\text{tile} / 575.0$)
- `[11]`: Double Elixir Indicator ($1.0$ if $\text{time\_left} \le 60\text{s}$ or in overtime, else $0.0$)

### Single Global Card Vocabulary & Unlocked List
- Uses a unified 135-card vocabulary from `CARD_DATABASE`.
- `configs/unlocked_cards.json` defines the editable set of global IDs unlocked by the beginner account (default starter deck: Arrows 1, Fireball 2, Knight 20, Archers 21, Goblins 22, Spear Goblins 23, Bomber 34, Minions 43).
- Hand card classifier matches only against unlocked cards. Unknown crops are logged to `unknown_cards/` for review.

### Legality Action Masks
- **Legal Card Mask `(5,)`:** `[True, slot0_affordable, slot1_affordable, slot2_affordable, slot3_affordable]`. Slot 0 (Wait) is always legal. Slots 1..4 are legal only if `cost <= elixir`.
- **Legal Tile Mask `(576,)`:**
  - Troops & Buildings: Restricted to ally territory (rows 16..31) plus pocket territory if the corresponding enemy princess tower is destroyed ($\le 0.05$).
  - Spells: All 576 tiles are legal.

---

## 2. Policy Architectures

### A. Neural Factored Policy: `DeckConditionedPolicy`
- **Summed Embeddings:** Combines a learned identity embedding with static card features:
  $$\mathbf{e}_{\text{card}} = \text{Embedding}(135, 16) + \text{Linear}(\mathbf{f}_{\text{static}} \to 16)$$
  Features: `cost`, `is_spell`, `is_building`, `targets_air`, `targets_ground`, `range_bucket`, `speed_bucket`. A card never seen during training still produces a valid embedding.
- **Factored Prediction:**
  1. `predict_card(...)` predicts the card slot (0 = Wait, 1..4 = Hand Slots).
  2. If card > 0, `predict_tile(..., card_slot=chosen_slot)` predicts the deployment tile conditioned on the chosen card embedding and its specific legal tile mask.

### B. Strategic Heuristic Engine: `RuleBotPolicy`
Deterministic beginner strategy engine for instant live play without pre-trained weights:
- Defends threats on the ally half with the cheapest valid counter.
- Targets enemy unit clusters with damage spells (Arrows / Fireball).
- Pushes the bridge with tanks/troops when elixir $\ge 9.0$ (prevents elixir leaking).
- Never plays unaffordable cards or illegal tiles.

---

## 3. Windows 11 Live Execution & Safety Controls

### Safety Rails
- **Binary Resolution:** Resolves ADB via `--adb-path`, `ADB_PATH` env var, `shutil.which("adb")`, and standard BlueStacks installation paths. Fails loudly if ADB or the target device is missing.
- **Pixel Scaling:** Queries `wm size` from device and scales tap coordinates to device pixels (with `--width` and `--height` overrides).
- **Rate Limiting & Cooldown:** Enforces an 800ms minimum action cooldown between deployments and an optional `--max-actions` limit.
- **Panic Hotkey:** Press **F12** at any time to trigger an immediate emergency stop.
- **Screen-State Gating:** Actions are only dispatched when the screen state is `"battle"`. The agent remains idle in `"menu"` and `"result"`.

### Live Execution Commands

#### 1. Test ADB Actuation (Dry-Run / Dummy Mode)
```powershell
python main_live.py --dummy --dry-run
```

#### 2. Run with Heuristic RuleBot (No Weights Required)
```powershell
python main_live.py --rule-bot --dry-run
```
To send real clicks to BlueStacks, omit `--dry-run`:
```powershell
python main_live.py --rule-bot --device 127.0.0.1:5555
```

#### 3. Run with Neural Policy (Requires Weights and Checkpoint)
```powershell
python main_live.py --weights ./checkpoints/yolov8_cr_units.pt --checkpoint ./checkpoints/cr_zero_policy_best.pt --device 127.0.0.1:5555 --log-dir ./data/live_matches
```

---

## 4. Google Colab Training Pipeline

Refer to [COLAB_SETUP.md](file:///c:/Users/varun/Projects/clashroyale/COLAB_SETUP.md) for step-by-step notebook setup.

1. **Scrape Gameplay VoDs & Reconstruct Trajectories:**
   ```bash
   python -c "
   from colab_pipeline.yt_action_extractor import YouTubeTrajectoryExtractor
   extractor = YouTubeTrajectoryExtractor()
   extractor.process_video('https://www.youtube.com/watch?v=VIDEO_ID', output_npz_path='./data/trajectories/shard_001.npz')
   "
   ```

2. **Train Policy Network:**
   ```bash
   python colab_pipeline/train_bc.py --data_dir ./data/trajectories --epochs 25 --batch_size 64
   ```
   Splits train/val by match and weights samples by match outcome (win > draw > loss). Exports `cr_zero_policy_best.pt` and `cr_zero_policy_best.onnx`.

---

## 5. Verification & Tests

Run the test suites:
```powershell
# 1. Run full unit and regression test suite
python tests/test_all_requirements.py

# 2. Test real frame perception (requires --weights)
python tests/test_real_frame.py --weights ./checkpoints/yolov8_cr_units.pt

# 3. Card template clustering tool
python tools/build_card_templates.py
```
