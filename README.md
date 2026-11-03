# CR-Zero: Autonomous Clash Royale AI Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/tests-25%2F25%20passed-brightgreen.svg)]()

**CR-Zero** is an end-to-end autonomous AI agent for *Clash Royale*, combining real-time computer vision, opponent cycle/elixir tracking, heuristic decision engines, and a dual-stage deep policy (Behavior Cloning + Advantage-Weighted Regression).

The system operates across two coordinated subsystems:
1. **Windows 11 Live Actuation Engine:** Captures the BlueStacks 5 client window ($540 \times 960$ portrait) via DirectX Desktop Duplication / ADB memory streaming at 15–30 FPS with ~40ms inter-tap latency, parses arena states at 5 Hz, tracks card hands and elixir, evaluates actions, and dispatches safety-gated taps.
2. **Dual-Stage Offline RL & Imitation Pipeline:** Trains a **Deck-Conditioned Policy Network** on GPU using **Behavior Cloning (BC)** with match-outcome weighting ($1.5\times$ for wins, $0.5\times$ for losses), followed by **Advantage-Weighted Regression (AWR)** fine-tuning ($w_t = \text{clip}(\exp(A_t / \beta), 0.05, 10.0)$ normalized to mean $1.0$) based on forward 10-second tower damage deltas and elixir trade differentials.

---

## Autonomous Match Record

CR-Zero has demonstrated continuous autonomous battle loops on live ladder and training camp, securing **5 consecutive verified victories** using its tactical RuleBot heuristic engine (including 3–0 Three-Crown sweeps against ladder opponents, advancing trophies from 28 $\to$ 58). (In parallel, BC card/tile prior on 30 Vanguard matches (no board state) achieves 43.2% card accuracy and 16.2% tile within 2 tiles).

![Official In-Game Battle Log](official_battle_log.png)

---

## System Architecture

```
                                  [Offline Datasets]
              +-------------------------------------------------------+
              | - VanguardX101/IL_Replay (17.8M actions / 252k games) |
              | - Nitesh-04/Clash-Royale-Cards-Data (Official stats)  |
              | - chrisrca/clash-royale-tv-replays (540x960 frames)   |
              | - wty-yy/Clash-Royale-Detection-Dataset (26k sprites)  |
              +-------------------------------------------------------+
                                          |
                              [tools/ingest_*.py]
                                          |
                                          v
+------------------------+   +---------------------------------------+
| BlueStacks 5 (540x960) |   | Standardized Trajectory Shards (.npz) |
+------------------------+   |  - Spatial Arena Grid (32, 32, 18)    |
            |                |  - Scalar Features (12,)              |
   [DirectX / Fast ADB]      |  - Legal Action Masks (5,) & (576,)   |
            |                +---------------------------------------+
            v                                     |
+------------------------+                        v
| Perception Engine      |   +---------------------------------------+
|  - Screen State Gate   |   | Stage 1: Behavior Cloning (BC)        |
|  - Template Hand Match |   |  - Cross-Entropy + Outcome Weighting  |
|  - Elixir Color Tracker|   +---------------------------------------+
|  - Tower HP Detectors  |                        |
+------------------------+                        v
            |                +---------------------------------------+
            v                | Stage 2: Offline RL (AWR)             |
+------------------------+   |  - Advantage-Weighted Regression      |
| Strategic Policy       |   |  - Tower Damage & Elixir Advantage    |
|  - 135 Card Embeddings |   +---------------------------------------+
|  - Opponent Tracker    |                        |
|  - Spell Lead Aiming   |                        v
+------------------------+   <-- [checkpoints/cr_zero_policy_best.pt]
            |
            v
+------------------------+
| Safety ADB Harness     |
|  - 800ms Cooldown      |
|  - Resolution Mapping  |
|  - F12 Emergency Panic |
+------------------------+
```

---

## 1. Integrated Public Datasets & Repositories

| Dataset / Repository | Platform / Creator | Volume & Format | Integration in CR-Zero |
| :--- | :--- | :--- | :--- |
| [`VanguardX101/IL_Replay`](https://huggingface.co/datasets/VanguardX101/IL_Replay) | Hugging Face | 17.8 Million card actions / 252k Ranked matches in Parquet | Ingested via `tools/ingest_vanguard_replays.py` into 30 canonical `schema_version=2` `.npz` shards matching $(32, 18)$ spatial coordinates |
| [`Nitesh-04/Clash-Royale-Cards-Data`](https://huggingface.co/datasets/Nitesh-04/Clash-Royale-Cards-Data) | Hugging Face | Official Supercell card JSON (`clash_royale_cards_1.json`) | Ingested via `tools/update_card_features.py`, populating official attributes (air targeting, swarm, speed, range) |
| [`wty-yy/Clash-Royale-Detection-Dataset`](https://github.com/wty-yy/Clash-Royale-Detection-Dataset) | GitHub | 26,798 transparent in-arena sprites & segments | Cloned in `data/cr_detection_dataset` for card classification templates and in-arena unit segmentation |
| [`chrisrca/clash-royale-tv-replays`](https://huggingface.co/datasets/chrisrca/clash-royale-tv-replays) | Hugging Face | 52,900 frames at 10 FPS | Native $540 \times 960$ Android emulator recordings across all 31 arenas |
| [`clash-royale-bhjq1`](https://universe.roboflow.com/clash-royale-data-collection/clash-royale-bhjq1) | Roboflow Universe | 12,022 annotated gameplay images | 92.3% mAP@50 YOLO detection baseline |

---

## 2. Perception & Observation Specifications

Canonical schema is strictly defined in [`configs/observation_schema.py`](configs/observation_schema.py) (`SCHEMA_VERSION = 2`).

### Spatial Arena Grid: `(32, 32, 18)` (`float32`)
The arena is mapped to 32 rows $\times$ 18 columns ($576$ tiles). Stacked across the 4 most recent frames ($4 \times 8 = 32$ channels):
- **Ch 0 (`Ch.ALLY_GROUND`):** Ally ground troops count
- **Ch 1 (`Ch.ALLY_AIR`):** Ally air troops count
- **Ch 2 (`Ch.ALLY_BUILDING`):** Ally buildings & towers (King at `(31, 9)`, Princesses at `(26, 3)` and `(26, 14)`)
- **Ch 3 (`Ch.ENEMY_GROUND`):** Enemy ground troops count
- **Ch 4 (`Ch.ENEMY_AIR`):** Enemy air troops count
- **Ch 5 (`Ch.ENEMY_BUILDING`):** Enemy buildings & towers (King at `(0, 9)`, Princesses at `(5, 3)` and `(5, 14)`)
- **Ch 6 (`Ch.SPELL`):** Active spells zone
- **Ch 7 (`Ch.DEPLOYABLE`):** Legal deployable mask (including river rows 15–16 bridges at cols 3–4 and 13–14, and enemy pocket territory when a tower falls)

### Scalar Features: `(12,)` (`float32`, Normalized to $[0, 1]$)
- `[0]` (`Sc.ALLY_ELIXIR`): Ally Elixir ($0.0 \dots 1.0$)
- `[1]` (`Sc.ENEMY_ELIXIR`): Estimated Enemy Elixir (from `OpponentTracker`, $0.0 \dots 1.0$)
- `[2]` (`Sc.TIME_LEFT`): Time Remaining ($t_{\text{left}} / 180.0$)
- `[3]` (`Sc.ALLY_KING`): Ally King Tower HP ($0.0 \dots 1.0$)
- `[4]` (`Sc.ALLY_LEFT`): Ally Left Princess Tower HP ($0.0 \dots 1.0$)
- `[5]` (`Sc.ALLY_RIGHT`): Ally Right Princess Tower HP ($0.0 \dots 1.0$)
- `[6]` (`Sc.ENEMY_KING`): Enemy King Tower HP ($0.0 \dots 1.0$)
- `[7]` (`Sc.ENEMY_LEFT`): Enemy Left Princess Tower HP ($0.0 \dots 1.0$)
- `[8]` (`Sc.ENEMY_RIGHT`): Enemy Right Princess Tower HP ($0.0 \dots 1.0$)
- `[9]` (`Sc.LAST_CARD`): Last Played Card Slot ($s / 4.0$, persists until next play)
- `[10]` (`Sc.LAST_TILE`): Last Played Tile Index ($\text{idx} / 575.0$, persists until next play)
- `[11]` (`Sc.DOUBLE_ELIXIR`): Double Elixir Indicator ($1.0$ if $t_{\text{left}} \le 60\text{s}$, else $0.0$)

> [!IMPORTANT]
> **Zero Label Leakage**: Match outcome is sequestered exclusively in the `match_outcome` metadata ($+1$ for victory, $-1$ for defeat) for offline loss weighting. Scalar `[11]` strictly represents the double elixir game phase, preventing post-match label leakage into the policy input.

### Legality Action Masks
- **Card Mask `(5,)`:** `[Wait=True, slot0_affordable, slot1_affordable, slot2_affordable, slot3_affordable]`.
- **Tile Mask `(576,)`:** Troops & buildings restricted to ally territory (rows 16..31) plus pocket territory if an enemy tower is destroyed. Spells and miner/goblin barrel can target any tile across the entire board.

---

## 3. Strategic Engine & Predictive Spell Physics

### Dynamic Spell Lead-Aiming
CR-Zero solves the projectile travel latency problem by calculating flight time from the King Tower:
$$t_{\text{flight}} = 1.0 + \frac{d}{8.0}$$
* **Stationary Units:** Targeted at their exact location with $0.0$ tile lead.
* **Marching Enemy Swarms:** Leading target by $+1.5$ to $+3.0$ tiles along the Y-axis into their path toward the bridge or princess tower.
* **Tower Finisher Protection:** 35-second opening lock prevents premature damage spells on full-HP towers.

### Opponent Elixir & Card Cycle Tracker
* Tracks enemy elixir expenditure and regeneration (1 elixir per 2.8s in normal time; 1 per 1.4s in double elixir).
* Tracks enemy 4-card cycle in memory to anticipate counter-pushes.
* Identifies meta archetypes (Log Bait, 2.6 Hog, Golem Beatdown) and predicts unrevealed deck cards.

---

## 4. Live Execution & Safety Controls

### Safety Rails
* **Resolution Independence:** Automatically detects device rendering bounds and normalizes tap coordinates to client resolution ($540 \times 960$ or $720 \times 1280$).
* **Emergency Brake (F12):** Pressing **F12** at any moment triggers a graceful shutdown, aborting pending ADB taps and releasing file handles.
* **State Machine Guard:** Actuation is strictly gated to the `"battle"` state. Menus, loading screens, and victory prompts are handled by autonomous UI handlers.
* **Deployment Cooldown:** Enforces an 800ms minimum action cooldown between card drops to prevent server-side action dropping.

### Live Commands

#### Autonomous Ladder Match Runner (5 Consecutive Wins)
```powershell
python tools/play_5_wins.py --target-wins 5 --max-games 10
```

#### Live Match Play via Policy or Heuristics
```powershell
# 1. Run with Stage 2 AWR trained neural policy:
python main_live.py --weights weights/yolo_units.pt --checkpoint checkpoints/cr_zero_policy_best.pt

# 2. Run with heuristic RuleBot (no neural weights required):
python main_live.py --rule-bot

# 3. Dry-run perception verification (safe test without real clicks):
python main_live.py --rule-bot --dry-run
```

---

## 5. Dual-Stage Training Pipeline

Interactive notebooks:
* Local workspace / IDE: [`train_cr_zero.ipynb`](train_cr_zero.ipynb)
* Google Colab: [`colab_pipeline/cr_zero_training.ipynb`](colab_pipeline/cr_zero_training.ipynb)

### CLI Training Commands

```powershell
# 1. Update official card features from Hugging Face
python tools/update_card_features.py

# 2. Ingest Vanguard ranked replays for Stage 1 Prior Pretraining
python tools/ingest_vanguard_replays.py --max-matches 30 --output-dir data/pretraining/vanguard

# 3. Execute Stage 1 Behavior Cloning (BC) Prior Pretraining
python colab_pipeline/train_bc.py --data_dir data/pretraining/vanguard --save_dir checkpoints --epochs 10 --batch_size 64

# 4. Execute Stage 2 Heuristic Reward-Weighted BC (AWR-style advantage weighting on live/VoD trajectories)
# NOTE: train_awr.py strictly requires genuine battlefield states and rejects Vanguard replays
python colab_pipeline/train_awr.py --shards-dir data/trajectories --init-checkpoint checkpoints/cr_zero_policy_best.pt --epochs 15 --batch-size 64 --beta 0.5
```

---

## 6. Verification & Test Suite

Run the full pytest suite:
```powershell
pytest tests/ -v
```

```
tests/test_all_requirements.py::TestCRZeroRequirements::test_01_global_card_vocabulary_and_unlocked PASSED
tests/test_all_requirements.py::TestCRZeroRequirements::test_02_regression_card_legality_and_affordability PASSED
tests/test_all_requirements.py::TestCRZeroRequirements::test_03_card_embedding_and_unseen_card PASSED
tests/test_all_requirements.py::TestCRZeroRequirements::test_04_canonicalization PASSED
tests/test_all_requirements.py::TestCRZeroRequirements::test_05_rule_bot_heuristics PASSED
tests/test_all_requirements.py::TestCRZeroRequirements::test_06_mocked_adb_safety_controls PASSED
tests/test_all_requirements.py::TestCRZeroRequirements::test_07_outcome_weighted_dataset PASSED
tests/test_awr_and_tracker.py::TestAWRAndTracker::test_adb_action_lifecycle_and_acknowledgement PASSED
tests/test_awr_and_tracker.py::TestAWRAndTracker::test_awr_dataset_weighting PASSED
tests/test_awr_and_tracker.py::TestAWRAndTracker::test_awr_rejects_vanguard_data PASSED
tests/test_awr_and_tracker.py::TestAWRAndTracker::test_awr_strict_timestamp_validation PASSED
tests/test_awr_and_tracker.py::TestAWRAndTracker::test_awr_training_step PASSED
tests/test_awr_and_tracker.py::TestAWRAndTracker::test_meta_deck_identification_and_prediction PASSED
tests/test_awr_and_tracker.py::TestAWRAndTracker::test_opponent_cycle_and_in_hand PASSED
tests/test_awr_and_tracker.py::TestAWRAndTracker::test_opponent_tracker_elixir_and_cycle PASSED
tests/test_golden_hand.py::test_golden_frame_known_hand_detection PASSED
tests/test_golden_hand.py::test_golden_frame_blank_and_unknown_slots_return_zero PASSED
tests/test_golden_hand.py::test_no_synthetic_sprite_templates_present PASSED
tests/test_schema_parity.py::test_no_target_leakage_and_hand_alignment PASSED
tests/test_schema_parity.py::test_vanguard_scalars_follow_live_encoder_layout PASSED
tests/test_schema_parity.py::test_unreconstructable_hand_is_rejected_not_guessed PASSED
tests/test_schema_parity.py::test_card_key_normalization_xbow_and_minipekka PASSED
tests/test_schema_parity.py::test_strict_schema_enforcement PASSED
tests/test_schema_parity.py::test_vanguard_ambiguity_rejection PASSED
tests/test_schema_parity.py::test_play_live_match_shard_contains_schema_version PASSED
============================= 25 passed in 10.88s =============================
```

---

## License

This project is licensed under the [MIT License](LICENSE).
Clash Royale is a registered trademark of Supercell Oy. This project is an independent open-source research initiative not affiliated with, endorsed, or sponsored by Supercell.
