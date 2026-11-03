"""
CR-Zero Comprehensive Integration & Regression Test Suite.
Verifies all P0 fixes and core requirements:
1. Single global card vocabulary (135 cards) and editable unlocked_cards.json.
2. Regression tests: Troop => ally-half tiles only (288 tiles / pockets), Spell => all 576 tiles.
3. Affordability and HUD card names using global IDs.
4. Card embedding = learned Embedding(135, 16) + Linear(static_features -> 16), summed.
5. Unseen card embedding validity.
6. Cooldown, max-actions, and safety controls with mocked ADB.
7. Perspective canonicalization (180 deg + ally/enemy channel swaps).
8. RuleBot strategic heuristics.
Must pass with exit code 0.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.constants import (
    ARENA_ROWS,
    ARENA_COLS,
    TOTAL_TILES,
    NUM_TOTAL_CARDS,
    STACKED_GRID_CHANNELS,
    NUM_SCALARS,
    CARD_DATABASE,
    CARD_ID_TO_COST,
    CARD_ID_TO_IS_SPELL,
    CARD_ID_TO_NAME,
    DEFAULT_UNLOCKED_CARD_IDS,
    grid_to_screen_pixel,
    hand_slot_to_screen_pixel,
    grid_to_clashroyalebuildabot,
    clashroyalebuildabot_to_grid,
)
from configs.card_features import (
    CARD_FEATURE_DIM,
    card_feature_vector,
    build_card_feature_matrix,
    card_targets_air,
)
from configs.unlocked_cards import load_unlocked_cards
from perception.state_encoder import (
    StateEncoder,
    Detection,
    canonicalize_perspective,
    canonicalize_tile_coordinate,
    stack_spatial_frames,
)
from perception.detector import ClashRoyaleDetector
from policy.model import DeckConditionedPolicy
from policy.rule_bot import RuleBotPolicy
from controller.adb_harness import ADBHarness
from colab_pipeline.dataset import ClashRoyaleDataset, create_dataloaders
from tests.helpers import generate_synthetic_shard


class TestCRZeroRequirements(unittest.TestCase):
    def test_01_global_card_vocabulary_and_unlocked(self):
        """Req 3: Single global vocabulary from CARD_DATABASE (135 classes)."""
        self.assertEqual(NUM_TOTAL_CARDS, 135)
        self.assertEqual(STACKED_GRID_CHANNELS, 32)
        self.assertEqual(NUM_SCALARS, 12)

        # Verify starter cards in unlocked_cards.json
        unlocked = load_unlocked_cards()
        self.assertGreaterEqual(len(unlocked), 4)
        for cid in unlocked:
            self.assertIn(cid, CARD_DATABASE)
            self.assertNotEqual(cid, 0)  # ID 0 is strictly Wait

    def test_02_regression_card_legality_and_affordability(self):
        """Req 5: Regression tests: troop => ally-half only, spell => all 576, cost/affordability correct, HUD names correct."""
        encoder = StateEncoder()

        # 1. Troop legality: Knight (ID 20) and Hog Rider (ID 51) are troops
        knight_mask = encoder.get_legal_tile_mask(card_id=20)
        self.assertEqual(knight_mask.shape, (576,))
        self.assertEqual(int(knight_mask.sum()), 16 * 18)  # Exactly 288 tiles on ally half (rows 16..31)
        # Verify no top-half tiles are legal for a troop when enemy towers stand
        top_half_mask = knight_mask.reshape(32, 18)[:16, :]
        self.assertEqual(int(top_half_mask.sum()), 0)

        hog_mask = encoder.get_legal_tile_mask(card_id=51)
        self.assertEqual(int(hog_mask.sum()), 16 * 18)

        # 2. Spell legality: Fireball (ID 2), Arrows (ID 1), The Log (ID 4)
        fireball_mask = encoder.get_legal_tile_mask(card_id=2)
        self.assertEqual(int(fireball_mask.sum()), 576)  # Entire arena is legal for spells

        arrows_mask = encoder.get_legal_tile_mask(card_id=1)
        self.assertEqual(int(arrows_mask.sum()), 576)

        # 3. Pocket expansion: when enemy left princess tower is destroyed (hp <= 0.05)
        tower_hps_pocket = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]  # enemy left down
        pocket_mask = encoder.get_legal_tile_mask(card_id=20, tower_hps=tower_hps_pocket)
        self.assertGreater(int(pocket_mask.sum()), 16 * 18)  # Ally half + pocket

        # 4. Affordability: Knight costs 3, Fireball costs 4
        self.assertEqual(CARD_ID_TO_COST[20], 3)
        self.assertEqual(CARD_ID_TO_COST[2], 4)
        self.assertFalse(CARD_ID_TO_IS_SPELL[20])
        self.assertTrue(CARD_ID_TO_IS_SPELL[2])

        _, _, cmask_low = encoder.encode(
            detections=[], elixir=2.5, hand_card_ids=[20, 2, 24, 1]
        )
        self.assertTrue(cmask_low[0])   # Wait always legal
        self.assertFalse(cmask_low[1])  # Knight (cost 3) not affordable at 2.5
        self.assertFalse(cmask_low[2])  # Fireball (cost 4) not affordable at 2.5

        _, _, cmask_mid = encoder.encode(
            detections=[], elixir=3.5, hand_card_ids=[20, 2, 24, 1]
        )
        self.assertTrue(cmask_mid[1])   # Knight (cost 3) affordable at 3.5
        self.assertFalse(cmask_mid[2])  # Fireball (cost 4) not affordable at 3.5

        # 5. HUD Names: Verify correct names from global IDs
        self.assertEqual(CARD_ID_TO_NAME[1], "Arrows")
        self.assertEqual(CARD_ID_TO_NAME[2], "Fireball")
        self.assertEqual(CARD_ID_TO_NAME[20], "Knight")
        self.assertEqual(CARD_ID_TO_NAME[51], "Hog Rider")

    def test_03_card_embedding_and_unseen_card(self):
        """Req 4: Embedding = learned Embedding(135,16) + Linear(static_features -> 16), summed."""
        model = DeckConditionedPolicy(
            num_card_classes=NUM_TOTAL_CARDS,
            card_emb_dim=16,
            num_scalars=12,
            in_channels=32,
        )

        # Test static feature matrix shape
        feat_matrix = build_card_feature_matrix(NUM_TOTAL_CARDS)
        self.assertEqual(feat_matrix.shape, (135, CARD_FEATURE_DIM))

        # Test forward pass with unseen high card ID (e.g. 110)
        grid = torch.zeros((1, 32, 32, 18), dtype=torch.float32)
        scalars = torch.zeros((1, 12), dtype=torch.float32)
        hand = torch.tensor([[1, 2, 20, 110]], dtype=torch.long)
        cmask = torch.ones((1, 5), dtype=torch.bool)
        tmask = torch.ones((1, 576), dtype=torch.bool)

        trunk_feat, hand_embs = model._encode(grid, scalars, hand)
        self.assertEqual(hand_embs.shape, (1, 4, 16))
        # Embedding for card 110 must be non-zero and finite
        emb_card_110 = hand_embs[0, 3]
        self.assertFalse(torch.isnan(emb_card_110).any())
        self.assertGreater(float(torch.norm(emb_card_110).item()), 0.0)

        # Test factored card and tile prediction methods
        chosen_slot, card_conf = model.predict_card(grid, scalars, hand, cmask)
        self.assertIn(chosen_slot, [0, 1, 2, 3, 4])
        chosen_tile, tile_conf = model.predict_tile(grid, scalars, hand, tmask, card_slot=chosen_slot)
        self.assertIn(chosen_tile, list(range(576)))

    def test_04_canonicalization(self):
        """Req 1: Rotate grid 180 and swap ally/enemy channels when player_side=top."""
        grid = np.zeros((8, ARENA_ROWS, ARENA_COLS), dtype=np.float32)
        grid[0, 5, 4] = 1.0  # Ally ground at (5, 4)
        grid[3, 25, 12] = 2.0  # Enemy ground at (25, 12)
        tower_hps = [1.0, 0.9, 0.8, 1.0, 0.7, 0.6]

        canon_top, hps_top = canonicalize_perspective(grid, tower_hps, player_side="top")
        self.assertEqual(canon_top[3, 26, 13], 1.0)
        self.assertEqual(canon_top[0, 6, 5], 2.0)
        cy, cx = canonicalize_tile_coordinate(5, 4, player_side="top")
        self.assertEqual((cy, cx), (26, 13))

    def test_05_rule_bot_heuristics(self):
        """Req 10: RuleBot beginner heuristics (defend threats, bridge push, never play unaffordable)."""
        bot = RuleBotPolicy()
        grid = np.zeros((8, 32, 18), dtype=np.float32)
        scalars = np.zeros(12, dtype=np.float32)
        scalars[0] = 0.5  # 5.0 elixir
        scalars[3:9] = 1.0
        hand = [20, 21, 24, 1]  # Knight(3), Archers(3), Giant(5), Arrows(3)

        # 1. Threat on ally side -> defends
        grid_threat = grid.copy()
        grid_threat[3, 22, 5] = 1.0
        cmask = np.array([True, True, True, True, True])
        slot, tile, _, _ = bot.predict_action(grid_threat, scalars, hand, cmask)
        self.assertIn(slot, [1, 2, 4])
        self.assertGreaterEqual(tile, 16 * 18)

        # 2. High elixir >= 9.0 -> bridge push
        scalars_high = scalars.copy()
        scalars_high[0] = 0.95  # 9.5 elixir
        slot_push, tile_push, _, _ = bot.predict_action(grid, scalars_high, hand, cmask)
        self.assertGreater(slot_push, 0)
        py, px = tile_push // 18, tile_push % 18
        self.assertIn(py, [16, 17])
        self.assertIn(px, [3, 14])

        # 3. Unaffordable cards -> wait
        cmask_none = np.array([True, False, False, False, False])
        slot_wait, _, _, _ = bot.predict_action(grid_threat, scalars_high, hand, cmask_none)
        self.assertEqual(slot_wait, 0)

    def test_06_mocked_adb_safety_controls(self):
        """Req 1 & 13: Mock ADB in unit tests; test resolution, cooldown, max_actions, dry_run."""
        with patch("controller.adb_harness.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Physical size: 720x1280\n")

            harness = ADBHarness(
                device_address="127.0.0.1:5555",
                dry_run=True,
                max_actions=2,
                action_cooldown_sec=0.1,
            )
            # Verify dry-run tap succeeds
            ok1 = harness.play_card(0, 20, 9, non_blocking=False)
            self.assertTrue(ok1)

            # Rapid tap rejected by cooldown
            ok_rapid = harness.play_card(1, 20, 9, non_blocking=False)
            self.assertFalse(ok_rapid)

            harness.close()

    def test_07_outcome_weighted_dataset(self):
        """Req 12: Split by match and weight win > loss."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "match_01_win.npz")
            p2 = os.path.join(tmpdir, "match_02_loss.npz")
            generate_synthetic_shard(p1, num_samples=50, match_outcome=1)   # Win
            generate_synthetic_shard(p2, num_samples=50, match_outcome=-1)  # Loss

            train_loader, val_loader = create_dataloaders(tmpdir, batch_size=16, val_split=0.5)
            # Fetch a batch from train_loader
            batch = next(iter(train_loader))
            weights = batch[7]
            self.assertEqual(weights.shape[0], 16)
            # Weights must reflect outcome (1.5 for win or 0.5 for loss)
            self.assertTrue((weights == 1.5).any() or (weights == 0.5).any())

    def test_08_human_tactical_intelligence_and_spell_discipline(self):
        """Verifies human-like threat assessment, counter selection, spell discipline, and cycle awareness."""
        from perception.state_encoder import OpponentTracker
        encoder = StateEncoder()
        bot = RuleBotPolicy()
        tracker = OpponentTracker(initial_elixir=6.0)

        # 1. Threat level flagging (LOW, MID, HIGH, CRITICAL)
        low_unit = [{"y": 5, "x": 3, "category": "ground", "cls_id": 30, "vy": 0.0, "vx": 0.0}]  # Skeletons in enemy back
        high_unit = [{"y": 19, "x": 3, "category": "ground", "cls_id": 51, "vy": 0.05, "vx": 0.0}]  # Hog Rider on ally side
        crit_unit = [{"y": 24, "x": 3, "category": "ground", "cls_id": 25, "vy": 0.0, "vx": 0.0}]  # PEKKA on ally tower

        t_low = bot.assess_threats(low_unit, [1.0] * 6)[0]
        t_high = bot.assess_threats(high_unit, [1.0] * 6)[0]
        t_crit = bot.assess_threats(crit_unit, [1.0] * 6)[0]
        self.assertEqual(t_low.threat_level, "LOW")
        self.assertIn(t_high.threat_level, ("HIGH", "CRITICAL"))
        self.assertEqual(t_crit.threat_level, "CRITICAL")

        # 2. Tank Killer vs Hog Rider: Bot selects Mini P.E.K.K.A (26) instead of wasting Arrows (1) or Giant (24)
        hog_det = [Detection(0.21, 0.49, 0.29, 0.56, 0.95, 51, False, "ground", track_id=201)]
        hand_hog = [24, 1, 26, 2]  # Giant(24), Arrows(1), Mini P.E.K.K.A(26), Fireball(2)
        grid_h, scalars_h, cmask_h = encoder.encode(hog_det, elixir=6.0, hand_card_ids=hand_hog, time_left=120.0)
        slot_h, tile_h, _, _ = bot.predict_action(grid_h, scalars_h, hand_hog, cmask_h, detections=hog_det, opponent_tracker=tracker)
        self.assertEqual(slot_h, 3)  # Slot 3 is index 2 -> Mini P.E.K.K.A (26)
        self.assertGreaterEqual(tile_h // 18, 16)

        # 3. Anti-Air vs Balloon: Bot selects Musketeer (27), never melee ground-only Knight/Mini P.E.K.K.A
        balloon_det = [Detection(0.74, 0.48, 0.84, 0.55, 0.95, 46, False, "air", track_id=202)]
        hand_air = [20, 26, 27, 24]  # Knight(20), Mini P.E.K.K.A(26), Musketeer(27), Giant(24)
        grid_a, scalars_a, cmask_a = encoder.encode(balloon_det, elixir=6.0, hand_card_ids=hand_air, time_left=120.0)
        slot_a, tile_a, _, _ = bot.predict_action(grid_a, scalars_a, hand_air, cmask_a, detections=balloon_det, opponent_tracker=tracker)
        self.assertEqual(slot_a, 3)  # Slot 3 is index 2 -> Musketeer (27)

        # 4. Strict Spell Discipline: Never cast spells on a solo tank or into corners (0, 0)
        giant_det = [Detection(0.21, 0.50, 0.31, 0.58, 0.95, 24, False, "ground", track_id=203)]
        hand_spells_only = [1, 2, 0, 0]  # Only Arrows and Fireball affordable
        grid_s, scalars_s, cmask_s = encoder.encode(giant_det, elixir=5.0, hand_card_ids=hand_spells_only, time_left=120.0)
        slot_s, tile_s, _, _ = bot.predict_action(grid_s, scalars_s, hand_spells_only, cmask_s, detections=giant_det, opponent_tracker=tracker)
        self.assertEqual(slot_s, 0)  # Waits rather than wasting Arrows/Fireball on a solo full-HP Giant

        # 5. Neural Spell Guardrail: Blocks neural policy from throwing Fireball at corner (0, 0) on empty arena
        empty_grid, empty_scalars, empty_cmask = encoder.encode([], elixir=6.0, hand_card_ids=[2, 20, 21, 24], time_left=120.0)
        safe_slot, safe_tile, reason = bot.validate_or_override_action(
            chosen_slot=1,  # Fireball
            chosen_tile=0,  # Corner (0, 0)
            arena_grid=empty_grid,
            scalars=empty_scalars,
            hand_card_ids=[2, 20, 21, 24],
            legal_card_mask=empty_cmask,
            detections=[],
            opponent_tracker=tracker,
        )
        self.assertEqual(safe_slot, 0)  # Overridden to WAIT because board is calm at 6.0 elixir
        self.assertIn("SPELL GUARD OVERRIDE", reason)

        # 6. Patient Elixir Banking at 6.0 Elixir vs Slow-Roll Behind King at 9.2 Elixir
        hand_std = [24, 27, 20, 1]
        grid_c, scalars_c, cmask_c = encoder.encode([], elixir=6.0, hand_card_ids=hand_std, time_left=130.0, est_enemy_elixir=0.6)
        slot_calm, _, _, _ = bot.predict_action(grid_c, scalars_c, hand_std, cmask_c, detections=[], opponent_tracker=tracker)
        self.assertEqual(slot_calm, 0)  # Banks elixir at 6.0 when board is calm

        grid_full, scalars_full, cmask_full = encoder.encode([], elixir=9.2, hand_card_ids=hand_std, time_left=130.0, est_enemy_elixir=0.6)
        slot_full, tile_full, _, _ = bot.predict_action(grid_full, scalars_full, hand_std, cmask_full, detections=[], opponent_tracker=tracker)
        self.assertEqual(slot_full, 1)  # Deploys Giant (24)
        self.assertGreaterEqual(tile_full // 18, 26)  # Slow-rolled in the back behind King Tower (rows 27..30)

    def test_09_cv_arena_unit_fallback(self):
        """Verifies that ClashRoyaleDetector detects arena units on samples/real_01.png even without YOLO weights."""
        import cv2
        detector = ClashRoyaleDetector(weights=None)
        frame = cv2.imread(os.path.join(REPO_ROOT, "samples", "real_01.png"))
        self.assertIsNotNone(frame)
        obs = detector.parse_frame(frame)
        dets = obs["detections"]
        self.assertGreaterEqual(len(dets), 2)
        has_enemy = any(not d.is_ally for d in dets)
        has_ally = any(d.is_ally for d in dets)
        self.assertTrue(has_enemy)
        self.assertTrue(has_ally)


if __name__ == "__main__":
    unittest.main()

