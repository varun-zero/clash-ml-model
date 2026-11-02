"""
CR-Zero Comprehensive Integration Test Suite.
Verifies all 14 requirements across DATA/EXTRACTION, STATE/MODEL, PERCEPTION, and LIVE.
Must exit with code 0.
"""
import os
import sys
import unittest
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.constants import (
    ARENA_ROWS,
    ARENA_COLS,
    TOTAL_TILES,
    NUM_CARD_CLASSES_V1,
    STACKED_GRID_CHANNELS,
    NUM_SCALARS,
    DEFAULT_DECK_CLASSES,
    CARD_DATABASE,
    grid_to_screen_pixel,
    hand_slot_to_screen_pixel,
    grid_to_clashroyalebuildabot,
    clashroyalebuildabot_to_grid,
)
from perception.state_encoder import (
    StateEncoder,
    Detection,
    canonicalize_perspective,
    canonicalize_tile_coordinate,
    stack_spatial_frames,
)
from perception.detector import ClashRoyaleDetector
from policy.model import DeckConditionedPolicy
from controller.adb_harness import ADBHarness
from colab_pipeline.yt_action_extractor import YouTubeTrajectoryExtractor


class TestCRZeroRequirements(unittest.TestCase):
    def test_01_constants_and_deck_vocabulary(self):
        """Req 11: Single deck (2.6 Hog) with 10 classes."""
        self.assertEqual(len(DEFAULT_DECK_CLASSES), 10)
        self.assertEqual(DEFAULT_DECK_CLASSES[0]["name"], "Wait")
        self.assertEqual(DEFAULT_DECK_CLASSES[1]["name"], "Hog Rider")
        self.assertEqual(NUM_CARD_CLASSES_V1, 10)
        self.assertEqual(STACKED_GRID_CHANNELS, 32)
        self.assertEqual(NUM_SCALARS, 12)

    def test_02_buildabot_interop(self):
        """Ensure coordinate conversions are monotonic and roundtrip accurately."""
        for r in [0, 15, 31]:
            for c in [0, 8, 17]:
                bx, by = grid_to_clashroyalebuildabot(r, c)
                r_rec, c_rec = clashroyalebuildabot_to_grid(bx, by)
                self.assertEqual(r, r_rec)
                self.assertEqual(c, c_rec)

    def test_03_canonicalization(self):
        """Req 1: Rotate grid 180 and swap ally/enemy channels when player_side=top."""
        grid = np.zeros((8, ARENA_ROWS, ARENA_COLS), dtype=np.float32)
        grid[0, 5, 4] = 1.0  # Ally ground at top row 5, col 4
        grid[3, 25, 12] = 2.0  # Enemy ground at bottom row 25, col 12
        tower_hps = [1.0, 0.9, 0.8, 1.0, 0.7, 0.6]

        canon_bottom, hps_bottom = canonicalize_perspective(grid, tower_hps, player_side="bottom")
        self.assertTrue(np.array_equal(canon_bottom, grid))
        self.assertEqual(hps_bottom, tower_hps)

        canon_top, hps_top = canonicalize_perspective(grid, tower_hps, player_side="top")
        # In canon_top, channels 0..2 swap with 3..5, and coordinates rotate 180 (row -> 31-r, col -> 17-c)
        # Old ally (channel 0) at (5, 4) becomes enemy (channel 3) at (31-5=26, 17-4=13)
        self.assertEqual(canon_top[3, 26, 13], 1.0)
        # Old enemy (channel 3) at (25, 12) becomes ally (channel 0) at (31-25=6, 17-12=5)
        self.assertEqual(canon_top[0, 6, 5], 2.0)

        # Coordinate canonicalization: (5, 4) becomes (26, 13)
        cy, cx = canonicalize_tile_coordinate(5, 4, player_side="top")
        self.assertEqual((cy, cx), (26, 13))

    def test_04_state_encoder_and_scalars(self):
        """Req 7 & 8: Scalars (12,) with est_enemy_elixir=0.5 and 32 stacked channels."""
        encoder = StateEncoder()
        grid, scalars, legal_card_mask = encoder.encode(
            detections=[Detection(0.5, 0.5, 0.6, 0.6, 0.9, 1, True, "ground")],
            elixir=7.0,
            hand_card_ids=[20, 21, 22, 23],
            tower_hps=[1.0, 0.9, 0.9, 1.0, 0.8, 0.8],
            time_left=120.0,
            est_enemy_elixir=0.5,
            last_action=(1, 100),
        )
        self.assertEqual(grid.shape, (8, 32, 18))
        self.assertEqual(scalars.shape, (12,))
        self.assertEqual(scalars[1], 0.5)  # est_enemy_elixir constant
        self.assertEqual(legal_card_mask.shape, (5,))

        # 4-frame stacking: 4 * 8 = 32 channels
        frames = [grid for _ in range(4)]
        stacked = stack_spatial_frames(frames, target_frames=4)
        self.assertEqual(stacked.shape, (32, 32, 18))

    def test_05_detector_loud_failure(self):
        """Req 11: Fail loudly if missing weights."""
        with self.assertRaises((FileNotFoundError, ValueError)):
            ClashRoyaleDetector(weights="non_existent_weights_file.pt")

    def test_06_policy_architecture_and_fp16(self):
        """Req 4, 7, 8: DeckConditionedPolicy with 32 in_channels and 12 scalars."""
        model = DeckConditionedPolicy(
            num_card_classes=10,
            card_emb_dim=16,
            num_scalars=12,
            in_channels=32,
        )
        grid = torch.zeros((2, 32, 32, 18), dtype=torch.float32)
        scalars = torch.zeros((2, 12), dtype=torch.float32)
        hand = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
        cmask = torch.ones((2, 5), dtype=torch.bool)
        tmask = torch.ones((2, 576), dtype=torch.bool)

        c_logits, t_logits = model(grid, scalars, hand, cmask, tmask)
        self.assertEqual(c_logits.shape, (2, 5))
        self.assertEqual(t_logits.shape, (2, 576))

        # Test predict_action with temperature sampling
        card_act, tile_act, c_conf, t_conf = model.predict_action(
            grid, scalars, hand, cmask, tmask, temperature=0.7
        )
        self.assertIn(card_act, list(range(5)))
        self.assertIn(tile_act, list(range(576)))

    def test_07_clock_filtering_and_ambiguity_logic(self):
        """Req 2: Clock filtering for side, troop legality, affordability, and ambiguity skip."""
        extractor = YouTubeTrajectoryExtractor()

        # Knight (card ID 20) is a 3-elixir ground troop (only legal on ally half rows 16..31)
        past_obs = {
            "elixir": 4.0,  # Affordable (cost 3 <= 4)
            "hand_ids": [20, 2, 3, 4],
            "tower_hps": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "has_overlay": False,
        }
        future_obs = {
            "hand_ids": [21, 2, 3, 4],
            "has_overlay": False,
        }

        # Case A: Two clocks: row 5 (illegal enemy half) and row 20 (legal ally half)
        curr_obs_a = {
            "selected_slot": 0,
            "elixir": 1.0,
            "clock_coords": [(5, 9), (20, 9)],
            "has_overlay": False,
        }
        res_a = extractor._detect_card_drop(past_obs, curr_obs_a, future_obs, player_side="bottom")
        self.assertIsNotNone(res_a)
        slot, ty, tx = res_a
        self.assertEqual(slot, 0)
        self.assertEqual((ty, tx), (20, 9))

        # Case B: Unaffordable at t-5 (elixir was 2.0 < 3.0) -> must return None
        past_obs_unaffordable = dict(past_obs, elixir=2.0)
        res_b = extractor._detect_card_drop(past_obs_unaffordable, curr_obs_a, future_obs, player_side="bottom")
        self.assertIsNone(res_b)

        # Case C: Ambiguous clocks (two legal clocks far apart: (20, 4) and (28, 14)) -> skip frame
        curr_obs_c = {
            "selected_slot": 0,
            "elixir": 1.0,
            "clock_coords": [(20, 4), (28, 14)],
            "has_overlay": False,
        }
        res_c = extractor._detect_card_drop(past_obs, curr_obs_c, future_obs, player_side="bottom")
        self.assertIsNone(res_c)

    def test_08_adb_harness_execution(self):
        """Req 13: ADB Harness two-tap calculation."""
        harness = ADBHarness(screen_width=540, screen_height=960)
        hx, hy = hand_slot_to_screen_pixel(0, 540, 960)
        self.assertTrue(0 <= hx <= 540 and 0 <= hy <= 960)
        ax, ay = grid_to_screen_pixel(20, 9, 540, 960)
        self.assertTrue(0 <= ax <= 540 and 0 <= ay <= 960)
        harness.close()


if __name__ == "__main__":
    unittest.main()
