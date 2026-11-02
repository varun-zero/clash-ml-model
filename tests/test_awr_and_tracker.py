"""
Unit and regression tests for Stage 2 AWR and OpponentTracker enhancements.
"""
import os
import sys
import unittest
import numpy as np
import torch

from perception.state_encoder import Detection, OpponentTracker, StateEncoder
from colab_pipeline.train_awr import AWRTrajectoryDataset, train_awr_epoch
from policy.model import DeckConditionedPolicy


class TestAWRAndTracker(unittest.TestCase):
    def test_opponent_tracker_elixir_and_cycle(self):
        """Tests OpponentTracker passive elixir accumulation, card cost deduction, and deck logging."""
        tracker = OpponentTracker(initial_elixir=5.0)
        self.assertAlmostEqual(tracker.get_elixir(), 5.0, places=1)

        # 1. Passive regeneration after 2.8 seconds -> +1.0 elixir (total 6.0)
        t0 = tracker.last_update_time
        enemy_det = [
            Detection(x1=0.2, y1=0.2, x2=0.3, y2=0.3, confidence=0.9, cls_id=24, is_ally=False, track_id=101)  # Giant (cost 5)
        ]
        elixir = tracker.update(current_time=t0 + 2.8, enemy_detections=enemy_det, double_elixir=False)
        # 5.0 + 1.0 (passive) - 5.0 (giant cost) = 1.0 elixir
        self.assertAlmostEqual(elixir, 1.0, places=1)
        self.assertIn(24, tracker.seen_cards)
        self.assertEqual(tracker.total_elixir_spent, 5)

        # 2. Existing track ID should not deduct cost again
        elixir2 = tracker.update(current_time=t0 + 5.6, enemy_detections=enemy_det, double_elixir=False)
        # 1.0 + 1.0 (passive) - 0 = 2.0 elixir
        self.assertAlmostEqual(elixir2, 2.0, places=1)

        # 3. State encoder integration
        encoder = StateEncoder()
        grid, scalars, cmask = encoder.encode(
            detections=enemy_det,
            elixir=7.0,
            hand_card_ids=[1, 2, 20, 21],
            opponent_tracker=tracker,
        )
        # scalars[0] is ally elixir norm (0.7), scalars[1] is enemy elixir norm (0.2)
        self.assertAlmostEqual(scalars[0], 0.7, places=2)
        self.assertAlmostEqual(scalars[1], 0.2, places=2)

    def test_opponent_cycle_and_in_hand(self):
        """Tests that cards accurately cycle back to hand after 4 intermediate plays."""
        tracker = OpponentTracker(initial_elixir=10.0)
        t0 = 1000.0

        # Plays: Barrel(12), Princess(49), Knight(20), Gang(64), Rocket(6)
        tracker.update(t0 + 0, [Detection(0.5, 0.2, 0.6, 0.3, 0.9, 12, False, track_id=1)])
        tracker.update(t0 + 1, [Detection(0.5, 0.2, 0.6, 0.3, 0.9, 49, False, track_id=2)])
        tracker.update(t0 + 2, [Detection(0.5, 0.2, 0.6, 0.3, 0.9, 20, False, track_id=3)])
        tracker.update(t0 + 3, [Detection(0.5, 0.2, 0.6, 0.3, 0.9, 64, False, track_id=4)])

        # Before 5th play: Barrel is 1 play away, not in hand
        self.assertFalse(tracker.is_card_in_hand(12))
        self.assertEqual(tracker.get_card_cycle_distance(12), 1)

        # 5th play: Rocket (6) -> Barrel cycles back into hand!
        tracker.update(t0 + 4, [Detection(0.5, 0.2, 0.6, 0.3, 0.9, 6, False, track_id=5)])
        self.assertTrue(tracker.is_card_in_hand(12))
        self.assertEqual(tracker.get_card_cycle_distance(12), 0)
        self.assertEqual(tracker.get_card_cycle_distance(49), 1)
        self.assertEqual(tracker.get_card_cycle_distance(6), 4)

    def test_meta_deck_identification_and_prediction(self):
        """Tests meta archetype recognition and unrevealed card predictions."""
        from configs.meta_decks import identify_meta_deck, _cid

        # 1. Log Bait detection
        log_bait_seen = [_cid("Goblin Barrel"), _cid("Princess"), _cid("Knight")]
        res = identify_meta_deck(log_bait_seen)
        self.assertEqual(res["matched_deck_name"], "Classic Log Bait")
        self.assertEqual(res["archetype"], "Spell Bait")
        self.assertGreaterEqual(res["confidence"], 0.8)
        self.assertIn("Rocket", res["predicted_unrevealed_names"])
        self.assertIn("The Log", res["predicted_unrevealed_names"])

        # 2. 2.6 Hog detection
        hog_seen = [_cid("Hog Rider"), _cid("Musketeer"), _cid("Cannon")]
        res_hog = identify_meta_deck(hog_seen)
        self.assertEqual(res_hog["matched_deck_name"], "2.6 Hog Cycle")
        self.assertEqual(res_hog["archetype"], "Fast Cycle")
        self.assertGreaterEqual(res_hog["confidence"], 0.8)

    def test_awr_dataset_weighting(self):
        """Tests that AWRTrajectoryDataset correctly computes advantage weights."""
        test_shard = "data/trajectories/match_shard_001_win.npz"
        if not os.path.exists(test_shard):
            self.skipTest(f"Shard {test_shard} not found")

        dataset = AWRTrajectoryDataset([test_shard], beta=0.5, augment_horizontal_flip=False)
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        # (grid, scalar, hand, cmask, tmask, acard, atile, weight)
        self.assertEqual(len(sample), 8)
        weight = float(sample[7])
        self.assertGreaterEqual(weight, 0.05)
        self.assertLessEqual(weight, 15.0)

    def test_awr_training_step(self):
        """Tests a single training step of AWR with mixed precision."""
        device = torch.device("cpu")
        model = DeckConditionedPolicy().to(device)
        test_shard = "data/trajectories/match_shard_001_win.npz"
        if not os.path.exists(test_shard):
            self.skipTest(f"Shard {test_shard} not found")

        dataset = AWRTrajectoryDataset([test_shard], beta=0.5, augment_horizontal_flip=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(enabled=False)

        metrics = train_awr_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            tile_loss_weight=1.0,
        )
        self.assertIn("loss", metrics)
        self.assertIn("card_acc", metrics)
        self.assertGreater(metrics["loss"], 0.0)


if __name__ == "__main__":
    unittest.main()

