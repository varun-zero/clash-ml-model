"""
Unit and regression tests for Stage 2 AWR and OpponentTracker enhancements.
"""
import glob
import os
import sys
import unittest
import numpy as np
import torch

from perception.state_encoder import Detection, OpponentTracker, StateEncoder
from colab_pipeline.train_awr import AWRTrajectoryDataset, train_awr_epoch
from policy.model import DeckConditionedPolicy
from configs.observation_schema import SCHEMA_VERSION


def get_or_create_test_shard() -> str:
    """Finds an existing valid schema_version=2 shard, or creates a synthetic fixture for CI."""
    for shard in sorted(glob.glob("data/trajectories/*.npz")):
        try:
            with np.load(shard) as d:
                if (
                    int(d.get("schema_version", -1)) == SCHEMA_VERSION
                    and len(d.get("action_card", [])) >= 10
                    and "rewards" in d
                ):
                    return shard
        except Exception:
            continue

    os.makedirs("data/trajectories", exist_ok=True)
    fixture_path = "data/trajectories/fixture_test_shard.npz"
    N = 25
    grid = np.zeros((N, 32, 32, 18), dtype=np.float32)
    scalars = np.zeros((N, 12), dtype=np.float32)
    scalars[:, 0] = 0.7  # ally elixir
    scalars[:, 1] = 0.5  # enemy elixir
    scalars[:, 2] = 0.8  # time
    scalars[:, 3:6] = 1.0  # ally towers
    scalars[:, 6:9] = 1.0  # enemy towers
    hand = np.tile(np.array([1, 2, 20, 24], dtype=np.int64), (N, 1))
    cmask = np.ones((N, 5), dtype=bool)
    tmask = np.ones((N, 576), dtype=bool)
    acard = np.ones(N, dtype=np.int64)
    atile = np.full(N, 100, dtype=np.int64)
    rewards = np.ones(N, dtype=np.float32) * 1.5
    timestamps = np.linspace(0, 10, N, dtype=np.float32)

    np.savez_compressed(
        fixture_path,
        arena_grid=grid,
        scalars=scalars,
        hand_ids=hand,
        legal_card_mask=cmask,
        legal_tile_mask=tmask,
        action_card=acard,
        action_tile=atile,
        rewards=rewards,
        match_outcome=np.int32(1),
        timestamps=timestamps,
        schema_version=np.int32(SCHEMA_VERSION),
    )
    return fixture_path


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
        test_shard = get_or_create_test_shard()
        dataset = AWRTrajectoryDataset([test_shard], beta=0.5, augment_horizontal_flip=False)
        self.assertGreater(len(dataset), 0)
        sample = dataset[0]
        # (grid, scalar, hand, cmask, tmask, acard, atile, weight)
        self.assertEqual(len(sample), 8)
        weight = float(sample[7])
        self.assertGreaterEqual(weight, 0.01)
        self.assertLessEqual(weight, 15.0)

    def test_awr_training_step(self):
        """Tests a single training step of AWR with mixed precision."""
        device = torch.device("cpu")
        model = DeckConditionedPolicy().to(device)
        test_shard = get_or_create_test_shard()

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

    def test_awr_rejects_vanguard_data(self):
        """P0 Guard: AWR must strictly refuse Vanguard pretraining shards lacking real board/rewards."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            p_vanguard = os.path.join(tmpdir, "vanguard_sample.npz")
            N = 20
            np.savez_compressed(
                p_vanguard,
                arena_grid=np.zeros((N, 32, 32, 18), dtype=np.float32),
                scalars=np.zeros((N, 12), dtype=np.float32),
                hand_ids=np.ones((N, 4), dtype=np.int64),
                legal_card_mask=np.ones((N, 5), dtype=bool),
                legal_tile_mask=np.ones((N, 576), dtype=bool),
                action_card=np.ones(N, dtype=np.int64),
                action_tile=np.zeros(N, dtype=np.int64),
                timestamps=np.linspace(0, 10, N, dtype=np.float32),
                match_outcome=np.int32(1),
                schema_version=np.int32(SCHEMA_VERSION),
                dataset_type=np.str_("pretraining_vanguard"),
                has_ground_truth_board=np.bool_(False),
            )
            with self.assertRaises(ValueError) as ctx:
                AWRTrajectoryDataset([p_vanguard], beta=0.5)
            self.assertIn("Ineligible dataset source for AWR", str(ctx.exception))

    def test_awr_strict_timestamp_validation(self):
        """P1 Guard: AWR must reject missing, non-monotonic, or NaN timestamps when computing advantages."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            N = 20
            base_kwargs = dict(
                arena_grid=np.zeros((N, 32, 32, 18), dtype=np.float32),
                scalars=np.ones((N, 12), dtype=np.float32),
                hand_ids=np.ones((N, 4), dtype=np.int64),
                legal_card_mask=np.ones((N, 5), dtype=bool),
                legal_tile_mask=np.ones((N, 576), dtype=bool),
                action_card=np.ones(N, dtype=np.int64),
                action_tile=np.zeros(N, dtype=np.int64),
                match_outcome=np.int32(1),
                schema_version=np.int32(SCHEMA_VERSION),
            )

            # 1. Missing timestamps without rewards
            p_missing_ts = os.path.join(tmpdir, "missing_ts.npz")
            np.savez_compressed(p_missing_ts, **base_kwargs)
            with self.assertRaises(ValueError) as ctx1:
                AWRTrajectoryDataset([p_missing_ts], beta=0.5)
            self.assertIn("Missing valid 'timestamps' array", str(ctx1.exception))

            # 2. Non-monotonic timestamps
            p_non_mono = os.path.join(tmpdir, "non_monotonic.npz")
            bad_ts = np.array([0.0, 1.0, 0.5] + [2.0] * (N - 3), dtype=np.float32)
            np.savez_compressed(p_non_mono, timestamps=bad_ts, **base_kwargs)
            with self.assertRaises(ValueError) as ctx2:
                AWRTrajectoryDataset([p_non_mono], beta=0.5)
            self.assertIn("strictly non-decreasing", str(ctx2.exception))

            # 3. NaN timestamps
            p_nan = os.path.join(tmpdir, "nan_ts.npz")
            nan_ts = np.linspace(0, 10, N, dtype=np.float32)
            nan_ts[5] = np.nan
            np.savez_compressed(p_nan, timestamps=nan_ts, **base_kwargs)
            with self.assertRaises(ValueError) as ctx3:
                AWRTrajectoryDataset([p_nan], beta=0.5)
            self.assertIn("non-finite", str(ctx3.exception))

    def test_adb_action_lifecycle_and_acknowledgement(self):
        """P0 Guard: Tests asynchronous ADB action lifecycle, IDs, and tap acknowledgement."""
        from controller.adb_harness import ADBHarness
        harness = ADBHarness(dry_run=True, action_cooldown_sec=0.01)

        # 1. Dispatch non-blocking play
        ok = harness.play_card(0, 18, 5, non_blocking=True)
        self.assertTrue(ok)
        aid = harness.get_last_action_id()
        self.assertGreater(aid, 0)

        # 2. Wait for background thread completion
        completed = harness.wait_for_action(aid, timeout=0.5)
        self.assertTrue(completed)
        self.assertTrue(harness.is_action_completed(aid))

        # 3. Drain acknowledgement queue
        acks = harness.get_completed_actions()
        self.assertGreaterEqual(len(acks), 1)
        self.assertEqual(acks[0]["action_id"], aid)
        self.assertTrue(acks[0]["success"])

        harness.close()


if __name__ == "__main__":
    unittest.main()
