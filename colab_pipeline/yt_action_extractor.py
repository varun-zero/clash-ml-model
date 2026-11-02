"""
Retroactive Action Reconstruction from YouTube Clash Royale Gameplay VoDs.
Features:
- Sample by CAP_PROP_POS_MSEC (100ms intervals = 10 FPS).
- Canonicalizes perspective with per-video --player_side {top,bottom} (ally always bottom).
- Primary slot signal: `selected` detection, with hand-slot diff fallback.
- Emote / text overlay frame filtering.
- Clock selection filtering: correct side, legal for card type, affordable at t-5, unambiguous.
- Stacks last 4 frames of arena_grid (32 channels) + last_action in (12,) scalars.
- Ratio-controlled Wait sampling (~33% of dataset, skipping within 10 frames after a play).
"""
from typing import Dict, List, Tuple, Optional, Union
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import cv2
import numpy as np
from collections import deque

from configs.constants import (
    ARENA_ROWS,
    ARENA_COLS,
    TOTAL_TILES,
    CARD_ID_TO_COST,
    CARD_ID_TO_IS_SPELL,
    get_default_deployable_mask,
)
from perception.detector import ClashRoyaleDetector
from perception.state_encoder import (
    StateEncoder,
    canonicalize_perspective,
    canonicalize_tile_coordinate,
    stack_spatial_frames,
)
from colab_pipeline.yt_downloader import download_youtube_vod


class YouTubeTrajectoryExtractor:
    """
    Watches a cropped 9:16 Clash Royale video from YouTube at 10 FPS (100ms by POS_MSEC).
    Pairs pre-play state (t - 0.5s) with detected deployment clock at t.
    """
    def __init__(
        self,
        detector: Optional[ClashRoyaleDetector] = None,
        lookback_frames: int = 5,   # 5 frames @ 10 FPS = 500ms
        lookahead_frames: int = 2,  # 2 frames @ 10 FPS = 200ms
        wait_ratio: float = 0.5,    # 1 Wait for every 2 Card plays = 33.3% waits
    ):
        self.detector = detector or ClashRoyaleDetector()
        self.encoder = StateEncoder()
        self.lookback = lookback_frames
        self.lookahead = lookahead_frames
        self.wait_ratio = wait_ratio
        self.window_size = lookback_frames + lookahead_frames + 1

    def calculate_default_viewport_crop(self, full_w: int, full_h: int) -> Tuple[int, int, int, int]:
        """Calculates centered 9:16 aspect ratio phone screen crop inside a 16:9 video."""
        viewport_w = int(round(full_h * (9.0 / 16.0)))
        viewport_x = max(0, (full_w - viewport_w) // 2)
        return viewport_x, 0, viewport_w, full_h

    def process_video(
        self,
        video_source: str,
        player_side: str = "bottom",
        viewport_crop: Optional[Tuple[int, int, int, int]] = None,
        output_npz_path: Optional[str] = None,
        max_duration_seconds: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extracts trajectory from YouTube URL or local .mp4 video path.
        """
        if video_source.startswith("http://") or video_source.startswith("https://"):
            video_path = download_youtube_vod(video_source)
        else:
            video_path = video_source

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")

        full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if viewport_crop is None:
            crop_x, crop_y, crop_w, crop_h = self.calculate_default_viewport_crop(full_w, full_h)
        else:
            crop_x, crop_y, crop_w, crop_h = viewport_crop

        print(f"[Extractor] Processing {video_path} (Side={player_side}, Crop=({crop_x}, {crop_y}, {crop_w}, {crop_h}))")

        buffer = deque(maxlen=self.window_size)
        grid_history = deque(maxlen=4)

        states_grid: List[np.ndarray] = []
        states_scalar: List[np.ndarray] = []
        states_hand: List[np.ndarray] = []
        legal_card_masks: List[np.ndarray] = []
        legal_tile_masks: List[np.ndarray] = []
        actions_card: List[int] = []
        actions_tile: List[int] = []
        timestamps: List[float] = []

        candidate_wait_samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []

        cooldown = 0
        step_idx = 0
        last_play_step = -999
        extracted_actions = 0
        last_action = (0, 0)

        # POS_MSEC based sampling: 100ms = 10 FPS
        target_step_ms = 100.0
        next_target_msec = 0.0

        while cap.isOpened():
            ret, full_frame = cap.read()
            if not ret:
                break

            current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if current_msec < next_target_msec:
                continue
            next_target_msec = current_msec + target_step_ms
            step_idx += 1

            current_time = current_msec / 1000.0
            if max_duration_seconds and current_time > max_duration_seconds:
                break

            # 9:16 phone viewport crop
            game_frame = full_frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            if game_frame.shape[0] < 100 or game_frame.shape[1] < 100:
                continue

            obs = self.detector.parse_frame(
                game_frame,
                time_left=max(0.0, 180.0 - (current_time % 180.0)),
                est_enemy_elixir=0.5,
                last_action=last_action,
            )

            # Canonicalize perspective if player is on top
            if player_side.lower() == "top":
                canon_grid, canon_hps = canonicalize_perspective(
                    obs["arena_grid"], obs["tower_hps"], player_side="top"
                )
                obs["arena_grid"] = canon_grid
                obs["tower_hps"] = canon_hps

            grid_history.append(obs["arena_grid"].copy())
            stacked_grid = stack_spatial_frames(list(grid_history), target_frames=4)
            obs["stacked_grid"] = stacked_grid

            buffer.append((step_idx, current_time, obs))

            if len(buffer) < self.window_size:
                continue

            if cooldown > 0:
                cooldown -= 1
                continue

            past_step, past_time, past_obs = buffer[0]
            curr_step, curr_time, curr_obs = buffer[self.lookback]
            future_step, future_time, future_obs = buffer[-1]

            # Skip frames with emote / text overlay
            if curr_obs.get("has_overlay", False) or past_obs.get("has_overlay", False):
                continue

            # Detect drop event
            drop_event = self._detect_card_drop(past_obs, curr_obs, future_obs, player_side=player_side)

            if drop_event is not None:
                slot_idx, tile_y, tile_x = drop_event
                card_id = past_obs["hand_ids"][slot_idx] if slot_idx < len(past_obs["hand_ids"]) else 0

                states_grid.append(past_obs["stacked_grid"])
                states_scalar.append(past_obs["scalars"])
                states_hand.append(np.array(past_obs["hand_ids"], dtype=np.int64))
                legal_card_masks.append(past_obs["legal_card_mask"])

                tile_mask = self.encoder.get_legal_tile_mask(card_id, tower_hps=past_obs.get("tower_hps"))
                legal_tile_masks.append(tile_mask)

                chosen_card = slot_idx + 1
                chosen_tile = tile_y * ARENA_COLS + tile_x
                actions_card.append(chosen_card)
                actions_tile.append(chosen_tile)
                timestamps.append(past_time)

                last_action = (chosen_card, chosen_tile)
                last_play_step = curr_step
                extracted_actions += 1
                cooldown = 8  # 800ms cooldown
            else:
                # Wait candidate: must be > 10 steps after a card play, no clocks, elixir not full
                if (curr_step - last_play_step > 10) and len(curr_obs["clock_coords"]) == 0 and past_obs["elixir"] < 9.5:
                    candidate_wait_samples.append((
                        past_obs["stacked_grid"],
                        past_obs["scalars"],
                        np.array(past_obs["hand_ids"], dtype=np.int64),
                        past_obs["legal_card_mask"],
                        past_time
                    ))

        cap.release()

        # Ratio-controlled Wait sampling: ~33% of dataset (1 Wait per 2 Plays)
        target_waits = int(round(extracted_actions * self.wait_ratio))
        if target_waits > 0 and len(candidate_wait_samples) > 0:
            sample_indices = np.random.choice(
                len(candidate_wait_samples),
                size=min(target_waits, len(candidate_wait_samples)),
                replace=False
            )
            for idx in sample_indices:
                w_grid, w_scalar, w_hand, w_cmask, w_time = candidate_wait_samples[idx]
                states_grid.append(w_grid)
                states_scalar.append(w_scalar)
                states_hand.append(w_hand)
                legal_card_masks.append(w_cmask)

                w_tile_mask = self.encoder.get_legal_tile_mask(card_id=0)
                legal_tile_masks.append(w_tile_mask)

                actions_card.append(0)  # Wait
                actions_tile.append(0)
                timestamps.append(w_time)

        dataset = {
            "arena_grid": np.array(states_grid, dtype=np.float32),
            "scalars": np.array(states_scalar, dtype=np.float32),
            "hand_ids": np.array(states_hand, dtype=np.int64),
            "legal_card_mask": np.array(legal_card_masks, dtype=bool),
            "legal_tile_mask": np.array(legal_tile_masks, dtype=bool),
            "action_card": np.array(actions_card, dtype=np.int64),
            "action_tile": np.array(actions_tile, dtype=np.int64),
            "timestamps": np.array(timestamps, dtype=np.float32),
            "rewards": np.full(len(actions_card), fill_value=1.0, dtype=np.float32),
            "match_outcome": np.int32(1),
        }

        print(f"[Extractor] Complete: {extracted_actions} actions, {target_waits} waits (Total: {len(actions_card)} samples)")

        if output_npz_path:
            self.save_shard(output_npz_path, dataset)

        return dataset

    def _detect_card_drop(
        self,
        past_obs: dict,
        curr_obs: dict,
        future_obs: dict,
        player_side: str = "bottom",
    ) -> Optional[Tuple[int, int, int]]:
        """
        Clock selection and card slot matching:
        - Primary slot signal: selected detection
        - Fallback: hand-slot diff
        - Filters clocks: correct side, legal for card type, affordable at t-5
        - Discards if 0 or ambiguous (>1 non-clustered) clocks.
        """
        raw_clocks = curr_obs.get("clock_coords", [])
        if not raw_clocks:
            return None

        # 1. Determine slot index
        slot_idx: Optional[int] = curr_obs.get("selected_slot")

        if slot_idx is None:
            # Fallback: check which hand slot changed between past and future
            past_hand = past_obs["hand_ids"]
            future_hand = future_obs["hand_ids"]
            changed_slots = [
                i for i in range(min(4, len(past_hand), len(future_hand)))
                if past_hand[i] != future_hand[i] and past_hand[i] != 0
            ]
            if len(changed_slots) == 1:
                slot_idx = changed_slots[0]
            else:
                # Elixir diff fallback
                elixir_diff = past_obs["elixir"] - curr_obs["elixir"]
                if elixir_diff >= 1.5:
                    best_slot = -1
                    best_cost_err = 999.0
                    for i in range(4):
                        cid = past_hand[i] if i < len(past_hand) else 0
                        if cid > 0:
                            cost = CARD_ID_TO_COST.get(cid, 99)
                            err = abs(cost - elixir_diff)
                            if err < best_cost_err and err <= 1.5:
                                best_cost_err = err
                                best_slot = i
                    if best_slot != -1:
                        slot_idx = best_slot

        if slot_idx is None or slot_idx < 0 or slot_idx > 3:
            return None

        # 2. Check affordability at t-5
        card_id = past_obs["hand_ids"][slot_idx] if slot_idx < len(past_obs["hand_ids"]) else 0
        card_cost = CARD_ID_TO_COST.get(card_id, 99)
        if past_obs["elixir"] < card_cost:
            return None

        is_spell = CARD_ID_TO_IS_SPELL.get(card_id, False)

        # 3. Filter clocks for legality and side
        valid_clocks: List[Tuple[int, int]] = []
        for cy, cx in raw_clocks:
            # Canonicalize clock coordinate
            cy_c, cx_c = canonicalize_tile_coordinate(cy, cx, player_side=player_side)

            # Spells can be cast anywhere (all 576 tiles)
            if is_spell:
                valid_clocks.append((cy_c, cx_c))
            else:
                # Troops must be placed on ally side (rows 16..31) or valid pockets
                enemy_left_down = past_obs["tower_hps"][4] <= 0.05
                enemy_right_down = past_obs["tower_hps"][5] <= 0.05
                is_ally_half = cy_c >= 16
                is_left_pocket = enemy_left_down and (11 <= cy_c < 16) and (0 <= cx_c <= 8)
                is_right_pocket = enemy_right_down and (11 <= cy_c < 16) and (9 <= cx_c <= 17)

                if is_ally_half or is_left_pocket or is_right_pocket:
                    valid_clocks.append((cy_c, cx_c))

        if len(valid_clocks) == 0:
            return None

        # If exactly 1 valid clock, select it
        if len(valid_clocks) == 1:
            return slot_idx, valid_clocks[0][0], valid_clocks[0][1]

        # If multiple valid clocks: check if they are clustered together (within 2 tiles)
        c0_y, c0_x = valid_clocks[0]
        all_clustered = all(
            abs(cy - c0_y) <= 2 and abs(cx - c0_x) <= 2
            for cy, cx in valid_clocks
        )
        if all_clustered:
            avg_y = int(round(np.mean([c[0] for c in valid_clocks])))
            avg_x = int(round(np.mean([c[1] for c in valid_clocks])))
            return slot_idx, avg_y, avg_x

        # Ambiguous multiple distinct clocks -> skip frame
        return None

    @staticmethod
    def save_shard(output_path: str, dataset: Dict[str, np.ndarray]):
        """Saves trajectory dataset as compressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(
            output_path,
            arena_grid=dataset["arena_grid"],
            scalars=dataset["scalars"],
            hand_ids=dataset["hand_ids"],
            legal_card_mask=dataset["legal_card_mask"],
            legal_tile_mask=dataset["legal_tile_mask"],
            action_card=dataset["action_card"],
            action_tile=dataset["action_tile"],
            timestamps=dataset["timestamps"],
            rewards=dataset.get("rewards", np.ones(len(dataset["action_card"]), dtype=np.float32)),
            match_outcome=dataset.get("match_outcome", np.int32(1)),
        )
        print(f"[Extractor] Saved shard to {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Extract CR-Zero Trajectories from YouTube Gameplay VoD")
    parser.add_argument("--video", type=str, required=True, help="YouTube URL or local MP4 path")
    parser.add_argument("--player_side", type=str, default="bottom", choices=["top", "bottom"], help="Player camera side")
    parser.add_argument("--output", type=str, default="./data/trajectories/shard_001.npz", help="Output .npz path")
    parser.add_argument("--max_seconds", type=float, default=None, help="Max duration to extract")
    args = parser.parse_args()

    extractor = YouTubeTrajectoryExtractor()
    extractor.process_video(
        video_source=args.video,
        player_side=args.player_side,
        output_npz_path=args.output,
        max_duration_seconds=args.max_seconds,
    )


# Alias for backward compatibility
YouTubeActionExtractor = YouTubeTrajectoryExtractor


if __name__ == "__main__":
    main()
