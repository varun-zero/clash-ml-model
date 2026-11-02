"""
Retroactive Action Reconstruction from YouTube Clash Royale Gameplay VoDs.
Samples frames at 10 FPS, pairs pre-play state (t - 0.5s) with detected deployment clock (t),
balances with 1 Wait frame per 2 Action frames, and exports compressed .npz shards.
"""
from typing import Dict, List, Tuple, Optional, Union
import os
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
from perception.state_encoder import StateEncoder
from colab_pipeline.yt_downloader import download_youtube_vod


class YouTubeTrajectoryExtractor:
    """
    Watches a cropped 9:16 Clash Royale video from YouTube at 10 FPS.
    Pairs (State at t - 0.5s) with (Action detected at t via deployment clock).
    """
    def __init__(
        self,
        detector: Optional[ClashRoyaleDetector] = None,
        lookback_frames: int = 5,   # 5 frames @ 10 FPS = 500ms
        lookahead_frames: int = 2,  # 2 frames @ 10 FPS = 200ms
        wait_ratio: float = 0.5,    # 1 Wait sample for every 2 Action samples
    ):
        self.detector = detector or ClashRoyaleDetector()
        self.encoder = StateEncoder()
        self.lookback = lookback_frames
        self.lookahead = lookahead_frames
        self.wait_ratio = wait_ratio
        self.window_size = lookback_frames + lookahead_frames + 1

    def calculate_default_viewport_crop(self, full_w: int, full_h: int) -> Tuple[int, int, int, int]:
        """
        Calculates centered 9:16 aspect ratio phone screen crop inside a 16:9 video.
        """
        viewport_w = int(round(full_h * (9.0 / 16.0)))
        viewport_x = max(0, (full_w - viewport_w) // 2)
        return viewport_x, 0, viewport_w, full_h

    def process_video(
        self,
        video_source: str,
        viewport_crop: Optional[Tuple[int, int, int, int]] = None,
        output_npz_path: Optional[str] = None,
        max_duration_seconds: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extracts trajectory from YouTube URL or local .mp4 video path.
        Args:
            video_source: Local filepath or https:// YouTube URL
            viewport_crop: (x, y, w, h) pixel crop bounding box for the phone screen
            output_npz_path: Optional destination to save compressed .npz shard
            max_duration_seconds: Optional ceiling on seconds processed
        """
        # Download if a URL is provided
        if video_source.startswith("http://") or video_source.startswith("https://"):
            video_path = download_youtube_vod(video_source)
        else:
            video_path = video_source

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Crop calculation
        if viewport_crop is None:
            crop_x, crop_y, crop_w, crop_h = self.calculate_default_viewport_crop(full_w, full_h)
        else:
            crop_x, crop_y, crop_w, crop_h = viewport_crop

        # 10 FPS step
        frame_step = max(1, int(round(video_fps / 10.0)))
        print(f"[Extractor] Processing {video_path}: FPS={video_fps:.1f}, Step={frame_step}, Crop=({crop_x}, {crop_y}, {crop_w}, {crop_h})")

        # Sliding window buffer: stores (frame_idx, obs_dict)
        buffer = deque(maxlen=self.window_size)

        # Datasets
        states_grid: List[np.ndarray] = []
        states_scalar: List[np.ndarray] = []
        states_hand: List[np.ndarray] = []
        legal_card_masks: List[np.ndarray] = []
        legal_tile_masks: List[np.ndarray] = []
        actions_card: List[int] = []
        actions_tile: List[int] = []
        timestamps: List[float] = []

        candidate_wait_samples: List[Tuple[dict, float]] = []

        cooldown = 0
        frame_idx = 0
        extracted_actions = 0

        while cap.isOpened():
            ret, full_frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % frame_step != 0:
                continue

            current_time = frame_idx / video_fps
            if max_duration_seconds and current_time > max_duration_seconds:
                break

            # Crop phone viewport (9:16)
            game_frame = full_frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            if game_frame.shape[0] < 100 or game_frame.shape[1] < 100:
                continue

            # Perception parse
            obs = self.detector.parse_frame(game_frame, time_left=max(0.0, 180.0 - (current_time % 180.0)))
            buffer.append((frame_idx, current_time, obs))

            # Need full window (lookback + candidate_t + lookahead)
            if len(buffer) < self.window_size:
                continue

            if cooldown > 0:
                cooldown -= 1
                continue

            # Candidate trigger frame `t` is at index: self.lookback
            # Past state (t - 0.5s) is at index: 0
            # Future state (t + 0.2s) is at index: -1
            past_idx, past_time, past_obs = buffer[0]
            curr_idx, curr_time, curr_obs = buffer[self.lookback]
            future_idx, future_time, future_obs = buffer[-1]

            # Detect deployment clock at frame `t` and card drop between (t - 5) and (t + 2)
            drop_event = self._detect_card_drop(past_obs, curr_obs, future_obs)

            if drop_event is not None:
                slot_idx, tile_y, tile_x = drop_event
                card_id = past_obs["hand_ids"][slot_idx] if slot_idx < len(past_obs["hand_ids"]) else 0

                # State at t - 0.5s (before card left hand)
                states_grid.append(past_obs["arena_grid"])
                states_scalar.append(past_obs["scalars"])
                states_hand.append(np.array(past_obs["hand_ids"], dtype=np.int64))
                legal_card_masks.append(past_obs["legal_card_mask"])

                # Legal tile mask for this card
                tile_mask = self.encoder.get_legal_tile_mask(card_id)
                legal_tile_masks.append(tile_mask)

                # Action: slot_idx + 1 (1..4)
                actions_card.append(slot_idx + 1)
                actions_tile.append(tile_y * ARENA_COLS + tile_x)
                timestamps.append(past_time)

                extracted_actions += 1
                cooldown = 8  # 800ms cooldown so the same clock icon isn't recorded repeatedly
            else:
                # Potential candidate Wait frame (no clocks detected, elixir not full)
                if len(curr_obs["clock_coords"]) == 0 and past_obs["elixir"] < 9.5:
                    candidate_wait_samples.append((past_obs, past_time))

        cap.release()

        # ----------------------------------------------------------------------
        # Balance dataset: Sample 1 Wait frame for every 2 Action frames
        # ----------------------------------------------------------------------
        target_waits = int(round(extracted_actions * self.wait_ratio))
        if target_waits > 0 and len(candidate_wait_samples) > 0:
            sample_indices = np.random.choice(
                len(candidate_wait_samples),
                size=min(target_waits, len(candidate_wait_samples)),
                replace=False
            )
            for idx in sample_indices:
                w_obs, w_time = candidate_wait_samples[idx]
                states_grid.append(w_obs["arena_grid"])
                states_scalar.append(w_obs["scalars"])
                states_hand.append(np.array(w_obs["hand_ids"], dtype=np.int64))
                legal_card_masks.append(w_obs["legal_card_mask"])

                # Wait tile mask: tile 0 valid
                w_tile_mask = self.encoder.get_legal_tile_mask(card_id=0)
                legal_tile_masks.append(w_tile_mask)

                actions_card.append(0)  # 0 = Wait
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
        }

        print(f"[Extractor] Finished {video_path}: {extracted_actions} actions, {target_waits} waits (Total {len(actions_card)} samples)")

        if output_npz_path:
            self.save_shard(output_npz_path, dataset)

        return dataset

    def _detect_card_drop(
        self,
        past_obs: dict,
        curr_obs: dict,
        future_obs: dict
    ) -> Optional[Tuple[int, int, int]]:
        """
        Matches a newly appeared `clock` detection in `curr_obs` with a card slot
        that changed between `past_obs` (t - 0.5s) and `future_obs` (t + 0.2s).
        Returns (slot_idx, tile_y, tile_x) or None.
        """
        if not curr_obs.get("clock_coords"):
            return None

        # Check which card slot changed between past and future
        past_hand = past_obs["hand_ids"]
        future_hand = future_obs["hand_ids"]

        changed_slots = [
            i for i in range(min(4, len(past_hand), len(future_hand)))
            if past_hand[i] != future_hand[i] and past_hand[i] != 0
        ]

        # If hand card classification was noisy or multiple slots changed, check elixir drop
        if len(changed_slots) != 1:
            # Fallback: find slot where card cost matches elixir drop
            elixir_diff = past_obs["elixir"] - curr_obs["elixir"]
            if elixir_diff >= 1.5:
                # Find slot with cost closest to elixir diff
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
                    changed_slots = [best_slot]

        if len(changed_slots) != 1:
            return None

        slot_idx = changed_slots[0]
        tile_y, tile_x = curr_obs["clock_coords"][0]
        return slot_idx, tile_y, tile_x

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
        )
        print(f"[Extractor] Saved shard to {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")

