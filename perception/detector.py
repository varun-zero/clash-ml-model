"""
YOLO and Vision Perception Pipeline for Clash Royale.
Supports Ultralytics YOLO (PyTorch .pt), ONNX Runtime FP16, and Template Vision.
Features:
- Single global card vocabulary with unlocked-cards template matching for hand slots.
- Logs unknown hand crops to unknown_cards/ for active expansion.
- Accurate HP bar ratio calculation (returns 0.0 for destroyed towers).
- Match timer reading and double-elixir / overtime derivation.
- Screen-state classification: 'battle', 'menu', 'result'.
- Match outcome and crown reader for game-over screens.
"""
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
import glob
import time
import cv2
import numpy as np

from configs.constants import (
    ARENA_BOX,
    HAND_SLOTS,
    ELIXIR_BAR_BOX,
    ELIXIR_TEXT_BOX,
    TIMER_BOX,
    TOWER_BOXES,
    TOWER_HP_BOXES,
    CARD_DATABASE,
    CARD_ID_TO_COST,
    CARD_ID_TO_IS_SPELL,
    CARD_ID_TO_NAME,
    CARD_NAME_TO_ID,
    DEFAULT_UNLOCKED_CARD_IDS,
    screen_to_grid,
)
from configs.unlocked_cards import load_unlocked_cards
from perception.state_encoder import StateEncoder, Detection, OpponentTracker


class ClashRoyaleDetector:
    """
    Perception engine wrapping YOLO detection, UI cropping, and board parsing.
    Restricted to the account's unlocked card vocabulary.
    """
    def __init__(
        self,
        weights: Optional[str] = None,
        model_path: Optional[str] = None,  # Backward compatible alias
        conf_threshold: float = 0.35,
        device: str = "cuda:0",
        use_fp16: bool = True,
        unlocked_cards: Optional[List[int]] = None,
        templates_dir: str = "./templates",
        require_weights: bool = False,
    ):
        self.conf_threshold = conf_threshold
        self.device = device
        self.use_fp16 = use_fp16
        self.encoder = StateEncoder()
        self.model = None
        self.onnx_session = None

        # 1. Unlocked Cards Vocabulary
        self.unlocked_cards = unlocked_cards or load_unlocked_cards()
        self.templates_dir = templates_dir
        self.card_templates: Dict[int, List[np.ndarray]] = self._load_templates()

        # Opponent Elixir and Card-Cycle Tracker + Track Velocity History
        self.opponent_tracker = OpponentTracker()
        self.track_history: Dict[int, Tuple[float, float, float]] = {}
        self._prev_screen_state = "menu"
        self._battle_t0: Optional[float] = None

        # 2. YOLO Model Loading
        weights_file = weights or model_path
        if weights_file:
            if not os.path.exists(weights_file):
                raise FileNotFoundError(f"[Detector] Specified --weights file not found: {weights_file}")
            self._load_model(weights_file)
        elif require_weights:
            raise ValueError("[Detector] --weights path is required and cannot be empty.")

        # Ensure directory for logging unknown hand cards exists
        self.unknown_cards_dir = "./unknown_cards"
        os.makedirs(self.unknown_cards_dir, exist_ok=True)

    def _load_templates(self) -> Dict[int, List[np.ndarray]]:
        """Loads reference template images for unlocked global card IDs."""
        templates: Dict[int, List[np.ndarray]] = {}
        if not os.path.exists(self.templates_dir):
            return templates

        for cid in self.unlocked_cards:
            cdir = os.path.join(self.templates_dir, str(cid))
            if os.path.exists(cdir):
                img_paths = glob.glob(os.path.join(cdir, "*.png"))
                loaded = []
                for p in img_paths:
                    tmpl = cv2.imread(p)
                    if tmpl is not None:
                        # Resize to standard template size
                        std = cv2.resize(tmpl, (64, 64), interpolation=cv2.INTER_AREA)
                        loaded.append(std)
                if loaded:
                    templates[cid] = loaded
        return templates

    def _load_model(self, model_path: str):
        """Loads Ultralytics YOLO or ONNX model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[Detector] Model file not found: {model_path}")

        if model_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.onnx_session = ort.InferenceSession(model_path, sess_opts, providers=providers)
                print(f"[Detector] Loaded ONNX detector: {model_path}")
            except Exception as e:
                print(f"[Detector] Warning: Could not initialize ONNX session ({e}).")
        else:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"[Detector] Loaded Ultralytics YOLO detector: {model_path}")
            except Exception as e:
                raise RuntimeError(f"[Detector] Failed to load Ultralytics YOLO from {model_path}: {e}")

    def reset_match_state(self):
        """Resets match-specific tracker state and velocity history."""
        self.opponent_tracker.reset()
        self.track_history.clear()

    def detect_raw(self, frame: np.ndarray) -> List[Detection]:
        """Runs object detection with ByteTrack tracking on the frame and returns Detection objects."""
        h, w = frame.shape[:2]
        detections: List[Detection] = []

        if self.model is not None:
            now = time.time()
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=self.conf_threshold,
                    device=self.device,
                    half=self.use_fp16,
                    verbose=False
                )
            except Exception:
                results = self.model.predict(
                    frame,
                    conf=self.conf_threshold,
                    device=self.device,
                    half=self.use_fp16,
                    verbose=False
                )
            for r in results:
                for box in r.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    name = self.model.names.get(cls_id, "").lower()

                    if "emote" in name or "banner" in name or "overlay" in name or "text" in name:
                        category = "emote"
                    elif "selected" in name or "card_selected" in name or "highlight" in name:
                        category = "selected"
                    elif "clock" in name or "drop" in name or cls_id == 999:
                        category = "clock"
                    elif "air" in name or "dragon" in name or "minion" in name or "balloon" in name:
                        category = "air"
                    elif "tower" in name or "hut" in name or "cannon" in name:
                        category = "building"
                    elif "spell" in name or "barrel" in name:
                        category = "spell"
                    else:
                        category = "ground"

                    is_ally = not name.startswith("enemy_")
                    track_id = None
                    vx, vy = 0.0, 0.0
                    if hasattr(box, "id") and box.id is not None:
                        try:
                            track_id = int(box.id[0].cpu().numpy())
                            cx = float((coords[0] + coords[2]) / 2.0 / w)
                            cy = float(coords[3] / h)
                            if track_id in self.track_history:
                                px, py, pt = self.track_history[track_id]
                                dt = max(1e-4, now - pt)
                                vx = float((cx - px) / dt)
                                vy = float((cy - py) / dt)
                            self.track_history[track_id] = (cx, cy, now)
                        except Exception:
                            pass

                    detections.append(
                        Detection(
                            x1=float(coords[0] / w),
                            y1=float(coords[1] / h),
                            x2=float(coords[2] / w),
                            y2=float(coords[3] / h),
                            confidence=conf,
                            cls_id=cls_id,
                            is_ally=is_ally,
                            category=category,
                            track_id=track_id,
                            vx=vx,
                            vy=vy,
                        )
                    )
        return detections

    def classify_screen_state(self, frame: np.ndarray) -> str:
        """
        Classifies screen state into:
        - 'battle': In active match (elixir bar and hand slots present)
        - 'result': Victory / Defeat / Draw crown screen
        - 'menu': Main lobby / chest / shop interface
        """
        h, w = frame.shape[:2]

        # 1. Check for Bottom Navigation Bar (Shop, Cards, Battle, Clan, Events) -> Strictly 'menu'
        nav_roi = frame[int(0.91 * h):int(0.99 * h), :]
        if nav_roi.size > 0:
            hsv_nav = cv2.cvtColor(nav_roi, cv2.COLOR_BGR2HSV)
            yellow_nav = cv2.inRange(hsv_nav, np.array([15, 120, 120]), np.array([35, 255, 255]))
            if np.sum(yellow_nav > 0) > 500:
                return "menu"

        # 2. Check for Elixir Bar in Battle (pink elixir or dark elixir bar background)
        bx1, by1, bx2, by2 = ELIXIR_BAR_BOX
        x1, y1 = int(bx1 * w), int(by1 * h)
        x2, y2 = int(bx2 * w), int(by2 * h)
        bar_roi = frame[y1:y2, x1:x2]
        if bar_roi.size > 0:
            hsv_bar = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
            pink_mask = cv2.inRange(hsv_bar, np.array([135, 60, 60]), np.array([170, 255, 255]))
            if np.sum(pink_mask > 0) > 80:
                return "battle"

        # 3. Check for Victory / Defeat banner at top center
        banner_roi = frame[int(0.08 * h):int(0.25 * h), int(0.20 * w):int(0.80 * w)]
        if banner_roi.size > 0:
            hsv_banner = cv2.cvtColor(banner_roi, cv2.COLOR_BGR2HSV)
            gold_mask = cv2.inRange(hsv_banner, np.array([15, 120, 120]), np.array([35, 255, 255]))
            red_mask1 = cv2.inRange(hsv_banner, np.array([0, 120, 100]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv_banner, np.array([170, 120, 100]), np.array([180, 255, 255]))
            red_mask = red_mask1 | red_mask2
            if np.sum(gold_mask > 0) > 1500 or np.sum(red_mask > 0) > 1500:
                # Also check for "OK" button at bottom
                btn_roi = frame[int(0.80 * h):int(0.95 * h), int(0.30 * w):int(0.70 * w)]
                if btn_roi.size > 0:
                    btn_hsv = cv2.cvtColor(btn_roi, cv2.COLOR_BGR2HSV)
                    yellow_mask = cv2.inRange(btn_hsv, np.array([15, 120, 120]), np.array([35, 255, 255]))
                    if np.sum(yellow_mask > 0) > 400:
                        return "result"

        # If dark elixir bar background exists, it is battle
        if bar_roi.size > 0:
            hsv_bar = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
            dark_mask = cv2.inRange(hsv_bar, np.array([0, 0, 10]), np.array([180, 80, 80]))
            if np.sum(dark_mask > 0) > (bar_roi.size * 0.20):
                return "battle"

        return "menu"

    def read_match_result(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Reads outcome ('victory', 'defeat', 'draw') and crown count from match result screen.
        """
        h, w = frame.shape[:2]
        banner_roi = frame[int(0.08 * h):int(0.25 * h), int(0.20 * w):int(0.80 * w)]
        outcome = "unknown"
        if banner_roi.size > 0:
            hsv = cv2.cvtColor(banner_roi, cv2.COLOR_BGR2HSV)
            gold_mask = cv2.inRange(hsv, np.array([15, 120, 120]), np.array([35, 255, 255]))
            red_mask1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([170, 120, 100]), np.array([180, 255, 255]))
            red_mask = red_mask1 | red_mask2

            gold_pixels = np.sum(gold_mask > 0)
            red_pixels = np.sum(red_mask > 0)
            if gold_pixels > red_pixels and gold_pixels > 800:
                outcome = "victory"
            elif red_pixels > 800:
                outcome = "defeat"
            else:
                outcome = "draw"

        # Count crowns in middle region
        crown_roi = frame[int(0.30 * h):int(0.55 * h), int(0.20 * w):int(0.80 * w)]
        crowns = 0
        if crown_roi.size > 0:
            hsv_crown = cv2.cvtColor(crown_roi, cv2.COLOR_BGR2HSV)
            crown_mask = cv2.inRange(hsv_crown, np.array([15, 140, 140]), np.array([35, 255, 255]))
            contours, _ = cv2.findContours(crown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 150:
                    crowns += 1
        crowns = int(np.clip(crowns, 0, 3))
        return {
            "outcome": outcome,
            "ally_crowns": crowns if outcome == "victory" else 0,
            "enemy_crowns": 0 if outcome == "victory" else crowns,
        }

    def read_elixir(self, frame: np.ndarray) -> float:
        """Reads elixir value (0.0 to 10.0) from pink bar."""
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = ELIXIR_BAR_BOX
        x1, y1 = int(bx1 * w), int(by1 * h)
        x2, y2 = int(bx2 * w), int(by2 * h)

        bar_crop = frame[y1:y2, x1:x2]
        if bar_crop.size == 0:
            return 5.0

        hsv = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2HSV)
        pink_mask = cv2.inRange(hsv, np.array([135, 70, 70]), np.array([170, 255, 255]))

        col_sums = np.sum(pink_mask > 0, axis=0)
        filled_cols = np.sum(col_sums > (bar_crop.shape[0] * 0.20))
        ratio = filled_cols / float(bar_crop.shape[1])
        elixir = float(np.clip(ratio * 10.0, 0.0, 10.0))
        return elixir

    def read_hand_cards(self, frame: np.ndarray) -> List[int]:
        """
        Reads the 4 card IDs in the player's hand slots 0..3 using template matching.
        Restricted to the account's unlocked_cards.
        Returns 0 for unknown card and logs unknown crop to unknown_cards/.
        """
        h, w = frame.shape[:2]
        hand_ids: List[int] = []

        for slot_idx, box in enumerate(HAND_SLOTS):
            bx1, by1, bx2, by2 = box
            x1, y1 = int(bx1 * w), int(by1 * h)
            x2, y2 = int(bx2 * w), int(by2 * h)

            # Inward padding to focus on card portrait
            crop_w = x2 - x1
            crop_h = y2 - y1
            pad_x = int(crop_w * 0.12)
            pad_y = int(crop_h * 0.12)
            card_crop = frame[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]

            if card_crop.size == 0:
                hand_ids.append(0)
                continue

            card_crop_std = cv2.resize(card_crop, (64, 64), interpolation=cv2.INTER_AREA)

            best_card_id = 0
            best_score = 0.0

            # Match against templates of unlocked cards only
            for cid in self.unlocked_cards:
                if cid in self.card_templates:
                    for tmpl in self.card_templates[cid]:
                        res = cv2.matchTemplate(card_crop_std, tmpl, cv2.TM_CCOEFF_NORMED)
                        score = float(res[0, 0])
                        if score > best_score:
                            best_score = score
                            best_card_id = cid

            if best_score >= 0.40:
                hand_ids.append(best_card_id)
            else:
                # Log unknown crop for template expansion
                ts = int(time.time() * 1000)
                unknown_path = os.path.join(self.unknown_cards_dir, f"unknown_slot{slot_idx}_{ts}.png")
                cv2.imwrite(unknown_path, card_crop)
                hand_ids.append(0)  # unknown: never guess a card (guesses corrupt cost masks + hand embeddings)

        return hand_ids

    def read_tower_hps(self, frame: np.ndarray) -> List[float]:
        """
        Reads HP ratios [0.0..1.0] for the 6 towers using TOWER_HP_BOXES:
        [ally_king, ally_left, ally_right, enemy_king, enemy_left, enemy_right].
        Returns 0.0 for destroyed towers (enables pocket deployments).
        """
        h, w = frame.shape[:2]
        hps = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        tower_keys = [
            "ally_king", "ally_princess_left", "ally_princess_right",
            "enemy_king", "enemy_princess_left", "enemy_princess_right"
        ]

        for i, key in enumerate(tower_keys):
            bx1, by1, bx2, by2 = TOWER_HP_BOXES.get(key, TOWER_BOXES[key])
            x1, y1 = int(bx1 * w), int(by1 * h)
            x2, y2 = int(bx2 * w), int(by2 * h)
            bar_roi = frame[y1:y2, x1:x2]
            if bar_roi.size == 0:
                hps[i] = 1.0
                continue

            hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
            is_ally = i < 3

            # Green/blue health bar for ally, red for enemy
            if is_ally:
                green_mask = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
                blue_mask = cv2.inRange(hsv, np.array([90, 80, 80]), np.array([130, 255, 255]))
                active_mask = green_mask | blue_mask
            else:
                red_m1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
                red_m2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
                active_mask = red_m1 | red_m2

            active_pixels = np.sum(active_mask > 0)
            # Destroyed tower check: if virtually no health bar pixels exist
            if active_pixels < 4:
                # If king tower and no damage bar, king is full HP (undamaged king hides bar until hit)
                if "king" in key:
                    hps[i] = 1.0
                else:
                    # Destroyed princess tower
                    hps[i] = 0.0
            else:
                col_sums = np.sum(active_mask > 0, axis=0)
                active_cols = np.sum(col_sums > 0)
                ratio = float(np.clip(active_cols / float(bar_roi.shape[1]), 0.0, 1.0))
                hps[i] = ratio

        return hps

    def read_timer(self, frame: np.ndarray) -> Tuple[float, bool]:
        """
        Reads match timer from TIMER_BOX.
        Returns:
            - time_left: float seconds remaining (0.0 to 180.0/300.0)
            - is_double_elixir: bool (True if <= 60s or in overtime)
        """
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = TIMER_BOX
        x1, y1 = int(bx1 * w), int(by1 * h)
        x2, y2 = int(bx2 * w), int(by2 * h)

        timer_roi = frame[y1:y2, x1:x2]
        time_left = 180.0
        if timer_roi.size > 0:
            gray = cv2.cvtColor(timer_roi, cv2.COLOR_BGR2GRAY)
            white_mask = gray > 180
            white_pixel_count = np.sum(white_mask)
            # If text is present, estimate based on activity or default match countdown
            if white_pixel_count > 20:
                time_left = 120.0

        is_double_elixir = time_left <= 60.0
        return time_left, is_double_elixir

    def parse_frame(
        self,
        frame: np.ndarray,
        time_left: Optional[float] = None,
        est_enemy_elixir: float = 0.5,
        last_action: Tuple[int, int] = (0, 0),
        clock: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Processes frame into full observation dict. `clock` = video/stream time in seconds
        (offline extraction); default is wall time."""
        screen_state = self.classify_screen_state(frame)
        _now = clock if clock is not None else time.monotonic()
        if screen_state == "battle" and self._prev_screen_state != "battle":
            self.reset_match_state()          # new battle: clear tracker / velocity history
            self._battle_t0 = _now
        self._prev_screen_state = screen_state
        raw_detections = self.detect_raw(frame) if screen_state == "battle" else []

        elixir = self.read_elixir(frame) if screen_state == "battle" else 0.0
        hand_ids = self.read_hand_cards(frame) if screen_state == "battle" else [0, 0, 0, 0]
        tower_hps = self.read_tower_hps(frame) if screen_state == "battle" else [1.0] * 6

        detected_time, is_double_elixir = self.read_timer(frame)  # placeholder reader (120/180 only)
        if time_left is None and screen_state == "battle" and self._battle_t0 is not None:
            elapsed = _now - self._battle_t0                     # elapsed-time clock since battle start
            detected_time = max(0.0, 180.0 - elapsed)
            is_double_elixir = elapsed >= 120.0
        elif time_left is not None:
            is_double_elixir = time_left <= 60.0
        actual_time = time_left if time_left is not None else detected_time

        selected_slot: Optional[int] = None
        has_overlay = False
        clock_coords: List[Tuple[int, int]] = []
        filtered_units: List[Detection] = []

        for d in raw_detections:
            if d.category == "emote":
                has_overlay = True
            elif d.category == "clock":
                cx = (d.x1 + d.x2) / 2.0
                cy = d.y2
                grid_pos = screen_to_grid(cx, cy)
                if grid_pos is not None:
                    clock_coords.append(grid_pos)
            elif d.category == "selected":
                for s_idx, s_box in enumerate(HAND_SLOTS):
                    sx1, sy1, sx2, sy2 = s_box
                    if sx1 <= d.x1 <= sx2 and sy1 <= d.y1 <= sy2:
                        selected_slot = s_idx
                        break
            else:
                filtered_units.append(d)

        if screen_state == "battle":
            self.opponent_tracker.update(
                current_time=_now,
                enemy_detections=filtered_units,
                double_elixir=is_double_elixir
            )

        arena_grid, scalars, legal_card_mask = self.encoder.encode(
            detections=filtered_units,
            elixir=elixir,
            hand_card_ids=hand_ids,
            tower_hps=tower_hps,
            time_left=actual_time,
            est_enemy_elixir=est_enemy_elixir,
            opponent_tracker=self.opponent_tracker if screen_state == "battle" else None,
            last_action=last_action,
        )

        return {
            "screen_state": screen_state,
            "arena_grid": arena_grid,
            "scalars": scalars,
            "legal_card_mask": legal_card_mask,
            "hand_ids": hand_ids,
            "elixir": elixir,
            "clock_coords": clock_coords,
            "tower_hps": tower_hps,
            "selected_slot": selected_slot,
            "has_overlay": has_overlay,
            "time_left": actual_time,
            "is_double_elixir": is_double_elixir,
            "detections": filtered_units,
            "opponent_tracker": self.opponent_tracker,
        }
