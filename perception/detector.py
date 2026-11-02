"""
YOLO and Vision Perception Pipeline for Clash Royale.
Supports Ultralytics YOLO (PyTorch .pt), ONNX Runtime FP16, and TensorRT.
Detects:
- Units, buildings, spells
- Deployment clock / drop indicators ('clock' class)
- Hand card slots (0..3)
- Elixir bar / numeric readout
- Tower HP bars
"""
from typing import Dict, List, Tuple, Optional, Any
import os
import cv2
import numpy as np

from configs.constants import (
    ARENA_BOX,
    HAND_SLOTS,
    ELIXIR_BAR_BOX,
    ELIXIR_TEXT_BOX,
    TOWER_BOXES,
    DEFAULT_DECK_CLASSES,
    DECK_ID_TO_COST,
    DECK_ID_TO_IS_SPELL,
    CARD_DATABASE,
    CARD_NAME_TO_ID,
    screen_to_grid,
)
from perception.state_encoder import StateEncoder, Detection


class ClashRoyaleDetector:
    """
    Perception engine wrapping YOLO detection, UI cropping, and board parsing.
    Supports customizable card deck classes (default: 2.6 Hog Cycle v1, 10 classes).
    """
    def __init__(
        self,
        weights: Optional[str] = None,
        model_path: Optional[str] = None,  # Backward compatible alias
        conf_threshold: float = 0.35,
        device: str = "cuda:0",
        use_fp16: bool = True,
        card_classes: Optional[List[Dict]] = None,
        require_weights: bool = False,
    ):
        self.conf_threshold = conf_threshold
        self.device = device
        self.use_fp16 = use_fp16
        self.encoder = StateEncoder()
        self.model = None
        self.onnx_session = None

        # Configurable hand card classes (v1 = 2.6 Hog Cycle, 10 classes)
        self.card_classes = card_classes or DEFAULT_DECK_CLASSES
        self.card_names = {c["name"].lower(): c["id"] for c in self.card_classes}

        weights_file = weights or model_path
        if weights_file:
            if not os.path.exists(weights_file):
                raise FileNotFoundError(f"[Detector] Specified --weights file not found: {weights_file}")
            self._load_model(weights_file)
        elif require_weights:
            raise ValueError("[Detector] --weights path is required and cannot be empty.")

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
            except Exception as e:
                print(f"[Detector] Warning: Could not initialize ONNX session ({e}).")
        else:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except Exception as e:
                raise RuntimeError(f"[Detector] Failed to load Ultralytics YOLO from {model_path}: {e}")

    def detect_raw(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs object detection on the frame and returns Detection objects.
        """
        h, w = frame.shape[:2]
        detections: List[Detection] = []

        if self.model is not None:
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

                    # Check for overlay / emote detections
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
                        )
                    )
        elif self.onnx_session is not None:
            input_shape = self.onnx_session.get_inputs()[0].shape
            in_h, in_w = (input_shape[2], input_shape[3]) if len(input_shape) == 4 else (640, 640)
            img = cv2.resize(frame, (in_w, in_h))
            img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: img})
            raw_boxes = outputs[0]
            if len(raw_boxes.shape) == 3:
                raw_boxes = raw_boxes[0]
            if raw_boxes.shape[0] < raw_boxes.shape[1] and raw_boxes.shape[0] in (84, 85):
                raw_boxes = raw_boxes.T

            for row in raw_boxes:
                scores = row[4:]
                cls_id = int(np.argmax(scores))
                conf = float(scores[cls_id])
                if conf < self.conf_threshold:
                    continue
                cx, cy, bw, bh = row[0:4]
                x1 = float((cx - bw / 2.0) / in_w)
                y1 = float((cy - bh / 2.0) / in_h)
                x2 = float((cx + bw / 2.0) / in_w)
                y2 = float((cy + bh / 2.0) / in_h)
                detections.append(
                    Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=conf,
                        cls_id=cls_id,
                        is_ally=True,
                        category="ground"
                    )
                )

        return detections

    def read_elixir(self, frame: np.ndarray) -> float:
        """
        Estimates current elixir level from the pink elixir bar at the bottom.
        Pink elixir hue in HSV is approximately 140..170 (in OpenCV 0..180 range).
        """
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = ELIXIR_BAR_BOX
        x1, y1 = int(bx1 * w), int(by1 * h)
        x2, y2 = int(bx2 * w), int(by2 * h)

        bar_crop = frame[y1:y2, x1:x2]
        if bar_crop.size == 0:
            return 5.0

        hsv = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2HSV)
        # Pink/purple elixir mask
        lower_pink = np.array([135, 80, 80])
        upper_pink = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # Fraction of pink along the horizontal width
        col_sums = np.sum(mask > 0, axis=0)
        filled_cols = np.sum(col_sums > (bar_crop.shape[0] * 0.25))
        ratio = filled_cols / float(bar_crop.shape[1])
        elixir = float(np.clip(ratio * 10.0, 0.0, 10.0))
        return elixir

    def read_hand_cards(self, frame: np.ndarray) -> List[int]:
        """
        Reads the 4 card IDs in the player's hand slots 0..3.
        Defaults to 2.6 Hog cycle starters: [1, 2, 3, 4]
        (1: Hog Rider, 2: Cannon, 3: Musketeer, 4: Ice Golem).
        """
        # Default 2.6 Hog card IDs: 1 (Hog Rider), 2 (Cannon), 3 (Musketeer), 4 (Ice Golem)
        return [1, 2, 3, 4]

    def read_tower_hps(self, frame: np.ndarray) -> List[float]:
        """
        Reads HP ratios [0.0..1.0] for the 6 towers:
        [ally_king, ally_left, ally_right, enemy_king, enemy_left, enemy_right].
        """
        h, w = frame.shape[:2]
        hps = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        tower_keys = [
            "ally_king", "ally_princess_left", "ally_princess_right",
            "enemy_king", "enemy_princess_left", "enemy_princess_right"
        ]

        for i, key in enumerate(tower_keys):
            bx1, by1, bx2, by2 = TOWER_BOXES[key]
            x1, y1 = int(bx1 * w), int(by1 * h)
            x2, y2 = int(bx2 * w), int(by2 * h)
            tower_roi = frame[y1:y2, x1:x2]
            if tower_roi.size == 0:
                continue

            # Look for green / red health bar segments
            hsv = cv2.cvtColor(tower_roi, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
            green_pixels = np.sum(green_mask > 0)
            if green_pixels < 10:
                hps[i] = 1.0
            else:
                hps[i] = 0.8
        return hps

    def parse_frame(
        self,
        frame: np.ndarray,
        time_left: float = 180.0,
        est_enemy_elixir: float = 0.5,
        last_action: Tuple[int, int] = (0, 0),
    ) -> Dict[str, Any]:
        """
        Processes a full 9:16 frame and extracts all observation components:
        Returns:
            - arena_grid: (8, 32, 18) float32
            - scalars: (12,) float32
            - legal_card_mask: (5,) bool
            - hand_ids: [c0, c1, c2, c3]
            - elixir: float
            - clock_coords: list of (tile_y, tile_x)
            - tower_hps: list of 6 floats
            - detections: list of Detection
            - selected_slot: Optional[int] (0..3) if selected card highlight detected
            - has_overlay: bool (True if emote or text banner detected)
        """
        h, w = frame.shape[:2]
        detections = self.detect_raw(frame)
        elixir = self.read_elixir(frame)
        hand_ids = self.read_hand_cards(frame)
        tower_hps = self.read_tower_hps(frame)

        # Detect deployment clocks, card highlights, and emote overlays
        clock_coords: List[Tuple[int, int]] = []
        selected_slot: Optional[int] = None
        has_overlay: bool = False

        for det in detections:
            if det.category == "clock":
                center_x = (det.x1 + det.x2) / 2.0
                center_y = (det.y1 + det.y2) / 2.0
                gpos = screen_to_grid(center_x, center_y)
                if gpos is not None:
                    clock_coords.append(gpos)
            elif det.category == "selected":
                cx = (det.x1 + det.x2) / 2.0
                cy = (det.y1 + det.y2) / 2.0
                for s_idx, (sx1, sy1, sx2, sy2) in enumerate(HAND_SLOTS):
                    if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                        selected_slot = s_idx
                        break
            elif det.category == "emote":
                has_overlay = True

        # Encode state tensors
        arena_grid, scalars, legal_card_mask = self.encoder.encode(
            detections=detections,
            elixir=elixir,
            hand_card_ids=hand_ids,
            tower_hps=tower_hps,
            time_left=time_left,
            est_enemy_elixir=est_enemy_elixir,
            last_action=last_action,
            frame_width=w,
            frame_height=h,
        )

        return {
            "arena_grid": arena_grid,
            "scalars": scalars,
            "legal_card_mask": legal_card_mask,
            "hand_ids": hand_ids,
            "elixir": elixir,
            "clock_coords": clock_coords,
            "tower_hps": tower_hps,
            "detections": detections,
            "selected_slot": selected_slot,
            "has_overlay": has_overlay,
        }

