"""
State encoder for Clash Royale board state.
Transforms raw YOLO object detections and UI readouts into:
- (8, 32, 18) float32 spatial tensor
- (12,) float32 scalar vector
- (5,) boolean legal card action mask
- (576,) boolean legal tile placement masks
"""
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np

from configs.constants import (
    ARENA_ROWS,
    ARENA_COLS,
    ARENA_GRID,
    TOTAL_TILES,
    ARENA_BOX,
    CARD_DATABASE,
    CARD_ID_TO_COST,
    CARD_ID_TO_IS_SPELL,
    CARD_ID_TO_CATEGORY,
    screen_to_grid,
    get_default_deployable_mask,
)


@dataclass
class Detection:
    """Represents a single detection from YOLO or template matching."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    cls_id: int
    is_ally: bool
    category: str = "ground"  # 'ground', 'air', 'building', 'spell', 'clock'


class StateEncoder:
    """
    Encodes raw visual perception and UI readings into neural network inputs.
    """
    def __init__(self):
        self.arena_rows = ARENA_ROWS
        self.arena_cols = ARENA_COLS

    def parse_detection_array(
        self,
        raw_detections: Union[np.ndarray, List[List[float]], List[Detection]],
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> List[Detection]:
        """
        Parses raw detections into a list of Detection objects.
        Accepts:
            - np.ndarray of shape (N, 7): [x1, y1, x2, y2, conf, cls_id, ally_flag]
            - List of Detection dataclass instances
        """
        parsed: List[Detection] = []
        if isinstance(raw_detections, list) and len(raw_detections) > 0 and isinstance(raw_detections[0], Detection):
            return raw_detections

        if raw_detections is None or len(raw_detections) == 0:
            return parsed

        for item in raw_detections:
            if isinstance(item, Detection):
                parsed.append(item)
                continue

            # Assuming [x1, y1, x2, y2, conf, cls_id, ally_flag]
            x1, y1, x2, y2, conf, cls_id, ally_flag = item[:7]
            
            # If coordinates are pixel based, normalize with frame dims
            if frame_width and frame_height and (x2 > 1.0 or y2 > 1.0):
                x1 /= frame_width
                y1 /= frame_height
                x2 /= frame_width
                y2 /= frame_height

            is_ally = bool(ally_flag == 1 or ally_flag is True)
            cls_id_int = int(cls_id)
            category = CARD_ID_TO_CATEGORY.get(cls_id_int, "ground")

            parsed.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(conf),
                    cls_id=cls_id_int,
                    is_ally=is_ally,
                    category=category,
                )
            )

        return parsed

    def encode(
        self,
        detections: Union[np.ndarray, List[Detection]],
        elixir: float,
        hand_card_ids: List[int],
        tower_hps: Optional[List[float]] = None,
        time_left: float = 180.0,
        est_enemy_elixir: float = 5.0,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main encoding routine.
        Args:
            detections: List of detections or array [x1, y1, x2, y2, conf, cls_id, ally_flag]
            elixir: Current ally elixir (0.0 to 10.0)
            hand_card_ids: List of 4 card IDs currently in hand
            tower_hps: List of 6 floats [0.0..1.0] for:
                       [ally_king, ally_left, ally_right, enemy_king, enemy_left, enemy_right]
            time_left: Match time remaining in seconds (e.g. 180.0 down to 0.0)
            est_enemy_elixir: Estimated enemy elixir (0.0 to 10.0)
            frame_width, frame_height: Optional viewport dimensions for coordinate normalization

        Returns:
            arena_grid: (8, 32, 18) np.float32
            scalars: (12,) np.float32
            legal_card_mask: (5,) bool
        """
        # Default tower HPs if not provided (all at full 1.0)
        if tower_hps is None or len(tower_hps) < 6:
            tower_hps = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        parsed_dets = self.parse_detection_array(detections, frame_width, frame_height)

        # ----------------------------------------------------------------------
        # 1. Build Spatial Tensor: (8, 32, 18)
        # Channel 0: ally_ground_count
        # Channel 1: ally_air_count
        # Channel 2: ally_building
        # Channel 3: enemy_ground_count
        # Channel 4: enemy_air_count
        # Channel 5: enemy_building
        # Channel 6: spell_active
        # Channel 7: deployable_zone_mask
        # ----------------------------------------------------------------------
        arena_grid = np.zeros((8, self.arena_rows, self.arena_cols), dtype=np.float32)

        # Determine if enemy princess towers are down for pocket deployment
        enemy_left_down = tower_hps[4] <= 0.05
        enemy_right_down = tower_hps[5] <= 0.05
        deployable_mask = get_default_deployable_mask(enemy_left_down, enemy_right_down)
        arena_grid[7] = deployable_mask.astype(np.float32)

        for det in parsed_dets:
            # Anchor coordinate: bottom-center of bounding box for units
            foot_x = (det.x1 + det.x2) / 2.0
            foot_y = det.y2

            grid_pos = screen_to_grid(foot_x, foot_y)
            if grid_pos is None:
                continue

            gy, gx = grid_pos
            cat = det.category.lower()

            if det.is_ally:
                if cat == "ground":
                    arena_grid[0, gy, gx] += 1.0
                elif cat == "air":
                    arena_grid[1, gy, gx] += 1.0
                elif cat == "building":
                    arena_grid[2, gy, gx] = 1.0
                elif cat == "spell":
                    arena_grid[6, gy, gx] = 1.0
            else:
                if cat == "ground":
                    arena_grid[3, gy, gx] += 1.0
                elif cat == "air":
                    arena_grid[4, gy, gx] += 1.0
                elif cat == "building":
                    arena_grid[5, gy, gx] = 1.0
                elif cat == "spell":
                    arena_grid[6, gy, gx] = 1.0

        # ----------------------------------------------------------------------
        # 2. Build Scalars Vector: (12,) normalized to [0, 1]
        # ----------------------------------------------------------------------
        ally_elixir_norm = float(np.clip(elixir / 10.0, 0.0, 1.0))
        enemy_elixir_norm = float(np.clip(est_enemy_elixir / 10.0, 0.0, 1.0))
        time_norm = float(np.clip(time_left / 300.0, 0.0, 1.0))

        ak_hp = float(np.clip(tower_hps[0], 0.0, 1.0))
        al_hp = float(np.clip(tower_hps[1], 0.0, 1.0))
        ar_hp = float(np.clip(tower_hps[2], 0.0, 1.0))
        ek_hp = float(np.clip(tower_hps[3], 0.0, 1.0))
        el_hp = float(np.clip(tower_hps[4], 0.0, 1.0))
        er_hp = float(np.clip(tower_hps[5], 0.0, 1.0))

        is_double_elixir = 1.0 if time_left <= 120.0 else 0.0
        is_overtime = 1.0 if time_left <= 60.0 and (ak_hp > 0 and ek_hp > 0) else 0.0
        ally_king_active = 1.0 if (ak_hp < 0.999 or al_hp <= 0.05 or ar_hp <= 0.05) else 0.0

        scalars = np.array([
            ally_elixir_norm,
            enemy_elixir_norm,
            time_norm,
            ak_hp,
            al_hp,
            ar_hp,
            ek_hp,
            el_hp,
            er_hp,
            is_double_elixir,
            is_overtime,
            ally_king_active,
        ], dtype=np.float32)

        # ----------------------------------------------------------------------
        # 3. Build Legal Card Action Mask: (5,)
        # [0: Wait is always legal,
        #  1: slot 0 affordable,
        #  2: slot 1 affordable,
        #  3: slot 2 affordable,
        #  4: slot 3 affordable]
        # ----------------------------------------------------------------------
        legal_card_mask = np.zeros(5, dtype=bool)
        legal_card_mask[0] = True  # Waiting is always an option

        for i in range(4):
            card_id = hand_card_ids[i] if i < len(hand_card_ids) else 0
            if card_id > 0:
                cost = CARD_ID_TO_COST.get(card_id, 99)
                if cost <= elixir:
                    legal_card_mask[i + 1] = True

        return arena_grid, scalars, legal_card_mask

    def get_legal_tile_mask(
        self,
        card_id: int,
        deployable_grid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns a boolean mask of shape (576,) indicating legal deployment tiles
        for the specified card.
        - Spells, Miner, Goblin Drill: all 576 tiles are legal.
        - Troops/Buildings: restricted to deployable ally territory.
        - Wait (ID 0): tile 0 is set True (arbitrary anchor since no tile is deployed).
        """
        if card_id == 0:
            mask = np.zeros(TOTAL_TILES, dtype=bool)
            mask[0] = True
            return mask

        is_spell = CARD_ID_TO_IS_SPELL.get(card_id, False)
        if is_spell:
            return np.ones(TOTAL_TILES, dtype=bool)

        if deployable_grid is None:
            deployable_grid = get_default_deployable_mask()

        # Flatten (32, 18) -> (576,)
        return deployable_grid.flatten().astype(bool)

