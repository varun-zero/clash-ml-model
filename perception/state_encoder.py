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


def canonicalize_perspective(
    arena_grid: np.ndarray,
    tower_hps: List[float],
    player_side: str = "bottom",
) -> Tuple[np.ndarray, List[float]]:
    """
    Canonicalizes perspective so ally is always on the bottom.
    If player_side == 'top':
      - Rotates arena grid 180 degrees.
      - Swaps ally (channels 0,1,2) and enemy (channels 3,4,5).
      - Swaps ally and enemy tower HPs (with left/right mirror).
    """
    if player_side.lower() == "bottom":
        return arena_grid, tower_hps

    # 1. Rotate 180 degrees on spatial axes (height=1, width=2)
    # rot90 twice flips both vertical and horizontal
    canon_grid = np.rot90(arena_grid, 2, axes=(1, 2)).copy()

    # 2. Swap ally and enemy channels:
    # 0 (ally_ground) <-> 3 (enemy_ground)
    # 1 (ally_air)    <-> 4 (enemy_air)
    # 2 (ally_bldg)   <-> 5 (enemy_bldg)
    ally_channels = canon_grid[0:3].copy()
    enemy_channels = canon_grid[3:6].copy()
    canon_grid[0:3] = enemy_channels
    canon_grid[3:6] = ally_channels

    # 3. Swap tower HPs: [ally_k, ally_l, ally_r, enemy_k, enemy_l, enemy_r]
    # For top player, enemy towers become their ally towers, mirrored horizontally
    canon_hps = [
        tower_hps[3], tower_hps[5], tower_hps[4],
        tower_hps[0], tower_hps[2], tower_hps[1]
    ]

    return canon_grid, canon_hps


def canonicalize_tile_coordinate(tile_y: int, tile_x: int, player_side: str = "bottom") -> Tuple[int, int]:
    """Rotates tile coordinate 180 degrees if player_side is 'top'."""
    if player_side.lower() == "top":
        return (ARENA_ROWS - 1) - tile_y, (ARENA_COLS - 1) - tile_x
    return tile_y, tile_x


def stack_spatial_frames(grid_frames: List[np.ndarray], target_frames: int = 4) -> np.ndarray:
    """
    Stacks last 4 frames of arena_grid (each 8, 32, 18) into (32, 32, 18) float32.
    If fewer than target_frames exist, repeats the oldest frame.
    """
    if not grid_frames:
        return np.zeros((target_frames * 8, ARENA_ROWS, ARENA_COLS), dtype=np.float32)

    frames = list(grid_frames)
    while len(frames) < target_frames:
        frames.insert(0, frames[0].copy())
    recent_frames = frames[-target_frames:]
    return np.concatenate(recent_frames, axis=0).astype(np.float32)


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
        est_enemy_elixir: float = 0.5,  # Constant 0.5 in v1
        last_action: Tuple[int, int] = (0, 0),  # (action_card 0..4, action_tile 0..575)
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
            est_enemy_elixir: Constant 0.5 in v1
            last_action: Tuple of (last_card_slot, last_tile_idx)
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
        # Features:
        # [0]: ally_elixir_norm
        # [1]: est_enemy_elixir = 0.5 (constant in v1)
        # [2]: time_norm (time_left / 300.0)
        # [3..8]: 6x tower_hps [ak, al, ar, ek, el, er]
        # [9]: last_action_card / 4.0
        # [10]: last_action_tile / 575.0
        # [11]: is_double_elixir (1.0 if time_left <= 120.0 else 0.0)
        # ----------------------------------------------------------------------
        ally_elixir_norm = float(np.clip(elixir / 10.0, 0.0, 1.0))
        enemy_elixir_norm = 0.5  # Constant in v1
        time_norm = float(np.clip(time_left / 300.0, 0.0, 1.0))

        ak_hp = float(np.clip(tower_hps[0], 0.0, 1.0))
        al_hp = float(np.clip(tower_hps[1], 0.0, 1.0))
        ar_hp = float(np.clip(tower_hps[2], 0.0, 1.0))
        ek_hp = float(np.clip(tower_hps[3], 0.0, 1.0))
        el_hp = float(np.clip(tower_hps[4], 0.0, 1.0))
        er_hp = float(np.clip(tower_hps[5], 0.0, 1.0))

        last_card, last_tile = last_action
        last_card_norm = float(np.clip(last_card / 4.0, 0.0, 1.0))
        last_tile_norm = float(np.clip(last_tile / max(1.0, float(TOTAL_TILES - 1)), 0.0, 1.0))
        is_double_elixir = 1.0 if time_left <= 120.0 else 0.0

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
            last_card_norm,
            last_tile_norm,
            is_double_elixir,
        ], dtype=np.float32)

        assert scalars.shape == (12,), f"Expected scalars shape (12,), got {scalars.shape}"

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

