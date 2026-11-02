"""
State encoder for Clash Royale board state.

All card IDs are GLOBAL ``CARD_DATABASE`` IDs. There is no separate local deck
vocabulary. Outputs:
- (8, 32, 18) float32 spatial tensor
- (12,) float32 scalar vector
- (5,) bool legal card action mask
- (576,) bool legal tile masks via ``get_legal_tile_mask``
"""
from typing import List, Dict, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass
import time
import numpy as np

from configs.constants import (
    ARENA_ROWS,
    ARENA_COLS,
    ARENA_GRID,
    TOTAL_TILES,
    ARENA_BOX,
    CARD_ID_TO_COST,
    CARD_ID_TO_IS_SPELL,
    CARD_ID_TO_CATEGORY,
    WAIT_CARD_ID,
    screen_to_grid,
    get_default_deployable_mask,
)


@dataclass
class Detection:
    """Represents one detection from YOLO or template matching with ByteTrack tracking."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    cls_id: int
    is_ally: bool
    category: str = "ground"  # ground, air, building, spell, clock, selected, emote
    track_id: Optional[int] = None
    vx: float = 0.0
    vy: float = 0.0


class OpponentTracker:
    """
    Tracks opponent's elixir, card cycle, and active units across match frames.
    - Elixir: starts at 5.0, regenerates at 1/2.8s (or 1/1.4s during double elixir).
    - Deduction: when a new enemy unit track ID appears, looks up cost in CARD_ID_TO_COST and deducts it.
    - Card Cycle: tracks unique enemy cards seen (up to 8 cards) and active rotation.
    """
    def __init__(self, initial_elixir: float = 5.0):
        self.enemy_elixir = float(initial_elixir)
        self.last_update_time = time.time()
        self.seen_track_ids: Set[int] = set()
        self.seen_cards: List[int] = []  # Unique cards identified in opponent deck
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.total_elixir_spent: int = 0

    def reset(self, initial_elixir: float = 5.0):
        """Resets tracker state at the beginning of a match."""
        self.enemy_elixir = float(initial_elixir)
        self.last_update_time = time.time()
        self.seen_track_ids.clear()
        self.seen_cards.clear()
        self.active_tracks.clear()
        self.total_elixir_spent = 0

    def update(
        self,
        current_time: float,
        enemy_detections: List[Detection],
        double_elixir: bool = False
    ) -> float:
        """
        Updates opponent elixir and registers any newly spawned units.
        """
        # 1. Passive regeneration over dt
        dt = max(0.0, current_time - self.last_update_time)
        self.last_update_time = current_time
        regen_rate = 1.0 / 1.4 if double_elixir else 1.0 / 2.8
        self.enemy_elixir = min(10.0, self.enemy_elixir + dt * regen_rate)

        # 2. Check for newly spawned enemy tracks
        for d in enemy_detections:
            if d.is_ally:
                continue
            tid = d.track_id
            cid = d.cls_id

            if tid is not None:
                if tid not in self.seen_track_ids:
                    self.seen_track_ids.add(tid)
                    cost = CARD_ID_TO_COST.get(cid, 3)
                    self.enemy_elixir = max(0.0, self.enemy_elixir - cost)
                    self.total_elixir_spent += cost

                    if cid not in self.seen_cards and len(self.seen_cards) < 8:
                        self.seen_cards.append(cid)

                    self.active_tracks[tid] = {
                        "cls_id": cid,
                        "spawn_time": current_time,
                        "cost": cost,
                    }

        return float(np.clip(self.enemy_elixir, 0.0, 10.0))

    def get_elixir(self) -> float:
        """Returns current estimated enemy elixir [0.0..10.0]."""
        return float(np.clip(self.enemy_elixir, 0.0, 10.0))

    def get_normalized_elixir(self) -> float:
        """Returns normalized enemy elixir [0.0..1.0] for model scalars[1]."""
        return float(np.clip(self.enemy_elixir / 10.0, 0.0, 1.0))

    def get_cycle_summary(self) -> Dict[str, Any]:
        """Returns summary of opponent tracked deck and elixir."""
        return {
            "estimated_elixir": round(self.enemy_elixir, 1),
            "seen_cards_count": len(self.seen_cards),
            "seen_card_ids": list(self.seen_cards),
            "total_elixir_spent": self.total_elixir_spent,
            "active_tracks_count": len(self.active_tracks),
        }


def canonicalize_perspective(
    arena_grid: np.ndarray,
    tower_hps: List[float],
    player_side: str = "bottom",
) -> Tuple[np.ndarray, List[float]]:
    """Rotate a top-side recording so the ally is always on the bottom."""
    if player_side.lower() == "bottom":
        return arena_grid, tower_hps

    canon_grid = np.rot90(arena_grid, 2, axes=(1, 2)).copy()

    ally_channels = canon_grid[0:3].copy()
    enemy_channels = canon_grid[3:6].copy()
    canon_grid[0:3] = enemy_channels
    canon_grid[3:6] = ally_channels

    canon_hps = [
        tower_hps[3],
        tower_hps[5],
        tower_hps[4],
        tower_hps[0],
        tower_hps[2],
        tower_hps[1],
    ]
    return canon_grid, canon_hps


def canonicalize_tile_coordinate(tile_y: int, tile_x: int, player_side: str = "bottom") -> Tuple[int, int]:
    """Rotate a tile coordinate if the recording is from the top perspective."""
    if player_side.lower() == "top":
        return (ARENA_ROWS - 1) - tile_y, (ARENA_COLS - 1) - tile_x
    return tile_y, tile_x


def stack_spatial_frames(grid_frames: List[np.ndarray], target_frames: int = 4) -> np.ndarray:
    """Stack the last N (8, 32, 18) frames into (8*N, 32, 18)."""
    if not grid_frames:
        return np.zeros((target_frames * 8, ARENA_ROWS, ARENA_COLS), dtype=np.float32)

    frames = list(grid_frames)
    while len(frames) < target_frames:
        frames.insert(0, frames[0].copy())
    recent_frames = frames[-target_frames:]
    return np.concatenate(recent_frames, axis=0).astype(np.float32)


class StateEncoder:
    """Encodes visual perception and UI readings into neural-network inputs."""

    def __init__(self):
        self.arena_rows = ARENA_ROWS
        self.arena_cols = ARENA_COLS

    def parse_detection_array(
        self,
        raw_detections: Union[np.ndarray, List[List[float]], List[Detection]],
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> List[Detection]:
        """Parse detections into Detection objects."""
        parsed: List[Detection] = []
        if isinstance(raw_detections, list) and len(raw_detections) > 0 and isinstance(raw_detections[0], Detection):
            return raw_detections

        if raw_detections is None or len(raw_detections) == 0:
            return parsed

        for item in raw_detections:
            if isinstance(item, Detection):
                parsed.append(item)
                continue

            x1, y1, x2, y2, conf, cls_id, ally_flag = item[:7]
            if frame_width and frame_height and (x2 > 1.0 or y2 > 1.0):
                x1 /= frame_width
                y1 /= frame_height
                x2 /= frame_width
                y2 /= frame_height

            is_ally = bool(ally_flag == 1 or ally_flag is True)
            cls_id_int = int(cls_id)
            parsed.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(conf),
                    cls_id=cls_id_int,
                    is_ally=is_ally,
                    category=CARD_ID_TO_CATEGORY.get(cls_id_int, "ground"),
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
        est_enemy_elixir: float = 0.5,
        opponent_tracker: Optional[OpponentTracker] = None,
        last_action: Tuple[int, int] = (0, 0),
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        double_elixir: Optional[bool] = None,
        overtime: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode one frame into spatial grid (8, 32, 18), scalars (12,), and legal_card_mask (5,).

        Scalars (12):
            0 ally elixir / 10
            1 estimated enemy elixir (from OpponentTracker or fallback)
            2 time remaining / 180
            3 ally king HP
            4 ally left princess HP
            5 ally right princess HP
            6 enemy king HP
            7 enemy left princess HP
            8 enemy right princess HP
            9 last card slot / 4
            10 last tile / 575
            11 double-elixir indicator
        """
        if tower_hps is None or len(tower_hps) < 6:
            tower_hps = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        parsed_dets = self.parse_detection_array(detections, frame_width, frame_height)

        if opponent_tracker is not None:
            enemy_elixir_norm = opponent_tracker.get_normalized_elixir()
        else:
            enemy_elixir_norm = float(np.clip(est_enemy_elixir, 0.0, 1.0))

        arena_grid = np.zeros((8, self.arena_rows, self.arena_cols), dtype=np.float32)
        enemy_left_down = tower_hps[4] <= 0.05
        enemy_right_down = tower_hps[5] <= 0.05
        arena_grid[7] = get_default_deployable_mask(enemy_left_down, enemy_right_down).astype(np.float32)

        for det in parsed_dets:
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

        ally_elixir_norm = float(np.clip(elixir / 10.0, 0.0, 1.0))
        time_norm = float(np.clip(time_left / 180.0, 0.0, 1.0))

        ak_hp = float(np.clip(tower_hps[0], 0.0, 1.0))
        al_hp = float(np.clip(tower_hps[1], 0.0, 1.0))
        ar_hp = float(np.clip(tower_hps[2], 0.0, 1.0))
        ek_hp = float(np.clip(tower_hps[3], 0.0, 1.0))
        el_hp = float(np.clip(tower_hps[4], 0.0, 1.0))
        er_hp = float(np.clip(tower_hps[5], 0.0, 1.0))

        last_card, last_tile = last_action
        last_card_norm = float(np.clip(last_card / 4.0, 0.0, 1.0))
        last_tile_norm = float(np.clip(last_tile / max(1.0, float(TOTAL_TILES - 1)), 0.0, 1.0))

        if double_elixir is None:
            double_elixir = time_left <= 60.0
        double_elixir_norm = 1.0 if double_elixir else 0.0

        scalars = np.array(
            [
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
                double_elixir_norm,
            ],
            dtype=np.float32,
        )
        if scalars.shape != (12,):
            raise AssertionError(f"Expected scalars shape (12,), got {scalars.shape}")

        legal_card_mask = np.zeros(5, dtype=bool)
        legal_card_mask[0] = True
        for i in range(4):
            card_id = int(hand_card_ids[i]) if i < len(hand_card_ids) else WAIT_CARD_ID
            if card_id > 0:
                cost = CARD_ID_TO_COST.get(card_id, 99)
                if cost <= elixir:
                    legal_card_mask[i + 1] = True

        return arena_grid, scalars, legal_card_mask

    def get_legal_tile_mask(
        self,
        card_id: int,
        deployable_grid: Optional[np.ndarray] = None,
        tower_hps: Optional[List[float]] = None,
        enemy_left_down: bool = False,
        enemy_right_down: bool = False,
    ) -> np.ndarray:
        """Return a (576,) legal-placement mask for a global card ID.

        Spells (including Miner/Goblin Drill placement) are legal anywhere.
        Troops/buildings are restricted to the ally half plus destroyed-tower
        pockets. Wait (0) uses tile 0 as a harmless anchor.
        """
        card_id = int(card_id)
        if card_id == WAIT_CARD_ID:
            mask = np.zeros(TOTAL_TILES, dtype=bool)
            mask[0] = True
            return mask

        if CARD_ID_TO_IS_SPELL.get(card_id, False):
            return np.ones(TOTAL_TILES, dtype=bool)

        if tower_hps is not None and len(tower_hps) >= 6:
            enemy_left_down = enemy_left_down or tower_hps[4] <= 0.05
            enemy_right_down = enemy_right_down or tower_hps[5] <= 0.05

        if deployable_grid is None:
            deployable_grid = get_default_deployable_mask(enemy_left_down, enemy_right_down)

        return deployable_grid.flatten().astype(bool)
