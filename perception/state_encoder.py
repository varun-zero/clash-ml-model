"""
State encoder and opponent intelligence tracker for Clash Royale board state.

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
    CARD_ID_TO_NAME,
    WAIT_CARD_ID,
    screen_to_grid,
    get_default_deployable_mask,
)
from configs.card_features import (
    get_card_roles,
    get_intrinsic_danger,
)
from configs.meta_decks import identify_meta_deck
from configs.observation_schema import Ch, Sc


@dataclass
class Detection:
    """Represents one detection from YOLO or template/CV matching with tracking."""

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
    Advanced real-time intelligence tracker for the opponent:
    1. Elixir tracking: dynamic rate (single/double/triple), tracking elixir leaked at 10.0.
    2. Card cycle tracking: tracks exact rotation history and total cards played.
       - Each played card enters position 4 (needs 4 other cards before returning).
       - Tracks which 4 cards are currently estimated to be IN HAND.
    3. Spatial placement memory & next-placement prediction:
       - Learns opponent lane bias (left vs right) and deployment depth habits.
       - Predicts exact (tile_y, tile_x) where the opponent will deploy their next threat.
    4. Counter reservation advisories:
       - Signals when to hold Arrows (vs Goblin Barrel / Swarm), Mini P.E.K.K.A (vs Hog / Giant / Tank),
         or Musketeer / Archers (vs Balloon / Air).
    5. Meta Deck Identification: matches revealed cards against competitive archetype signatures.
    """
    def __init__(self, initial_elixir: float = 5.0):
        self.enemy_elixir = float(initial_elixir)
        self.last_update_time = time.time()
        self.seen_track_ids: Set[int] = set()
        self.seen_cards: List[int] = []  # Unique cards identified in opponent deck (max 8)
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self.total_elixir_spent: int = 0
        self.elixir_leaked: float = 0.0

        # Card cycle rotation tracking
        self.play_history: List[int] = []  # Chronological order of opponent card plays
        self.card_last_play_idx: Dict[int, int] = {}  # cid -> index in play_history

        # Spatial placement memory for predicting opponent habits
        self.spatial_play_history: List[Dict[str, Any]] = []
        self.left_lane_plays: int = 0
        self.right_lane_plays: int = 0
        self._recent_untracked_spawns: List[Tuple[float, int, int, int]] = []  # (time, gy, gx, cid)

    def reset(self, initial_elixir: float = 5.0):
        """Resets tracker state at the beginning of a match."""
        self.enemy_elixir = float(initial_elixir)
        self.last_update_time = time.time()
        self.seen_track_ids.clear()
        self.seen_cards.clear()
        self.active_tracks.clear()
        self.total_elixir_spent = 0
        self.elixir_leaked = 0.0
        self.play_history.clear()
        self.card_last_play_idx.clear()
        self.spatial_play_history.clear()
        self.left_lane_plays = 0
        self.right_lane_plays = 0
        self._recent_untracked_spawns.clear()

    def _register_card_play(
        self,
        cid: int,
        current_time: float,
        grid_pos: Optional[Tuple[int, int]] = None,
        track_id: Optional[int] = None,
    ):
        """Records a confirmed opponent card play, deducts cost, and updates cycle & spatial memory."""
        cost = CARD_ID_TO_COST.get(cid, 3) if cid > 0 else 3
        self.enemy_elixir = max(0.0, self.enemy_elixir - cost)
        self.total_elixir_spent += cost

        if cid > 0:
            play_idx = len(self.play_history)
            self.play_history.append(cid)
            self.card_last_play_idx[cid] = play_idx

            if cid not in self.seen_cards and len(self.seen_cards) < 8:
                self.seen_cards.append(cid)

        gy, gx = grid_pos if grid_pos is not None else (14, 14)
        if gx < 9:
            self.left_lane_plays += 1
            lane = "left"
        else:
            self.right_lane_plays += 1
            lane = "right"

        self.spatial_play_history.append({
            "cls_id": cid,
            "time": current_time,
            "tile_y": gy,
            "tile_x": gx,
            "lane": lane,
            "cost": cost,
        })

        if track_id is not None:
            self.active_tracks[track_id] = {
                "cls_id": cid,
                "spawn_time": current_time,
                "cost": cost,
                "tile_y": gy,
                "tile_x": gx,
            }

    def update(
        self,
        current_time: float,
        enemy_detections: List[Detection],
        double_elixir: bool = False,
        triple_elixir: bool = False,
    ) -> float:
        """
        Updates opponent elixir and registers any newly spawned units.
        Supports both ByteTrack track_id detections and spatial-temporal deduplication
        when track_id is absent.
        """
        # 1. Passive regeneration over dt
        dt = max(0.0, current_time - self.last_update_time)
        self.last_update_time = current_time

        if triple_elixir:
            regen_rate = 1.0 / 0.93  # Triple elixir (overtime)
        elif double_elixir:
            regen_rate = 1.0 / 1.4   # Double elixir (last 60s)
        else:
            regen_rate = 1.0 / 2.8   # Normal single elixir

        potential_elixir = self.enemy_elixir + dt * regen_rate
        if potential_elixir > 10.0:
            self.elixir_leaked += (potential_elixir - 10.0)
            self.enemy_elixir = 10.0
        else:
            self.enemy_elixir = potential_elixir

        # Prune old untracked spawn memory (> 4.5s old)
        self._recent_untracked_spawns = [
            s for s in self._recent_untracked_spawns if (current_time - s[0]) <= 4.5
        ]

        # 2. Check for newly spawned enemy tracks
        for d in enemy_detections:
            if d.is_ally:
                continue
            tid = d.track_id
            cid = int(d.cls_id)
            grid_pos = screen_to_grid((d.x1 + d.x2) / 2.0, d.y2)

            if tid is not None and cid > 0:
                if tid not in self.seen_track_ids:
                    self.seen_track_ids.add(tid)
                    self._register_card_play(cid, current_time, grid_pos=grid_pos, track_id=tid)
                elif tid in self.active_tracks and grid_pos is not None:
                    self.active_tracks[tid]["tile_y"] = grid_pos[0]
                    self.active_tracks[tid]["tile_x"] = grid_pos[1]
            elif tid is None and cid > 0 and grid_pos is not None:
                gy, gx = grid_pos
                # Check if this unit matches a recently seen untracked spawn within 4 tiles
                already_seen = any(
                    (s_cid == cid and abs(gy - s_gy) <= 4 and abs(gx - s_gx) <= 3)
                    for (_, s_gy, s_gx, s_cid) in self._recent_untracked_spawns
                )
                if not already_seen:
                    self._recent_untracked_spawns.append((current_time, gy, gx, cid))
                    self._register_card_play(cid, current_time, grid_pos=grid_pos, track_id=None)

        return float(np.clip(self.enemy_elixir, 0.0, 10.0))

    def get_elixir(self) -> float:
        """Returns current estimated enemy elixir [0.0..10.0]."""
        return float(np.clip(self.enemy_elixir, 0.0, 10.0))

    def get_normalized_elixir(self) -> float:
        """Returns normalized enemy elixir [0.0..1.0] for model scalars[1]."""
        return float(np.clip(self.enemy_elixir / 10.0, 0.0, 10.0))

    def get_cards_played_count(self) -> int:
        """Returns total number of cards the opponent has played in the current match."""
        return len(self.play_history)

    def get_favored_lane(self) -> str:
        """Returns 'left' or 'right' based on opponent's observed placement history."""
        if self.left_lane_plays > self.right_lane_plays:
            return "left"
        return "right"

    def get_card_cycle_distance(self, cid: int) -> int:
        """
        Returns number of card plays remaining until card returns to hand:
        0 = In hand right now
        1 = 1 play away
        2 = 2 plays away
        3 = 3 plays away
        4 = Just played (4 plays away)
        """
        if cid not in self.card_last_play_idx:
            # Never played yet -> in hand or unrevealed draw pile
            return 0
        last_idx = self.card_last_play_idx[cid]
        plays_since = len(self.play_history) - 1 - last_idx
        return max(0, 4 - plays_since)

    def is_card_in_hand(self, cid: int) -> bool:
        """True if the card is estimated to be in the opponent's 4-card hand."""
        return self.get_card_cycle_distance(cid) == 0

    def get_estimated_hand(self) -> List[Dict[str, Any]]:
        """
        Returns cards currently estimated to be in the opponent's active hand.
        """
        in_hand = []
        for cid in self.seen_cards:
            if self.is_card_in_hand(cid):
                cost = CARD_ID_TO_COST.get(cid, 0)
                in_hand.append({
                    "id": cid,
                    "name": CARD_ID_TO_NAME.get(cid, f"Card_{cid}"),
                    "cost": cost,
                    "affordable": bool(cost <= self.enemy_elixir),
                })
        return in_hand

    def get_cycle_queue(self) -> List[Dict[str, Any]]:
        """
        Returns cards currently cycling back to hand with their cycle distance (1..4).
        """
        queue = []
        for cid in self.seen_cards:
            dist = self.get_card_cycle_distance(cid)
            if dist > 0:
                cost = CARD_ID_TO_COST.get(cid, 0)
                queue.append({
                    "id": cid,
                    "name": CARD_ID_TO_NAME.get(cid, f"Card_{cid}"),
                    "cost": cost,
                    "plays_away": dist,
                })
        return sorted(queue, key=lambda x: x["plays_away"])

    def predict_opponent_placement(self, card_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Predicts where the opponent is most likely to place their next card based on
        their card cycle, elixir, card combat role, and observed lane preferences.
        """
        favored_lane = self.get_favored_lane()
        lane_x = 3 if favored_lane == "left" else 14

        if card_id is None:
            next_preds = self.predict_next_card()
            if next_preds:
                card_id = next_preds[0]["id"]
            else:
                card_id = 0

        cid = int(card_id or 0)
        roles = get_card_roles(cid)

        # Look for historical placement of this exact card by the opponent
        prev_for_card = [p for p in self.spatial_play_history if p["cls_id"] == cid]
        if prev_for_card:
            last_p = prev_for_card[-1]
            pred_y, pred_x = int(last_p["tile_y"]), int(last_p["tile_x"])
            zone = f"{last_p['lane']}_custom"
        elif cid in (12, 55, 113, 16):  # Goblin Barrel, Miner, Goblin Drill, Graveyard -> direct on our tower
            pred_y, pred_x = 24, lane_x
            zone = f"ally_princess_{favored_lane}"
        elif "tank" in roles and CARD_ID_TO_COST.get(cid, 4) >= 6:
            # Heavy tank (Golem, Lava Hound, P.E.K.K.A) -> usually started in back when elixir is high
            pred_y, pred_x = (4, 7 if favored_lane == "left" else 10) if self.enemy_elixir >= 7.0 else (14, lane_x)
            zone = f"back_{favored_lane}" if self.enemy_elixir >= 7.0 else f"bridge_{favored_lane}"
        elif "win_condition" in roles or "fast" in roles or cid in (51, 59, 69, 74, 81, 110, 109):
            # Hog Rider, Battle Ram, Bandit, X-Bow, Mortar -> bridge placement
            pred_y, pred_x = 15, lane_x
            zone = f"bridge_{favored_lane}"
        else:
            pred_y, pred_x = 14, lane_x
            zone = f"bridge_{favored_lane}"

        return {
            "card_id": cid,
            "card_name": CARD_ID_TO_NAME.get(cid, "Unknown"),
            "predicted_tile": (pred_y, pred_x),
            "zone": zone,
            "favored_lane": favored_lane,
        }

    def get_counter_hold_flags(self) -> Dict[str, Any]:
        """
        Determines which counters the bot should hold in reserve based on the
        opponent's currently tracked hand and next-up cycle card.
        """
        near_cycle_ids: Set[int] = set()
        for cid in self.seen_cards:
            if self.get_card_cycle_distance(cid) <= 1:
                near_cycle_ids.add(cid)

        meta = self.get_meta_deck()
        if meta.get("confidence", 0.0) >= 0.6:
            for u_cid in meta.get("predicted_unrevealed_cards", [])[:2]:
                near_cycle_ids.add(int(u_cid))

        hold_small_spell = False
        hold_tank_killer = False
        hold_anti_air = False
        avoid_ground_swarm = False

        for cid in near_cycle_ids:
            roles = get_card_roles(cid)
            if "spell_bait" in roles or cid in (12, 31, 44, 64, 49):
                hold_small_spell = True
            if "tank" in roles or "building_targeter" in roles or cid in (24, 51, 37, 25, 78, 108, 53):
                hold_tank_killer = True
            if "air_unit" in roles or cid in (46, 54, 29, 44, 60):
                hold_anti_air = True
            if "splash" in roles or "small_spell" in roles:
                avoid_ground_swarm = True

        return {
            "hold_small_spell": hold_small_spell,
            "hold_tank_killer": hold_tank_killer,
            "hold_anti_air": hold_anti_air,
            "avoid_ground_swarm": avoid_ground_swarm,
            "near_cycle_ids": sorted(near_cycle_ids),
        }

    def predict_next_card(self) -> List[Dict[str, Any]]:
        """
        Predicts which cards the opponent is most likely to play next.
        Ranks by in-hand presence, affordability, and threat weight.
        """
        meta = self.get_meta_deck()
        candidates = []

        # 1. Revealed cards in hand
        for card_info in self.get_estimated_hand():
            cid = card_info["id"]
            cost = card_info["cost"]
            affordable = card_info["affordable"]

            danger = get_intrinsic_danger(cid)
            is_win_con = cid in (24, 51, 12, 37, 25, 46, 110, 109, 108, 93, 113)
            urgency = 2.0 if is_win_con else (1.5 if cost <= 3 else 1.0 + danger * 0.05)
            score = (10.0 if affordable else 2.0) * urgency

            placement = self.predict_opponent_placement(cid)
            candidates.append({
                "id": cid,
                "name": card_info["name"],
                "cost": cost,
                "affordable": affordable,
                "in_hand": True,
                "confidence": round(min(1.0, score / 15.0), 2),
                "is_revealed": True,
                "predicted_tile": placement["predicted_tile"],
                "predicted_zone": placement["zone"],
            })

        # 2. Predicted unrevealed cards from matched meta deck
        for cid in meta.get("predicted_unrevealed_cards", []):
            cost = CARD_ID_TO_COST.get(cid, 3)
            affordable = bool(cost <= self.enemy_elixir)
            placement = self.predict_opponent_placement(cid)
            candidates.append({
                "id": cid,
                "name": CARD_ID_TO_NAME.get(cid, f"Card_{cid}"),
                "cost": cost,
                "affordable": affordable,
                "in_hand": False,  # candidate unrevealed
                "confidence": round(meta.get("confidence", 0.5) * 0.7, 2),
                "is_revealed": False,
                "predicted_tile": placement["predicted_tile"],
                "predicted_zone": placement["zone"],
            })

        return sorted(candidates, key=lambda x: (x["affordable"], x["confidence"]), reverse=True)

    def get_meta_deck(self) -> Dict[str, Any]:
        """Identifies opponent meta deck archetype using revealed cards."""
        return identify_meta_deck(self.seen_cards)

    def get_cycle_summary(self) -> Dict[str, Any]:
        """Returns summary of opponent tracked deck and elixir."""
        meta = self.get_meta_deck()
        return {
            "estimated_elixir": round(self.enemy_elixir, 1),
            "elixir_leaked": round(self.elixir_leaked, 1),
            "cards_played_count": len(self.play_history),
            "seen_cards_count": len(self.seen_cards),
            "seen_card_ids": list(self.seen_cards),
            "seen_card_names": [CARD_ID_TO_NAME.get(cid, f"Card_{cid}") for cid in self.seen_cards],
            "total_elixir_spent": self.total_elixir_spent,
            "active_tracks_count": len(self.active_tracks),
            "favored_lane": self.get_favored_lane(),
            "archetype": meta.get("archetype", "Unknown"),
            "matched_deck": meta.get("matched_deck_name", "Unknown"),
            "confidence": meta.get("confidence", 0.0),
        }

    def get_intel_report(self) -> Dict[str, Any]:
        """Full intelligence dossier for opponent in current battle."""
        meta = self.get_meta_deck()
        return {
            "elixir": round(self.enemy_elixir, 1),
            "elixir_leaked": round(self.elixir_leaked, 1),
            "cards_played_count": len(self.play_history),
            "revealed_count": f"{len(self.seen_cards)}/8",
            "favored_lane": self.get_favored_lane(),
            "seen_cards": [
                {
                    "id": cid,
                    "name": CARD_ID_TO_NAME.get(cid, f"Card_{cid}"),
                    "cost": CARD_ID_TO_COST.get(cid, 0),
                    "cycle_distance": self.get_card_cycle_distance(cid),
                    "in_hand": self.is_card_in_hand(cid),
                }
                for cid in self.seen_cards
            ],
            "estimated_hand": self.get_estimated_hand(),
            "cycle_queue": self.get_cycle_queue(),
            "next_card_predictions": self.predict_next_card()[:4],
            "counter_holds": self.get_counter_hold_flags(),
            "meta_deck": meta,
        }

    def format_console_banner(self) -> str:
        """Generates formatted multi-line ASCII intelligence report."""
        meta = self.get_meta_deck()
        e_hand = [f"{c['name']}({c['cost']})" for c in self.get_estimated_hand()]
        q_cards = [f"{c['name']} ({c['plays_away']} away)" for c in self.get_cycle_queue()]
        preds = [
            f"{p['name']} @ {p.get('predicted_zone', 'bridge')} ({'AFFORDABLE' if p['affordable'] else 'saving'})"
            for p in self.predict_next_card()[:3]
        ]

        lines = [
            f"[OPPONENT INTEL] Elixir: {self.enemy_elixir:.1f}/10.0 (Spent: {self.total_elixir_spent}, Plays: {len(self.play_history)}) | Archetype: {meta['matched_deck_name']} ({int(meta['confidence']*100)}%)",
            f"  -> Revealed ({len(self.seen_cards)}/8): {', '.join([CARD_ID_TO_NAME.get(c, str(c)) for c in self.seen_cards]) if self.seen_cards else 'None'}",
            f"  -> Estimated In Hand: {', '.join(e_hand) if e_hand else 'Waiting for rotation'}",
            f"  -> Cycling Back: {', '.join(q_cards) if q_cards else 'All in rotation'}",
            f"  -> Next Play & Placement Prediction: {', '.join(preds) if preds else 'Unknown'}",
            f"  -> Counter Strategy: {meta.get('counter_strategy', 'None')}",
        ]
        return "\n".join(lines)


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
        arena_grid[Ch.DEPLOYABLE] = get_default_deployable_mask(enemy_left_down, enemy_right_down).astype(np.float32)

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
                    arena_grid[Ch.ALLY_GROUND, gy, gx] += 1.0
                elif cat == "air":
                    arena_grid[Ch.ALLY_AIR, gy, gx] += 1.0
                elif cat == "building":
                    arena_grid[Ch.ALLY_BUILDING, gy, gx] = 1.0
                elif cat == "spell":
                    arena_grid[Ch.SPELL, gy, gx] = 1.0
            else:
                if cat == "ground":
                    arena_grid[Ch.ENEMY_GROUND, gy, gx] += 1.0
                elif cat == "air":
                    arena_grid[Ch.ENEMY_AIR, gy, gx] += 1.0
                elif cat == "building":
                    arena_grid[Ch.ENEMY_BUILDING, gy, gx] = 1.0
                elif cat == "spell":
                    arena_grid[Ch.SPELL, gy, gx] = 1.0

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

        scalars = np.zeros(12, dtype=np.float32)
        scalars[Sc.ALLY_ELIXIR] = ally_elixir_norm
        scalars[Sc.ENEMY_ELIXIR] = enemy_elixir_norm
        scalars[Sc.TIME_LEFT] = time_norm
        scalars[Sc.ALLY_KING] = ak_hp
        scalars[Sc.ALLY_LEFT] = al_hp
        scalars[Sc.ALLY_RIGHT] = ar_hp
        scalars[Sc.ENEMY_KING] = ek_hp
        scalars[Sc.ENEMY_LEFT] = el_hp
        scalars[Sc.ENEMY_RIGHT] = er_hp
        scalars[Sc.LAST_CARD] = last_card_norm
        scalars[Sc.LAST_TILE] = last_tile_norm
        scalars[Sc.DOUBLE_ELIXIR] = double_elixir_norm
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
