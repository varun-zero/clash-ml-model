"""
Human-Like Tactical Policy & Threat Assessment Engine for Clash Royale ("RuleBot").

Replaces rigid, repetitive bot patterns with a situational human decision loop:
1. Multi-Tier Threat Assessment:
   - Evaluates every enemy troop/cluster and flags it as LOW, MID, HIGH, or CRITICAL threat
     based on unit role (win-condition, tank, splash, ranged support, swarm), proximity to
     ally towers, velocity, and backline support synergy.
2. Opponent Memory, Elixir & Card Cycle Awareness:
   - Integrates with OpponentTracker to track opponent elixir, total cards played,
     4-card hand rotation, cycling queue, and predicted next placement tile.
   - Reserves key counters (e.g., Arrows for Goblin Barrel/Swarm, Mini P.E.K.K.A for
     Hog Rider/Giant/Tank, Musketeer/Archers for Balloon/Air) when those threats are
     in the opponent's hand or 1 card away in cycle.
3. Authentic Human Defensive Geometry:
   - Center "Kill Zone" Pulls (rows 20..22, cols 7..10) so both Princess Towers engage melee threats.
   - Opposite-Lane Kiting (row 19, col 8..9) against heavy melee units.
   - Direct Backline Assassination (Knight / Mini P.E.K.K.A dropped directly on top of
     enemy Witch / Musketeer / Wizard / Bomber once they cross the river).
   - Outer Flank Ranged Sniping (cols 1..2 or 15..16) to avoid enemy splash and spells.
4. Strict Spell Discipline:
   - Never casts spells in arena corners or on empty grass.
   - Requires verified positive elixir value (swarms for Arrows; medium clusters or
     troop+tower double value for Fireball) or verified lethal tower finish.
5. Counter-Push Translation, Opposite-Lane Punish & Patient Back-of-King Slow Pushes:
   - Banks elixir patiently when the board is calm instead of dumping solo units at the bridge.
   - Places tanks in front of surviving defenders or ranged support 3 tiles behind advancing tanks.
"""
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from configs.constants import (
    ARENA_ROWS,
    ARENA_COLS,
    TOTAL_TILES,
    WAIT_CARD_ID,
    CARD_ID_TO_COST,
    CARD_ID_TO_IS_SPELL,
    CARD_ID_TO_CATEGORY,
    CARD_ID_TO_NAME,
    CARD_DATABASE,
    screen_to_grid,
)
from configs.card_features import (
    card_targets_air,
    card_targets_ground,
    get_card_roles,
    get_intrinsic_danger,
    score_counter_matchup,
)
from perception.state_encoder import Detection, StateEncoder, OpponentTracker


@dataclass
class ThreatAssessment:
    """Structured threat evaluation for an enemy unit or cluster on the arena."""

    tile_y: int
    tile_x: int
    category: str
    cls_id: int
    card_name: str
    danger_score: float
    threat_level: str  # "LOW", "MID", "HIGH", "CRITICAL"
    cluster_size: int
    cluster_elixir: float
    has_backline_support: bool = False
    backline_tile: Optional[Tuple[int, int]] = None
    backline_cid: int = 0
    vy: float = 0.0
    vx: float = 0.0
    recommended_style: str = "center_kill_zone"


class RuleBotPolicy:
    """
    Human-like tactical decision engine for Clash Royale.
    Assesses board condition, threat levels, elixir differential, and opponent card cycle
    before selecting a card and placement coordinate.
    """
    def __init__(self):
        self.encoder = StateEncoder()
        self.last_tank_time = 0.0
        self.last_tank_lane = 14  # Default right lane x
        self.last_decision_reason: str = "Initializing tactical engine"
        self.last_threat_summary: List[Dict[str, Any]] = []
        self._play_step_counter: int = 0

    @staticmethod
    def is_valid_spell_tile(tile_y: int, tile_x: int) -> bool:
        """
        Returns True if (tile_y, tile_x) is inside the playable combat arena and
        NOT in the dead corners or extreme outer borders of the screen.
        """
        if not (2 <= tile_y <= 29 and 1 <= tile_x <= 16):
            return False
        # Exclude top and bottom corners where troops never walk
        if tile_y <= 4 and (tile_x <= 2 or tile_x >= 15):
            return False
        if tile_y >= 28 and (tile_x <= 2 or tile_x >= 15):
            return False
        return True

    def calculate_predictive_spell_tile(
        self,
        spell_cid: int,
        target_y: int,
        target_x: int,
        detections: Optional[List[Detection]] = None,
    ) -> int:
        """
        Calculates the lead-predicted impact tile for projectile spells (Fireball, Arrows, etc.).
        Compensates for server deploy delay (1.0s) + distance-based flight time from King Tower (28, 9).
        Strictly clamps impact coordinates to valid lane corridors so spells are never thrown into corners.
        """
        king_y, king_x = 28, 9
        dist = float(np.hypot(king_y - target_y, king_x - target_x))
        # Arrows travel faster (~10 tiles/sec) than Fireball (~7.5 tiles/sec)
        proj_speed = 10.0 if spell_cid in (1, 3, 4, 13) else 7.5
        t_flight = 1.0 + (dist / proj_speed)

        # Check if enemy is stationary (already attacking one of our towers or is a building)
        is_at_left_princess = (target_y >= 22 and 1 <= target_x <= 5)
        is_at_right_princess = (target_y >= 22 and 12 <= target_x <= 16)
        is_at_king = (target_y >= 26 and 6 <= target_x <= 11)

        if is_at_left_princess or is_at_right_princess or is_at_king:
            lead_y = 0.0
            lead_x = 0.0
        else:
            measured_vy = None
            measured_vx = None
            if detections:
                for d in detections:
                    if not d.is_ally:
                        d_pos = screen_to_grid((d.x1 + d.x2) / 2.0, d.y2)
                        if d_pos and abs(d_pos[0] - target_y) <= 2 and abs(d_pos[1] - target_x) <= 2:
                            if hasattr(d, "vy") and abs(d.vy) > 0.01:
                                measured_vy = d.vy * 32.0
                                measured_vx = getattr(d, "vx", 0.0) * 18.0
                                break

            if measured_vy is not None and measured_vy > 0.15:
                lead_y = float(np.clip(measured_vy * t_flight, 1.0, 3.2))
            else:
                # Standard troop marching speed (~1.05 tiles/sec downward toward ally side)
                lead_y = float(np.clip(1.05 * t_flight, 1.2, 2.5))

            if measured_vx is not None and abs(measured_vx) > 0.1:
                lead_x = float(np.clip(measured_vx * t_flight, -1.5, 1.5))
            else:
                # Units steer toward the bridge (col 3 or 14) on enemy half, then toward Princess Tower (col 3 or 14)
                target_lane_x = 3 if target_x < 9 else 14
                lead_x = 0.4 * float(np.sign(target_lane_x - target_x))

        pred_y = int(np.clip(round(target_y + lead_y), 3, 27))
        pred_x = int(np.clip(round(target_x + lead_x), 2, 15))
        return pred_y * ARENA_COLS + pred_x

    def extract_board_units(
        self,
        arena_grid: np.ndarray,
        detections: Optional[List[Detection]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extracts structured enemy and ally unit lists from detections or spatial grid channels.
        Each item has keys: y, x, category, cls_id, vy, vx.
        """
        enemy_units: List[Dict[str, Any]] = []
        ally_units: List[Dict[str, Any]] = []

        if detections:
            for d in detections:
                pos = screen_to_grid((d.x1 + d.x2) / 2.0, d.y2)
                if pos is not None:
                    unit_info = {
                        "y": int(pos[0]),
                        "x": int(pos[1]),
                        "category": str(d.category).lower(),
                        "cls_id": int(d.cls_id),
                        "vy": float(getattr(d, "vy", 0.0)),
                        "vx": float(getattr(d, "vx", 0.0)),
                    }
                    if d.is_ally:
                        ally_units.append(unit_info)
                    else:
                        enemy_units.append(unit_info)

        # If detections were empty, fall back to the latest frame in arena_grid
        if not enemy_units and not ally_units and arena_grid is not None:
            latest_grid = arena_grid[-8:] if (arena_grid.ndim == 3 and arena_grid.shape[0] >= 8) else (
                arena_grid[0, -8:] if arena_grid.ndim == 4 else arena_grid
            )
            ch_eground = latest_grid[3]
            ch_eair = latest_grid[4]
            ch_ebuild = latest_grid[5]
            for y, x in zip(*np.nonzero(ch_eground > 0)):
                count = int(max(1, round(float(ch_eground[y, x]))))
                for _ in range(count):
                    enemy_units.append({"y": int(y), "x": int(x), "category": "ground", "cls_id": 0, "vy": 0.0, "vx": 0.0})
            for y, x in zip(*np.nonzero(ch_eair > 0)):
                count = int(max(1, round(float(ch_eair[y, x]))))
                for _ in range(count):
                    enemy_units.append({"y": int(y), "x": int(x), "category": "air", "cls_id": 0, "vy": 0.0, "vx": 0.0})
            for y, x in zip(*np.nonzero(ch_ebuild > 0)):
                enemy_units.append({"y": int(y), "x": int(x), "category": "building", "cls_id": 0, "vy": 0.0, "vx": 0.0})

            ch_aground = latest_grid[0]
            ch_aair = latest_grid[1]
            for y, x in zip(*np.nonzero(ch_aground > 0)):
                ally_units.append({"y": int(y), "x": int(x), "category": "ground", "cls_id": 0, "vy": 0.0, "vx": 0.0})
            for y, x in zip(*np.nonzero(ch_aair > 0)):
                ally_units.append({"y": int(y), "x": int(x), "category": "air", "cls_id": 0, "vy": 0.0, "vx": 0.0})

        return enemy_units, ally_units

    def assess_threats(
        self,
        enemy_units: List[Dict[str, Any]],
        tower_hps: List[float],
    ) -> List[ThreatAssessment]:
        """
        Evaluates all enemy units on the board, computes their composite danger score,
        flags each as LOW / MID / HIGH / CRITICAL, detects tank+support combos,
        and recommends a human defensive placement style.
        """
        assessments: List[ThreatAssessment] = []
        if not enemy_units:
            self.last_threat_summary = []
            return assessments

        ally_left_hp = tower_hps[1] if len(tower_hps) >= 3 else 1.0
        ally_right_hp = tower_hps[2] if len(tower_hps) >= 3 else 1.0

        for u in enemy_units:
            uy, ux = u["y"], u["x"]
            ucat = u["category"]
            ucid = u["cls_id"]
            cname = CARD_ID_TO_NAME.get(ucid, f"Enemy_{ucat.capitalize()}") if ucid > 0 else f"Enemy_{ucat.capitalize()}"
            roles = get_card_roles(ucid)

            # 1. Intrinsic card danger (1.5 .. 10.0)
            base_danger = get_intrinsic_danger(ucid, ucat)

            # 2. Cluster analysis within 3 tiles
            neighbors = [
                n for n in enemy_units
                if abs(n["y"] - uy) <= 3 and abs(n["x"] - ux) <= 3
            ]
            cluster_size = len(neighbors)
            cluster_elixir = sum(
                float(CARD_ID_TO_COST.get(n["cls_id"], 3)) if n["cls_id"] > 0 else 2.5
                for n in neighbors
            )
            # Discount duplicate elixir sum for natural multi-unit swarms
            if cluster_size >= 3:
                cluster_elixir = min(cluster_elixir, 4.0 + (cluster_size - 2) * 1.2)

            # 3. Check for Tank + Backline Support synergy (e.g. Giant at row 18 with Witch/Musketeer at row 15)
            has_backline = False
            backline_tile = None
            backline_cid = 0
            if ("tank" in roles or "mini_tank" in roles or ucid in (24, 37, 25, 20, 78)):
                for n in enemy_units:
                    if n is u:
                        continue
                    if 1 <= (uy - n["y"]) <= 6 and abs(n["x"] - ux) <= 4:
                        n_roles = get_card_roles(n["cls_id"])
                        if "ranged_support" in n_roles or "splash" in n_roles or n["cls_id"] in (27, 32, 115, 34, 21):
                            has_backline = True
                            backline_tile = (n["y"], n["x"])
                            backline_cid = n["cls_id"]
                            break

            # 4. Spatial proximity multiplier
            if uy >= 22:
                # Directly threatening / locked onto our Princess or King Tower
                prox_mult = 1.85
            elif uy >= 16:
                # Crossed river onto ally territory
                prox_mult = 1.40 + (uy - 16) * 0.07
            elif uy >= 12:
                # Approaching bridge / river
                prox_mult = 1.05
            else:
                # Deep on enemy backfield
                prox_mult = 0.55

            # Boost urgency if attacking a damaged ally tower
            lane_tower_hp = ally_left_hp if ux < 9 else ally_right_hp
            tower_vuln_bonus = 1.5 if lane_tower_hp < 0.40 else (0.7 if lane_tower_hp < 0.70 else 0.0)
            cluster_bonus = min(3.0, (cluster_size - 1) * 0.85) + (1.5 if has_backline else 0.0)

            danger_score = float(base_danger * prox_mult + cluster_bonus + tower_vuln_bonus)

            if danger_score >= 10.0 or (uy >= 22 and base_danger >= 4.5):
                threat_level = "CRITICAL"
            elif danger_score >= 7.0 or (uy >= 16 and base_danger >= 4.5):
                threat_level = "HIGH"
            elif danger_score >= 4.0 or uy >= 15:
                threat_level = "MID"
            else:
                threat_level = "LOW"

            # 5. Recommend authentic human defensive placement style
            if has_backline and backline_tile is not None and backline_tile[0] >= 16:
                style = "backline_assassinate"
            elif ucat == "air" or "air_unit" in roles:
                style = "outer_flank_snipe"
            elif "ranged_support" in roles or "splash" in roles:
                style = "direct_surround" if uy >= 16 else "center_kill_zone"
            elif "building_targeter" in roles or ucid in (24, 51, 37, 46, 59):
                style = "direct_intercept"
            elif ("tank_killer" in roles or "melee" in roles) and uy <= 20:
                style = "opposite_lane_kite" if danger_score >= 8.0 else "center_kill_zone"
            else:
                style = "center_kill_zone"

            assessments.append(
                ThreatAssessment(
                    tile_y=uy,
                    tile_x=ux,
                    category=ucat,
                    cls_id=ucid,
                    card_name=cname,
                    danger_score=round(danger_score, 2),
                    threat_level=threat_level,
                    cluster_size=cluster_size,
                    cluster_elixir=round(cluster_elixir, 1),
                    has_backline_support=has_backline,
                    backline_tile=backline_tile,
                    backline_cid=backline_cid,
                    vy=u["vy"],
                    vx=u["vx"],
                    recommended_style=style,
                )
            )

        # Sort highest danger first
        assessments.sort(key=lambda t: (t.danger_score, t.tile_y), reverse=True)
        self.last_threat_summary = [
            {
                "card": a.card_name,
                "pos": (a.tile_y, a.tile_x),
                "level": a.threat_level,
                "score": a.danger_score,
                "style": a.recommended_style,
            }
            for a in assessments[:4]
        ]
        return assessments

    def _apply_anti_clump_spacing(
        self,
        def_y: int,
        def_x: int,
        ally_units: List[Dict[str, Any]],
        is_left_lane: bool,
    ) -> Tuple[int, int]:
        """
        Prevents stacking ranged defenders right on top of existing ally units or damaged towers,
        denying the opponent 2-for-1 Fireball / Poison / Rocket value.
        """
        if not ally_units:
            return def_y, def_x

        too_close = any(
            np.hypot(a["y"] - def_y, a["x"] - def_x) <= 2.2
            for a in ally_units
        )
        if not too_close:
            return def_y, def_x

        # Try shifting 2-3 tiles toward the opposite side of the lane
        candidate_offsets = [(0, 3), (0, -3), (1, 2), (1, -2), (-1, 3), (-1, -3)]
        best_pos = (def_y, def_x)
        best_min_dist = 0.0
        for dy, dx in candidate_offsets:
            ny = int(np.clip(def_y + dy, 17, 27))
            nx = int(np.clip(def_x + dx, 1, 16))
            min_dist = min(np.hypot(a["y"] - ny, a["x"] - nx) for a in ally_units)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = (ny, nx)
                if min_dist >= 2.8:
                    break
        return best_pos

    def _compute_human_defensive_tile(
        self,
        defender_cid: int,
        threat: ThreatAssessment,
        tower_hps: List[float],
        ally_units: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Computes human-like defensive placement coordinates tailored to the defender's role
        and the threat's attack profile (4-3 / 4-2 Building Plant, Kill Zone pull,
        Opposite-Lane Kite, Outer Snipe, Backline Assassination, or Direct Intercept).
        """
        ty, tx = threat.tile_y, threat.tile_x
        def_roles = get_card_roles(defender_cid)
        enemy_roles = get_card_roles(threat.cls_id)
        is_left_lane = tx < 9

        # Subtle deterministic micro-offset (0 or 1 tile) so placements adapt organically
        micro_shift = (self._play_step_counter % 2)

        # Case 1: Backline Assassination (drop melee unit right on top of enemy support behind tank)
        if (
            threat.recommended_style == "backline_assassinate"
            and threat.backline_tile is not None
            and ("mini_tank" in def_roles or "tank_killer" in def_roles or defender_cid in (20, 26, 33))
        ):
            by, bx = threat.backline_tile
            def_y = int(np.clip(by + 1, 16, 26))
            def_x = int(np.clip(bx, 2, 15))

        # Case 2: Defensive Building ("4-3 Standard Pull" vs "4-2 Anti-Pig-Push Plant")
        elif "defensive_building" in def_roles or "building" in def_roles:
            is_fast_bypass = (
                "fast_bypass_win_condition" in enemy_roles
                or threat.cls_id in (51, 59, 74, 82)
            )
            if ty >= 22:
                # Threat is already at tower: plant directly in front of tower to body-block
                def_y = 22
                def_x = 5 if is_left_lane else 12
            elif is_fast_bypass and (tx <= 3 or tx >= 14):
                # "4-2 Anti-Bypass Plant" (2 tiles south of river, slightly same-lane biased)
                # Prevents outer-tile pig-pushed Hog Rider / Ram from bypassing the building
                def_y = 18
                def_x = 7 if is_left_lane else 10
            else:
                # Classic "4-3 Standard Pull" (3 tiles south of river, 1 tile into opposite lane)
                # Maximizes tank walking distance, engages BOTH Princess Towers, and avoids Fireball tower splash
                def_y = 19
                def_x = 9 if is_left_lane else 8

        # Case 3: Ranged Defender (Musketeer, Archers, Spear Goblins, Bomber, Wizard, Ice/Electro Wizard)
        elif "ranged_support" in def_roles or defender_cid in (21, 23, 27, 34):
            if threat.category == "air" or threat.recommended_style == "outer_flank_snipe":
                # Place on safe anti-splash flank or cross-lane sniping position
                if ty >= 21:
                    def_y = int(np.clip(23 + micro_shift, 21, 26))
                    def_x = 6 if is_left_lane else 11
                else:
                    def_y = int(np.clip(ty + 4, 20, 25))
                    def_x = (6 + micro_shift) if is_left_lane else (11 - micro_shift)
            else:
                # Ground threat: keep ranged unit 4-5 tiles away in the center kill-zone corridor
                def_y = int(np.clip(max(20, ty + 3), 20, 25))
                def_x = (7 if is_left_lane else 10)

            if ally_units:
                def_y, def_x = self._apply_anti_clump_spacing(def_y, def_x, ally_units, is_left_lane)

        # Case 4: Direct Surround / On-Top Drop against Enemy Ranged / Glass Cannon on Ally Half
        elif threat.recommended_style == "direct_surround" and ty >= 16:
            def_y = int(np.clip(ty + 1, 16, 27))
            def_x = int(np.clip(tx, 2, 15))

        # Case 5: Direct Intercept against Building-Targeting Win Conditions (Giant, Hog, Goblin Barrel)
        elif threat.recommended_style == "direct_intercept":
            if threat.cls_id == 12:  # Goblin Barrel on tower -> drop directly on tower
                def_y = int(np.clip(ty, 24, 26))
                def_x = 3 if is_left_lane else 14
            elif ty >= 23:
                # Already on our tower: place right next to tower
                def_y = int(np.clip(ty, 22, 26))
                def_x = int(np.clip(tx, 2, 15))
            else:
                # Intercept directly in its lane path 2 tiles ahead
                def_y = int(np.clip(max(17, ty + 2), 17, 23))
                lane_center_x = 3 if is_left_lane else 14
                def_x = int(np.clip(round(0.6 * tx + 0.4 * lane_center_x), 2, 15))

        # Case 6: Opposite-Lane Kiting for Heavy Melee Threats (P.E.K.K.A, Prince, Mega Knight, Mini P.E.K.K.A)
        elif (threat.recommended_style == "opposite_lane_kite" or defender_cid == 61) and 16 <= ty <= 20:
            def_y = 19 + micro_shift
            # Place 1 tile across the center line so the melee threat walks across both tower ranges
            def_x = 9 if is_left_lane else 8

        # Case 7: Standard Center "Kill Zone" Pull (drags enemy toward middle of ally arena)
        else:
            if ty >= 22:
                # Threat is already close to our Princess Tower: block between threat and tower
                def_y = int(np.clip(ty + 1, 22, 26))
                def_x = int(np.clip(tx + (1 if is_left_lane else -1), 2, 15))
            else:
                # Pull toward center kill-zone (rows 20..21, cols 7..10)
                def_y = int(np.clip(max(19, ty + 2), 19, 22))
                def_x = (7 + micro_shift) if is_left_lane else (10 - micro_shift)

        tile_mask = self.encoder.get_legal_tile_mask(defender_cid, tower_hps=tower_hps)
        tile_idx = def_y * ARENA_COLS + def_x
        if not tile_mask[tile_idx]:
            legal_indices = np.where(tile_mask)[0]
            if len(legal_indices) > 0:
                tile_idx = int(legal_indices[np.argmin(np.abs(legal_indices - tile_idx))])
        return tile_idx

    def _evaluate_spell_opportunity(
        self,
        spell_slots: List[Tuple[int, int, int, bool]],
        threats: List[ThreatAssessment],
        enemy_units: List[Dict[str, Any]],
        ally_units: List[Dict[str, Any]],
        tower_hps: List[float],
        elapsed_sec: float,
        time_left: float = 180.0,
        elixir: float = 5.0,
        hold_small_spell: bool = False,
        detections: Optional[List[Detection]] = None,
    ) -> Optional[Tuple[int, int, float, str]]:
        """
        Evaluates whether any spell or global-deploy win condition in hand achieves high human-grade value.
        Returns (slot_1_based, tile_idx, confidence, reason) or None.
        Never casts spells on empty tiles, corners, or negative-value solo tanks.
        """
        if not spell_slots:
            return None

        enemy_king_hp = tower_hps[3]
        enemy_left_hp = tower_hps[4]
        enemy_right_hp = tower_hps[5]

        # 1. Verified Lethal Tower Finisher / Late-Game Tiebreaker Chip
        if elapsed_sec >= 35.0:
            is_urgent_endgame = time_left <= 28.0
            for s_idx, s_cid, s_cost, _ in spell_slots:
                # Miner (55), Goblin Drill (113), Goblin Barrel (12), Graveyard (16) are global win conditions
                if s_cid in (55, 113, 12, 16):
                    continue
                if s_cid in (2, 5, 6, 7):  # Big spells (Fireball, Lightning, Rocket, Poison)
                    crit_threshold = 0.16 if is_urgent_endgame else 0.12
                else:
                    crit_threshold = 0.08 if is_urgent_endgame else 0.05

                if 0.02 < enemy_left_hp <= crit_threshold:
                    tile = 7 * ARENA_COLS + 3
                    return s_idx + 1, tile, 0.98, f"[SPELL FINISHER] {CARD_ID_TO_NAME.get(s_cid)} on Left Princess Tower (HP: {enemy_left_hp*100:.0f}%)"
                if 0.02 < enemy_right_hp <= crit_threshold:
                    tile = 7 * ARENA_COLS + 14
                    return s_idx + 1, tile, 0.98, f"[SPELL FINISHER] {CARD_ID_TO_NAME.get(s_cid)} on Right Princess Tower (HP: {enemy_right_hp*100:.0f}%)"
                if (enemy_left_hp <= 0.05 or enemy_right_hp <= 0.05) and (0.02 < enemy_king_hp <= crit_threshold):
                    tile = 4 * ARENA_COLS + 9
                    return s_idx + 1, tile, 0.98, f"[SPELL FINISHER] {CARD_ID_TO_NAME.get(s_cid)} on Enemy King Tower (HP: {enemy_king_hp*100:.0f}%)"

        # 2. Evaluate Global Win Conditions (Goblin Barrel, Miner, Goblin Drill, Graveyard)
        for s_idx, s_cid, s_cost, _ in spell_slots:
            if s_cid in (12, 55, 113, 16):
                sname = CARD_ID_TO_NAME.get(s_cid, "Global Win-Con")
                has_tank_pushing = any(12 <= a["y"] <= 19 for a in ally_units)
                calm_board = not any(t.threat_level in ("CRITICAL", "HIGH") for t in threats)
                if has_tank_pushing or (calm_board and elixir >= 8.0) or (calm_board and time_left <= 25.0 and elixir >= 5.0):
                    target_x = 3 if (enemy_left_hp > 0.05 and (enemy_left_hp <= enemy_right_hp or enemy_right_hp <= 0.05)) else 14
                    # Vary Miner/Drill placement slightly around tower (row 6 or 7) so opponent cannot blind-predict
                    target_y = 6 if (self._play_step_counter % 2 == 0) else 7
                    tile = target_y * ARENA_COLS + target_x
                    reason = f"[GLOBAL WIN-CON] {sname} onto Enemy Princess Tower ({target_y}, {target_x})"
                    return s_idx + 1, tile, 0.94, reason

        if not enemy_units:
            return None

        # 3. Evaluate each damage spell against enemy clusters & high-value targets
        best_play: Optional[Tuple[float, int, int, str]] = None  # (value_score, slot_1_based, tile, reason)

        for s_idx, s_cid, s_cost, _ in spell_slots:
            if s_cid in (12, 55, 113, 16):
                continue
            sname = CARD_ID_TO_NAME.get(s_cid, "Spell")
            roles = get_card_roles(s_cid)
            blast_radius = 3.5 if ("small_spell" in roles or s_cid == 1) else 2.6

            for u in enemy_units:
                # Ignore targets sitting in invalid corner tiles
                if not self.is_valid_spell_tile(u["y"], u["x"]):
                    continue

                in_blast = [
                    n for n in enemy_units
                    if np.hypot(n["y"] - u["y"], n["x"] - u["x"]) <= blast_radius
                ]
                count = len(in_blast)
                avg_y = int(round(np.mean([n["y"] for n in in_blast])))
                avg_x = int(round(np.mean([n["x"] for n in in_blast])))

                # Estimate elixir value of units inside blast radius
                swarm_hits = sum(
                    1 for n in in_blast
                    if n["cls_id"] in (12, 31, 44, 64, 49, 63, 87, 43, 72, 22, 23, 30, 81)
                    or "swarm" in get_card_roles(n["cls_id"])
                )
                support_hits = sum(
                    1 for n in in_blast
                    if n["cls_id"] in (27, 28, 32, 115, 34, 21, 80, 85, 65, 50, 75, 36)
                    or "ranged_support" in get_card_roles(n["cls_id"])
                )
                only_heavy_tanks = all(
                    n["cls_id"] in (24, 37, 25, 78, 54, 93, 108) or "tank" in get_card_roles(n["cls_id"])
                    for n in in_blast
                ) and count == 1

                # NEVER cast damage spells on a solo full-health tank!
                if only_heavy_tanks:
                    continue

                if s_cid == 1 or "small_spell" in roles:
                    # Arrows / Small Spell: requires either a high-priority swarm card (Barrel/Skarmy/Horde/Princess)
                    # OR >= 3 enemy units clustered together, OR >= 2 units with at least 1 swarm/glass cannon
                    is_priority_bait = any(n["cls_id"] in (12, 31, 44, 64, 49) for n in in_blast)
                    if is_priority_bait:
                        val = 10.5
                    elif count >= 3:
                        val = 7.5 + count * 0.9 - (2.0 if hold_small_spell else 0.0)
                    elif count >= 2 and swarm_hits >= 1 and avg_y >= 14:
                        val = 6.2 - (2.5 if hold_small_spell else 0.0)
                    else:
                        continue

                else:
                    # Fireball / Big Spell (cost >= 4):
                    # Requires >= 3 clustered units, OR >= 2 units including ranged/support/medium troops,
                    # OR hitting a support unit right next to an enemy Princess Tower (Tower + Troop double value!)
                    near_enemy_left_tower = (enemy_left_hp > 0.05 and abs(avg_y - 7) <= 3 and abs(avg_x - 3) <= 3)
                    near_enemy_right_tower = (enemy_right_hp > 0.05 and abs(avg_y - 7) <= 3 and abs(avg_x - 14) <= 3)
                    tower_bonus = 2.5 if (near_enemy_left_tower or near_enemy_right_tower) else 0.0

                    if count >= 3:
                        val = 8.5 + count * 0.8 + tower_bonus
                    elif count >= 2 and (support_hits >= 1 or avg_y >= 15):
                        val = 7.2 + support_hits * 1.2 + tower_bonus
                    elif count == 1 and support_hits >= 1 and tower_bonus > 0:
                        # Single high-value support (Musketeer/Witch/Wizard) walking past enemy Princess Tower!
                        val = 7.4
                    else:
                        continue

                if val >= 6.0:
                    lead_tile = self.calculate_predictive_spell_tile(
                        spell_cid=s_cid,
                        target_y=avg_y,
                        target_x=avg_x,
                        detections=detections,
                    )
                    ly, lx = lead_tile // ARENA_COLS, lead_tile % ARENA_COLS
                    if not self.is_valid_spell_tile(ly, lx):
                        continue
                    reason = (
                        f"[SPELL VALUE] {sname} -> Tile ({ly:02d}, {lx:02d}) hitting {count} enemy unit(s) "
                        f"(Value score: {val:.1f})"
                    )
                    if best_play is None or val > best_play[0]:
                        best_play = (val, s_idx + 1, lead_tile, reason)

        if best_play is not None:
            _, chosen_slot, chosen_tile, reason = best_play
            return chosen_slot, chosen_tile, 0.94, reason

        return None

    def predict_action(
        self,
        arena_grid: np.ndarray,
        scalars: np.ndarray,
        hand_card_ids: List[int],
        legal_card_mask: np.ndarray,
        legal_tile_mask: Optional[np.ndarray] = None,
        detections: Optional[List[Detection]] = None,
        opponent_tracker: Optional[OpponentTracker] = None,
    ) -> Tuple[int, int, float, float]:
        """
        Computes (chosen_slot, chosen_tile, card_conf, tile_conf) using human-like tactical reasoning:
        chosen_slot: 0 = Wait, 1..4 = Hand Slot 0..3
        chosen_tile: 0..575
        """
        self._play_step_counter += 1

        # 1. Parse scalar state & match tempo (Single vs Double/Triple Elixir)
        elixir = float(scalars[0] * 10.0) if (scalars is not None and len(scalars) > 0) else 0.0
        enemy_elixir = (
            opponent_tracker.get_elixir()
            if opponent_tracker is not None
            else (float(scalars[1] * 10.0) if (scalars is not None and len(scalars) > 1) else 5.0)
        )
        time_left = float(scalars[2] * 180.0) if (scalars is not None and len(scalars) > 2) else 180.0
        elapsed_sec = max(0.0, 180.0 - time_left)
        is_double_elixir = (
            (bool(scalars[11] >= 0.5) if (scalars is not None and len(scalars) > 11) else False)
            or time_left <= 60.0
        )

        tower_hps = list(scalars[3:9]) if (scalars is not None and len(scalars) >= 9) else [1.0] * 6
        ally_left_hp = tower_hps[1]
        ally_right_hp = tower_hps[2]
        enemy_left_hp = tower_hps[4]
        enemy_right_hp = tower_hps[5]
        enemy_left_down = enemy_left_hp <= 0.05
        enemy_right_down = enemy_right_hp <= 0.05

        # 2. Gather affordable cards in hand
        affordable_slots: List[Tuple[int, int, int, bool]] = []
        for slot_idx in range(4):
            if slot_idx < len(hand_card_ids):
                cid = int(hand_card_ids[slot_idx])
                if cid > 0 and legal_card_mask[slot_idx + 1]:
                    cost = CARD_ID_TO_COST.get(cid, 99)
                    is_spell = bool(CARD_ID_TO_IS_SPELL.get(cid, False) and CARD_ID_TO_CATEGORY.get(cid) == "spell")
                    affordable_slots.append((slot_idx, cid, cost, is_spell))

        if not affordable_slots:
            self.last_decision_reason = f"[WAIT] No affordable cards in hand (Elixir: {elixir:.1f})"
            return 0, 0, 1.0, 1.0

        # 3. Extract Opponent Intelligence & Counter Reservation Flags
        hold_flags = (
            opponent_tracker.get_counter_hold_flags()
            if opponent_tracker is not None
            else {"hold_small_spell": False, "hold_tank_killer": False, "hold_anti_air": False, "avoid_ground_swarm": False}
        )
        opp_cards_played = opponent_tracker.get_cards_played_count() if opponent_tracker is not None else 0

        # 4. Extract Board Units & Assess Threats (LOW / MID / HIGH / CRITICAL)
        enemy_units, ally_units = self.extract_board_units(arena_grid, detections)
        threats = self.assess_threats(enemy_units, tower_hps)

        # Choose primary target lane (weakest standing enemy tower)
        if enemy_left_down and not enemy_right_down:
            target_lane_x = 14
        elif enemy_right_down and not enemy_left_down:
            target_lane_x = 3
        elif enemy_left_hp < enemy_right_hp:
            target_lane_x = 3
        else:
            target_lane_x = 14

        spell_slots = [s for s in affordable_slots if s[3] or s[1] in (12, 55, 113, 16)]
        troop_slots = [s for s in affordable_slots if not s[3] and s[1] not in (12, 55, 113, 16)]

        # ======================================================================
        # PRIORITY 1: High-Value Spell / Swarm Cleanse / Lethal Tower Finisher
        # ======================================================================
        spell_play = self._evaluate_spell_opportunity(
            spell_slots=spell_slots,
            threats=threats,
            enemy_units=enemy_units,
            ally_units=ally_units,
            tower_hps=tower_hps,
            elapsed_sec=elapsed_sec,
            time_left=time_left,
            elixir=elixir,
            hold_small_spell=hold_flags["hold_small_spell"],
            detections=detections,
        )
        if spell_play is not None:
            s_slot, s_tile, s_conf, s_reason = spell_play
            self.last_decision_reason = s_reason
            return s_slot, s_tile, s_conf, s_conf

        # ======================================================================
        # PRIORITY 2: Threat-Prioritized Defense (CRITICAL / HIGH / MID Threats)
        # ======================================================================
        active_threats = [
            t for t in threats
            if t.tile_y >= 15 or (t.threat_level in ("CRITICAL", "HIGH") and t.tile_y >= 13)
        ]
        if active_threats:
            primary_threat = active_threats[0]

            # Check if we already have sufficient ally defenders engaged near this threat
            nearby_allies = [
                a for a in ally_units
                if abs(a["y"] - primary_threat.tile_y) <= 5 and abs(a["x"] - primary_threat.tile_x) <= 5
            ]
            # Avoid overcommitting on a LOW/MID threat if 2+ ally troops are already defending it
            if primary_threat.threat_level in ("LOW", "MID") and len(nearby_allies) >= 2 and elixir < 9.0:
                self.last_decision_reason = (
                    f"[HOLD] Ally defenders already engaging {primary_threat.card_name} "
                    f"({primary_threat.threat_level}) — conserving elixir ({elixir:.1f})"
                )
                return 0, 0, 1.0, 1.0

            scored_defenders: List[Tuple[float, int, int, bool]] = []
            for slot_idx, cid, cost, is_spell in affordable_slots:
                matchup_score = score_counter_matchup(
                    defender_cid=cid,
                    enemy_cid=primary_threat.cls_id,
                    enemy_category=primary_threat.category,
                    cluster_size=primary_threat.cluster_size,
                    cluster_elixir=primary_threat.cluster_elixir,
                )

                # If enemy tank has a backline support (e.g. Giant + Witch/Wizard/Musketeer) on our side,
                # strongly favor melee assassins (Knight, Mini P.E.K.K.A, Valkyrie) to drop directly onto the backline!
                if primary_threat.has_backline_support and primary_threat.backline_tile is not None:
                    c_roles = get_card_roles(cid)
                    if ("mini_tank" in c_roles or "tank_killer" in c_roles) and cid in (20, 26, 33, 52, 53):
                        matchup_score += 3.5

                # Apply opponent cycle reservation penalties for non-critical threats
                if primary_threat.threat_level not in ("CRITICAL", "HIGH"):
                    if cid == 1 and hold_flags["hold_small_spell"]:
                        matchup_score -= 4.0
                    if cid == 26 and hold_flags["hold_tank_killer"]:
                        matchup_score -= 3.0
                    if cid in (27, 21) and hold_flags["hold_anti_air"] and primary_threat.category != "air":
                        matchup_score -= 2.0

                if matchup_score > -15.0:
                    scored_defenders.append((matchup_score, slot_idx, cid, is_spell))

            if scored_defenders:
                scored_defenders.sort(key=lambda x: x[0], reverse=True)
                best_score, best_slot, best_cid, best_is_spell = scored_defenders[0]

                if best_is_spell:
                    tile_idx = self.calculate_predictive_spell_tile(
                        spell_cid=best_cid,
                        target_y=primary_threat.tile_y,
                        target_x=primary_threat.tile_x,
                        detections=detections,
                    )
                else:
                    tile_idx = self._compute_human_defensive_tile(
                        defender_cid=best_cid,
                        threat=primary_threat,
                        tower_hps=tower_hps,
                        ally_units=ally_units,
                    )

                ty_out, tx_out = tile_idx // ARENA_COLS, tile_idx % ARENA_COLS
                self.last_decision_reason = (
                    f"[DEFEND {primary_threat.threat_level} THREAT] {primary_threat.card_name} at "
                    f"({primary_threat.tile_y},{primary_threat.tile_x}) -> Counter with "
                    f"{CARD_ID_TO_NAME.get(best_cid)} at ({ty_out},{tx_out}) [{primary_threat.recommended_style}]"
                )
                return best_slot + 1, tile_idx, 0.96, 0.96

        # ======================================================================
        # PRIORITY 3: Counter-Push Synergy (Support Surviving Defenders / Tanks)
        # ======================================================================
        if troop_slots and ally_units and elixir >= 4.0:
            # Find furthest advancing ally unit on our half or crossing bridge
            ally_on_field = [a for a in ally_units if 12 <= a["y"] <= 27]
            if ally_on_field:
                lead_ally = min(ally_on_field, key=lambda a: a["y"])  # lowest y = closest to enemy side
                ay, ax = lead_ally["y"], lead_ally["x"]
                ally_lane_x = 3 if ax < 9 else 14
                lead_roles = get_card_roles(lead_ally["cls_id"])

                # 3A: We have a Tank / Mini-Tank advancing -> place ranged / splash support 3 tiles behind it!
                if ("tank" in lead_roles or "mini_tank" in lead_roles or lead_ally["cls_id"] in (24, 20, 37, 25)) and 14 <= ay <= 25:
                    support_candidates = [
                        s for s in troop_slots
                        if ("ranged_support" in get_card_roles(s[1]) or "splash" in get_card_roles(s[1]) or s[1] in (27, 21, 34, 43, 23))
                    ]
                    if support_candidates:
                        # Prefer Musketeer (27), Bomber (34), Archers (21), Minions (43)
                        support_candidates.sort(key=lambda s: (s[1] in (27, 34, 21), s[2]), reverse=True)
                        s_slot, s_cid, _, _ = support_candidates[0]
                        sup_y = int(np.clip(ay + 3, 17, 28))
                        sup_x = int(np.clip(ax + (1 if ax < 9 else -1), 2, 15))
                        sup_y, sup_x = self._apply_anti_clump_spacing(sup_y, sup_x, ally_units, ax < 9)
                        tile_mask = self.encoder.get_legal_tile_mask(s_cid, tower_hps=tower_hps)
                        sup_tile = sup_y * ARENA_COLS + sup_x
                        if not tile_mask[sup_tile]:
                            legal_indices = np.where(tile_mask)[0]
                            sup_tile = int(legal_indices[np.argmin(np.abs(legal_indices - sup_tile))])
                        self.last_decision_reason = (
                            f"[COUNTER-PUSH SUPPORT] Supporting ally tank at ({ay},{ax}) with "
                            f"{CARD_ID_TO_NAME.get(s_cid)} behind at ({sup_tile // ARENA_COLS},{sup_tile % ARENA_COLS})"
                        )
                        return s_slot + 1, sup_tile, 0.93, 0.93

                # 3B: We have surviving Ranged Defenders approaching river (rows 17..22) -> drop Tank in front at bridge!
                if 17 <= ay <= 22 and elixir >= 5.0:
                    front_tanks = [
                        s for s in troop_slots
                        if ("tank" in get_card_roles(s[1]) or s[1] in (24, 20))
                    ]
                    if front_tanks:
                        front_tanks.sort(key=lambda s: s[2], reverse=True)  # Prefer Giant (24) over Knight (20)
                        t_slot, t_cid, _, _ = front_tanks[0]
                        self.last_tank_lane = ally_lane_x
                        bridge_y = 17
                        bridge_tile = bridge_y * ARENA_COLS + ally_lane_x
                        tile_mask = self.encoder.get_legal_tile_mask(t_cid, tower_hps=tower_hps)
                        if not tile_mask[bridge_tile]:
                            legal_indices = np.where(tile_mask)[0]
                            bridge_tile = int(legal_indices[np.argmin(np.abs(legal_indices - bridge_tile))])
                        self.last_decision_reason = (
                            f"[TANK IN FRONT] Placing {CARD_ID_TO_NAME.get(t_cid)} at bridge ({bridge_y},{ally_lane_x}) "
                            f"to tank for surviving defender at ({ay},{ax})"
                        )
                        return t_slot + 1, bridge_tile, 0.94, 0.94

        # ======================================================================
        # PRIORITY 4: Opposite-Lane Punish When Opponent Overcommits in Back
        # ======================================================================
        backfield_enemies = [t for t in threats if t.tile_y <= 11 and t.danger_score >= 4.0]
        elixir_advantage = elixir - enemy_elixir
        if troop_slots and elixir >= 6.0 and (backfield_enemies or (enemy_elixir <= 2.2 and elixir_advantage >= 4.0)):
            if backfield_enemies:
                enemy_lane_left = backfield_enemies[0].tile_x < 9
                punish_lane_x = 14 if enemy_lane_left else 3
            else:
                punish_lane_x = target_lane_x

            # Pick fast pressure or win-condition troop (Giant, Hog Rider, Mini P.E.K.K.A, Knight, Minions)
            punish_candidates = [
                s for s in troop_slots
                if s[1] in (24, 51, 26, 20, 43, 22) or "win_condition" in get_card_roles(s[1]) or "fast" in get_card_roles(s[1])
            ]
            if not punish_candidates:
                punish_candidates = troop_slots

            punish_candidates.sort(key=lambda s: (s[1] == 24, s[2]), reverse=True)
            p_slot, p_cid, _, _ = punish_candidates[0]
            self.last_tank_lane = punish_lane_x
            punish_y = 17
            punish_tile = punish_y * ARENA_COLS + punish_lane_x
            tile_mask = self.encoder.get_legal_tile_mask(p_cid, tower_hps=tower_hps)
            if not tile_mask[punish_tile]:
                legal_indices = np.where(tile_mask)[0]
                punish_tile = int(legal_indices[np.argmin(np.abs(legal_indices - punish_tile))])
            self.last_decision_reason = (
                f"[PUNISH PUSH] Opponent low elixir ({enemy_elixir:.1f}) / backfield commit -> "
                f"Pressuring bridge ({punish_y},{punish_lane_x}) with {CARD_ID_TO_NAME.get(p_cid)}"
            )
            return p_slot + 1, punish_tile, 0.92, 0.92

        # ======================================================================
        # PRIORITY 5: Pocket Deployment When an Enemy Princess Tower is Down
        # ======================================================================
        if (enemy_left_down or enemy_right_down) and elixir >= 6.5 and troop_slots:
            pocket_candidates = [
                s for s in troop_slots
                if not (s[1] == 26 and hold_flags["hold_tank_killer"])
                and "passive_building" not in get_card_roles(s[1])
            ] or troop_slots
            pocket_candidates.sort(key=lambda s: s[2], reverse=True)
            p_slot, p_cid, _, _ = pocket_candidates[0]
            tile_mask = self.encoder.get_legal_tile_mask(p_cid, tower_hps=tower_hps)

            pocket_y = 13
            pocket_x = 4 if enemy_left_down else 13
            pocket_tile = pocket_y * ARENA_COLS + pocket_x
            if not tile_mask[pocket_tile]:
                legal_indices = np.where(tile_mask)[0]
                if len(legal_indices) > 0:
                    pocket_tile = int(legal_indices[np.argmin(np.abs(legal_indices - pocket_tile))])
            self.last_decision_reason = (
                f"[POCKET ATTACK] Enemy princess tower down -> Deploying {CARD_ID_TO_NAME.get(p_cid)} "
                f"in pocket ({pocket_tile // ARENA_COLS},{pocket_tile % ARENA_COLS})"
            )
            return p_slot + 1, pocket_tile, 0.91, 0.91

        # ======================================================================
        # PRIORITY 6: Tempo-Aware Proactive Push / Investment (1x vs 2x Elixir)
        # ======================================================================
        # In 2x/3x elixir, initiate pushes at 7.2 elixir to maintain tempo and never leak;
        # in 1x single elixir, wait until 8.5 elixir for safe banking.
        proactive_threshold = 7.2 if is_double_elixir else 8.5
        if elixir >= proactive_threshold and troop_slots:
            push_candidates = [
                s for s in troop_slots
                if not (s[1] == 26 and hold_flags["hold_tank_killer"] and len(troop_slots) > 1)
            ]

            # 0. Passive Elixir Collector (45): place safely in front of King Tower or behind healthier Princess Tower
            pump_slots = [s for s in push_candidates if s[1] == 45]
            if pump_slots and not threats:
                p_slot, p_cid, _, _ = pump_slots[0]
                pump_y = 27
                pump_x = 8 if ally_left_hp >= ally_right_hp else 9
                pump_tile = pump_y * ARENA_COLS + pump_x
                self.last_decision_reason = f"[ELIXIR PUMP] Investing Elixir Collector safely at ({pump_y},{pump_x})"
                return p_slot + 1, pump_tile, 0.92, 0.92

            # 0B. Offensive Siege Building (X-Bow 110, Mortar 109): place 1 tile behind bridge to lock tower!
            siege_slots = [s for s in push_candidates if "siege_building" in get_card_roles(s[1])]
            if siege_slots and not any(t.tile_x < 9 if target_lane_x == 3 else t.tile_x >= 9 for t in threats):
                sg_slot, sg_cid, _, _ = siege_slots[0]
                siege_y, siege_x = 17, target_lane_x
                siege_tile = siege_y * ARENA_COLS + siege_x
                self.last_decision_reason = (
                    f"[SIEGE LOCK] Deploying {CARD_ID_TO_NAME.get(sg_cid)} at bridge ({siege_y},{siege_x}) "
                    f"to lock onto Enemy Princess Tower"
                )
                return sg_slot + 1, siege_tile, 0.93, 0.93

            # 1. If Giant (24) or heavy Beatdown Tank (Golem 37, Lava Hound 54, Goblin Giant 83, Electro Giant 93) is in hand:
            tank_slots = [s for s in push_candidates if s[1] in (24, 37, 54, 83, 93, 108)]
            if tank_slots:
                g_slot, g_cid, _, _ = tank_slots[0]
                self.last_tank_lane = target_lane_x
                if enemy_elixir <= 1.5 or time_left <= 20.0 or len(ally_units) > 0 or g_cid == 108:
                    deploy_y, deploy_x = 17, target_lane_x
                    style_tag = "Bridge Push"
                else:
                    deploy_y = 28
                    deploy_x = 7 if target_lane_x == 3 else 10
                    style_tag = "Slow-Roll Behind King"

                deploy_tile = deploy_y * ARENA_COLS + deploy_x
                tile_mask = self.encoder.get_legal_tile_mask(g_cid, tower_hps=tower_hps)
                if not tile_mask[deploy_tile]:
                    legal_indices = np.where(tile_mask)[0]
                    deploy_tile = int(legal_indices[np.argmin(np.abs(legal_indices - deploy_tile))])
                self.last_decision_reason = (
                    f"[INITIATE PUSH - {style_tag}] Elixir {elixir:.1f}/10 -> "
                    f"Deploying {CARD_ID_TO_NAME.get(g_cid)} at ({deploy_tile // ARENA_COLS},{deploy_tile % ARENA_COLS})"
                )
                return g_slot + 1, deploy_tile, 0.92, 0.92

            # 2. Otherwise, cycle a safe investment troop in the back behind King Tower (Archers split, Knight, Musketeer, Spear Goblins)
            cycle_priority = {21: 6, 23: 5, 20: 4, 27: 3, 34: 2, 43: 1}
            push_candidates.sort(key=lambda s: (cycle_priority.get(s[1], 0), -s[2]), reverse=True)
            c_slot, c_cid, _, _ = push_candidates[0]
            self.last_tank_lane = target_lane_x

            if enemy_elixir <= 1.5 or time_left <= 20.0 or "fast_bypass_win_condition" in get_card_roles(c_cid):
                cycle_y, cycle_x = 17, target_lane_x
            elif "defensive_building" in get_card_roles(c_cid):
                # Pre-plant defensive building in 4-3 center spot at high elixir
                cycle_y, cycle_x = 19, (9 if target_lane_x == 3 else 8)
            elif c_cid in (21, 23, 22, 30, 48, 70, 82):
                # Split multi-unit cards in the center behind King Tower!
                cycle_y = 29
                cycle_x = 8 if target_lane_x == 3 else 9
            else:
                # Single unit investment behind King Tower in target lane
                cycle_y = 28
                cycle_x = 7 if target_lane_x == 3 else 10

            cycle_tile = cycle_y * ARENA_COLS + cycle_x
            tile_mask = self.encoder.get_legal_tile_mask(c_cid, tower_hps=tower_hps)
            if not tile_mask[cycle_tile]:
                legal_indices = np.where(tile_mask)[0]
                cycle_tile = int(legal_indices[np.argmin(np.abs(legal_indices - cycle_tile))])
            self.last_decision_reason = (
                f"[INVESTMENT CYCLE] Elixir {elixir:.1f}/10 (Opp Cards Played: {opp_cards_played}) -> "
                f"Cycling {CARD_ID_TO_NAME.get(c_cid)} in back at ({cycle_tile // ARENA_COLS},{cycle_tile % ARENA_COLS})"
            )
            return c_slot + 1, cycle_tile, 0.88, 0.88

        # ======================================================================
        # PRIORITY 7: Emergency Full-Elixir Spell Cycle (Only if all 4 cards are spells at >= 9.6 elixir)
        # ======================================================================
        if elixir >= 9.6 and not troop_slots and spell_slots:
            s_slot, s_cid, _, _ = spell_slots[0]
            tower_tile = (7 * ARENA_COLS + 3) if (not enemy_left_down and (enemy_left_hp <= enemy_right_hp or enemy_right_down)) else (7 * ARENA_COLS + 14)
            self.last_decision_reason = (
                f"[OVERFLOW SPELL CHIP] Hand has only spells at {elixir:.1f} elixir -> "
                f"Chipping Enemy Princess Tower with {CARD_ID_TO_NAME.get(s_cid)}"
            )
            return s_slot + 1, tower_tile, 0.80, 0.80

        # ======================================================================
        # PRIORITY 8: Patient Human Wait (Bank Elixir & Observe Opponent)
        # ======================================================================
        self.last_decision_reason = (
            f"[WAIT & ASSESS] Elixir: {elixir:.1f}/10 vs Opp: {enemy_elixir:.1f}/10 "
            f"(Opp plays: {opp_cards_played}) — Banking elixir for optimal trade"
        )
        return 0, 0, 1.0, 1.0

    def compute_tactical_action_priors(
        self,
        arena_grid: np.ndarray,
        scalars: np.ndarray,
        hand_card_ids: List[int],
        legal_card_mask: np.ndarray,
        detections: Optional[List[Detection]] = None,
        opponent_tracker: Optional[OpponentTracker] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes structured logit priors (card_prior[5], tile_prior[576]) from the tactical engine.
        Used by DeckConditionedPolicy.predict_hybrid_action() to fuse learned neural representations
        with domain-accurate counter matchups, 4-3 building geometry, and spell discipline.
        """
        card_prior = np.zeros(5, dtype=np.float32)
        tile_prior = np.zeros(TOTAL_TILES, dtype=np.float32)

        rb_slot, rb_tile, rb_conf, _ = self.predict_action(
            arena_grid=arena_grid,
            scalars=scalars,
            hand_card_ids=hand_card_ids,
            legal_card_mask=legal_card_mask,
            detections=detections,
            opponent_tracker=opponent_tracker,
        )

        # Boost the recommended tactical slot
        card_prior[rb_slot] += 2.5 * float(rb_conf)

        if rb_slot > 0:
            ry, rx = rb_tile // ARENA_COLS, rb_tile % ARENA_COLS
            # Add a smooth 2D Gaussian spatial prior around the recommended tile (sigma = 1.5 tiles)
            ys, xs = np.ogrid[:ARENA_ROWS, :ARENA_COLS]
            dist_sq = (ys - ry) ** 2 + (xs - rx) ** 2
            spatial_bonus = 3.0 * np.exp(-dist_sq / (2.0 * (1.5 ** 2)))
            tile_prior = spatial_bonus.flatten().astype(np.float32)

        # Explicitly penalize invalid spell corner tiles
        for y in range(ARENA_ROWS):
            for x in range(ARENA_COLS):
                if not self.is_valid_spell_tile(y, x):
                    tile_prior[y * ARENA_COLS + x] -= 3.0

        return card_prior, tile_prior

    def validate_or_override_action(
        self,
        chosen_slot: int,
        chosen_tile: int,
        arena_grid: np.ndarray,
        scalars: np.ndarray,
        hand_card_ids: List[int],
        legal_card_mask: np.ndarray,
        detections: Optional[List[Detection]] = None,
        opponent_tracker: Optional[OpponentTracker] = None,
    ) -> Tuple[int, int, str]:
        """
        Tactical guardrail for neural policy predictions:
        1. Intercepts wasted spells (casting a spell on empty ground or in arena corners).
        2. Intercepts ignored CRITICAL threats on our side when neural policy chooses Wait or opposite-lane push.
        3. Intercepts repetitive bridge-only placements when a defensive pull is required.
        """
        enemy_units, _ = self.extract_board_units(arena_grid, detections)
        tower_hps = list(scalars[3:9]) if (scalars is not None and len(scalars) >= 9) else [1.0] * 6
        threats = self.assess_threats(enemy_units, tower_hps)

        # 1. If a CRITICAL or HIGH threat is on our side, ensure we defend it properly
        urgent_threats = [t for t in threats if t.threat_level in ("CRITICAL", "HIGH") and t.tile_y >= 16]
        if urgent_threats:
            rb_slot, rb_tile, _, _ = self.predict_action(
                arena_grid=arena_grid,
                scalars=scalars,
                hand_card_ids=hand_card_ids,
                legal_card_mask=legal_card_mask,
                detections=detections,
                opponent_tracker=opponent_tracker,
            )
            if rb_slot > 0:
                return rb_slot, rb_tile, self.last_decision_reason

        # 2. If neural policy chose a card, validate spell usage & corner bounds
        if chosen_slot > 0 and (chosen_slot - 1) < len(hand_card_ids):
            cid = int(hand_card_ids[chosen_slot - 1])
            ty, tx = chosen_tile // ARENA_COLS, chosen_tile % ARENA_COLS
            is_spell = bool(CARD_ID_TO_IS_SPELL.get(cid, False) and CARD_ID_TO_CATEGORY.get(cid) == "spell")

            if is_spell:
                # Reject spell in corners or when no enemy units / low-HP tower is near target tile
                near_enemies = [u for u in enemy_units if np.hypot(u["y"] - ty, u["x"] - tx) <= 4.0]
                near_low_tower = (
                    (tower_hps[4] <= 0.16 and abs(ty - 7) <= 2 and abs(tx - 3) <= 2)
                    or (tower_hps[5] <= 0.16 and abs(ty - 7) <= 2 and abs(tx - 14) <= 2)
                )
                if not self.is_valid_spell_tile(ty, tx) or (len(near_enemies) < 2 and not near_low_tower):
                    rb_slot, rb_tile, _, _ = self.predict_action(
                        arena_grid=arena_grid,
                        scalars=scalars,
                        hand_card_ids=hand_card_ids,
                        legal_card_mask=legal_card_mask,
                        detections=detections,
                        opponent_tracker=opponent_tracker,
                    )
                    return rb_slot, rb_tile, f"[SPELL GUARD OVERRIDE] Blocked wasted {CARD_ID_TO_NAME.get(cid)} at ({ty},{tx}) -> {self.last_decision_reason}"

        return chosen_slot, chosen_tile, "Neural policy action verified"

