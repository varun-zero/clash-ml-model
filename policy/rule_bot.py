"""
Rule-Based Beginner Heuristic Policy for Clash Royale ("RuleBot").
Implements strategic beginner heuristics without requiring neural network weights:
1. Defend near threats on the ally half with the cheapest valid counter.
2. Target enemy swarms and clusters with damage spells (Arrows / Fireball).
3. Proactive tank pushes (Giant / Mini PEKKA / Knight) at bridge when elixir >= 5.5.
4. Support follow-up (Musketeer / Archers / Minions) placed behind tanks.
5. Pocket deployment when enemy princess tower is destroyed.
6. Spell finisher on low-HP enemy towers.
7. Strict legality enforcement: never plays unaffordable cards or illegal tiles.
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from configs.constants import (
    ARENA_ROWS,
    ARENA_COLS,
    TOTAL_TILES,
    WAIT_CARD_ID,
    CARD_ID_TO_COST,
    CARD_ID_TO_IS_SPELL,
    CARD_DATABASE,
    screen_to_grid,
)
from configs.card_features import card_targets_air, card_targets_ground
from perception.state_encoder import Detection, StateEncoder


class RuleBotPolicy:
    """
    Deterministic rule-based agent executing fundamental Clash Royale strategies.
    Provides a strong baseline and generates clean trajectory data for training.
    """
    def __init__(self):
        self.encoder = StateEncoder()
        self.last_tank_time = 0.0
        self.last_tank_lane = 14  # Default right bridge

    def calculate_predictive_spell_tile(
        self,
        spell_cid: int,
        target_y: int,
        target_x: int,
        detections: Optional[List[Detection]] = None,
    ) -> int:
        """
        Calculates the lead-predicted impact tile for projectile spells (Fireball, Arrows).
        Compensates for server deploy delay (1.0s) + distance-based flight time from King Tower (28, 9).
        Leads marching enemy units forward along their lane vector so the spell lands exactly
        when the troops walk into the blast radius instead of trailing behind them.
        """
        # King Tower launch origin
        king_y, king_x = 28, 9

        # Distance from King Tower to target
        dist = np.hypot(king_y - target_y, king_x - target_x)
        # Server delay (1.0s) + projectile travel speed (~8.0 tiles/sec)
        t_flight = 1.0 + (dist / 8.0)

        # Check if enemy is stationary (attacking our tower)
        # Left Princess Tower: (24, 3); Right Princess Tower: (24, 14); King Tower: (28, 9)
        is_at_left_princess = (target_y >= 23 and 1 <= target_x <= 5)
        is_at_right_princess = (target_y >= 23 and 12 <= target_x <= 16)
        is_at_king = (target_y >= 26 and 7 <= target_x <= 11)

        if is_at_left_princess or is_at_right_princess or is_at_king:
            # Unit is stationary attacking our tower - zero lead needed
            lead_y = 0.0
            lead_x = 0.0
        else:
            # Unit is moving towards player side (increasing y)
            measured_vy = None
            if detections:
                for d in detections:
                    if not d.is_ally:
                        d_pos = screen_to_grid((d.x1 + d.x2) / 2.0, d.y2)
                        if d_pos and abs(d_pos[0] - target_y) <= 2 and abs(d_pos[1] - target_x) <= 2:
                            if hasattr(d, "vy") and abs(d.vy) > 0.01:
                                measured_vy = d.vy * 32.0
                                break

            if measured_vy is not None and measured_vy > 0.2:
                lead_y = float(np.clip(measured_vy * t_flight, 1.2, 3.5))
            else:
                # Default marching speed: ~1.1 tiles/sec -> lead ~ 1.8 to 2.8 tiles down
                lead_y = float(np.clip(1.1 * t_flight, 1.5, 3.0))

            # Lateral steering towards target lane Princess Tower
            target_lane_x = 3 if target_x < 9 else 14
            lead_x = 0.5 * np.sign(target_lane_x - target_x)

        pred_y = int(np.clip(round(target_y + lead_y), 0, ARENA_ROWS - 1))
        pred_x = int(np.clip(round(target_x + lead_x), 0, ARENA_COLS - 1))
        return pred_y * ARENA_COLS + pred_x

    def predict_action(
        self,
        arena_grid: np.ndarray,
        scalars: np.ndarray,
        hand_card_ids: List[int],
        legal_card_mask: np.ndarray,
        legal_tile_mask: Optional[np.ndarray] = None,
        detections: Optional[List[Detection]] = None,
    ) -> Tuple[int, int, float, float]:
        """
        Computes (chosen_slot, chosen_tile, card_conf, tile_conf).
        chosen_slot: 0 = Wait, 1..4 = Hand Slot 0..3
        chosen_tile: 0..575
        """
        # 1. Check affordable cards
        affordable_slots: List[Tuple[int, int, int, bool]] = []
        elixir = scalars[0] * 10.0 if scalars is not None and len(scalars) > 0 else 0.0

        for slot_idx in range(4):
            if slot_idx < len(hand_card_ids):
                cid = int(hand_card_ids[slot_idx])
                if cid > 0 and legal_card_mask[slot_idx + 1]:
                    cost = CARD_ID_TO_COST.get(cid, 99)
                    is_spell = CARD_ID_TO_IS_SPELL.get(cid, False)
                    affordable_slots.append((slot_idx, cid, cost, is_spell))

        # If no card is affordable, we must Wait
        if not affordable_slots:
            return 0, 0, 1.0, 1.0

        # Tower HPs from scalars [3..8]
        # [0: ally_king, 1: ally_left, 2: ally_right, 3: enemy_king, 4: enemy_left, 5: enemy_right]
        tower_hps = list(scalars[3:9]) if scalars is not None and len(scalars) >= 9 else [1.0] * 6
        enemy_king_hp = tower_hps[3]
        enemy_left_hp = tower_hps[4]
        enemy_right_hp = tower_hps[5]

        enemy_left_down = enemy_left_hp <= 0.05
        enemy_right_down = enemy_right_hp <= 0.05

        # Choose primary push lane (target the lower HP or non-destroyed tower)
        if enemy_left_down and not enemy_right_down:
            target_lane_x = 14  # Push right tower
        elif enemy_right_down and not enemy_left_down:
            target_lane_x = 3   # Push left tower
        elif enemy_left_hp < enemy_right_hp:
            target_lane_x = 3   # Left has lower HP
        else:
            target_lane_x = 14  # Default right lane

        # Extract enemy and ally units from arena grid or parsed detections
        enemy_units = []
        ally_units = []
        if detections:
            from configs.constants import screen_to_grid
            for d in detections:
                pos = screen_to_grid((d.x1 + d.x2) / 2.0, d.y2)
                if pos:
                    if d.is_ally:
                        ally_units.append((pos[0], pos[1], d.category.lower(), d.cls_id))
                    else:
                        enemy_units.append((pos[0], pos[1], d.category.lower()))
        else:
            # Fallback: scan channels
            ch_eground = arena_grid[3] if arena_grid.ndim == 3 else arena_grid[0, 3]
            ch_eair = arena_grid[4] if arena_grid.ndim == 3 else arena_grid[0, 4]
            for y, x in zip(*np.nonzero(ch_eground > 0)):
                enemy_units.append((int(y), int(x), "ground"))
            for y, x in zip(*np.nonzero(ch_eair > 0)):
                enemy_units.append((int(y), int(x), "air"))

            ch_aground = arena_grid[0] if arena_grid.ndim == 3 else arena_grid[0, 0]
            for y, x in zip(*np.nonzero(ch_aground > 0)):
                ally_units.append((int(y), int(x), "ground", 0))

        # ----------------------------------------------------------------------
        # Strategy A: Threat Defense on Ally Half (rows 16..31)
        # ----------------------------------------------------------------------
        ally_threats = [u for u in enemy_units if u[0] >= 16]
        if ally_threats:
            ally_threats.sort(key=lambda u: u[0], reverse=True)
            deepest_threat = ally_threats[0]
            ty, tx, tcat = deepest_threat

            candidates = []
            for slot_idx, cid, cost, is_spell in affordable_slots:
                if is_spell:
                    candidates.append((cost + 2, slot_idx, cid, is_spell))
                    continue
                if tcat == "air":
                    if card_targets_air(cid):
                        candidates.append((cost, slot_idx, cid, is_spell))
                else:
                    candidates.append((cost, slot_idx, cid, is_spell))

            if candidates:
                candidates.sort(key=lambda c: c[0])
                _, best_slot, best_cid, best_is_spell = candidates[0]

                if best_is_spell:
                    # Predictive lead targeting for defensive spells
                    tile_idx = self.calculate_predictive_spell_tile(
                        spell_cid=best_cid,
                        target_y=ty,
                        target_x=tx,
                        detections=detections,
                    )
                else:
                    # Position placement slightly in front of threat or towards center
                    def_y = min(31, max(16, ty + 2))
                    def_x = int(np.clip(tx + (1 if tx < 9 else -1), 1, ARENA_COLS - 2))

                    tile_mask = self.encoder.get_legal_tile_mask(best_cid, tower_hps=tower_hps)
                    tile_idx = def_y * ARENA_COLS + def_x
                    if not tile_mask[tile_idx]:
                        legal_indices = np.where(tile_mask)[0]
                        if len(legal_indices) > 0:
                            tile_idx = int(legal_indices[np.argmin(np.abs(legal_indices - tile_idx))])

                return best_slot + 1, tile_idx, 0.95, 0.95

        # ----------------------------------------------------------------------
        # Strategy B: Spell on Clusters or Low-HP Tower Finisher
        # ----------------------------------------------------------------------
        spell_slots = [s for s in affordable_slots if s[3]]
        if spell_slots:
            time_left = scalars[2] * 180.0 if (scalars is not None and len(scalars) > 2) else 180.0
            elapsed = 180.0 - time_left

            # Finisher: ONLY fire on tower if:
            # 1. Match is at least 35 seconds in (avoid false tower detections at match start)
            # 2. Tower is genuinely at critical health (<= 0.15 for Fireball, <= 0.05 for Arrows)
            if elapsed >= 35.0:
                for s_idx, s_cid, s_cost, _ in spell_slots:
                    crit_threshold = 0.15 if s_cid == 2 else 0.05
                    if 0.02 < enemy_left_hp <= crit_threshold:
                        return s_idx + 1, 7 * ARENA_COLS + 3, 0.98, 0.98
                    elif 0.02 < enemy_right_hp <= crit_threshold:
                        return s_idx + 1, 7 * ARENA_COLS + 14, 0.98, 0.98
                    elif 0.02 < enemy_king_hp <= crit_threshold:
                        return s_idx + 1, 5 * ARENA_COLS + 9, 0.98, 0.98

            # Cluster damage: 2+ enemies close together with PREDICTIVE LEAD AIMING
            if len(enemy_units) >= 2:
                for u1 in enemy_units:
                    neighbors = [
                        u2 for u2 in enemy_units
                        if abs(u1[0] - u2[0]) <= 3 and abs(u1[1] - u2[1]) <= 3
                    ]
                    if len(neighbors) >= 2:
                        avg_y = int(round(np.mean([n[0] for n in neighbors])))
                        avg_x = int(round(np.mean([n[1] for n in neighbors])))
                        best_spell = min(spell_slots, key=lambda s: s[2])
                        # Calculate predictive lead tile ahead of the moving swarm!
                        lead_tile = self.calculate_predictive_spell_tile(
                            spell_cid=best_spell[1],
                            target_y=avg_y,
                            target_x=avg_x,
                            detections=detections,
                        )
                        return best_spell[0] + 1, lead_tile, 0.92, 0.92

        # ----------------------------------------------------------------------
        # Strategy C: Pocket Attack if an Enemy Tower is Down
        # ----------------------------------------------------------------------
        if (enemy_left_down or enemy_right_down) and elixir >= 4.0:
            troop_slots = [s for s in affordable_slots if not s[3]]
            if troop_slots:
                # Deploy high DPS troop directly into enemy territory pocket
                troop_slots.sort(key=lambda s: s[2], reverse=True)
                p_slot, p_cid, _, _ = troop_slots[0]
                tile_mask = self.encoder.get_legal_tile_mask(p_cid, tower_hps=tower_hps)

                # Target row 13 or 14 towards the remaining tower or king
                pocket_y = 13
                pocket_x = 3 if enemy_left_down else 14
                pocket_tile = pocket_y * ARENA_COLS + pocket_x
                if tile_mask[pocket_tile]:
                    return p_slot + 1, pocket_tile, 0.92, 0.92
                else:
                    legal_indices = np.where(tile_mask)[0]
                    if len(legal_indices) > 0:
                        pocket_tile = int(legal_indices[np.argmin(np.abs(legal_indices - pocket_tile))])
                        return p_slot + 1, pocket_tile, 0.90, 0.90

        # ----------------------------------------------------------------------
        # Strategy D: Proactive Bridge Push & Support (Elixir >= 5.0)
        # ----------------------------------------------------------------------
        if elixir >= 5.0:
            troop_slots = [s for s in affordable_slots if not s[3]]
            if troop_slots:
                # 1. If Giant (ID 24) is affordable, deploy Giant at bridge!
                giant_slots = [s for s in troop_slots if s[1] == 24]
                if giant_slots:
                    g_slot, g_cid, _, _ = giant_slots[0]
                    self.last_tank_lane = target_lane_x
                    bridge_tile = 17 * ARENA_COLS + target_lane_x
                    tile_mask = self.encoder.get_legal_tile_mask(g_cid, tower_hps=tower_hps)
                    if not tile_mask[bridge_tile]:
                        legal_indices = np.where(tile_mask)[0]
                        bridge_tile = int(legal_indices[np.argmin(np.abs(legal_indices - bridge_tile))])
                    return g_slot + 1, bridge_tile, 0.95, 0.95

                # 2. Support troop: if we have ally units pushing, or elixir >= 7.0
                support_slots = [s for s in troop_slots if s[1] in (27, 21, 43, 26, 20)]  # Musketeer, Archers, Minions, Mini PEKKA, Knight
                if support_slots and (len(ally_units) > 0 or elixir >= 6.5):
                    # Sort to pick best support
                    s_slot, s_cid, _, _ = support_slots[0]
                    support_y = 19
                    support_tile = support_y * ARENA_COLS + self.last_tank_lane
                    tile_mask = self.encoder.get_legal_tile_mask(s_cid, tower_hps=tower_hps)
                    if not tile_mask[support_tile]:
                        legal_indices = np.where(tile_mask)[0]
                        support_tile = int(legal_indices[np.argmin(np.abs(legal_indices - support_tile))])
                    return s_slot + 1, support_tile, 0.90, 0.90

                # 3. If elixir >= 8.5 (avoid leaking), play any affordable troop at bridge
                if elixir >= 8.5:
                    troop_slots.sort(key=lambda s: s[2], reverse=True)
                    t_slot, t_cid, _, _ = troop_slots[0]
                    bridge_tile = 17 * ARENA_COLS + target_lane_x
                    tile_mask = self.encoder.get_legal_tile_mask(t_cid, tower_hps=tower_hps)
                    if not tile_mask[bridge_tile]:
                        legal_indices = np.where(tile_mask)[0]
                        bridge_tile = int(legal_indices[np.argmin(np.abs(legal_indices - bridge_tile))])
                    return t_slot + 1, bridge_tile, 0.85, 0.85

        # ----------------------------------------------------------------------
        # Strategy E: Passive Wait (Conserve Elixir)
        # ----------------------------------------------------------------------
        return 0, 0, 1.0, 1.0
