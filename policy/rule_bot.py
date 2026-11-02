"""
Rule-Based Beginner Heuristic Policy for Clash Royale ("RuleBot").
Implements strategic beginner heuristics without requiring neural network weights:
1. Defend near threats on the ally half with the cheapest valid counter.
2. Target enemy swarms and clusters with damage spells (Arrows / Fireball).
3. Push the bridge when elixir >= 9.0 (avoid elixir leaking).
4. Strict legality enforcement: never plays unaffordable cards or illegal tiles.
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
        enemy_left_hp = tower_hps[4]
        enemy_right_hp = tower_hps[5]

        # Extract enemy units from arena grid or parsed detections
        enemy_units = []
        if detections:
            for d in detections:
                if not d.is_ally:
                    # Anchor tile
                    from configs.constants import screen_to_grid
                    pos = screen_to_grid((d.x1 + d.x2) / 2.0, d.y2)
                    if pos:
                        enemy_units.append((pos[0], pos[1], d.category.lower()))
        else:
            # Fallback: scan enemy ground (channel 3) and enemy air (channel 4)
            # If 32 stacked channels, use current frame channels 3 and 4
            ch_ground = arena_grid[3] if arena_grid.ndim == 3 else arena_grid[0, 3]
            ch_air = arena_grid[4] if arena_grid.ndim == 3 else arena_grid[0, 4]

            g_ys, g_xs = np.nonzero(ch_ground > 0)
            for y, x in zip(g_ys, g_xs):
                enemy_units.append((int(y), int(x), "ground"))
            a_ys, a_xs = np.nonzero(ch_air > 0)
            for y, x in zip(a_ys, a_xs):
                enemy_units.append((int(y), int(x), "air"))

        # ----------------------------------------------------------------------
        # Strategy A: Threat Defense on Ally Half (rows 16..31)
        # ----------------------------------------------------------------------
        ally_threats = [u for u in enemy_units if u[0] >= 16]
        if ally_threats:
            # Sort by danger: deepest in our territory (highest row)
            ally_threats.sort(key=lambda u: u[0], reverse=True)
            deepest_threat = ally_threats[0]
            ty, tx, tcat = deepest_threat

            # Look for valid counter in hand
            candidates = []
            for slot_idx, cid, cost, is_spell in affordable_slots:
                if is_spell:
                    candidates.append((cost + 2, slot_idx, cid, is_spell))  # Spells slightly deprioritized for defense
                    continue
                if tcat == "air":
                    if card_targets_air(cid):
                        candidates.append((cost, slot_idx, cid, is_spell))
                else:
                    candidates.append((cost, slot_idx, cid, is_spell))

            if candidates:
                candidates.sort(key=lambda c: c[0])  # Pick cheapest counter
                _, best_slot, best_cid, best_is_spell = candidates[0]

                # Position placement slightly in front of or towards center
                def_y = min(31, max(16, ty + 2))
                def_x = int(np.clip(tx + (1 if tx < 9 else -1), 1, ARENA_COLS - 2))

                tile_mask = self.encoder.get_legal_tile_mask(best_cid, tower_hps=tower_hps)
                tile_idx = def_y * ARENA_COLS + def_x
                if not tile_mask[tile_idx]:
                    # Find nearest legal tile
                    legal_indices = np.where(tile_mask)[0]
                    if len(legal_indices) > 0:
                        tile_idx = int(legal_indices[np.argmin(np.abs(legal_indices - tile_idx))])

                return best_slot + 1, tile_idx, 0.95, 0.95

        # ----------------------------------------------------------------------
        # Strategy B: Spell on Enemy Clusters
        # ----------------------------------------------------------------------
        spell_slots = [s for s in affordable_slots if s[3]]  # is_spell == True
        if spell_slots and len(enemy_units) >= 2:
            # Check for clusters of 2+ enemies within radius 3
            for i, u1 in enumerate(enemy_units):
                neighbors = [
                    u2 for j, u2 in enumerate(enemy_units)
                    if abs(u1[0] - u2[0]) <= 3 and abs(u1[1] - u2[1]) <= 3
                ]
                if len(neighbors) >= 2:
                    avg_y = int(round(np.mean([n[0] for n in neighbors])))
                    avg_x = int(round(np.mean([n[1] for n in neighbors])))
                    target_tile = int(np.clip(avg_y * ARENA_COLS + avg_x, 0, TOTAL_TILES - 1))
                    best_spell = min(spell_slots, key=lambda s: s[2])
                    return best_spell[0] + 1, target_tile, 0.90, 0.90

        # ----------------------------------------------------------------------
        # Strategy C: Bridge Push (High Elixir Pressure >= 9.0)
        # ----------------------------------------------------------------------
        if elixir >= 9.0:
            troop_slots = [s for s in affordable_slots if not s[3]]
            if troop_slots:
                # Prefer heavier troop / tank, or lowest cost troop to cycle
                troop_slots.sort(key=lambda s: s[2], reverse=True)
                push_slot, push_cid, _, _ = troop_slots[0]

                # Choose lane with lower enemy princess tower HP
                # Left bridge: (16, 3), Right bridge: (16, 14)
                if enemy_left_hp <= enemy_right_hp:
                    bridge_y, bridge_x = 16, 3
                else:
                    bridge_y, bridge_x = 16, 14

                bridge_tile = bridge_y * ARENA_COLS + bridge_x
                tile_mask = self.encoder.get_legal_tile_mask(push_cid, tower_hps=tower_hps)
                if not tile_mask[bridge_tile]:
                    bridge_y = 17
                    bridge_tile = bridge_y * ARENA_COLS + bridge_x

                return push_slot + 1, bridge_tile, 0.85, 0.85

        # ----------------------------------------------------------------------
        # Strategy D: Passive Wait (Conserve Elixir)
        # ----------------------------------------------------------------------
        return 0, 0, 1.0, 1.0

