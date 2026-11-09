"""
Rule-Based Scripted Bot for Clash Royale.
Used for:
- Behavioral Cloning warm-start (AlphaStar Stage 1)
- League baseline (prevents agent from unlearning basic defense during self-play)
"""

import numpy as np
from typing import Dict, Any
from cr_alphastar_agent.config import ARENA_WIDTH, ARENA_HEIGHT, CARDS_CATALOG, SPELL_CARD_IDS


class ScriptedBot:
    """
    Heuristic rule-based bot operating in canonical perspective:
    - Analyzes incoming threats on friendly half (y < 16) and mounts defense.
    - If no immediate threat and elixir >= 9.2, plays a push/win condition to avoid elixir leakage.
    - Otherwise returns No-Op (slot 4) to bank elixir.
    """
    def __init__(self, player_id: int = 2):
        self.player_id = player_id

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        scalars = obs["scalars"]
        own_elixir = scalars[0] * 10.0  # Denormalize
        card_mask = obs["card_mask"]
        entities = obs["entities"]
        entity_mask = obs["entity_mask"]

        # 1. Check for incoming threats on our half (canonical y < 0.5 -> y < 16)
        threats = []
        for i in range(len(entities)):
            if entity_mask[i]:
                continue
            e = entities[i]
            # e: [card_id_norm, is_enemy, x_norm, y_norm, hp_ratio, is_air, state]
            is_enemy = (e[1] > 0.0)
            y_norm = e[3]
            # If enemy unit crossed the river onto friendly side
            if is_enemy and y_norm < 0.5 and e[4] > 0.05:
                threats.append(e)

        # 2. Defensive reaction if threatened and have elixir
        if threats:
            threats.sort(key=lambda t: t[3])
            primary_threat = threats[0]
            tx = int(np.clip(primary_threat[2] * ARENA_WIDTH, 1, ARENA_WIDTH - 2))
            ty = int(np.clip(primary_threat[3] * ARENA_HEIGHT, 3, 14))

            affordable_slots = [s for s in range(4) if card_mask[s]]
            if affordable_slots:
                chosen_slot = int(np.random.choice(affordable_slots))
                deploy_y = max(1, ty - 2)
                tile_idx = deploy_y * ARENA_WIDTH + tx
                return np.array([chosen_slot, tile_idx], dtype=np.int64)

        # 3. Offense / Push building at high elixir (>= 9.2) to prevent elixir leakage
        if own_elixir >= 9.2:
            affordable_slots = [s for s in range(4) if card_mask[s]]
            if affordable_slots:
                chosen_slot = int(np.random.choice(affordable_slots))
                # Place behind King tower (canonical y=2) or at bridge (canonical y=14)
                if np.random.rand() < 0.5:
                    deploy_tile = 2 * ARENA_WIDTH + 9
                else:
                    lane_x = 3 if np.random.rand() < 0.5 else 14
                    deploy_tile = 14 * ARENA_WIDTH + lane_x
                return np.array([chosen_slot, deploy_tile], dtype=np.int64)

        # 4. Default to bank elixir: Wait / No-Op
        return np.array([4, 0], dtype=np.int64)


class TacticalProBot:
    """
    Advanced Tactical Bot executing Human Pro Micro and Macro mechanics:
    1. Back-of-King Slow Push (canonical y=2..3): Banks elixir by building pushes behind King tower.
    2. Tank Tricking (Bridge Meat-Shielding): Drops high-HP tanks in front of vulnerable ranged support troops.
    3. Opposite-Lane Center Kiting: Pulls bridge-crossing melee threats into the dual-Princess-Tower center kill zone.
    4. Center-Back Troop Splitting: Splits swarms/pairs on the center tile line (x=9) to pressure both lanes.
    5. Spell Bait & Tricky Barrel: Punishes recent opponent spell usage with swarms/Goblin Barrel on tower.
    """
    def __init__(self, player_id: int = 2):
        self.player_id = player_id
        self.step_count = 0
        self.last_opp_spell_step = -999

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        self.step_count += 1
        scalars = obs["scalars"]
        own_elixir = float(scalars[0] * 10.0)
        opp_elixir_est = float(scalars[1] * 10.0)
        card_mask = obs["card_mask"]
        hand_cards = obs["hand_cards"][:4]
        entities = obs["entities"]
        entity_mask = obs["entity_mask"]

        affordable_slots = [s for s in range(4) if card_mask[s]]
        if not affordable_slots:
            return np.array([4, 0], dtype=np.int64)

        # Classify active entities in canonical frame
        # e: [card_id_norm, side (+1 own, -1 opp), x_norm, y_norm, hp_ratio, is_air, state]
        threats = []
        friendly_support = []
        friendly_tanks = []

        for i in range(len(entities)):
            if entity_mask[i]:
                continue
            e = entities[i]
            if e[4] <= 0.02:
                continue
            side = e[1]
            cid = int(round(e[0] * 16.0))
            norm_x = e[2]
            norm_y = e[3]
            is_air = (e[5] > 0.5)
            ex = norm_x * ARENA_WIDTH
            ey = norm_y * ARENA_HEIGHT

            # Enemy units on our half (y < 16.0)
            if side < 0.0:  # Enemy
                if ey < 16.0 and i >= 6:  # Skip towers
                    threats.append({
                        "cid": cid, "x": ex, "y": ey, "hp": e[4],
                        "is_air": is_air, "norm": e
                    })
            else:  # Friendly
                if i >= 6:  # Skip towers
                    if cid in (2, 4):  # Ranged support (Musketeer, Wizard, Archers, Witch)
                        friendly_support.append({"cid": cid, "x": ex, "y": ey, "hp": e[4]})
                    elif cid in (1, 3, 5, 10, 11):  # Tank / Mini-Tank (Giant, Pekka, Knight, MiniPekka, etc.)
                        friendly_tanks.append({"cid": cid, "x": ex, "y": ey, "hp": e[4]})

        # -------------------------------------------------------------
        # TACTIC 1: Opposite-Lane Center Kiting (High Priority Defense)
        # -------------------------------------------------------------
        ground_melee_threats = [t for t in threats if not t["is_air"] and t["y"] >= 9.0]
        if ground_melee_threats:
            # Check for cheap distraction cards in hand (Skeletons=6, Minions=7, Knight=1, Goblins=14, Spirits=6)
            cheap_slots = [s for s in affordable_slots if hand_cards[s] in (6, 7, 1, 14)]
            if cheap_slots:
                target_threat = min(ground_melee_threats, key=lambda t: t["y"])
                chosen_slot = cheap_slots[0]
                # If threat is pushing Left Lane (x < 8.0), kite into Right Lane Center (x=10.0, y=11.0)
                if target_threat["x"] < 8.0:
                    kite_x = 10
                    kite_y = 11
                # If threat is pushing Right Lane (x > 10.0), kite into Left Lane Center (x=7.5, y=11.0)
                elif target_threat["x"] > 10.0:
                    kite_x = 7
                    kite_y = 11
                else:
                    kite_x = 9
                    kite_y = 10
                tile_idx = kite_y * ARENA_WIDTH + kite_x
                return np.array([chosen_slot, tile_idx], dtype=np.int64)

        # Standard Defense if threatened and not kiting
        if threats:
            threats.sort(key=lambda t: t["y"])
            pri_threat = threats[0]
            tx = int(np.clip(pri_threat["x"], 1, ARENA_WIDTH - 2))
            ty = int(np.clip(pri_threat["y"], 3, 14))
            chosen_slot = int(np.random.choice(affordable_slots))
            deploy_y = max(1, ty - 2)
            tile_idx = deploy_y * ARENA_WIDTH + tx
            return np.array([chosen_slot, tile_idx], dtype=np.int64)

        # -------------------------------------------------------------
        # TACTIC 2: Tank Tricking (Bridge Meat-Shielding)
        # -------------------------------------------------------------
        # If a friendly ranged support troop is approaching bridge (y in [10.0, 14.5])
        bridge_support = [s for s in friendly_support if 10.0 <= s["y"] <= 14.5]
        if bridge_support:
            tank_slots = [s for s in affordable_slots if hand_cards[s] in (3, 5, 1, 10, 11)]
            if tank_slots:
                chosen_support = bridge_support[0]
                chosen_slot = tank_slots[0]
                # Drop tank 2 tiles ahead closer to river (y=14..15) on same lane
                shield_x = int(np.clip(chosen_support["x"], 2, ARENA_WIDTH - 3))
                shield_y = min(15, int(chosen_support["y"]) + 2)
                tile_idx = shield_y * ARENA_WIDTH + shield_x
                return np.array([chosen_slot, tile_idx], dtype=np.int64)

        # -------------------------------------------------------------
        # TACTIC 3: Spell Bait & Tricky Barrel
        # -------------------------------------------------------------
        # If holding Goblin Barrel (14) or Hog Rider (11)
        barrel_slots = [s for s in affordable_slots if hand_cards[s] == 14]
        if barrel_slots and (opp_elixir_est < 4.5 or np.random.rand() < 0.35):
            chosen_slot = barrel_slots[0]
            # Tricky Barrel: Offset slightly behind enemy princess tower (x=3 or 14, y=27)
            target_x = 3 if np.random.rand() < 0.5 else 14
            target_y = 27  # Behind enemy princess tower
            tile_idx = target_y * ARENA_WIDTH + target_x
            return np.array([chosen_slot, tile_idx], dtype=np.int64)

        hog_slots = [s for s in affordable_slots if hand_cards[s] == 11]
        if hog_slots and opp_elixir_est < 4.0:
            chosen_slot = hog_slots[0]
            # Bridge rush on weaker lane
            bridge_x = 3 if np.random.rand() < 0.5 else 14
            bridge_y = 15
            tile_idx = bridge_y * ARENA_WIDTH + bridge_x
            return np.array([chosen_slot, tile_idx], dtype=np.int64)

        # -------------------------------------------------------------
        # TACTIC 4: Center-Back Troop Splitting
        # -------------------------------------------------------------
        if own_elixir >= 8.5:
            split_slots = [s for s in affordable_slots if hand_cards[s] in (2, 6, 7)]
            if split_slots and np.random.rand() < 0.40:
                chosen_slot = split_slots[0]
                # Drop dead center on dividing line behind towers (x=9, y=5)
                tile_idx = 5 * ARENA_WIDTH + 9
                return np.array([chosen_slot, tile_idx], dtype=np.int64)

        # -------------------------------------------------------------
        # TACTIC 5: Back-of-King Slow Push (Investment Deployment)
        # -------------------------------------------------------------
        if own_elixir >= 8.8:
            heavy_slots = [s for s in affordable_slots if hand_cards[s] in (3, 4, 1, 5, 10, 12)]
            if heavy_slots:
                chosen_slot = heavy_slots[0]
                # Drop behind King Tower (x=8 or 10, y=2)
                corner_x = 8 if np.random.rand() < 0.5 else 10
                tile_idx = 2 * ARENA_WIDTH + corner_x
                return np.array([chosen_slot, tile_idx], dtype=np.int64)

        # Default: Bank elixir
        return np.array([4, 0], dtype=np.int64)
