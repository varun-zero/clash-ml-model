"""
High-Performance Headless Clash Royale Replica Environment (Gymnasium API Compatible).
Features:
- Canonical Player 2 Perspective with full Y-axis mirroring and channel swapping.
- Always includes active Crown Towers in entities (Guarantees Transformer NEVER encounters empty entity boards / NaN).
- 18x32 grid arena simulation with King and Princess towers, bridge bottlenecks, and river.
- Dense shaped rewards (tower damage + elixir trade efficiency) + Terminal win/loss (+1.0 / -1.0).
- Action masking for card affordability and legal deployment tiles (including spell full-arena targeting).
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from cr_alphastar_agent.config import (
    ARENA_WIDTH, ARENA_HEIGHT, TOTAL_TILES, MAX_ENTITIES,
    TICKS_PER_SEC, STEP_DT, ACTION_REPEAT, MAX_MATCH_STEPS, REGULAR_TIME_SECS,
    ELIXIR_START, ELIXIR_MAX, ELIXIR_RATE_NORMAL, ELIXIR_RATE_DOUBLE,
    CARDS_CATALOG, SPELL_CARD_IDS, DEFAULT_DECK
)


class Unit:
    """Active troop, building, or spell projectile on the arena."""
    def __init__(self, card_id: int, owner: int, x: float, y: float):
        cfg = CARDS_CATALOG[card_id]
        self.card_id = card_id
        self.owner = owner  # 1 or 2
        self.x = float(x)
        self.y = float(y)
        self.max_hp = float(cfg.get("hp", 100))
        self.hp = self.max_hp
        self.dmg = float(cfg.get("dmg", 50))
        self.speed = float(cfg.get("speed", 1.0))
        self.attack_range = float(cfg.get("range", 1.2))
        self.is_air = bool(cfg.get("air", False))
        self.is_spell = bool(cfg.get("spell", False))
        self.is_building = bool(cfg.get("building", False))
        self.targets = cfg.get("targets", "ground")
        self.splash = float(cfg.get("splash", 0.0))
        self.attack_cooldown = 0.0
        self.hit_speed = 1.2  # seconds per attack
        self.state = 1.0  # 1.0 = alive/moving/attacking

    @property
    def hp_ratio(self) -> float:
        return max(0.0, min(1.0, self.hp / self.max_hp)) if self.max_hp > 0 else 0.0


class Tower:
    """Crown Tower (King or Princess). Always present as an active entity."""
    def __init__(self, owner: int, tower_type: str, x: float, y: float):
        self.owner = owner  # 1 or 2
        self.tower_type = tower_type  # "princess_left", "princess_right", "king"
        self.x = float(x)
        self.y = float(y)
        if tower_type == "king":
            self.max_hp = 4000.0
            self.dmg = 109.0
            self.attack_range = 7.0
            self.is_activated = False  # King activates when damaged or princess falls
        else:
            self.max_hp = 2534.0
            self.dmg = 90.0
            self.attack_range = 7.5
            self.is_activated = True
        self.hp = self.max_hp
        self.hit_speed = 0.8
        self.attack_cooldown = 0.0
        self.alive = True

    @property
    def hp_ratio(self) -> float:
        return max(0.0, min(1.0, self.hp / self.max_hp))


class ClashRoyaleReplicaEnv:
    """
    Headless Clash Royale replica with Gymnasium API compatibility.
    Player 1 starts at bottom (y ~ 0..15), Player 2 starts at top (y ~ 16..31).
    Canonical perspective ensures both players see their own towers at bottom (y < 16).
    Uses Action Repeat (frame-skip) of 4 physics ticks per step (~400ms decision cadence).
    """
    def __init__(self, seed: Optional[int] = None, action_repeat: int = ACTION_REPEAT):
        self.rng = np.random.default_rng(seed)
        self.action_repeat = action_repeat
        self.episode_micro_rewards = {1: 0.0, 2: 0.0}
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.elapsed_time = 0.0
        self.is_overtime = False
        self.episode_micro_rewards = {1: 0.0, 2: 0.0}

        # Elixir state
        self.elixir = {1: float(ELIXIR_START), 2: float(ELIXIR_START)}

        # Decks and Hands (4 hand cards + 1 next card, followed by draw cycle)
        self.deck = {
            1: list(DEFAULT_DECK),
            2: list(DEFAULT_DECK)
        }
        self.rng.shuffle(self.deck[1])
        self.rng.shuffle(self.deck[2])

        self.hand = {
            1: self.deck[1][:4],
            2: self.deck[2][:4]
        }
        self.next_card = {
            1: self.deck[1][4],
            2: self.deck[2][4]
        }
        self.deck_queue = {
            1: self.deck[1][5:],
            2: self.deck[2][5:]
        }
        self.last_spell_time = {1: -999999.0, 2: -999999.0}

        # Initialize Crown Towers
        # Player 1 (Bottom): Princess L (3.5, 6.5), Princess R (14.5, 6.5), King (9.0, 2.5)
        # Player 2 (Top): Princess L (3.5, 25.5), Princess R (14.5, 25.5), King (9.0, 29.5)
        self.towers: Dict[int, Dict[str, Tower]] = {
            1: {
                "princess_left": Tower(1, "princess_left", 3.5, 6.5),
                "princess_right": Tower(1, "princess_right", 14.5, 6.5),
                "king": Tower(1, "king", 9.0, 2.5),
            },
            2: {
                "princess_left": Tower(2, "princess_left", 3.5, 25.5),
                "princess_right": Tower(2, "princess_right", 14.5, 25.5),
                "king": Tower(2, "king", 9.0, 29.5),
            }
        }

        self.units: List[Unit] = []
        obs_p1 = self.get_obs(player_id=1)
        return obs_p1, {}

    def get_obs(self, player_id: int = 1) -> Dict[str, np.ndarray]:
        """
        Constructs observation dictionary in canonical perspective:
        Friendly side is always at bottom (y < 16), Enemy is at top (y >= 16).
        """
        opp_id = 2 if player_id == 1 else 1

        # 1. Scalars: [own_elixir_norm, opp_elixir_est_norm, time_norm, is_overtime]
        own_elx = self.elixir[player_id] / ELIXIR_MAX
        opp_elx = self.elixir[opp_id] / ELIXIR_MAX  # Observable or estimated
        time_norm = min(1.0, self.elapsed_time / MAX_MATCH_STEPS)
        is_ot = 1.0 if self.is_overtime else 0.0
        scalars = np.array([own_elx, opp_elx, time_norm, is_ot], dtype=np.float32)

        # 2. Hand Cards (4 hand cards + 1 next card)
        hand_cards = np.array(self.hand[player_id] + [self.next_card[player_id]], dtype=np.int64)

        # 3. Entities (Crown Towers ALWAYS included first -> guarantees NO empty boards / NaN)
        # Vector: [card_id_norm, side, x_norm, y_norm, hp_ratio, is_air, state] (7 dims)
        entity_list = []

        # Add all 6 Crown Towers
        for p in [player_id, opp_id]:
            is_enemy = 1.0 if p == opp_id else 0.0
            for t_name, tower in self.towers[p].items():
                if tower.alive:
                    # Canonical coordinates
                    if player_id == 1:
                        cx, cy = tower.x, tower.y
                    else:
                        cx, cy = tower.x, 32.0 - tower.y
                    card_code = 0.01 if tower.tower_type == "king" else 0.02
                    entity_list.append([
                        card_code,
                        is_enemy,
                        cx / float(ARENA_WIDTH),
                        cy / float(ARENA_HEIGHT),
                        tower.hp_ratio,
                        0.0,  # is_air
                        1.0   # state alive
                    ])

        # Add active troops/buildings
        for u in self.units:
            if u.hp > 0:
                is_enemy = 1.0 if u.owner == opp_id else 0.0
                if player_id == 1:
                    cx, cy = u.x, u.y
                else:
                    cx, cy = u.x, 32.0 - u.y
                entity_list.append([
                    u.card_id / float(len(CARDS_CATALOG) + 1),
                    is_enemy,
                    np.clip(cx / float(ARENA_WIDTH), 0.0, 1.0),
                    np.clip(cy / float(ARENA_HEIGHT), 0.0, 1.0),
                    u.hp_ratio,
                    1.0 if u.is_air else 0.0,
                    u.state
                ])

        # Truncate to MAX_ENTITIES
        entity_count = min(len(entity_list), MAX_ENTITIES)
        entities = np.zeros((MAX_ENTITIES, 7), dtype=np.float32)
        entity_mask = np.ones(MAX_ENTITIES, dtype=bool)  # True = padded/empty

        if entity_count > 0:
            entities[:entity_count] = np.array(entity_list[:entity_count], dtype=np.float32)
            entity_mask[:entity_count] = False

        # 4. Spatial Grid: [6, 32, 18]
        # Channels: [friendly_ground, friendly_air, enemy_ground, enemy_air, towers_buildings, valid_deployment]
        grid = np.zeros((6, ARENA_HEIGHT, ARENA_WIDTH), dtype=np.float32)

        for u in self.units:
            if u.hp > 0:
                is_opp = (u.owner == opp_id)
                gx = int(np.clip(u.x, 0, ARENA_WIDTH - 1))
                gy = int(np.clip(u.y if player_id == 1 else 31 - u.y, 0, ARENA_HEIGHT - 1))
                if not is_opp:
                    ch = 1 if u.is_air else 0
                else:
                    ch = 3 if u.is_air else 2
                grid[ch, gy, gx] += (u.hp_ratio)

        # Channel 4: Crown Towers on grid
        for p in [player_id, opp_id]:
            sign = -1.0 if p == opp_id else 1.0
            for tower in self.towers[p].values():
                if tower.alive:
                    gx = int(np.clip(tower.x, 0, ARENA_WIDTH - 1))
                    gy = int(np.clip(tower.y if player_id == 1 else 31 - tower.y, 0, ARENA_HEIGHT - 1))
                    grid[4, gy, gx] = sign * tower.hp_ratio

        # Channel 5: Legal deployment zone in canonical frame
        # Standard deployment is friendly half (y <= 15).
        # If enemy princess tower is destroyed, can place up to enemy side in that lane
        opp_princess_left_dead = not self.towers[opp_id]["princess_left"].alive
        opp_princess_right_dead = not self.towers[opp_id]["princess_right"].alive

        deploy_mask = np.zeros((ARENA_HEIGHT, ARENA_WIDTH), dtype=bool)
        deploy_mask[:16, :] = True  # Friendly half always valid

        if opp_princess_left_dead:
            deploy_mask[16:24, :9] = True  # Expanded left pocket
        if opp_princess_right_dead:
            deploy_mask[16:24, 9:] = True  # Expanded right pocket

        grid[5] = deploy_mask.astype(np.float32)

        # 5. Action Masks
        # Card mask: slots 0..3 affordable + slot 4 No-Op always allowed
        card_mask = np.zeros(5, dtype=bool)
        current_elixir = self.elixir[player_id]
        for slot in range(4):
            card_id = self.hand[player_id][slot]
            cost = CARDS_CATALOG[card_id]["cost"]
            if current_elixir >= cost:
                card_mask[slot] = True
        card_mask[4] = True  # No-Op / Wait is always legal

        # Flattened tile mask: [576]
        tile_mask = deploy_mask.flatten()

        return {
            "scalars": scalars,
            "hand_cards": hand_cards,
            "entities": entities,
            "entity_mask": entity_mask,
            "arena_grid": grid,
            "card_mask": card_mask,
            "tile_mask": tile_mask
        }

    def _spawn_card(self, player_id: int, card_slot: int, tile_x: int, tile_y: int):
        """Spawns unit(s) or triggers spell for a player."""
        card_id = self.hand[player_id][card_slot]
        cfg = CARDS_CATALOG[card_id]
        cost = cfg["cost"]

        # Deduct elixir
        self.elixir[player_id] = max(0.0, self.elixir[player_id] - cost)

        # Cycle card
        self.hand[player_id][card_slot] = self.next_card[player_id]
        self.deck_queue[player_id].append(card_id)
        self.next_card[player_id] = self.deck_queue[player_id].pop(0)

        # Spell handling
        if cfg.get("spell", False):
            # Instant or area spell effect
            radius = float(cfg.get("range", 2.5))
            dmg = float(cfg.get("dmg", 200.0))
            opp_id = 2 if player_id == 1 else 1
            # Damage enemy units in radius
            for u in self.units:
                if u.owner == opp_id and u.hp > 0:
                    dist = math.hypot(u.x - tile_x, u.y - tile_y)
                    if dist <= radius:
                        u.hp -= dmg
            # Damage enemy towers in radius (reduced crown tower damage 35%)
            tower_dmg = dmg * 0.35
            for tower in self.towers[opp_id].values():
                if tower.alive:
                    dist = math.hypot(tower.x - tile_x, tower.y - tile_y)
                    if dist <= radius + 1.0:
                        tower.hp -= tower_dmg
                        tower.is_activated = True
                        if tower.hp <= 0:
                            tower.alive = False
            return

        # Troop / Building spawn
        count = cfg.get("count", 1)
        for i in range(count):
            ox = tile_x + (0.5 * (i % 2) - 0.25)
            oy = tile_y + (0.5 * (i // 2) - 0.25)
            self.units.append(Unit(card_id, player_id, ox, oy))

    def step_both(self, p1_action: np.ndarray, p2_action: np.ndarray) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray], float, float, bool, bool, Dict[str, Any]
    ]:
        """
        Executes simultaneous actions for Player 1 and Player 2 with action_repeat frame-skipping.
        Actions are deployed on sub-tick 0, followed by action_repeat internal physics updates.
        """
        # 1. Execute Player 1 Action on sub-tick 0
        p1_slot, p1_tile = int(p1_action[0]), int(p1_action[1])
        p1_slow_roll = 0.0
        p1_spell_bait = 0.0
        if 0 <= p1_slot <= 3:
            p1_card = self.hand[1][p1_slot]
            if self.elixir[1] >= CARDS_CATALOG[p1_card]["cost"]:
                tx = p1_tile % ARENA_WIDTH
                ty = p1_tile // ARENA_WIDTH
                no_enemy_1 = not any(u.owner == 2 and u.alive and u.y < 16.0 for u in self.units)
                if ty <= 3 and self.elixir[1] >= 8.5 and no_enemy_1 and not CARDS_CATALOG[p1_card].get("spell", False):
                    p1_slow_roll = 0.05
                is_swarm_or_win = p1_card in (6, 7, 11, 14)
                if is_swarm_or_win and (self.elapsed_time - self.last_spell_time[2]) <= 3.0:
                    p1_spell_bait = 0.05
                if CARDS_CATALOG[p1_card].get("spell", False):
                    self.last_spell_time[1] = self.elapsed_time
                self._spawn_card(player_id=1, card_slot=p1_slot, tile_x=tx, tile_y=ty)

        # 2. Execute Player 2 Action on sub-tick 0 (Canonical Y-mirror back to environment frame)
        p2_slot, p2_tile = int(p2_action[0]), int(p2_action[1])
        p2_slow_roll = 0.0
        p2_spell_bait = 0.0
        if 0 <= p2_slot <= 3:
            p2_card = self.hand[2][p2_slot]
            if self.elixir[2] >= CARDS_CATALOG[p2_card]["cost"]:
                p2_cx = p2_tile % ARENA_WIDTH
                p2_cy = p2_tile // ARENA_WIDTH
                env_tx = p2_cx
                env_ty = 31 - p2_cy
                no_enemy_2 = not any(u.owner == 1 and u.alive and u.y > 16.0 for u in self.units)
                if p2_cy <= 3 and self.elixir[2] >= 8.5 and no_enemy_2 and not CARDS_CATALOG[p2_card].get("spell", False):
                    p2_slow_roll = 0.05
                is_swarm_or_win = p2_card in (6, 7, 11, 14)
                if is_swarm_or_win and (self.elapsed_time - self.last_spell_time[1]) <= 3.0:
                    p2_spell_bait = 0.05
                if CARDS_CATALOG[p2_card].get("spell", False):
                    self.last_spell_time[2] = self.elapsed_time
                self._spawn_card(player_id=2, card_slot=p2_slot, tile_x=env_tx, tile_y=env_ty)

        # 3. Simulate internal physics ticks across action_repeat (accumulating shaped rewards)
        p1_total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.action_repeat):
            self.step_count += 1
            self.elapsed_time += STEP_DT

            if self.elapsed_time >= REGULAR_TIME_SECS:
                self.is_overtime = True

            elixir_rate = ELIXIR_RATE_DOUBLE if self.is_overtime else ELIXIR_RATE_NORMAL
            self.elixir[1] = min(ELIXIR_MAX, self.elixir[1] + elixir_rate * STEP_DT)
            self.elixir[2] = min(ELIXIR_MAX, self.elixir[2] + elixir_rate * STEP_DT)

            # 1.5x Offensive Pressure Incentive on lowest-HP enemy Princess Tower
            p1_enemy_pts = [t for name, t in self.towers[2].items() if name != "king" and t.alive]
            lowest_p2_pt = min(p1_enemy_pts, key=lambda t: t.hp) if p1_enemy_pts else None

            p2_enemy_pts = [t for name, t in self.towers[1].items() if name != "king" and t.alive]
            lowest_p1_pt = min(p2_enemy_pts, key=lambda t: t.hp) if p2_enemy_pts else None

            p2_tower_hps_before = {k: t.hp for k, t in self.towers[2].items()}
            p1_tower_hps_before = {k: t.hp for k, t in self.towers[1].items()}

            self._simulate_tick()

            p1_dmg_dealt = 0.0
            for k, t in self.towers[2].items():
                dmg = max(0.0, p2_tower_hps_before[k] - t.hp)
                p1_dmg_dealt += (dmg * 1.5) if (lowest_p2_pt is not None and t == lowest_p2_pt) else dmg

            p1_dmg_taken = 0.0
            for k, t in self.towers[1].items():
                dmg = max(0.0, p1_tower_hps_before[k] - t.hp)
                p1_dmg_taken += (dmg * 1.5) if (lowest_p1_pt is not None and t == lowest_p1_pt) else dmg

            p1_total_reward += (p1_dmg_dealt - p1_dmg_taken) * 0.001

            # Termination checks
            if not self.towers[1]["king"].alive:
                terminated = True
                p1_total_reward -= 1.0
                break
            elif not self.towers[2]["king"].alive:
                terminated = True
                p1_total_reward += 1.0
                break
            elif self.step_count >= MAX_MATCH_STEPS:
                truncated = True
                break

        p2_total_reward = -p1_total_reward

        # Tactical Micro Pseudo-Rewards
        p1_kite = 0.0
        p2_kite = 0.0
        p1_towers_alive = (self.towers[1]["princess_left"].alive and self.towers[1]["princess_right"].alive)
        p2_towers_alive = (self.towers[2]["princess_left"].alive and self.towers[2]["princess_right"].alive)

        for u in self.units:
            if not u.alive:
                continue
            if u.owner == 2 and u.y < 16.0:
                if (7.5 <= u.x <= 10.5 and p1_towers_alive) or (
                    ((u.x - 3.5)**2 + (u.y - 6.5)**2)**0.5 <= 7.5 and ((u.x - 14.5)**2 + (u.y - 6.5)**2)**0.5 <= 7.5
                ):
                    p1_kite = 0.015
            elif u.owner == 1 and u.y > 16.0:
                if (7.5 <= u.x <= 10.5 and p2_towers_alive) or (
                    ((u.x - 3.5)**2 + (u.y - 25.5)**2)**0.5 <= 7.5 and ((u.x - 14.5)**2 + (u.y - 25.5)**2)**0.5 <= 7.5
                ):
                    p2_kite = 0.015

        p1_tank = 0.0
        p2_tank = 0.0
        tanks_1 = [u for u in self.units if u.owner == 1 and u.alive and getattr(u, "max_hp", 0) >= 900]
        supports_1 = [u for u in self.units if u.owner == 1 and u.alive and getattr(u, "range", 1.0) >= 3.5]
        for t in tanks_1:
            for s in supports_1:
                if t.y > s.y and ((t.x - s.x)**2 + (t.y - s.y)**2)**0.5 <= 4.0:
                    p1_tank = 0.015
                    break
            if p1_tank > 0:
                break

        tanks_2 = [u for u in self.units if u.owner == 2 and u.alive and getattr(u, "max_hp", 0) >= 900]
        supports_2 = [u for u in self.units if u.owner == 2 and u.alive and getattr(u, "range", 1.0) >= 3.5]
        for t in tanks_2:
            for s in supports_2:
                if t.y < s.y and ((t.x - s.x)**2 + (t.y - s.y)**2)**0.5 <= 4.0:
                    p2_tank = 0.015
                    break
            if p2_tank > 0:
                break

        # Accumulate micro-shaping pseudo-rewards with strict +0.25 cumulative episode cap (prevents micro reward farming)
        MAX_MICRO_EPISODE = 0.25
        p1_micro_raw = p1_slow_roll + p1_spell_bait + p1_kite + p1_tank
        max_allowed_1 = max(0.0, MAX_MICRO_EPISODE - self.episode_micro_rewards[1])
        p1_micro = min(p1_micro_raw, max_allowed_1)
        self.episode_micro_rewards[1] += p1_micro

        p2_micro_raw = p2_slow_roll + p2_spell_bait + p2_kite + p2_tank
        max_allowed_2 = max(0.0, MAX_MICRO_EPISODE - self.episode_micro_rewards[2])
        p2_micro = min(p2_micro_raw, max_allowed_2)
        self.episode_micro_rewards[2] += p2_micro

        p1_total_reward += p1_micro
        p2_total_reward += p2_micro

        if truncated and not terminated:
            # Overtime / max steps tiebreaker
            p1_crowns = sum(1 for t in self.towers[2].values() if not t.alive)
            p2_crowns = sum(1 for t in self.towers[1].values() if not t.alive)
            if p1_crowns > p2_crowns:
                p1_total_reward += 1.0
                p2_total_reward -= 1.0
            elif p2_crowns > p1_crowns:
                p1_total_reward -= 1.0
                p2_total_reward += 1.0
            else:
                # Sudden-death tiebreaker: compare lowest remaining active tower HP
                p1_min_hp = min((t.hp for t in self.towers[1].values() if t.alive), default=0.0)
                p2_min_hp = min((t.hp for t in self.towers[2].values() if t.alive), default=0.0)
                if p1_min_hp > p2_min_hp:
                    p1_total_reward += 0.5
                    p2_total_reward -= 0.5
                elif p2_min_hp > p1_min_hp:
                    p1_total_reward -= 0.5
                    p2_total_reward += 0.5
                else:
                    # Turtle Trap Stalemate Draw Penalty: Both policies lose reward for passive draws
                    p1_total_reward -= 0.5
                    p2_total_reward -= 0.5

        obs_p1 = self.get_obs(player_id=1)
        obs_p2 = self.get_obs(player_id=2)

        info = {
            "p1_crowns": sum(1 for t in self.towers[2].values() if not t.alive),
            "p2_crowns": sum(1 for t in self.towers[1].values() if not t.alive),
            "step": self.step_count,
            "elapsed_time": self.elapsed_time
        }

        return obs_p1, obs_p2, p1_total_reward, p2_total_reward, terminated, truncated, info

    def _simulate_tick(self):
        """Advances physical troop movements, targeting, and tower defense."""
        # 1. Update Unit attacks and movements
        for u in self.units:
            if u.hp <= 0:
                continue

            opp_id = 2 if u.owner == 1 else 1

            # Find nearest target (enemy unit or tower)
            target = None
            min_dist = float("inf")

            # Check enemy units
            if u.targets in ("ground", "both"):
                for eu in self.units:
                    if eu.owner == opp_id and eu.hp > 0:
                        if not eu.is_air or u.targets == "both":
                            d = math.hypot(eu.x - u.x, eu.y - u.y)
                            if d < min_dist:
                                min_dist = d
                                target = eu

            # Check enemy towers
            for tower in self.towers[opp_id].values():
                if tower.alive:
                    d = math.hypot(tower.x - u.x, tower.y - u.y)
                    # Building targeters prioritize towers over units
                    if u.targets == "building":
                        if d < min_dist:
                            min_dist = d
                            target = tower
                    elif target is None or d < min_dist:
                        min_dist = d
                        target = tower

            if target is not None:
                # Attack if in range
                if min_dist <= u.attack_range:
                    u.attack_cooldown -= STEP_DT
                    if u.attack_cooldown <= 0:
                        target.hp -= u.dmg
                        u.attack_cooldown = u.hit_speed
                        if isinstance(target, Tower) and target.hp <= 0:
                            target.alive = False
                else:
                    # Move towards target
                    dx = target.x - u.x
                    dy = target.y - u.y
                    norm = math.hypot(dx, dy)
                    if norm > 0.01:
                        move_dist = u.speed * STEP_DT
                        u.x += (dx / norm) * move_dist
                        u.y += (dy / norm) * move_dist
            else:
                # Default march forward (Player 1 moves up +y, Player 2 moves down -y)
                dir_y = 1.0 if u.owner == 1 else -1.0
                u.y += dir_y * u.speed * STEP_DT

        # 2. Towers defend
        for pid in [1, 2]:
            opp_id = 2 if pid == 1 else 1
            for tower in self.towers[pid].values():
                if not tower.alive:
                    continue
                if tower.tower_type == "king" and not tower.is_activated:
                    continue

                tower.attack_cooldown -= STEP_DT
                # Find nearest enemy within tower range
                closest_enemy = None
                closest_d = tower.attack_range
                for u in self.units:
                    if u.owner == opp_id and u.hp > 0:
                        d = math.hypot(u.x - tower.x, u.y - tower.y)
                        if d <= closest_d:
                            closest_d = d
                            closest_enemy = u

                if closest_enemy is not None and tower.attack_cooldown <= 0:
                    closest_enemy.hp -= tower.dmg
                    tower.attack_cooldown = tower.hit_speed

        # Remove dead units
        self.units = [u for u in self.units if u.hp > 0]

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Standard Gymnasium single-agent step where Player 2 takes No-Op (or bot)."""
        p2_noop = np.array([4, 0], dtype=np.int64)
        obs_p1, _, r1, _, term, trunc, info = self.step_both(action, p2_noop)
        return obs_p1, r1, term, trunc, info

