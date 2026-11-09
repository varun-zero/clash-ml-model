"""
Full-Fidelity Headless Clash Royale Environment Adapter.
Bridges the parameter-driven crsim engine (164 cards, decoded logic tables, level-scaling)
to the AlphaStarCRNet and OpenAI Five PPO/GAE trainer pipeline.

Guarantees:
- Canonical Player 2 Perspective symmetry (Y-axis mirror + friendly/enemy channel inversion).
- Crown Towers always in the first 6 entity slots (guarantees Transformer never encounters empty entity boards / NaN).
- Action repeat (frame-skip = 4 ticks / 400ms decision cadence).
- FP32-safe action masks (card availability + legal deployment zones).
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from crsim.engine import (
    Battle, Tower as CrTower, Unit as CrUnit, Projectile as CrProj,
    load_model, ARENA_W, ARENA_H, RIVER_Y, RIVER_HALF, MAX_ELIXIR
)
from cr_alphastar_agent.config import (
    ARENA_WIDTH, ARENA_HEIGHT, TOTAL_TILES, MAX_ENTITIES,
    ACTION_REPEAT, DEFAULT_DECK, CARDS_CATALOG, SPELL_CARD_IDS
)

# Card ID to crsim card name mapping
ID_TO_CRSIM_NAME = {
    1: "Knight",
    2: "Archer",
    3: "Giant",
    4: "Musketeer",
    5: "MiniPekka",
    6: "Skeletons",
    7: "Minions",
    8: "Fireball",
    9: "Arrows",
    10: "Valkyrie",
    11: "HogRider",
    12: "BabyDragon",
    13: "Cannon",
    14: "GoblinBarrel",
    15: "MegaMinion",
    16: "Zap",
}
CRSIM_NAME_TO_ID = {v: k for k, v in ID_TO_CRSIM_NAME.items()}
CRSIM_NAME_TO_ID.update({
    # Common name variants & exact strings
    "Archers": 2,
    "Hog Rider": 11,
    "Mini PEKKA": 5,
    "Baby Dragon": 12,
    "Goblin Barrel": 14,
    "Mega Minion": 15,
    # 1: Melee Mini-Tanks
    "Knight": 1,
    "IceGolemite": 1,
    "Ice Golem": 1,
    "Ghost": 1,
    "Bandit": 1,
    "BossBandit": 1,
    "Miner": 1,
    # 2: Ranged Single/Duo
    "Archer": 2,
    "Princess": 2,
    "DartGoblin": 2,
    "BlowdartGoblin": 2,
    "SpearGoblins": 2,
    # 3: Heavy Tanks
    "Giant": 3,
    "Golem": 3,
    "LavaHound": 3,
    "RoyalGiant": 3,
    # 4: High-DPS Ranged / Spawners
    "Musketeer": 4,
    "Wizard": 4,
    "ElectroWizard": 4,
    "Witch": 4,
    "DarkWitch": 4,
    "FlyingMachine": 4,
    # 5: Heavy Melee / Tank Busters (Authentic PEKKA & MiniPekka)
    "MiniPekka": 5,
    "Pekka": 5,
    "PEKKA": 5,
    "Prince": 5,
    "EliteBarbarians": 5,
    # 6: Cheap Ground Swarm / Cycle
    "Skeletons": 6,
    "SkeletonArmy": 6,
    "Goblins": 6,
    "Goblin": 6,
    "GoblinGang": 6,
    "IceSpirits": 6,
    "IceSpirit": 6,
    "Tombstone": 6,
    # 7: Air Swarms
    "Minions": 7,
    "MinionHorde": 7,
    "Bats": 7,
    # 8: Medium/Heavy Spells
    "Fireball": 8,
    "Poison": 8,
    "Rocket": 8,
    "Lightning": 8,
    "Earthquake": 8,
    # 9: Area Spells
    "Arrows": 9,
    "Snowball": 9,
    # 10: Splash Mini-Tanks & Rushers
    "Valkyrie": 10,
    "DarkPrince": 10,
    "Bomber": 10,
    "Bowler": 10,
    "MegaKnight": 10,
    # 11: Bridge Rushers / Win Conditions
    "HogRider": 11,
    "BattleRam": 11,
    "RamRider": 11,
    "Balloon": 11,
    "WallBreakers": 11,
    # 12: Air Dragons & Flying Splash
    "BabyDragon": 12,
    "InfernoDragon": 12,
    "SkeletonDragons": 12,
    # 13: Defensive Buildings
    "Cannon": 13,
    "Tesla": 13,
    "InfernoTower": 13,
    "BombTower": 13,
    "Mortar": 13,
    "Xbow": 13,
    # 14: Direct Spell Win Condition / Bait
    "GoblinBarrel": 14,
    # 15: Heavy Single-Target Air
    "MegaMinion": 15,
    # 16: Fast Small Spells
    "Zap": 16,
    "Log": 16,
    "BarbLog": 16,
    "Tornado": 16,
})

# Unit name to representative card ID mapping for entities tensor
UNIT_NAME_TO_CARD_ID = {
    "KingTower": 0,
    "PrincessTower": 0,
    "TowerPrincess": 0,
    # Melee Mini-Tanks
    "Knight": 1,
    "KnightHero": 1,
    "IceGolemite": 1,
    "IceGolem": 1,
    "SuperIceGolemite": 1,
    "Barbarian": 1,
    "Barbarian_EV1": 1,
    "AngryBarbarian": 1,
    "Berserker": 1,
    "Bandit": 1,
    "BossBandit": 1,
    "Miner": 1,
    "RoyalRecruit": 1,
    # Ranged Single/Spread
    "Archer": 2,
    "Archers": 2,
    "Archer_EV1": 2,
    "BlowdartGoblin": 2,
    "DartGoblin": 2,
    "SpearGoblin": 2,
    "Princess": 2,
    "Princess_EV1": 2,
    "Firecracker": 2,
    "Firecracker_EV1": 2,
    # Heavy Tanks
    "Giant": 3,
    "Golem": 3,
    "Golemite": 3,
    "LavaHound": 3,
    "RoyalGiant": 3,
    "ElectroGiant": 3,
    "ElixirGolem": 3,
    # High-DPS Ranged & Spawners
    "Musketeer": 4,
    "ThreeMusketeers": 4,
    "Wizard": 4,
    "ElectroWizard": 4,
    "IceWizard": 4,
    "Witch": 4,
    "DarkWitch": 4,
    "MagicArcher": 4,
    "Hunter": 4,
    "FlyingMachine": 4,
    # Tank Busters / Heavy Melee
    "MiniPekka": 5,
    "Pekka": 5,
    "Pekka_EV1": 5,
    "Prince": 5,
    "EliteBarbarian": 5,
    "Lumberjack": 5,
    # Ground Swarm
    "Skeleton": 6,
    "Skeletons": 6,
    "SkeletonArmy": 6,
    "IceSpirits": 6,
    "IceSpirit": 6,
    "Guard": 6,
    "Tombstone": 6,
    # Air Swarm
    "Minion": 7,
    "Minions": 7,
    "MinionHorde": 7,
    "Bat": 7,
    "Bats": 7,
    "Bat_EV1": 7,
    "LavaPup": 7,
    # Splash Mini-Tanks & Rushers
    "Valkyrie": 10,
    "Valkyrie_EV1": 10,
    "DarkPrince": 10,
    "MegaKnight": 10,
    "Bomber": 10,
    "Bomber_EV1": 10,
    "Executioner": 10,
    "AxeMan": 10,
    "AxeMan_EV1": 10,
    "Bowler": 10,
    "Ghost": 10,
    "RoyalGhost": 10,
    # Fast Building Targeters
    "HogRider": 11,
    "BattleRam": 11,
    "BattleRam_EV1": 11,
    "RamRider": 11,
    "WallBreaker": 11,
    "WallBreakers": 11,
    "Balloon": 11,
    # Air Flying Units
    "BabyDragon": 12,
    "BabyDragon_EV1": 12,
    "InfernoDragon": 12,
    "SkeletonDragon": 12,
    "Phoenix": 12,
    # Defensive Buildings
    "Cannon": 13,
    "Cannon_EV1": 13,
    "Tesla": 13,
    "InfernoTower": 13,
    "BombTower": 13,
    "Mortar": 13,
    "XBow": 13,
    "GoblinCage": 13,
    "GoblinDrill": 13,
    # Fast Ground Runners / Goblins
    "Goblin": 14,
    "Goblins": 14,
    "Goblin_Stab": 14,
    "BarrelGoblin": 14,
    "GoblinGang": 14,
    "RascalBoy": 1,
    "RascalGirl": 2,
    # Air High Single-Target
    "MegaMinion": 15,
}

TOWER_PENALTY = 0.001
CROWN_REWARD = 1.0
WIN_REWARD = 3.0


class CrSimReplicaEnv:
    """
    Gymnasium-compatible environment backed by crsim.engine.Battle.
    Provides identical observation shapes and action signatures as ClashRoyaleReplicaEnv.
    """
    def __init__(
        self,
        blue_deck: Optional[List[int]] = None,
        red_deck: Optional[List[int]] = None,
        seed: Optional[int] = None,
        action_repeat: int = ACTION_REPEAT,
        model: Optional[dict] = None
    ):
        self.blue_deck_ids = list(blue_deck or DEFAULT_DECK)
        self.red_deck_ids = list(red_deck or DEFAULT_DECK)
        self.action_repeat = action_repeat
        self.model = model if model is not None else load_model()
        self.rng = np.random.default_rng(seed)
        self.battle: Optional[Battle] = None
        self.prev_tower_hp = [{}, {}]
        self.episode_micro_rewards = [0.0, 0.0]
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        deck0 = [cid if isinstance(cid, str) else ID_TO_CRSIM_NAME.get(cid, "Knight") for cid in self.blue_deck_ids]
        deck1 = [cid if isinstance(cid, str) else ID_TO_CRSIM_NAME.get(cid, "Knight") for cid in self.red_deck_ids]

        self.battle = Battle([deck0, deck1], model=self.model, seed=int(self.rng.integers(0, 2**31)))
        self.prev_tower_hp = [self._get_tower_hps(0), self._get_tower_hps(1)]
        self.last_spell_time = [-999999, -999999]
        self.episode_micro_rewards = [0.0, 0.0]
        return self.get_obs(1), {}

    def _get_tower_hps(self, team: int) -> Dict[str, float]:
        """Returns dict of tower HPs for a team: king, princess_left, princess_right."""
        b = self.battle
        towers = [u for u in b.units if isinstance(u, CrTower) and u.team == team]
        out = {"king": 0.0, "princess_left": 0.0, "princess_right": 0.0}
        for t in towers:
            kind = getattr(t, "kind", "")
            if kind == "KingTower":
                out["king"] = float(t.hp)
            elif t.x < 9.0:
                out["princess_left"] = float(t.hp)
            else:
                out["princess_right"] = float(t.hp)
        return out

    def get_obs(self, player_id: int) -> Dict[str, np.ndarray]:
        """
        Returns canonical perspective observation for player_id (1 or 2).
        player_id 1 is team 0 (blue, bottom).
        player_id 2 is team 1 (red, top).
        """
        b = self.battle
        team = player_id - 1
        opp_team = 1 - team

        # 1. Scalars: [own_elixir/10, opp_elixir/10, time_remaining/180, is_overtime]
        own_elixir = float(np.clip(b.elixir[team] / 10.0, 0.0, 1.0))
        opp_elixir = float(np.clip(b.elixir[opp_team] / 10.0, 0.0, 1.0))
        time_left = max(0.0, (180_000 - b.t) / 180_000)
        is_ot = 1.0 if b.t >= 180_000 else 0.0
        scalars = np.array([own_elixir, opp_elixir, time_left, is_ot], dtype=np.float32)

        # 2. Hand Cards: [4 hand cards, 1 next card]
        hand_cnames = b.hand[team]
        cycle_cnames = b.cycle[team]
        hand_ids = [CRSIM_NAME_TO_ID.get(c, 1) for c in hand_cnames]
        next_id = CRSIM_NAME_TO_ID.get(cycle_cnames[0], 1) if cycle_cnames else 1
        hand_cards = np.array(hand_ids + [next_id], dtype=np.int64)

        # 3. Entities & Entity Mask: Shape [40, 7]
        # Feature: [card_id_norm, side (+1/-1), x_norm, y_norm, hp_ratio, is_air, state]
        entities = np.zeros((MAX_ENTITIES, 7), dtype=np.float32)
        entity_mask = np.ones(MAX_ENTITIES, dtype=bool)

        # Sort towers first so Crown Towers always occupy slots 0..5
        all_towers = [u for u in b.units if isinstance(u, CrTower)]
        regular_units = [u for u in b.units if not isinstance(u, CrTower) and u.hp > 0]
        active_entities = all_towers + regular_units

        for idx, u in enumerate(active_entities[:MAX_ENTITIES]):
            cid = UNIT_NAME_TO_CARD_ID.get(u.name, 1)
            is_own = (u.team == team)
            side = 1.0 if is_own else -1.0

            # Canonical Coordinate Remapping
            if player_id == 1:
                cx = u.x
                cy = u.y
            else:
                # Mirror for player 2: x remains x, y mirrored 32 - y
                cx = u.x
                cy = 32.0 - u.y

            norm_x = np.clip(cx / ARENA_W, 0.0, 1.0)
            norm_y = np.clip(cy / ARENA_H, 0.0, 1.0)
            hp_ratio = float(np.clip(u.hp / max(1, u.maxhp), 0.0, 1.0))
            is_air = 1.0 if getattr(u, "flying", False) else 0.0
            state = 1.0 if u.hp > 0 else 0.0

            entities[idx] = [cid / 16.0, side, norm_x, norm_y, hp_ratio, is_air, state]
            entity_mask[idx] = False

        # Fallback guard
        if entity_mask.all():
            entity_mask[0] = False

        # 4. Spatial Grid State: Shape (6, 32, 18)
        grid = np.zeros((6, ARENA_HEIGHT, ARENA_WIDTH), dtype=np.float32)

        for u in b.units:
            if u.hp <= 0:
                continue
            is_own = (u.team == team)
            gx = int(np.clip(u.x, 0, ARENA_WIDTH - 1))
            gy = int(np.clip(u.y if player_id == 1 else 31 - u.y, 0, ARENA_HEIGHT - 1))
            hp_frac = float(u.hp / max(1, u.maxhp))

            if isinstance(u, CrTower) or getattr(u, "is_building", False):
                # Channel 4: Buildings & Towers
                grid[4, gy, gx] = (1.0 if is_own else -1.0) * hp_frac
            else:
                is_flying = getattr(u, "flying", False)
                if is_own:
                    grid[1 if is_flying else 0, gy, gx] += hp_frac
                else:
                    grid[3 if is_flying else 2, gy, gx] += hp_frac

        # Channel 5: broad friendly-side deployment prior; action legality is
        # computed per card below from the engine to avoid masking valid spells.
        deploy_mask = np.zeros((ARENA_HEIGHT, ARENA_WIDTH), dtype=bool)
        deploy_mask[:16, :] = True  # Friendly half

        # If enemy princess tower dead, expand pocket
        opp_towers = self._get_tower_hps(opp_team)
        if opp_towers["princess_left"] <= 0:
            deploy_mask[16:24, :9] = True
        if opp_towers["princess_right"] <= 0:
            deploy_mask[16:24, 9:] = True

        grid[5] = deploy_mask.astype(np.float32)

        # 5. Card and placement masks. The hand remains four distinct slots;
        # placement legality comes from the simulator for each card.
        card_mask = np.zeros(5, dtype=bool)
        card_tiles = np.zeros((5, TOTAL_TILES), dtype=bool)
        for slot in range(4):
            if slot < len(b.hand[team]):
                cname = b.hand[team][slot]
                for tile in range(TOTAL_TILES):
                    x, y = tile % ARENA_WIDTH, tile // ARENA_WIDTH
                    if player_id == 2:
                        y = ARENA_HEIGHT - 1 - y
                    if b.legal_action(team, slot, float(x), float(y)):
                        card_tiles[slot, tile] = True
                card_mask[slot] = bool(card_tiles[slot].any())
        card_mask[4] = True  # No-Op / Wait is always legal
        card_tiles[4] = True
        # Keep legacy tile_mask as union; network uses per-card mask when present.
        tile_mask = card_tiles[:4].any(axis=0)

        return {
            "scalars": scalars,
            "hand_cards": hand_cards,
            "entities": entities,
            "entity_mask": entity_mask,
            "arena_grid": grid,
            "card_mask": card_mask,
            "tile_mask": tile_mask
            ,"card_tile_mask": card_tiles
        }

    def step_both(self, p1_action: np.ndarray, p2_action: np.ndarray) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray], float, float, bool, bool, Dict[str, Any]
    ]:
        """
        Executes actions for Player 1 and Player 2 and advances action_repeat ticks.
        """
        b = self.battle
        if b is None or b.done:
            obs1 = self.get_obs(1)
            obs2 = self.get_obs(2)
            return obs1, obs2, 0.0, 0.0, True, False, {}

        # 1. Player 1 Action
        p1_slot, p1_tile = int(p1_action[0]), int(p1_action[1])
        p1_slow_roll_bonus = 0.0
        p1_spell_bait_bonus = 0.0
        if 0 <= p1_slot <= 3 and p1_slot < len(b.hand[0]):
            cname = b.hand[0][p1_slot]
            cost = b.card_cost(cname)
            if b.elixir[0] >= cost and b.play_cd[0] <= 0:
                tx = float(p1_tile % ARENA_WIDTH)
                ty = float(p1_tile // ARENA_WIDTH)
                if b.legal_action(0, p1_slot, tx, ty):
                    # Slow-Roll Investment Bonus (+0.05): Deploy behind King (ty <= 3.0), elixir >= 8.5, no enemy on friendly side
                    no_enemy_p1 = not any(u.team == 1 and u.hp > 0 and u.y < 16.0 for u in b.units)
                    if ty <= 3.0 and b.elixir[0] >= 8.5 and no_enemy_p1 and b._is_unit_card(b._card_meta(cname)):
                        p1_slow_roll_bonus = 0.05
                    # Spell-Bait Punish Bonus (+0.05): Swarm or win condition deployed within 3s of opponent exhausting a spell
                    is_swarm_or_win = cname in ("GoblinBarrel", "HogRider", "Skeletons", "Bats", "GoblinGang", "Minions")
                    if is_swarm_or_win and (b.t - self.last_spell_time[1]) <= 3000:
                        p1_spell_bait_bonus = 0.05
                    if cname in ("Zap", "Log", "Arrows", "Fireball", "Poison"):
                        self.last_spell_time[0] = b.t
                    b.play(0, p1_slot, tx, ty)

        # 2. Player 2 Action (Mirror Y back to engine frame)
        p2_slot, p2_tile = int(p2_action[0]), int(p2_action[1])
        p2_slow_roll_bonus = 0.0
        p2_spell_bait_bonus = 0.0
        if 0 <= p2_slot <= 3 and p2_slot < len(b.hand[1]):
            cname = b.hand[1][p2_slot]
            cost = b.card_cost(cname)
            if b.elixir[1] >= cost and b.play_cd[1] <= 0:
                p2_cx = float(p2_tile % ARENA_WIDTH)
                p2_cy = float(p2_tile // ARENA_WIDTH)
                env_tx = p2_cx
                env_ty = 31.0 - p2_cy
                if b.legal_action(1, p2_slot, env_tx, env_ty):
                    no_enemy_p2 = not any(u.team == 0 and u.hp > 0 and u.y > 16.0 for u in b.units)
                    if p2_cy <= 3.0 and b.elixir[1] >= 8.5 and no_enemy_p2 and b._is_unit_card(b._card_meta(cname)):
                        p2_slow_roll_bonus = 0.05
                    is_swarm_or_win = cname in ("GoblinBarrel", "HogRider", "Skeletons", "Bats", "GoblinGang", "Minions")
                    if is_swarm_or_win and (b.t - self.last_spell_time[0]) <= 3000:
                        p2_spell_bait_bonus = 0.05
                    if cname in ("Zap", "Log", "Arrows", "Fireball", "Poison"):
                        self.last_spell_time[1] = b.t
                    b.play(1, p2_slot, env_tx, env_ty)

        # 3. Simulate action_repeat ticks
        for _ in range(self.action_repeat):
            if b.done:
                break
            b.step([])

        # 4. Rewards with 1.5x Offensive Pressure Incentive on lowest-HP enemy Princess Tower
        cur_t0 = self._get_tower_hps(0)
        cur_t1 = self._get_tower_hps(1)

        # Identify lowest-HP active princess towers before step
        p1_enemy_pt = [k for k in ("princess_left", "princess_right") if self.prev_tower_hp[1].get(k, 0) > 0]
        lowest_t1 = min(p1_enemy_pt, key=lambda k: self.prev_tower_hp[1][k]) if p1_enemy_pt else None

        p0_enemy_pt = [k for k in ("princess_left", "princess_right") if self.prev_tower_hp[0].get(k, 0) > 0]
        lowest_t0 = min(p0_enemy_pt, key=lambda k: self.prev_tower_hp[0][k]) if p0_enemy_pt else None

        t1_lost_weighted = 0.0
        for k in cur_t1:
            dmg = max(0.0, self.prev_tower_hp[1][k] - cur_t1[k])
            t1_lost_weighted += (dmg * 1.5) if (lowest_t1 is not None and k == lowest_t1) else dmg

        t0_lost_weighted = 0.0
        for k in cur_t0:
            dmg = max(0.0, self.prev_tower_hp[0][k] - cur_t0[k])
            t0_lost_weighted += (dmg * 1.5) if (lowest_t0 is not None and k == lowest_t0) else dmg

        r1 = (t1_lost_weighted - t0_lost_weighted) * TOWER_PENALTY
        r2 = (t0_lost_weighted - t1_lost_weighted) * TOWER_PENALTY

        # 5. Tactical Micro Shaping Pseudo-Rewards
        p1_kite_bonus = 0.0
        p2_kite_bonus = 0.0
        p1_tank_bonus = 0.0
        p2_tank_bonus = 0.0

        p1_towers_alive = (cur_t0["princess_left"] > 0 and cur_t0["princess_right"] > 0)
        p2_towers_alive = (cur_t1["princess_left"] > 0 and cur_t1["princess_right"] > 0)

        # Dual-Tower Kite Bonus (+0.015 per RL step):
        for u in b.units:
            if u.hp <= 0 or isinstance(u, CrTower) or getattr(u, "is_building", False):
                continue
            # P1 Kiting: Enemy unit (team 1) pulled into center or double-tower range on P1 side (y < 16.0)
            if u.team == 1 and u.y < 16.0:
                if (7.5 <= u.x <= 10.5 and p1_towers_alive) or (
                    ((u.x - 3.5)**2 + (u.y - 6.5)**2)**0.5 <= 7.5 and ((u.x - 14.5)**2 + (u.y - 6.5)**2)**0.5 <= 7.5
                ):
                    p1_kite_bonus = 0.015
            # P2 Kiting: Enemy unit (team 0) pulled into center or double-tower range on P2 side (y > 16.0)
            elif u.team == 0 and u.y > 16.0:
                if (7.5 <= u.x <= 10.5 and p2_towers_alive) or (
                    ((u.x - 3.5)**2 + (u.y - 25.5)**2)**0.5 <= 7.5 and ((u.x - 14.5)**2 + (u.y - 25.5)**2)**0.5 <= 7.5
                ):
                    p2_kite_bonus = 0.015

        # Tank-Trick Shielding Bonus (+0.015 per RL step):
        tanks_0 = [u for u in b.units if u.team == 0 and u.hp > 0 and not isinstance(u, CrTower) and not getattr(u, "is_building", False) and getattr(u, "maxhp", 0) >= 900]
        supports_0 = [u for u in b.units if u.team == 0 and u.hp > 0 and not isinstance(u, CrTower) and not getattr(u, "is_building", False) and getattr(u, "range", 1.0) >= 3.5]
        for t in tanks_0:
            for s in supports_0:
                if t.y > s.y and ((t.x - s.x)**2 + (t.y - s.y)**2)**0.5 <= 4.0:
                    p1_tank_bonus = 0.015
                    break
            if p1_tank_bonus > 0:
                break

        tanks_1 = [u for u in b.units if u.team == 1 and u.hp > 0 and not isinstance(u, CrTower) and not getattr(u, "is_building", False) and getattr(u, "maxhp", 0) >= 900]
        supports_1 = [u for u in b.units if u.team == 1 and u.hp > 0 and not isinstance(u, CrTower) and not getattr(u, "is_building", False) and getattr(u, "range", 1.0) >= 3.5]
        for t in tanks_1:
            for s in supports_1:
                if t.y < s.y and ((t.x - s.x)**2 + (t.y - s.y)**2)**0.5 <= 4.0:
                    p2_tank_bonus = 0.015
                    break
            if p2_tank_bonus > 0:
                break

        # Accumulate micro-shaping pseudo-rewards with strict +0.25 cumulative episode cap (prevents micro reward farming)
        MAX_MICRO_EPISODE = 0.25
        p1_micro_raw = p1_slow_roll_bonus + p1_spell_bait_bonus + p1_kite_bonus + p1_tank_bonus
        max_allowed_p1 = max(0.0, MAX_MICRO_EPISODE - self.episode_micro_rewards[0])
        p1_micro = min(p1_micro_raw, max_allowed_p1)
        self.episode_micro_rewards[0] += p1_micro

        p2_micro_raw = p2_slow_roll_bonus + p2_spell_bait_bonus + p2_kite_bonus + p2_tank_bonus
        max_allowed_p2 = max(0.0, MAX_MICRO_EPISODE - self.episode_micro_rewards[1])
        p2_micro = min(p2_micro_raw, max_allowed_p2)
        self.episode_micro_rewards[1] += p2_micro

        r1 += p1_micro
        r2 += p2_micro

        self.prev_tower_hp = [cur_t0, cur_t1]

        # Terminal rewards
        terminated = b.done
        truncated = False

        if b.done:
            if b.result == 0:
                r1 += WIN_REWARD + CROWN_REWARD * b.crowns[0]
                r2 -= WIN_REWARD * 0.5
            elif b.result == 1:
                r2 += WIN_REWARD + CROWN_REWARD * b.crowns[1]
                r1 -= WIN_REWARD * 0.5
            else:
                # Stalemate / Turtle Trap Penalty: Penalize passive draws to incentivize closing games
                r1 -= 0.5
                r2 -= 0.5

        obs1 = self.get_obs(1)
        obs2 = self.get_obs(2)
        return obs1, obs2, float(r1), float(r2), terminated, truncated, {"crowns": b.crowns}
