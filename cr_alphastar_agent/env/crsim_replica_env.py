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
    load_model, ARENA_W, ARENA_H, RIVER_Y, RIVER_HALF, MAX_ELIXIR, TICK_MS
)
from cr_alphastar_agent.config import (
    ARENA_WIDTH, ARENA_HEIGHT, TOTAL_TILES, MAX_ENTITIES,
    ACTION_REPEAT, DEFAULT_DECK, CARDS_CATALOG, SPELL_CARD_IDS,
    WIN_CONDITION_IDS, DEFENSIVE_CARD_IDS, DECK_NAME_ALIASES,
    OPP_ELIXIR_SOURCE, DEFENSE_PRIORITY_WEIGHT, SPELL_VALUE_WEIGHT,
    SPELL_MISUSE_WEIGHT, SUPPORT_DEFEND_WEIGHT, PLACEMENT_VARIETY_WEIGHT,
    SPELL_SAFE_MASK, ELIXIR_LEAK_WEIGHT
)
from cr_alphastar_agent.env.opponent_elixir import OpponentElixirTracker
from cr_alphastar_agent.env.tactics import (
    SUPPORT_CARD_IDS,
    UnitView,
    evaluate_spell_cast,
    is_spell,
    perspective,
    threat_pressure,
    threats_on_our_side,
    touches_enemy_king,
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
    "Mini P.E.K.K.A": 5,
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
        model: Optional[dict] = None,
        opponent_elixir_source: str = OPP_ELIXIR_SOURCE,
        elixir_leak_weight: float = ELIXIR_LEAK_WEIGHT
    ):
        self.blue_deck_ids = list(blue_deck or DEFAULT_DECK)
        self.red_deck_ids = list(red_deck or DEFAULT_DECK)
        self.action_repeat = action_repeat
        self.model = model if model is not None else load_model()
        self.rng = np.random.default_rng(seed)
        self.battle: Optional[Battle] = None
        self.prev_tower_hp = [{}, {}]
        self.episode_micro_rewards = [0.0, 0.0]
        # Public-information opponent elixir trackers, one per seat.
        # tracker[team] tracks the elixir of the *opponent* of `team`.
        self.opponent_elixir_source = opponent_elixir_source
        self.elixir_leak_weight = float(elixir_leak_weight)
        self._elixir_trackers = {0: OpponentElixirTracker(), 1: OpponentElixirTracker()}
        self._recent_tiles = {0: [], 1: []}
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        def _to_crsim(cid):
            if isinstance(cid, int):
                return ID_TO_CRSIM_NAME.get(cid, "Knight")
            s = str(cid)
            return DECK_NAME_ALIASES.get(s, s)

        deck0 = [_to_crsim(cid) for cid in self.blue_deck_ids]
        deck1 = [_to_crsim(cid) for cid in self.red_deck_ids]

        self.battle = Battle([deck0, deck1], model=self.model, seed=int(self.rng.integers(0, 2**31)))
        self.prev_tower_hp = [self._get_tower_hps(0), self._get_tower_hps(1)]
        self.last_spell_time = [-999999, -999999]
        self.episode_micro_rewards = [0.0, 0.0]
        self._recent_tiles = {0: [], 1: []}
        for tracker in self._elixir_trackers.values():
            tracker.reset()
        return self.get_obs(1), {}

    def _observe_opponent_play(self, team: int, card_name: str, t_ms: int) -> None:
        """Feed a *public* play event into the opponent's elixir tracker.

        The card's cost is public information (the card is visible on the
        board), so this does not expose the opponent's private counter.
        """
        if self.opponent_elixir_source != "tracker":
            return
        cost = self.battle.card_cost(card_name)
        self._elixir_trackers[1 - team].observe_play(cost, t_ms / 1000.0)

    def _opponent_elixir(self, team: int) -> float:
        """Estimated opponent elixir for `team` (0..10)."""
        if self.opponent_elixir_source != "tracker":
            opp = 1 - team
            return float(self.battle.elixir[opp])
        est = self._elixir_trackers[team].estimate(self.battle.t / 1000.0)
        return float(np.clip(est.mean, 0.0, MAX_ELIXIR))

    def _unit_views(self, team: int) -> List[UnitView]:
        """Canonical-perspective view of the living units on the board.

        For team 1 the arena is Y-mirrored (canonical_y = 31 - engine_y),
        which is the same convention the observation uses.
        """
        b = self.battle
        out: List[UnitView] = []
        for u in b.units:
            if u.hp <= 0 or isinstance(u, CrTower):
                continue
            cid = UNIT_NAME_TO_CARD_ID.get(u.name, 1)
            is_own = (u.team == team)
            cx = u.x
            cy = u.y if team == 0 else (ARENA_HEIGHT - 1) - u.y
            out.append(UnitView(
                card_id=cid,
                x=float(cx),
                y=float(cy),
                hp=float(u.hp),
                is_enemy=not is_own,
                is_air=bool(getattr(u, "flying", False)),
            ))
        return out

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

        # 1. Scalars: [own_elixir/10, opp_elixir_est/10, time_remaining/180, is_overtime]
        #    The opponent's elixir is *estimated* from public plays (see
        #    OpponentElixirTracker) unless the legacy truth source is forced.
        own_elixir = float(np.clip(b.elixir[team] / 10.0, 0.0, 1.0))
        opp_elixir = float(np.clip(self._opponent_elixir(team) / 10.0, 0.0, 1.0))
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
        persp = perspective(team)
        active_enemies = [u for u in b.units if not isinstance(u, CrTower) and u.team != team and u.hp > 0]
        threats_our_half = [u for u in active_enemies if persp.is_own_half(u.y if player_id == 1 else 32.0 - u.y)]
        
        for slot in range(4):
            if slot < len(b.hand[team]):
                cname = b.hand[team][slot]
                cid = CRSIM_NAME_TO_ID.get(cname, 1)
                for tile in range(TOTAL_TILES):
                    cx, cy = tile % ARENA_WIDTH, tile // ARENA_WIDTH   # canonical
                    ex, ey = (cx, cy) if player_id == 1 else (cx, ARENA_HEIGHT - 1 - cy)
                    if not b.legal_action(team, slot, float(ex), float(ey)):
                        continue
                    if cid == 11:  # HogRider: strictly Left and Right bridges (never own lane defense)
                        if not (cy == 14 and (2 <= cx <= 4 or 13 <= cx <= 15)):
                            continue
                    if is_spell(cid):
                        # UNCONDITIONALLY forbid all spells on tile_y <= 11 (ZERO EXCEPTIONS)
                        if cy <= 11:
                            continue
                        if touches_enemy_king(float(cx), float(cy), team):
                            continue
                        # Strictly restricted to 2-tile radius around enemy units (lead_y = max(12, enemy_y - 1))
                        in_spell_radius = False
                        for u in active_enemies:
                            uy = u.y if player_id == 1 else 32.0 - u.y
                            ux = u.x if player_id == 1 else u.x
                            if uy >= 11.5:
                                lead_y = max(12, int(round(uy)) - 1)
                                target_x = int(round(ux))
                                if abs(cx - target_x) <= 2 and abs(cy - lead_y) <= 2:
                                    in_spell_radius = True
                                    break
                        if not in_spell_radius:
                            continue
                    elif len(threats_our_half) > 0 and cid not in (13,):
                        # Troop defenders deploy directly in the SAME LANE as the threat
                        primary_ex = threats_our_half[0].x if player_id == 1 else threats_our_half[0].x
                        lane_min_x = int(max(1, primary_ex - 2))
                        lane_max_x = int(min(16, primary_ex + 2))
                        if not (8 <= cy <= 12 and lane_min_x <= cx <= lane_max_x):
                            continue
                    card_tiles[slot, tile] = True
                card_mask[slot] = bool(card_tiles[slot].any())
                
                # Gate spell slots: require >= 1 enemy units at tile_y >= 12
                if is_spell(cid):
                    enemies_target_zone = [
                        u for u in active_enemies
                        if (u.y if player_id == 1 else 32.0 - u.y) >= 12.0
                    ]
                    if len(enemies_target_zone) == 0:
                        card_mask[slot] = False

        # Win-condition defense guard: If under threat on our half, never waste building-targeters
        # (Giant, HogRider) if an affordable defensive card is present in hand
        if len(threats_our_half) > 0:
            has_affordable_defense = any(
                card_mask[s] and CRSIM_NAME_TO_ID.get(b.hand[team][s], 1) in DEFENSIVE_CARD_IDS
                for s in range(min(4, len(b.hand[team])))
            )
            if has_affordable_defense:
                for s in range(min(4, len(b.hand[team]))):
                    if CRSIM_NAME_TO_ID.get(b.hand[team][s], 1) in WIN_CONDITION_IDS:
                        card_mask[s] = False

        # Low-Elixir Idle Gate: When elixir < 7.5 and 0 enemy units on arena, bank elixir
        own_elixir = float(b.elixir[team])
        if own_elixir < 7.5 and len(active_enemies) == 0:
            card_mask[:4] = False

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

    # -------------------------------------------------------------- shaping
    def _tactical_shaping(self, team: int, cname: str, tx: float, ty: float) -> float:
        """Tactical shaping for one deployment.

        Fixes the three failures observed in live matches:

        * spells thrown at our own tower / at the enemy king tower (which
          activates it) -> penalised, value spells -> rewarded;
        * pushing on the other lane while the opponent chews on our towers
          -> penalised, defending while threatened -> rewarded;
        * deploying on the exact same tile over and over -> penalised.
        """
        b = self.battle
        units = self._unit_views(team)
        # Card *names* need the card table, not the unit table.
        cid = CRSIM_NAME_TO_ID.get(cname, UNIT_NAME_TO_CARD_ID.get(cname, 1))
        shaping = 0.0

        # --- Spell value / misuse -----------------------------------------
        if is_spell(cid):
            decision = evaluate_spell_cast(cid, tx, ty, units, team=team)
            if decision.allowed:
                shaping += SPELL_VALUE_WEIGHT * (1.0 + 0.5 * (decision.hits - 1))
            else:
                shaping -= SPELL_MISUSE_WEIGHT
        else:
            # --- Defense priority -------------------------------------------
            persp = perspective(team)
            threats = threats_on_our_side(units, team)
            if threats:
                pressure = threat_pressure(units, team)
                # Pushing means deploying past our own half.
                offensive_push = not persp.is_own_half(ty)
                if offensive_push:
                    shaping -= DEFENSE_PRIORITY_WEIGHT * pressure
                else:
                    shaping += DEFENSE_PRIORITY_WEIGHT * pressure * 0.5

            # --- Protect our own support units ----------------------------
            own_support = [
                u for u in units
                if (not u.is_enemy) and u.card_id in SUPPORT_CARD_IDS and u.hp > 0
            ]
            enemy_near_support = any(
                u.is_enemy and ((u.x - s.x) ** 2 + (u.y - s.y) ** 2) ** 0.5 <= 5.0
                for u in units for s in own_support
            )
            if own_support and enemy_near_support and ty <= 8.0:
                shaping += SUPPORT_DEFEND_WEIGHT

        # --- Placement variety ---------------------------------------------
        recent = self._recent_tiles[team]
        if recent:
            if recent[-1] == (int(tx), int(ty)):
                shaping -= PLACEMENT_VARIETY_WEIGHT * 3.0
            elif (int(tx), int(ty)) in recent[-4:]:
                shaping -= PLACEMENT_VARIETY_WEIGHT
        return shaping

    def _record_tile(self, team: int, tx: float, ty: float) -> None:
        self._recent_tiles[team].append((int(tx), int(ty)))
        if len(self._recent_tiles[team]) > 8:
            del self._recent_tiles[team][:-8]

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
        p1_tactical = 0.0
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
                    p1_tactical = self._tactical_shaping(0, cname, tx, ty)
                    self._record_tile(0, tx, ty)
                    b.play(0, p1_slot, tx, ty)
                    self._observe_opponent_play(0, cname, b.t)

        # 2. Player 2 Action (Mirror Y back to engine frame)
        p2_slot, p2_tile = int(p2_action[0]), int(p2_action[1])
        p2_slow_roll_bonus = 0.0
        p2_spell_bait_bonus = 0.0
        p2_tactical = 0.0
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
                    p2_tactical = self._tactical_shaping(1, cname, p2_cx, p2_cy)
                    self._record_tile(1, p2_cx, p2_cy)
                    b.play(1, p2_slot, env_tx, env_ty)
                    self._observe_opponent_play(1, cname, b.t)

        # 3. Simulate action_repeat ticks
        elixir_mid = [float(b.elixir[0]), float(b.elixir[1])]
        regen_per_step = 10.0 * TICK_MS * self.action_repeat / max(1.0, b._bar_ms())
        for _ in range(self.action_repeat):
            if b.done:
                break
            b.step([])

        # 3b. Elixir overflow penalty.
        # Elixir that regenerates while the bar is full is thrown away.  With
        # no penalty the agent learns to bank elixir and wait (it played only
        # 14 % of the decisions), because nothing ever charges it for the
        # waste.  The penalty makes cycling cards profitable again.
        p1_leak = 0.0
        p2_leak = 0.0
        if self.elixir_leak_weight > 0.0:
            p1_leak = max(0.0, elixir_mid[0] + regen_per_step - MAX_ELIXIR)
            p2_leak = max(0.0, elixir_mid[1] + regen_per_step - MAX_ELIXIR)

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
        # Tactical shaping computed at action time (spell value/misuse,
        # defense priority, placement variety).
        p1_tactical_bonus = p1_tactical
        p2_tactical_bonus = p2_tactical

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
        p1_micro_raw = (p1_slow_roll_bonus + p1_spell_bait_bonus + p1_kite_bonus
                        + p1_tank_bonus + max(0.0, p1_tactical_bonus))
        max_allowed_p1 = max(0.0, MAX_MICRO_EPISODE - self.episode_micro_rewards[0])
        p1_micro = min(p1_micro_raw, max_allowed_p1)
        self.episode_micro_rewards[0] += p1_micro

        p2_micro_raw = (p2_slow_roll_bonus + p2_spell_bait_bonus + p2_kite_bonus
                        + p2_tank_bonus + max(0.0, p2_tactical_bonus))
        max_allowed_p2 = max(0.0, MAX_MICRO_EPISODE - self.episode_micro_rewards[1])
        p2_micro = min(p2_micro_raw, max_allowed_p2)
        self.episode_micro_rewards[1] += p2_micro

        # Tactical penalties are applied immediately (uncapped, but bounded by
        # their own weight) so the agent learns to stop wasting spells and
        # stop pushing while its own towers are being destroyed.
        r1 += p1_micro + min(0.0, p1_tactical_bonus)
        r2 += p2_micro + min(0.0, p2_tactical_bonus)

        # Elixir overflow penalty (opponent waste is our advantage).
        if self.elixir_leak_weight > 0.0:
            r1 -= self.elixir_leak_weight * p1_leak
            r2 -= self.elixir_leak_weight * p2_leak

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
