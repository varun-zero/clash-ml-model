"""Gym-style environment wrapper around crsim.engine.Battle.

Two-player symmetric env. Every agent always acts in its OWN coordinate
frame (blue frame: your side is low-y). The wrapper mirrors positions for
the red agent so strategies can be learned in a mirrored, symmetric space.

    env = CrSimEnv([deck_blue, deck_red], seed=...)
    obs, info = env.reset()
    obs, rew, done, info = env.step([a_blue, a_red])

Action: tuple (hand_idx 0..3, x, y) with x,y in tiles of the agent's own
frame, or None to do nothing this step.
Reward: tower-damage delta (normalized) per step + terminal crowns bonus.
"""
from __future__ import annotations

import math
import random

from .engine import (ARENA_H, ARENA_W, Battle, Tower, Unit, MAX_ELIXIR)

TOWER_PENALTY = 0.001      # reward per point of tower damage dealt
KING_DAMAGE_BONUS = 0.002  # extra weight for king tower damage
CROWN_REWARD = 1.0
WIN_REWARD = 3.0


def _mirror(x: float, y: float):
    """Blue frame -> red frame (position on the red agent's own side)."""
    return ARENA_W - x, ARENA_H - y


class CrSimEnv:
    def __init__(self, decks, model: dict | None = None, *,
                 seed: int | None = None,
                 invalid_jail: bool = True):
        self.decks = decks
        self.model = model
        self.rng = random.Random(seed)
        self.invalid_jail = invalid_jail
        self.battle: Battle | None = None
        self.prev_tower_hp = [(0, 0, 0), (0, 0, 0)]  # per team: king, pr, pr
        self._tower_entities_cache = {}

    # ---------------------------------------------------------------- api
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.battle = Battle(self.decks, self.model, seed=self.rng.randrange(2**31))
        self.prev_tower_hp = [self._tower_hp(0), self._tower_hp(1)]
        return self.obs(0), self.obs(1)

    def step(self, actions):
        """actions: [ (idx, x, y) | None , (idx, x, y) | None ] in own frames."""
        b = self.battle
        engine_actions = []
        for team, act in enumerate(actions):
            if act is None:
                continue
            idx, x, y = act
            ix, iy = (x, y) if team == 0 else _mirror(x, y)
            if b.legal_action(team, idx, ix, iy):
                engine_actions.append((team, idx, ix, iy))
            elif self.invalid_jail:
                pass  # illegal: skipped silently (use legal mask in your algo)
        b.step(engine_actions)
        rewards = self._rewards()
        obs0, obs1 = self.obs(0), self.obs(1)
        return obs0, obs1, rewards, self.done

    # ------------------------------------------------------------- helpers
    def _towers(self, team: int):
        b = self.battle
        own = [u for u in b.units if isinstance(u, Tower) and u.team == team]
        ks = [u for u in own if u.kind == "KingTower"]
        ps = sorted([u for u in own if u.kind in ("PrincessTower", "TowerPrincess")],
                    key=lambda u: u.x)
        while len(ps) < 2:
            ps.append(None)
        return (ks[0] if ks else None, ps[0], ps[1])

    def _tower_hp(self, team: int):
        t = self._towers(team)
        return tuple(u.hp if u else 0 for u in t)

    def _rewards(self):
        b = self.battle
        rewards = [0.0, 0.0]
        for team in (0, 1):
            hp = self._tower_hp(team)
            prev = self.prev_tower_hp[team]
            lost = sum(p - c for p, c in zip(prev, hp))
            dealt = sum(self.prev_tower_hp[1 - team][i] - hp_own
                        for i, hp_own in enumerate(self._tower_hp(1 - team)))
            r = (dealt - lost) * TOWER_PENALTY
            # king damage counts double (already included in dealt)
            rewards[team] += r
            self.prev_tower_hp[team] = hp
        if b.done and b.result is not None:
            rewards[b.result] += WIN_REWARD + CROWN_REWARD * b.crowns[b.result]
            rewards[1 - b.result] -= WIN_REWARD * 0.5
        return rewards

    def obs(self, team: int) -> dict:
        """Team-specific observation, in the agent's own frame."""
        b = self.battle
        if b is None:
            return {}
        def remap(x, y):
            return (x, y) if team == 0 else _mirror(x, y)

        entities = []
        for u in b.units:
            if u.hp <= 0:
                continue
            x, y = remap(u.x, u.y)
            side = 1 if u.team == team else -1
            entities.append({
                "id": u.id, "side": side, "name": u.name,
                "x": x, "y": y,
                "hp_frac": u.hp / max(1, u.maxhp),
                "flying": 1 if u.flying else 0,
                "building": 1 if u.is_building or isinstance(u, Tower) else 0,
                "tower": 1 if isinstance(u, Tower) else 0,
                "activated": 1 if isinstance(u, Tower) and u.kind == "KingTower"
                              and u.activated else 0,
            })
        own_t, foe_t = self._towers(team), self._towers(1 - team)

        def tv(t):
            return ([u.hp if u else 0 for u in t] if t else [0, 0, 0])

        hand = b.hand_view(team)
        return {
            "elixir": b.elixir[team],
            "hand": [{"card": h["card"], "cost": h["cost"]} for h in hand],
            "crowns_self": b.crowns[team],
            "crowns_foe": b.crowns[1 - team],
            "towers_self": tv(own_t),
            "towers_foe": tv(foe_t),
            "entities": entities,
            "t_ms": b.t,
            "time_ms_left": max(0, 180_000 - b.t) if b.t < 180_000 else 0,
            "play_cd": b.play_cd[team],
        }

    @property
    def done(self):
        return self.battle.done if self.battle else False

    # ------------------------------------------------- action space helper
    # sample points across the whole arena (own-half deploy + enemy-half spells)
    _GRID = [(2.0, 2.0), (2.0, 7.0), (2.0, 12.0), (5.5, 9.0), (5.5, 13.0),
             (9.0, 9.0), (9.0, 13.0), (12.5, 9.0), (12.5, 13.0), (16.0, 12.0),
             (3.5, 14.0), (14.5, 14.0), (9.0, 4.0), (5.5, 4.0), (12.5, 4.0)]

    def legal_actions(self, team: int) -> list[tuple[int, float, float]]:
        """Grid sample of legal plays (for random agents / action masking)."""
        b = self.battle
        if b is None:
            return []
        out = []
        hand = b.hand_view(team)
        for idx in range(len(hand)):
            cost = hand[idx]["cost"]
            if b.elixir[team] < cost or b.play_cd[team] > 0:
                continue
            is_summon = bool(b._card_meta(hand[idx]["card"]).get("SummonCharacter"))
            for gx, gy in self._GRID:
                if not is_summon and gy > 14.0:
                    continue  # (keep spells off the last row; sample is fine)
                x, y = gx, gy
                if team == 1:
                    x, y = _mirror(x, y)
                if b.legal_action(team, idx, x, y):
                    ax, ay = (x, y) if team == 0 else _mirror(x, y)
                    out.append((idx, ax, ay))
        return out

    def is_spell(self, team: int, idx: int) -> bool:
        b = self.battle
        card = b.hand[team][idx]
        return not bool(b._card_meta(card).get("SummonCharacter"))
