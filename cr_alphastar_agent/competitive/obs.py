"""Public-information observation contract and legal action geometry (v2).

Everything here is derived from what a player can see:

* own hand, own next card, own deck, own elixir and own play cooldown;
* all units / buildings / towers on the board (except enemy invisible units);
* tower HP, crowns and the match clock;
* the opponent's *observed* plays (card identity, position, time);
* a public-deck belief about the opponent.

The opponent's true elixir, hand and next card are **not** inputs.  They are
returned only in the `aux_*` fields for training-time auxiliary supervision and
are stripped before the observation reaches a policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np

from crsim.engine import ARENA_H, ARENA_W, MAX_ELIXIR, RIVER_HALF, RIVER_Y, Tower, Unit

N_CELLS_X = 36
N_CELLS_Y = 64
N_CELLS = N_CELLS_X * N_CELLS_Y  # 2,304 half-tile cells
CELL_SIZE = 0.5
PATCH_X = 9
PATCH_Y = 16
N_PATCHES = PATCH_X * PATCH_Y  # 144 two-tile patches
MAX_UNITS = 192
PAST_K = 4
HAND_SIZE = 4
DECK_SIZE = 8

UNIT_FEATURES = (
    "is_own", "hp_frac", "maxhp_norm", "flying", "building", "tower",
    "king_tower", "activated", "x", "y", "dist_river", "speed_norm",
    "range_norm", "shield_frac", "invisible", "is_clone",
)
UNIT_FEATURE_DIM = len(UNIT_FEATURES)

GLOBAL_FEATURES = (
    "own_elixir", "time_norm", "overtime", "double_elixir", "triple_elixir",
    "crowns_own", "crowns_opp",
    "own_king_hp", "own_left_hp", "own_right_hp",
    "opp_king_hp", "opp_left_hp", "opp_right_hp",
    "own_king_alive", "own_left_alive", "own_right_alive",
    "opp_king_alive", "opp_left_alive", "opp_right_alive",
    "play_cd", "since_own_play", "since_opp_play", "has_own_play", "has_opp_play",
    "belief_elixir", "belief_ess", "revealed_frac", "belief_entropy",
    "units_own", "units_enemy", "affordable_hand", "hand_avg_cost",
    "next_cost", "deck_avg_cost", "elixir_lead", "own_tower_total", "opp_tower_total",
)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURES)


@lru_cache(maxsize=1)
def cell_xy() -> np.ndarray:
    """[2304, 2] cell centres in viewer frame (x right, y down from own wall)."""
    xs = (np.arange(N_CELLS_X, dtype=np.float32) + 0.5) * CELL_SIZE
    ys = (np.arange(N_CELLS_Y, dtype=np.float32) + 0.5) * CELL_SIZE
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)


@lru_cache(maxsize=1)
def patch_xy() -> np.ndarray:
    """[144, 2] patch centres in viewer frame (2-tile patches)."""
    xs = (np.arange(PATCH_X, dtype=np.float32) + 0.5) * 2.0
    ys = (np.arange(PATCH_Y, dtype=np.float32) + 0.5) * 2.0
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)


@lru_cache(maxsize=1)
def cell_patch() -> np.ndarray:
    """[2304] the patch index containing each fine cell."""
    cx = np.arange(N_CELLS, dtype=np.int64) % N_CELLS_X
    cy = np.arange(N_CELLS, dtype=np.int64) // N_CELLS_X
    # A patch is 2 tiles = 4 half-tile cells on each axis.
    return (cy // 4) * PATCH_X + (cx // 4)


def cell_index(x: float, y: float, *, clip: bool = True) -> int:
    """Viewer-frame tile position -> half-tile cell index."""
    if clip:
        x = min(max(float(x), 0.0), ARENA_W - 1e-6)
        y = min(max(float(y), 0.0), ARENA_H - 1e-6)
    cx = int(x / CELL_SIZE)
    cy = int(y / CELL_SIZE)
    return cy * N_CELLS_X + cx


def cell_center(idx: int) -> tuple[float, float]:
    idx = int(np.clip(idx, 0, N_CELLS - 1))
    cx = idx % N_CELLS_X
    cy = idx // N_CELLS_X
    return (cx + 0.5) * CELL_SIZE, (cy + 0.5) * CELL_SIZE


def view_xy(x: float, y: float, team: int) -> tuple[float, float]:
    return (x, y) if team == 0 else (ARENA_W - x, ARENA_H - y)


def engine_xy(x: float, y: float, team: int) -> tuple[float, float]:
    return (x, y) if team == 0 else (ARENA_W - x, ARENA_H - y)


def _engine_coords_for_viewer(team: int) -> tuple[np.ndarray, np.ndarray]:
    xy = cell_xy()
    if team == 0:
        return xy[:, 0], xy[:, 1]
    return ARENA_W - xy[:, 0], ARENA_H - xy[:, 1]


def legal_cell_mask(battle, team: int, card: str) -> np.ndarray:
    """Vectorised equivalent of ``Battle.legal_action`` over all 2,304 cells."""
    mask = np.ones(N_CELLS, dtype=bool)
    if not card or battle.done:
        return np.zeros(N_CELLS, dtype=bool)
    meta = battle._card_meta(card)
    if not meta:
        return np.zeros(N_CELLS, dtype=bool)
    if battle.elixir[team] + 1e-9 < battle.card_cost(card) or battle.play_cd[team] > 0:
        return np.zeros(N_CELLS, dtype=bool)
    if card == "Mirror" and not battle.last_played[team]:
        return np.zeros(N_CELLS, dtype=bool)
    ex, ey = _engine_coords_for_viewer(team)
    # Own-half restriction exactly as engine: summon cards without the
    # CanDeployOnEnemySide flag, irrespective of whether the summon is a
    # building or a troop.  The no-deploy exclusion around buildings only
    # applies inside this same branch (spells may be placed anywhere).
    if meta.get("SummonCharacter") and not meta.get("CanDeployOnEnemySide"):
        vy = cell_xy()[:, 1]
        mask &= vy <= (RIVER_Y - RIVER_HALF)
        bdef = battle.model.get("buildings", {}) or {}
        for u in battle.units:
            if u.hp <= 0 or not (u.is_building or isinstance(u, Tower)):
                continue
            d = bdef.get(u.name, {}) or {}
            w = float(d.get("NoDeploySizeW") or 0) / 2.0
            h = float(d.get("NoDeploySizeH") or 0) / 2.0
            if w and h:
                # Replicates Battle._own_half_ok's w/2 comparison.
                mask &= ~((np.abs(ex - u.x) <= w / 2.0) & (np.abs(ey - u.y) <= h / 2.0))
    # Engine bounds are always satisfied by cell centres, but keep the check.
    mask &= (ex >= 0.0) & (ex <= ARENA_W) & (ey >= 0.0) & (ey <= ARENA_H)
    return mask


def view_card_mask(battle, team: int) -> np.ndarray:
    mask = np.zeros(HAND_SIZE, dtype=bool)
    for i, card in enumerate(battle.hand[team][:HAND_SIZE]):
        mask[i] = bool(legal_cell_mask(battle, team, card).any())
    return mask


def view_cell_masks(battle, team: int) -> np.ndarray:
    out = np.zeros((HAND_SIZE, N_CELLS), dtype=bool)
    for i, card in enumerate(battle.hand[team][:HAND_SIZE]):
        out[i] = legal_cell_mask(battle, team, card)
    return out


def _tower_slots(battle, team: int):
    towers = [u for u in battle.units
              if isinstance(u, Tower) and u.team == team and u.hp > 0]
    king = next((u for u in towers if u.kind == "KingTower"), None)
    princess = sorted((u for u in towers if u.kind != "KingTower"),
                      key=lambda u: view_xy(u.x, u.y, team)[0])
    left = princess[0] if len(princess) > 0 else None
    right = princess[1] if len(princess) > 1 else None
    return king, left, right


def tower_potential(battle, team: int, initial_total_hp: float,
                    *, tower_weight: float = 0.5, crown_weight: float = 0.15) -> float:
    own = _tower_slots(battle, team)
    opp = _tower_slots(battle, 1 - team)
    own_hp = sum((t.hp if t else 0.0) for t in own)
    opp_hp = sum((t.hp if t else 0.0) for t in opp)
    denom = max(1.0, float(initial_total_hp))
    tower_term = tower_weight * (own_hp - opp_hp) / denom
    crown_term = crown_weight * ((battle.crowns[team] - battle.crowns[1 - team]) / 3.0)
    return float(tower_term + crown_term)


@dataclass
class Observation:
    unit_ids: np.ndarray
    unit_feats: np.ndarray
    unit_mask: np.ndarray
    global_feats: np.ndarray
    hand_ids: np.ndarray
    hand_mask: np.ndarray
    deck_ids: np.ndarray
    deck_mask: np.ndarray
    next_id: int
    past_own: np.ndarray
    past_opp: np.ndarray
    belief: np.ndarray
    card_mask: np.ndarray
    cell_masks: np.ndarray
    wait_mask: np.ndarray
    # Training-only privileged labels (never part of the policy input).
    aux_opp_elixir: float = 0.0
    aux_opp_hand: np.ndarray | None = None
    aux_opp_next: int = 0
    phi: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    def policy_arrays(self) -> dict[str, np.ndarray]:
        """The subset that may safely reach the network."""
        return {
            "unit_ids": self.unit_ids,
            "unit_feats": self.unit_feats,
            "unit_mask": self.unit_mask,
            "global_feats": self.global_feats,
            "hand_ids": self.hand_ids,
            "hand_mask": self.hand_mask,
            "deck_ids": self.deck_ids,
            "deck_mask": self.deck_mask,
            "next_id": np.asarray(self.next_id),
            "past_own": self.past_own,
            "past_opp": self.past_opp,
            "belief": self.belief,
        }


def _past_array(items: Sequence[tuple[str, float, float, float]], t_ms: float,
                vocab, team: int) -> np.ndarray:
    out = np.zeros((PAST_K, 5), dtype=np.float32)
    for j, (card, ex, ey, when) in enumerate(list(items)[-PAST_K:][::-1]):
        vx, vy = view_xy(ex, ey, team)
        out[j, 0] = float(vocab.id(card))
        out[j, 1] = vx / ARENA_W
        out[j, 2] = vy / ARENA_H
        out[j, 3] = np.clip((t_ms - when) / 30_000.0, 0.0, 1.0)
        out[j, 4] = 1.0
    return out


def build_observation(
    battle,
    team: int,
    belief,
    *,
    vocab,
    past_own: Sequence[tuple[str, float, float, float]] = (),
    past_opp: Sequence[tuple[str, float, float, float]] = (),
    initial_total_hp: float = 1.0,
    max_units: int = MAX_UNITS,
    tower_weight: float = 0.5,
    crown_weight: float = 0.15,
) -> Observation:
    """Build the public observation for ``team`` from the current battle state."""
    belief.update_time(float(battle.t))
    unit_ids = np.zeros(max_units, dtype=np.int32)
    unit_feats = np.zeros((max_units, UNIT_FEATURE_DIM), dtype=np.float32)
    unit_mask = np.zeros(max_units, dtype=bool)

    units = []
    for u in battle.units:
        if u.hp <= 0:
            continue
        if u.team != team and getattr(u, "invisible", False):
            continue  # invisible enemy units are not public information
        units.append(u)
    units.sort(key=lambda u: (0 if isinstance(u, Tower) else 1, int(u.id)))
    units = units[:max_units]

    n_own_units = 0
    n_enemy_units = 0
    for i, u in enumerate(units):
        is_own = u.team == team
        if is_own:
            n_own_units += 1
        else:
            n_enemy_units += 1
        vx, vy = view_xy(u.x, u.y, team)
        is_tower = isinstance(u, Tower)
        unit_ids[i] = vocab.id(u.name)
        unit_feats[i] = (
            float(is_own),
            float(np.clip(u.hp / max(1.0, float(u.maxhp)), 0.0, 1.0)),
            float(np.clip(u.maxhp / 8000.0, 0.0, 1.0)),
            float(bool(u.flying)),
            float(bool(u.is_building or is_tower)),
            float(is_tower),
            float(is_tower and u.kind == "KingTower"),
            float(bool(is_tower and u.kind == "KingTower" and getattr(u, "activated", False))),
            float(vx / ARENA_W),
            float(vy / ARENA_H),
            float(abs(vy - RIVER_Y) / (ARENA_H / 2.0)),
            float(np.clip(u.speed_t / 8.0, 0.0, 1.0)),
            float(np.clip(u.range_t / 8.0, 0.0, 1.0)),
            float(np.clip(getattr(u, "shield_hp", 0) / max(1.0, float(u.maxhp)), 0.0, 1.0)),
            float(bool(getattr(u, "invisible", False))),
            float(bool(getattr(u, "is_clone", False))),
        )
        unit_mask[i] = True

    own_t = _tower_slots(battle, team)
    opp_t = _tower_slots(battle, 1 - team)

    def hp(t):
        return float(np.clip(t.hp / max(1.0, float(t.maxhp)), 0.0, 1.0)) if t else 0.0

    own_hp = sum((t.hp if t else 0) for t in own_t)
    opp_hp = sum((t.hp if t else 0) for t in opp_t)
    own_max = sum((t.maxhp if t else 0) for t in own_t)
    opp_max = sum((t.maxhp if t else 0) for t in opp_t)

    hand = list(battle.hand[team])[:HAND_SIZE]
    hand_ids = np.zeros(HAND_SIZE, dtype=np.int32)
    hand_mask = np.zeros(HAND_SIZE, dtype=bool)
    for i, c in enumerate(hand):
        hand_ids[i] = vocab.id(c)
        hand_mask[i] = True
    deck = list(battle.decks[team])[:DECK_SIZE]
    deck_ids = np.zeros(DECK_SIZE, dtype=np.int32)
    deck_mask = np.zeros(DECK_SIZE, dtype=bool)
    for i, c in enumerate(deck):
        deck_ids[i] = vocab.id(c)
        deck_mask[i] = True
    next_card = battle.cycle[team][0] if battle.cycle[team] else ""
    next_id = int(vocab.id(next_card))

    card_mask = view_card_mask(battle, team)
    cell_masks = view_cell_masks(battle, team)
    wait_mask = deck_mask & ~np.isin(deck_ids, hand_ids[hand_mask])

    now = float(battle.t)
    since_own = (now - past_own[-1][3]) / 30_000.0 if past_own else 1.0
    since_opp = (now - past_opp[-1][3]) / 30_000.0 if past_opp else 1.0
    bsum = belief.summary()
    elixir = float(battle.elixir[team])
    hand_costs = [battle.card_cost(c) for c in hand]
    deck_costs = [battle.card_cost(c) for c in deck]

    global_feats = np.array([
        np.clip(elixir / MAX_ELIXIR, 0.0, 1.0),
        np.clip(now / 180_000.0, 0.0, 2.0),
        float(now >= 180_000.0),
        float(now >= 120_000.0),
        float(now >= 240_000.0),
        battle.crowns[team] / 3.0,
        battle.crowns[1 - team] / 3.0,
        hp(own_t[0]), hp(own_t[1]), hp(own_t[2]),
        hp(opp_t[0]), hp(opp_t[1]), hp(opp_t[2]),
        float(own_t[0] is not None), float(own_t[1] is not None), float(own_t[2] is not None),
        float(opp_t[0] is not None), float(opp_t[1] is not None), float(opp_t[2] is not None),
        np.clip(battle.play_cd[team] / 1000.0, 0.0, 1.0),
        np.clip(since_own, 0.0, 1.0),
        np.clip(since_opp, 0.0, 1.0),
        float(bool(past_own)),
        float(bool(past_opp)),
        np.clip(bsum["elixir"] / MAX_ELIXIR, 0.0, 1.0),
        float(bsum["ess_frac"]),
        min(1.0, len(bsum["revealed"]) / 8.0),
        float(bsum["hand_entropy"] / math.log(max(2, vocab.n_cards))),
        min(n_own_units / 20.0, 1.0),
        min(n_enemy_units / 20.0, 1.0),
        float(card_mask.sum()) / HAND_SIZE,
        (sum(hand_costs) / max(1, len(hand_costs))) / 10.0,
        battle.card_cost(next_card) / 10.0 if next_card else 0.0,
        (sum(deck_costs) / max(1, len(deck_costs))) / 10.0,
        float(np.clip((elixir - bsum["elixir"]) / 10.0, -1.0, 1.0)),
        own_hp / max(1.0, own_max),
        opp_hp / max(1.0, opp_max),
    ], dtype=np.float32)

    # Privileged aux labels, returned separately from the policy arrays.
    opp_hand = np.zeros(vocab.n_cards, dtype=np.float32)
    for c in battle.hand[1 - team][:HAND_SIZE]:
        opp_hand[vocab.id(c)] = 1.0
    opp_next = battle.cycle[1 - team][0] if battle.cycle[1 - team] else ""
    phi = tower_potential(battle, team, initial_total_hp,
                          tower_weight=tower_weight, crown_weight=crown_weight)

    return Observation(
        unit_ids=unit_ids,
        unit_feats=unit_feats,
        unit_mask=unit_mask,
        global_feats=global_feats,
        hand_ids=hand_ids,
        hand_mask=hand_mask,
        deck_ids=deck_ids,
        deck_mask=deck_mask,
        next_id=next_id,
        past_own=_past_array(past_own, now, vocab, team),
        past_opp=_past_array(past_opp, now, vocab, team),
        belief=belief.belief_vector(),
        card_mask=card_mask,
        cell_masks=cell_masks,
        wait_mask=wait_mask,
        aux_opp_elixir=float(np.clip(battle.elixir[1 - team] / MAX_ELIXIR, 0.0, 1.0)),
        aux_opp_hand=opp_hand,
        aux_opp_next=int(vocab.id(opp_next)),
        phi=float(phi),
        info={"team": team, "t_ms": now},
    )


def collate_observations(obs_list: Sequence[Observation]) -> dict[str, np.ndarray]:
    """Stack variable-count observations into fixed-shape arrays."""
    n = len(obs_list)
    belief_dim = obs_list[0].belief.shape[0]
    return {
        "unit_ids": np.stack([o.unit_ids for o in obs_list]),
        "unit_feats": np.stack([o.unit_feats for o in obs_list]),
        "unit_mask": np.stack([o.unit_mask for o in obs_list]),
        "global_feats": np.stack([o.global_feats for o in obs_list]),
        "hand_ids": np.stack([o.hand_ids for o in obs_list]),
        "hand_mask": np.stack([o.hand_mask for o in obs_list]),
        "deck_ids": np.stack([o.deck_ids for o in obs_list]),
        "deck_mask": np.stack([o.deck_mask for o in obs_list]),
        "next_id": np.asarray([o.next_id for o in obs_list], dtype=np.int64),
        "past_own": np.stack([o.past_own for o in obs_list]),
        "past_opp": np.stack([o.past_opp for o in obs_list]),
        "belief": np.stack([o.belief for o in obs_list]),
        "card_mask": np.stack([o.card_mask for o in obs_list]),
        "cell_masks": np.stack([o.cell_masks for o in obs_list]),
        "wait_mask": np.stack([o.wait_mask for o in obs_list]),
        "aux_opp_elixir": np.asarray([o.aux_opp_elixir for o in obs_list], dtype=np.float32),
        "aux_opp_hand": np.stack([o.aux_opp_hand for o in obs_list]),
        "aux_opp_next": np.asarray([o.aux_opp_next for o in obs_list], dtype=np.int64),
        "phi": np.asarray([o.phi for o in obs_list], dtype=np.float32),
    }


@dataclass
class Action:
    """Factored competitive action (gate -> card -> cell; wait pointer)."""
    gate: int = 0
    hand_slot: int = 0
    cell: int = 0
    wait_slot: int = -1

    def as_array(self) -> np.ndarray:
        return np.asarray([self.gate, self.hand_slot, self.cell, self.wait_slot], dtype=np.int64)


def decode_action(action: Action, battle, team: int) -> tuple[int, int, float, float] | None:
    """Return an engine action tuple, or None for wait / illegal."""
    if action is None or int(action.gate) == 0:
        return None
    if battle.done:
        return None
    idx = int(action.hand_slot)
    if not (0 <= idx < len(battle.hand[team])):
        return None
    vx, vy = cell_center(int(action.cell))
    ex, ey = engine_xy(vx, vy, team)
    if not battle.legal_action(team, idx, ex, ey):
        return None
    return (team, idx, float(ex), float(ey))


def action_from_engine_play(battle, team: int, hand_idx: int, x: float, y: float) -> Action:
    vx, vy = view_xy(float(x), float(y), team)
    return Action(gate=1, hand_slot=int(hand_idx), cell=cell_index(vx, vy), wait_slot=-1)
