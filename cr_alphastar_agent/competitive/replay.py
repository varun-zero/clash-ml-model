"""Replay ingestion and behaviour-cloning data for the competitive v2 agent.

Supported inputs
----------------
* this repository's JSON replays (``blue_deck`` / ``red_deck`` / ``events``);
* the public HuggingFace ``VanguardX101/IL_Replay`` parquet schema
  (``replay_tag`` + ``payload_json`` with ``battle.*.players[0].deck`` and
  ``events[*].kind == "play_card"``).

Because a replay stores only card plays, not full per-tick frames, the state at
each play is reconstructed by re-simulating the recorded plays in ``crsim``.
That is exact for hands when the recorded card is in the current hand, and a
documented approximation when the replay's hand order differs.  The generated
rows are still valuable for BC; they are not a substitute for a full-frame
replay corpus.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from crsim.engine import ARENA_H, ARENA_W, Battle, Tower

from .belief import OpponentBelief
from .cards import CardVocab, default_vocab
from .obs import (
    Action, Observation, build_observation, cell_center, cell_index,
    collate_observations, engine_xy, legal_cell_mask, view_xy,
)
from .policy import obs_to_tensors


@dataclass
class ReplayEvent:
    t_ms: float
    team: int
    card: str
    x: float
    y: float
    hand_idx: int = -1


@dataclass
class ReplayRecord:
    replay_id: str
    decks: list[list[str]]
    events: list[ReplayEvent]
    winner: int | None = None
    duration_ms: float = 0.0
    source: str = ""


def _canonical_deck(deck: Sequence[str], vocab: CardVocab) -> list[str] | None:
    out = []
    for c in deck:
        name = vocab.name(vocab.id(c))
        if name in ("<unknown>", "<pad>"):
            return None
        out.append(name)
    if len(out) != 8 or len(set(out)) != 8:
        return None
    if any(not vocab.card_meta(c) for c in out):
        return None
    return out


def parse_local_replay(path: str | Path, vocab: CardVocab | None = None) -> ReplayRecord | None:
    vocab = vocab or default_vocab()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    decks = [_canonical_deck(raw.get("blue_deck") or [], vocab),
             _canonical_deck(raw.get("red_deck") or [], vocab)]
    if decks[0] is None or decks[1] is None:
        return None
    events = []
    for e in raw.get("events") or []:
        card = vocab.name(vocab.id(e.get("card", "")))
        if card in ("<unknown>", "<pad>"):
            continue
        x = float(e.get("x", 0.0))
        y = float(e.get("y", 0.0))
        # This repository's deploy logger writes world tiles (18 x 32).
        if x > ARENA_W + 0.5 or y > ARENA_H + 0.5:
            x /= 1000.0
            y /= 1000.0
        events.append(ReplayEvent(
            t_ms=float(e.get("t_ms", 0.0)), team=int(e.get("team", 0)),
            card=card, x=x, y=y, hand_idx=int(e.get("hand_idx", -1)),
        ))
    winner = None
    w = str(raw.get("winner", "")).lower()
    if w in ("blue", "team0", "0"):
        winner = 0
    elif w in ("red", "team1", "1"):
        winner = 1
    events.sort(key=lambda e: e.t_ms)
    return ReplayRecord(
        replay_id=str(raw.get("id") or Path(path).stem),
        decks=[decks[0], decks[1]], events=events, winner=winner,
        duration_ms=float(raw.get("duration_ms", events[-1].t_ms if events else 0.0)),
        source=str(path),
    )


def parse_hf_payload(payload: dict | str, tag: str = "", vocab: CardVocab | None = None) -> ReplayRecord | None:
    vocab = vocab or default_vocab()
    p = json.loads(payload) if isinstance(payload, str) else payload
    battle = p.get("battle") or {}
    sides = {}
    for side in ("team", "opponent"):
        players = (battle.get(side) or {}).get("players") or []
        if not players:
            return None
        sides[side] = players[0]
    decks = []
    for side in ("team", "opponent"):
        cards = [c.get("card_key", c.get("card", "")) for c in sides[side].get("deck") or []]
        deck = _canonical_deck(cards, vocab)
        if deck is None:
            return None
        decks.append(deck)
    events = []
    for e in p.get("events") or []:
        if e.get("kind") != "play_card":
            continue
        sf = e.get("source_fields") or {}
        if sf.get("data_x") is None or sf.get("data_y") is None:
            continue
        side = e.get("side")
        team = 0 if side in ("team", 0) else 1
        card = vocab.name(vocab.id(e.get("card_key") or e.get("card") or ""))
        if card in ("<unknown>", "<pad>"):
            continue
        # HF source coordinates are engine units (1000 per tile) and data_t is
        # at 20 Hz; convert both to the local 100 ms engine convention.
        x = float(sf["data_x"]) / 1000.0
        y = float(sf["data_y"]) / 1000.0
        events.append(ReplayEvent(
            t_ms=float(sf.get("data_t", 0)) * 50.0, team=team,
            card=card, x=x, y=y, hand_idx=int(sf.get("data_i", -1) or -1),
        ))
    if not events:
        return None
    crowns = []
    for side in ("team", "opponent"):
        crowns.append(int((battle.get(side) or {}).get("crowns", 0) or 0))
    winner = None if crowns[0] == crowns[1] else (0 if crowns[0] > crowns[1] else 1)
    events.sort(key=lambda e: e.t_ms)
    return ReplayRecord(
        replay_id=tag or "hf", decks=decks, events=events, winner=winner,
        duration_ms=events[-1].t_ms, source="hf",
    )


def iter_hf_parquet(paths: Sequence[str | Path], limit: int | None = None) -> Iterator[tuple[str, dict | str]]:
    """Yield (replay_tag, payload) from parquet parts.

    pyarrow is optional; install it with ``pip install pyarrow``.
    """
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("reading HF parquet requires pyarrow (pip install pyarrow)") from exc
    n = 0
    for path in paths:
        table = pq.read_table(str(path), columns=["replay_tag", "payload_json"])
        tags = table.column("replay_tag").to_pylist()
        payloads = table.column("payload_json").to_pylist()
        for tag, payload in zip(tags, payloads):
            yield str(tag), payload
            n += 1
            if limit is not None and n >= limit:
                return


def align_hand(battle: Battle, team: int, card: str, hand_idx: int = -1) -> int | None:
    """Make ``card`` available in the team's hand, returns its slot."""
    hand = battle.hand[team]
    if card in hand:
        return hand.index(card)
    if card in battle.cycle[team]:
        idx = hand_idx if 0 <= hand_idx < len(hand) else 0
        old = hand[idx]
        pos = battle.cycle[team].index(card)
        battle.cycle[team][pos] = old
        hand[idx] = card
        return idx
    return None


def _nearest_legal_cell(battle: Battle, team: int, hand_idx: int, vx: float, vy: float) -> int:
    mask = legal_cell_mask(battle, team, battle.hand[team][hand_idx])
    if not mask.any():
        return -1
    from .obs import cell_xy
    xy = cell_xy()
    d = (xy[:, 0] - vx) ** 2 + (xy[:, 1] - vy) ** 2
    d[~mask] = 1e9
    return int(np.argmin(d))


def _advance(battle: Battle, t_ms: float) -> None:
    while battle.t < int(round(t_ms)) and not battle.done:
        battle.step([])


@dataclass
class BCExample:
    obs: Observation
    gate: int
    hand_slot: int
    cell: int
    wait_slot: int
    source: str = ""
    replay_id: str = ""
    value_target: float = 0.0
    hand_slot_valid: bool = True


def build_examples_for_team(
    record: ReplayRecord,
    team: int,
    *,
    vocab: CardVocab | None = None,
    wait_prob: float = 0.35,
    max_wait_per_event: int = 2,
    seed: int = 0,
    max_units: int = 192,
) -> list[BCExample]:
    vocab = vocab or default_vocab()
    rng = random.Random(seed)
    battle = Battle([record.decks[0], record.decks[1]], seed=seed)
    beliefs = {
        t: OpponentBelief(vocab, deck_prior=[record.decks[0], record.decks[1]],
                          n_particles=64, seed=seed + 13 * t)
        for t in (0, 1)
    }
    history: dict[int, list[tuple[str, float, float, float]]] = {0: [], 1: []}
    initial_total_hp = max(1.0, sum(float(u.maxhp) for u in battle.units if isinstance(u, Tower)))
    examples: list[BCExample] = []
    last_t = 0.0
    value_target = 0.0
    if record.winner is not None:
        value_target = 1.0 if record.winner == team else -1.0

    def obs_now() -> Observation:
        return build_observation(
            battle, team, beliefs[team], vocab=vocab,
            past_own=history[team], past_opp=history[1 - team],
            initial_total_hp=initial_total_hp, max_units=max_units,
        )

    # Iterate *all* events so the opponent's plays also advance the engine and
    # update the learner's public belief.  Examples are emitted only for the
    # imitated seat.
    for ei, ev in enumerate(record.events):
        if ev.t_ms < last_t:
            continue
        if ev.team == team:
            # Wait examples on a 500 ms grid between the previous event and
            # this one (the next real action for this seat).
            n_wait = 0
            t = last_t + 500.0
            while t <= ev.t_ms - 250.0 and n_wait < max_wait_per_event and not battle.done:
                _advance(battle, t)
                if battle.done:
                    break
                obs = obs_now()
                if obs.card_mask.any() and rng.random() < wait_prob:
                    nxt = next((x for x in record.events[ei:] if x.team == team and x.t_ms > battle.t), None)
                    wait_slot = -1
                    if nxt is not None:
                        cid = vocab.id(nxt.card)
                        hit = np.where(obs.deck_ids == cid)[0]
                        if len(hit) and not (obs.hand_mask & (obs.hand_ids == cid)).any():
                            wait_slot = int(hit[0])
                    examples.append(BCExample(obs, gate=0, hand_slot=0, cell=0,
                                              wait_slot=wait_slot, source=record.source,
                                              replay_id=record.replay_id, value_target=value_target))
                    n_wait += 1
                t += 500.0
        _advance(battle, ev.t_ms)
        if battle.done:
            break
        slot = align_hand(battle, ev.team, ev.card, ev.hand_idx)
        if slot is None:
            last_t = float(ev.t_ms)
            continue
        vx, vy = view_xy(ev.x, ev.y, ev.team)
        cell = _nearest_legal_cell(battle, ev.team, slot, vx, vy)
        if cell < 0:
            last_t = float(ev.t_ms)
            continue
        cvx, cvy = cell_center(cell)
        ex, ey = engine_xy(cvx, cvy, ev.team)
        if not battle.legal_action(ev.team, slot, ex, ey):
            last_t = float(ev.t_ms)
            continue
        if ev.team == team:
            examples.append(BCExample(obs_now(), gate=1, hand_slot=slot, cell=cell,
                                      wait_slot=-1, source=record.source,
                                      replay_id=record.replay_id, value_target=value_target))
        if battle.play(ev.team, slot, ex, ey):
            beliefs[1 - ev.team].observe_play(ev.card, float(battle.t))
            history[ev.team].append((ev.card, float(ex), float(ey), float(battle.t)))
        last_t = float(ev.t_ms)
    return examples


def build_dataset_from_records(
    records: Iterable[ReplayRecord],
    *,
    vocab: CardVocab | None = None,
    wait_prob: float = 0.35,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    vocab = vocab or default_vocab()
    examples: list[BCExample] = []
    for i, rec in enumerate(records):
        for team in (0, 1):
            examples.extend(build_examples_for_team(rec, team, vocab=vocab,
                                                    wait_prob=wait_prob, seed=seed + i * 7 + team))
    return examples_to_arrays(examples, vocab=vocab)


def examples_to_arrays(examples: Sequence[BCExample], *, vocab: CardVocab | None = None) -> dict[str, np.ndarray]:
    vocab = vocab or default_vocab()
    if not examples:
        return {}
    coll = collate_observations([e.obs for e in examples])
    n = len(examples)
    out = {
        **coll,
        "gate": np.asarray([e.gate for e in examples], dtype=np.int64),
        "hand_slot": np.asarray([e.hand_slot for e in examples], dtype=np.int64),
        "cell": np.asarray([e.cell for e in examples], dtype=np.int64),
        "wait_slot": np.asarray([e.wait_slot for e in examples], dtype=np.int64),
        "value_target": np.asarray([e.value_target for e in examples], dtype=np.float32),
        "cell_mask": np.stack([
            e.obs.cell_masks[e.hand_slot] if 0 <= e.hand_slot < e.obs.cell_masks.shape[0]
            else np.zeros(e.obs.cell_masks.shape[-1], dtype=bool)
            for e in examples
        ]),
        "source": np.asarray([e.source for e in examples], dtype="U160"),
        "replay_id": np.asarray([e.replay_id for e in examples], dtype="U160"),
    }
    # The chosen-card mask is enough for BC; the full [N, 4, 2304] tensor is
    # ~9 KB per example and would dominate large replay corpora.
    out.pop("cell_masks", None)
    return out


def save_dataset(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=False)
    return {k: data[k] for k in data.files}


class CompetitiveDataset:
    """Minimal in-memory dataset wrapper for BC training."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        self.n = int(arrays["gate"].shape[0]) if "gate" in arrays else 0

    def __len__(self) -> int:
        return self.n

    def batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        idx = np.asarray(idx)
        return {k: v[idx] for k, v in self.arrays.items()}


def load_local_replays(directory: str | Path, vocab: CardVocab | None = None,
                       limit: int | None = None) -> list[ReplayRecord]:
    vocab = vocab or default_vocab()
    out = []
    for p in sorted(Path(directory).glob("*.json")):
        rec = parse_local_replay(p, vocab)
        if rec is not None:
            out.append(rec)
        if limit is not None and len(out) >= limit:
            break
    return out


# torch is imported lazily in type hints above; keep the module importable
# without torch in lightweight replay tools.
try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None
