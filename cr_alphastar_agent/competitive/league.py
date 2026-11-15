"""PFSP league, scripted opponents and snapshot opponents for competitive v2."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .belief import OpponentBelief
from .cards import CardVocab, default_vocab
from .obs import (
    Action, Observation, GLOBAL_FEATURE_DIM, UNIT_FEATURE_DIM,
    cell_center, cell_index, cell_xy, engine_xy, view_xy,
)
from .policy import ActionSampler
from .model import CompetitiveNet


@dataclass
class Snapshot:
    path: str
    step: int
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5) / max(1, self.games)

    @property
    def score_rate(self) -> float:
        """Draws count as half a win for PFSP weighting."""
        return (self.wins + 0.5 * self.draws + 0.5) / (self.games + 1.0)

    @property
    def name(self) -> str:
        return Path(self.path).stem

    def to_dict(self) -> dict:
        return {
            "path": self.path, "step": self.step, "games": self.games,
            "wins": self.wins, "losses": self.losses, "draws": self.draws,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        return cls(path=d["path"], step=int(d.get("step", 0)), games=int(d.get("games", 0)),
                   wins=int(d.get("wins", 0)), losses=int(d.get("losses", 0)),
                   draws=int(d.get("draws", 0)))


@dataclass
class Matchup:
    opponent_kind: str            # scripted | random | snapshot | current | wait
    opponent_name: str
    learner_deck: list[str]
    opponent_deck: list[str]
    seed: int
    learner_team: int
    group_id: int
    snapshot_path: str | None = None
    profile: str = "tactical"


class ScriptedOpponent:
    """Public-observation heuristic opponent used as a never-forgotten baseline."""

    def __init__(self, profile: str = "tactical", seed: int = 0):
        self.profile = profile
        self.name = f"scripted:{profile}"
        self.rng = random.Random(seed)

    def reset(self, env) -> None:
        pass

    def act(self, obs: Observation, env) -> Action:
        own_elixir = float(obs.global_feats[0] * 10.0)
        playable = np.where(obs.card_mask)[0]
        if len(playable) == 0:
            return Action(gate=0)
        vocab = env.vocab
        card_of = {int(s): vocab.name(int(obs.hand_ids[s])) for s in playable}
        cards = list(card_of.values())

        # Board threats: enemy units on our half (viewer y < 16).
        threats = []
        for i in range(len(obs.unit_mask)):
            if not obs.unit_mask[i]:
                continue
            f = obs.unit_feats[i]
            if f[0] > 0.5:  # own
                continue
            if f[5] > 0.5:  # tower
                continue
            if f[9] >= 0.5 or f[1] <= 0.02:
                continue
            threats.append((float(f[9]), float(f[8]), float(f[1]), float(f[4]),
                            int(obs.unit_ids[i])))
        threats.sort(key=lambda t: t[0])

        def nearest_legal(slot: int, x: float, y: float) -> int:
            mask = obs.cell_masks[slot]
            if not mask.any():
                return -1
            xy = cell_xy()
            d = (xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2
            d[~mask] = 1e9
            return int(np.argmin(d))

        # Spell on a cluster of at least three enemies, otherwise save it.
        if threats and self.profile == "tactical":
            spell_slots = [s for s in playable if vocab.is_spell(card_of[int(s)])]
            if spell_slots:
                tx = np.mean([t[1] for t in threats[:4]])
                ty = np.mean([t[0] for t in threats[:4]])
                if len(threats) >= 3:
                    s = int(spell_slots[0])
                    c = nearest_legal(s, tx, ty)
                    if c >= 0:
                        return Action(gate=1, hand_slot=s, cell=c)

        # Defend with the cheapest affordable unit near the leading threat.
        if threats:
            unit_slots = [s for s in playable if not vocab.is_spell(card_of[int(s)])]
            if unit_slots:
                def cost(slot):
                    return vocab.info(card_of[int(slot)]).cost
                s = int(min(unit_slots, key=cost))
                tx, ty = threats[0][1], threats[0][0]
                # Place behind the threat, on our side of it.
                ty = max(1.0, ty - 2.0)
                c = nearest_legal(s, tx, ty)
                if c >= 0:
                    return Action(gate=1, hand_slot=s, cell=c)

        # Support an existing push or start one at the bridge.
        attack_units = []
        for i in range(len(obs.unit_mask)):
            if not obs.unit_mask[i]:
                continue
            f = obs.unit_feats[i]
            if f[0] > 0.5 and f[5] < 0.5 and f[9] > 0.55 and f[1] > 0.05:
                attack_units.append((float(f[8]), float(f[9])))
        if attack_units and own_elixir >= 7.0:
            tx = float(np.mean([a[0] for a in attack_units]))
            ty = 13.5
            slots = [s for s in playable if not vocab.is_spell(card_of[int(s)])]
            if slots:
                s = int(slots[0])
                c = nearest_legal(s, tx, ty)
                if c >= 0:
                    return Action(gate=1, hand_slot=s, cell=c)

        threshold = 8.2 if self.profile == "tactical" else 9.2
        if own_elixir >= threshold:
            slots = [s for s in playable if not vocab.is_spell(card_of[int(s)])]
            if not slots:
                return Action(gate=0)
            s = int(max(slots, key=lambda z: vocab.info(card_of[int(z)]).cost))
            # Back-of-king or bridge, with a small random lane choice.
            if self.rng.random() < 0.5:
                tx, ty = 9.0, 2.5
            else:
                tx, ty = (3.5 if self.rng.random() < 0.5 else 14.5), 13.5
            c = nearest_legal(s, tx, ty)
            if c >= 0:
                return Action(gate=1, hand_slot=s, cell=c)
        return Action(gate=0)


class RandomOpponent:
    name = "random"

    def __init__(self, seed: int = 0, play_prob: float = 0.45):
        self.rng = random.Random(seed)
        self.play_prob = play_prob

    def reset(self, env) -> None:
        pass

    def act(self, obs: Observation, env) -> Action:
        slots = np.where(obs.card_mask)[0]
        if len(slots) == 0 or self.rng.random() > self.play_prob:
            return Action(gate=0)
        s = int(self.rng.choice(slots.tolist()))
        cells = np.where(obs.cell_masks[s])[0]
        if len(cells) == 0:
            return Action(gate=0)
        return Action(gate=1, hand_slot=s, cell=int(self.rng.choice(cells.tolist())))


class ModelOpponent:
    """A frozen competitive checkpoint playing from its own public view."""

    def __init__(self, model: CompetitiveNet, device: torch.device, name: str,
                 temperature: float = 1.0, greedy: bool = False, seed: int = 0):
        self.model = model.eval()
        self.device = device
        self.name = name
        self.sampler = ActionSampler(self.model, device, seed=seed)
        self.temperature = float(temperature)
        self.greedy = bool(greedy)

    def reset(self, env) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: Observation, env) -> Action:
        actions, _ = self.sampler.sample(obs, greedy=self.greedy, temperature=self.temperature)
        return actions[0]


def load_competitive_model(path: str | Path, device: torch.device,
                           vocab: CardVocab | None = None) -> tuple[CompetitiveNet, dict, CardVocab]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    from .cards import vocab_from_checkpoint
    v = vocab or vocab_from_checkpoint(payload)
    cfg = payload.get("model_config") or {}
    model = CompetitiveNet(
        n_cards=v.n_cards,
        d_model=cfg.get("d_model", 192),
        layers=cfg.get("layers", 4),
        heads=cfg.get("heads", 4),
        dim_feedforward=cfg.get("dim_feedforward", 512),
        dropout=cfg.get("dropout", 0.1),
        d_card=cfg.get("d_card", 64),
        unit_feat_dim=cfg.get("unit_feat_dim", UNIT_FEATURE_DIM),
        global_feat_dim=cfg.get("global_feat_dim", GLOBAL_FEATURE_DIM),
        belief_dim=cfg.get("belief_dim"),
        n_hand=cfg.get("n_hand", 4),
        n_deck=cfg.get("n_deck", 8),
        n_past=cfg.get("n_past", 4),
        n_fourier=cfg.get("n_fourier", 4),
        n_value_bins=cfg.get("n_value_bins", 7),
    )
    state = payload.get("model_state") or payload.get("state_dict") or payload
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, payload, v


class League:
    """Prioritised fictitious self-play with a deck pool and frozen snapshots."""

    def __init__(
        self,
        cfg: dict,
        vocab: CardVocab,
        deck_pool: Sequence[Sequence[str]],
        *,
        checkpoint_dir: str | Path,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        current_model: CompetitiveNet | None = None,
    ):
        self.cfg = cfg
        self.vocab = vocab
        self.deck_pool = [list(d) for d in deck_pool]
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.rng = random.Random(seed)
        self.current_model = current_model
        self.snapshots: list[Snapshot] = []
        self.deck_usage: dict[tuple[str, ...], int] = {}
        self._group_counter = 0
        self.index_path = self.checkpoint_dir / "league_index.json"
        self.load_index()

        learner = cfg.get("learner_decks") or []
        self.learner_decks = [list(d) for d in learner] if learner else list(self.deck_pool)
        self.opponent_decks = list(self.deck_pool)
        self.scripted_ratio = float(cfg.get("scripted_ratio", 0.25))
        self.snapshot_ratio = float(cfg.get("snapshot_ratio", 0.55))
        self.self_ratio = float(cfg.get("self_ratio", 0.20))
        self.random_ratio = float(cfg.get("random_ratio", 0.0))
        self.wait_ratio = float(cfg.get("wait_ratio", 0.0))
        self.mirror_prob = float(cfg.get("mirror_prob", 0.1))
        self.pfsp_power = float(cfg.get("pfsp_power", 2.0))
        self.opponent_temperature = float(cfg.get("opponent_temperature", 1.0))
        self.snapshot_keep = int(cfg.get("snapshot_keep", 20))
        self._snapshot_cache: dict[str, ModelOpponent] = {}

    # --------------------------------------------------------------- snapshots
    def load_index(self) -> None:
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.snapshots = [Snapshot.from_dict(d) for d in data.get("snapshots", [])]
                self._group_counter = int(data.get("group_counter", 0))
            except Exception:
                self.snapshots = []
        # Drop entries whose files were moved or pruned outside the index.
        self.snapshots = [s for s in self.snapshots if Path(s.path).exists()]
        # Also discover checkpoints that arrived outside the index.
        known = {str(Path(s.path).resolve()) for s in self.snapshots}
        for p in sorted(self.checkpoint_dir.glob("*.pt")):
            if str(p.resolve()) not in known:
                step = 0
                try:
                    payload = torch.load(str(p), map_location="cpu", weights_only=False)
                    step = int(payload.get("update", payload.get("step", 0)))
                except Exception:
                    pass
                self.snapshots.append(Snapshot(str(p), step))
        self.snapshots.sort(key=lambda s: s.step)

    def save_index(self) -> None:
        data = {"snapshots": [s.to_dict() for s in self.snapshots],
                "group_counter": self._group_counter}
        self.index_path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def add_snapshot(self, path: str | Path, step: int) -> Snapshot:
        snap = Snapshot(str(Path(path).resolve()), int(step))
        self.snapshots.append(snap)
        self.snapshots.sort(key=lambda s: s.step)
        while len(self.snapshots) > self.snapshot_keep:
            removed = self.snapshots.pop(0)
            try:
                if Path(removed.path).exists() and removed.path != str(Path(path).resolve()):
                    Path(removed.path).unlink()
            except OSError:
                pass
            self._snapshot_cache.pop(removed.path, None)
        self.save_index()
        return snap

    def record_result(self, snapshot_path: str | None, learner_won: bool, draw: bool) -> None:
        if not snapshot_path:
            return
        for s in self.snapshots:
            if str(Path(s.path)) == str(Path(snapshot_path)):
                s.games += 1
                if draw:
                    s.draws += 1
                elif learner_won:
                    s.wins += 1
                else:
                    s.losses += 1
                break

    # ---------------------------------------------------------------- selection
    def _pick_deck(self, decks: Sequence[Sequence[str]]) -> list[str]:
        # Least-used quartile sampling: broad coverage early, then balanced.
        counts = np.asarray([self.deck_usage.get(tuple(sorted(d)), 0) for d in decks], dtype=np.float32)
        order = np.argsort(counts + np.random.rand(len(decks)) * 1e-3)
        take = max(1, len(order) // 4)
        idx = int(self.rng.choice(order[:take].tolist()))
        deck = list(decks[idx])
        self.deck_usage[tuple(sorted(deck))] = self.deck_usage.get(tuple(sorted(deck)), 0) + 1
        return deck

    def _snapshot_weights(self) -> list[float]:
        if not self.snapshots:
            return []
        weights = []
        for s in self.snapshots:
            # pfsp: harder opponents (lower learner score rate) get more weight.
            wr = s.score_rate if s.games >= 3 else 0.35
            weights.append(max(0.02, (1.0 - wr) ** self.pfsp_power))
        total = sum(weights)
        return [w / total for w in weights]

    def sample_matchup(self, step: int = 0, group_size: int = 1) -> Matchup:
        self._group_counter += 1
        group_id = self._group_counter
        roll = self.rng.random()
        r = self.scripted_ratio
        s = r + self.snapshot_ratio
        c = s + self.self_ratio
        d = c + self.random_ratio
        e = d + self.wait_ratio
        # Normalise in case ratios do not sum to one.
        total = max(1e-6, e)
        roll *= total
        if roll < r or not self.snapshots and roll < s:
            profile = "tactical" if self.rng.random() < 0.7 else "basic"
            kind, name = "scripted", f"scripted:{profile}"
            snap = None
        elif roll < s and self.snapshots:
            weights = self._snapshot_weights()
            snap_obj = self.rng.choices(self.snapshots, weights=weights, k=1)[0]
            kind, name, snap = "snapshot", snap_obj.name, snap_obj.path
        elif roll < c and self.current_model is not None:
            kind, name, snap = "current", "current", None
        elif roll < d:
            kind, name, snap = "random", "random", None
        elif roll < e:
            kind, name, snap = "wait", "wait", None
        else:
            profile = "tactical"
            kind, name, snap = "scripted", f"scripted:{profile}", None

        learner_deck = self._pick_deck(self.learner_decks)
        if self.rng.random() < self.mirror_prob:
            opponent_deck = list(learner_deck)
        else:
            opponent_deck = self._pick_deck(self.opponent_decks)
        return Matchup(
            opponent_kind=kind, opponent_name=name,
            learner_deck=learner_deck, opponent_deck=opponent_deck,
            seed=self.rng.randrange(2**31), learner_team=self.rng.randrange(2),
            group_id=group_id, snapshot_path=snap, profile="tactical" if "tactical" in name else "basic",
        )

    # ---------------------------------------------------------------- opponent
    def make_opponent(self, matchup: Matchup):
        if matchup.opponent_kind == "scripted":
            return ScriptedOpponent(matchup.profile, seed=matchup.seed)
        if matchup.opponent_kind == "random":
            return RandomOpponent(seed=matchup.seed)
        if matchup.opponent_kind == "wait":
            from .env import WaitOpponent
            return WaitOpponent()
        if matchup.opponent_kind == "current":
            if self.current_model is None:
                return ScriptedOpponent(matchup.profile, seed=matchup.seed)
            return ModelOpponent(self.current_model, self.device, "current",
                                 temperature=self.opponent_temperature, seed=matchup.seed)
        if matchup.opponent_kind == "snapshot" and matchup.snapshot_path:
            path = str(Path(matchup.snapshot_path).resolve())
            if path not in self._snapshot_cache:
                model, _, _ = load_competitive_model(path, self.device, self.vocab)
                self._snapshot_cache[path] = ModelOpponent(
                    model, self.device, Path(path).stem,
                    temperature=self.opponent_temperature, seed=matchup.seed,
                )
            return self._snapshot_cache[path]
        return ScriptedOpponent(matchup.profile, seed=matchup.seed)
