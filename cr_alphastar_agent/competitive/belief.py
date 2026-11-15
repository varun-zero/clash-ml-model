"""Public-information opponent elixir and hand belief.

No method in this module reads the opponent's true elixir, hand or cycle.  The
only hidden-state access is through ``observe_play`` callers, which pass a card
identity and time that a player would see in the arena.  ``aux_targets`` is
training-only supervision and is never part of the observation returned to the
policy.

The estimator has two layers:

1. deterministic elixir accounting with the engine's public regeneration
   schedule and re-basing whenever an observed play proves the estimate low;
2. a particle filter over candidate 8-card opponent decks.  Each particle has a
   full deck order, hand and cycle, so updates from observed plays are exact
   for that hypothesis.  Candidate decks come from a public meta-deck prior
   (the configured league deck pool by default), and the filter degrades to a
   revealed-card fallback if the opponent plays a deck outside the prior.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from crsim.engine import MAX_ELIXIR, NORMAL_MS

START_ELIXIR = 5.0
# (from_ms, elixir per ms) -- engine bars 28 s / 14 s / 9.3 s.
REGEN_SCHEDULE: tuple[tuple[float, float], ...] = (
    (0.0, 10.0 / 28_000.0),
    (120_000.0, 10.0 / 14_000.0),
    (240_000.0, 10.0 / 9_300.0),
    (300_000.0, 0.0),
)


def regen_between(t0: float, t1: float, schedule=REGEN_SCHEDULE) -> float:
    if t1 <= t0:
        return 0.0
    total = 0.0
    for i, (lo, rate) in enumerate(schedule):
        hi = schedule[i + 1][0] if i + 1 < len(schedule) else float("inf")
        a, b = max(t0, lo), min(t1, hi)
        if b > a:
            total += rate * (b - a)
    return total


@dataclass
class ElixirCounter:
    """Deterministic estimate of the opponent's elixir from public events."""

    start: float = START_ELIXIR
    t_ms: float = 0.0
    est: float = START_ELIXIR
    plays: int = 0
    rebases: int = 0
    missed_cost: float = 0.0

    def reset(self) -> None:
        self.t_ms = 0.0
        self.est = float(self.start)
        self.plays = 0
        self.rebases = 0
        self.missed_cost = 0.0

    def at(self, t_ms: float) -> float:
        self.advance(t_ms)
        return self.est

    def advance(self, t_ms: float) -> None:
        if t_ms > self.t_ms:
            # Clamped regeneration is lost, so one aggregate clamp is exactly
            # equivalent to clamping per tick for a non-negative rate.
            self.est = min(MAX_ELIXIR, self.est + regen_between(self.t_ms, t_ms))
            self.t_ms = float(t_ms)

    def observe_play(self, t_ms: float, cost: float | None) -> None:
        self.advance(float(t_ms))
        self.plays += 1
        if cost is None or cost <= 0:
            return
        if self.est < cost:
            # The play was legal, so the estimate was necessarily low.  Re-base
            # to the least it could have been and count the correction.
            self.rebases += 1
            self.missed_cost += float(cost) - self.est
            self.est = float(cost)
        self.est = max(0.0, self.est - float(cost))


@dataclass
class _Particle:
    deck: tuple[str, ...]
    hand: list[str]
    cycle: list[str]
    weight: float = 1.0

    @classmethod
    def from_deck(cls, deck: Sequence[str], rng: random.Random) -> "_Particle":
        order = list(deck)
        rng.shuffle(order)
        return cls(tuple(order), order[:4], order[4:], 1.0)

    def play(self, card: str) -> bool:
        try:
            idx = self.hand.index(card)
        except ValueError:
            return False
        played = self.hand[idx]
        if not self.cycle:  # malformed particle; recover defensively
            self.hand[idx] = played
            return True
        self.hand[idx] = self.cycle[0]
        self.cycle.append(played)
        self.cycle.pop(0)
        return True


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 1e-12]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


class OpponentBelief:
    """Belief about one opponent, driven only by publicly observed plays.

    Parameters
    ----------
    vocab:
        Card vocabulary (provides the observation vector width and costs).
    deck_prior:
        Public meta deck pool.  The true opponent deck is *not* passed here.
    n_particles:
        Particle count.  128 is enough for the 8-card cycle problems in this
        engine and keeps rollout collection cheap.
    """

    def __init__(
        self,
        vocab,
        deck_prior: Sequence[Sequence[str]] | None = None,
        n_particles: int = 128,
        seed: int = 0,
        start_elixir: float = START_ELIXIR,
    ):
        self.vocab = vocab
        self.n_cards = int(vocab.n_cards)
        self.n_particles = max(8, int(n_particles))
        self.rng = random.Random(seed)
        self.deck_prior = [tuple(d) for d in (deck_prior or []) if len(d) == 8]
        self.counter = ElixirCounter(start=float(start_elixir))
        self.revealed: set[str] = set()
        self.revealed_counts: dict[str, int] = {}
        self.last_play_ms: float | None = None
        self.particles: list[_Particle] = []
        self._deck_card_names = vocab.deck_card_names()
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        self.counter.reset()
        self.revealed = set()
        self.revealed_counts = {}
        self.last_play_ms = None
        self.particles = []
        self._init_particles()

    def _init_particles(self) -> None:
        prior = self.deck_prior
        if not prior:
            return
        self.particles = []
        for i in range(self.n_particles):
            deck = prior[self.rng.randrange(len(prior))]
            self.particles.append(_Particle.from_deck(deck, self.rng))
        self._normalize()

    def _normalize(self) -> None:
        if not self.particles:
            return
        total = sum(p.weight for p in self.particles)
        if total <= 1e-12:
            # All hypotheses were rejected.  This can only happen if every
            # positive-weight particle diverges from the observed cycle; a
            # uniform reset keeps the filter usable instead of dividing by zero.
            uniform = 1.0 / len(self.particles)
            for p in self.particles:
                p.weight = uniform
            return
        for p in self.particles:
            p.weight /= total

    def _ess(self) -> float:
        if not self.particles:
            return 0.0
        denom = sum(p.weight * p.weight for p in self.particles)
        return 1.0 / denom if denom > 1e-30 else 0.0

    def _resample(self) -> None:
        if not self.particles:
            return
        n = len(self.particles)
        # Systematic resampling (low variance).
        weights = np.asarray([max(0.0, p.weight) for p in self.particles], dtype=np.float64)
        total = float(weights.sum())
        if total <= 1e-30:
            weights = np.full(n, 1.0 / n)
        else:
            weights = weights / total
        positions = (self.rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        new: list[_Particle] = []
        j = 0
        for pos in positions:
            while j < n - 1 and cumulative[j] < pos:
                j += 1
            src = self.particles[j]
            new.append(_Particle(src.deck, list(src.hand), list(src.cycle), 1.0 / n))
        self.particles = new

    def _fallback_after_mismatch(self) -> None:
        """Re-initialise particles when the observed deck is outside the prior.

        We keep the revealed cards (public) and sample the rest from the
        engine's deck-card list.  The cycle state is necessarily approximate
        after this, and the observation exposes the particle ESS so the model
        can learn to discount the belief.
        """
        revealed = [c for c in self._deck_card_names if c in self.revealed]
        pool = [c for c in self._deck_card_names if c not in self.revealed]
        if len(pool) < 8 - len(revealed):
            # Extremely defensive: allow repeats only if the engine catalogue
            # is smaller than a deck, which should never happen.
            pool = list(self._deck_card_names)
        self.particles = []
        for _ in range(self.n_particles):
            extra = self.rng.sample(pool, max(0, 8 - len(revealed)))
            deck = list(revealed) + extra
            if len(deck) < 8:
                deck += [pool[0]] * (8 - len(deck))
            self.rng.shuffle(deck)
            p = _Particle.from_deck(deck[:8], self.rng)
            self.particles.append(p)
        self._normalize()

    # --------------------------------------------------------------- updates
    def observe_play(self, card: str, t_ms: float, cost: float | None = None) -> None:
        card = str(card)
        if cost is None:
            cost = self.vocab.info(card).cost if self.vocab.card_meta(card) else None
        self.counter.observe_play(float(t_ms), cost)
        self.last_play_ms = float(t_ms)
        self.revealed.add(card)
        self.revealed_counts[card] = self.revealed_counts.get(card, 0) + 1

        if not self.particles:
            self._init_particles()
            if not self.particles:
                self._fallback_after_mismatch()
                return
        alive_weight = 0.0
        dead = 0
        for p in self.particles:
            if p.weight <= 0.0:
                dead += 1
                continue
            if p.play(card):
                alive_weight += p.weight
            else:
                p.weight = 0.0
                dead += 1
        # Weight zero means the hypothesis cannot explain the observed card.
        # Checking the *weighted* mass (not the raw particle count) prevents
        # previously-rejected particles from keeping a depleted filter alive.
        if alive_weight <= 1e-12:
            self._fallback_after_mismatch()
            return
        self._normalize()
        if dead or self._ess() < 0.5 * len(self.particles):
            self._resample()

    def update_time(self, t_ms: float) -> None:
        """Advance the elixir clock without claiming a play."""
        self.counter.advance(float(t_ms))

    # ------------------------------------------------------------ features
    def hand_probs(self) -> np.ndarray:
        out = np.zeros(self.n_cards, dtype=np.float32)
        total = sum(p.weight for p in self.particles)
        if total <= 1e-12:
            return out
        for p in self.particles:
            w = p.weight / total
            for c in p.hand:
                out[self.vocab.id(c)] += w
        return np.clip(out, 0.0, 1.0)

    def next_probs(self) -> np.ndarray:
        out = np.zeros(self.n_cards, dtype=np.float32)
        total = sum(p.weight for p in self.particles)
        if total <= 1e-12:
            return out
        for p in self.particles:
            if p.cycle:
                out[self.vocab.id(p.cycle[0])] += p.weight / total
        return np.clip(out, 0.0, 1.0)

    def deck_probs(self) -> np.ndarray:
        out = np.zeros(self.n_cards, dtype=np.float32)
        total = sum(p.weight for p in self.particles)
        if total <= 1e-12:
            return out
        for p in self.particles:
            w = p.weight / total
            for c in p.deck:
                out[self.vocab.id(c)] += w
        return np.clip(out, 0.0, 1.0)

    def summary(self) -> dict:
        hand = self.hand_probs()
        total = sum(p.weight for p in self.particles) or 1.0
        ess = self._ess()
        return {
            "elixir": float(self.counter.est),
            "plays": int(self.counter.plays),
            "rebases": int(self.counter.rebases),
            "revealed": sorted(self.revealed),
            "ess_frac": float(ess / max(1, len(self.particles))),
            "hand_entropy": _entropy(hand),
            "n_particles": len(self.particles),
        }

    def belief_vector(self) -> np.ndarray:
        """Fixed-size public belief input: hand, next, deck probabilities + scalars."""
        ess_frac = self._ess() / max(1, len(self.particles))
        entropy = _entropy(self.hand_probs())
        max_entropy = math.log(max(2, self.n_cards))
        since_last = ((self.counter.t_ms - self.last_play_ms) / 30_000.0
                      if self.last_play_ms is not None else 1.0)
        scalars = np.array([
            np.clip(self.counter.est / MAX_ELIXIR, 0.0, 1.0),
            np.clip(ess_frac, 0.0, 1.0),
            min(1.0, len(self.revealed) / 8.0),
            np.clip(entropy / max_entropy, 0.0, 1.0),
            np.clip(since_last, 0.0, 1.0),
            min(1.0, self.counter.rebases / 8.0),
        ], dtype=np.float32)
        return np.concatenate([self.hand_probs(), self.next_probs(), self.deck_probs(), scalars])

    @property
    def belief_dim(self) -> int:
        return 3 * self.n_cards + 6
