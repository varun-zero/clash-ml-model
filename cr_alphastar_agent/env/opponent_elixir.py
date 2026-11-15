"""Explicit, public-information opponent elixir tracker.

The tracker reconstructs how much elixir the opponent holds using only
information a real player can see:

* the match clock (public),
* the public elixir regeneration schedule,
* the fact that the opponent played a card (their unit/spell appears on the
  board, and every card shows its cost in the hand).

It never reads the opponent's private elixir counter.  The estimate is
therefore an *interval* rather than a point value, because plays can be
missed (an unseen spell, a unit that dies before we noticed it).

The regeneration schedule mirrors ``crsim.engine.Battle`` exactly:

===========  =========  ===================
phase        window     elixir per second
===========  =========  ===================
normal       0 - 120 s  10 / 28 s
double       120 - 240 s  10 / 14 s
triple       240 s +     10 / 9.3 s
===========  =========  ===================

Usage in the simulator
----------------------
``CrSimReplicaEnv`` keeps one tracker per seat and feeds it the *cost* of
every card the opponent plays.  Card costs are public information, so this
is not a hidden-state leak: the tracker only ever sees "a 4-elixir card was
played at t = 42.3 s".

Usage on a live emulator
------------------------
Feed the same events from screen understanding (or from a deck tracker):

    tracker.reset()
    tracker.observe_play(cost=4, t_s=42.3)
    mean, low, high = tracker.estimate(t_s=45.0)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

# Clash Royale timings (seconds).
BAR_NORMAL_S = 28.0        # 10 elixir per 28 s during normal time
BAR_DOUBLE_S = 14.0        # last 2 minutes (incl. overtime start)
BAR_TRIPLE_S = 9.3         # overtime
DOUBLE_ELIXIR_START_S = 120.0
TRIPLE_ELIXIR_START_S = 240.0
ELIXIR_PERIOD_S = 2.8      # 1 elixir per 2.8 s during normal time
START_ELIXIR = 5.0
MAX_ELIXIR = 10.0


@dataclass
class ElixirEstimate:
    """Interval estimate of the opponent's elixir."""

    mean: float
    low: float
    high: float
    confidence: float
    elapsed_s: float

    @property
    def spread(self) -> float:
        return max(0.0, self.high - self.low)

    def as_vector(self) -> list[float]:
        """Four floats suitable for an observation tensor."""
        return [
            float(self.mean) / MAX_ELIXIR,
            float(self.low) / MAX_ELIXIR,
            float(self.high) / MAX_ELIXIR,
            float(self.confidence),
        ]

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": round(self.mean, 3),
            "low": round(self.low, 3),
            "high": round(self.high, 3),
            "confidence": round(self.confidence, 3),
            "elapsed_s": round(self.elapsed_s, 2),
        }


class OpponentElixirTracker:
    """Tracks the opponent's elixir from public plays only."""

    def __init__(
        self,
        *,
        start_elixir: float = START_ELIXIR,
        max_elixir: float = MAX_ELIXIR,
        bar_normal_s: float = BAR_NORMAL_S,
        bar_double_s: float = BAR_DOUBLE_S,
        bar_triple_s: float = BAR_TRIPLE_S,
        double_elixir_start_s: float = DOUBLE_ELIXIR_START_S,
        triple_elixir_start_s: float = TRIPLE_ELIXIR_START_S,
        unknown_play_cost: float = 3.0,
        base_uncertainty: float = 0.15,
        uncertainty_per_minute: float = 0.35,
        uncertainty_per_unknown: float = 0.45,
        uncertainty_per_play: float = 0.03,
        assume_missed_plays: bool = False,
    ):
        self.start_elixir = float(start_elixir)
        self.max_elixir = float(max_elixir)
        self.bar_normal_s = float(bar_normal_s)
        self.bar_double_s = float(bar_double_s)
        self.bar_triple_s = float(bar_triple_s)
        self.double_elixir_start_s = float(double_elixir_start_s)
        self.triple_elixir_start_s = float(triple_elixir_start_s)
        self.unknown_play_cost = float(unknown_play_cost)
        self.base_uncertainty = float(base_uncertainty)
        self.uncertainty_per_minute = float(uncertainty_per_minute)
        self.uncertainty_per_unknown = float(uncertainty_per_unknown)
        self.uncertainty_per_play = float(uncertainty_per_play)
        # In the simulator every play is observed exactly, so the estimate is
        # tight.  On a live emulator some plays can be missed, so the
        # uncertainty is allowed to grow with the match clock.
        self.assume_missed_plays = bool(assume_missed_plays)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._spent: float = 0.0
        self._unknown_plays: int = 0
        self._known_plays: int = 0
        self._overflow: float = 0.0
        self._last_t: float = 0.0
        self._recent: Deque[float] = deque(maxlen=32)

    # ------------------------------------------------------------- accounting
    def _regen(self, t_s: float) -> float:
        """Total elixir regenerated by time ``t_s`` (piecewise linear)."""
        if t_s <= 0.0:
            return 0.0
        total = 0.0
        # normal phase
        normal_end = min(t_s, self.double_elixir_start_s)
        if normal_end > 0.0:
            total += 10.0 * normal_end / self.bar_normal_s
        # double phase
        double_end = min(t_s, self.triple_elixir_start_s)
        if double_end > self.double_elixir_start_s:
            total += 10.0 * (double_end - self.double_elixir_start_s) / self.bar_double_s
        # triple phase (overtime)
        if t_s > self.triple_elixir_start_s:
            total += 10.0 * (t_s - self.triple_elixir_start_s) / self.bar_triple_s
        return total

    def _phase_bar_s(self, t_s: float) -> float:
        if t_s >= self.triple_elixir_start_s:
            return self.bar_triple_s
        if t_s >= self.double_elixir_start_s:
            return self.bar_double_s
        return self.bar_normal_s

    # --------------------------------------------------------------- updates
    def observe_play(self, cost: float, t_s: float) -> None:
        """Record that the opponent played a card of the given cost."""
        c = max(0.0, float(cost))
        self._spent += c
        self._known_plays += 1
        self._recent.append(float(t_s))
        self._last_t = max(self._last_t, float(t_s))

    def observe_unknown_play(self, t_s: float) -> None:
        """Record a play we saw but could not price (missed card)."""
        self._unknown_plays += 1
        self._spent += self.unknown_play_cost
        self._recent.append(float(t_s))
        self._last_t = max(self._last_t, float(t_s))

    # -------------------------------------------------------------- estimate
    def estimate(self, t_s: Optional[float] = None) -> ElixirEstimate:
        """Interval estimate of the opponent's elixir at time ``t_s``."""
        t = self._last_t if t_s is None else float(t_s)
        t = max(t, 0.0)

        gross = self.start_elixir + self._regen(t) - self._spent
        # Regeneration above the cap is wasted, so the true value can be lower
        # than our accounting; treat that as extra uncertainty.
        overflow = max(0.0, gross - self.max_elixir)
        self._overflow = overflow

        high = min(self.max_elixir, gross)
        uncertainty = (
            self.base_uncertainty
            + self.uncertainty_per_unknown * self._unknown_plays
            + self.uncertainty_per_play * self._known_plays
            + 0.35 * overflow
        )
        if self.assume_missed_plays:
            # Uncertainty grows with the clock: a play we never saw would
            # silently reduce the opponent's pool.
            uncertainty += self.uncertainty_per_minute * (t / 60.0)
        low = max(0.0, high - uncertainty)
        mean = 0.5 * (low + high)
        confidence = max(0.0, min(1.0, 1.0 - uncertainty / max(1.0, self.max_elixir)))

        return ElixirEstimate(
            mean=mean, low=low, high=high, confidence=confidence, elapsed_s=t
        )

    def mean_elixir(self, t_s: Optional[float] = None) -> float:
        return self.estimate(t_s).mean

    def to_dict(self, t_s: Optional[float] = None) -> Dict[str, float]:
        d = self.estimate(t_s).to_dict()
        d.update(
            {
                "known_plays": float(self._known_plays),
                "unknown_plays": float(self._unknown_plays),
                "spent": round(self._spent, 2),
            }
        )
        return d


__all__ = [
    "BAR_DOUBLE_S",
    "BAR_NORMAL_S",
    "BAR_TRIPLE_S",
    "DOUBLE_ELIXIR_START_S",
    "ELIXIR_PERIOD_S",
    "ElixirEstimate",
    "MAX_ELIXIR",
    "OpponentElixirTracker",
    "START_ELIXIR",
    "TRIPLE_ELIXIR_START_S",
]
