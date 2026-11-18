"""Hidden opponent state: Elixir counter, 8-card cycle and counter availability.

A top-ladder player never reasons from the enemy units alone; they track what
the opponent *can* play next.  Clash Royale makes both hidden variables
publicly inferable:

* **Elixir** regenerates at a known rate and every played card has a known
  cost, so a counter can be kept exactly (within one pip of rounding).
* **The 8-card cycle** is public: after a card is played it cannot return to
  hand until four other cards have been played.

This module maintains both from observed spawns/spells and exposes them to
the tactical layer and the telemetry.  It never guesses a card that has not
been seen; unobserved slots stay explicitly unknown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Elixir generation: seconds per pip at each speed.
SECONDS_PER_PIP = {1: 2.8, 2: 1.4, 3: 0.93}
ELIXIR_MAX = 10.0
ELIXIR_START = 5.0
HAND_SIZE = 4
DECK_SIZE = 8
CYCLE_LOCKOUT = 4
IDLE_RESYNC_S = 3.0

#: Cards that answer a building-targeting win condition.  Kept here (rather
#: than hard-coded into the policy) so it can be tuned without retraining.
COUNTERS_TO_BUILDING_TARGETERS = frozenset({
    "Cannon", "MiniPekka", "Mini PEKKA", "SkeletonArmy", "Skeleton Army",
    "InfernoTower", "Inferno Tower", "Tesla", "Barbarians", "Fireball",
})


@dataclass
class OpponentPlay:
    card_name: str
    card_id: int
    cost: float
    t: float
    play_index: int
    kind: str = "spawn"          # spawn | spell


@dataclass
class OpponentState:
    elixir: float
    elixir_low: bool
    elixir_diff: float           # my_elixir - opp_elixir, normalised later
    cycle: List[str]
    seen_cards: List[str]
    plays: int
    counter_locked: bool
    counter_available: bool
    counter_name: Optional[str]
    counter_cards_in_hand: List[str]
    unknown_slots: int
    last_play_age_s: float
    opposite_lane_punish: bool


class OpponentTracker:
    """Track the opponent's elixir and card cycle from public plays."""

    def __init__(self, card_intel=None) -> None:
        self.card_intel = card_intel
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.elixir = ELIXIR_START
        self.my_elixir = ELIXIR_START
        self.last_t = 0.0
        self.last_play_t = 0.0
        self.plays: List[OpponentPlay] = []
        self.seen: List[str] = []
        self._spent_since_idle = 0.0

    @property
    def play_count(self) -> int:
        return len(self.plays)

    # ----------------------------------------------------------------- update
    def tick(self, t: float, my_elixir: float, speed: int = 1,
             my_activity: bool = False, opp_activity: bool = False) -> None:
        """Advance elixir generation and handle idle resynchronisation."""
        dt = max(0.0, t - self.last_t)
        self.last_t = t
        self.my_elixir = float(my_elixir)
        if dt > 0:
            self.elixir = min(ELIXIR_MAX,
                              self.elixir + dt / SECONDS_PER_PIP.get(speed, 1))
        # When both players have been idle for a few seconds, both bars are
        # full; this removes drift accumulated from missed plays.
        if not my_activity and not opp_activity and (t - self.last_play_t) > IDLE_RESYNC_S:
            self.elixir = ELIXIR_MAX

    def observe_play(self, card_name: str, t: float,
                     card_id: int = 0, kind: str = "spawn") -> Optional[OpponentPlay]:
        """Register an enemy card.  Returns the recorded play, or None."""
        cost = self._cost(card_name)
        if cost <= 0:
            return None
        self.tick(t, self.my_elixir, opp_activity=True)
        self.elixir = max(0.0, self.elixir - cost)
        self.last_play_t = t
        play = OpponentPlay(card_name, card_id, cost, t, len(self.plays), kind)
        self.plays.append(play)
        if card_name not in self.seen:
            self.seen.append(card_name)
        return play

    def _cost(self, name: str) -> float:
        if self.card_intel is None:
            return 0.0
        card = self.card_intel.get(name)
        return float(card.cost) if card is not None else 0.0

    # ----------------------------------------------------------------- query
    def cards_in_hand(self) -> List[str]:
        """The last four distinct cards played, oldest first.

        In Clash Royale a card leaves the hand when played and returns after
        four plays.  The four most recent distinct plays therefore model the
        current hand when all eight cards have been revealed.
        """
        if not self.plays:
            return []
        recent = self.plays[-(HAND_SIZE + 1):]
        seen: List[str] = []
        for play in recent:
            if play.card_name not in seen:
                seen.append(play.card_name)
        return seen[:HAND_SIZE]

    def is_locked(self, card_name: str) -> bool:
        """Is ``card_name`` known to be out of the opponent's hand?"""
        for play in reversed(self.plays):
            if play.card_name == card_name:
                return (len(self.plays) - play.play_index) < CYCLE_LOCKOUT
        return False

    def counter_status(self, counter_names: Sequence[str] = ()) -> Tuple[bool, bool, Optional[str]]:
        """(locked, available, name) for the most relevant known counter."""
        names = list(counter_names) or list(COUNTERS_TO_BUILDING_TARGETERS)
        in_hand = self.cards_in_hand()
        candidates = [n for n in names if n in self.seen]
        if not candidates:
            return False, False, None
        # Prefer a counter currently in the inferred hand.
        for name in candidates:
            if name in in_hand and not self.is_locked(name):
                return False, True, name
        for name in candidates:
            if self.is_locked(name):
                return True, False, name
        return False, False, candidates[0]

    def snapshot(self, my_elixir: Optional[float] = None, now: Optional[float] = None,
                 under_threat: bool = False) -> OpponentState:
        mine = self.my_elixir if my_elixir is None else float(my_elixir)
        locked, available, counter = self.counter_status()
        hand = self.cards_in_hand()
        low = self.elixir <= 2.5
        return OpponentState(
            elixir=round(self.elixir, 2),
            elixir_low=low,
            elixir_diff=round(mine - self.elixir, 2),
            cycle=[p.card_name for p in self.plays[-DECK_SIZE:]],
            seen_cards=list(self.seen),
            plays=len(self.plays),
            counter_locked=locked,
            counter_available=available,
            counter_name=counter,
            counter_cards_in_hand=hand,
            unknown_slots=max(0, DECK_SIZE - len(self.seen)),
            last_play_age_s=round((now or self.last_t) - self.last_play_t, 2),
            opposite_lane_punish=bool(low and not available and not under_threat),
        )


__all__ = ["OpponentPlay", "OpponentState", "OpponentTracker"]
