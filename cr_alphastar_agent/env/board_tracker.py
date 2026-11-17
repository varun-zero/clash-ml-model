"""Stateful live board understanding.

``detect_entities()`` returns a *bag* of blobs once every ~0.45 s.  That is
enough to drive a policy but not enough to answer the questions a defender
actually asks:

* where is each enemy troop going?
* which of my towers is in danger?
* is the opponent assembling a push?
* is a tower nearly dead?

This module keeps a short history of detections, smooths positions, estimates
velocity/destination, and turns the result into a :class:`BoardSnapshot` that
the live bridge uses to tighten its tactical masks.  It deliberately does
**not** change the network input dimensions, so the existing checkpoint stays
valid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cr_alphastar_agent.config import ARENA_HEIGHT, ARENA_WIDTH

#: Canonical y of the river (rows 0..15 are ours, 16..31 are the opponent's).
RIVER_Y = 16.0
#: A unit this far from a princess tower is considered a threat to it.
TOWER_THREAT_RADIUS = 6.0
#: How many consecutive ticks a unit must move toward a tower to count as a push.
PUSH_MIN_TICKS = 3
#: A group this large moving together is a push.
PUSH_MIN_UNITS = 2
#: How long a track survives without a detection (seconds).
TRACK_TTL_S = 4.0
#: Maximum number of tracks kept.
MAX_TRACKS = 24


@dataclass
class UnitTrack:
    """A smoothed enemy unit track in canonical tile coordinates."""

    track_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    hp: float = 1.0
    card_id: int = 5
    card_name: str = "unknown"
    confidence: float = 0.0
    age_ticks: int = 0
    missed_ticks: int = 0
    last_seen: float = 0.0
    history: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    @property
    def pushing_toward_our_half(self) -> bool:
        """Is it travelling toward the river / our side?"""
        return self.vy < -0.05

    @property
    def destination(self) -> Tuple[float, float]:
        """Rough predicted position three seconds ahead, clamped to the arena."""
        return (
            float(np.clip(self.x + 3.0 * self.vx, 0.0, ARENA_WIDTH - 1)),
            float(np.clip(self.y + 3.0 * self.vy, 0.0, ARENA_HEIGHT - 1)),
        )


@dataclass
class TowerState:
    """What the agent knows about one of our towers."""

    name: str
    x: float
    y: float
    hp_ratio: float = 1.0
    hp_known: bool = False
    incoming: int = 0
    threat: float = 0.0
    seconds_to_impact: Optional[float] = None


@dataclass
class BoardSnapshot:
    """Everything the defender needs for the current tick."""

    enemy_tracks: List[UnitTrack]
    towers: Dict[str, TowerState]
    under_threat: bool
    push_detected: bool
    push_lane_x: Optional[float]
    push_size: int
    most_dangerous_tower: Optional[str]
    estimated_enemy_elixir: float = 5.0

    @property
    def threat_x(self) -> Optional[float]:
        """Lane centre of the most advanced enemy, for same-lane defence."""
        if not self.enemy_tracks:
            return None
        return float(min(self.enemy_tracks, key=lambda u: u.y).x)

    @property
    def defense_target(self) -> Optional[Tuple[float, float]]:
        """Recommended defensive tile: in front of the endangered tower."""
        if self.most_dangerous_tower is None or self.most_dangerous_tower not in self.towers:
            if not self.enemy_tracks:
                return None
            u = min(self.enemy_tracks, key=lambda t: t.y)
            return (float(np.clip(u.x, 1, ARENA_WIDTH - 2)),
                    float(np.clip(u.y + 2.0, 8, 14)))
        tower = self.towers[self.most_dangerous_tower]
        return (float(np.clip(tower.x, 1, ARENA_WIDTH - 2)),
                float(np.clip(tower.y + 2.0, 8, 13)))


class BoardTracker:
    """Turns per-tick detections into a tactical board snapshot."""

    def __init__(self) -> None:
        self._tracks: Dict[int, UnitTrack] = {}
        self._next_id = 1
        self._tower_ratios: Dict[str, float] = {}
        self._last_update = 0.0

    # ------------------------------------------------------------------ tracks
    def _match(self, x: float, y: float, dt: float) -> Optional[int]:
        """Nearest *unclaimed* predicted track position.

        Predicting from velocity before matching is essential: two troops in
        the same lane are often only one tile apart, and a plain nearest
        match swaps their identities every tick, which destroys the velocity
        estimate and makes push detection impossible.
        """
        best_id = None
        best_d = 1.2
        for track in self._tracks.values():
            if track.track_id in self._claimed:
                continue
            px = track.x + track.vx * dt
            py = track.y + track.vy * dt
            d = float(np.hypot(px - x, py - y))
            if d < best_d:
                best_d = d
                best_id = track.track_id
        return best_id

    def update(self, enemy_units: Sequence[Dict], now: float,
               dt: Optional[float] = None) -> BoardSnapshot:
        """Update with this tick's raw enemy detections.

        ``enemy_units`` items need ``x``/``y``; optional ``card_id``,
        ``card_name``, ``confidence`` and ``hp`` are used when available.
        """
        if dt is None:
            dt = max(0.1, now - self._last_update) if self._last_update else 0.45
        self._last_update = now

        # Expire stale tracks.
        for tid in list(self._tracks):
            track = self._tracks[tid]
            if now - track.last_seen > TRACK_TTL_S:
                self._tracks.pop(tid, None)

        self._claimed: set = set()
        seen: set = set()
        # Match the most advanced units first: they are the ones that matter
        # for defence and the most likely to be confused with each other.
        ordered = sorted(enemy_units, key=lambda o: float(o.get("y", 0.0)))
        for obs in ordered:
            x = float(obs.get("x", 0.0))
            y = float(obs.get("y", 0.0))
            tid = self._match(x, y, dt)
            if tid is None:
                if len(self._tracks) >= MAX_TRACKS:
                    continue
                tid = self._next_id
                self._next_id += 1
                track = UnitTrack(track_id=tid, x=x, y=y,
                                  last_seen=now,
                                  hp=float(obs.get("hp", 1.0)),
                                  card_id=int(obs.get("card_id", 5)),
                                  card_name=str(obs.get("card_name", "unknown")),
                                  confidence=float(obs.get("confidence", 0.0)))
                self._tracks[tid] = track
            else:
                track = self._tracks[tid]
                # Exponential smoothing keeps the velocity estimate stable.
                alpha = 0.45
                new_x = alpha * x + (1.0 - alpha) * track.x
                new_y = alpha * y + (1.0 - alpha) * track.y
                track.vx = (new_x - track.x) / dt
                track.vy = (new_y - track.y) / dt
                track.x = new_x
                track.y = new_y
                track.hp = float(obs.get("hp", track.hp))
                if obs.get("card_name") and obs["card_name"] != "unknown":
                    track.card_name = str(obs["card_name"])
                    track.card_id = int(obs.get("card_id", track.card_id))
                    track.confidence = float(obs.get("confidence", track.confidence))
                track.missed_ticks = 0
                track.last_seen = now
                track.age_ticks += 1
            track.history.append((track.x, track.y))
            if len(track.history) > 12:
                track.history.pop(0)
            seen.add(tid)
            self._claimed.add(tid)

        for tid, track in self._tracks.items():
            if tid not in seen:
                track.missed_ticks += 1
                # coast the track briefly so we do not lose a unit during a
                # single missed detection
                if track.missed_ticks <= 2:
                    track.x = float(np.clip(track.x + track.vx * dt, 0, ARENA_WIDTH - 1))
                    track.y = float(np.clip(track.y + track.vy * dt, 0, ARENA_HEIGHT - 1))

        return self.snapshot(now)

    # ----------------------------------------------------------------- towers
    def set_tower_hp(self, name: str, ratio: float) -> None:
        self._tower_ratios[name] = float(np.clip(ratio, 0.0, 1.0))

    # --------------------------------------------------------------- snapshot
    def snapshot(self, now: float) -> BoardSnapshot:
        towers = {
            "princess_left": TowerState("princess_left", 3.5, 6.5,
                                        self._tower_ratios.get("princess_left", 1.0),
                                        "princess_left" in self._tower_ratios),
            "princess_right": TowerState("princess_right", 14.5, 6.5,
                                         self._tower_ratios.get("princess_right", 1.0),
                                         "princess_right" in self._tower_ratios),
            "king": TowerState("king", 9.0, 3.0,
                               self._tower_ratios.get("king", 1.0),
                               "king" in self._tower_ratios),
        }

        live = [t for t in self._tracks.values() if t.missed_ticks <= 2]
        for track in live:
            for tower in towers.values():
                d = float(np.hypot(track.x - tower.x, track.y - tower.y))
                if d <= TOWER_THREAT_RADIUS:
                    tower.incoming += 1
                    closing = -track.vy if track.y > tower.y else track.vy
                    tower.threat += max(0.0, 1.0 - d / TOWER_THREAT_RADIUS) * (1.0 + closing)
                    if track.speed > 0.05:
                        tower.seconds_to_impact = min(
                            tower.seconds_to_impact or 1e9, d / track.speed
                        )

        under_threat = any(t.threat > 0.15 for t in towers.values())

        # A push = at least two units, at or above the bridge, moving together
        # toward our half.
        pushers = [t for t in live
                   if t.y >= RIVER_Y - 3.0 and t.pushing_toward_our_half
                   and t.age_ticks >= PUSH_MIN_TICKS]
        push_detected = len(pushers) >= PUSH_MIN_UNITS
        push_lane = float(np.median([t.x for t in pushers])) if pushers else None

        most_dangerous = None
        best = 0.0
        for name, tower in towers.items():
            score = tower.threat + (1.0 - tower.hp_ratio) * 0.5
            if score > best:
                best = score
                most_dangerous = name

        return BoardSnapshot(
            enemy_tracks=list(live),
            towers=towers,
            under_threat=under_threat,
            push_detected=push_detected,
            push_lane_x=push_lane,
            push_size=len(pushers),
            most_dangerous_tower=most_dangerous if best > 0.15 else None,
        )

    def reset(self) -> None:
        self._tracks.clear()
        self._tower_ratios.clear()
        self._last_update = 0.0


__all__ = ["BoardSnapshot", "BoardTracker", "TowerState", "UnitTrack"]
