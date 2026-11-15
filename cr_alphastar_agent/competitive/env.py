"""Competitive (v2) environment: public observations, factored actions, 500 ms cadence.

One ``CompetitiveEnv`` owns a match and a learner seat.  The opponent is any
object with ``act(obs, env) -> Action``.  Both seats use the public observation
contract; each seat has its own opponent-belief filter, so neither policy can
see the other's hand or elixir.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from crsim.engine import Battle, Tower

from .belief import OpponentBelief
from .cards import CardVocab, default_vocab
from .obs import (
    Action, MAX_UNITS, Observation, build_observation, decode_action,
    tower_potential,
)


@dataclass
class CompetitiveEnvConfig:
    decision_ticks: int = 5          # 5 x 100 ms = 500 ms
    max_units: int = MAX_UNITS
    n_particles: int = 128
    tower_weight: float = 0.5
    crown_weight: float = 0.15
    start_elixir: float = 5.0
    illegal_action_penalty: float = 0.0


class WaitOpponent:
    name = "wait"

    def act(self, obs: Observation, env: "CompetitiveEnv") -> Action:
        return Action(gate=0)


class CompetitiveEnv:
    def __init__(
        self,
        deck_pool: Sequence[Sequence[str]],
        cfg: CompetitiveEnvConfig | None = None,
        *,
        seed: int = 0,
        vocab: CardVocab | None = None,
    ):
        self.cfg = cfg or CompetitiveEnvConfig()
        self.vocab = vocab or default_vocab()
        self.deck_pool = [list(d) for d in deck_pool]
        self.rng = random.Random(seed)
        self.battle: Battle | None = None
        self.learner_team = 0
        self.opponent: Any = WaitOpponent()
        self.opponent_name = "wait"
        self.beliefs: dict[int, OpponentBelief] = {}
        self.history: dict[int, list[tuple[str, float, float, float]]] = {0: [], 1: []}
        self.initial_total_hp = 1.0
        self.last_obs: dict[int, Observation] = {}
        self.decisions = 0
        self.invalid_actions = 0
        self.decks: list[list[str]] = [[], []]

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        learner_deck: Sequence[str],
        opponent_deck: Sequence[str],
        *,
        seed: int | None = None,
        learner_team: int | None = None,
        opponent: Any | None = None,
        opponent_name: str | None = None,
    ) -> Observation:
        seed = self.rng.randrange(2**31) if seed is None else int(seed)
        learner_team = self.rng.randrange(2) if learner_team is None else int(learner_team)
        learner_deck = list(learner_deck)
        opponent_deck = list(opponent_deck)
        self.decks = [learner_deck if learner_team == 0 else opponent_deck,
                      opponent_deck if learner_team == 0 else learner_deck]
        self.battle = Battle(self.decks, seed=seed)
        self.learner_team = learner_team
        self.opponent = opponent if opponent is not None else WaitOpponent()
        self.opponent_name = opponent_name or getattr(self.opponent, "name", type(self.opponent).__name__)
        self.beliefs = {
            team: OpponentBelief(
                self.vocab,
                deck_prior=self.deck_pool,
                n_particles=self.cfg.n_particles,
                seed=seed + 97 * (team + 1),
                start_elixir=self.cfg.start_elixir,
            )
            for team in (0, 1)
        }
        self.history = {0: [], 1: []}
        self.initial_total_hp = max(
            1.0,
            sum(float(u.maxhp) for u in self.battle.units if isinstance(u, Tower)),
        )
        self.decisions = 0
        self.invalid_actions = 0
        self.last_obs = {}
        if hasattr(self.opponent, "reset"):
            self.opponent.reset(self)
        obs = self.observe(self.learner_team)
        return obs

    # ------------------------------------------------------------- observation
    def observe(self, team: int) -> Observation:
        assert self.battle is not None
        obs = build_observation(
            self.battle,
            team,
            self.beliefs[team],
            vocab=self.vocab,
            past_own=self.history[team],
            past_opp=self.history[1 - team],
            initial_total_hp=self.initial_total_hp,
            max_units=self.cfg.max_units,
            tower_weight=self.cfg.tower_weight,
            crown_weight=self.cfg.crown_weight,
        )
        self.last_obs[team] = obs
        return obs

    def observe_learner(self) -> Observation:
        return self.observe(self.learner_team)

    def observe_opponent(self) -> Observation:
        return self.observe(1 - self.learner_team)

    # ------------------------------------------------------------------- step
    def step(self, learner_action: Action | None) -> tuple[Observation, float, bool, dict]:
        if self.battle is None:
            raise RuntimeError("reset() must be called first")
        if self.battle.done:
            return self.observe_learner(), 0.0, True, self._info()

        opp_obs = self.observe(1 - self.learner_team)
        opp_action = self.opponent.act(opp_obs, self)

        actions: list[tuple[int, int, float, float]] = []
        played: dict[int, tuple[str, float]] = {}
        # Capture hand cards before the engine advances the cycle.
        for team, action in ((self.learner_team, learner_action), (1 - self.learner_team, opp_action)):
            if action is None or int(action.gate) == 0:
                continue
            idx = int(action.hand_slot)
            if 0 <= idx < len(self.battle.hand[team]):
                card = self.battle.hand[team][idx]
            else:
                card = ""
            decoded = decode_action(action, self.battle, team)
            if decoded is None:
                if team == self.learner_team:
                    self.invalid_actions += 1
                continue
            actions.append(decoded)
            played[team] = (card, 0.0)

        # Apply the simultaneous actions on the first tick, then let the engine
        # run out the decision window.
        self.battle.step(actions)
        for _ in range(max(0, int(self.cfg.decision_ticks) - 1)):
            if self.battle.done:
                break
            self.battle.step([])
        self.decisions += 1

        # Public event ingestion: the opponent sees the card appear, not the
        # hand it came from.
        for team, (card, _) in played.items():
            if card:
                self.beliefs[1 - team].observe_play(card, float(self.battle.t))
                self.history[team].append((card, 0.0, 0.0, float(self.battle.t)))
                # Coordinates are not needed by the belief, but the obs past
                # window uses them; fill from the action that was played.
                # (The engine action carries legal engine coordinates.)
        if actions:
            # Replace the placeholder positions with the actual deployment
            # coordinates.  This is public information (the unit appears there).
            action_by_team = {a[0]: a for a in actions}
            for team, (card, _) in list(played.items()):
                a = action_by_team.get(team)
                if a is None:
                    continue
                _, _, ex, ey = a
                if self.history[team]:
                    self.history[team][-1] = (card, float(ex), float(ey), float(self.battle.t))

        done = bool(self.battle.done)
        reward = 0.0
        if done:
            if self.battle.result is None:
                reward = 0.0
            elif int(self.battle.result) == self.learner_team:
                reward = 1.0
            else:
                reward = -1.0
        obs = self.observe(self.learner_team)
        if done:
            obs.phi = 0.0  # potential is zero at the terminal state
        info = self._info()
        return obs, reward, done, info

    # ------------------------------------------------------------------ helper
    def _info(self) -> dict:
        b = self.battle
        assert b is not None
        return {
            "t_ms": float(b.t),
            "done": bool(b.done),
            "result": b.result,
            "crowns": list(b.crowns),
            "learner_team": self.learner_team,
            "learner_deck": list(self.decks[self.learner_team]),
            "opponent_deck": list(self.decks[1 - self.learner_team]),
            "opponent": self.opponent_name,
            "invalid_actions": self.invalid_actions,
            "decisions": self.decisions,
        }

    # ---------------------------------------------------------------- rollout
    def run_episode(
        self,
        learner_policy,
        learner_deck: Sequence[str],
        opponent_deck: Sequence[str],
        *,
        seed: int | None = None,
        learner_team: int | None = None,
        opponent: Any | None = None,
        opponent_name: str | None = None,
    ) -> dict:
        """Convenience loop used by BC/evaluation.  ``learner_policy(obs, env)``
        returns an Action.  Returns the episode summary plus the recorded rows."""
        obs = self.reset(learner_deck, opponent_deck, seed=seed, learner_team=learner_team,
                         opponent=opponent, opponent_name=opponent_name)
        rows = []
        total_reward = 0.0
        while True:
            action = learner_policy(obs, self)
            rows.append((obs, action))
            obs, reward, done, info = self.step(action)
            total_reward += reward
            if done:
                break
        info = dict(info)
        info["total_reward"] = total_reward
        info["rows"] = rows
        return info
