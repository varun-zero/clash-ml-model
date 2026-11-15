"""Paired evaluation with Wilson confidence intervals (v2)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .env import CompetitiveEnv, CompetitiveEnvConfig
from .league import ScriptedOpponent, ModelOpponent
from .obs import Action, Observation
from .policy import ActionSampler


def wilson_ci(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class EvalResult:
    matches: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    ci_low: float
    ci_high: float
    by_opponent: dict[str, dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "matches": self.matches, "wins": self.wins, "losses": self.losses,
            "draws": self.draws, "win_rate": self.win_rate,
            "ci95": [self.ci_low, self.ci_high], "by_opponent": self.by_opponent,
        }


def model_policy(model, device: torch.device, *, greedy: bool = False,
                 temperature: float = 1.0, seed: int = 0) -> Callable[[Observation, CompetitiveEnv], Action]:
    sampler = ActionSampler(model, device, seed=seed)

    def act(obs: Observation, env: CompetitiveEnv) -> Action:
        actions, _ = sampler.sample(obs, greedy=greedy, temperature=temperature)
        return actions[0]

    return act


def _run_one(
    model,
    device,
    vocab,
    deck_pool: Sequence[Sequence[str]],
    learner_deck: Sequence[str],
    opponent_deck: Sequence[str],
    seed: int,
    learner_team: int,
    opponent,
    *,
    greedy: bool,
) -> int:
    """Return 1 for learner win, -1 loss, 0 draw."""
    env = CompetitiveEnv(deck_pool, CompetitiveEnvConfig(), seed=seed, vocab=vocab)
    env.reset(learner_deck, opponent_deck, seed=seed, learner_team=learner_team,
              opponent=opponent, opponent_name=getattr(opponent, "name", "opponent"))
    policy = model_policy(model, device, greedy=greedy, seed=seed + 17)
    while True:
        action = policy(env.observe_learner(), env)
        _, _, done, info = env.step(action)
        if done:
            if info["result"] is None:
                return 0
            return 1 if int(info["result"]) == learner_team else -1


def evaluate_paired(
    model,
    device: torch.device,
    vocab,
    deck_pool: Sequence[Sequence[str]],
    matchups: Sequence[tuple[Sequence[str], Sequence[str], int]],
    opponent_factory: Callable[[], Any],
    *,
    greedy: bool = False,
    opponent_name: str = "opponent",
) -> EvalResult:
    wins = losses = draws = 0
    by: dict[str, dict[str, int]] = {}
    for learner_deck, opp_deck, seed in matchups:
        for learner_team in (0, 1):
            opponent = opponent_factory()
            r = _run_one(model, device, vocab, deck_pool, learner_deck, opp_deck,
                         int(seed), learner_team, opponent, greedy=greedy)
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1
            key = opponent_name
            d = by.setdefault(key, {"wins": 0, "losses": 0, "draws": 0})
            d["wins" if r > 0 else "losses" if r < 0 else "draws"] += 1
    n = wins + losses + draws
    lo, hi = wilson_ci(wins, n)
    return EvalResult(n, wins, losses, draws, wins / n if n else 0.0, lo, hi, by)
