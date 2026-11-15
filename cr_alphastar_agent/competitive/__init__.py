"""Competitive v2 stack: public-information observations, belief modelling,
transformer policy, PPO/KL-leash training, PFSP league and replay BC.

This package is additive to the v1 ``cr_alphastar_agent`` stack.  It does not
import or modify any v1 checkpoint, model or environment.
"""
from __future__ import annotations

__all__ = [
    "cards",
    "belief",
    "obs",
    "env",
    "model",
    "policy",
    "ppo",
    "league",
    "replay",
    "bc",
    "evaluation",
]
