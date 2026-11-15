"""Live-match data capture for online learning.

When the agent plays a real match against the ladder, the transitions are
logged to a replay file.  Those replays can be turned into a behaviour-cloning
dataset (``tools/build_live_dataset.py``) and used to fine-tune the policy,
so every real match improves the model instead of being thrown away.

This module is used by ``deploy_mumu.py`` to record matches; it is
deliberately free of emulator-specific code so it can also be unit tested.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class LiveMatchRecorder:
    """Records the decisions of a live match into a JSONL file.

    Each line is one decision:

    .. code-block:: json

        {"t_ms": 12000, "obs": {...}, "action": [2, 301], "reward": 0.0,
         "done": false, "result": null}

    ``obs`` holds the same tensors the policy consumed, serialised as lists.
    """

    path: str
    session: Dict[str, Any] = field(default_factory=dict)
    _handle: Any = None
    _count: int = 0

    def __post_init__(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a", encoding="utf-8")

    # ------------------------------------------------------------------ api
    def record(self, obs: Dict[str, np.ndarray], action: Sequence[int],
               *, reward: float = 0.0, done: bool = False,
               result: Optional[int] = None, t_ms: int = 0) -> None:
        row = {
            "t_ms": int(t_ms),
            "obs": {k: np.asarray(v).tolist() for k, v in obs.items()},
            "action": [int(a) for a in action],
            "reward": float(reward),
            "done": bool(done),
            "result": result,
            "session": self.session,
        }
        self._handle.write(json.dumps(row) + "\n")
        self._handle.flush()
        self._count += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "LiveMatchRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def load_live_replays(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Load recorded live decisions from one or more JSONL files."""
    rows: List[Dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            files = sorted(p.glob("*.jsonl"))
        else:
            files = [p]
        for f in files:
            if not f.exists():
                continue
            with open(f, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return rows


def rows_to_dataset(rows: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Convert recorded rows into the arrays the BC trainer consumes."""
    from cr_alphastar_agent.config import TOTAL_TILES

    keys = (
        "scalars", "hand_cards", "entities", "entity_mask", "arena_grid",
        "card_mask", "tile_mask",
    )
    obs_store: Dict[str, List[np.ndarray]] = {k: [] for k in keys}
    card_tile: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    values: List[float] = []

    for row in rows:
        obs = row.get("obs") or {}
        action = row.get("action")
        if not action or len(action) < 2:
            continue
        try:
            arrays = {k: np.asarray(obs[k]) for k in keys if k in obs}
        except (KeyError, ValueError):
            continue
        if len(arrays) != len(keys):
            continue
        for k in keys:
            obs_store[k].append(arrays[k])
        ctm = obs.get("card_tile_mask")
        if ctm is not None:
            card_tile.append(np.asarray(ctm))
        actions.append(np.asarray(action[:2], dtype=np.int64))
        values.append(float(row.get("reward", 0.0)))

    if not actions:
        return {}

    dataset: Dict[str, np.ndarray] = {
        k: np.stack(v).astype(np.float32) if k not in ("hand_cards", "card_mask", "entity_mask")
        else np.stack(v) for k, v in obs_store.items()
    }
    dataset["hand_cards"] = dataset["hand_cards"].astype(np.int64)
    dataset["card_mask"] = dataset["card_mask"].astype(bool)
    dataset["entity_mask"] = dataset["entity_mask"].astype(bool)
    dataset["action"] = np.stack(actions).astype(np.int64)
    dataset["value"] = np.asarray(values, dtype=np.float32)
    if card_tile and len(card_tile) == len(actions):
        dataset["card_tile_mask"] = np.stack(card_tile).astype(bool)
    return dataset


__all__ = ["LiveMatchRecorder", "load_live_replays", "rows_to_dataset"]
