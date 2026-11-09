"""
AlphaStar Prioritized Fictitious Self-Play (PFSP) League Pool Manager.
Manages opponent selection for Player 2 during parallel environment rollouts:
- 20% ScriptedBot (guarantees fundamental defense memory)
- 50% Prioritized Historical League Snapshots (prioritizes competitive past versions)
- 30% Live Mirror Self-Play (tracks moving meta)
"""

import os
import glob
import copy
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from cr_alphastar_agent.config import (
    PFSP_RUSH_EXPLOITER_RATIO, PFSP_SCRIPTED_RATIO, PFSP_HISTORICAL_RATIO, PFSP_CURRENT_RATIO,
    SNAPSHOT_INTERVAL
)
from cr_alphastar_agent.env.scripted_bot import ScriptedBot, TacticalProBot
from cr_alphastar_agent.env.rush_exploiter_bot import RushExploiterBot


class SnapshotRecord:
    def __init__(self, step: int, path: str, state_dict: dict):
        self.step = step
        self.path = path
        self.state_dict = state_dict
        self.games_played = 0
        self.wins_against_live = 0

    @property
    def win_rate_against_live(self) -> float:
        if self.games_played == 0:
            return 0.5  # Neutral default prior
        return self.wins_against_live / self.games_played


class SelfPlayLeagueManager:
    def __init__(self, pool_dir: str = "./checkpoints/pool", device: str = "cpu"):
        self.pool_dir = pool_dir
        self.device = torch.device(device)
        os.makedirs(pool_dir, exist_ok=True)

        self.scripted_bot = ScriptedBot(player_id=2)
        self.rush_bot = RushExploiterBot(player_id=2)
        self.tactical_bot = TacticalProBot(player_id=2)
        self.snapshots: List[SnapshotRecord] = []
        self._load_existing_snapshots()

    def _load_existing_snapshots(self):
        ckpt_files = sorted(glob.glob(os.path.join(self.pool_dir, "snapshot_*.pt")))
        for f in ckpt_files:
            try:
                ckpt = torch.load(f, map_location=self.device)
                step = ckpt.get("step", 0)
                record = SnapshotRecord(step, f, ckpt["model_state_dict"])
                self.snapshots.append(record)
            except Exception as e:
                print(f"[SelfPlayLeague] Error loading snapshot {f}: {e}")
        if self.snapshots:
            print(f"[SelfPlayLeague] Initialized with {len(self.snapshots)} historical snapshots from {self.pool_dir}")

    def add_snapshot(self, model: torch.nn.Module, step: int):
        """Snapshots current model into the frozen league pool."""
        snap_path = os.path.join(self.pool_dir, f"snapshot_{step:08d}.pt")
        # Detach and clone state dict to CPU
        cpu_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        state = {
            "step": step,
            "model_state_dict": cpu_state
        }
        snap_tmp = os.path.join(self.pool_dir, f"snap_{os.getpid()}_{int(time.time()*1000)}.tmp")
        for attempt in range(3):
            try:
                torch.save(state, snap_tmp)
                os.replace(snap_tmp, snap_path)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[SelfPlayLeague] Warning: could not save snapshot {snap_path}: {e}")
                else:
                    time.sleep(0.15 * (attempt + 1))
            finally:
                if os.path.exists(snap_tmp):
                    try:
                        os.remove(snap_tmp)
                    except Exception:
                        pass
        record = SnapshotRecord(step, snap_path, cpu_state)
        self.snapshots.append(record)
        print(f"[SelfPlayLeague] New historical snapshot saved: {snap_path} (Pool size: {len(self.snapshots)})")

    def sample_opponent_type(self, pure_self_play: bool = False) -> str:
        """
        Samples opponent category:
        If pure_self_play is True (Stage 3 Human Tactical League):
          - 25% TacticalProBot (demonstrating back-of-king slow pushes, kiting, tank-tricking)
          - 45% Prioritized Historical League Snapshots of our neural network
          - 30% Live Mirror Self-Play of our neural network
        If pure_self_play is False:
          - Standard AlphaStar League distribution (Rush, Scripted baseline, Historical, Mirror)
        """
        if pure_self_play:
            r = np.random.rand()
            if r < 0.25:
                return "tactical_pro"
            elif r < 0.70 and self.snapshots:
                return "historical"
            else:
                return "current"

        r = np.random.rand()
        if r < PFSP_RUSH_EXPLOITER_RATIO:
            return "rush"
        elif r < (PFSP_RUSH_EXPLOITER_RATIO + PFSP_SCRIPTED_RATIO):
            return "scripted"
        elif r < (PFSP_RUSH_EXPLOITER_RATIO + PFSP_SCRIPTED_RATIO + PFSP_HISTORICAL_RATIO) and self.snapshots:
            return "historical"
        else:
            return "current"

    def sample_historical_snapshot(self) -> SnapshotRecord:
        """
        AlphaStar PFSP: Prioritizes snapshots that have high win rate against current policy.
        Uses weighting: P(opp) proportional to (win_rate_against_live + 0.1)
        """
        if not self.snapshots:
            raise ValueError("No historical snapshots available.")

        weights = np.array([max(0.05, s.win_rate_against_live) for s in self.snapshots], dtype=np.float32)
        probs = weights / weights.sum()
        chosen_idx = np.random.choice(len(self.snapshots), p=probs)
        return self.snapshots[chosen_idx]

    def record_match_result(self, snapshot: Optional[SnapshotRecord], won_against_live: bool):
        """Updates win-rate tracking for prioritized sampling."""
        if snapshot is not None:
            snapshot.games_played += 1
            if won_against_live:
                snapshot.wins_against_live += 1

