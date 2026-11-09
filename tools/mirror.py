"""Mirror-match fairness check: identical decks must be ~50/50."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crsim.engine import load_model, Battle

M = load_model()
DECK = ["HogRider", "Knight", "Archer", "Cannon", "Fireball", "Arrows",
        "Skeletons", "Musketeer"]

from crsim import env as crsim_env
wins = {0: 0, 1: 0, None: 0}
crown_sums = [0, 0]
t0 = time.perf_counter(); total_steps = 0
N = 30
for g in range(N):
    e = crsim_env.CrSimEnv([DECK, DECK], model=M, seed=g)
    e.reset()
    steps = 0
    while not e.done and steps < 4800:
        a0 = e.rng.choice(e.legal_actions(0)) if e.legal_actions(0) else None
        a1 = e.rng.choice(e.legal_actions(1)) if e.legal_actions(1) else None
        if not (a0 or a1):
            e.battle.step([])
            steps += 1
            continue
        e.step([a0, a1])
        steps += 1
    r = e.battle.result
    wins[r if r is not None else None] = wins.get(r if r is not None else None, 0) + 1
    crown_sums[0] += e.battle.crowns[0]
    crown_sums[1] += e.battle.crowns[1]
    total_steps += steps
dt = time.perf_counter() - t0
print(f"mirror {N} games: blue={wins[0]} red={wins[1]} draws={wins[None]} "
      f"avg crowns {crown_sums[0]/N:.2f}-{crown_sums[1]/N:.2f} "
      f"({total_steps} steps, {dt:.1f}s, {total_steps/dt:.0f} steps/s)")
