"""Smoke test: random-vs-random battles. Sanity + speed check.

    python tools/smoke.py [n_battles]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crsim.engine import load_model
from crsim import env as crsim_env

MODEL = load_model()

# a standard-ish hog cycle for sanity; validated against the logic tables
DECK_A = ["HogRider", "Knight", "Archer", "Cannon", "Fireball", "Arrows",
          "Skeletons", "Musketeer"]
DECK_B = ["Giant", "MiniPekka", "Goblins", "Wizard", "Zap", "Bomber",
          "Minions", "Valkyrie"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    results = {"b wins": 0, "r wins": 0, "draw": 0}
    t0 = time.perf_counter()
    ticks = 0
    for game in range(n):
        e = crsim_env.CrSimEnv([DECK_A, DECK_B], model=MODEL, seed=game)
        obs0, obs1 = e.reset()
        steps = 0
        while not e.done and steps < 4800:
            a0 = e.rng.choice(e.legal_actions(0)) if e.legal_actions(0) else None
            a1 = e.rng.choice(e.legal_actions(1)) if e.legal_actions(1) else None
            obs0, obs1, rewards, done = e.step([a0, a1])
            steps += 1
        ticks += steps
        winner = e.battle.result
        hp0 = obs0["towers_self"]
        hp1 = obs1["towers_self"]
        print(f"game {game}: t={e.battle.t/1000:.0f}s crowns "
              f"{e.battle.crowns[0]}-{e.battle.crowns[1]} result={winner} "
              f"blue towers {hp0} red towers {hp1}")
        if winner is None:
            results["draw"] += 1
        else:
            results["b wins" if winner == 0 else "r wins"] += 1
    dt = time.perf_counter() - t0
    print(f"{n} battles, {ticks} sim steps, {dt:.2f}s wall -> "
          f"{ticks/dt:.0f} steps/s agent-driven")
    print(results)


if __name__ == "__main__":
    main()
