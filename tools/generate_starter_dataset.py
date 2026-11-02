"""
Generates starter match trajectory shards in data/trajectories for immediate training.
Each shard represents a full match with global card IDs, stacked 32ch grids,
12-dim scalars, legality masks, and match outcome tags.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.helpers import generate_synthetic_shard


def main():
    traj_dir = os.path.join(REPO_ROOT, "data", "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    # Shard 1: Match 1 - Victory (outcome=1)
    shard_1 = os.path.join(traj_dir, "match_shard_001_win.npz")
    generate_synthetic_shard(shard_1, num_samples=200, match_outcome=1)
    print(f"[Dataset] Generated {shard_1} (200 frames, Outcome: Win)")

    # Shard 2: Match 2 - Defeat (outcome=-1)
    shard_2 = os.path.join(traj_dir, "match_shard_002_loss.npz")
    generate_synthetic_shard(shard_2, num_samples=200, match_outcome=-1)
    print(f"[Dataset] Generated {shard_2} (200 frames, Outcome: Loss)")

    # Shard 3: Match 3 - Victory (outcome=1)
    shard_3 = os.path.join(traj_dir, "match_shard_003_win.npz")
    generate_synthetic_shard(shard_3, num_samples=200, match_outcome=1)
    print(f"[Dataset] Generated {shard_3} (200 frames, Outcome: Win)")

    print("[Dataset] Starter trajectory dataset ready in data/trajectories/.")


if __name__ == "__main__":
    main()

