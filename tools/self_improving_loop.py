"""
CR-Zero: Autonomous Continual Learning & Self-Improving Reinforcement Loop.
Iteratively plays live battles, logs state-action-reward trajectories,
triggers Advantage-Weighted Regression (AWR) retraining, and hot-reloads
improved policy weights without manual intervention.
"""
import os
import sys
import glob
import time
import argparse
import subprocess
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def count_existing_shards(shards_dir: str) -> int:
    return len(glob.glob(os.path.join(shards_dir, "*.npz")))


def run_single_match(
    checkpoint_path: str,
    device: str = "127.0.0.1:5556",
    shards_dir: str = "data/trajectories",
    use_rule_bot: bool = False,
) -> Dict[str, any]:
    """Runs a single autonomous match using tools/play_live_match.py."""
    cmd = [
        sys.executable, "tools/play_live_match.py",
        "--device", device,
        "--log-dir", shards_dir,
        "--checkpoint", checkpoint_path,
        "--weights", "weights/yolo_units.pt",
        "--save-trajectory",
    ]
    if use_rule_bot:
        cmd.append("--rule-bot")

    print(f"\n[Self-Improve] Launching match with checkpoint: {checkpoint_path}...")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = proc.stdout + proc.stderr
        duration = time.time() - t0

        won = "MATCH_WON" in output or "Victory detected" in output or "VICTORY" in output
        lost = "MATCH_LOST" in output or "Defeat detected" in output or "DEFEAT" in output
        result = "WIN" if won else ("LOSS" if lost else "UNKNOWN")

        print(f"[Self-Improve] Match completed in {duration:.1f}s. Outcome: {result}")
        return {"outcome": result, "duration": duration, "output": output}
    except subprocess.TimeoutExpired:
        print("[Self-Improve] Match timed out after 300s.")
        return {"outcome": "TIMEOUT", "duration": 300.0, "output": ""}
    except Exception as e:
        print(f"[Self-Improve] Match execution error: {e}")
        return {"outcome": "ERROR", "duration": 0.0, "output": str(e)}


def retrain_policy_awr(
    shards_dir: str = "data/trajectories",
    init_checkpoint: str = "checkpoints/cr_zero_policy_best.pt",
    epochs: int = 10,
    batch_size: int = 64,
    beta: float = 0.5,
    lr: float = 2e-4,
) -> bool:
    """Executes Stage 2 Advantage-Weighted Regression retraining."""
    print(f"\n{'='*70}")
    print(f"[Self-Improve Retraining] Launching AWR fine-tuning on aggregated buffer ({shards_dir})...")
    print(f"{'='*70}")

    cmd = [
        sys.executable, "colab_pipeline/train_awr.py",
        "--shards-dir", shards_dir,
        "--init-checkpoint", init_checkpoint,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--beta", str(beta),
        "--lr", str(lr),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = proc.stdout + proc.stderr
        print(proc.stdout)
        if proc.returncode == 0 and "Stage 2 Advantage-Weighted Regression Training Complete!" in output:
            print("[Self-Improve Retraining] Success! Checkpoint promoted and ONNX model updated.")
            return True
        else:
            print(f"[Self-Improve Retraining] Error or non-zero exit code: {proc.returncode}")
            return False
    except Exception as e:
        print(f"[Self-Improve Retraining] Failed to run AWR trainer: {e}")
        return False


def run_self_improving_pipeline(
    target_wins: int = 5,
    retrain_every: int = 3,
    max_matches: int = 20,
    checkpoint: str = "checkpoints/cr_zero_policy_awr_best.pt",
    device: str = "127.0.0.1:5556",
    shards_dir: str = "data/trajectories",
    epochs_per_round: int = 10,
):
    """Orchestrates continual play, experience accumulation, and RL improvement."""
    print(f"{'='*70}")
    print(f"CR-Zero Autonomous Self-Improving Continual RL Loop")
    print(f"Target Wins: {target_wins} | Retrain Every: {retrain_every} matches | Device: {device}")
    print(f"{'='*70}\n")

    wins = 0
    total_played = 0
    shards_at_last_retrain = count_existing_shards(shards_dir)
    print(f"[Init] Initial trajectory shard count in buffer: {shards_at_last_retrain}")

    while wins < target_wins and total_played < max_matches:
        total_played += 1
        print(f"\n>>> Game #{total_played} (Current Wins: {wins}/{target_wins}) <<<")

        match_res = run_single_match(
            checkpoint_path=checkpoint,
            device=device,
            shards_dir=shards_dir,
        )

        if match_res["outcome"] == "WIN":
            wins += 1

        current_shards = count_existing_shards(shards_dir)
        new_shards = current_shards - shards_at_last_retrain
        print(f"[Buffer Status] Total shards: {current_shards} (+{new_shards} new shards since last retrain)")

        # Trigger self-improvement retraining trigger
        if new_shards >= retrain_every:
            print(f"\n[Self-Improvement Trigger] {new_shards} new match trajectories collected!")
            success = retrain_policy_awr(
                shards_dir=shards_dir,
                init_checkpoint=checkpoint,
                epochs=epochs_per_round,
            )
            if success:
                shards_at_last_retrain = count_existing_shards(shards_dir)
                print(f"[Hot-Reload] Latest weights active for subsequent matches.")

        time.sleep(2.0)

    print(f"\n{'='*70}")
    print(f"[Self-Improvement Complete] Summary: {wins}/{total_played} wins achieved.")
    print(f"Total Trajectory Shards in Buffer: {count_existing_shards(shards_dir)}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="CR-Zero Continual Learning Self-Improving Pipeline")
    parser.add_argument("--target-wins", type=int, default=5, help="Stop after reaching target victories")
    parser.add_argument("--retrain-every", type=int, default=3, help="Trigger AWR fine-tuning after N matches")
    parser.add_argument("--max-matches", type=int, default=20, help="Maximum matches to attempt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cr_zero_policy_best.pt")
    parser.add_argument("--device", type=str, default="127.0.0.1:5556")
    parser.add_argument("--shards-dir", type=str, default="data/trajectories")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run_self_improving_pipeline(
        target_wins=args.target_wins,
        retrain_every=args.retrain_every,
        max_matches=args.max_matches,
        checkpoint=args.checkpoint,
        device=args.device,
        shards_dir=args.shards_dir,
        epochs_per_round=args.epochs,
    )


if __name__ == "__main__":
    main()

