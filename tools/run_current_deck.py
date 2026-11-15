"""Train the current deck on the local GPU while monitoring the run live.

Runs the full ClashAI-style pipeline on this machine:

1. behaviour cloning on the tactical teacher (``TacticalCoachBot``),
2. PPO + GAE reinforcement learning against scripted bots,

and prints a live report (learning curve, win-rate trend, first-match vs
last-match behavioural comparison) while the training runs.

Usage:
    python tools/run_current_deck.py --updates 200 --bc-matches 40
    python tools/run_current_deck.py --updates 400 --envs 8 --watch 60
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.monitor_training import load_json, plot_curve, print_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + monitor the current deck")
    ap.add_argument("--updates", type=int, default=200,
                    help="Number of PPO updates (default: 200)")
    ap.add_argument("--envs", type=int, default=4,
                    help="Parallel environments (default: 4)")
    ap.add_argument("--rollout-steps", type=int, default=256,
                    help="Steps per environment per rollout (default: 256)")
    ap.add_argument("--bc-matches", type=int, default=40,
                    help="Behaviour-cloning matches before RL (default: 40)")
    ap.add_argument("--bc-epochs", type=int, default=3,
                    help="Behaviour-cloning epochs (default: 3)")
    ap.add_argument("--teacher", type=str, default="coach",
                    choices=["coach", "tactical", "scripted"])
    ap.add_argument("--opponents", type=str, default="scripted",
                    choices=["scripted", "mixed", "self_play"])
    ap.add_argument("--deck", type=str, default="current")
    ap.add_argument("--dir", default="checkpoints/current_deck",
                    help="Output directory for checkpoints and telemetry")
    ap.add_argument("--watch", type=int, default=30,
                    help="Seconds between monitor refreshes (0 = only at the end)")
    ap.add_argument("--window", type=int, default=20,
                    help="Matches per window for the first-vs-last comparison")
    ap.add_argument("--split-update", type=int, default=0,
                    help="Compare the matches before this update with the ones after it")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training.log"

    cmd = [
        sys.executable, "-u", str(ROOT / "train_colab.py"),
        "--total-updates", str(args.updates),
        "--num-envs", str(args.envs),
        "--rollout-steps", str(args.rollout_steps),
        "--deck", args.deck,
        "--opponents", args.opponents,
        "--bc-matches", str(args.bc_matches),
        "--bc-epochs", str(args.bc_epochs),
        "--bc-teacher", args.teacher,
        "--checkpoint-dir", str(out_dir),
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print("=" * 78)
    print("  CLASHBOT - CURRENT DECK TRAINING (local GPU)")
    print("=" * 78)
    print(f"  deck        : {args.deck}")
    print(f"  opponents   : {args.opponents}")
    print(f"  plan        : BC {args.bc_matches} matches / {args.bc_epochs} epochs "
          f"-> {args.updates} PPO updates")
    print(f"  batch       : {args.envs} envs x {args.rollout_steps} steps "
          f"= {args.envs * args.rollout_steps} samples/update")
    print(f"  output      : {out_dir}")
    print(f"  log         : {log_path}")
    print("=" * 78 + "\n")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log,
                                stderr=subprocess.STDOUT, env=env)
        print(f"[Run] training process started (pid {proc.pid})")
        print(f"[Run] tail the log with:  tail -f {log_path}\n")

        try:
            while proc.poll() is None:
                time.sleep(max(5, args.watch))
                history = load_json(out_dir / "training_history.json")
                matches = load_json(out_dir / "match_log.json")
                print("\033c", end="")
                print_report(history, matches, args.window, args.split_update)
                plot_curve(history, matches, out_dir / "learning_curve.png")
                # Echo the tail of the training log so the numbers agree.
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                        tail = fh.read().splitlines()[-3:]
                    for line in tail:
                        print(f"  [log] {line}")
                except Exception:
                    pass
                print()
        except KeyboardInterrupt:
            print("\n[Run] interrupted - stopping training.")
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()

        # Final report.
        history = load_json(out_dir / "training_history.json")
        matches = load_json(out_dir / "match_log.json")
        print("\033c", end="")
        print_report(history, matches, args.window, args.split_update)
        plot_curve(history, matches, out_dir / "learning_curve.png")
        print(f"\n[Run] finished with exit code {proc.returncode}")
        print(f"[Run] checkpoint: {out_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
