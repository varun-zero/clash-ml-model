"""
Google Colab Tesla T4 Training Script for AlphaStar Clash Royale RL Agent.
Enforces 100% Pure Neural Self-Play (0% Scripted Bots) with Exact Engine Match Tracking,
FP16 Mixed Precision, Automatic Google Drive Persistence, and Real-Time Curve Generation.
"""

import os
import sys
import time
import shutil
import argparse
import datetime
import torch

# Ensure the repository root is in sys.path when this module is run directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cr_alphastar_agent.config import (
    get_hardware_profile, PPO_LR, ROLLOUT_STEPS, NUM_ENVS, EFFECTIVE_BATCH_SIZE,
    resolve_deck_ids, resolve_deck_names, ELIXIR_LEAK_WEIGHT
)
from cr_alphastar_agent.training.hybrid_ppo_trainer import (
    ENTROPY_PRESETS, SNAPSHOT_OPPONENT_RATE,
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.training.hybrid_ppo_trainer import HybridPPOTrainer
from cr_alphastar_agent.training.plot_progress import generate_training_plot


def detect_environment():
    """Detects whether running on Google Colab or local machine."""
    is_colab = "google.colab" in sys.modules or os.path.exists("/content")
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if has_cuda else 0.0
    return {
        "is_colab": is_colab,
        "has_cuda": has_cuda,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb
    }


def main():
    parser = argparse.ArgumentParser(description="Google Colab Pure Self-Play AlphaStar Trainer")
    parser.add_argument("--total-updates", type=int, default=1000,
                        help="Total PPO updates to run (default: 1000)")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Parallel headless game environments (default: 8 for T4)")
    parser.add_argument("--rollout-steps", type=int, default=256,
                        help="Steps per environment per rollout (8 x 256 = 2048 samples/update)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Local checkpoint directory")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/clashbot_training",
                        help="Google Drive backup directory for checkpoints and curves")
    parser.add_argument("--lr", type=float, default=PPO_LR,
                        help="AdamW learning rate")
    parser.add_argument("--deck", type=str, default="current",
                        help="Deck preset for the learner (current|starter|hog26|giant) or a "
                             "comma-separated card list. Default: current (the account deck).")
    parser.add_argument("--opponents", type=str, default="scripted",
                        choices=["scripted", "mixed", "self_play"],
                        help="Opponent source: scripted (rule bots), mixed (ramped), self_play (league)")
    parser.add_argument("--init-bc", type=str, default=None,
                        help="Behaviour-cloning checkpoint to initialise the policy from "
                             "(produced by warmstart_bc.py --out-name bc.pt)")
    parser.add_argument("--bc-matches", type=int, default=0,
                        help="If > 0, run behaviour cloning first (matches) and use the "
                             "result as the RL initialisation")
    parser.add_argument("--bc-epochs", type=int, default=3,
                        help="Behaviour cloning epochs when --bc-matches > 0")
    parser.add_argument("--bc-teacher", type=str, default="coach",
                        choices=["coach", "tactical", "scripted"],
                        help="Behaviour-cloning teacher bot")
    parser.add_argument("--elixir-leak-weight", type=float, default=ELIXIR_LEAK_WEIGHT,
                        help="Penalty per elixir wasted at the 10-elixir cap.  Without it the "
                             "agent learns to bank elixir and wait instead of cycling cards.")
    parser.add_argument("--entropy-preset", type=str, default="spec", choices=["spec", "diverse"],
                        help="Entropy bonus preset: 'spec' (reference coefficients) or "
                             "'diverse' (20x stronger tile bonus so the tile head cannot "
                             "collapse onto a few anchors)")
    parser.add_argument("--snapshot-opponent-rate", type=float, default=SNAPSHOT_OPPONENT_RATE,
                        help="Share of the league that is a frozen snapshot of our own past "
                             "policy (0 = scripted bots only).  Without it the league has no "
                             "learning signal and the win rate stops improving.")
    args = parser.parse_args()

    bc_checkpoint = args.init_bc
    entropy_preset = ENTROPY_PRESETS[args.entropy_preset]
    env_info = detect_environment()
    opponent_desc = {
        "scripted": "Scripted Bots (TacticalCoach / TacticalPro / Rush)",
        "mixed": "Mixed (scripted first, then PFSP league)",
        "self_play": "100% Pure Neural Self-Play (League Snapshots + Mirror)",
    }.get(args.opponents, args.opponents)
    print("=" * 80)
    print("   ⚔️  CLASHBOT ALPHASTAR // COLAB BC + RL TRAINING ENGINE  ⚔️")
    print("=" * 80)
    print(f" Environment : {'Google Colab' if env_info['is_colab'] else 'Local Machine'}")
    print(f" Accelerator : {env_info['gpu_name']} ({env_info['vram_gb']:.1f} GB VRAM)")
    print(f" Mode        : {opponent_desc}")
    print(f" Deck        : {args.deck}")
    print(f" BC Init     : {bc_checkpoint or 'none - resuming latest.pt if present'}")
    print(f" Entropy     : {args.entropy_preset} preset")
    print(f" Batch Scale : {args.num_envs} envs x {args.rollout_steps} steps = {args.num_envs * args.rollout_steps} samples/update")
    print(f" Checkpoints : {os.path.abspath(args.checkpoint_dir)}")

    # Drive sync check
    drive_available = os.path.exists("/content/drive/MyDrive") or os.path.exists(os.path.dirname(args.drive_dir))
    if drive_available:
        os.makedirs(args.drive_dir, exist_ok=True)
        print(f" Drive Sync  : Active -> {args.drive_dir}")
        # If checkpoints exist in Drive but not locally, copy them
        drive_latest = os.path.join(args.drive_dir, "latest.pt")
        local_latest = os.path.join(args.checkpoint_dir, "latest.pt")
        if os.path.exists(drive_latest) and not os.path.exists(local_latest):
            print(f" [Drive Sync] Restoring latest checkpoint from Google Drive...")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            shutil.copy2(drive_latest, local_latest)
            drive_history = os.path.join(args.drive_dir, "training_history.json")
            local_history = os.path.join(args.checkpoint_dir, "training_history.json")
            if os.path.exists(drive_history):
                shutil.copy2(drive_history, local_history)
    else:
        print(f" Drive Sync  : Disabled (Drive not mounted. Checkpoints saved locally in {args.checkpoint_dir})")
    print("=" * 80 + "\n")

    # Initialize neural model
    model = AlphaStarCRNet()

    # Optional behaviour-cloning stage before RL (ClashAI-style pipeline:
    # clone the tactical teacher, then improve with PPO against scripted bots).
    if args.bc_matches > 0:
        # If starting a fresh BC stage, archive prior telemetry so the curve starts clean
        for fname in ["training_history.json", "match_log.json"]:
            fpath = os.path.join(args.checkpoint_dir, fname)
            if os.path.exists(fpath):
                prev_path = os.path.join(args.checkpoint_dir, f"{os.path.splitext(fname)[0]}_prev.json")
                shutil.copy2(fpath, prev_path)
                try:
                    os.remove(fpath)
                except Exception:
                    pass

        from cr_alphastar_agent.warmstart_bc import run_warmstart
        bc_path = os.path.join(args.checkpoint_dir, "bc.pt")
        print("=" * 80)
        print(f"   BEHAVIOUR CLONING STAGE: {args.bc_matches} matches, {args.bc_epochs} epochs, "
              f"teacher={args.bc_teacher}, deck={args.deck}")
        print("=" * 80)
        run_warmstart(
            num_matches=args.bc_matches,
            epochs=args.bc_epochs,
            lr=1e-3,
            checkpoint_dir=args.checkpoint_dir,
            env_backend="crsim",
            deck=resolve_deck_names(args.deck),
            teacher=args.bc_teacher,
            out_name="bc.pt",
        )
        bc_checkpoint = bc_path

    # Initialize trainer
    trainer = HybridPPOTrainer(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        lr=args.lr,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        pure_self_play=(args.opponents == "self_play"),
        env_backend="crsim",
        deck=args.deck,
        opponent_mode=args.opponents,
        init_checkpoint=bc_checkpoint,
        elixir_leak_weight=args.elixir_leak_weight,
        entropy_card_ceiling=entropy_preset["card_ceiling"],
        entropy_tile_ceiling=entropy_preset["tile_ceiling"],
        entropy_card_coef=entropy_preset["card_coef"],
        entropy_tile_coef=entropy_preset["tile_coef"],
        snapshot_opponent_rate=args.snapshot_opponent_rate,
    )

    batch_samples = args.num_envs * args.rollout_steps
    start_update = trainer.update_count

    print(f"[Self-Play Initialized] Global Step: {trainer.global_step:,} | Starting Update: {start_update}")
    print(f"[Match Record] Total: {trainer.total_matches} matches | {trainer.total_p1_wins} Wins • {trainer.total_p2_wins} Losses • {trainer.total_draws} Draws\n")

    for update in range(start_update, start_update + args.total_updates):
        t0 = time.time()

        # 1. Collect parallel pure self-play rollouts
        buffer = trainer.collect_rollouts()
        t_rollout = time.time() - t0

        # 2. PPO update with dynamic gradient accumulation
        t1 = time.time()
        metrics = trainer.ppo_update(buffer)
        t_update = time.time() - t1

        total_time = t_rollout + t_update
        fps = int(batch_samples / max(1e-4, total_time))

        w = metrics.get("total_wins", 0)
        l = metrics.get("total_losses", 0)
        d = metrics.get("total_draws", 0)
        m = metrics.get("total_matches", 0)
        wr = metrics.get("win_rate", 50.0)

        bw = metrics.get("batch_wins", 0)
        bl = metrics.get("batch_losses", 0)
        bd = metrics.get("batch_draws", 0)
        bm = metrics.get("batch_matches", 0)
        bwr = metrics.get("batch_win_rate", 50.0)

        print(
            f"Update {trainer.update_count:04d} | "
            f"Step {trainer.global_step:07d} | "
            f"Batch [{bw}W-{bl}L-{bd}D ({bwr:.0f}%)] | "
            f"Cumulative [{w}W-{l}L-{d}D / {m}m ({wr:.1f}%)] | "
            f"PG: {metrics['pg_loss']:.4f} | "
            f"V: {metrics['v_loss']:.4f} | "
            f"Ent: {metrics['entropy']:.3f} | "
            f"FPS: {fps} | "
            f"Time: {total_time:.2f}s",
            flush=True
        )

        # 3. Google Drive Backup
        if drive_available and trainer.update_count % 5 == 0:
            try:
                local_latest = os.path.join(args.checkpoint_dir, "latest.pt")
                local_hist = os.path.join(args.checkpoint_dir, "training_history.json")
                local_plot = os.path.join(args.checkpoint_dir, "clashbot_training_curve.png")

                if os.path.exists(local_latest):
                    shutil.copy2(local_latest, os.path.join(args.drive_dir, "latest.pt"))
                if os.path.exists(local_hist):
                    shutil.copy2(local_hist, os.path.join(args.drive_dir, "training_history.json"))
                if os.path.exists(local_plot):
                    shutil.copy2(local_plot, os.path.join(args.drive_dir, "clashbot_training_curve.png"))
            except Exception as e:
                print(f"[Drive Sync] Warning: {e}")


if __name__ == "__main__":
    main()

