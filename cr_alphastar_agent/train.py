"""
Main Entry Point for AlphaStar + OpenAI Five Clash Royale Agent Training.
Supports:
- Auto-detection of Colab T4 (16GB VRAM) vs Local RTX 3050 (4-6GB VRAM) vs CPU.
- Continuous auto-resumption from latest.pt.
- Custom checkpoint directories (e.g., Google Drive sync).
- --smoke-test flag for fast 256-sample verification runs vs full 2048-sample production runs.
"""

import os
import sys

# Ensure clashbot root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import argparse
import torch

from cr_alphastar_agent.config import (
    get_hardware_profile, PPO_LR, ROLLOUT_STEPS, NUM_ENVS
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.training.hybrid_ppo_trainer import HybridPPOTrainer
from cr_alphastar_agent.training.obsidian_sync import sync_batch_to_obsidian


def main():
    parser = argparse.ArgumentParser(description="AlphaStar + OpenAI Five Clash Royale Trainer")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Fast smoke-test mode (2 updates, 4 envs x 64 steps = 256 samples/update)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Path to checkpoint and pool directory (e.g., /content/drive/MyDrive/cr_ckpts on Colab)")
    parser.add_argument("--total-updates", type=int, default=None,
                        help="Total number of PPO update iterations to run (default: 1000, smoke-test: 2)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help=f"Number of parallel headless environments (default: {NUM_ENVS})")
    parser.add_argument("--rollout-steps", type=int, default=None,
                        help=f"Steps per environment per rollout (default: {ROLLOUT_STEPS}, smoke-test: 64)")
    parser.add_argument("--lr", type=float, default=PPO_LR,
                        help="AdamW learning rate")
    parser.add_argument(
        "--pure-self-play",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use neural self-play only (disable with --no-pure-self-play; 0%% scripted bots)",
    )
    parser.add_argument("--env-backend", type=str, default="crsim", choices=["crsim", "replica"],
                        help="Physics engine backend (crsim = 100%% APK data-driven, replica = lightweight)")
    args = parser.parse_args()

    # Determine execution scale
    if args.smoke_test:
        total_updates = args.total_updates or 2
        num_envs = args.num_envs or 4
        rollout_steps = args.rollout_steps or 64
        mode_str = "SMOKE-TEST MODE"
    else:
        total_updates = args.total_updates or 1000
        num_envs = args.num_envs or NUM_ENVS
        rollout_steps = args.rollout_steps or ROLLOUT_STEPS
        mode_str = "PRODUCTION MODE (2048 samples/update)"

    print("=" * 75)
    print(f"      AlphaStar + OpenAI Five Hybrid Clash Royale Agent [{mode_str}]")
    print("=" * 75)

    # Initialize model
    model = AlphaStarCRNet()

    # Initialize trainer
    trainer = HybridPPOTrainer(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        lr=args.lr,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        pure_self_play=args.pure_self_play,
        env_backend=args.env_backend
    )

    batch_samples = num_envs * rollout_steps
    print(f"\n[Trainer Initialized] Checkpoints: {os.path.abspath(args.checkpoint_dir)}")
    print(f"[Self-Play Mode] {'Pure Neural AI vs AI (0% Scripted Bots)' if args.pure_self_play else 'Curriculum'}")
    print(f"[Training Plan] {total_updates} updates | {num_envs} parallel envs | {rollout_steps} steps/rollout")
    print(f"[Batch Size] {batch_samples} samples per update (Action Repeat: 4 ticks/step)\n")

    start_update = trainer.update_count
    for update in range(start_update, start_update + total_updates):
        t0 = time.time()

        # 1. Collect parallel rollouts
        buffer = trainer.collect_rollouts()
        t_rollout = time.time() - t0

        # 2. PPO update with dynamic gradient accumulation
        t1 = time.time()
        metrics = trainer.ppo_update(buffer)
        t_update = time.time() - t1

        total_time = t_rollout + t_update
        fps = int(batch_samples / max(1e-4, total_time))

        print(
            f"Update {trainer.update_count:04d} | "
            f"Step {trainer.global_step:07d} | "
            f"Record: {metrics.get('total_wins', 0)}W-{metrics.get('total_losses', 0)}L-{metrics.get('total_draws', 0)}D ({metrics.get('win_rate', 0.0):.1f}%) | "
            f"PG: {metrics['pg_loss']:.4f} | "
            f"V: {metrics['v_loss']:.4f} | "
            f"Ent: {metrics['entropy']:.3f} | "
            f"KL: {metrics['approx_kl']:.5f} | "
            f"FPS: {fps} | "
            f"Time: {total_time:.2f}s",
            flush=True
        )

        # 3. Synchronize live batch metrics and training graph into user's Obsidian Vault note
        try:
            sync_batch_to_obsidian(
                update_count=trainer.update_count,
                global_step=trainer.global_step,
                metrics=metrics,
                fps=fps,
                total_time=total_time,
                phase_desc=trainer.current_phase_desc,
                history_data=trainer.training_history
            )
        except Exception as e:
            pass


if __name__ == "__main__":
    main()
