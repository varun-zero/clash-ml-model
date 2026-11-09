"""Generate training curves for local use and an optional Obsidian vault."""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_PNG = os.path.join(PROJECT_ROOT, "checkpoints", "clashbot_training_curve.png")
PROJECT_PNG = os.path.join(PROJECT_ROOT, "clashbot_training_curve.png")


def generate_training_plot(history_data: list, output_paths: list = None) -> str:
    """
    Renders a 3-panel dark-themed telemetry visualization:
    1. Win Rate (%) Over Updates (vs Historical AI & Mirror Self-Play)
    2. Policy Gradient Loss & Value Critic MSE
    3. Match Outcomes Breakdown (Wins, Losses, Draws)
    """
    if not history_data:
        return ""

    if output_paths is None:
        output_paths = [LOCAL_PNG, PROJECT_PNG]
        obsidian_vault = os.environ.get("CLASHBOT_OBSIDIAN_VAULT")
        if obsidian_vault:
            output_paths.insert(
                0, os.path.join(obsidian_vault, "clashbot_training_curve.png")
            )

    # Extract series
    updates = [h.get("update", i + 1) for i, h in enumerate(history_data)]
    steps = [h.get("step", 0) for h in history_data]
    pg_losses = [h.get("pg_loss", 0.0) for h in history_data]
    v_losses = [h.get("v_loss", 0.0) for h in history_data]
    win_rates = [h.get("win_rate", 50.0) for h in history_data]
    c_wins = [h.get("total_wins", 0) for h in history_data]
    c_losses = [h.get("total_losses", 0) for h in history_data]
    c_draws = [h.get("total_draws", 0) for h in history_data]

    # Theme colors
    bg_color = "#07090e"
    panel_color = "#0f131c"
    grid_color = "#1e2638"
    text_color = "#f1f5f9"
    muted_color = "#94a3b8"
    blue = "#38bdf8"
    emerald = "#10b981"
    rose = "#f43f5e"
    amber = "#f59e0b"
    purple = "#c084fc"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "text.color": text_color,
        "axes.labelcolor": text_color,
        "xtick.color": muted_color,
        "ytick.color": muted_color,
    })

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), facecolor=bg_color, dpi=130)
    plt.subplots_adjust(hspace=0.38, top=0.92, bottom=0.08, left=0.08, right=0.94)

    latest = history_data[-1]
    cur_step = latest.get("step", 0)
    cur_upd = latest.get("update", len(history_data))
    cur_wr = latest.get("win_rate", 50.0)
    cur_matches = latest.get("total_matches", 0)
    cur_w = latest.get("total_wins", 0)
    cur_l = latest.get("total_losses", 0)
    cur_d = latest.get("total_draws", 0)

    fig.suptitle(
        f"ALPHASTAR // NEURAL SELF-PLAY TELEMETRY  •  UPDATE {cur_upd:04d} (STEP {cur_step:,})\n"
        f"Pure AI vs AI Self-Play  |  Matches: {cur_matches:,}  |  Win Rate: {cur_wr:.1f}%  |  Wins: {cur_w} • Losses: {cur_l} • Draws: {cur_d}",
        fontsize=12, fontweight="bold", color=text_color, y=0.97
    )

    # -------------------------------------------------------------
    # Panel 1: Win Rate (%) Over Updates
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor(panel_color)
    for spine in ax1.spines.values():
        spine.set_color(grid_color)
    ax1.grid(True, color=grid_color, linestyle="--", alpha=0.6)

    ax1.plot(updates, win_rates, color=blue, linewidth=2.2, label="Agent Win Rate (%)", zorder=4)
    ax1.fill_between(updates, win_rates, 50, where=[w >= 50 for w in win_rates],
                     color=emerald, alpha=0.15, interpolate=True)
    ax1.fill_between(updates, win_rates, 50, where=[w < 50 for w in win_rates],
                     color=rose, alpha=0.15, interpolate=True)

    # Reference lines
    ax1.axhline(50, color=muted_color, linestyle=":", linewidth=1.2, alpha=0.8, label="Parity Baseline (50%)")
    ax1.axhline(60, color=emerald, linestyle="--", linewidth=1.2, alpha=0.8, label="Decisive Threshold (60%)")

    ax1.set_ylabel("Win Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(max(0, min(win_rates) - 10), min(100, max(win_rates) + 10))
    ax1.set_title("1. Head-to-Head Win Rate Against Self-Play Pool & Past Checkpoints", fontsize=11, fontweight="bold", pad=8, loc="left", color=blue)
    ax1.legend(loc="lower right", facecolor=panel_color, edgecolor=grid_color, fontsize=8)

    # -------------------------------------------------------------
    # Panel 2: PPO Policy & Value Loss
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor(panel_color)
    for spine in ax2.spines.values():
        spine.set_color(grid_color)
    ax2.grid(True, color=grid_color, linestyle="--", alpha=0.6)

    ax2.plot(updates, pg_losses, color=purple, linewidth=1.8, label="Policy Loss (PG)", zorder=3)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(updates, v_losses, color=amber, linewidth=1.8, label="Value Loss (Critic MSE)", zorder=3)
    ax2_twin.set_ylabel("Value Loss (MSE)", fontsize=10, fontweight="bold", color=amber)
    ax2_twin.tick_params(axis="y", labelcolor=amber)
    for spine in ax2_twin.spines.values():
        spine.set_color(grid_color)

    ax2.set_ylabel("Policy Loss", fontsize=10, fontweight="bold", color=purple)
    ax2.tick_params(axis="y", labelcolor=purple)
    ax2.set_title("2. PPO Optimization Losses (Policy Gradient Improvement & Critic MSE Convergence)", fontsize=11, fontweight="bold", pad=8, loc="left", color=purple)

    # -------------------------------------------------------------
    # Panel 3: Cumulative Match Outcomes (Wins, Losses, Draws)
    # -------------------------------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor(panel_color)
    for spine in ax3.spines.values():
        spine.set_color(grid_color)
    ax3.grid(True, color=grid_color, linestyle="--", alpha=0.6)

    ax3.plot(updates, c_wins, color=emerald, linewidth=2.0, label=f"Total Wins ({cur_w})", zorder=4)
    ax3.plot(updates, c_losses, color=rose, linewidth=2.0, label=f"Total Losses ({cur_l})", zorder=4)
    ax3.plot(updates, c_draws, color=amber, linewidth=1.8, linestyle="--", label=f"Total Draws ({cur_d})", zorder=4)

    ax3.set_xlabel("Training Update Iteration", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Cumulative Games", fontsize=10, fontweight="bold")
    ax3.set_title("3. Cumulative Game Outcomes in Self-Play (Genuine Engine Statistics)", fontsize=11, fontweight="bold", pad=8, loc="left", color=emerald)
    ax3.legend(loc="upper left", facecolor=panel_color, edgecolor=grid_color, fontsize=8)

    # Format ticks
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Save to all target paths
    saved_paths = []
    for p in output_paths:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            plt.savefig(p, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
            saved_paths.append(p)
        except Exception as e:
            pass
    plt.close(fig)

    return saved_paths[0] if saved_paths else ""

