"""Training monitor: learning curve, behavioural drift, first-vs-last match.

Reads the artefacts written by ``HybridPPOTrainer``:

* ``checkpoints/<dir>/training_history.json`` - per-update metrics
* ``checkpoints/<dir>/match_log.json``       - per-match behavioural telemetry

and produces

* a multi-panel learning curve PNG,
* a console report with the win-rate trend,
* a "first N matches vs last N matches" comparison showing concretely what
  the agent changed (spell waste, ignored defences, placement spread).

Usage:
    python tools/monitor_training.py --dir checkpoints/current_deck
    python tools/monitor_training.py --dir checkpoints/current_deck --watch 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

from cr_alphastar_agent.config import CARDS_CATALOG  # noqa: E402


# ----------------------------------------------------------------- loading
def load_json(path: Path, retries: int = 3) -> List[dict]:
    """Load a JSON list, retrying briefly if it is being rewritten.

    The trainer writes its telemetry atomically now, but a stale or
    half-written file must never be reported as "no data" - that reads as
    a regression rather than a transient.
    """
    for attempt in range(retries):
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, list) else []
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.4)
    return []


def rolling(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    out = []
    acc = []
    for v in values:
        acc.append(v)
        if len(acc) > window:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


# ------------------------------------------------------------------ metrics
def match_win_rate(matches: List[dict]) -> float:
    if not matches:
        return 0.0
    wins = sum(1 for m in matches if m.get("result") == "WIN")
    draws = sum(1 for m in matches if m.get("result") == "DRAW")
    return (wins + 0.5 * draws) / len(matches)


def compare_windows(matches: List[dict], window: int) -> Optional[Dict[str, Any]]:
    """Compare the first ``window`` matches with the last ``window`` ones."""
    if len(matches) < max(4, window):
        return None
    first = matches[:window]
    last = matches[-window:]
    return {"first": _aggregate(first), "last": _aggregate(last), "window": window}


def split_by_update(matches: List[dict], split_update: int) -> Dict[str, Any]:
    """Compare the matches before and after a given update.

    Used to measure whether a resumed run (with a relaxed KL leash, for
    example) actually improved on the run it continued from.
    """
    before = [m for m in matches if int(m.get("update", 0)) < split_update]
    after = [m for m in matches if int(m.get("update", 0)) >= split_update]
    return {
        "split_update": split_update,
        "before": _aggregate(before),
        "after": _aggregate(after),
    }


def _aggregate(rows: List[dict]) -> Dict[str, float]:
    n = max(1, len(rows))
    deploys = [float(r.get("deploys", 0)) for r in rows]
    return {
        "matches": len(rows),
        "win_rate": match_win_rate(rows),
        "deploys_per_match": sum(deploys) / n,
        "waits_per_match": sum(float(r.get("waits", 0)) for r in rows) / n,
        "spells": sum(float(r.get("spells", 0)) for r in rows),
        "spells_own_half": sum(float(r.get("spells_own_half", 0)) for r in rows),
        "spells_king_tower": sum(float(r.get("spells_king_tower", 0)) for r in rows),
        "pushes_while_threatened": sum(float(r.get("pushes_while_threatened", 0)) for r in rows),
        "defensive_responses": sum(float(r.get("defensive_responses", 0)) for r in rows),
        "distinct_tiles": sum(float(r.get("distinct_tiles", 0)) for r in rows) / n,
        "tile_entropy": sum(float(r.get("tile_entropy", 0.0)) for r in rows) / n,
    }

    a, b = _aggregate(first), _aggregate(last)
    return {"first": a, "last": b, "window": window}


def print_report(history: List[dict], matches: List[dict], window: int = 20,
                 split_update: int = 0) -> None:
    print("=" * 78)
    print("  CLASHBOT TRAINING MONITOR")
    print("=" * 78)

    if history:
        last = history[-1]
        print(f"\n[PROGRESS] update {last.get('update')} | step {last.get('step'):,}")
        print(f"  cumulative : {last.get('total_wins', 0)}W-"
              f"{last.get('total_losses', 0)}L-{last.get('total_draws', 0)}D "
              f"over {last.get('total_matches', 0)} matches "
              f"({last.get('win_rate', 0.0):.1f}%)")
        print(f"  pg_loss    : {last.get('pg_loss', 0.0):+.4f}   "
              f"v_loss {last.get('v_loss', 0.0):.4f}   "
              f"entropy {last.get('entropy', 0.0):.3f}")
        recent = history[-10:]
        if recent:
            avg = sum(h.get("win_rate", 0.0) for h in recent) / len(recent)
            print(f"  last 10 updates mean win rate: {avg:.1f}%")
    else:
        print("\n[PROGRESS] no training history yet")

    if not matches:
        print("\n[MATCHES] no completed matches yet")
        return

    print(f"\n[MATCHES] {len(matches)} completed")
    wr = match_win_rate(matches)
    half = max(1, len(matches) // 2)
    first_half = match_win_rate(matches[:half])
    second_half = match_win_rate(matches[half:])
    print(f"  overall win rate      : {wr * 100:.1f}%")
    print(f"  first half of run     : {first_half * 100:.1f}%")
    print(f"  second half of run    : {second_half * 100:.1f}%")
    delta = (second_half - first_half) * 100
    arrow = "UP" if delta > 1 else ("DOWN" if delta < -1 else "FLAT")
    print(f"  trend                 : {arrow} ({delta:+.1f} points)")

    cmp = compare_windows(matches, window)
    if cmp is None:
        print(f"\n  (need more than {window} matches for a first-vs-last comparison)")
        return

    if split_update > 0:
        split = split_by_update(matches, split_update)
        before, after = split["before"], split["after"]
        if before["matches"] and after["matches"]:
            print(f"\n[PHASE SPLIT AT UPDATE {split_update}]")
            print(f"  before: {before['matches']} matches @ {before['win_rate'] * 100:.1f}%")
            print(f"  after : {after['matches']} matches @ {after['win_rate'] * 100:.1f}%  "
                  f"({(after['win_rate'] - before['win_rate']) * 100:+.1f} points)")
            print(f"  deploys/match {before['deploys_per_match']:.1f} -> "
                  f"{after['deploys_per_match']:.1f}   "
                  f"tile entropy {before['tile_entropy']:.2f} -> {after['tile_entropy']:.2f} bits")

    a, b = cmp["first"], cmp["last"]
    print(f"\n[FIRST {cmp['window']} MATCHES  vs  LAST {cmp['window']} MATCHES]")
    rows = [
        ("win rate", "win_rate", "{:.1f}%", 100.0),
        ("deploys / match", "deploys_per_match", "{:.1f}", 1.0),
        ("waits / match", "waits_per_match", "{:.1f}", 1.0),
        ("spells cast", "spells", "{:.0f}", 1.0),
        ("spells in OWN half", "spells_own_half", "{:.0f}", 1.0),
        ("spells at ENEMY KING", "spells_king_tower", "{:.0f}", 1.0),
        ("pushes while threatened", "pushes_while_threatened", "{:.0f}", 1.0),
        ("defensive responses", "defensive_responses", "{:.0f}", 1.0),
        ("distinct tiles / match", "distinct_tiles", "{:.1f}", 1.0),
        ("tile entropy (bits)", "tile_entropy", "{:.2f}", 1.0),
    ]
    print(f"  {'metric':<26}{'first':>14}{'last':>14}{'change':>14}")
    print("  " + "-" * 66)
    for label, key, fmt, scale in rows:
        fv = a.get(key, 0.0) * scale
        lv = b.get(key, 0.0) * scale
        print(f"  {label:<26}{fmt.format(fv):>14}{fmt.format(lv):>14}"
              f"{fmt.format(lv - fv):>14}")

    print("\n[WHAT CHANGED]")
    notes = []
    if b["spells_own_half"] < a["spells_own_half"]:
        notes.append(f"stopped wasting spells in our own half "
                     f"({a['spells_own_half']:.0f} -> {b['spells_own_half']:.0f})")
    if b["spells_king_tower"] < a["spells_king_tower"]:
        notes.append(f"stopped activating the enemy king tower "
                     f"({a['spells_king_tower']:.0f} -> {b['spells_king_tower']:.0f})")
    if b["pushes_while_threatened"] < a["pushes_while_threatened"]:
        notes.append(f"stopped pushing while defending "
                     f"({a['pushes_while_threatened']:.0f} -> {b['pushes_while_threatened']:.0f})")
    if b["defensive_responses"] > a["defensive_responses"]:
        notes.append(f"answers threats more often "
                     f"({a['defensive_responses']:.0f} -> {b['defensive_responses']:.0f})")
    if b["distinct_tiles"] > a["distinct_tiles"]:
        notes.append(f"uses more distinct placement tiles "
                     f"({a['distinct_tiles']:.1f} -> {b['distinct_tiles']:.1f})")
    if b["tile_entropy"] > a["tile_entropy"] + 0.05:
        notes.append(f"placement spread grew "
                     f"({a['tile_entropy']:.2f} -> {b['tile_entropy']:.2f} bits)")
    if b["win_rate"] > a["win_rate"] + 0.01:
        notes.append(f"win rate improved "
                     f"({a['win_rate'] * 100:.1f}% -> {b['win_rate'] * 100:.1f}%)")
    if not notes:
        print("  (no statistically meaningful drift yet - keep training)")
    for note in notes:
        print(f"  - {note}")


def plot_curve(history: List[dict], matches: List[dict], out_path: Path,
               mark_update: int = 0) -> None:
    """Eight-panel training curve.

    Panels: cumulative win rate, losses, policy entropy, card-cycle rate,
    win rate vs steps, per-match rolling win rate, placement spread and the
    BC-anchor KL leash.
    """
    if not HAS_MPL:
        print("[Monitor] matplotlib unavailable - skipping plot")
        return
    if not history and not matches:
        print("[Monitor] nothing to plot yet")
        return

    fig, axes = plt.subplots(2, 4, figsize=(23, 9))
    fig.patch.set_facecolor("#12141c")
    for ax in axes.flat:
        ax.set_facecolor("#1b1e29")
        ax.tick_params(colors="#c8ccd8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#39405a")
        ax.grid(True, color="#2a3048", alpha=0.6, linewidth=0.6)
        ax.title.set_color("#e8ecf6")
        ax.xaxis.label.set_color("#9aa3bd")
        ax.yaxis.label.set_color("#9aa3bd")

    if mark_update > 0:
        for ax in axes.flat:
            ax.axvline(mark_update, color="#fbbf24", linestyle="--", linewidth=1.4,
                       alpha=0.85)

    if history:
        updates = [h.get("update", i) for i, h in enumerate(history)]
        steps = [h.get("step", 0) for h in history]
        wr = [h.get("win_rate", 0.0) for h in history]

        axes[0, 0].plot(updates, wr, color="#4ade80", linewidth=1.8)
        axes[0, 0].set_title("Cumulative win rate (%)")
        axes[0, 0].set_xlabel("update")

        axes[0, 1].plot(updates, [h.get("pg_loss", 0.0) for h in history],
                        color="#60a5fa", linewidth=1.5, label="pg")
        axes[0, 1].plot(updates, [h.get("v_loss", 0.0) for h in history],
                        color="#f472b6", linewidth=1.5, label="value")
        axes[0, 1].set_title("Losses")
        axes[0, 1].set_xlabel("update")
        axes[0, 1].legend(facecolor="#1b1e29", edgecolor="#39405a",
                          labelcolor="#c8ccd8", fontsize=8)

        axes[0, 2].plot(updates, [h.get("entropy", 0.0) for h in history],
                        color="#fbbf24", linewidth=1.5, label="total")
        axes[0, 2].plot(updates, [h.get("card_entropy", 0.0) for h in history],
                        color="#c084fc", linewidth=1.2, label="card")
        axes[0, 2].plot(updates, [h.get("tile_entropy", 0.0) for h in history],
                        color="#38bdf8", linewidth=1.2, label="tile")
        axes[0, 2].set_title("Policy entropy")
        axes[0, 2].set_xlabel("update")
        axes[0, 2].legend(facecolor="#1b1e29", edgecolor="#39405a",
                          labelcolor="#c8ccd8", fontsize=8)

        axes[1, 0].plot(steps, wr, color="#4ade80", linewidth=1.8)
        axes[1, 0].set_title("Win rate vs environment steps")
        axes[1, 0].set_xlabel("step")

        # BC anchor leash telemetry.
        if any("anchor_kl" in h for h in history):
            axes[1, 3].plot(updates, [h.get("anchor_kl", 0.0) for h in history],
                            color="#fb7185", linewidth=1.5, label="anchor KL")
            axes[1, 3].plot(updates, [h.get("bc_kl_target", 0.0) for h in history],
                            color="#fbbf24", linewidth=1.2, linestyle="--", label="KL target")
            axes[1, 3].plot(updates, [h.get("bc_kl_beta", 0.0) for h in history],
                            color="#22d3ee", linewidth=1.2, label="leash beta")
            axes[1, 3].set_title("BC anchor KL leash")
            axes[1, 3].set_xlabel("update")
            axes[1, 3].legend(facecolor="#1b1e29", edgecolor="#39405a",
                              labelcolor="#c8ccd8", fontsize=8)

    if matches:
        idx = list(range(1, len(matches) + 1))
        wr_match = rolling([1.0 if m.get("result") == "WIN" else
                            (0.5 if m.get("result") == "DRAW" else 0.0)
                            for m in matches], 15)
        axes[1, 1].plot(idx, wr_match, color="#4ade80", linewidth=1.8)
        axes[1, 1].set_title("Per-match win rate (rolling 15)")
        axes[1, 1].set_xlabel("match")
        axes[1, 1].set_ylim(-0.05, 1.05)

        axes[1, 2].plot(idx, [m.get("tile_entropy", 0.0) for m in matches],
                        color="#a78bfa", linewidth=1.5, label="tile entropy")
        axes[1, 2].plot(idx, [m.get("distinct_tiles", 0.0) for m in matches],
                        color="#22d3ee", linewidth=1.2, label="distinct tiles")
        axes[1, 2].set_title("Placement spread")
        axes[1, 2].set_xlabel("match")
        axes[1, 2].legend(facecolor="#1b1e29", edgecolor="#39405a",
                          labelcolor="#c8ccd8", fontsize=8)

        # Card cycle: how often the agent actually plays a card.  Only
        # meaningful for matches logged with the fixed telemetry (which
        # counts the wait actions as well).
        if any("decisions" in m for m in matches):
            deploys = [float(m.get("deploys", 0)) for m in matches]
            decisions = []
            for m, d in zip(matches, deploys):
                total = float(m.get("decisions", 0)) or (d + float(m.get("waits", 0)))
                decisions.append(total)
            cycle = [100.0 * d / max(1.0, t) for d, t in zip(deploys, decisions)]
            axes[0, 3].plot(idx, cycle, color="#34d399", linewidth=1.5)
            axes[0, 3].set_title("Card cycle rate (% of decisions played)")
            axes[0, 3].set_xlabel("match")
            axes[0, 3].set_ylim(0, 100)
        else:
            axes[0, 3].set_title("Card cycle rate (no telemetry)")
            axes[0, 3].text(0.5, 0.5, "waits were not logged\nin these matches",
                            ha="center", va="center", color="#9aa3bd", fontsize=9,
                            transform=axes[0, 3].transAxes)

    fig.suptitle("ClashBot - Current Deck Training (BC + RL)", color="#f8fafc", fontsize=15)
    if mark_update > 0:
        fig.text(0.5, 0.945, f"dashed line = update {mark_update} (run resumed, relaxed KL leash)",
                 color="#fbbf24", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Monitor] Learning curve -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor ClashBot training progress")
    ap.add_argument("--dir", default="checkpoints/current_deck",
                    help="Checkpoint directory to monitor")
    ap.add_argument("--window", type=int, default=20,
                    help="Number of matches per window for the first-vs-last comparison")
    ap.add_argument("--plot", default=None, help="Where to write the learning curve PNG")
    ap.add_argument("--watch", type=int, default=0,
                    help="Re-print the report every N seconds (0 = print once)")
    ap.add_argument("--split-update", type=int, default=0,
                    help="Compare the matches before this update with the ones after it "
                         "(e.g. the resume boundary of a continued run); also marks it "
                         "on the learning curve")
    args = ap.parse_args()

    base = Path(args.dir)
    plot_path = Path(args.plot) if args.plot else base / "learning_curve.png"

    while True:
        history = load_json(base / "training_history.json")
        matches = load_json(base / "match_log.json")
        print("\033c", end="")
        print_report(history, matches, args.window, args.split_update)
        plot_curve(history, matches, plot_path, args.split_update)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
