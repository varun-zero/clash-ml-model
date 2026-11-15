"""Paired evaluation for competitive v2 checkpoints, with Wilson 95% CIs.

Examples:
    python tools/eval_paired.py --checkpoint checkpoints/competitive/latest.pt \
        --vs scripted --pairs 20
    python tools/eval_paired.py --checkpoint checkpoints/competitive/latest.pt \
        --vs snapshot --snapshot checkpoints/competitive/pool/step_000001000.pt --pairs 20

Every pair is played twice with the seats swapped, so deck/seat bias cancels.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.competitive.cards import default_vocab, vocab_from_checkpoint
from cr_alphastar_agent.competitive.evaluation import evaluate_paired
from cr_alphastar_agent.competitive.league import (
    ModelOpponent, ScriptedOpponent, load_competitive_model,
)


def held_out_decks(vocab, train_pool, n: int, seed: int):
    """Random decks that are absent from the training pool."""
    keys = {tuple(sorted(d)) for d in train_pool}
    rng = random.Random(seed + 991)
    out = []
    for _ in range(max(8, n * 20)):
        if len(out) >= n:
            break
        try:
            deck = vocab.random_deck(rng)
        except ValueError:
            break
        if tuple(sorted(deck)) not in keys:
            out.append(deck)
    return out or [list(d) for d in train_pool[:max(1, n)]]


def build_matchups(decks, pairs: int, seed: int):
    rng = random.Random(seed)
    matchups = []
    for i in range(pairs):
        learner = decks[i % len(decks)]
        opp = decks[(i * 3 + 1) % len(decks)]
        if tuple(sorted(learner)) == tuple(sorted(opp)) and len(decks) > 1:
            opp = decks[(i * 3 + 2) % len(decks)]
        matchups.append((learner, opp, rng.randrange(2**31)))
    return matchups


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vs", choices=("scripted", "random", "snapshot", "wait"),
                    default="scripted")
    ap.add_argument("--snapshot", default=None, help="checkpoint for --vs snapshot")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--held-out", action=argparse.BooleanOptionalAction, default=True,
                    help="evaluate on random decks absent from the training pool")
    ap.add_argument("--json", default=None, help="write the result JSON here")
    args = ap.parse_args(argv)

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = vocab_from_checkpoint(payload)
    model, _, _ = load_competitive_model(args.checkpoint, device, vocab)

    from cr_alphastar_agent.competitive.league import RandomOpponent
    if args.vs == "scripted":
        factory = lambda: ScriptedOpponent("tactical", seed=args.seed)
        name = "scripted:tactical"
    elif args.vs == "random":
        factory = lambda: RandomOpponent(seed=args.seed)
        name = "random"
    elif args.vs == "wait":
        from cr_alphastar_agent.competitive.env import WaitOpponent
        factory = lambda: WaitOpponent()
        name = "wait"
    else:
        if not args.snapshot:
            ap.error("--vs snapshot requires --snapshot PATH")
        snap_model, _, _ = load_competitive_model(args.snapshot, device, vocab)
        factory = lambda: ModelOpponent(snap_model, device, Path(args.snapshot).stem,
                                        temperature=args.temperature, greedy=args.greedy,
                                        seed=args.seed)
        name = Path(args.snapshot).stem

    deck_pool = [list(d) for d in (payload.get("deck_pool") or [])]
    train_cfg = payload.get("train_config") or {}
    if not deck_pool:
        from cr_alphastar_agent.competitive.cards import deck_pool_from_config
        deck_pool = deck_pool_from_config(
            train_cfg.get("league", {}) or {}, vocab, args.seed,
        )
    explicit_eval = (train_cfg.get("league", {}) or {}).get("eval_decks") or []
    if explicit_eval:
        eval_decks = [list(d) for d in explicit_eval]
    elif args.held_out:
        eval_decks = held_out_decks(vocab, deck_pool, 4, args.seed)
    else:
        eval_decks = deck_pool
    matchups = build_matchups(eval_decks, args.pairs, args.seed)
    result = evaluate_paired(
        model, device, vocab, deck_pool, matchups, factory,
        greedy=args.greedy, opponent_name=name,
    )
    print(f"vs {name}: {result.matches} matches "
          f"{result.wins}W/{result.losses}L/{result.draws}D "
          f"wr={result.win_rate:.3f} CI95=[{result.ci_low:.3f},{result.ci_high:.3f}]")
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
