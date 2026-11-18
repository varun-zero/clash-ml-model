#!/usr/bin/env python3
"""Head-to-head strength test between two CRSIM checkpoints.

Plays the PFSP checkpoint against the frozen baseline on identical decks and
seeds, with the seats swapped on every pair so deck/seat bias cancels out.
Actions are greedy (argmax) on both sides for a deterministic, reproducible
comparison of the two policies rather than of their sampling noise.

Usage::

    python tools/head_to_head.py --matches 100 --envs 16 --json .tmp/h2h.json

The legacy baseline checkpoint uses the old 4-scalar / 7-entity observation;
``adapt_legacy_state_dict`` zero-pads those input columns so it behaves exactly
as before.  Both networks otherwise run through the same AlphaStarCRNet code
path and the same CrSimReplicaEnv physics as training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cr_alphastar_agent.config import CURRENT_DECK_IDS  # noqa: E402
from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv  # noqa: E402
from cr_alphastar_agent.model.alphastar_cr_net import (  # noqa: E402
    AlphaStarCRNet, adapt_legacy_state_dict,
)


def load_net(path: str, device: torch.device) -> AlphaStarCRNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    state = adapt_legacy_state_dict(state)
    net = AlphaStarCRNet().to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def batch_obs(obs_list, device: torch.device):
    return {
        k: torch.from_numpy(np.stack([o[k] for o in obs_list])).to(device)
        for k in obs_list[0]
    }


def act_batched(net: AlphaStarCRNet, obs_list, device: torch.device) -> np.ndarray:
    if not obs_list:
        return np.empty((0, 2), dtype=np.int64)
    batch = batch_obs(obs_list, device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            actions, _, _, _ = net.get_action_and_value(batch, deterministic=True)
    return actions.squeeze(0).cpu().numpy() if len(obs_list) == 1 else actions.cpu().numpy()


def wilson_interval(wins: int, total: int, z: float = 1.96):
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--challenger", default="checkpoints/pfsp_run/latest.pt",
                    help="policy under test (player A)")
    ap.add_argument("--baseline", default="checkpoints/latest.pt",
                    help="frozen reference policy (player B)")
    ap.add_argument("--matches", type=int, default=100, help="total games")
    ap.add_argument("--envs", type=int, default=16, help="parallel sims")
    ap.add_argument("--seed", type=int, default=20251118)
    ap.add_argument("--max-steps", type=int, default=800,
                    help="decision cap per game (sim naturally ends at ~750)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[head-to-head] A={args.challenger}\n[head-to-head] B={args.baseline}")
    print(f"[head-to-head] {args.matches} games, {args.envs} parallel envs, device={device}")
    nets = {
        "A": load_net(args.challenger, device),
        "B": load_net(args.baseline, device),
    }
    names = {"A": os.path.basename(args.challenger), "B": os.path.basename(args.baseline)}

    n = min(args.envs, args.matches)
    deck = list(CURRENT_DECK_IDS)
    envs = [CrSimReplicaEnv(blue_deck=deck, red_deck=deck, seed=args.seed + i)
            for i in range(n)]
    # Even envs: A is Player 1.  Odd envs: B is Player 1.  Seat bias cancels.
    seats = [("A", "B") if i % 2 == 0 else ("B", "A") for i in range(n)]
    cur_p1 = [envs[i].reset(seed=args.seed + i)[0] for i in range(n)]
    game_id = [args.seed + i for i in range(n)]
    steps = [0] * n
    results = []  # dicts
    next_game = args.seed + n
    t0 = time.time()
    last_reported = 0

    while len(results) < args.matches:
        # Gather every env's P1 and P2 observation, grouped by which net acts.
        pending = {"A": [], "B": []}
        for i in range(n):
            p1_name, p2_name = seats[i]
            pending[p1_name].append((i, cur_p1[i]))
            pending[p2_name].append((i, envs[i].get_obs(player_id=2)))

        actions = {}
        for name in ("A", "B"):
            entries = pending[name]
            acts = act_batched(nets[name], [o for _, o in entries], device)
            for (i, _), a in zip(entries, acts):
                actions[(i, name)] = a

        for i in range(n):
            p1_name, p2_name = seats[i]
            a1 = actions[(i, p1_name)]
            a2 = actions[(i, p2_name)]
            nxt, _, _, _, term, trunc, _ = envs[i].step_both(a1, a2)
            steps[i] += 1
            done = bool(term or trunc) or steps[i] >= args.max_steps
            if not done:
                cur_p1[i] = nxt
                continue

            battle = envs[i].battle
            c1 = int(battle.crowns[0]) if battle else 0
            c2 = int(battle.crowns[1]) if battle else 0
            winner = p1_name if c1 > c2 else (p2_name if c2 > c1 else "DRAW")
            results.append({
                "game": game_id[i],
                "winner": winner,
                "p1": p1_name, "p2": p2_name,
                "crowns_p1": c1, "crowns_p2": c2,
                "steps": steps[i],
                "capped": steps[i] >= args.max_steps,
            })
            if len(results) >= args.matches:
                break
            cur_p1[i] = envs[i].reset(seed=next_game)[0]
            game_id[i] = next_game
            steps[i] = 0
            next_game += 1

        if len(results) - last_reported >= 10:
            last_reported = len(results)
            print(f"  {len(results)}/{args.matches} games  ({time.time() - t0:.0f}s)", flush=True)

    a_wins = sum(r["winner"] == "A" for r in results)
    b_wins = sum(r["winner"] == "B" for r in results)
    draws = sum(r["winner"] == "DRAW" for r in results)
    capped = sum(r["capped"] for r in results)
    decisive = a_wins + b_wins
    low, high = wilson_interval(a_wins, decisive)
    a_as_p1 = [r for r in results if r["p1"] == "A"]
    a_as_p2 = [r for r in results if r["p2"] == "A"]

    print("\n" + "=" * 68)
    print(f"  A = {names['A']}")
    print(f"  B = {names['B']}")
    print("=" * 68)
    print(f"  A wins: {a_wins}   B wins: {b_wins}   draws: {draws}"
          + (f"   (cap-hit games: {capped})" if capped else ""))
    print(f"  A decisive win rate: {a_wins / max(1, decisive):.3f} "
          f"(Wilson 95% CI [{low:.3f}, {high:.3f}])")
    print(f"  A as P1: {sum(r['winner']=='A' for r in a_as_p1)}/{len(a_as_p1)}  |  "
          f"A as P2: {sum(r['winner']=='A' for r in a_as_p2)}/{len(a_as_p2)}")
    print(f"  elapsed {time.time() - t0:.0f}s")
    print("=" * 68)

    if args.json:
        payload = {
            "challenger": args.challenger, "baseline": args.baseline,
            "matches": len(results), "a_wins": a_wins, "b_wins": b_wins,
            "draws": draws, "capped": capped, "a_win_rate": a_wins / max(1, decisive),
            "ci95": [low, high],
            "a_as_p1": {"wins": sum(r["winner"]=="A" for r in a_as_p1), "games": len(a_as_p1)},
            "a_as_p2": {"wins": sum(r["winner"]=="A" for r in a_as_p2), "games": len(a_as_p2)},
            "results": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
