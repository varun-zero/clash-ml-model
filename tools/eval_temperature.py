"""Benchmark a trained checkpoint with low-temperature exploitation.

Samples the policy at temperature ``T`` (both heads) against a rule-based
opponent and reports the *exploitation* win rate - the win rate the agent
would achieve playing its best moves instead of exploring.

Usage:
    python tools/eval_temperature.py --checkpoint checkpoints/current_deck/latest.pt \
        --temperature 0.35 --matches 50 --opponent coach
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cr_alphastar_agent.config import (  # noqa: E402
    ARENA_HEIGHT, ARENA_WIDTH, SPELL_CARD_IDS, resolve_deck_names, get_hardware_profile,
)
from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv  # noqa: E402
from cr_alphastar_agent.env.scripted_bot import ScriptedBot, TacticalProBot  # noqa: E402
from cr_alphastar_agent.env.tactical_coach import TacticalCoachBot  # noqa: E402
from cr_alphastar_agent.env.tactics import perspective, touches_enemy_king  # noqa: E402
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet  # noqa: E402

OPPONENTS = {
    "coach": TacticalCoachBot,
    "tactical": TacticalProBot,
    "scripted": ScriptedBot,
}


def load_model(path: str, device: torch.device,
               top_p: float = 0.85) -> AlphaStarCRNet:
    model = AlphaStarCRNet(top_p=top_p).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    return model


def to_tensors(obs: dict, device: torch.device) -> dict:
    return {k: torch.from_numpy(np.asarray(v)).unsqueeze(0).to(device) for k, v in obs.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Low-temperature exploitation benchmark")
    ap.add_argument("--checkpoint", default="checkpoints/current_deck/latest.pt")
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--matches", type=int, default=50)
    ap.add_argument("--opponent", default="coach", choices=sorted(OPPONENTS))
    ap.add_argument("--deck", default="current")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    ap.add_argument("--greedy", action="store_true",
                    help="Use argmax instead of temperature sampling")
    ap.add_argument("--top-p", type=float, default=0.85,
                    help="Nucleus sampling threshold for the tile head (0 disables it)")
    args = ap.parse_args()

    hw = get_hardware_profile()
    device = torch.device(args.device or hw["device"])
    model = load_model(args.checkpoint, device, top_p=args.top_p)
    deck = resolve_deck_names(args.deck)
    opp_cls = OPPONENTS[args.opponent]

    persp = perspective(0)
    wins = losses = draws = 0
    crowns = [0, 0]
    deploys = 0
    spells = spells_own_half = spells_king = 0
    distinct_tiles = set()
    per_match_tiles = []
    wait_actions = 0

    for match in range(args.matches):
        env = CrSimReplicaEnv(seed=args.seed * 1000 + match,
                              blue_deck=deck, red_deck=deck)
        opp = opp_cls(player_id=2, seed=args.seed + match)
        obs, _ = env.reset()
        match_tiles = set()
        for _ in range(500):
            obs2 = env.get_obs(2)
            with torch.no_grad():
                tensors = to_tensors(obs, device)
                if args.greedy:
                    action, _, _, _ = model.get_action_and_value(tensors, deterministic=True)
                else:
                    action, _, _, _ = model.get_action_and_value(
                        tensors, temperature=args.temperature)
            slot = int(action[0, 0].item())
            tile = int(action[0, 1].item())

            if slot != 4:
                deploys += 1
                distinct_tiles.add(tile)
                match_tiles.add(tile)
                cid = int(obs["hand_cards"][slot])
                tx = float(tile % ARENA_WIDTH)
                ty = float(tile // ARENA_WIDTH)
                if cid in SPELL_CARD_IDS:
                    spells += 1
                    if persp.is_own_half(ty):
                        spells_own_half += 1
                    if touches_enemy_king(tx, ty, 0):
                        spells_king += 1
            else:
                wait_actions += 1

            opp_action = opp.act(obs2)
            obs, _, _, _, term, trunc, _ = env.step_both(
                action[0].cpu().numpy(), opp_action)
            if term or trunc:
                break

        battle = env.battle
        per_match_tiles.append(len(match_tiles))
        crowns[0] += int(battle.crowns[0])
        crowns[1] += int(battle.crowns[1])
        if battle.result == 0:
            wins += 1
        elif battle.result == 1:
            losses += 1
        else:
            draws += 1

    total = max(1, wins + losses + draws)
    print("=" * 74)
    print(f"  EXPLOITATION BENCHMARK  (T = {args.temperature}"
          f"{', greedy' if args.greedy else ''})")
    print("=" * 74)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  opponent   : {opp_cls.__name__}")
    print(f"  deck       : {args.deck}")
    print(f"  top-p      : {args.top_p if args.top_p > 0 else 'disabled'}")
    print(f"  matches    : {total}")
    print(f"  record     : {wins}W - {losses}L - {draws}D")
    print(f"  win rate   : {100.0 * (wins + 0.5 * draws) / total:.1f}%")
    print(f"  crowns     : {crowns[0]} - {crowns[1]}")
    print(f"  deploys/match   : {deploys / total:.1f}")
    print(f"  waits/match     : {wait_actions / total:.1f}")
    print(f"  spells/match    : {spells / total:.1f}")
    print(f"  spells own half : {spells_own_half}")
    print(f"  spells at king  : {spells_king}")
    print(f"  distinct tiles  : {len(distinct_tiles)} overall, "
          f"{sum(per_match_tiles) / max(1, len(per_match_tiles)):.1f} per match")
    print("=" * 74)


if __name__ == "__main__":
    main()
