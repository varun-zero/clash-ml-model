"""
Headless Win-Rate Benchmark and Visual Evaluation for AlphaStar CR Agent.
Evaluates the trained AlphaStarCRNet policy against ScriptedBot over N matches.

Usage:
    python evaluate_visual.py --eval-matches 50 [--checkpoint checkpoints/latest.pt] [--render]
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

from cr_alphastar_agent.config import (
    ARENA_WIDTH, ARENA_HEIGHT, TOTAL_TILES, CARDS_CATALOG, get_hardware_profile
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.env.replica_env import ClashRoyaleReplicaEnv
from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv
from cr_alphastar_agent.env.scripted_bot import ScriptedBot, TacticalProBot


def classify_tactical_zone(x: float, y: float) -> str:
    """
    Classifies a card deployment coordinate (x, y) in the canonical friendly frame
    into one of the 5 key tactical deployment zones.
    """
    if y <= 4.0:
        return "Back-of-King Slow-Roll"
    elif 7.5 <= x <= 10.5 and 4.0 < y <= 8.0:
        return "Center Split"
    elif 7.0 <= x <= 11.0 and 8.0 < y <= 13.0:
        return "Center Kill-Zone / Kite"
    elif 13.0 < y <= 15.5:
        return "Bridge / Tank-Trick"
    elif y > 16.0:
        return "Deep Enemy Spell"
    else:
        return "Flank Lane Defense"


def evaluate(checkpoint_path: str, eval_matches: int = 50, device_str: str = None,
             render: bool = False, backend: str = "crsim", opponent_type: str = "scripted"):
    if device_str is None:
        hw = get_hardware_profile()
        device = torch.device(hw["device"])
    else:
        device = torch.device(device_str)

    opp_title = "TacticalProBot" if opponent_type == "tactical" else "ScriptedBot"
    print("=" * 75)
    print(f"      AlphaStar CR Agent vs {opp_title} Benchmark Evaluation")
    print("=" * 75)
    print(f"[Device]           {device}")
    print(f"[Backend]          {backend.upper()}")
    print(f"[Opponent]         {opp_title}")
    print(f"[Checkpoint]       {os.path.abspath(checkpoint_path)}")
    print(f"[Matches to Play]  {eval_matches}\n")

    # Load Model
    model = AlphaStarCRNet().to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        global_step = ckpt.get("global_step", 0)
        update_count = ckpt.get("update_count", 0)
        print(f"[Loaded Checkpoint] Step {global_step} | Update {update_count}")
    else:
        model.load_state_dict(ckpt)
        print("[Loaded Raw Model Weights]")

    model.eval()

    # Opponent Bot
    if opponent_type == "tactical":
        opponent = TacticalProBot(player_id=2)
    else:
        opponent = ScriptedBot(player_id=2)

    wins = 0
    losses = 0
    draws = 0
    agent_crowns_total = 0
    opp_crowns_total = 0
    match_durations = []
    card_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    # 5 Key Tactical Zones + Flank Defense
    tactical_zones = {
        "Back-of-King Slow-Roll": 0,
        "Center Split": 0,
        "Center Kill-Zone / Kite": 0,
        "Bridge / Tank-Trick": 0,
        "Deep Enemy Spell": 0,
        "Flank Lane Defense": 0,
    }

    t_start = time.time()

    for match_idx in range(1, eval_matches + 1):
        if backend == "crsim":
            env = CrSimReplicaEnv(seed=1000 + match_idx)
        else:
            env = ClashRoyaleReplicaEnv(seed=1000 + match_idx)
        obs_p1 = env.get_obs(player_id=1)
        obs_p2 = env.get_obs(player_id=2)

        done = False
        step_count = 0
        p1_crowns = 0
        p2_crowns = 0

        while not done:
            step_count += 1

            # Prepare tensor for Player 1 (Learned Agent)
            with torch.no_grad():
                obs_tensors = {
                    "scalars": torch.as_tensor(obs_p1["scalars"], dtype=torch.float32, device=device).unsqueeze(0),
                    "hand_cards": torch.as_tensor(obs_p1["hand_cards"], dtype=torch.long, device=device).unsqueeze(0),
                    "entities": torch.as_tensor(obs_p1["entities"], dtype=torch.float32, device=device).unsqueeze(0),
                    "entity_mask": torch.as_tensor(obs_p1["entity_mask"], dtype=torch.bool, device=device).unsqueeze(0),
                    "arena_grid": torch.as_tensor(obs_p1["arena_grid"], dtype=torch.float32, device=device).unsqueeze(0),
                    "card_mask": torch.as_tensor(obs_p1["card_mask"], dtype=torch.bool, device=device).unsqueeze(0),
                    "tile_mask": torch.as_tensor(obs_p1["tile_mask"], dtype=torch.bool, device=device).unsqueeze(0),
                }
                actions, _, _, _ = model.get_action_and_value(obs_tensors)
                p1_action = actions[0].cpu().numpy()
                slot = int(p1_action[0])
                card_counts[slot] += 1

                # Track Tactical Zone Deployment if an active card was deployed
                if 0 <= slot <= 3:
                    tile_idx = int(p1_action[1])
                    tx = float(tile_idx % ARENA_WIDTH)
                    ty = float(tile_idx // ARENA_WIDTH)
                    zone = classify_tactical_zone(tx, ty)
                    tactical_zones[zone] += 1

            # Player 2
            p2_action = opponent.act(obs_p2)

            # Step environment
            step_ret = env.step_both(p1_action, p2_action)
            obs_p1, obs_p2, r1, r2, term1, term2, info = step_ret
            done = term1 or term2

        # Match concluded
        if hasattr(env, "battle") and env.battle is not None:
            b = env.battle
            p1_crowns = b.crowns[0]
            p2_crowns = b.crowns[1]
            if b.result == 0:
                result = "WIN  [Agent]"
                wins += 1
            elif b.result == 1:
                result = "LOSS [Bot]  "
                losses += 1
            else:
                result = "DRAW        "
                draws += 1
        else:
            p1_crowns = info.get("p1_crowns", 0)
            p2_crowns = info.get("p2_crowns", 0)
            if p1_crowns > p2_crowns:
                result = "WIN  [Agent]"
                wins += 1
            elif p2_crowns > p1_crowns:
                result = "LOSS [Bot]  "
                losses += 1
            else:
                p1_min = min((t.hp for t in env.towers[1].values() if t.alive), default=0.0)
                p2_min = min((t.hp for t in env.towers[2].values() if t.alive), default=0.0)
                if p1_min > p2_min:
                    result = "WIN* [Agent]"
                    wins += 1
                elif p2_min > p1_min:
                    result = "LOSS*[Bot]  "
                    losses += 1
                else:
                    result = "DRAW        "
                    draws += 1

        agent_crowns_total += p1_crowns
        opp_crowns_total += p2_crowns
        duration_s = (env.battle.t / 1000.0) if hasattr(env, "battle") else (step_count * env.action_repeat * 0.1)
        match_durations.append(duration_s)

        curr_win_rate = (wins / match_idx) * 100.0
        print(f"Match {match_idx:02d}/{eval_matches:02d} | Result: {result} | Crowns: {p1_crowns}-{p2_crowns} | Steps: {step_count:03d} ({match_durations[-1]:.1f}s) | WinRate: {curr_win_rate:.1f}%")

    total_time = time.time() - t_start
    decisive_matches = wins + losses
    decisive_win_rate = (wins / decisive_matches * 100.0) if decisive_matches > 0 else 0.0
    overall_win_rate = (wins / eval_matches) * 100.0

    print("\n" + "=" * 75)
    print("                         BENCHMARK RESULTS")
    print("=" * 75)
    print(f"Total Matches:            {eval_matches}")
    print(f"Agent Wins:               {wins} ({overall_win_rate:.1f}%)")
    print(f"{opp_title} Wins:         {losses} ({(losses / eval_matches) * 100.0:.1f}%)")
    print(f"Draws:                    {draws} ({(draws / eval_matches) * 100.0:.1f}%)")
    print(f"Decisive Win Rate:        {decisive_win_rate:.1f}% (Wins / Decisive)")
    print(f"Average Crowns (Agent):   {agent_crowns_total / eval_matches:.2f}")
    print(f"Average Crowns (Bot):     {opp_crowns_total / eval_matches:.2f}")
    print(f"Average Match Duration:   {np.mean(match_durations):.1f}s")
    print(f"Total Evaluation Time:    {total_time:.2f}s ({eval_matches / max(1e-4, total_time):.1f} matches/s)")
    print("-" * 75)
    total_actions = sum(card_counts.values()) or 1
    print("Card Slot Distribution:")
    for slot in range(4):
        print(f"  Slot {slot}: {card_counts[slot]} plays ({card_counts[slot] / total_actions * 100:.1f}%)")
    print(f"  Wait (Slot 4): {card_counts[4]} ticks ({card_counts[4] / total_actions * 100:.1f}%)")

    # Tactical Placement Zone Telemetry Breakdown
    total_deployments = sum(tactical_zones.values()) or 1
    print("-" * 75)
    print("🎯 Tactical Placement Zone Telemetry (Spatial Deployment Breakdown):")
    for zone, count in tactical_zones.items():
        pct = (count / total_deployments) * 100.0
        bar = "█" * int(round(pct / 4))
        print(f"  • {zone:<25}: {count:4d} cards ({pct:5.1f}%)  {bar}")
    print("=" * 75)

    # Sync to Obsidian vault
    try:
        from cr_alphastar_agent.training.obsidian_sync import sync_tactical_zones_to_obsidian
        sync_tactical_zones_to_obsidian(
            tactical_zones=tactical_zones,
            total_matches=eval_matches,
            win_rate=overall_win_rate,
            opp_name=opp_title
        )
    except Exception:
        pass

    if overall_win_rate >= 65.0:
        print("[SUCCESS] Agent achieved target metric (>65% win rate)! Ready for Colab scaling.")
    elif decisive_win_rate >= 65.0:
        print(f"[TARGET ACHIEVED] Decisive Win Rate is {decisive_win_rate:.1f}% (>65%) with {draws} tactical draws.")
    else:
        print(f"[NOTE] Win rate is {overall_win_rate:.1f}%. Continue training updates for further dominance.")

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": overall_win_rate,
        "decisive_win_rate": decisive_win_rate,
        "tactical_zones": tactical_zones
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaStar CR Agent Evaluator")
    parser.add_argument("--eval-matches", type=int, default=50, help="Number of evaluation matches")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Path to checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to evaluate on (cuda/cpu)")
    parser.add_argument("--backend", type=str, default="crsim", choices=["crsim", "replica"],
                        help="Physics engine backend (crsim or replica)")
    parser.add_argument("--opponent", type=str, default="scripted", choices=["scripted", "tactical"],
                        help="Opponent bot policy (scripted or tactical)")
    parser.add_argument("--render", action="store_true", help="Print visual match progress")
    args = parser.parse_args()

    evaluate(checkpoint_path=args.checkpoint, eval_matches=args.eval_matches,
             device_str=args.device, render=args.render, backend=args.backend,
             opponent_type=args.opponent)
