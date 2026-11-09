"""
Quantitative Evaluation & Benchmark Suite for AlphaStar Clash Royale Agent.
Pits the trained policy (checkpoints/latest.pt) against heuristic bots, random agents,
or historical league snapshots across full-fidelity crsim matches.

Usage:
    python eval_agent.py [--checkpoint checkpoints/latest.pt] [--games 20] [--opponent scripted]
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

from crsim.engine import load_model, Battle
from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv, CRSIM_NAME_TO_ID
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.env.scripted_bot import ScriptedBot
from cr_alphastar_agent.env.rush_exploiter_bot import RushExploiterBot
from tools.ui import POPULAR_DECKS


def run_evaluation(
    checkpoint_path: str = "checkpoints/latest.pt",
    num_games: int = 20,
    opponent_type: str = "scripted",
    deck_agent_name: str = "Hog 2.6 Cycle",
    deck_opp_name: str = "Giant Beatdown",
    device_name: str = "auto"
):
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 75)
    print("      AlphaStar Clash Royale Agent - Quantitative Evaluation Suite")
    print("=" * 75)
    print(f"[Model Checkpoint] {checkpoint_path}")
    print(f"[Device] {device}")
    print(f"[Opponent Baseline] {opponent_type.upper()}")
    print(f"[Matchup] Agent: {deck_agent_name} vs Opponent: {deck_opp_name}")
    print(f"[Sample Size] {num_games} matches (Alternating sides for fairness)\n")

    # Load agent
    agent_net = AlphaStarCRNet().to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        agent_net.load_state_dict(ckpt["model_state_dict"])
        step = ckpt.get("global_step", "N/A")
        update = ckpt.get("update_count", "N/A")
        print(f"[Agent Loaded] Trained for {step} steps ({update} PPO updates)")
    else:
        print(f"[Warning] Checkpoint '{checkpoint_path}' not found! Evaluating random policy weights.")

    agent_net.eval()

    # Load opponent if opponent is another checkpoint
    opp_net = None
    if opponent_type.endswith(".pt") and os.path.exists(opponent_type):
        opp_net = AlphaStarCRNet().to(device)
        opp_ckpt = torch.load(opponent_type, map_location=device)
        opp_net.load_state_dict(opp_ckpt["model_state_dict"])
        opp_net.eval()
        print(f"[Opponent Loaded] Frozen snapshot from {opponent_type}")

    # Metrics
    results = {
        "agent_wins": 0,
        "opp_wins": 0,
        "draws": 0,
        "agent_crowns": 0,
        "opp_crowns": 0,
        "durations_sec": [],
        "agent_tower_hp": [],
        "opp_tower_hp": []
    }

    b_ids = [CRSIM_NAME_TO_ID.get(c, 1) for c in POPULAR_DECKS.get(deck_agent_name, POPULAR_DECKS["Hog 2.6 Cycle"])]
    r_ids = [CRSIM_NAME_TO_ID.get(c, 1) for c in POPULAR_DECKS.get(deck_opp_name, POPULAR_DECKS["Giant Beatdown"])]

    t0_all = time.time()

    for g in range(num_games):
        # Alternate player sides: even games agent=Player 1 (blue), odd games agent=Player 2 (red)
        agent_is_p1 = (g % 2 == 0)
        agent_pid = 1 if agent_is_p1 else 2
        opp_pid = 2 if agent_is_p1 else 1

        deck1 = b_ids if agent_is_p1 else r_ids
        deck2 = r_ids if agent_is_p1 else b_ids

        env = CrSimReplicaEnv(blue_deck=deck1, red_deck=deck2, seed=1000 + g)
        scripted_bot = ScriptedBot(player_id=opp_pid)
        rush_bot = RushExploiterBot(player_id=opp_pid)

        obs_p1, _ = env.reset()

        step_count = 0
        max_steps = 1200  # Up to 8 minutes match limit

        while not env.battle.done and step_count < max_steps:
            obs_p2 = env.get_obs(2)

            # Determine Agent Action
            agent_obs = obs_p1 if agent_is_p1 else obs_p2
            batch_agent_obs = {
                k: torch.from_numpy(v).unsqueeze(0).to(device)
                for k, v in agent_obs.items()
            }
            with torch.no_grad():
                agent_action, _, _, _ = agent_net.get_action_and_value(batch_agent_obs)
            agent_act = agent_action[0].cpu().numpy()

            # Determine Opponent Action
            opp_obs = obs_p2 if agent_is_p1 else obs_p1
            if opp_net is not None:
                batch_opp_obs = {
                    k: torch.from_numpy(v).unsqueeze(0).to(device)
                    for k, v in opp_obs.items()
                }
                with torch.no_grad():
                    opp_action, _, _, _ = opp_net.get_action_and_value(batch_opp_obs)
                opp_act = opp_action[0].cpu().numpy()
            elif opponent_type == "random":
                c_mask = opp_obs["card_mask"]
                valid_cards = np.where(c_mask)[0]
                c_slot = np.random.choice(valid_cards) if len(valid_cards) > 0 else 4
                t_mask = opp_obs["tile_mask"]
                valid_tiles = np.where(t_mask)[0]
                t_idx = np.random.choice(valid_tiles) if len(valid_tiles) > 0 else 0
                opp_act = np.array([c_slot, t_idx])
            elif opponent_type == "rush":
                opp_act = rush_bot.act(opp_obs)
            else:
                # Scripted heuristic bot
                opp_act = scripted_bot.act(opp_obs)

            # Assemble action pair
            p1_act = agent_act if agent_is_p1 else opp_act
            p2_act = opp_act if agent_is_p1 else agent_act

            next_obs_p1, _, _, _, term, trunc, _ = env.step_both(p1_act, p2_act)
            obs_p1 = next_obs_p1
            step_count += 1
            if term or trunc:
                break

        # Record match results
        b = env.battle
        crowns_p1 = b.crowns[0]
        crowns_p2 = b.crowns[1]
        c_agent = crowns_p1 if agent_is_p1 else crowns_p2
        c_opp = crowns_p2 if agent_is_p1 else crowns_p1

        results["agent_crowns"] += c_agent
        results["opp_crowns"] += c_opp
        duration_sec = b.t / 1000.0
        results["durations_sec"].append(duration_sec)

        if b.result is None:
            results["draws"] += 1
            outcome = "DRAW"
        elif (b.result == 0 and agent_is_p1) or (b.result == 1 and not agent_is_p1):
            results["agent_wins"] += 1
            outcome = "AGENT WIN"
        else:
            results["opp_wins"] += 1
            outcome = "OPPONENT WIN"

        side_str = "Blue (P1)" if agent_is_p1 else "Red (P2)"
        print(f"Game {g+1:02d}/{num_games:02d} | Agent Side: {side_str} | Result: {outcome:12s} | Crowns: {c_agent}-{c_opp} | Time: {duration_sec:.1f}s")

    elapsed_all = time.time() - t0_all
    win_rate = (results["agent_wins"] / num_games) * 100.0
    loss_rate = (results["opp_wins"] / num_games) * 100.0
    draw_rate = (results["draws"] / num_games) * 100.0
    avg_dur = np.mean(results["durations_sec"]) if results["durations_sec"] else 0.0

    print("\n" + "=" * 75)
    print("                           EVALUATION REPORT")
    print("=" * 75)
    print(f"Total Games Played:        {num_games}")
    print(f"Agent Win Rate:            {win_rate:.1f}% ({results['agent_wins']} Wins)")
    print(f"Opponent Win Rate:         {loss_rate:.1f}% ({results['opp_wins']} Losses)")
    print(f"Draw Rate:                 {draw_rate:.1f}% ({results['draws']} Draws)")
    print(f"Total Crowns:              Agent {results['agent_crowns']} - {results['opp_crowns']} Opponent")
    print(f"Average Crown Advantage:   {(results['agent_crowns'] - results['opp_crowns']) / num_games:+.2f} per game")
    print(f"Average Match Duration:    {avg_dur:.1f} seconds")
    print(f"Total Evaluation Time:     {elapsed_all:.1f} seconds")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AlphaStar Clash Royale Agent Evaluator")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--games", type=int, default=10,
                        help="Number of evaluation games (default: 10)")
    parser.add_argument("--opponent", type=str, default="scripted",
                        help="Opponent baseline: 'scripted', 'random', or path to snapshot .pt")
    parser.add_argument("--deck-agent", type=str, default="Hog 2.6 Cycle",
                        help="Deck for agent (Hog 2.6 Cycle, Giant Beatdown, Spell Bait, Bridge Spam)")
    parser.add_argument("--deck-opp", type=str, default="Giant Beatdown",
                        help="Deck for opponent")
    parser.add_argument("--device", type=str, default="auto",
                        help="Evaluation device ('auto', 'cuda', 'cpu')")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        num_games=args.games,
        opponent_type=args.opponent,
        deck_agent_name=args.deck_agent,
        deck_opp_name=args.deck_opp,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
