"""
AlphaStar League Tournament Runner for Clash Royale.
Conducts round-robin or Swiss tournaments across historical model checkpoints,
pool snapshots, and baseline bots (ScriptedBot, RandomBot) over full-fidelity crsim physics.

Computes Elo ratings, crown differentials, and generates a head-to-head win-rate cross-table.

Usage:
    python tools/league_tournament.py [--games-per-pair 2] [--max-candidates 6] [--device auto]
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
from typing import Dict, List, Any, Tuple

# Dynamic parent path resolution
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crsim.engine import load_model, Battle
from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv, CRSIM_NAME_TO_ID
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.env.scripted_bot import ScriptedBot
from cr_alphastar_agent.env.rush_exploiter_bot import RushExploiterBot
from tools.ui import POPULAR_DECKS


class RandomBot:
    """Baseline bot that plays random legal moves when elixir is available."""
    def __init__(self, player_id: int = 2):
        self.player_id = player_id

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        card_mask = obs["card_mask"]
        tile_mask = obs["tile_mask"]
        valid_cards = np.where(card_mask)[0]
        if len(valid_cards) == 0:
            return np.array([4, 0], dtype=np.int64)

        chosen_card = int(np.random.choice(valid_cards))
        if chosen_card == 4:
            return np.array([4, 0], dtype=np.int64)

        valid_tiles = np.where(tile_mask)[0]
        if len(valid_tiles) == 0:
            return np.array([4, 0], dtype=np.int64)

        chosen_tile = int(np.random.choice(valid_tiles))
        return np.array([chosen_card, chosen_tile], dtype=np.int64)


class PlayerCandidate:
    def __init__(self, name: str, kind: str, path: str = None, device: torch.device = None):
        self.name = name
        self.kind = kind  # "model", "scripted", "random"
        self.path = path
        self.device = device
        self.net = None
        self.elo = 1200.0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.crowns_for = 0
        self.crowns_against = 0

        if self.kind == "model" and path and os.path.exists(path):
            self.net = AlphaStarCRNet().to(device)
            ckpt = torch.load(path, map_location=device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.net.load_state_dict(state_dict)
            self.net.eval()

    def get_action(self, obs: Dict[str, np.ndarray], player_id: int) -> np.ndarray:
        if self.kind == "model":
            tensor_obs = {
                k: torch.from_numpy(v).unsqueeze(0).to(self.device)
                for k, v in obs.items()
            }
            with torch.no_grad():
                action, _, _, _ = self.net.get_action_and_value(tensor_obs, deterministic=True)
            return action.squeeze(0).cpu().numpy()
        elif self.kind == "scripted":
            bot = ScriptedBot(player_id=player_id)
            return bot.act(obs)
        elif self.kind == "rush":
            bot = RushExploiterBot(player_id=player_id)
            return bot.act(obs)
        elif self.kind == "random":
            bot = RandomBot(player_id=player_id)
            return bot.act(obs)
        return np.array([4, 0], dtype=np.int64)


def play_match(
    p1: PlayerCandidate,
    p2: PlayerCandidate,
    deck1_ids: List[int],
    deck2_ids: List[int],
    seed: int = 42,
    max_steps: int = 1200
) -> Tuple[int, int, int]:
    """
    Simulates a match between p1 (Player 1 / Blue) and p2 (Player 2 / Red).
    Returns (winner_id, crowns_p1, crowns_p2) where winner_id is 1, 2, or 0 (draw).
    """
    env = CrSimReplicaEnv(blue_deck=deck1_ids, red_deck=deck2_ids, seed=seed)
    obs_p1, _ = env.reset()

    for _ in range(max_steps):
        obs_p2 = env.get_obs(player_id=2)

        act1 = p1.get_action(obs_p1, player_id=1)
        act2 = p2.get_action(obs_p2, player_id=2)

        obs_p1, obs_p2, r1, r2, term, trunc, info = env.step_both(act1, act2)
        if term or trunc:
            break

    b = env.battle
    c1 = b.crowns[0] if b else 0
    c2 = b.crowns[1] if b else 0
    winner = 0
    if b:
        if b.result == 0:
            winner = 1
        elif b.result == 1:
            winner = 2
    return winner, c1, c2


def update_elo(p_a: PlayerCandidate, p_b: PlayerCandidate, score_a: float, k: float = 32.0):
    """
    Standard Elo rating update:
    score_a = 1.0 (A wins), 0.5 (draw), 0.0 (B wins)
    """
    exp_a = 1.0 / (1.0 + 10.0 ** ((p_b.elo - p_a.elo) / 400.0))
    exp_b = 1.0 - exp_a

    score_b = 1.0 - score_a

    p_a.elo += k * (score_a - exp_a)
    p_b.elo += k * (score_b - exp_b)


def run_tournament(
    games_per_pair: int = 2,
    max_candidates: int = 6,
    device_name: str = "auto"
):
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 80)
    print("           ALPHASTAR CLASH ROYALE LEAGUE TOURNAMENT")
    print("=" * 80)
    print(f"[Device] {device}")
    print(f"[Games Per Pair] {games_per_pair * 2} matches (Alternating Blue/Red)")

    # 1. Discover candidates
    candidates: List[PlayerCandidate] = []

    # Check latest
    latest_path = os.path.join(ROOT, "checkpoints", "latest.pt")
    if os.path.exists(latest_path):
        candidates.append(PlayerCandidate(name="AlphaStar (latest)", kind="model", path=latest_path, device=device))

    # Pool snapshots (sample evenly spaced milestones up to max_candidates - 4)
    pool_files = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "pool", "*.pt")))
    max_pool_slots = max(1, max_candidates - 4)
    if len(pool_files) > max_pool_slots:
        indices = np.linspace(0, len(pool_files) - 1, max_pool_slots, dtype=int)
        selected_pools = [pool_files[idx] for idx in sorted(set(indices))]
    else:
        selected_pools = pool_files

    for pf in selected_pools:
        name = "Pool:" + os.path.basename(pf).replace(".pt", "").replace("snapshot_", "s_")
        candidates.append(PlayerCandidate(name=name, kind="model", path=pf, device=device))

    # Baselines (Always included)
    candidates.append(PlayerCandidate(name="ScriptedBot", kind="scripted"))
    candidates.append(PlayerCandidate(name="RushExploiter", kind="rush"))
    candidates.append(PlayerCandidate(name="RandomBot", kind="random"))

    num_p = len(candidates)
    print(f"[Participants ({num_p})]")
    for idx, c in enumerate(candidates):
        print(f"  {idx+1}. {c.name:24s} [{c.kind}]")
    print()

    # Pre-select decks
    deck_blue_cards = POPULAR_DECKS["Hog 2.6 Cycle"]
    deck_red_cards = POPULAR_DECKS["Giant Beatdown"]
    deck1_ids = [CRSIM_NAME_TO_ID.get(c, 1) for c in deck_blue_cards]
    deck2_ids = [CRSIM_NAME_TO_ID.get(c, 1) for c in deck_red_cards]

    # Head-to-Head win matrix: matrix[i][j] = (wins_i, draws, wins_j)
    h2h_matrix = {i: {j: [0, 0, 0] for j in range(num_p)} for i in range(num_p)}

    total_matches = num_p * (num_p - 1) * games_per_pair
    match_counter = 0
    t0 = time.time()

    # Round Robin Tournament
    for i in range(num_p):
        for j in range(i + 1, num_p):
            pA = candidates[i]
            pB = candidates[j]

            for g in range(games_per_pair):
                match_counter += 1
                # Sub-match 1: pA is Player 1 (Blue), pB is Player 2 (Red)
                winner, c1, c2 = play_match(pA, pB, deck1_ids, deck2_ids, seed=1000 + match_counter)
                pA.crowns_for += c1
                pA.crowns_against += c2
                pB.crowns_for += c2
                pB.crowns_against += c1

                if winner == 1:
                    pA.wins += 1
                    pB.losses += 1
                    update_elo(pA, pB, 1.0)
                    h2h_matrix[i][j][0] += 1
                    h2h_matrix[j][i][2] += 1
                elif winner == 2:
                    pB.wins += 1
                    pA.losses += 1
                    update_elo(pA, pB, 0.0)
                    h2h_matrix[i][j][2] += 1
                    h2h_matrix[j][i][0] += 1
                else:
                    pA.draws += 1
                    pB.draws += 1
                    update_elo(pA, pB, 0.5)
                    h2h_matrix[i][j][1] += 1
                    h2h_matrix[j][i][1] += 1

                # Sub-match 2: Canonical symmetry swap (pB is Blue, pA is Red)
                match_counter += 1
                winner, c1, c2 = play_match(pB, pA, deck1_ids, deck2_ids, seed=5000 + match_counter)
                pB.crowns_for += c1
                pB.crowns_against += c2
                pA.crowns_for += c2
                pA.crowns_against += c1

                if winner == 1:
                    pB.wins += 1
                    pA.losses += 1
                    update_elo(pB, pA, 1.0)
                    h2h_matrix[j][i][0] += 1
                    h2h_matrix[i][j][2] += 1
                elif winner == 2:
                    pA.wins += 1
                    pB.losses += 1
                    update_elo(pB, pA, 0.0)
                    h2h_matrix[j][i][2] += 1
                    h2h_matrix[i][j][0] += 1
                else:
                    pA.draws += 1
                    pB.draws += 1
                    update_elo(pB, pA, 0.5)
                    h2h_matrix[j][i][1] += 1
                    h2h_matrix[i][j][1] += 1

                print(f"[{match_counter:2d}/{total_matches}] {pA.name:20s} vs {pB.name:20s} -> Elo: {pA.name[:10]} {pA.elo:.0f} | {pB.name[:10]} {pB.elo:.0f}")

    elapsed = time.time() - t0

    # Sort participants by Elo
    ranked = sorted(candidates, key=lambda c: c.elo, reverse=True)

    print("\n" + "=" * 80)
    print("                     FINAL LEAGUE STANDINGS & ELO")
    print("=" * 80)
    print(f"{'Rank':<5} {'Participant':<24} {'Elo':<7} {'W-L-D':<10} {'Win %':<8} {'Crown Diff':<10}")
    print("-" * 80)

    for rank, p in enumerate(ranked, 1):
        total_g = p.wins + p.losses + p.draws
        win_pct = (p.wins / total_g * 100.0) if total_g > 0 else 0.0
        c_diff = p.crowns_for - p.crowns_against
        diff_str = f"+{c_diff}" if c_diff > 0 else str(c_diff)
        wld_str = f"{p.wins}-{p.losses}-{p.draws}"
        print(f"{rank:<5} {p.name:<24} {p.elo:<7.1f} {wld_str:<10} {win_pct:<7.1f}% {diff_str:<10}")

    print("=" * 80)
    print(f"Completed {match_counter} tournament matches in {elapsed:.1f}s ({match_counter/elapsed:.1f} matches/s)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaStar League Tournament")
    parser.add_argument("--games-per-pair", type=int, default=1, help="Pairs of symmetric matches per matchup")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum candidates to include")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    args = parser.parse_args()

    run_tournament(
        games_per_pair=args.games_per_pair,
        max_candidates=args.max_candidates,
        device_name=args.device
    )
