"""
OpenAI Five PPO + Generalized Advantage Estimation (GAE) Trainer with Hybrid Colab T4 / Local RTX 3050 Handoff.
Features:
- Micro-batching + Gradient Accumulation preserves identical Effective Batch Size of 256 on both T4 and RTX 3050.
- FP16 Mixed Precision via torch.amp with numerical safeguards.
- Seamless checkpoint saving and resumption via checkpoints/latest.pt.
"""

import os
import glob
import time
import copy
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from cr_alphastar_agent.config import (
    get_hardware_profile, PPO_LR, GAMMA, GAE_LAMBDA, CLIP_COEF,
    ENT_COEF, VF_COEF, MAX_GRAD_NORM, PPO_EPOCHS, ROLLOUT_STEPS,
    NUM_ENVS, EFFECTIVE_BATCH_SIZE, SNAPSHOT_INTERVAL, ARENA_WIDTH, ARENA_HEIGHT,
    TOTAL_TILES, MAX_ENTITIES, D_MODEL, POPULAR_DECKS,
    sample_random_deck, STAGE_1_END_UPDATE, STAGE_2_END_UPDATE
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.env.replica_env import ClashRoyaleReplicaEnv
from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv
from cr_alphastar_agent.training.self_play_pool import SelfPlayLeagueManager
from cr_alphastar_agent.training.plot_progress import generate_training_plot


class RolloutBuffer:
    """Stores on-policy trajectories collected across N parallel environments."""
    def __init__(self, num_steps: int, num_envs: int, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.total_samples = num_steps * num_envs

        # Observations buffer
        self.obs = {
            "scalars": torch.zeros((num_steps, num_envs, 4), dtype=torch.float32, device=device),
            "hand_cards": torch.zeros((num_steps, num_envs, 5), dtype=torch.long, device=device),
            "entities": torch.zeros((num_steps, num_envs, MAX_ENTITIES, 7), dtype=torch.float32, device=device),
            "entity_mask": torch.ones((num_steps, num_envs, MAX_ENTITIES), dtype=torch.bool, device=device),
            "arena_grid": torch.zeros((num_steps, num_envs, 6, ARENA_HEIGHT, ARENA_WIDTH), dtype=torch.float32, device=device),
            "card_mask": torch.zeros((num_steps, num_envs, 5), dtype=torch.bool, device=device),
            "tile_mask": torch.zeros((num_steps, num_envs, TOTAL_TILES), dtype=torch.bool, device=device),
            "card_tile_mask": torch.zeros((num_steps, num_envs, 5, TOTAL_TILES), dtype=torch.bool, device=device),
        }

        # Step variables
        self.actions = torch.zeros((num_steps, num_envs, 2), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    def insert(self, step: int, obs: dict, actions: torch.Tensor, logprobs: torch.Tensor,
               rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor):
        for k in self.obs:
            self.obs[k][step] = obs[k]
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values

    def get_flattened(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flattens (T, N) dimensions into a unified batch dimension (T*N)."""
        b_obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in self.obs.items()}
        b_actions = self.actions.reshape(-1, 2)
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        return b_obs, b_actions, b_logprobs, b_advantages, b_returns


class HybridPPOTrainer:
    def __init__(self, model: AlphaStarCRNet, checkpoint_dir: str = "./checkpoints",
                 lr: float = PPO_LR, num_envs: int = NUM_ENVS, rollout_steps: int = ROLLOUT_STEPS,
                 env_backend: str = "crsim", pure_self_play: bool = True):
        self.hw = get_hardware_profile()
        self.device = torch.device(self.hw["device"])
        self.env_backend = env_backend
        self.pure_self_play = pure_self_play
        print(f"[HybridPPOTrainer] Hardware Profile: {self.hw['profile_name']} on {self.hw['device_name']}")
        print(f"[HybridPPOTrainer] Backend: {self.env_backend.upper()} (Physics & Rules Fidelity)")
        print(f"[HybridPPOTrainer] Opponent Mode: {'100% PURE AI vs AI (Self-Play)' if pure_self_play else 'Curriculum Staged'}")
        print(f"[HybridPPOTrainer] Micro-batch size: {self.hw['micro_batch']}, Gradient Accumulation: {self.hw['accum_steps']}")

        self.model = model.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        self.pool_dir = os.path.join(checkpoint_dir, "pool")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.pool_dir, exist_ok=True)

        self.league_manager = SelfPlayLeagueManager(pool_dir=self.pool_dir, device=self.hw["device"])

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        use_cuda_amp = self.hw["fp16"] and self.hw["device"] == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
        self.use_amp = use_cuda_amp

        self.global_step = 0
        self.update_count = 0
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps

        # Match statistics & history tracking (No random or guessed data)
        self.total_matches = 0
        self.total_p1_wins = 0
        self.total_p2_wins = 0
        self.total_draws = 0
        self.batch_matches = 0
        self.batch_p1_wins = 0
        self.batch_p2_wins = 0
        self.batch_draws = 0
        self.batch_tactical_zones = {
            "Back-of-King Slow-Roll": 0,
            "Center Split": 0,
            "Center Kill-Zone / Kite": 0,
            "Bridge / Tank-Trick": 0,
            "Deep Enemy Spell": 0,
            "Flank Lane Defense": 0,
        }
        self.history_file = os.path.join(checkpoint_dir, "training_history.json")
        self.training_history: List[dict] = []
        self._load_training_history()

        # Create parallel environments
        if env_backend == "crsim":
            self.envs = [CrSimReplicaEnv(seed=1000 + i) for i in range(num_envs)]
        else:
            self.envs = [ClashRoyaleReplicaEnv(seed=1000 + i) for i in range(num_envs)]
        self.env_opponents: List[Dict[str, Any]] = [{"type": "current", "model": None, "record": None} for _ in range(num_envs)]

        # Frozen shadow network for historical evaluation when needed
        self.shadow_opp_model = AlphaStarCRNet().to(self.device)
        self.shadow_opp_model.eval()

        # Resume if checkpoint exists
        self.load_latest_checkpoint()

    def _load_training_history(self):
        """Restores genuine match history from JSON on disk."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.training_history = json.load(f)
                if self.training_history:
                    last = self.training_history[-1]
                    self.total_matches = last.get("total_matches", 0)
                    self.total_p1_wins = last.get("total_wins", 0)
                    self.total_p2_wins = last.get("total_losses", 0)
                    self.total_draws = last.get("total_draws", 0)
                    print(f"[HybridPPOTrainer] Loaded match history: {self.total_matches} matches ({self.total_p1_wins}W - {self.total_p2_wins}L - {self.total_draws}D)")
            except Exception as e:
                print(f"[HybridPPOTrainer] Note: fresh match history initialized: {e}")

    def _save_training_history(self):
        """Persists genuine match history to JSON on disk."""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.training_history, f, indent=2)
        except Exception:
            pass

    def _sample_opponent_for_single_env(self, env_idx: int):
        """Assigns Player 2 opponent for a specific environment (Human Tactical Self-Play League)."""
        is_stage_3 = (self.pure_self_play or self.update_count > STAGE_2_END_UPDATE)
        opp_type = self.league_manager.sample_opponent_type(pure_self_play=is_stage_3)
        if opp_type == "historical" and self.league_manager.snapshots:
            record = self.league_manager.sample_historical_snapshot()
            self.env_opponents[env_idx] = {"type": "historical", "record": record}
        else:
            self.env_opponents[env_idx] = {"type": opp_type, "record": None}

    def _sample_opponents_for_envs(self):
        """AlphaStar PFSP: Sets up Player 2 opponent for each parallel env at rollout start."""
        for i in range(self.num_envs):
            self._sample_opponent_for_single_env(i)

    def _get_p2_action(self, env_idx: int, p2_obs: dict) -> np.ndarray:
        """Determines Player 2 action according to the assigned opponent type."""
        opp_info = self.env_opponents[env_idx]
        opp_type = opp_info["type"]

        if opp_type == "tactical_pro":
            return self.league_manager.tactical_bot.act(p2_obs)
        elif opp_type == "scripted":
            return self.league_manager.scripted_bot.act(p2_obs)
        elif opp_type == "rush":
            return self.league_manager.rush_bot.act(p2_obs)

        # Convert observation dict to PyTorch tensors
        tensor_obs = {
            k: torch.from_numpy(v).unsqueeze(0).to(self.device)
            for k, v in p2_obs.items()
        }

        with torch.no_grad():
            if opp_type == "historical" and opp_info["record"] is not None:
                self.shadow_opp_model.load_state_dict(opp_info["record"].state_dict)
                action, _, _, _ = self.shadow_opp_model.get_action_and_value(tensor_obs)
            else:
                # Live mirror self-play
                action, _, _, _ = self.model.get_action_and_value(tensor_obs)

        return action.squeeze(0).cpu().numpy()

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor, next_value: torch.Tensor,
                    gamma: float = GAMMA, gae_lambda: float = GAE_LAMBDA) -> Tuple[torch.Tensor, torch.Tensor]:
        """OpenAI Five Generalized Advantage Estimation (GAE)."""
        num_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0.0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values
        return advantages, returns

    def _sample_decks_for_phase(self) -> Tuple[List[int], List[int]]:
        """
        AlphaStar Staged Deck Curriculum:
        - Stage 1 (Updates 1-100): Fixed Scripted Bot Archetype Sets (cycling popular competitive decks)
        - Stage 2 (Updates 101-300): Completely Random 8-Card Decks Every Match vs Scripted Bots
        - Stage 3 (Updates 301+): Random Decks & Pure Self-Play (Bot vs Itself, NOT Scripted)
        """
        if self.update_count <= STAGE_1_END_UPDATE:
            deck_names = ["PEKKA Bridge Spam", "Classic Log Bait", "Hog 2.6 Cycle", "Giant Beatdown", "Double Dragon Beatdown", "Mega Knight Bait", "Lavaloon Air Beatdown"]
            idx = (self.update_count // 25) % len(deck_names)
            b_name = "Hog 2.6 Cycle"
            r_name = deck_names[idx]
            return POPULAR_DECKS[b_name], POPULAR_DECKS[r_name]
        elif self.update_count <= STAGE_2_END_UPDATE:
            # Stage 2: Fresh random 8-card deck for both Blue and Red EVERY single match!
            return sample_random_deck(), sample_random_deck()
        else:
            # Stage 3: Pure self-play with random decks (70% full random, 30% tournament archetypes)
            if np.random.rand() < 0.7:
                return sample_random_deck(), sample_random_deck()
            else:
                deck_names = ["PEKKA Bridge Spam", "Classic Log Bait", "Hog 2.6 Cycle", "Giant Beatdown", "Double Dragon Beatdown", "Mega Knight Bait", "Lavaloon Air Beatdown"]
                return POPULAR_DECKS[np.random.choice(deck_names)], POPULAR_DECKS[np.random.choice(deck_names)]

    @property
    def current_phase_desc(self) -> str:
        if self.update_count <= STAGE_1_END_UPDATE:
            return "Stage 1: Scripted Bot Sets (Fixed Archetypes)"
        elif self.update_count <= STAGE_2_END_UPDATE:
            return "Stage 2: Random Decks vs Scripted Bot Sets"
        else:
            return "Stage 3: Pure AlphaStar Self-Play (Bot vs Bot)"

    def collect_rollouts(self) -> RolloutBuffer:
        """Collects on-policy rollouts across parallel environments."""
        buffer = RolloutBuffer(self.rollout_steps, self.num_envs, self.device)

        # Reset batch match accumulators
        self.batch_matches = 0
        self.batch_p1_wins = 0
        self.batch_p2_wins = 0
        self.batch_draws = 0
        for k in self.batch_tactical_zones:
            self.batch_tactical_zones[k] = 0

        # Current observations
        cur_obs_p1 = [env.get_obs(player_id=1) for env in self.envs]

        self._sample_opponents_for_envs()

        for step in range(self.rollout_steps):
            # Batch Player 1 observations
            batched_obs = {
                k: torch.stack([torch.from_numpy(cur_obs_p1[i][k]) for i in range(self.num_envs)]).to(self.device)
                for k in cur_obs_p1[0]
            }

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    p1_actions, p1_logprobs, _, p1_values = self.model.get_action_and_value(batched_obs)

            p1_actions_np = p1_actions.cpu().numpy()

            # Track spatial deployment breakdown across 5 tactical zones
            for act in p1_actions_np:
                slot = int(act[0])
                if 0 <= slot <= 3:
                    t_idx = int(act[1])
                    tx = float(t_idx % ARENA_WIDTH)
                    ty = float(t_idx // ARENA_WIDTH)
                    if ty <= 4.0:
                        self.batch_tactical_zones["Back-of-King Slow-Roll"] += 1
                    elif 7.5 <= tx <= 10.5 and 4.0 < ty <= 8.0:
                        self.batch_tactical_zones["Center Split"] += 1
                    elif 7.0 <= tx <= 11.0 and 8.0 < ty <= 13.0:
                        self.batch_tactical_zones["Center Kill-Zone / Kite"] += 1
                    elif 13.0 < ty <= 15.5:
                        self.batch_tactical_zones["Bridge / Tank-Trick"] += 1
                    elif ty > 16.0:
                        self.batch_tactical_zones["Deep Enemy Spell"] += 1
                    else:
                        self.batch_tactical_zones["Flank Lane Defense"] += 1

            step_rewards = []
            step_dones = []

            # Step parallel environments
            for env_idx in range(self.num_envs):
                env = self.envs[env_idx]
                p2_obs = env.get_obs(player_id=2)
                p2_action = self._get_p2_action(env_idx, p2_obs)

                next_p1_obs, _, r1, r2, term, trunc, info = env.step_both(p1_actions_np[env_idx], p2_action)
                done = term or trunc

                step_rewards.append(r1)
                step_dones.append(1.0 if done else 0.0)

                if done:
                    # Accurately record match result directly from simulation battle engine
                    b = env.battle
                    p1_crowns = b.crowns[0] if b else 0
                    p2_crowns = b.crowns[1] if b else 0
                    res = b.result if b else None

                    if res == 0 or p1_crowns > p2_crowns:
                        p2_won = False
                        self.total_p1_wins += 1
                        self.batch_p1_wins += 1
                    elif res == 1 or p2_crowns > p1_crowns:
                        p2_won = True
                        self.total_p2_wins += 1
                        self.batch_p2_wins += 1
                    else:
                        p2_won = False
                        self.total_draws += 1
                        self.batch_draws += 1

                    self.total_matches += 1
                    self.batch_matches += 1

                    # Record match result for PFSP league weighting
                    opp_record = self.env_opponents[env_idx]["record"]
                    self.league_manager.record_match_result(opp_record, won_against_live=p2_won)

                    # Refresh Player 2 opponent for the new match
                    self._sample_opponent_for_single_env(env_idx)

                    # Configure fresh decks for current curriculum phase (randomized every match in Stage 2 & 3)
                    b_deck, r_deck = self._sample_decks_for_phase()
                    if hasattr(env, "blue_deck_ids"):
                        env.blue_deck_ids = list(b_deck)
                        env.red_deck_ids = list(r_deck)
                    elif hasattr(env, "deck"):
                        env.deck = {1: list(b_deck), 2: list(r_deck)}
                    next_p1_obs, _ = env.reset()

                cur_obs_p1[env_idx] = next_p1_obs

            rewards_tensor = torch.tensor(step_rewards, dtype=torch.float32, device=self.device)
            dones_tensor = torch.tensor(step_dones, dtype=torch.float32, device=self.device)

            buffer.insert(
                step=step,
                obs=batched_obs,
                actions=p1_actions,
                logprobs=p1_logprobs,
                rewards=rewards_tensor,
                dones=dones_tensor,
                values=p1_values
            )

        # Bootstrap next value for GAE
        final_batched_obs = {
            k: torch.stack([torch.from_numpy(cur_obs_p1[i][k]) for i in range(self.num_envs)]).to(self.device)
            for k in cur_obs_p1[0]
        }
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                next_value = self.model.get_value(final_batched_obs)

        advantages, returns = self.compute_gae(buffer.rewards, buffer.values, buffer.dones, next_value)
        buffer.advantages = advantages
        buffer.returns = returns

        self.global_step += (self.rollout_steps * self.num_envs)
        return buffer

    def ppo_update(self, buffer: RolloutBuffer, epochs: int = PPO_EPOCHS) -> Dict[str, float]:
        """
        Runs PPO update using dynamic Gradient Accumulation:
        T4 and RTX 3050 produce mathematically identical gradient steps.
        """
        b_obs, b_actions, b_logprobs, b_advantages, b_returns = buffer.get_flattened()
        # Advantage normalization
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_samples = b_actions.size(0)
        micro_bs = self.hw["micro_batch"]
        accum_steps = self.hw["accum_steps"]

        metrics = {
            "pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
            "card_entropy": 0.0, "card_entropy_active": 0.0,
            "tile_entropy": 0.0, "approx_kl": 0.0
        }
        num_updates = 0

        for _ in range(epochs):
            perm = torch.randperm(total_samples, device=self.device)
            self.optimizer.zero_grad()

            for step_idx, start in enumerate(range(0, total_samples, micro_bs)):
                end = min(start + micro_bs, total_samples)
                mb_idx = perm[start:end]
                mb_obs = {k: v[mb_idx] for k, v in b_obs.items()}

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    _, newlogprob, (card_ent, tile_ent), newvalue = self.model.get_action_and_value(
                        mb_obs, b_actions[mb_idx], return_entropy_breakdown=True
                    )
                    logratio = newlogprob - b_logprobs[mb_idx]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()

                    # OpenAI Five Clipped Surrogate Objective
                    mb_adv = b_advantages[mb_idx]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((newvalue - b_returns[mb_idx]) ** 2).mean()
                    card_entropy_mean = card_ent.mean()
                    tile_entropy_mean = tile_ent.mean()
                    # Sharpened Tile Entropy: 0.04 * card_entropy + 0.0008 * tile_entropy
                    # Anneals spatial blur from 5.33 down to sharp tactical anchors at back-of-king, kite, and bridge
                    entropy_loss = 0.04 * card_entropy_mean + 0.0008 * tile_entropy_mean

                    loss = (pg_loss - entropy_loss + VF_COEF * v_loss) / accum_steps

                self.scaler.scale(loss).backward()

                # Track metrics (including active decision entropy where multiple choices exist)
                multi_choice = (mb_obs["card_mask"].sum(dim=-1) > 1)
                active_card_ent = card_ent[multi_choice].mean() if multi_choice.any() else card_entropy_mean

                metrics["pg_loss"] += pg_loss.item()
                metrics["v_loss"] += v_loss.item()
                metrics["entropy"] += (card_entropy_mean + tile_entropy_mean).item()
                metrics["card_entropy"] += card_entropy_mean.item()
                metrics["card_entropy_active"] += active_card_ent.item()
                metrics["tile_entropy"] += tile_entropy_mean.item()
                metrics["approx_kl"] += approx_kl.item()
                num_updates += 1

                if (step_idx + 1) % accum_steps == 0 or (end >= total_samples):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

        self.update_count += 1

        # Match statistics & genuine win rates (non-random, engine-verified)
        total_p1 = self.total_p1_wins
        total_p2 = self.total_p2_wins
        total_d = self.total_draws
        total_m = self.total_matches
        win_rate = (total_p1 / max(1, total_m)) * 100.0

        batch_p1 = self.batch_p1_wins
        batch_p2 = self.batch_p2_wins
        batch_d = self.batch_draws
        batch_m = self.batch_matches
        batch_win_rate = (batch_p1 / max(1, batch_m)) * 100.0 if batch_m > 0 else 50.0

        # Reset per-batch match counters
        self.batch_matches = 0
        self.batch_p1_wins = 0
        self.batch_p2_wins = 0
        self.batch_draws = 0

        # Periodic Snapshot to PFSP League
        if self.update_count % SNAPSHOT_INTERVAL == 0:
            self.league_manager.add_snapshot(self.model, self.global_step)

        # Save latest checkpoint
        self.save_checkpoint()

        # Compute tactical zone percentage distribution
        total_zone_deploys = sum(self.batch_tactical_zones.values()) or 1
        zone_pcts = {k: round((v / total_zone_deploys) * 100.0, 1) for k, v in self.batch_tactical_zones.items()}

        avg_metrics = {k: v / max(1, num_updates) for k, v in metrics.items()}
        avg_metrics.update({
            "total_matches": total_m,
            "total_wins": total_p1,
            "total_losses": total_p2,
            "total_draws": total_d,
            "win_rate": win_rate,
            "batch_matches": batch_m,
            "batch_wins": batch_p1,
            "batch_losses": batch_p2,
            "batch_draws": batch_d,
            "batch_win_rate": batch_win_rate,
            "tactical_zones": dict(self.batch_tactical_zones),
            "tactical_zone_pcts": zone_pcts,
        })

        # Append to telemetry history and generate obsidian graph
        history_entry = {
            "step": self.global_step,
            "update": self.update_count,
            "pg_loss": float(avg_metrics.get("pg_loss", 0.0)),
            "v_loss": float(avg_metrics.get("v_loss", 0.0)),
            "entropy": float(avg_metrics.get("entropy", 0.0)),
            "total_matches": total_m,
            "total_wins": total_p1,
            "total_losses": total_p2,
            "total_draws": total_d,
            "win_rate": float(win_rate),
            "batch_matches": batch_m,
            "batch_wins": batch_p1,
            "batch_losses": batch_p2,
            "batch_draws": batch_d,
            "batch_win_rate": float(batch_win_rate),
        }
        self.training_history.append(history_entry)
        self._save_training_history()

        try:
            generate_training_plot(self.training_history)
        except Exception as e:
            print(f"[HybridPPOTrainer] Warning: could not render plot: {e}")

        return avg_metrics

    def save_checkpoint(self):
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        state = {
            "global_step": self.global_step,
            "update_count": self.update_count,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "total_matches": self.total_matches,
            "total_p1_wins": self.total_p1_wins,
            "total_p2_wins": self.total_p2_wins,
            "total_draws": self.total_draws,
        }

        # Robust atomic save with retries to withstand Windows file locking
        tmp_path = os.path.join(self.checkpoint_dir, f"latest_{os.getpid()}_{int(time.time()*1000)}.tmp")
        for attempt in range(5):
            try:
                torch.save(state, tmp_path)
                os.replace(tmp_path, latest_path)
                break
            except Exception as e:
                if attempt == 4:
                    print(f"[HybridPPOTrainer] Warning: could not save latest.pt after 5 attempts: {e}")
                else:
                    time.sleep(0.15 * (attempt + 1))
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        # Periodic numbered checkpoint
        if self.update_count % 10 == 0:
            num_path = os.path.join(self.checkpoint_dir, f"ckpt_step_{self.global_step:08d}.pt")
            num_tmp = os.path.join(self.checkpoint_dir, f"ckpt_{os.getpid()}_{int(time.time()*1000)}.tmp")
            for attempt in range(3):
                try:
                    torch.save(state, num_tmp)
                    os.replace(num_tmp, num_path)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"[HybridPPOTrainer] Warning: could not save {num_path}: {e}")
                    else:
                        time.sleep(0.15 * (attempt + 1))
                finally:
                    if os.path.exists(num_tmp):
                        try:
                            os.remove(num_tmp)
                        except Exception:
                            pass

    def load_latest_checkpoint(self):
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        if os.path.exists(latest_path):
            try:
                ckpt = torch.load(latest_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scaler_state_dict" in ckpt and bool(ckpt["scaler_state_dict"]) and self.use_amp:
                    self.scaler.load_state_dict(ckpt["scaler_state_dict"])
                self.global_step = ckpt.get("global_step", 0)
                self.update_count = ckpt.get("update_count", 0)
                self.total_matches = ckpt.get("total_matches", self.total_matches)
                self.total_p1_wins = ckpt.get("total_p1_wins", self.total_p1_wins)
                self.total_p2_wins = ckpt.get("total_p2_wins", self.total_p2_wins)
                self.total_draws = ckpt.get("total_draws", self.total_draws)
                print(f"[HybridPPOTrainer] Loaded latest checkpoint from Step {self.global_step} (Update {self.update_count}) | Record: {self.total_p1_wins}W-{self.total_p2_wins}L-{self.total_draws}D ({self.total_matches} matches)")
            except Exception as e:
                print(f"[HybridPPOTrainer] Warning: could not load checkpoint: {e}")
