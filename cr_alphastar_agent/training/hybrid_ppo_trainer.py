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
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

from cr_alphastar_agent.config import (
    get_hardware_profile, PPO_LR, GAMMA, GAE_LAMBDA, CLIP_COEF,
    ENT_COEF, VF_COEF, MAX_GRAD_NORM, PPO_EPOCHS, ROLLOUT_STEPS,
    NUM_ENVS, EFFECTIVE_BATCH_SIZE, SNAPSHOT_INTERVAL, ARENA_WIDTH, ARENA_HEIGHT,
    TOTAL_TILES, MAX_ENTITIES, D_MODEL, POPULAR_DECKS,
    ENTITY_FEATURES, SCALAR_FEATURES,
    sample_random_deck, STAGE_1_END_UPDATE, STAGE_2_END_UPDATE,
    CURRENT_DECK, CURRENT_DECK_IDS, resolve_deck_ids, resolve_deck_names,
    SPELL_CARD_IDS, ELIXIR_LEAK_WEIGHT
)
from cr_alphastar_agent.model.alphastar_cr_net import (
    AlphaStarCRNet, adapt_legacy_state_dict,
)
from cr_alphastar_agent.env.replica_env import ClashRoyaleReplicaEnv
from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv
from cr_alphastar_agent.env.tactical_coach import (
    TacticalCoachBot,
    _make_aggro_bot,
    _make_defensive_bot,
    _make_spell_bot,
)
from cr_alphastar_agent.env.tactics import perspective as _perspective, touches_enemy_king
from cr_alphastar_agent.training.self_play_pool import SelfPlayLeagueManager
from cr_alphastar_agent.training.plot_progress import generate_training_plot

# The learner always plays the blue (bottom) seat, so its canonical frame is
# the seat-0 perspective.
persp = _perspective(0)

# Hinge-clamped entropy bonus.  The bonus only pays out when entropy drops
# below the ceiling, so PPO can never push the policy *above* the sharpness
# that behaviour cloning produced.
ENTROPY_CARD_CEILING = 0.90
ENTROPY_TILE_CEILING = 2.80
ENTROPY_CARD_COEF = 0.015
#: Raised from 0.0005 for PFSP league training: the old value let the tile
#: head collapse onto a single tile ((2, 13) in live matches).
ENTROPY_TILE_COEF = 0.004

# After a behaviour-cloning warm start the critic needs a few updates to
# settle before the actor is allowed to move.
CRITIC_WARMUP_UPDATES = 8
CRITIC_WARMUP_TARGET_LOSS = 0.04

# Weight of the tile distribution inside the BC anchor KL.
TILE_KL_WEIGHT = 0.25

# Share of the league that is a frozen snapshot of our own past policy even
# in scripted mode.  A league of three fixed bots gives the learner no
# learning signal - the win rate oscillates around 60 % with no trend.
SNAPSHOT_OPPONENT_RATE = 0.20

# Progressive KL target expansion.
# The leash holds the policy near the clone while the critic settles, then
# relaxes so PPO can improve beyond the teacher.
KL_RAMP_START_UPDATE = 50     # keep the tight target until here
KL_RAMP_END_UPDATE = 250      # fully expanded by here
KL_TARGET_MAX = 0.25          # final (loose) KL target
BC_KL_BETA_CAP = 20.0         # maximum leash beta at update 0
BC_KL_BETA_DECAY = 0.992      # per-update decay of the beta cap

# Named entropy presets.  "spec" is the reference formulation; "diverse"
# strengthens the tile bonus so the tile head cannot collapse onto a handful
# of anchors (the previous run fell from 30.4 distinct tiles per match to
# 4.0, which is exactly the "always the same three places" failure).
ENTROPY_PRESETS: dict = {
    "spec": {
        "card_ceiling": ENTROPY_CARD_CEILING,
        "tile_ceiling": ENTROPY_TILE_CEILING,
        "card_coef": ENTROPY_CARD_COEF,
        "tile_coef": ENTROPY_TILE_COEF,
    },
    "diverse": {
        "card_ceiling": 0.90,
        "tile_ceiling": 4.00,
        "card_coef": 0.015,
        "tile_coef": 0.010,
    },
}


def _atomic_write_json(path: str, payload: Any) -> None:
    """Write JSON atomically (temp file + rename).

    A reader (the training monitor) can otherwise catch the file half
    written and parse a truncated list, which silently produces wrong
    statistics - that is exactly how a bogus "placement collapse" was
    reported once.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _new_match_stats() -> dict:
    """Fresh per-match behavioural counters."""
    return {
        "deploys": 0,
        "waits": 0,
        "spells": 0,
        "spells_own_half": 0,
        "spells_king_tower": 0,
        "pushes_while_threatened": 0,
        "defensive_responses": 0,
        "tiles": set(),
        "cards": [],
    }


def _entropy_of_counts(counts, universe: int) -> float:
    """Shannon entropy (bits) of a tile-usage distribution."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    import math
    entropy = 0.0
    for value in counts:
        if value:
            p = value / total
            entropy -= p * math.log2(p)
    return entropy


def _tile_entropy(tiles, universe: int) -> float:
    """Entropy (bits) of the tile-usage histogram, over ``universe`` tiles.

    A policy that always deploys on three squares scores ~1.6 bits; a policy
    that spreads over 30 squares scores ~4.9 bits.  The value is therefore a
    direct measure of "the agent stopped using the same three places".
    """
    counts = [0] * universe
    for tile in tiles:
        counts[int(tile)] += 1
    return _entropy_of_counts(counts, universe)


class RolloutBuffer:
    """Stores on-policy trajectories collected across N parallel environments."""
    def __init__(self, num_steps: int, num_envs: int, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.total_samples = num_steps * num_envs

        # Observations buffer
        self.obs = {
            "scalars": torch.zeros((num_steps, num_envs, SCALAR_FEATURES), dtype=torch.float32, device=device),
            "hand_cards": torch.zeros((num_steps, num_envs, 5), dtype=torch.long, device=device),
            "entities": torch.zeros((num_steps, num_envs, MAX_ENTITIES, ENTITY_FEATURES), dtype=torch.float32, device=device),
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
        # The nucleus (top-p support) that was used to *sample* each action.
        # It has to be replayed during the PPO update: the nucleus shrinks as
        # the tile head learns, so recomputing it would make the sampled tile
        # impossible and give it a -1e9 log-probability, which blows the KL
        # metric up to ~1e7 and destroys the importance ratio.
        self.nucleus = torch.ones((num_steps, num_envs, TOTAL_TILES), dtype=torch.bool, device=device)

        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    def insert(self, step: int, obs: dict, actions: torch.Tensor, logprobs: torch.Tensor,
               rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor,
               nucleus: Optional[torch.Tensor] = None):
        for k in self.obs:
            self.obs[k][step] = obs[k]
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values
        if nucleus is not None:
            self.nucleus[step] = nucleus

    def get_flattened(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flattens (T, N) dimensions into a unified batch dimension (T*N)."""
        b_obs = {k: v.reshape(-1, *v.shape[2:]) for k, v in self.obs.items()}
        b_actions = self.actions.reshape(-1, 2)
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        return b_obs, b_actions, b_logprobs, b_advantages, b_returns

    def get_flattened_nucleus(self) -> torch.Tensor:
        """The per-sample sampling nucleus, flattened to (T*N, TOTAL_TILES)."""
        return self.nucleus.reshape(-1, self.nucleus.shape[-1])


class HybridPPOTrainer:
    def __init__(self, model: AlphaStarCRNet, checkpoint_dir: str = "./checkpoints",
                 lr: float = PPO_LR, num_envs: int = NUM_ENVS, rollout_steps: int = ROLLOUT_STEPS,
                 env_backend: str = "crsim", pure_self_play: bool = True,
                 deck: Any = "current", opponent_mode: str = "self_play",
                 init_checkpoint: Optional[str] = None,
                 critic_warmup_updates: int = CRITIC_WARMUP_UPDATES,
                 bc_anchor_kl: float = 0.05,
                 bc_kl_beta: float = 1.0,
                 bc_kl_target: float = 0.02,
                 kl_ramp_start_update: int = KL_RAMP_START_UPDATE,
                 kl_ramp_end_update: int = KL_RAMP_END_UPDATE,
                 kl_target_max: float = KL_TARGET_MAX,
                 bc_kl_beta_cap: float = BC_KL_BETA_CAP,
                 bc_kl_beta_decay: float = BC_KL_BETA_DECAY,
                 elixir_leak_weight: float = ELIXIR_LEAK_WEIGHT,
                 entropy_card_ceiling: float = ENTROPY_CARD_CEILING,
                 entropy_tile_ceiling: float = ENTROPY_TILE_CEILING,
                 entropy_card_coef: float = ENTROPY_CARD_COEF,
                 entropy_tile_coef: float = ENTROPY_TILE_COEF,
                 snapshot_opponent_rate: float = SNAPSHOT_OPPONENT_RATE):
        self.hw = get_hardware_profile()
        self.device = torch.device(self.hw["device"])
        self.env_backend = env_backend
        self.pure_self_play = pure_self_play
        # Deck curriculum: "current" plays the account's deck, "curriculum"
        # cycles the POPULAR_DECKS archetypes, "random" samples fresh decks.
        self.deck_mode = deck if isinstance(deck, str) else "custom"
        self.learner_deck_ids = resolve_deck_ids(deck)
        # The simulator needs real card names and exactly eight distinct
        # cards, so the environment always receives engine names.
        self.learner_deck_names = resolve_deck_names(deck)
        # Opponent source: "self_play" (league), "scripted" (rule bots only),
        # "mixed" (both).
        self.opponent_mode = opponent_mode
        print(f"[HybridPPOTrainer] Hardware Profile: {self.hw['profile_name']} on {self.hw['device_name']}")
        print(f"[HybridPPOTrainer] Backend: {self.env_backend.upper()} (Physics & Rules Fidelity)")
        print(f"[HybridPPOTrainer] Opponent Mode: {'100% PURE AI vs AI (Self-Play)' if pure_self_play else 'Curriculum Staged'}"
              f" | league={opponent_mode}")
        print(f"[HybridPPOTrainer] Learner Deck: {self.deck_mode} -> {self.learner_deck_ids}")
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
        # Tile-usage histogram: the direct measure of spatial collapse.
        self.batch_tiles: Counter = Counter()
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

        # Per-match behavioural log (first match vs last match comparison).
        self.match_log_file = os.path.join(checkpoint_dir, "match_log.json")
        self.match_log: List[dict] = []
        self._load_match_log()
        self._match_stats = [_new_match_stats() for _ in range(num_envs)]

        # Create parallel environments
        if env_backend == "crsim":
            self.envs = [CrSimReplicaEnv(seed=1000 + i,
                                        elixir_leak_weight=elixir_leak_weight)
                         for i in range(num_envs)]
        else:
            self.envs = [ClashRoyaleReplicaEnv(seed=1000 + i) for i in range(num_envs)]
        self.elixir_leak_weight = float(elixir_leak_weight)
        if self.elixir_leak_weight > 0.0:
            print(f"[HybridPPOTrainer] Elixir leak penalty active: "
                  f"{self.elixir_leak_weight} per wasted elixir")

        # Hinge-clamped entropy bonus (configurable).
        self.entropy_card_ceiling = float(entropy_card_ceiling)
        self.entropy_tile_ceiling = float(entropy_tile_ceiling)
        self.entropy_card_coef = float(entropy_card_coef)
        self.entropy_tile_coef = float(entropy_tile_coef)
        print(f"[HybridPPOTrainer] Entropy bonus: card coef {self.entropy_card_coef} "
              f"(ceiling {self.entropy_card_ceiling}), tile coef {self.entropy_tile_coef} "
              f"(ceiling {self.entropy_tile_ceiling})")
        self.env_opponents: List[Dict[str, Any]] = [{"type": "current", "model": None, "record": None} for _ in range(num_envs)]

        # Rule-based opponents used when the league is restricted to scripted
        # opponents ("scripted" / "mixed" modes).  The variants differ in
        # *behaviour* (aggression, hoarding, spell-happy), which gives the
        # learner opponents it has not already solved.
        self.coach_bot = TacticalCoachBot(player_id=2, seed=17)
        self.opponent_bots: Dict[str, Any] = {
            "coach": self.coach_bot,
            "coach_aggro": _make_aggro_bot(player_id=2, seed=23),
            "coach_defensive": _make_defensive_bot(player_id=2, seed=29),
            "coach_spell": _make_spell_bot(player_id=2, seed=31),
            "tactical_pro": self.league_manager.tactical_bot,
            "rush": self.league_manager.rush_bot,
        }
        # Probability of facing a frozen snapshot of our own past policy
        # even in scripted mode.  Without this the league has no learning
        # signal at all and the win rate oscillates without a trend.
        self.snapshot_opponent_rate = float(snapshot_opponent_rate)

        # Frozen shadow network for historical evaluation when needed
        self.shadow_opp_model = AlphaStarCRNet().to(self.device)
        self.shadow_opp_model.eval()

        # Critic warmup: after a fresh BC clone the value head is calibrated
        # during BC, but it still needs a few PPO updates to settle on the
        # RL distribution.  During the warmup only the value head is
        # updated - the actor stays exactly where the clone put it.
        self.critic_warmup_updates = int(critic_warmup_updates)
        self._critic_warmup_active = False
        self._bc_initialised = False

        # BC anchor: a frozen copy of the cloned policy acts as a reference.
        # PPO is allowed to improve on the clone but not to drift away from
        # it - this is what stopped the policy from being washed out.
        self.bc_anchor_kl = float(bc_anchor_kl)
        # Progressive KL target expansion: the leash is tight while the
        # critic settles and then relaxes so PPO can outgrow the teacher.
        self.kl_ramp_start_update = int(kl_ramp_start_update)
        self.kl_ramp_end_update = int(kl_ramp_end_update)
        self.kl_target_max = float(kl_target_max)
        self.bc_kl_beta_cap = float(bc_kl_beta_cap)
        self.bc_kl_beta_decay = float(bc_kl_beta_decay)
        self.bc_reference_model: Optional[AlphaStarCRNet] = None
        self.bc_kl_beta = float(bc_kl_beta)
        self.bc_kl_target = float(bc_kl_target)

        # Optional behaviour-cloning warm start.  An explicit --init-bc
        # checkpoint takes precedence over the RL resume checkpoint, otherwise
        # a stale latest.pt (e.g. restored from Drive) would silently
        # overwrite the freshly cloned weights.
        if init_checkpoint:
            self.load_weights_from(init_checkpoint)
            self._setup_bc_reference_and_warmup()
        else:
            self.load_latest_checkpoint()

        # If a dedicated bc_reference.pt exists and reference model hasn't been instantiated yet
        ref_path = os.path.join(self.checkpoint_dir, "bc_reference.pt")
        if os.path.exists(ref_path) and self.bc_reference_model is None and self.bc_anchor_kl > 0.0:
            try:
                ref_ckpt = torch.load(ref_path, map_location=self.device, weights_only=False)
                ref_state = ref_ckpt.get("model_state_dict", ref_ckpt)
                self.bc_reference_model = AlphaStarCRNet().to(self.device)
                self.bc_reference_model.load_state_dict(adapt_legacy_state_dict(ref_state))
                self.bc_reference_model.eval()
                for param in self.bc_reference_model.parameters():
                    param.requires_grad_(False)
                print(f"[HybridPPOTrainer] Loaded BC reference anchor from {ref_path}")
            except Exception as e:
                print(f"[HybridPPOTrainer] Note: could not load bc_reference.pt: {e}")

    def _setup_bc_reference_and_warmup(self, state_dict: Optional[dict] = None,
                                       critic_warmup_active: Optional[bool] = None):
        """Initialises the frozen reference model for the KL leash and arms critic warmup."""
        self._bc_initialised = True
        if critic_warmup_active is None:
            self._critic_warmup_active = (self.update_count < self.critic_warmup_updates)
        else:
            self._critic_warmup_active = bool(critic_warmup_active)

        if self._critic_warmup_active:
            remaining = max(0, self.critic_warmup_updates - self.update_count)
            print(f"[HybridPPOTrainer] Critic warmup armed: {remaining} update(s) "
                  f"(value head only) before actor is released.")

        if self.bc_anchor_kl > 0.0 and self.bc_reference_model is None:
            self.bc_reference_model = AlphaStarCRNet().to(self.device)
            ref_dict = state_dict if state_dict is not None else copy.deepcopy(self.model.state_dict())
            self.bc_reference_model.load_state_dict(adapt_legacy_state_dict(ref_dict))
            self.bc_reference_model.eval()
            for param in self.bc_reference_model.parameters():
                param.requires_grad_(False)
            print(f"[HybridPPOTrainer] BC anchor active: KL leash to the cloned policy "
                  f"(beta={self.bc_kl_beta:.3f}, target={self._current_kl_target():.4f}, "
                  f"ramp {self.kl_ramp_start_update}->{self.kl_ramp_end_update})")

    def _current_kl_target(self) -> float:
        """Progressive KL target for the BC anchor leash.

        The leash was a bungee cord: with a fixed target of 0.02 the policy
        could never outgrow the teacher and the win rate froze at ~60 %.
        The target is therefore held tight while the critic settles (up to
        ``kl_ramp_start_update``) and then expanded linearly to
        ``kl_target_max`` by ``kl_ramp_end_update``, so PPO can optimise
        freely beyond the cloned behaviour once the advantages are clean.
        """
        if self.update_count <= self.kl_ramp_start_update:
            return self.bc_kl_target
        if self.update_count >= self.kl_ramp_end_update:
            return self.kl_target_max
        span = max(1, self.kl_ramp_end_update - self.kl_ramp_start_update)
        frac = (self.update_count - self.kl_ramp_start_update) / span
        return self.bc_kl_target + frac * (self.kl_target_max - self.bc_kl_target)

    def _current_beta_cap(self) -> float:
        """Maximum allowed leash beta; decays slightly on every update."""
        return max(0.5, self.bc_kl_beta_cap * (self.bc_kl_beta_decay ** self.update_count))

    def _bc_anchor_kl(self, obs: Dict[str, torch.Tensor],
                      actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean KL(BC reference || live policy) over a minibatch.

        The reference is the frozen behaviour-cloned policy, the live policy
        is the one being trained.  A growing KL means PPO is drifting away
        from the cloned behaviour - the entropy clamp and the critic warmup
        soften that drift, this leash is the hard guarantee against it.

        The KL is measured on the autoregressive card-slot distribution (the
        decision that determines *which* card is played) and, when the taken
        actions are available, on the tile distribution as well.
        """
        if self.bc_reference_model is None:
            return torch.zeros((), device=self.device)

        card_mask = obs["card_mask"].bool()
        with torch.no_grad():
            ref_core = self.bc_reference_model.forward_core(
                obs["scalars"], obs["hand_cards"], obs["entities"],
                obs["entity_mask"], obs["arena_grid"])
            ref_card_logits = self.bc_reference_model.card_head(ref_core).float()
            ref_card_logits = ref_card_logits.masked_fill(~card_mask, -1e9)
            ref_card_dist = torch.distributions.Categorical(logits=ref_card_logits)

        live_core = self.model.forward_core(
            obs["scalars"], obs["hand_cards"], obs["entities"],
            obs["entity_mask"], obs["arena_grid"])
        live_card_logits = self.model.card_head(live_core).float()
        live_card_logits = live_card_logits.masked_fill(~card_mask, -1e9)
        live_card_dist = torch.distributions.Categorical(logits=live_card_logits)

        card_kl = torch.distributions.kl_divergence(ref_card_dist, live_card_dist).mean()

        if actions is None:
            return card_kl

        # Tile distribution conditioned on the card that was actually taken.
        slots = actions[:, 0].clamp(0, 4).view(-1, 1, 1).expand(-1, 1, TOTAL_TILES)
        with torch.no_grad():
            ref_emb = self.bc_reference_model.chosen_action_emb(actions[:, 0])
            ref_tile_logits = self.bc_reference_model.tile_head(
                torch.cat([ref_core, ref_emb], dim=-1)).float()
            ref_tile_logits = ref_tile_logits.masked_fill(
                ~torch.gather(obs["card_tile_mask"].bool(), 1, slots).squeeze(1), -1e9)
            ref_tile_dist = torch.distributions.Categorical(logits=ref_tile_logits)

        live_emb = self.model.chosen_action_emb(actions[:, 0])
        live_tile_logits = self.model.tile_head(
            torch.cat([live_core, live_emb], dim=-1)).float()
        live_tile_logits = live_tile_logits.masked_fill(
            ~torch.gather(obs["card_tile_mask"].bool(), 1, slots).squeeze(1), -1e9)
        live_tile_dist = torch.distributions.Categorical(logits=live_tile_logits)

        tile_kl = torch.distributions.kl_divergence(ref_tile_dist, live_tile_dist).mean()
        return card_kl + TILE_KL_WEIGHT * tile_kl

    def load_weights_from(self, path: str) -> bool:
        """Load model weights (BC warm start) without touching RL counters."""
        if not path or not os.path.exists(path):
            print(f"[HybridPPOTrainer] Init checkpoint not found: {path}")
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(adapt_legacy_state_dict(state))
            print(f"[HybridPPOTrainer] Initialised weights from {os.path.basename(path)}")
            return True
        except Exception as exc:
            print(f"[HybridPPOTrainer] Warning: could not load {path}: {exc}")
            return False

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
        """Persists genuine match history to JSON on disk (atomically)."""
        try:
            _atomic_write_json(self.history_file, self.training_history)
        except Exception:
            pass

    def _load_match_log(self) -> None:
        """Restores the per-match behavioural log from disk."""
        if not os.path.exists(self.match_log_file):
            return
        try:
            with open(self.match_log_file, "r", encoding="utf-8") as f:
                self.match_log = json.load(f)
            print(f"[HybridPPOTrainer] Loaded match log: {len(self.match_log)} matches")
        except Exception as exc:
            print(f"[HybridPPOTrainer] Note: fresh match log initialized: {exc}")

    def _save_match_log(self) -> None:
        """Appends the per-match behavioural log to disk (atomically)."""
        try:
            _atomic_write_json(self.match_log_file, self.match_log[-2000:])
        except Exception:
            pass

    def _sample_opponent_for_single_env(self, env_idx: int):
        """Assigns Player 2 opponent for a specific environment.

        ``opponent_mode``:
          * ``"scripted"``  - rule-based bots only (coach / tactical / rush).
            Used while bootstrapping on the account deck so the learner faces
            predictable, well-behaved opponents first.
          * ``"self_play"`` - the full PFSP league (historical + mirror).
          * ``"mixed"``     - both, weighted towards scripted early on.
        """
        if self.opponent_mode == "scripted":
            # A slice of the league is our own frozen past: without it the
            # learner faces a fixed target and stops improving.
            if (self.snapshot_opponent_rate > 0.0
                    and self.league_manager.snapshots
                    and np.random.rand() < self.snapshot_opponent_rate):
                record = self.league_manager.sample_historical_snapshot()
                self.env_opponents[env_idx] = {"type": "historical", "record": record}
                return
            opp_type = str(np.random.choice(list(self.opponent_bots.keys())))
            self.env_opponents[env_idx] = {"type": opp_type, "record": None}
            return

        if self.opponent_mode == "mixed":
            # Ramp from scripted to league over the first 100 updates.
            scripted_share = max(0.0, 1.0 - self.update_count / 100.0)
            if np.random.rand() < scripted_share:
                r = np.random.rand()
                opp_type = "coach" if r < 0.5 else ("tactical_pro" if r < 0.8 else "rush")
                self.env_opponents[env_idx] = {"type": opp_type, "record": None}
                return

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

        if opp_type in self.opponent_bots:
            return self.opponent_bots[opp_type].act(p2_obs)
        if opp_type == "scripted":
            return self.league_manager.scripted_bot.act(p2_obs)

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

    def _sample_decks_for_phase(self) -> Tuple[List[str], List[str]]:
        """
        Deck curriculum.  Returns (learner deck, opponent deck) as *engine
        card names* - the simulator rejects decks that are not eight distinct
        real cards.

        ``deck_mode``:
          * ``"current"``  - the learner always plays the account deck
            (Knight, Mini PEKKA, Archers, Musketeer, Arrows, Fireball, Giant,
            Minions) while the opponent draws a varied deck.  This is the
            mode used to master the deck we actually play on the ladder.
          * ``"curriculum"`` - AlphaStar staged curriculum (archetypes ->
            random decks -> self-play).
          * ``"random"``   - fresh random decks every match.
          * ``"custom"``   - a caller-supplied deck list for the learner.
        """
        if self.deck_mode == "current":
            # Opponent still varies so the learner does not overfit one matchup.
            if np.random.rand() < 0.5:
                opp = resolve_deck_names("random")
            else:
                deck_names = ["Hog 2.6 Cycle", "Giant Beatdown", "Classic Log Bait"]
                opp = resolve_deck_names(POPULAR_DECKS[np.random.choice(deck_names)])
            return list(self.learner_deck_names), opp

        if self.deck_mode == "random":
            return resolve_deck_names("random"), resolve_deck_names("random")

        if self.deck_mode == "custom":
            return list(self.learner_deck_names), resolve_deck_names("random")

        # AlphaStar Staged Deck Curriculum:
        # - Stage 1 (Updates 1-100): Fixed Scripted Bot Archetype Sets (cycling popular competitive decks)
        # - Stage 2 (Updates 101-300): Completely Random 8-Card Decks Every Match vs Scripted Bots
        # - Stage 3 (Updates 301+): Random Decks & Pure Self-Play (Bot vs Itself, NOT Scripted)
        if self.update_count <= STAGE_1_END_UPDATE:
            deck_names = ["PEKKA Bridge Spam", "Classic Log Bait", "Hog 2.6 Cycle", "Giant Beatdown", "Double Dragon Beatdown", "Mega Knight Bait", "Lavaloon Air Beatdown"]
            idx = (self.update_count // 25) % len(deck_names)
            b_name = "Hog 2.6 Cycle"
            r_name = deck_names[idx]
            return (resolve_deck_names(POPULAR_DECKS[b_name]),
                    resolve_deck_names(POPULAR_DECKS[r_name]))
        elif self.update_count <= STAGE_2_END_UPDATE:
            # Stage 2: Fresh random 8-card deck for both Blue and Red EVERY single match!
            return resolve_deck_names("random"), resolve_deck_names("random")
        else:
            # Stage 3: Pure self-play with random decks (70% full random, 30% tournament archetypes)
            if np.random.rand() < 0.7:
                return resolve_deck_names("random"), resolve_deck_names("random")
            else:
                deck_names = ["PEKKA Bridge Spam", "Classic Log Bait", "Hog 2.6 Cycle", "Giant Beatdown", "Double Dragon Beatdown", "Mega Knight Bait", "Lavaloon Air Beatdown"]
                return (resolve_deck_names(POPULAR_DECKS[np.random.choice(deck_names)]),
                        resolve_deck_names(POPULAR_DECKS[np.random.choice(deck_names)]))

    @property
    def current_phase_desc(self) -> str:
        if self.deck_mode == "current":
            return f"Current Deck Mastery ({self.learner_deck_ids}) vs {self.opponent_mode}"
        if self.deck_mode == "random":
            return "Random Deck Training"
        if self.update_count <= STAGE_1_END_UPDATE:
            return "Stage 1: Scripted Bot Sets (Fixed Archetypes)"
        elif self.update_count <= STAGE_2_END_UPDATE:
            return "Stage 2: Random Decks vs Scripted Bot Sets"
        else:
            return "Stage 3: Pure AlphaStar Self-Play (Bot vs Bot)"

    def _telemetry_step(self, env_idx: int, action: np.ndarray,
                        obs: Dict[str, np.ndarray]) -> None:
        """Accumulate per-match behavioural telemetry for the learner.

        The counters exist so the training monitor can compare the *first*
        matches with the *last* ones and state concretely what changed:
        how many spells were wasted, how many pushes ignored a defence, and
        how many distinct tiles the agent used.
        """
        stats = self._match_stats[env_idx]
        slot = int(action[0])
        if not (0 <= slot <= 3):
            stats["waits"] += 1
            return

        tile = int(action[1])
        tx = float(tile % ARENA_WIDTH)
        ty = float(tile // ARENA_WIDTH)
        hand = obs.get("hand_cards")
        card_id = int(hand[slot]) if hand is not None and slot < len(hand) else 0

        stats["deploys"] += 1
        stats["tiles"].add(tile)
        stats["cards"].append(card_id)

        if card_id in SPELL_CARD_IDS:
            stats["spells"] += 1
            if persp.is_own_half(ty):
                stats["spells_own_half"] += 1
            if touches_enemy_king(tx, ty, 0):
                stats["spells_king_tower"] += 1
        else:
            if stats["threatened"]:
                # A troop deployed while the opponent was on our half.
                if not persp.is_own_half(ty):
                    stats["pushes_while_threatened"] += 1
                else:
                    stats["defensive_responses"] += 1

    def _telemetry_threat(self, env_idx: int, obs: Dict[str, np.ndarray]) -> None:
        """Flag whether the opponent currently has units on our half."""
        entities = obs.get("entities")
        mask = obs.get("entity_mask")
        if entities is None or mask is None:
            return
        threat = False
        for i in range(len(mask)):
            if mask[i]:
                continue
            e = entities[i]
            if float(e[1]) < 0.0 and float(e[4]) > 0.05:
                y = float(e[3]) * ARENA_HEIGHT
                if persp.is_own_half(y):
                    threat = True
                    break
        self._match_stats[env_idx]["threatened"] = threat

    def _telemetry_finish(self, env_idx: int, result: Optional[int],
                          p1_crowns: int, p2_crowns: int,
                          opponent_type: str) -> None:
        """Close a match and append it to the persistent match log."""
        stats = self._match_stats[env_idx]
        entry = {
            "match": self.total_matches,
            "update": self.update_count,
            "opponent": opponent_type,
            "result": ("WIN" if result == 0 else "LOSS" if result == 1 else "DRAW"),
            "crowns": f"{p1_crowns}-{p2_crowns}",
            "deploys": stats["deploys"],
            "waits": stats["waits"],
            "decisions": stats["deploys"] + stats["waits"],
            "spells": stats["spells"],
            "spells_own_half": stats["spells_own_half"],
            "spells_king_tower": stats["spells_king_tower"],
            "pushes_while_threatened": stats["pushes_while_threatened"],
            "defensive_responses": stats["defensive_responses"],
            "distinct_tiles": len(stats["tiles"]),
            "tile_entropy": float(_tile_entropy(stats["tiles"], TOTAL_TILES)),
        }
        self.match_log.append(entry)
        self._save_match_log()
        self._match_stats[env_idx] = _new_match_stats()

    def collect_rollouts(self) -> RolloutBuffer:
        """Collects on-policy rollouts across parallel environments."""
        buffer = RolloutBuffer(self.rollout_steps, self.num_envs, self.device)

        # Reset batch match accumulators
        self.batch_matches = 0
        self.batch_p1_wins = 0
        self.batch_p2_wins = 0
        self.batch_draws = 0
        self.batch_tiles.clear()
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
                    p1_actions, p1_logprobs, _, p1_values, p1_nucleus = \
                        self.model.get_action_and_value(
                            batched_obs, return_entropy_breakdown=True, return_nucleus=True)

            p1_actions_np = p1_actions.cpu().numpy()

            # Track spatial deployment breakdown across 5 tactical zones
            for env_idx, act in enumerate(p1_actions_np):
                slot = int(act[0])

                # Behavioural telemetry (what the agent is actually doing).
                # This has to run for *every* decision, including the waits,
                # otherwise the "waits per match" figure is silently wrong.
                self._telemetry_threat(env_idx, cur_obs_p1[env_idx])
                self._telemetry_step(env_idx, act, cur_obs_p1[env_idx])

                if 0 <= slot <= 3:
                    t_idx = int(act[1])
                    self.batch_tiles[t_idx] += 1
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

                    # Close the behavioural telemetry for this match.
                    self._telemetry_finish(
                        env_idx, res, p1_crowns, p2_crowns,
                        self.env_opponents[env_idx]["type"],
                    )

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
                values=p1_values,
                nucleus=p1_nucleus,
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
        b_nucleus = buffer.get_flattened_nucleus()
        # Advantage normalization
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_samples = b_actions.size(0)
        micro_bs = self.hw["micro_batch"]
        accum_steps = self.hw["accum_steps"]

        # Critic warmup: only the value head moves until the critic has
        # settled, so the actor cannot be dragged around by noisy advantages.
        critic_warmup = self._critic_warmup_active

        metrics = {
            "pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
            "card_entropy": 0.0, "card_entropy_active": 0.0,
            "tile_entropy": 0.0, "approx_kl": 0.0, "anchor_kl": 0.0
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
                        mb_obs, b_actions[mb_idx], return_entropy_breakdown=True,
                        # Replay the nucleus that was used to sample the action
                        # so the evaluated distribution matches the behaviour
                        # policy exactly (otherwise the ratio explodes).
                        nucleus_mask=b_nucleus[mb_idx],
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

                    # Hinge-clamped entropy bonus.
                    # The unconditional bonus kept *pushing* entropy upwards:
                    # PPO drove the policy from the BC entropy of 4.25 to 6.20
                    # and washed the cloned behaviour out.  The bonus is now
                    # clamped so it only regularises entropy that has fallen
                    # *below* the tactical floor - it can never raise entropy
                    # above the level the behaviour cloning produced.
                    card_ent_bonus = torch.clamp(card_ent, max=self.entropy_card_ceiling)
                    tile_ent_bonus = torch.clamp(tile_ent, max=self.entropy_tile_ceiling)
                    entropy_loss = (self.entropy_card_coef * card_ent_bonus
                                    + self.entropy_tile_coef * tile_ent_bonus).mean()

                    # BC anchor: KL between the live policy and the frozen
                    # clone.  Keeps PPO from drifting away from the
                    # behaviour it was initialised with.
                    anchor_kl = torch.zeros((), device=self.device)
                    if self.bc_reference_model is not None and not critic_warmup:
                        anchor_kl = self._bc_anchor_kl(mb_obs, b_actions[mb_idx])
                        anchor_penalty = self.bc_kl_beta * anchor_kl
                    else:
                        anchor_penalty = torch.zeros((), device=self.device)

                    if critic_warmup:
                        # Critic warmup: only the value head is updated, the
                        # actor is left exactly where behaviour cloning put it.
                        loss = (VF_COEF * v_loss) / accum_steps
                    else:
                        loss = (pg_loss - entropy_loss + anchor_penalty
                                + VF_COEF * v_loss) / accum_steps

                self.scaler.scale(loss).backward()

                if critic_warmup:
                    # Freeze every parameter except the value head so the
                    # actor cannot move until the critic is calibrated.
                    for name, param in self.model.named_parameters():
                        if not name.startswith("value_head"):
                            param.grad = None

                # Track metrics (including active decision entropy where multiple choices exist)
                multi_choice = (mb_obs["card_mask"].sum(dim=-1) > 1)
                active_card_ent = card_ent[multi_choice].mean() if multi_choice.any() else card_entropy_mean

                if critic_warmup:
                    # The actor does not move, but its distribution is still
                    # measurable — logging zero here made the whole warmup
                    # phase look like a broken entropy metric.
                    metrics["pg_loss"] += 0.0
                else:
                    metrics["pg_loss"] += pg_loss.item()
                metrics["entropy"] += (card_entropy_mean + tile_entropy_mean).item()
                metrics["v_loss"] += v_loss.item()
                metrics["card_entropy"] += card_entropy_mean.item()
                metrics["card_entropy_active"] += active_card_ent.item()
                metrics["tile_entropy"] += tile_entropy_mean.item()
                metrics["approx_kl"] += approx_kl.item()
                metrics["anchor_kl"] += float(anchor_kl.detach())
                num_updates += 1

                if (step_idx + 1) % accum_steps == 0 or (end >= total_samples):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

        self.update_count += 1

        # Critic warmup exit: release the actor once the critic has settled
        # below the target loss, or when the warmup budget is used up.
        if self._critic_warmup_active:
            warm_v_loss = metrics["v_loss"] / max(1, num_updates)
            if self.update_count >= self.critic_warmup_updates or warm_v_loss < CRITIC_WARMUP_TARGET_LOSS:
                self._critic_warmup_active = False
                print(f"[HybridPPOTrainer] Critic warmup complete after {self.update_count} update(s) "
                      f"(v_loss={warm_v_loss:.4f}). Actor released.")

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

        # Dynamic adaptation of the BC anchor KL leash beta, against the
        # *progressive* KL target (see `_current_kl_target`).
        if self.bc_reference_model is not None and not critic_warmup:
            mean_anchor_kl = avg_metrics.get("anchor_kl", 0.0)
            target = self._current_kl_target()
            beta_cap = self._current_beta_cap()
            if mean_anchor_kl > 1.5 * target:
                self.bc_kl_beta = min(beta_cap, self.bc_kl_beta * 1.5)
            elif mean_anchor_kl < (target / 1.5):
                self.bc_kl_beta = max(0.1, self.bc_kl_beta / 1.5)
        avg_metrics["bc_kl_beta"] = float(self.bc_kl_beta)
        avg_metrics["bc_kl_target"] = float(self._current_kl_target())

        avg_metrics.update({
            "total_matches": total_m,
            "total_wins": total_p1,
            "total_losses": total_p2,
            "total_draws": total_d,
            "win_rate": win_rate,
            "critic_warmup": 1.0 if self._critic_warmup_active else 0.0,
            "bc_initialised": 1.0 if self._bc_initialised else 0.0,
            "batch_matches": batch_m,
            "batch_wins": batch_p1,
            "batch_losses": batch_p2,
            "batch_draws": batch_d,
            "batch_win_rate": batch_win_rate,
            "tactical_zones": dict(self.batch_tactical_zones),
            "tactical_zone_pcts": zone_pcts,
            "distinct_tiles": len(self.batch_tiles),
            "top_tile_share": round(
                max(self.batch_tiles.values()) / max(1, sum(self.batch_tiles.values())), 4
            ),
        })

        # Append to telemetry history and generate obsidian graph
        history_entry = {
            "step": self.global_step,
            "update": self.update_count,
            "pg_loss": float(avg_metrics.get("pg_loss", 0.0)),
            "v_loss": float(avg_metrics.get("v_loss", 0.0)),
            "entropy": float(avg_metrics.get("entropy", 0.0)),
            "anchor_kl": float(avg_metrics.get("anchor_kl", 0.0)),
            "bc_kl_beta": float(avg_metrics.get("bc_kl_beta", 1.0)),
            "critic_warmup": float(avg_metrics.get("critic_warmup", 0.0)),
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
            "distinct_tiles": len(self.batch_tiles),
            "top_tile_share": float(avg_metrics.get("top_tile_share", 0.0)),
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
            "bc_warmstart": False,
            "bc_initialised": self._bc_initialised,
            "critic_warmup_active": self._critic_warmup_active,
            "bc_kl_beta": self.bc_kl_beta,
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
                self.model.load_state_dict(adapt_legacy_state_dict(ckpt["model_state_dict"]))
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scaler_state_dict" in ckpt and bool(ckpt["scaler_state_dict"]) and self.use_amp:
                    self.scaler.load_state_dict(ckpt["scaler_state_dict"])
                self.global_step = ckpt.get("global_step", 0)
                self.update_count = ckpt.get("update_count", 0)
                self.total_matches = ckpt.get("total_matches", self.total_matches)
                self.total_p1_wins = ckpt.get("total_p1_wins", self.total_p1_wins)
                self.total_p2_wins = ckpt.get("total_p2_wins", self.total_p2_wins)
                self.total_draws = ckpt.get("total_draws", self.total_draws)
                if "bc_kl_beta" in ckpt:
                    self.bc_kl_beta = float(ckpt["bc_kl_beta"])

                # If this checkpoint was saved right after BC warmstart, or has warmup active:
                is_bc = bool(ckpt.get("bc_warmstart") or ckpt.get("bc_initialised") or self.update_count == 0)
                if is_bc:
                    warmup_flag = ckpt.get("critic_warmup_active", (self.update_count < self.critic_warmup_updates))
                    self._setup_bc_reference_and_warmup(
                        state_dict=ckpt.get("bc_reference_state_dict") or ckpt["model_state_dict"],
                        critic_warmup_active=warmup_flag
                    )
                print(f"[HybridPPOTrainer] Loaded latest checkpoint from Step {self.global_step} (Update {self.update_count}) | Record: {self.total_p1_wins}W-{self.total_p2_wins}L-{self.total_draws}D ({self.total_matches} matches)")
            except Exception as e:
                print(f"[HybridPPOTrainer] Warning: could not load checkpoint: {e}")
