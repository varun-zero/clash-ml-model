"""
AlphaStar Stage-1 Imitation Warm-Start (Behavioral Cloning).
Simulates teacher bots headless, records (obs, action) pairs in canonical coordinates,
and trains AlphaStarCRNet with supervised class-weighted cross-entropy before RL begins.

The default teacher is ``TacticalCoachBot``: defence-first, spell-disciplined and
placement-varied, so the cloned policy does not inherit the spell-waste,
push-instead-of-defend and same-three-tiles behaviour of the old scripted bots.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from cr_alphastar_agent.config import (
    get_hardware_profile, ARENA_WIDTH, ARENA_HEIGHT, TOTAL_TILES,
    MAX_ENTITIES, D_MODEL, BC_MATCHES, BC_EPOCHS,
    CURRENT_DECK_IDS, resolve_deck_ids, resolve_deck_names
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.env.replica_env import ClashRoyaleReplicaEnv
from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv
from cr_alphastar_agent.env.scripted_bot import ScriptedBot, TacticalProBot
from cr_alphastar_agent.env.tactical_coach import TacticalCoachBot, _make_cycle_bot

TEACHERS = {
    "coach": TacticalCoachBot,
    "cycle": _make_cycle_bot,
    "tactical": TacticalProBot,
    "scripted": ScriptedBot,
}

# Weight of the critic calibration term in the behaviour-cloning loss.
# Regressing the value head onto the discounted return-to-go during BC keeps
# the critic usable from the very first RL update.
BC_VALUE_COEF = 0.5
# Discount used for the recorded return-to-go targets.
BC_RETURN_GAMMA = 0.995


def collect_demonstration_data(num_matches: int = BC_MATCHES, max_steps_per_match: int = 400,
                               env_backend: str = "crsim", bot_cls = TacticalCoachBot,
                               deck: Optional[List[int]] = None,
                               opponent_bot_cls=None,
                               seed: int = 42,
                               gamma: float = 0.995
                               ) -> List[Tuple[Dict[str, np.ndarray], np.ndarray, float]]:
    """Runs bot vs bot games and captures (obs, action, return-to-go) triples.

    The discounted return-to-go ``G_t = sum_k gamma^k r_{t+k}`` is recorded so
    the critic can be calibrated during behaviour cloning.  An uncalibrated
    ``value_head`` produces noisy GAE advantages, which is what pushes the
    actor into a random policy right after the clone.
    """
    if env_backend == "crsim":
        env = CrSimReplicaEnv(seed=seed, blue_deck=deck, red_deck=deck)
    else:
        env = ClashRoyaleReplicaEnv(seed=seed)
    bot1 = bot_cls(player_id=1)
    bot2 = (opponent_bot_cls or bot_cls)(player_id=2)

    dataset: List[Tuple[Dict[str, np.ndarray], np.ndarray, float]] = []
    print(f"[WarmStart BC] Generating {num_matches} matches of {bot_cls.__name__} demonstrations...")

    for match in range(num_matches):
        obs_p1, _ = env.reset()
        # Per-seat buffers for one match.
        seat_rows = {1: [], 2: []}
        seat_rewards = {1: [], 2: []}

        for step in range(max_steps_per_match):
            obs_p2 = env.get_obs(player_id=2)

            act_p1 = bot1.act(obs_p1)
            act_p2 = bot2.act(obs_p2)

            # Record non-trivial decisions or balanced sample of no-ops
            # Keep all active card plays (slot != 4) and subsample No-Ops (~5%) so Wait holds ~35-45% probability on affordable states
            if act_p1[0] != 4 or np.random.rand() < 0.05:
                seat_rows[1].append((copy_obs(obs_p1), act_p1.copy(), step))
            if act_p2[0] != 4 or np.random.rand() < 0.05:
                seat_rows[2].append((copy_obs(obs_p2), act_p2.copy(), step))

            next_obs_p1, _, r1, r2, term, trunc, _ = env.step_both(act_p1, act_p2)
            # Rewards are recorded per timestep so the return-to-go can be
            # computed backwards over the whole match.
            seat_rewards[1].append(float(r1))
            seat_rewards[2].append(float(r2))
            obs_p1 = next_obs_p1
            if term or trunc:
                break

        for seat in (1, 2):
            rows = seat_rows[seat]
            rewards = seat_rewards[seat]
            if not rows:
                continue
            # Discounted return-to-go over the match timeline; each recorded
            # decision uses the value of the step it was taken on.
            returns = _returns_to_go(rewards, gamma)
            for obs, act, step in rows:
                idx = min(int(step), len(returns) - 1)
                dataset.append((obs, act, float(returns[idx])))

    print(f"[WarmStart BC] Collected {len(dataset)} transition pairs "
          f"(with discounted return-to-go targets).")
    return dataset


def _returns_to_go(rewards: List[float], gamma: float) -> List[float]:
    """Discounted return-to-go for every timestep of an episode."""
    out = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def copy_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in obs.items()}


def collate_fn(batch: List[Tuple[Dict[str, np.ndarray], np.ndarray, float]],
               device: torch.device):
    """Collate (obs, action, return-to-go) triples into batched tensors."""
    obs_list, act_list, ret_list = zip(*batch)
    batched_obs = {
        k: torch.stack([torch.from_numpy(o[k]) for o in obs_list]).to(device)
        for k in obs_list[0]
    }
    batched_actions = torch.stack([torch.from_numpy(a) for a in act_list]).to(device)
    batched_returns = torch.stack(
        [torch.tensor(float(r), dtype=torch.float32) for r in ret_list]).to(device)
    return batched_obs, batched_actions, batched_returns


def run_warmstart(num_matches: int = BC_MATCHES, epochs: int = BC_EPOCHS, batch_size: int = 64,
                  lr: float = 1e-3, checkpoint_dir: str = "./checkpoints", env_backend: str = "crsim",
                  tactical_anchors: bool = False, deck: Optional[List[int]] = None,
                  teacher: str = "coach", opponent_teacher: Optional[str] = None,
                  init_checkpoint: Optional[str] = None, out_name: str = "latest.pt"):
    hw = get_hardware_profile()
    device = torch.device(hw["device"])
    print(f"[WarmStart BC] Hardware: {hw['profile_name']} ({device})")
    print(f"[WarmStart BC] Backend: {env_backend.upper()}")
    if deck is not None:
        print(f"[WarmStart BC] Deck: {len(deck)} cards -> {list(deck)}")
    if tactical_anchors:
        print("[WarmStart BC] MODE: --tactical-anchors (Fine-tuning spatial anchor points with TacticalProBot)")

    os.makedirs(checkpoint_dir, exist_ok=True)
    bot_cls = TEACHERS.get(teacher, TacticalCoachBot)
    opp_cls = TEACHERS.get(opponent_teacher) if opponent_teacher else None
    print(f"[WarmStart BC] Teacher: {bot_cls.__name__}"
          + (f" | Opponent teacher: {opp_cls.__name__}" if opp_cls else ""))
    dataset = collect_demonstration_data(num_matches=num_matches, env_backend=env_backend,
                                         bot_cls=bot_cls, deck=deck, opponent_bot_cls=opp_cls,
                                         gamma=BC_RETURN_GAMMA)
    if not dataset:
        print("[WarmStart BC] No data collected!")
        return

    model = AlphaStarCRNet().to(device)

    # If tactical_anchors mode, load existing checkpoint to preserve RL policy knowledge
    saved_step = 0
    saved_update = 0
    saved_matches = 0
    saved_w = 0
    saved_l = 0
    saved_d = 0
    latest_path = os.path.join(checkpoint_dir, out_name)

    resume_path = init_checkpoint or (latest_path if tactical_anchors else None)
    if resume_path and os.path.exists(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            saved_step = ckpt.get("global_step", 0)
            saved_update = ckpt.get("update_count", 0)
            saved_matches = ckpt.get("total_matches", 0)
            saved_w = ckpt.get("total_p1_wins", 0)
            saved_l = ckpt.get("total_p2_wins", 0)
            saved_d = ckpt.get("total_draws", 0)
            print(f"[WarmStart BC] Loaded existing policy from Step {saved_step:,} (Update {saved_update}).")
        except Exception as e:
            print(f"[WarmStart BC] Warning loading latest checkpoint: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_samples = len(dataset)
    indices = np.arange(total_samples)

    # Class weights for card_head
    card_class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.3], device=device)

    print(f"[WarmStart BC] Training AlphaStarCRNet for {epochs} epochs (samples: {total_samples}, lr: {lr})...")
    if tactical_anchors:
        print(f"[WarmStart BC] Emphasizing tile_head spatial cross-entropy on back-of-king, kite, and bridge tank-trick coordinates.")

    model.train()
    for ep in range(epochs):
        np.random.shuffle(indices)
        ep_loss = 0.0
        ep_value_loss = 0.0
        batches = 0

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_data = [dataset[i] for i in indices[start:end]]
            b_obs, b_act, b_returns = collate_fn(batch_data, device)

            optimizer.zero_grad()

            # Forward core
            core = model.forward_core(
                b_obs["scalars"], b_obs["hand_cards"], b_obs["entities"],
                b_obs["entity_mask"], b_obs["arena_grid"]
            )

            # Head 1: Card slot prediction with class weights
            card_logits = model.card_head(core).float()
            card_mask = b_obs["card_mask"].bool()
            masked_card_logits = card_logits.masked_fill(~card_mask, -1e9)
            target_card = b_act[:, 0]

            # A teacher action must be legal for the card slot it names; if it
            # is not, the masked cross-entropy would explode.
            card_target_ok = card_mask.gather(
                1, target_card.clamp(0, 4).view(-1, 1)).squeeze(-1)
            card_loss = F.cross_entropy(
                masked_card_logits, target_card, weight=card_class_weights,
                reduction="none")
            card_loss = (card_loss * card_target_ok.float()).sum() / \
                max(1.0, card_target_ok.float().sum())

            # Head 2: Autoregressive Tile prediction conditioned on target card
            chosen_emb = model.chosen_action_emb(target_card)
            tile_input = torch.cat([core, chosen_emb], dim=-1)
            tile_logits = model.tile_head(tile_input).float()

            # Card-conditioned tile mask.
            # The observation carries a per-card mask (card_tile_mask); using
            # it (instead of the union mask) keeps the training-time mask
            # identical to the mask the policy sees at inference.
            hand_cards = b_obs["hand_cards"][:, :4]
            clamped_slot = target_card.clamp(0, 3).unsqueeze(-1)
            chosen_card_id = torch.gather(hand_cards, 1, clamped_slot).squeeze(-1)
            is_spell = torch.isin(chosen_card_id, model.spell_ids) & (target_card != 4)
            full_arena_mask = torch.ones_like(b_obs["tile_mask"], dtype=torch.bool)

            card_tile_mask = b_obs.get("card_tile_mask")
            if card_tile_mask is not None:
                slots = target_card.clamp(0, 4).view(-1, 1, 1).expand(-1, 1, TOTAL_TILES)
                effective_tile_mask = torch.gather(
                    card_tile_mask.bool(), 1, slots).squeeze(1)
            else:
                effective_tile_mask = torch.where(
                    is_spell.unsqueeze(-1), full_arena_mask, b_obs["tile_mask"].bool())

            masked_tile_logits = tile_logits.masked_fill(~effective_tile_mask, -1e9)
            target_tile = b_act[:, 1]

            # Tile loss only applies to active card play actions (target_card != 4), weighted by 3.0 for tactical anchors
            is_play_action = (target_card != 4)
            # Drop rows whose teacher target is not actually legal for that
            # card: training against a masked target would give the teacher
            # tile a ~1e9 loss and destroy the run.
            target_legal = effective_tile_mask.gather(1, target_tile.clamp(0, TOTAL_TILES - 1).view(-1, 1)).squeeze(-1)
            is_play_action = is_play_action & target_legal
            tile_weight = 3.0 if tactical_anchors else 2.0
            if is_play_action.any():
                tile_loss = tile_weight * F.cross_entropy(
                    masked_tile_logits[is_play_action],
                    target_tile[is_play_action]
                )
            else:
                tile_loss = torch.tensor(0.0, device=device)

            # Head 3: Critic calibration.
            # The value head is regressed onto the discounted return-to-go so
            # that the critic is already calibrated when RL starts.  Without
            # this the first PPO updates see a value loss of ~0.17, the GAE
            # advantages are dominated by critic noise and the actor is
            # pushed towards a random policy before it can learn anything.
            values = model.value_head(core).squeeze(-1).float()
            value_loss = F.mse_loss(values, b_returns)

            loss = card_loss + tile_loss + BC_VALUE_COEF * value_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            ep_loss += loss.item()
            ep_value_loss += float(value_loss.detach())
            batches += 1

        avg_loss = ep_loss / max(1, batches)
        avg_value_loss = ep_value_loss / max(1, batches)
        print(f"  [Epoch {ep+1}/{epochs}] Supervised Anchor Loss: {avg_loss:.4f} "
              f"(value calibration MSE: {avg_value_loss:.4f})")

    use_cuda_amp = hw["fp16"] and hw["device"] == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

    # Save to latest.pt, preserving RL progress if in tactical_anchors mode
    torch.save({
        "global_step": saved_step,
        "update_count": saved_update,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if use_cuda_amp else {},
        "total_matches": saved_matches,
        "total_p1_wins": saved_w,
        "total_p2_wins": saved_l,
        "total_draws": saved_d,
        "bc_warmstart": True,
        "bc_initialised": True,
        "critic_warmup_active": True,
    }, latest_path)

    # Save a dedicated reference copy for the BC Anchor KL leash
    ref_path = os.path.join(checkpoint_dir, "bc_reference.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "deck": deck,
    }, ref_path)
    print(f"[WarmStart BC Complete] Model successfully saved to {latest_path} (Global Step {saved_step:,})")
    print(f"[WarmStart BC Complete] Reference anchor copy saved to {ref_path}")


def main():
    parser = argparse.ArgumentParser(description="AlphaStar Stage-1 Behavioral Cloning Warm-Start")
    parser.add_argument("--smoke-test", action="store_true", help="Quick smoke test mode (10 matches, 2 epochs)")
    parser.add_argument("--tactical-anchors", action="store_true", help="Fine-tune spatial head with TacticalProBot demonstrations (50 matches, 2 epochs, lr=1e-4)")
    parser.add_argument("--num-matches", type=int, default=None, help=f"Number of simulated matches (default: {BC_MATCHES})")
    parser.add_argument("--epochs", type=int, default=None, help=f"Number of supervised training epochs (default: {BC_EPOCHS})")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-3, tactical-anchors: 1e-4)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Path to checkpoint directory")
    parser.add_argument("--env-backend", type=str, default="crsim", choices=["crsim", "replica"],
                        help="Physics engine backend (crsim = 100%% APK data-driven, replica = lightweight)")
    parser.add_argument("--deck", type=str, default="current",
                        help="Deck preset or comma-separated card names "
                             "(default: current = Knight, Mini PEKKA, Archers, Musketeer, "
                             "Arrows, Fireball, Giant, Minions)")
    parser.add_argument("--teacher", type=str, default="coach", choices=sorted(TEACHERS),
                        help="Behaviour-cloning teacher bot (default: coach = defence-first, "
                             "spell-disciplined, placement-varied)")
    parser.add_argument("--opponent-teacher", type=str, default=None, choices=sorted(TEACHERS),
                        help="Optional different teacher for the opposing seat")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Warm start from this checkpoint instead of random weights")
    parser.add_argument("--out-name", type=str, default="latest.pt",
                        help="Output checkpoint file name inside --checkpoint-dir")
    args = parser.parse_args()

    if args.deck.strip() == "none":
        deck_ids = None
    elif "," in args.deck:
        deck_ids = resolve_deck_names([s.strip() for s in args.deck.split(",") if s.strip()])
    else:
        deck_ids = resolve_deck_names(args.deck)

    if args.tactical_anchors:
        num_matches = args.num_matches or 50
        epochs = args.epochs or 2
        lr = args.lr or 1e-4
        print(f"[WarmStart BC] Running --tactical-anchors fine-tuning: {num_matches} matches, {epochs} epochs, lr={lr}")
    elif args.smoke_test:
        num_matches = args.num_matches or 10
        epochs = args.epochs or 2
        lr = args.lr or 1e-3
        print(f"[WarmStart BC] Running in --smoke-test mode: {num_matches} matches, {epochs} epochs.")
    else:
        num_matches = args.num_matches or BC_MATCHES
        epochs = args.epochs or BC_EPOCHS
        lr = args.lr or 1e-3

    run_warmstart(num_matches=num_matches, epochs=epochs, lr=lr, checkpoint_dir=args.checkpoint_dir,
                  env_backend=args.env_backend, tactical_anchors=args.tactical_anchors, deck=deck_ids,
                  teacher=args.teacher, opponent_teacher=args.opponent_teacher,
                  init_checkpoint=args.init_checkpoint, out_name=args.out_name)


if __name__ == "__main__":
    main()
