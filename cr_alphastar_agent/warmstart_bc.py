"""
AlphaStar Stage-1 Imitation Warm-Start (Behavioral Cloning).
Simulates ScriptedBot vs ScriptedBot headless matches, records (obs, action) pairs in canonical coordinates,
and trains AlphaStarCRNet with supervised class-weighted cross-entropy before RL begins.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

from cr_alphastar_agent.config import (
    get_hardware_profile, ARENA_WIDTH, ARENA_HEIGHT, TOTAL_TILES,
    MAX_ENTITIES, D_MODEL, BC_MATCHES, BC_EPOCHS
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.env.replica_env import ClashRoyaleReplicaEnv
from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv
from cr_alphastar_agent.env.scripted_bot import ScriptedBot, TacticalProBot


def collect_demonstration_data(num_matches: int = BC_MATCHES, max_steps_per_match: int = 400,
                               env_backend: str = "crsim", bot_cls = ScriptedBot) -> List[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    """Runs bot vs bot games and captures expert decision pairs."""
    if env_backend == "crsim":
        env = CrSimReplicaEnv(seed=42)
    else:
        env = ClashRoyaleReplicaEnv(seed=42)
    bot1 = bot_cls(player_id=1)
    bot2 = bot_cls(player_id=2)

    dataset = []
    print(f"[WarmStart BC] Generating {num_matches} matches of {bot_cls.__name__} demonstrations...")

    for match in range(num_matches):
        obs_p1, _ = env.reset()
        for step in range(max_steps_per_match):
            obs_p2 = env.get_obs(player_id=2)

            act_p1 = bot1.act(obs_p1)
            act_p2 = bot2.act(obs_p2)

            # Record non-trivial decisions or balanced sample of no-ops
            # Keep all active card plays (slot != 4) and subsample No-Ops (~5%) so Wait holds ~35-45% probability on affordable states
            if act_p1[0] != 4 or np.random.rand() < 0.05:
                dataset.append((copy_obs(obs_p1), act_p1.copy()))
            if act_p2[0] != 4 or np.random.rand() < 0.05:
                dataset.append((copy_obs(obs_p2), act_p2.copy()))

            next_obs_p1, _, _, _, term, trunc, _ = env.step_both(act_p1, act_p2)
            obs_p1 = next_obs_p1
            if term or trunc:
                break

    print(f"[WarmStart BC] Collected {len(dataset)} transition pairs.")
    return dataset


def copy_obs(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in obs.items()}


def collate_fn(batch: List[Tuple[Dict[str, np.ndarray], np.ndarray]], device: torch.device):
    obs_list, act_list = zip(*batch)
    batched_obs = {
        k: torch.stack([torch.from_numpy(o[k]) for o in obs_list]).to(device)
        for k in obs_list[0]
    }
    batched_actions = torch.stack([torch.from_numpy(a) for a in act_list]).to(device)
    return batched_obs, batched_actions


def run_warmstart(num_matches: int = BC_MATCHES, epochs: int = BC_EPOCHS, batch_size: int = 64,
                  lr: float = 1e-3, checkpoint_dir: str = "./checkpoints", env_backend: str = "crsim",
                  tactical_anchors: bool = False):
    hw = get_hardware_profile()
    device = torch.device(hw["device"])
    print(f"[WarmStart BC] Hardware: {hw['profile_name']} ({device})")
    print(f"[WarmStart BC] Backend: {env_backend.upper()}")
    if tactical_anchors:
        print("[WarmStart BC] MODE: --tactical-anchors (Fine-tuning spatial anchor points with TacticalProBot)")

    os.makedirs(checkpoint_dir, exist_ok=True)
    bot_cls = TacticalProBot if tactical_anchors else ScriptedBot
    dataset = collect_demonstration_data(num_matches=num_matches, env_backend=env_backend, bot_cls=bot_cls)
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
    latest_path = os.path.join(checkpoint_dir, "latest.pt")

    if tactical_anchors and os.path.exists(latest_path):
        try:
            ckpt = torch.load(latest_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            saved_step = ckpt.get("global_step", 0)
            saved_update = ckpt.get("update_count", 0)
            saved_matches = ckpt.get("total_matches", 0)
            saved_w = ckpt.get("total_p1_wins", 0)
            saved_l = ckpt.get("total_p2_wins", 0)
            saved_d = ckpt.get("total_draws", 0)
            print(f"[WarmStart BC] Loaded existing policy from Step {saved_step:,} (Update {saved_update}) for spatial anchoring.")
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
        batches = 0

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_data = [dataset[i] for i in indices[start:end]]
            b_obs, b_act = collate_fn(batch_data, device)

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

            card_loss = F.cross_entropy(masked_card_logits, target_card, weight=card_class_weights)

            # Head 2: Autoregressive Tile prediction conditioned on target card
            chosen_emb = model.chosen_action_emb(target_card)
            tile_input = torch.cat([core, chosen_emb], dim=-1)
            tile_logits = model.tile_head(tile_input).float()

            # Card-conditioned tile mask
            hand_cards = b_obs["hand_cards"][:, :4]
            clamped_slot = target_card.clamp(0, 3).unsqueeze(-1)
            chosen_card_id = torch.gather(hand_cards, 1, clamped_slot).squeeze(-1)
            is_spell = torch.isin(chosen_card_id, model.spell_ids) & (target_card != 4)
            full_arena_mask = torch.ones_like(b_obs["tile_mask"], dtype=torch.bool)
            effective_tile_mask = torch.where(is_spell.unsqueeze(-1), full_arena_mask, b_obs["tile_mask"].bool())

            masked_tile_logits = tile_logits.masked_fill(~effective_tile_mask, -1e9)
            target_tile = b_act[:, 1]

            # Tile loss only applies to active card play actions (target_card != 4), weighted by 3.0 for tactical anchors
            is_play_action = (target_card != 4)
            tile_weight = 3.0 if tactical_anchors else 2.0
            if is_play_action.any():
                tile_loss = tile_weight * F.cross_entropy(
                    masked_tile_logits[is_play_action],
                    target_tile[is_play_action]
                )
            else:
                tile_loss = torch.tensor(0.0, device=device)

            loss = card_loss + tile_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            ep_loss += loss.item()
            batches += 1

        avg_loss = ep_loss / max(1, batches)
        print(f"  [Epoch {ep+1}/{epochs}] Supervised Anchor Loss: {avg_loss:.4f}")

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
    }, latest_path)
    print(f"[WarmStart BC Complete] Model successfully saved to {latest_path} (Global Step {saved_step:,})")


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
    args = parser.parse_args()

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

    run_warmstart(num_matches=num_matches, epochs=epochs, lr=lr, checkpoint_dir=args.checkpoint_dir, env_backend=args.env_backend, tactical_anchors=args.tactical_anchors)


if __name__ == "__main__":
    main()
