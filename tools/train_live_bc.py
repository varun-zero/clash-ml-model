"""Turn real (ladder) matches into a behaviour-cloning dataset and fine-tune.

Real matches played by the deployed agent are recorded as JSONL by
``cr_alphastar_agent.env.live_recorder``.  This tool converts them into a
supervised dataset and continues training the policy on them, so every real
match improves the model.

Usage:
    python tools/train_live_bc.py --replays data/live --checkpoint checkpoints/latest.pt \
        --out checkpoints/live_bc.pt --epochs 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cr_alphastar_agent.config import (  # noqa: E402
    ARENA_WIDTH, TOTAL_TILES, get_hardware_profile,
)
from cr_alphastar_agent.env.live_recorder import load_live_replays, rows_to_dataset  # noqa: E402
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet  # noqa: E402


def build_tensors(dataset: dict, device: torch.device) -> tuple[dict, torch.Tensor]:
    obs = {
        "scalars": torch.from_numpy(dataset["scalars"]).float().to(device),
        "hand_cards": torch.from_numpy(dataset["hand_cards"]).long().to(device),
        "entities": torch.from_numpy(dataset["entities"]).float().to(device),
        "entity_mask": torch.from_numpy(dataset["entity_mask"]).bool().to(device),
        "arena_grid": torch.from_numpy(dataset["arena_grid"]).float().to(device),
        "card_mask": torch.from_numpy(dataset["card_mask"]).bool().to(device),
        "tile_mask": torch.from_numpy(dataset["tile_mask"]).bool().to(device),
    }
    if "card_tile_mask" in dataset:
        obs["card_tile_mask"] = torch.from_numpy(dataset["card_tile_mask"]).bool().to(device)
    actions = torch.from_numpy(dataset["action"]).long().to(device)
    return obs, actions


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the policy on real recorded matches")
    ap.add_argument("--replays", nargs="+", required=True,
                    help="JSONL file(s) or directory of recorded live matches")
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt",
                    help="Checkpoint to fine-tune from")
    ap.add_argument("--out", default="checkpoints/live_bc.pt",
                    help="Where to write the fine-tuned checkpoint")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="Only train on rows with at least this many playable options")
    args = ap.parse_args()

    hw = get_hardware_profile()
    device = torch.device(args.device or hw["device"])
    print(f"[Live BC] Device: {device}")

    rows = load_live_replays(args.replays)
    print(f"[Live BC] Loaded {len(rows)} recorded decisions")
    if not rows:
        print("[Live BC] Nothing to train on.")
        return

    dataset = rows_to_dataset(rows)
    if not dataset:
        print("[Live BC] Recorded rows did not contain usable observations.")
        return
    n = len(dataset["action"])
    print(f"[Live BC] Dataset: {n} samples")

    model = AlphaStarCRNet().to(device)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = payload.get("model_state_dict", payload)
        model.load_state_dict(state)
        print(f"[Live BC] Initialised from {ckpt_path}")
    else:
        print(f"[Live BC] WARNING: {ckpt_path} not found, starting from scratch")

    obs, actions = build_tensors(dataset, device)
    card_mask = obs["card_mask"]
    tile_mask = obs.get("card_tile_mask")
    if tile_mask is None:
        base = obs["tile_mask"]
        tile_mask = base.unsqueeze(1).expand(-1, 5, -1).clone()
        obs["card_tile_mask"] = tile_mask

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    indices = np.arange(n)
    model.train()

    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        total = 0.0
        seen = 0
        for start in range(0, n, args.batch_size):
            idx = indices[start:start + args.batch_size]
            mb_obs = {k: v[idx] for k, v in obs.items()}
            mb_actions = actions[idx]

            optimiser.zero_grad()
            core = model.forward_core(
                mb_obs["scalars"], mb_obs["hand_cards"], mb_obs["entities"],
                mb_obs["entity_mask"], mb_obs["arena_grid"],
            )

            card_logits = model.card_head(core).float()
            cmask = mb_obs["card_mask"]
            masked = card_logits.masked_fill(~cmask, -1e9)
            target_card = mb_actions[:, 0]
            card_ok = cmask.gather(1, target_card.clamp(0, 4).view(-1, 1)).squeeze(-1)
            card_loss = F.cross_entropy(masked, target_card, reduction="none")
            card_loss = (card_loss * card_ok.float()).sum() / max(1.0, card_ok.float().sum())

            chosen_emb = model.chosen_action_emb(target_card)
            tile_logits = model.tile_head(torch.cat([core, chosen_emb], dim=-1)).float()
            slots = target_card.clamp(0, 4).view(-1, 1, 1).expand(-1, 1, TOTAL_TILES)
            effective = torch.gather(mb_obs["card_tile_mask"].bool(), 1, slots).squeeze(1)
            masked_tile = tile_logits.masked_fill(~effective, -1e9)
            target_tile = mb_actions[:, 1]
            legal = effective.gather(1, target_tile.clamp(0, TOTAL_TILES - 1).view(-1, 1)).squeeze(-1)
            play = (target_card != 4) & legal
            if play.any():
                tile_loss = 2.0 * F.cross_entropy(masked_tile[play], target_tile[play])
            else:
                tile_loss = torch.zeros((), device=device)

            loss = card_loss + tile_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimiser.step()
            total += float(loss.detach())
            seen += 1
        print(f"[Live BC] epoch {epoch + 1}/{args.epochs} loss={total / max(1, seen):.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "source": "live_bc"}
    if ckpt_path.exists():
        original = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for key in ("global_step", "update_count", "total_matches", "total_p1_wins",
                    "total_p2_wins", "total_draws"):
            if key in original:
                payload[key] = original[key]
    torch.save(payload, out_path)
    print(f"[Live BC] Saved {out_path}")


if __name__ == "__main__":
    main()
