"""Bootstrap a competitive v2 model with behaviour cloning.

Example:
    python tools/train_bc.py --dataset data/pro_bc.npz \
        --out checkpoints/competitive/bc_init.pt --epochs 3

Then start PPO from the result:
    python train_competitive.py --config configs/competitive.yaml \
        --init-bc checkpoints/competitive/bc_init.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.competitive.bc import (
    BCConfig, build_model_from_config, save_bc_checkpoint, train_bc,
)
from cr_alphastar_agent.competitive.cards import default_vocab
from cr_alphastar_agent.competitive.replay import CompetitiveDataset, load_dataset


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--history", default=None, help="write per-epoch metrics JSON")
    args = ap.parse_args(argv)

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = default_vocab()
    dataset = CompetitiveDataset(load_dataset(args.dataset))
    print(f"[data] {len(dataset)} examples")
    model = build_model_from_config(
        {"d_model": args.d_model, "layers": args.layers}, vocab.n_cards,
    ).to(device)
    cfg = BCConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                   seed=args.seed)
    result = train_bc(model, dataset, cfg, device)
    save_bc_checkpoint(args.out, model, vocab, result)
    print(f"[done] wrote {args.out}")
    if args.history:
        Path(args.history).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
