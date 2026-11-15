"""Build a behaviour-cloning dataset from local replays or public HF parquet.

Examples:
    python tools/build_pro_dataset.py --replays replays --out data/pro_bc.npz
    python tools/build_pro_dataset.py --hf-parquet data/hf_part_0.parquet \
        --limit 2000 --out data/hf_bc.npz

The output is a compressed ``.npz`` consumed by ``tools/train_bc.py`` or
``train_competitive.py --init-bc``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.competitive.cards import default_vocab
from cr_alphastar_agent.competitive.replay import (
    build_dataset_from_records,
    build_examples_for_team,
    examples_to_arrays,
    iter_hf_parquet,
    load_local_replays,
    parse_hf_payload,
    save_dataset,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--replays", help="directory of local JSON replays")
    src.add_argument("--hf-parquet", nargs="+", help="HF parquet part(s)")
    ap.add_argument("--out", required=True, help="output .npz dataset")
    ap.add_argument("--limit", type=int, default=None, help="max replays to ingest")
    ap.add_argument("--wait-prob", type=float, default=0.35,
                    help="probability of emitting a wait example on the 500 ms grid")
    ap.add_argument("--teams", default="both", choices=("both", "0", "1"),
                    help="which seat(s) to imitate")
    args = ap.parse_args(argv)

    vocab = default_vocab()
    records = []
    if args.replays:
        records = load_local_replays(args.replays, vocab=vocab, limit=args.limit)
        print(f"[read] {len(records)} local replays")
    else:
        for tag, payload in iter_hf_parquet(args.hf_parquet, limit=args.limit):
            rec = parse_hf_payload(payload, tag=tag, vocab=vocab)
            if rec is not None:
                records.append(rec)
            if len(records) % 500 == 0 and len(records):
                print(f"[read] {len(records)} parquet replays")
        print(f"[read] {len(records)} parquet replays")

    if not records:
        print("[error] no usable replays (check card names / deck completeness)")
        return 1

    teams = (0, 1) if args.teams == "both" else (int(args.teams),)
    examples = []
    for i, rec in enumerate(records):
        for team in teams:
            examples.extend(build_examples_for_team(
                rec, team, vocab=vocab, wait_prob=args.wait_prob,
                seed=i * 7 + team,
            ))
        if (i + 1) % 200 == 0:
            print(f"[build] {i + 1}/{len(records)} replays, {len(examples)} examples")
    if not examples:
        print("[error] replay parsing produced no examples")
        return 1

    arrays = examples_to_arrays(examples, vocab=vocab)
    save_dataset(args.out, arrays)
    print(f"[done] {len(examples)} examples -> {args.out}")
    print(f"       keys: {', '.join(sorted(arrays))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
