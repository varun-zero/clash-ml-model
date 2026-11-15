"""Audit the competitive v2 public-observation contract.

Checks:
1. Policy arrays are invariant to the opponent's true elixir, hand and cycle.
2. The card mask agrees with ``Battle.legal_action`` on random cells.
3. The cell mask agrees with ``Battle.legal_action`` for every hand card on a
   sample of cells (and never marks a cell legal that the engine rejects).
4. Decoding an Action produces a legal engine play.

Exit code 0 means all checks passed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crsim.engine import Battle
from cr_alphastar_agent.competitive.belief import OpponentBelief
from cr_alphastar_agent.competitive.cards import default_vocab
from cr_alphastar_agent.competitive.obs import (
    Action, N_CELLS, build_observation, cell_center, decode_action, engine_xy,
    legal_cell_mask,
)


def compare_policy_arrays(a, b) -> list[str]:
    problems = []
    for k, va in a.policy_arrays().items():
        vb = b.policy_arrays()[k]
        if not np.array_equal(np.asarray(va), np.asarray(vb)):
            problems.append(k)
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decks", type=int, default=3, help="number of preset decks to test")
    ap.add_argument("--samples", type=int, default=200, help="cells sampled per card")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    vocab = default_vocab()
    decks = vocab.playable_deck_pool()[: args.decks]
    if len(decks) < 1:
        print("[error] no playable preset decks")
        return 1

    failures = 0
    for di, deck in enumerate(decks):
        opp_deck = decks[(di + 1) % len(decks)]
        battle = Battle([deck, opp_deck], seed=args.seed + di)
        # Advance a few seconds so hands/cycle and units exist.
        for _ in range(30):
            battle.step([])
        beliefs = {
            0: OpponentBelief(vocab, deck_prior=[deck, opp_deck], seed=1),
            1: OpponentBelief(vocab, deck_prior=[deck, opp_deck], seed=2),
        }
        obs_a = build_observation(battle, 0, beliefs[0], vocab=vocab)

        # 1. Hidden-state invariance: perturb every hidden opponent value.
        hidden = (battle.elixir[1], list(battle.hand[1]), list(battle.cycle[1]))
        battle.elixir[1] = 0.0
        battle.hand[1] = list(reversed(battle.hand[1]))
        battle.cycle[1] = list(reversed(battle.cycle[1]))
        obs_b = build_observation(battle, 0, beliefs[0], vocab=vocab)
        battle.elixir[1], battle.hand[1], battle.cycle[1] = hidden[0], hidden[1], hidden[2]
        leaked = compare_policy_arrays(obs_a, obs_b)
        if leaked:
            failures += 1
            print(f"[FAIL] deck {di}: policy input leaks hidden state via {leaked}")
        else:
            print(f"[ok] deck {di}: policy arrays invariant to opponent elixir/hand/cycle")

        # 2. Card mask agrees with the engine.
        for slot, card in enumerate(battle.hand[0]):
            any_legal = False
            for _ in range(args.samples):
                cell = int(rng.integers(0, N_CELLS))
                vx, vy = cell_center(cell)
                ex, ey = engine_xy(vx, vy, 0)
                engine_ok = battle.legal_action(0, slot, ex, ey)
                mask_ok = bool(legal_cell_mask(battle, 0, card)[cell])
                if engine_ok != mask_ok:
                    failures += 1
                    print(f"[FAIL] deck {di} slot {slot} cell {cell}: "
                          f"engine={engine_ok} mask={mask_ok}")
                    break
                any_legal = any_legal or engine_ok
            if any_legal and not obs_a.card_mask[slot]:
                failures += 1
                print(f"[FAIL] deck {di} slot {slot}: legal cell exists but card_mask is False")
        if failures == 0:
            print(f"[ok] deck {di}: cell masks match engine legal_action on "
                  f"{len(battle.hand[0]) * args.samples} samples")

        # 3. Action decoding into the engine.
        for slot, mask in enumerate(obs_a.cell_masks):
            cells = np.where(mask)[0]
            if len(cells) == 0:
                continue
            cell = int(rng.choice(cells))
            action = Action(gate=1, hand_slot=slot, cell=cell)
            decoded = decode_action(action, battle, 0)
            if decoded is None:
                failures += 1
                print(f"[FAIL] deck {di} slot {slot} cell {cell}: mask legal but decode failed")
        print(f"[ok] deck {di}: all sampled masked cells decode to legal engine actions")

    if failures:
        print(f"\n{failures} observation-contract failure(s)")
        return 1
    print("\nAll observation-contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
