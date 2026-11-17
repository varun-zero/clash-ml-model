#!/usr/bin/env python3
"""Offline ground-truth verification for the live ClashBot bridge.

This script does **not** need MuMu Player, ADB or a live match.  It proves
the three invariants that the previous live runs silently violated:

1. **10-pip discrete elixir reader** - ``read_elixir`` must return the exact
   number of filled elixir pips (0..10), must ignore the purple drop icon
   that sits left of the bar (``x < 0.255``) and must not be fooled by the
   white elixir counter the game draws over the first pips.

2. **Ground-truth match outcome** - ``detect_match_end_result`` must read the
   opponent crown zone (``y 0.24..0.36``), the player crown zone
   (``y 0.51..0.63``), the opponent ribbon (``y 0.14..0.23``) and the player
   result ribbon (``y 0.41..0.51``).  It must never use the bottom blue
   ``OK`` button and it must **never** default to ``"WIN"``.

3. **4-card cycle + tactical mask invariants** - a card that was just played
   must not be predicted in hand for the next three plays, and the shared
   tactical masks (``cr_alphastar_agent.env.tactical_masks``) must satisfy
   the spell / win-condition / same-lane-defence / elixir-bank rules that
   ``tools/diagnose_truth.py`` checks.

Usage::

    python tools/verify_ground_truth.py

Exit code 0 means every check passed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy_mumu import (  # noqa: E402
    ELIXIR_ICON_MAX_X,
    ELIXIR_PIP_COUNT,
    PerceptionAdapter,
    detect_match_end_result,
)
from cr_alphastar_agent.config import ARENA_WIDTH, TOTAL_TILES  # noqa: E402
from cr_alphastar_agent.env.tactical_masks import (  # noqa: E402
    ELIXIR_BANK_THRESHOLD,
    EnemyUnit,
    SPELL_BLOCKED_TILE_Y,
    SPELL_MIN_TILE_Y,
    THREAT_TILE_Y,
    build_tactical_masks,
)

# Card ids used below (see cr_alphastar_agent.config.CARDS_CATALOG).
KNIGHT, ARCHERS, MINIONS = 1, 2, 7
ARROWS, FIREBALL = 9, 8
GIANT, HOG_RIDER = 3, 11
MUSKETEER = 4

FAILURES: List[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


# ---------------------------------------------------------------------------
# Synthetic frame builders (identical geometry to the real emulator capture)
# ---------------------------------------------------------------------------
def _bgr(h: int, s: int, v: int) -> np.ndarray:
    return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]


MAGENTA = _bgr(150, 200, 255)   # bright magenta elixir pip
GOLD = _bgr(25, 220, 255)      # crown gold
PINK = _bgr(175, 200, 220)     # opponent ribbon
BLUE = _bgr(110, 200, 220)     # player ribbon


def make_elixir_frame(pips: int, with_icon: bool = False,
                      with_counter: str = "") -> np.ndarray:
    """900x1600 frame whose elixir bar shows ``pips`` filled pips.

    The empty segments are drawn as a dim purple track - that is what the
    real game renders for an empty bar (only the segment separators keep a
    purple tint), and it is what makes an *empty* bar distinguishable from a
    *hidden* one.
    """
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)
    magenta = _bgr(150, 200, 255)
    track = _bgr(150, 80, 120)
    x0, x1 = int(0.258 * 900), int(0.955 * 900)
    y0, y1 = int(0.966 * 1600), int(0.984 * 1600)
    frame[y0:y1, x0:x1] = track
    segment = (x1 - x0) / 10.0
    for index in range(max(0, min(10, pips))):
        a = x0 + int(round(index * segment))
        b = x0 + int(round((index + 1) * segment))
        frame[y0:y1, a:b] = magenta
    if with_icon:
        frame[y0:y1, int(0.08 * 900):int(0.11 * 900)] = magenta
    if with_counter:
        cv2.putText(frame, with_counter, (int(0.262 * 900), int(0.978 * 1600)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return frame


def make_end_screen(opponent_crowns: int = 0, player_crowns: int = 0,
                    with_ribbons: bool = True,
                    with_ok_button: bool = False) -> np.ndarray:
    """Synthetic Clash Royale post-match end screen."""
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)
    if with_ribbons:
        frame[int(0.24 * 1600):int(0.36 * 1600), :] = PINK
        frame[int(0.51 * 1600):int(0.63 * 1600), :] = BLUE
    for index in range(opponent_crowns):
        x = 120 + index * 200
        frame[int(0.15 * 1600):int(0.205 * 1600), x:x + 160] = GOLD
    for index in range(player_crowns):
        x = 120 + index * 200
        frame[int(0.42 * 1600):int(0.48 * 1600), x:x + 160] = GOLD
    if with_ok_button:
        frame[int(0.879 * 1600):int(0.941 * 1600), 350:550] = BLUE
    return frame


# ---------------------------------------------------------------------------
# 1. Elixir reader
# ---------------------------------------------------------------------------
def verify_elixir() -> None:
    print("\n" + "=" * 72)
    print("1. 10-PIP DISCRETE ELIXIR READER")
    print("=" * 72)
    adapter = PerceptionAdapter(device=torch.device("cpu"))

    check("pip count is 10", ELIXIR_PIP_COUNT == 10)
    check("drop icon boundary is x < 0.255", abs(ELIXIR_ICON_MAX_X - 0.255) < 1e-9,
          f"ELIXIR_ICON_MAX_X={ELIXIR_ICON_MAX_X}")

    ok = True
    for pips in range(0, 11):
        value = adapter.read_elixir(make_elixir_frame(pips))
        if value != float(pips):
            ok = False
            print(f"        pip count {pips} -> {value}")
    check("empty bar reads 0.0", adapter.read_elixir(make_elixir_frame(0)) == 0.0)
    check("every pip count 0..10 is read exactly", ok)

    # A frame without any elixir bar (the HUD is hidden) must NOT read 0.
    adapter.current_elixir = 4.0
    hidden = adapter.read_elixir(np.zeros((1600, 900, 3), dtype=np.uint8))
    check("a hidden elixir bar keeps the last known value", hidden == 4.0, f"read {hidden}")
    check("a hidden elixir bar is reported as invisible",
          adapter.elixir_visible is False)

    check("left drop icon is ignored (0 pips)",
          adapter.read_elixir(make_elixir_frame(0, with_icon=True)) == 0.0)
    check("left drop icon is ignored (4 pips)",
          adapter.read_elixir(make_elixir_frame(4, with_icon=True)) == 4.0)

    check("white elixir counter does not hide the pips (10)",
          adapter.read_elixir(make_elixir_frame(10, with_counter="10")) == 10.0)
    check("white elixir counter does not hide the pips (4)",
          adapter.read_elixir(make_elixir_frame(4, with_counter="4")) == 4.0)

    # Real capture regression (10/10 bar).
    real_frame = PROJECT_ROOT / ".tmp" / "diag_live.png"
    if real_frame.exists():
        frame = cv2.imread(str(real_frame))
        value = adapter.read_elixir(frame)
        check("real 900x1600 capture reads a full bar as 10", value == 10.0, f"read {value}")
    else:
        print("  [SKIP] real capture .tmp/diag_live.png not present")


# ---------------------------------------------------------------------------
# 2. Match outcome detector
# ---------------------------------------------------------------------------
def verify_outcome() -> None:
    print("\n" + "=" * 72)
    print("2. GROUND-TRUTH MATCH OUTCOME DETECTOR")
    print("=" * 72)

    result, p, o, delta = detect_match_end_result(make_end_screen(opponent_crowns=3))
    check("0-3 (opponent took three crowns) -> LOSS",
          result == "LOSS" and o == 3 and p == 0, f"{result} {p}-{o} delta={delta}")

    result, p, o, delta = detect_match_end_result(make_end_screen(player_crowns=1))
    check("1-0 (player win) -> WIN",
          result == "WIN" and p == 1 and o == 0, f"{result} {p}-{o} delta={delta}")

    result, p, o, delta = detect_match_end_result(make_end_screen(player_crowns=3))
    check("3-0 (player win) -> WIN", result == "WIN" and p == 3, f"{result} {p}-{o}")

    result, _, _, _ = detect_match_end_result(make_end_screen())
    check("ambiguous end screen -> UNVERIFIED", result == "UNVERIFIED", result)

    result, _, _, _ = detect_match_end_result(
        make_end_screen(with_ribbons=False, with_ok_button=True))
    check("bottom blue OK button is never evidence -> UNVERIFIED",
          result == "UNVERIFIED", result)

    result, _, _, _ = detect_match_end_result(np.zeros((1600, 900, 3), dtype=np.uint8))
    check("blank frame -> UNVERIFIED", result == "UNVERIFIED", result)

    result, _, _, _ = detect_match_end_result(
        make_end_screen(opponent_crowns=1, player_crowns=1))
    check("contradictory evidence -> UNVERIFIED (never guessed)",
          result == "UNVERIFIED", result)

    # The real captured loss screen.
    real_frame = PROJECT_ROOT / ".tmp" / "diag_live2.png"
    if real_frame.exists():
        frame = cv2.imread(str(real_frame))
        result, p, o, delta = detect_match_end_result(frame)
        check("real captured end screen -> LOSS", result == "LOSS", f"{result} {p}-{o}")
        check("real 1-crown loss pays -9", delta == -9, f"delta {delta}")
    else:
        print("  [SKIP] real capture .tmp/diag_live2.png not present")

    # The real captured two-crown loss screen (opponent took two crowns).
    two_crown = PROJECT_ROOT / ".tmp" / "endscreen_loss_2crowns.png"
    if two_crown.exists():
        frame = cv2.imread(str(two_crown))
        result, p, o, delta = detect_match_end_result(frame)
        check("real captured end screen -> LOSS", result == "LOSS", f"{result} {p}-{o}")
        check("real 2-crown loss uses the conservative model value", delta == -9, f"delta {delta}")
        check("opponent crown count is 2", o == 2, f"crowns {o}")
    else:
        print("  [SKIP] real capture .tmp/endscreen_loss_2crowns.png not present")

    # The real captured WIN screen (live match, +31 trophies).
    win_screen = PROJECT_ROOT / ".tmp" / "endscreen_win_1crown.png"
    if win_screen.exists():
        frame = cv2.imread(str(win_screen))
        result, p, o, delta = detect_match_end_result(frame)
        check("real captured end screen -> WIN", result == "WIN", f"{result} {p}-{o}")
        check("player crown count is 1", p == 1, f"crowns {p}")
    else:
        print("  [SKIP] real capture .tmp/endscreen_win_1crown.png not present")

    # The real captured overtime WIN screen (2-1): the player's crowns sit
    # exactly between the two ribbons.
    ot_screen = PROJECT_ROOT / ".tmp" / "endscreen_win_2crowns_ot.png"
    if ot_screen.exists():
        frame = cv2.imread(str(ot_screen))
        result, p, o, delta = detect_match_end_result(frame)
        check("real overtime end screen -> WIN", result == "WIN", f"{result} {p}-{o}")
        check("overtime score is 2-1", (p, o) == (2, 1), f"crowns {p}-{o}")
    else:
        print("  [SKIP] real capture .tmp/endscreen_win_2crowns_ot.png not present")

    # The real captured 3-0 win screen (with the reward chest popup): the
    # chest is a big gold box and must not be counted as a crown.
    three = PROJECT_ROOT / ".tmp" / "endscreen_win_3crowns.png"
    if three.exists():
        frame = cv2.imread(str(three))
        result, p, o, delta = detect_match_end_result(frame)
        check("real 3-crown end screen -> WIN", result == "WIN", f"{result} {p}-{o}")
        check("3-crown score is 3-0", (p, o) == (3, 0), f"crowns {p}-{o}")
    else:
        print("  [SKIP] real capture .tmp/endscreen_win_3crowns.png not present")

    # Never default to WIN.
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    outcomes = [
        detect_match_end_result(np.zeros((1600, 900, 3), dtype=np.uint8))[0],
        detect_match_end_result(make_end_screen())[0],
        detect_match_end_result(make_end_screen(with_ribbons=False, with_ok_button=True))[0],
    ]
    check("no ambiguous frame is ever reported as WIN",
          all(o != "WIN" for o in outcomes), str(outcomes))


# ---------------------------------------------------------------------------
# 3. Four-card cycle
# ---------------------------------------------------------------------------
def verify_cycle() -> None:
    print("\n" + "=" * 72)
    print("3. FOUR-CARD CYCLE ENFORCEMENT")
    print("=" * 72)
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)

    adapter.recent_deployed_cards.append("Arrows")
    adapter.read_hand(frame, force_refresh=True)
    check("a just-played card is not predicted again",
          "Arrows" not in adapter.tracked_hand, str(adapter.tracked_hand))

    adapter.recent_deployed_cards.clear()
    adapter.read_hand(frame, force_refresh=True)
    check("the card may be predicted again once the cycle window passed",
          len(adapter.tracked_hand) == 4)

    # The cycle queue must hold at most three cards.
    adapter.recent_deployed_cards.extend(["Fireball", "Knight", "Giant", "Musketeer"])
    check("cycle queue is bounded to three cards",
          len(adapter.recent_deployed_cards) == 3, str(list(adapter.recent_deployed_cards)))

    # A card played 3 times must not be predicted again.
    for _ in range(3):
        adapter.update_hand_on_play(0, deduct_elixir=False,
                                    card_name=adapter.tracked_hand[0])
    adapter.read_hand(frame, force_refresh=True)
    check("cards deployed in the last three plays are excluded",
          all(c not in adapter.tracked_hand for c in adapter.recent_deployed_cards),
          str(adapter.tracked_hand))


# ---------------------------------------------------------------------------
# 4. Tactical mask invariants (same rules tools/diagnose_truth.py checks)
# ---------------------------------------------------------------------------
def verify_masks() -> None:
    print("\n" + "=" * 72)
    print("4. TACTICAL MASK INVARIANTS (shared tactical_masks module)")
    print("=" * 72)

    # --- spells: zero unmasked tiles with tile_y <= 11 ---------------------
    bundle = build_tactical_masks(
        [FIREBALL, ARROWS, KNIGHT, GIANT], 10.0,
        [EnemyUnit(card_id=5, x=3.0, y=13.0)],
    )
    spell_tiles = bundle.card_tile_mask[0]
    low = int(np.sum(spell_tiles[: (SPELL_BLOCKED_TILE_Y + 1) * ARENA_WIDTH]))
    check("spell has 0 unmasked tiles with tile_y <= 11", low == 0, f"{low} tiles")

    spell_mask = bundle.card_tile_mask[1]
    low = int(np.sum(spell_mask[: (SPELL_BLOCKED_TILE_Y + 1) * ARENA_WIDTH]))
    check("second spell slot also has 0 unmasked tiles with tile_y <= 11", low == 0)

    # The unmasked spell tiles must sit inside a 2-tile radius of the enemy
    # (with the -1 y-lead).
    cy = max(SPELL_MIN_TILE_Y, int(round(13.0)) - 1)
    cx = int(round(3.0))
    rows = sorted({int(t) // ARENA_WIDTH for t in np.flatnonzero(spell_mask)})
    check("spell tiles stay within +-2 rows of the aimed row",
          all(abs(r - cy) <= 2 for r in rows), str(rows))
    check("spell tiles stay within +-2 columns of the enemy",
          all(abs((t % ARENA_WIDTH) - cx) <= 2 for t in np.flatnonzero(spell_mask)))

    # --- spell gating: no enemy in range -> spell masked out --------------
    bundle = build_tactical_masks([FIREBALL, KNIGHT, MUSKETEER, GIANT], 10.0, [])
    check("spell slot is masked out without a target", not bundle.card_mask[0])

    # --- win condition guard ----------------------------------------------
    bundle = build_tactical_masks(
        [GIANT, KNIGHT, MUSKETEER, ARCHERS], 10.0,
        [EnemyUnit(card_id=5, x=3.0, y=10.0)],
    )
    check("Giant is masked out while under threat with an affordable defender",
          not bundle.card_mask[0], str(bundle.card_mask))
    check("the defender stays available", bundle.card_mask[1])

    # Without a threat the win condition is playable again.
    bundle = build_tactical_masks(
        [GIANT, KNIGHT, MUSKETEER, ARCHERS], 10.0,
        [EnemyUnit(card_id=5, x=3.0, y=25.0)],
    )
    check("Giant is playable again without a threat", bundle.card_mask[0])

    # --- same-lane troop defence ------------------------------------------
    bundle = build_tactical_masks(
        [KNIGHT, MUSKETEER, ARCHERS, MINIONS], 10.0,
        [EnemyUnit(card_id=5, x=3.0, y=10.0)],
    )
    defence = bundle.card_tile_mask[0]
    xs = {int(t) % ARENA_WIDTH for t in np.flatnonzero(defence)}
    ys = {int(t) // ARENA_WIDTH for t in np.flatnonzero(defence)}
    check("troop defence is constrained to the threat lane",
          all(1 <= x <= 5 for x in xs), str(sorted(xs)))
    check("troop defence is constrained to y in 8..12",
          all(8 <= y <= 12 for y in ys), str(sorted(ys)))

    # --- empty board elixir bank ------------------------------------------
    bundle = build_tactical_masks([KNIGHT, MUSKETEER, ARCHERS, MINIONS], 5.0, [])
    check("empty board + low elixir masks all four card slots",
          not bundle.card_mask[:4].any(), str(bundle.card_mask))
    check("Wait stays legal", bool(bundle.card_mask[4]))
    bundle = build_tactical_masks([KNIGHT, MUSKETEER, ARCHERS, MINIONS],
                                  ELIXIR_BANK_THRESHOLD + 0.5, [])
    check("empty board + enough elixir allows playing",
          bundle.card_mask[:4].any(), str(bundle.card_mask))

    # --- the live bridge and the simulator must agree ----------------------
    from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv
    env = CrSimReplicaEnv(seed=17)
    env.reset()
    obs = env.get_obs(1)
    check("simulator still produces a per-card tile mask",
          obs["card_tile_mask"].shape == (5, TOTAL_TILES))
    check("simulator keeps Wait legal", bool(obs["card_mask"][4]))


def main() -> int:
    print("=" * 72)
    print("  ClashBot - OFFLINE GROUND-TRUTH VERIFICATION")
    print("=" * 72)
    print(f"  project root: {PROJECT_ROOT}")

    verify_elixir()
    verify_outcome()
    verify_cycle()
    verify_masks()

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"  RESULT: {len(FAILURES)}/{CHECKS} CHECKS FAILED")
        for name in FAILURES:
            print(f"    - {name}")
        print("=" * 72)
        return 1
    print(f"  RESULT: ALL {CHECKS} CHECKS PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
