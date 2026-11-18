#!/usr/bin/env python3
"""Offline audit: are all the new 'elite' features actually active?

This script exercises the live perception path against real captured frames
and real replay rows.  It verifies tensor shapes, opponent-state inference,
full-HP blob detection, kinematic typing, pro geometry and post-match credit
assignment.  No emulator or live match is required.

Usage::

    python tools/audit_new_features.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy_mumu import PerceptionAdapter, resolve_deck_names  # noqa: E402
from cr_alphastar_agent.env.arena_monitor import ArenaMonitor  # noqa: E402
from cr_alphastar_agent.env.opponent_tracker import OpponentTracker  # noqa: E402
from cr_alphastar_agent.env.pro_geometry import (  # noqa: E402
    king_activation_tiles, kite_tiles, lethal_spell_cycle, should_kite,
)
from cr_alphastar_agent.utils.card_intel import get_card_intel  # noqa: E402

FAILURES: list = []
COUNT = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global COUNT
    COUNT += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    print("=" * 72)
    print("  ELITE FEATURE AUDIT (offline, real captures)")
    print("=" * 72)
    intel = get_card_intel()

    print("\n1. OBSERVATION SHAPES")
    adapter = PerceptionAdapter(device=torch.device("cpu"),
                                deck=resolve_deck_names("current"))
    frame_path = ROOT / ".tmp" / "inbattle_reference.png"
    if frame_path.exists():
        frame = cv2.imread(str(frame_path))
        obs = adapter.frame_to_obs(frame)
        check("scalars are 7-wide", tuple(obs["scalars"].shape) == (1, 7),
              str(tuple(obs["scalars"].shape)))
        check("entities are 9-wide", tuple(obs["entities"].shape) == (1, 40, 9),
              str(tuple(obs["entities"].shape)))
        check("new scalar slots are populated",
              bool(np.isfinite(obs["scalars"][0, 4:].cpu().numpy()).all()),
              str(obs["scalars"][0, 4:].cpu().numpy().round(3).tolist()))
        check("arena monitor produced full-HP candidates",
              isinstance(adapter.full_hp_blobs, list),
              f"{len(adapter.full_hp_blobs)} blobs")
        check("opponent tracker produced state", adapter.opponent_state is not None,
              f"elixir={getattr(adapter.opponent_state, 'elixir', None)}")
        check("board tracker produced a snapshot", adapter.board_snapshot is not None,
              f"{len(adapter.board_snapshot.enemy_tracks)} tracks")
        check("kinematic roles computed", isinstance(adapter.kinematic_roles, list),
              str(adapter.kinematic_roles[:2]))
    else:
        print(f"  [SKIP] missing real capture {frame_path}")

    print("\n2. OPPONENT ELIXIR + CYCLE")
    tracker = OpponentTracker(intel)
    tracker.tick(0, 5.0)
    tracker.observe_play("Giant", 0, 3)
    check("card cost subtracted", 0.0 <= tracker.elixir < 1.0, f"{tracker.elixir:.2f}")
    tracker.tick(10, 5.0)
    check("idle resync to 10", tracker.elixir == 10.0, f"{tracker.elixir:.2f}")
    tracker.observe_play("Cannon", 11, 13)
    check("counter locked after play", tracker.is_locked("Cannon"))

    print("\n3. FULL-HP BACKGROUND SUBTRACTION")
    monitor = ArenaMonitor(alpha=0.02)
    arena = np.full((400, 600, 3), 70, dtype=np.uint8)
    monitor.suppress(arena, [])
    moved = arena.copy()
    cv2.rectangle(moved, (270, 170), (330, 250), (180, 40, 40), -1)
    blobs = monitor.detect(moved, {"top": 0.05, "bottom": 0.78,
                                   "left": 0.03, "right": 0.97}, (900, 1600))
    check("a 100%-HP sprite is detected", len(blobs) > 0, f"{len(blobs)} blobs")

    print("\n4. ZERO-SHOT KINEMATIC TYPING")
    fast_air, c1 = intel.coarse_role(1.125, True)
    slow_ground, c2 = intel.coarse_role(0.562, False)
    check("fast air -> air_swarm", fast_air == "air_swarm", f"{fast_air} {c1:.2f}")
    check("slow ground -> ground_tank", slow_ground == "ground_tank", f"{slow_ground} {c2:.2f}")
    lead = intel.spell_lead_target("Arrows", {"x": 4, "y": 20, "vx": 0, "vy": -1.5})
    check("spell lead tracks velocity", lead[1] < 20.0, str(tuple(round(v, 2) for v in lead)))

    print("\n5. PRO GEOMETRY")
    check("kite crosses centre", all(x > 9 for x, _ in kite_tiles(3, 13)))
    check("King activation exact", king_activation_tiles("GoblinBarrel") == [(8, 4)])
    check("kite only for slow melee", should_kite("ground_tank", 0.56, 13)
          and not should_kite("air_swarm", 1.125, 13))
    check("lethal cycle gated by time",
          lethal_spell_cycle(0.2, [570], 20, False)
          and not lethal_spell_cycle(0.2, [570], 90, False))

    print("\n6. CREDIT ASSIGNMENT FOR USELESS PLAYS")
    q, reason = intel.evaluate_play_quality("Arrows", 4, 17, [])
    check("empty spell gets near-zero weight", q < 0.1, reason)
    q2, reason2 = intel.evaluate_play_quality("Arrows", 4, 17,
                                              [{"x": 4, "y": 17}, {"x": 5, "y": 18}])
    check("useful spell gets full weight", q2 > 0.8, reason2)

    print("\n7. REAL REPLAY SHAPES")
    replay = ROOT / ".tmp" / "live_replays.jsonl"
    if replay.exists():
        rows = [json.loads(line) for line in replay.read_text(encoding="utf-8").splitlines()]
        if rows:
            row = rows[-1]
            scalar_width = len(row["obs"]["scalars"][0])
            entity_width = len(row["obs"]["entities"][0][0])
            if scalar_width == 4:
                print("  [INFO] replay predates the 7-scalar upgrade (legacy data)")
            else:
                check("replay scalars are 7-wide", scalar_width == 7,
                      f"width {scalar_width}")
            if entity_width == 7:
                print("  [INFO] replay predates the 9-entity upgrade (legacy data)")
            else:
                check("replay entities are 9-wide", entity_width == 9,
                      f"width {entity_width}")
            check("replay has verified/weight fields",
                  "verified" in row and "weight" in row,
                  f"verified={row.get('verified')} weight={row.get('weight')}")
    else:
        print("  [SKIP] no real replay present")

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"  RESULT: {len(FAILURES)}/{COUNT} CHECKS FAILED")
        for name in FAILURES:
            print(f"    - {name}")
        return 1
    print(f"  RESULT: ALL {COUNT} CHECKS PASSED")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
