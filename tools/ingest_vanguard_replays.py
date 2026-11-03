"""
Ingests Hugging Face VanguardX101/IL_Replay dataset into CR-Zero trajectory shards.
Dataset: VanguardX101/IL_Replay (17.8M actions / 252k replays)
Link: https://huggingface.co/datasets/VanguardX101/IL_Replay
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.constants import (
    ARENA_ROWS,
    ARENA_COLS,
    TOTAL_TILES,
    CARD_DATABASE,
    CARD_ID_TO_COST,
    CARD_ID_TO_IS_SPELL,
    CARD_NAME_TO_ID,
    RIVER_ROW_SOUTH,
    LEFT_BRIDGE_COLS,
    RIGHT_BRIDGE_COLS,
    STACKED_GRID_CHANNELS,
    NUM_SCALARS,
)
from perception.state_encoder import stack_spatial_frames


def normalize_card_key(card_key: str) -> str:
    """Normalizes dataset card keys (e.g., 'firecracker-ev1' -> 'firecracker')."""
    k = str(card_key).lower().strip()
    if "-ev" in k:
        k = k.split("-ev")[0]
    # Specific remappings
    remap = {
        "the-log": "the log",
        "mini-pekka": "mini p.e.k.k.a",
        "pekka": "p.e.k.k.a",
        "boss-bandit": "boss bandit",
        "giant-skeleton": "giant skeleton",
        "royal-recruits": "royal recruits",
        "royal-hogs": "royal hogs",
        "royal-giant": "royal giant",
        "royal-ghost": "royal ghost",
        "royal-delivery": "royal delivery",
        "three-musketeers": "three musketeers",
        "skeleton-army": "skeleton army",
        "skeleton-barrel": "skeleton barrel",
        "skeleton-dragons": "skeleton dragons",
        "skeleton-king": "skeleton king",
        "goblin-barrel": "goblin barrel",
        "goblin-gang": "goblin gang",
        "goblin-giant": "goblin giant",
        "goblin-cage": "goblin cage",
        "goblin-drill": "goblin drill",
        "goblin-hut": "goblin hut",
        "goblin-demolisher": "goblin demolisher",
        "goblin-machine": "goblin machine",
        "goblin-curse": "goblin curse",
        "spear-goblins": "spear goblins",
        "ice-spirit": "ice spirit",
        "fire-spirit": "fire spirit",
        "electro-spirit": "electro spirit",
        "heal-spirit": "heal spirit",
        "ice-wizard": "ice wizard",
        "electro-wizard": "electro wizard",
        "magic-archer": "magic archer",
        "archer-queen": "archer queen",
        "golden-knight": "golden knight",
        "mighty-miner": "mighty miner",
        "little-prince": "little prince",
        "dark-prince": "dark prince",
        "hog-rider": "hog rider",
        "ram-rider": "ram rider",
        "battle-ram": "battle ram",
        "battle-healer": "battle healer",
        "flying-machine": "flying machine",
        "inferno-dragon": "inferno dragon",
        "inferno-tower": "inferno tower",
        "bomb-tower": "bomb tower",
        "barbarian-hut": "barbarian hut",
        "barbarian-barrel": "barbarian barrel",
        "elite-barbarians": "elite barbarians",
        "minion-horde": "minion horde",
        "mega-minion": "mega minion",
        "mega-knight": "mega knight",
        "night-witch": "night witch",
        "mother-witch": "mother witch",
        "baby-dragon": "baby dragon",
        "electro-dragon": "electro dragon",
        "electro-giant": "electro giant",
        "elixir-collector": "elixir collector",
        "elixir-golem": "elixir golem",
        "giant-snowball": "giant snowball",
        "wall-breakers": "wall breakers",
        "suspicious-bush": "suspicious bush",
        "cannon-cart": "cannon cart",
        "rune-giant": "rune giant",
        "spirit-empress": "spirit empress",
    }
    k = remap.get(k, k.replace("-", " "))
    return k


def card_key_to_id(card_key: str) -> int:
    norm = normalize_card_key(card_key)
    if norm in CARD_NAME_TO_ID:
        return CARD_NAME_TO_ID[norm]
    # Fallback search
    for name, cid in CARD_NAME_TO_ID.items():
        if norm in name or name in norm:
            return cid
    return 0


def build_legal_tile_mask(card_id: int) -> np.ndarray:
    """Builds legal 576-tile deployment mask for a given card."""
    mask = np.zeros(TOTAL_TILES, dtype=bool)
    is_spell = CARD_ID_TO_IS_SPELL.get(card_id, False)

    # Any-tile cards (all spells, miner, goblin drill, goblin barrel)
    if is_spell or card_id in (12, 55, 113):
        mask[:] = True
        return mask

    # Standard troops/buildings: ally side (rows 16..31)
    for r in range(RIVER_ROW_SOUTH, ARENA_ROWS):
        for c in range(ARENA_COLS):
            mask[r * ARENA_COLS + c] = True

    # Bridges are legal deploy tiles
    for col in list(LEFT_BRIDGE_COLS) + list(RIGHT_BRIDGE_COLS):
        mask[15 * ARENA_COLS + col] = True

    return mask


def infer_initial_queue(deck: List[int], plays: List[int]) -> Optional[List[int]]:
    """Real CR hand = 4 slots + 4-card queue. The opening order is NOT in the replay,
    so recover an ordering [h0..h3, q0..q3] consistent with every observed play.
    Returns None if the deck is not 8 distinct cards or no ordering explains the plays."""
    from itertools import permutations
    if len(deck) != 8 or len(set(deck)) != 8:
        return None
    for perm in permutations(deck):
        hand, queue, ok = list(perm[:4]), list(perm[4:]), True
        for cid in plays:
            if cid not in hand:
                ok = False
                break
            s_ = hand.index(cid)
            hand[s_] = queue.pop(0)
            queue.append(cid)
        if ok:
            return list(perm)
    return None


def process_match(
    replay_meta: Dict,
    match_actions: List[Dict],
) -> Optional[Dict[str, np.ndarray]]:
    """Vanguard actions -> BC samples. Observation is built BEFORE the action is
    applied; the target action/tile is never written into the input; hand is
    snapshotted BEFORE the card cycles; scalar/channel layout comes from
    configs.observation_schema (identical to the live StateEncoder)."""
    from configs.observation_schema import Ch, Sc, SCHEMA_VERSION
    from configs.constants import get_default_deployable_mask

    result_str = str(replay_meta.get("result", "")).lower()
    outcome = 1 if result_str == "victory" else (-1 if result_str == "defeat" else 0)

    payload_str = replay_meta.get("payload_json", "{}")
    try:
        payload = json.loads(payload_str)
        deck_info = payload.get("battle", {}).get("team", {}).get("players", [{}])[0].get("deck", [])
        deck_card_ids = [card_key_to_id(c.get("card_key", "")) for c in deck_info]
        deck_card_ids = [c for c in deck_card_ids if c > 0]
    except Exception:
        deck_card_ids = []

    plays = []
    for a in match_actions:
        if a.get("side") == "team" and a.get("kind") == "play_card":
            cid = card_key_to_id(a.get("card_key", ""))
            if cid > 0:
                plays.append((float(a.get("time_seconds", 0.0)), cid, a))
    if len(plays) < 5:
        return None

    order = infer_initial_queue(deck_card_ids, [c for _, c, _ in plays])
    if order is None:
        return None  # hand cannot be reconstructed faithfully -> do not train on it
    hand, queue = order[:4], order[4:]

    base = np.zeros((8, ARENA_ROWS, ARENA_COLS), dtype=np.float32)
    base[Ch.ALLY_BUILDING, 31, 9] = base[Ch.ALLY_BUILDING, 26, 3] = base[Ch.ALLY_BUILDING, 26, 14] = 1.0
    base[Ch.ENEMY_BUILDING, 0, 9] = base[Ch.ENEMY_BUILDING, 5, 3] = base[Ch.ENEMY_BUILDING, 5, 14] = 1.0
    base[Ch.DEPLOYABLE] = get_default_deployable_mask(False, False).astype(np.float32)

    G, S, H, CM, TM, AC, AT, TS = [], [], [], [], [], [], [], []
    history, elixir, last_t = [], 5.0, 0.0
    last_card_n, last_tile_n = 0.0, 0.0

    for t, cid, act in plays:
        rate = (1.0 / 1.4) if t >= 120.0 else (1.0 / 2.8)
        elixir = min(10.0, elixir + max(0.0, t - last_t) * rate)
        last_t = t
        cost = CARD_ID_TO_COST.get(cid, 3)
        elixir = max(elixir, float(cost))  # a card was played, so the cost was available: resync

        cr_y = int(np.clip(31 - int(act.get("grid_y", 0)), 0, 31))
        cr_x = int(np.clip(int(act.get("grid_x", 9)), 0, 17))
        tile_idx = cr_y * ARENA_COLS + cr_x
        slot_idx = hand.index(cid)
        hand_before = list(hand)                      # snapshot BEFORE cycling

        history.append(base.copy())                   # target is NOT painted into the input
        stacked = stack_spatial_frames(history, target_frames=4)

        sc = np.zeros(NUM_SCALARS, dtype=np.float32)
        sc[Sc.ALLY_ELIXIR] = elixir / 10.0
        sc[Sc.ENEMY_ELIXIR] = 0.5                     # unknown in replay data
        sc[Sc.TIME_LEFT] = float(np.clip((180.0 - t) / 180.0, 0.0, 1.0))
        sc[Sc.ALLY_KING:Sc.ENEMY_RIGHT + 1] = 1.0     # tower HP unknown in replay data
        sc[Sc.LAST_CARD] = last_card_n
        sc[Sc.LAST_TILE] = last_tile_n
        sc[Sc.DOUBLE_ELIXIR] = 1.0 if t >= 120.0 else 0.0

        cmask = np.ones(5, dtype=bool)
        for s_ in range(4):
            if CARD_ID_TO_COST.get(hand_before[s_], 99) > elixir:
                cmask[s_ + 1] = False
        cmask[slot_idx + 1] = True
        tmask = build_legal_tile_mask(cid)
        tmask[tile_idx] = True

        G.append(stacked); S.append(sc); H.append(np.array(hand_before, dtype=np.int64))
        CM.append(cmask); TM.append(tmask); AC.append(slot_idx + 1); AT.append(tile_idx); TS.append(t)

        elixir = max(0.0, elixir - cost)              # apply action AFTER the sample is stored
        hand[slot_idx] = queue.pop(0)
        queue.append(cid)
        last_card_n, last_tile_n = (slot_idx + 1) / 4.0, tile_idx / float(TOTAL_TILES - 1)

    if len(G) < 5:
        return None
    return {
        "arena_grid": np.array(G, dtype=np.float32),
        "scalars": np.array(S, dtype=np.float32),
        "hand_ids": np.array(H, dtype=np.int64),
        "legal_card_mask": np.array(CM, dtype=bool),
        "legal_tile_mask": np.array(TM, dtype=bool),
        "action_card": np.array(AC, dtype=np.int64),
        "action_tile": np.array(AT, dtype=np.int64),
        "timestamps": np.array(TS, dtype=np.float32),
        "match_outcome": np.int32(outcome),
        "schema_version": np.int32(SCHEMA_VERSION),
        # NOTE: no "rewards" key. Vanguard has no per-action signal; do not feed it to AWR.
        # NOTE: no Wait samples and no board/enemy info; use for card/tile-prior pretraining only.
    }


def ingest_vanguard_replays(
    actions_path: str,
    replays_path: str,
    output_dir: str = "data/trajectories",
    max_matches: int = 50,
) -> List[str]:
    """Ingests Vanguard parquet files and outputs .npz shards."""
    print(f"[Vanguard Ingest] Loading replays from {replays_path}...")
    replays_df = pq.read_table(replays_path).to_pandas().set_index("replay_tag")

    print(f"[Vanguard Ingest] Loading actions from {actions_path}...")
    actions_table = pq.read_table(
        actions_path,
        columns=["replay_tag", "side", "kind", "card_key", "grid_x", "grid_y", "time_seconds"],
    )
    actions_df = actions_table.to_pandas()

    grouped = actions_df.groupby("replay_tag")
    print(f"[Vanguard Ingest] Available replays: {len(grouped)}. Processing up to {max_matches} matches...")

    os.makedirs(output_dir, exist_ok=True)
    created_shards = []
    count = 0

    import pandas as pd

    for replay_tag, group in grouped:
        if replay_tag not in replays_df.index:
            continue

        replay_meta = replays_df.loc[replay_tag]
        if isinstance(replay_meta, pd.DataFrame):
            replay_meta = replay_meta.iloc[0]

        meta_dict = replay_meta.to_dict() if hasattr(replay_meta, "to_dict") else dict(replay_meta)
        match_actions = group.sort_values("time_seconds").to_dict(orient="records")
        tensors = process_match(meta_dict, match_actions)
        if tensors is None:
            continue

        outcome_label = "win" if tensors["match_outcome"] == 1 else "loss"
        shard_name = f"vanguard_{count:03d}_{outcome_label}.npz"
        shard_path = os.path.join(output_dir, shard_name)

        np.savez_compressed(shard_path, **tensors)
        created_shards.append(shard_path)
        count += 1
        print(f"  [{count}/{max_matches}] Created {shard_name} ({len(tensors['action_card'])} actions | Outcome: {outcome_label})")

        if count >= max_matches:
            break

    print(f"[Vanguard Ingest] Done! Created {len(created_shards)} new trajectory shard(s) in {output_dir}.")
    return created_shards


def main():
    parser = argparse.ArgumentParser(description="Ingest VanguardX101/IL_Replay into CR-Zero shards.")
    parser.add_argument("--actions-file", type=str, default=None)
    parser.add_argument("--replays-file", type=str, default=None)
    parser.add_argument("--max-matches", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="data/trajectories")
    args = parser.parse_args()

    # Find cached files if not passed
    actions_path = args.actions_file
    replays_path = args.replays_file

    if not actions_path or not os.path.exists(actions_path):
        cache_base = os.path.expanduser("~/.cache/huggingface/hub/datasets--VanguardX101--IL_Replay")
        for root, _, files in os.walk(cache_base):
            for f in files:
                if f == "part-000000.parquet":
                    if "actions" in root and not actions_path:
                        actions_path = os.path.join(root, f)
                    elif "replays" in root and not replays_path:
                        replays_path = os.path.join(root, f)

    if not actions_path or not replays_path:
        print("[Vanguard Ingest] Error: Could not locate Vanguard parquet files in cache.")
        sys.exit(1)

    ingest_vanguard_replays(actions_path, replays_path, args.output_dir, args.max_matches)


if __name__ == "__main__":
    main()
