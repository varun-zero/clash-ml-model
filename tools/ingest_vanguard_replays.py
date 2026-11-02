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


def process_match(
    replay_meta: Dict,
    match_actions: List[Dict],
) -> Optional[Dict[str, np.ndarray]]:
    """
    Simulates a match from Vanguard actions and constructs full CR-Zero trajectory tensors.
    """
    result_str = str(replay_meta.get("result", "")).lower()
    if result_str == "victory":
        outcome = 1
    elif result_str == "defeat":
        outcome = -1
    else:
        outcome = 0

    # Extract 8-card player deck
    payload_str = replay_meta.get("payload_json", "{}")
    try:
        payload = json.loads(payload_str)
        team_deck_info = payload.get("battle", {}).get("team", {}).get("players", [{}])[0].get("deck", [])
        deck_card_ids = [card_key_to_id(c.get("card_key", "")) for c in team_deck_info if card_key_to_id(c.get("card_key", "")) > 0]
    except Exception:
        deck_card_ids = []

    # If deck parsing failed, collect unique played cards
    if len(deck_card_ids) < 4:
        played = []
        for a in match_actions:
            if a.get("side") == "team" and a.get("kind") == "play_card":
                cid = card_key_to_id(a.get("card_key", ""))
                if cid > 0 and cid not in played:
                    played.append(cid)
        deck_card_ids = played[:8]

    if len(deck_card_ids) < 4:
        return None

    # Hand simulation
    hand = deck_card_ids[:4]
    draw_pile = deck_card_ids[4:] if len(deck_card_ids) > 4 else []

    # Filter actions by time
    team_plays = [a for a in match_actions if a.get("side") == "team" and a.get("kind") == "play_card"]
    if len(team_plays) < 5:
        return None

    frames_grid = []
    frames_scalar = []
    frames_hand = []
    frames_cmask = []
    frames_tmask = []
    frames_card_act = []
    frames_tile_act = []
    frames_rewards = []

    current_grid_history = []
    simulated_elixir = 5.0
    last_t = 0.0

    # Base arena map with river and towers
    base_arena = np.zeros((8, ARENA_ROWS, ARENA_COLS), dtype=np.float32)
    # Channel 2: Friendly towers
    base_arena[2, 31, 9] = 1.0   # Ally King
    base_arena[2, 26, 3] = 1.0   # Ally Left Princess
    base_arena[2, 26, 14] = 1.0  # Ally Right Princess
    # Channel 5: Enemy towers
    base_arena[5, 0, 9] = 1.0    # Enemy King
    base_arena[5, 5, 3] = 1.0    # Enemy Left Princess
    base_arena[5, 5, 14] = 1.0   # Enemy Right Princess
    # Channel 6: River and Bridges
    base_arena[6, 15, :] = 1.0
    base_arena[6, 16, :] = 1.0
    for col in list(LEFT_BRIDGE_COLS) + list(RIGHT_BRIDGE_COLS):
        base_arena[6, 15, col] = 0.5
        base_arena[6, 16, col] = 0.5

    for act in team_plays:
        t = float(act.get("time_seconds", 0.0))
        dt = max(0.0, t - last_t)
        last_t = t

        # Elixir accumulation (1 elixir per 2.8s, double elixir after 120s)
        rate = (1.0 / 1.4) if t >= 120.0 else (1.0 / 2.8)
        simulated_elixir = min(10.0, simulated_elixir + dt * rate)

        cid = card_key_to_id(act.get("card_key", ""))
        if cid == 0:
            continue

        cost = CARD_ID_TO_COST.get(cid, 3)

        # Coordinate transformation: RoyaleAPI 0 is ally back line, 31 is enemy
        # CR-Zero 0 is enemy, 31 is ally -> cr_zero_y = 31 - grid_y
        raw_gx = int(act.get("grid_x", 9))
        raw_gy = int(act.get("grid_y", 0))
        cr_y = np.clip(31 - raw_gy, 0, 31)
        cr_x = np.clip(raw_gx, 0, 17)
        tile_idx = cr_y * ARENA_COLS + cr_x

        # Identify which slot matches this card
        slot_idx = -1
        if cid in hand:
            slot_idx = hand.index(cid)
        else:
            # Shift hand to accommodate card
            slot_idx = 0
            hand[0] = cid

        # Update board state
        current_frame = base_arena.copy()
        is_spell = CARD_ID_TO_IS_SPELL.get(cid, False)
        if is_spell:
            current_frame[7, cr_y, cr_x] = 1.0
        else:
            current_frame[0, cr_y, cr_x] = 1.0

        current_grid_history.append(current_frame)
        stacked_grid = stack_spatial_frames(current_grid_history, target_frames=4)

        # Scalar features (12)
        scalar_vec = np.zeros(NUM_SCALARS, dtype=np.float32)
        scalar_vec[0] = simulated_elixir / 10.0
        scalar_vec[1] = 0.5  # enemy elixir estimate
        scalar_vec[2] = 1.0  # ally king hp
        scalar_vec[3] = 1.0  # ally princess L hp
        scalar_vec[4] = 1.0  # ally princess R hp
        scalar_vec[5] = 1.0  # enemy king hp
        scalar_vec[6] = 1.0  # enemy princess L hp
        scalar_vec[7] = 1.0  # enemy princess R hp
        scalar_vec[8] = min(t / 180.0, 1.0)  # normalized match time
        scalar_vec[9] = 1.0 if t >= 120.0 else 0.0  # double elixir
        scalar_vec[10] = 0.0  # differential hp
        scalar_vec[11] = float(outcome)  # outcome bias

        # Card mask: slot 0 (wait=True), slots 1..4 (affordability)
        cmask = np.ones(5, dtype=bool)
        for s in range(4):
            card_cost = CARD_ID_TO_COST.get(hand[s], 99)
            if card_cost > simulated_elixir:
                cmask[s + 1] = False
        # The chosen card action must be marked legal in ground truth
        cmask[slot_idx + 1] = True

        tmask = build_legal_tile_mask(cid)
        tmask[tile_idx] = True

        # Spend elixir
        simulated_elixir = max(0.0, simulated_elixir - cost)

        # Cycle hand
        if draw_pile:
            next_c = draw_pile.pop(0)
            draw_pile.append(cid)
            hand[slot_idx] = next_c

        # Advantage reward
        adv_reward = outcome * 1.0 + (0.5 if outcome == 1 else -0.5)

        frames_grid.append(stacked_grid)
        frames_scalar.append(scalar_vec)
        frames_hand.append(np.array(hand, dtype=np.int64))
        frames_cmask.append(cmask)
        frames_tmask.append(tmask)
        frames_card_act.append(slot_idx + 1)
        frames_tile_act.append(tile_idx)
        frames_rewards.append(adv_reward)

    if len(frames_grid) < 5:
        return None

    return {
        "arena_grid": np.array(frames_grid, dtype=np.float32),
        "scalars": np.array(frames_scalar, dtype=np.float32),
        "scalar_features": np.array(frames_scalar, dtype=np.float32),
        "hand_ids": np.array(frames_hand, dtype=np.int64),
        "hand_cards": np.array(frames_hand, dtype=np.int64),
        "legal_card_mask": np.array(frames_cmask, dtype=bool),
        "card_mask": np.array(frames_cmask, dtype=bool),
        "legal_tile_mask": np.array(frames_tmask, dtype=bool),
        "tile_mask": np.array(frames_tmask, dtype=bool),
        "action_card": np.array(frames_card_act, dtype=np.int64),
        "card_action": np.array(frames_card_act, dtype=np.int64),
        "action_tile": np.array(frames_tile_act, dtype=np.int64),
        "tile_action": np.array(frames_tile_act, dtype=np.int64),
        "rewards": np.array(frames_rewards, dtype=np.float32),
        "match_outcome": np.int32(outcome),
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
        print(f"  [{count}/{max_matches}] Created {shard_name} ({len(tensors['card_action'])} actions | Outcome: {outcome_label})")

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
