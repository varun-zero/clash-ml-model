"""Single source of truth for CR-Zero observation layout. Every producer
(live encoder, YouTube extractor, Vanguard ingest, live logger) and consumer
(datasets, flip augmentation, RuleBot) must index through these enums."""
from enum import IntEnum
import numpy as np

SCHEMA_VERSION = 2  # bump on ANY change to channel/scalar meaning


class Ch(IntEnum):          # per-frame spatial channels (8); stacked x4 -> 32
    ALLY_GROUND = 0
    ALLY_AIR = 1
    ALLY_BUILDING = 2
    ENEMY_GROUND = 3
    ENEMY_AIR = 4
    ENEMY_BUILDING = 5
    SPELL = 6
    DEPLOYABLE = 7


class Sc(IntEnum):          # scalar features (12)
    ALLY_ELIXIR = 0
    ENEMY_ELIXIR = 1
    TIME_LEFT = 2           # time_left / 180
    ALLY_KING = 3
    ALLY_LEFT = 4
    ALLY_RIGHT = 5
    ENEMY_KING = 6
    ENEMY_LEFT = 7
    ENEMY_RIGHT = 8
    LAST_CARD = 9           # last PLAYED slot / 4 (persists until next play)
    LAST_TILE = 10          # last PLAYED tile / 575 (persists until next play)
    DOUBLE_ELIXIR = 11


ALLY_TOWER_SLICE = slice(Sc.ALLY_KING, Sc.ALLY_RIGHT + 1)
ENEMY_TOWER_SLICE = slice(Sc.ENEMY_KING, Sc.ENEMY_RIGHT + 1)


def check_schema(data, path: str = "") -> None:
    """Call right after np.load(). Strictly refuses shards without schema_version or with layout mismatch."""
    if "schema_version" not in data:
        raise ValueError(f"{path}: missing schema_version. Re-generate or explicitly migrate this shard.")
    v = int(data["schema_version"])
    if v != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version={v}, code expects {SCHEMA_VERSION}. Re-generate the shard.")
