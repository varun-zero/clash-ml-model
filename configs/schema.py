"""
Canonical Schema definition for CR-Zero (Alias / Re-export of observation_schema.py).
Defines GridChannel and ScalarIndex for type checking and backward/forward compatibility.
"""
from configs.observation_schema import (
    Ch,
    Sc,
    SCHEMA_VERSION,
    check_schema,
    ALLY_TOWER_SLICE,
    ENEMY_TOWER_SLICE,
)

# Explicit IntEnum aliases for readability matching audit specs:
GridChannel = Ch
ScalarIndex = Sc

__all__ = [
    "Ch",
    "Sc",
    "GridChannel",
    "ScalarIndex",
    "SCHEMA_VERSION",
    "check_schema",
    "ALLY_TOWER_SLICE",
    "ENEMY_TOWER_SLICE",
]
