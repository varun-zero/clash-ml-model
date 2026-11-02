"""Static card features for the global 135-card embedding vocabulary.

The learned per-ID embedding captures card identity. These features give the
policy a valid signal even for a card ID that never appeared in training:
``Embedding(id) + Linear(features(id))``. IDs missing from CARD_DATABASE get a
zero feature vector and still have a trainable embedding row.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from configs.constants import CARD_DATABASE, NUM_TOTAL_CARDS

CARD_FEATURE_NAMES: Tuple[str, ...] = (
    "cost_norm",
    "is_spell",
    "is_building",
    "targets_air",
    "targets_ground",
    "range_bucket",
    "speed_bucket",
)
CARD_FEATURE_DIM: int = len(CARD_FEATURE_NAMES)

# Cards whose attacks/spells can damage air units. This is intentionally
# conservative; unknown/new cards default to False and can be corrected here.
_AIR_TARGETING = {
    "arrows", "fireball", "zap", "lightning", "rocket", "poison", "freeze",
    "tornado", "giant snowball", "void", "archers", "spear goblins",
    "musketeer", "three musketeers", "minions", "minion horde", "baby dragon",
    "mega minion", "dart goblin", "electro wizard", "ice wizard", "princess",
    "magic archer", "flying machine", "firecracker", "electro dragon",
    "skeleton dragons", "witch", "mother witch", "hunter", "executioner",
    "zappies", "inferno dragon", "inferno tower", "phoenix", "little prince",
    "archer queen", "rascals", "ice spirit", "electro spirit", "fire spirit",
    "heal spirit", "skeleton king", "golden knight", "mighty miner",
}

# Coarse range buckets: 0 = melee/building-contact, 1 = short, 2 = medium,
# 3 = long/spell. These are intentionally coarse so off-by-one game patches
# do not invalidate the feature matrix.
_LONG_RANGE = {
    "mortar", "x-bow", "royal giant", "princess", "magic archer",
    "dart goblin", "firecracker", "rocket", "lightning", "arrows",
    "fireball", "poison", "freeze", "tornado", "earthquake", "void",
}
_SHORT_RANGE = {
    "bomber", "ice golem", "ice spirit", "fire spirit", "electro spirit",
    "heal spirit", "skeletons", "goblins", "spear goblins", "guards",
    "minions", "bats", "minion horde", "wall breakers", "goblin gang",
    "skeleton army", "barbarians", "royal recruits", "elite barbarians",
}
_MELEE = {
    "knight", "giant", "p.e.k.k.a", "mini p.e.k.k.a", "valkyrie", "golem",
    "goblin giant", "giant skeleton", "pekka", "prince", "dark prince",
    "lumberjack", "bandit", "royal ghost", "mega knight", "electro giant",
    "golden knight", "skeleton king", "mighty miner", "monk", "hog rider",
    "battle ram", "ram rider", "royal hogs", "fisherman", "battle healer",
    "suspicious bush", "goblin machine", "goblin demolisher", "goblin cage",
    "tombstone", "cannon", "bomb tower", "goblin hut", "barbarian hut",
    "furnace", "elixir collector", "tesla",
}
_SLOW = {
    "golem", "p.e.k.k.a", "pekka", "giant skeleton", "lava hound",
    "electro giant", "royal giant", "bowler", "executioner", "inferno tower",
    "elixir collector", "sparky", "giant",
}
_FAST = {
    "hog rider", "battle ram", "bandit", "royal ghost", "lumberjack",
    "elite barbarians", "dark prince", "prince", "miner", "wall breakers",
    "royal hogs", "ram rider", "goblin drill", "goblin giant", "fisherman",
    "royal recruits",
}
_VERY_FAST = {
    "ice spirit", "fire spirit", "electro spirit", "heal spirit", "skeletons",
    "goblins", "spear goblins", "bats", "minions", "minion horde",
    "goblin gang", "skeleton army", "goblin barrel",
}


def _name(card: Dict) -> str:
    return str(card.get("name", "")).strip().lower()


def card_feature_vector(card_id: int) -> np.ndarray:
    """Return the normalized static feature vector for one global card ID."""
    vector = np.zeros(CARD_FEATURE_DIM, dtype=np.float32)
    card = CARD_DATABASE.get(int(card_id))
    if card is None or int(card_id) == 0:
        return vector

    name = _name(card)
    is_spell = bool(card.get("is_spell", False))
    is_building = card.get("category") == "building"
    targets_air = name in _AIR_TARGETING
    targets_ground = True  # Every damaging card can hit ground; effects are separate.

    if name in _MELEE:
        range_bucket = 0
    elif name in _SHORT_RANGE:
        range_bucket = 1
    elif name in _LONG_RANGE or is_spell or is_building:
        range_bucket = 3
    else:
        range_bucket = 2

    if name in _SLOW:
        speed_bucket = 0
    elif name in _VERY_FAST:
        speed_bucket = 3
    elif name in _FAST:
        speed_bucket = 2
    else:
        speed_bucket = 1

    vector[:] = (
        min(float(card.get("cost", 0)) / 10.0, 1.0),
        float(is_spell),
        float(is_building),
        float(targets_air),
        float(targets_ground),
        float(range_bucket) / 3.0,
        float(speed_bucket) / 3.0,
    )
    return vector


def build_card_feature_matrix(num_cards: int = NUM_TOTAL_CARDS) -> np.ndarray:
    """Build (num_cards, CARD_FEATURE_DIM) float32 static features."""
    matrix = np.zeros((num_cards, CARD_FEATURE_DIM), dtype=np.float32)
    for card_id in CARD_DATABASE:
        if 0 <= int(card_id) < num_cards:
            matrix[int(card_id)] = card_feature_vector(int(card_id))
    return matrix


def card_targets_air(card_id: int) -> bool:
    return bool(card_feature_vector(card_id)[3])


def card_targets_ground(card_id: int) -> bool:
    return bool(card_feature_vector(card_id)[4])
