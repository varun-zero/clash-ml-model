"""Static card features and tactical combat roles for the global 135-card vocabulary.

The learned per-ID embedding captures card identity. These features give the
policy a valid signal even for a card ID that never appeared in training:
``Embedding(id) + Linear(features(id))``. IDs missing from CARD_DATABASE get a
zero feature vector and still have a trainable embedding row.

Also provides tactical combat role classifications, intrinsic threat scoring,
and counter-matchup evaluation used by the human-like tactical engine.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from configs.constants import (
    CARD_DATABASE,
    CARD_ID_TO_CATEGORY,
    CARD_ID_TO_COST,
    CARD_ID_TO_IS_SPELL,
    NUM_TOTAL_CARDS,
)

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
    "wizard", "tesla", "goblin hut", "furnace", "spirit empress", "vines",
    "goblin curse", "royal delivery", "rage",
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
    "furnace", "elixir collector", "tesla", "berserker", "boss bandit",
    "rune giant", "ronin",
}
_SLOW = {
    "golem", "p.e.k.k.a", "pekka", "giant skeleton", "lava hound",
    "electro giant", "royal giant", "bowler", "executioner", "inferno tower",
    "elixir collector", "sparky", "giant", "rune giant",
}
_FAST = {
    "hog rider", "battle ram", "bandit", "royal ghost", "lumberjack",
    "elite barbarians", "dark prince", "prince", "miner", "wall breakers",
    "royal hogs", "ram rider", "goblin drill", "goblin giant", "fisherman",
    "royal recruits", "boss bandit", "berserker",
}
_VERY_FAST = {
    "ice spirit", "fire spirit", "electro spirit", "heal spirit", "skeletons",
    "goblins", "spear goblins", "bats", "minions", "minion horde",
    "goblin gang", "skeleton army", "goblin barrel",
}

# ==============================================================================
# Tactical Combat Role Sets (by normalized card name)
# ==============================================================================

_WIN_CONDITIONS = {
    "giant", "hog rider", "balloon", "golem", "royal giant", "goblin barrel",
    "battle ram", "ram rider", "x-bow", "mortar", "lava hound", "graveyard",
    "electro giant", "wall breakers", "royal hogs", "goblin drill", "miner",
    "three musketeers", "sparky", "goblin giant", "elixir golem", "rune giant",
}

_BUILDING_TARGETERS = {
    "giant", "hog rider", "balloon", "golem", "royal giant", "battle ram",
    "ram rider", "lava hound", "electro giant", "wall breakers", "royal hogs",
    "goblin giant", "elixir golem", "ice golem", "skeleton barrel", "rune giant",
}

_TANKS = {
    "giant", "golem", "p.e.k.k.a", "lava hound", "mega knight", "electro giant",
    "royal giant", "giant skeleton", "goblin giant", "rune giant", "elixir golem",
}

_MINI_TANKS = {
    "knight", "valkyrie", "mini p.e.k.k.a", "dark prince", "prince", "ice golem",
    "battle ram", "bandit", "royal ghost", "lumberjack", "golden knight",
    "skeleton king", "mighty miner", "monk", "battle healer", "bowler",
    "cannon cart", "rascals", "goblin machine", "goblinstein", "boss bandit",
    "ronin", "berserker",
}

_TANK_KILLERS = {
    "mini p.e.k.k.a", "p.e.k.k.a", "prince", "inferno tower", "inferno dragon",
    "hunter", "mighty miner", "skeleton army", "barbarians", "elite barbarians",
    "goblins", "minion horde", "lumberjack", "goblin gang", "guards",
    "night witch", "little prince", "archer queen", "sparky",
}

_RANGED_SUPPORT = {
    "musketeer", "archers", "spear goblins", "bomber", "baby dragon", "witch",
    "wizard", "ice wizard", "electro wizard", "princess", "dart goblin",
    "executioner", "magic archer", "firecracker", "flying machine",
    "mother witch", "archer queen", "little prince", "skeleton dragons",
    "zappies", "three musketeers", "electro dragon", "hunter", "phoenix",
    "mega minion", "spirit empress",
}

_SPLASH_UNITS = {
    "bomber", "valkyrie", "baby dragon", "witch", "wizard", "dark prince",
    "princess", "ice wizard", "bowler", "executioner", "mega knight",
    "magic archer", "firecracker", "skeleton dragons", "skeleton king",
    "fire spirit", "ice spirit", "electro spirit", "bomb tower", "mortar",
    "goblin demolisher", "royal ghost", "sparky",
}

_SWARM_UNITS = {
    "goblins", "spear goblins", "skeletons", "skeleton army", "minions",
    "minion horde", "barbarians", "goblin barrel", "guards", "goblin gang",
    "bats", "royal recruits", "rascals", "royal hogs", "wall breakers",
    "skeleton barrel", "archers",
}

_SPELL_BAIT_SWARMS = {
    "goblin barrel", "skeleton army", "minion horde", "goblin gang",
    "princess", "dart goblin", "firecracker", "minions", "bats",
    "rascals", "wall breakers", "skeleton barrel",
}

_SMALL_SPELLS = {
    "arrows", "zap", "the log", "tornado", "earthquake", "barbarian barrel",
    "giant snowball", "royal delivery", "rage", "goblin curse", "vines",
}

_BIG_SPELLS = {
    "fireball", "lightning", "rocket", "poison", "freeze", "void",
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


def get_card_roles(card_id: int) -> Set[str]:
    """Return the set of tactical combat roles for a global card ID."""
    card = CARD_DATABASE.get(int(card_id))
    if card is None or int(card_id) == 0:
        return set()

    name = _name(card)
    roles: Set[str] = set()
    cat = str(card.get("category", "ground")).lower()
    is_spell = bool(card.get("is_spell", False))

    if name in _WIN_CONDITIONS:
        roles.add("win_condition")
    if name in _BUILDING_TARGETERS:
        roles.add("building_targeter")
    if name in _TANKS:
        roles.add("tank")
    if name in _MINI_TANKS:
        roles.add("mini_tank")
    if name in _TANK_KILLERS:
        roles.add("tank_killer")
    if name in _RANGED_SUPPORT:
        roles.add("ranged_support")
    if name in _SPLASH_UNITS:
        roles.add("splash")
    if name in _SWARM_UNITS:
        roles.add("swarm")
    if name in _SPELL_BAIT_SWARMS:
        roles.add("spell_bait")
    if name in _SMALL_SPELLS:
        roles.add("small_spell")
    if name in _BIG_SPELLS:
        roles.add("big_spell")
    if name in _AIR_TARGETING:
        roles.add("anti_air")
    if name in _MELEE:
        roles.add("melee")
    if name in _FAST or name in _VERY_FAST:
        roles.add("fast")
    if cat == "air":
        roles.add("air_unit")
    elif cat == "building":
        roles.add("building")
    elif is_spell and cat == "spell":
        roles.add("spell")

    return roles


def is_swarm_card(card_id: int) -> bool:
    return "swarm" in get_card_roles(card_id)


def is_tank_card(card_id: int) -> bool:
    roles = get_card_roles(card_id)
    return "tank" in roles or "mini_tank" in roles


def is_win_condition_card(card_id: int) -> bool:
    return "win_condition" in get_card_roles(card_id)


def is_splash_card(card_id: int) -> bool:
    return "splash" in get_card_roles(card_id)


def is_ranged_support_card(card_id: int) -> bool:
    return "ranged_support" in get_card_roles(card_id)


def is_tank_killer_card(card_id: int) -> bool:
    return "tank_killer" in get_card_roles(card_id)


def is_building_targeter_card(card_id: int) -> bool:
    return "building_targeter" in get_card_roles(card_id)


def get_intrinsic_danger(card_id: int, category: str = "ground") -> float:
    """
    Computes the base intrinsic threat score (1.0 .. 10.0) of an enemy unit.
    Accounts for win-condition priority, elixir cost, speed, and category.
    """
    cid = int(card_id)
    if cid <= 0 or cid not in CARD_DATABASE:
        # Generic detected unit when class ID is unknown
        if category == "air":
            return 5.2
        elif category == "building":
            return 4.5
        return 4.8

    cost = float(CARD_ID_TO_COST.get(cid, 3))
    roles = get_card_roles(cid)

    score = cost * 1.15
    if "win_condition" in roles:
        score += 2.8
    if "building_targeter" in roles:
        score += 1.2
    if "tank_killer" in roles:
        score += 1.5
    if "fast" in roles and ("win_condition" in roles or "tank_killer" in roles):
        score += 1.4
    if "air_unit" in roles:
        score += 0.8
    if "spell_bait" in roles and cid == 12:  # Goblin Barrel direct tower threat
        score += 3.0

    return float(np.clip(score, 1.5, 10.0))


def score_counter_matchup(
    defender_cid: int,
    enemy_cid: int,
    enemy_category: str = "ground",
    cluster_size: int = 1,
    cluster_elixir: float = 3.0,
) -> float:
    """
    Scores how well ``defender_cid`` counters an enemy unit or cluster.
    Higher score = stronger human-like counter choice and positive elixir trade.
    Returns negative values for invalid or poor counters.
    """
    defender_cid = int(defender_cid)
    enemy_cid = int(enemy_cid)
    if defender_cid <= 0 or defender_cid not in CARD_DATABASE:
        return -999.0

    def_cost = float(CARD_ID_TO_COST.get(defender_cid, 3))
    def_roles = get_card_roles(defender_cid)
    def_is_spell = bool(CARD_ID_TO_IS_SPELL.get(defender_cid, False) and CARD_ID_TO_CATEGORY.get(defender_cid) == "spell")

    enemy_cost = float(CARD_ID_TO_COST.get(enemy_cid, 4)) if enemy_cid > 0 else max(3.5, cluster_elixir)
    enemy_roles = get_card_roles(enemy_cid) if enemy_cid > 0 else set()
    is_air_enemy = (enemy_category == "air") or ("air_unit" in enemy_roles)
    is_swarm_enemy = (cluster_size >= 3) or ("swarm" in enemy_roles) or ("spell_bait" in enemy_roles)
    is_tank_enemy = ("tank" in enemy_roles) or ("building_targeter" in enemy_roles) or (enemy_cost >= 5.0 and not is_swarm_enemy)
    is_splash_enemy = ("splash" in enemy_roles)
    is_ranged_enemy = ("ranged_support" in enemy_roles)
    is_melee_heavy = ("tank_killer" in enemy_roles and "melee" in enemy_roles and not is_swarm_enemy)

    # 1. Hard rule: Air threats require air-targeting cards (or cheap building/tank kite in emergencies)
    if is_air_enemy and "anti_air" not in def_roles:
        if "building" in def_roles and ("building_targeter" in enemy_roles):
            return 2.5  # Building can pull Balloon/Lava Hound
        return -100.0

    # 2. Spells should NEVER be used as generic single-target troop counters unless swarm/barrel
    if def_is_spell:
        if defender_cid == 1:  # Arrows (cost 3)
            if enemy_cid in (12, 31, 44, 64, 49, 63, 87, 43, 72) or is_swarm_enemy:
                return 8.5 + min(4.0, (cluster_elixir - def_cost) * 1.2)
            return -25.0  # Never waste Arrows on a single Knight/Giant/Mini PEKKA
        if defender_cid == 2:  # Fireball (cost 4)
            if cluster_size >= 2 and cluster_elixir >= 5.0:
                return 7.5 + min(4.0, (cluster_elixir - def_cost) * 1.0)
            if enemy_cid in (28, 36, 32, 115, 27, 80, 85, 75) and cluster_size >= 2:
                return 8.0
            return -30.0  # Never waste Fireball on a single ground troop when troops can counter

    # 3. Offensive building-targeters (like Giant) cannot attack enemy troops!
    # Only use Giant defensively as a desperation kite sponge if nothing else is available.
    if "building_targeter" in def_roles and "building" not in def_roles:
        if defender_cid == 61:  # Ice Golem is a great 2-elixir kite
            return 4.5 if (is_melee_heavy or is_ranged_enemy) else 1.0
        return -8.0

    score = 5.0

    # 4. Matchup: Defending against Enemy Swarm (Skeleton Army, Minions, Goblin Gang, etc.)
    if is_swarm_enemy:
        if "splash" in def_roles:
            score += 5.5  # Bomber, Valkyrie, Baby Dragon crush swarms
        elif "ranged_support" in def_roles and def_cost <= 3.0:
            score += 3.5  # Archers, Spear Goblins, Minions clean up swarms well
        elif "swarm" in def_roles and def_cost <= 3.0:
            score += 2.5
        if defender_cid == 26:  # Mini P.E.K.K.A is terrible against swarms!
            score -= 6.0

    # 5. Matchup: Defending against Enemy Tank / Building-Targeter (Giant, Hog Rider, Golem, Balloon)
    if is_tank_enemy:
        if "tank_killer" in def_roles:
            score += 6.0  # Mini P.E.K.K.A, Goblins, Minions shred tanks
        elif "building" in def_roles:
            score += 5.0  # Cannon, Tesla, Inferno pull and DPS tanks
        elif "swarm" in def_roles and not is_splash_enemy:
            score += 4.0  # Goblins / Minions / Skeletons high DPS on single-target tank
        elif "mini_tank" in def_roles:
            score += 2.5  # Knight provides solid body + DPS
        elif "ranged_support" in def_roles:
            score += 3.0  # Musketeer / Archers provide safe ranged DPS
        if defender_cid == 34 and cluster_size == 1:
            score -= 2.0  # Bomber alone has lower single-target DPS vs Giant/Hog

    # 6. Matchup: Defending against Enemy Melee Heavy Hitter (Mini P.E.K.K.A, Prince, P.E.K.K.A, Knight)
    if is_melee_heavy:
        if "swarm" in def_roles:
            score += 5.5  # Goblins, Spear Goblins, Minions, Archers distract and melt single-target melee
        elif "mini_tank" in def_roles and def_cost <= 3.0:
            score += 4.5  # Knight in center kill-zone tanks and counters
        elif "ranged_support" in def_roles and def_cost <= 3.0:
            score += 4.0  # Archers / Spear Goblins pull to center

    # 7. Matchup: Defending against Enemy Splash / Ranged Support (Witch, Wizard, Musketeer, Bomber, Baby Dragon)
    if is_splash_enemy or is_ranged_enemy:
        if "swarm" in def_roles and is_splash_enemy and def_cost <= 3.0 and defender_cid in (22, 23, 30, 31, 64, 72):
            score -= 5.5  # Never feed squishy Goblins/Skeletons into Bomber/Valkyrie/Witch/Baby Dragon!
        if "mini_tank" in def_roles or defender_cid in (20, 26, 33):
            score += 5.5  # Knight / Mini P.E.K.K.A dropped directly on top assassinates ranged/splash units
        elif defender_cid == 27:  # Musketeer duels air/ranged units well
            score += 4.5

    # 8. Elixir efficiency bonus (favor positive elixir trades)
    elixir_trade = enemy_cost - def_cost
    score += float(np.clip(elixir_trade * 0.75, -2.5, 3.0))

    return float(score)
