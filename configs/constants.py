"""
Constants and coordinate mappings for CR-Zero (Clash Royale AI Agent).
Standardized on 9:16 portrait viewport and 32x18 arena grid tile conventions.
"""
from typing import Dict, Tuple, List, Optional
import numpy as np

# ==============================================================================
# 1. Grid & Viewport Dimensions
# ==============================================================================

# Arena Grid: 32 vertical rows (0 = enemy king side, 31 = ally king side)
#             18 horizontal columns (0 = leftmost lane, 17 = rightmost lane)
ARENA_GRID: Tuple[int, int] = (32, 18)
ARENA_ROWS: int = ARENA_GRID[0]
ARENA_COLS: int = ARENA_GRID[1]
TOTAL_TILES: int = ARENA_ROWS * ARENA_COLS  # 576

# River line (separating ally half from enemy half)
RIVER_ROW_NORTH: int = 15  # Last row of enemy side
RIVER_ROW_SOUTH: int = 16  # First row of ally side

# Bridge columns (2 tiles wide each)
LEFT_BRIDGE_COLS: Tuple[int, int] = (3, 4)
RIGHT_BRIDGE_COLS: Tuple[int, int] = (13, 14)

# ==============================================================================
# 2. Viewport Bounding Boxes (Normalized 0.0 to 1.0 of 9:16 Game Viewport)
# Box format: (xmin, ymin, xmax, ymax)
# ==============================================================================

# Arena boundary within the 9:16 frame
ARENA_BOX: Tuple[float, float, float, float] = (0.040, 0.100, 0.960, 0.780)

# The 4 playable cards in hand (slots 0..3 from left to right)
HAND_SLOTS: List[Tuple[float, float, float, float]] = [
    (0.130, 0.815, 0.315, 0.955),  # Slot 0
    (0.325, 0.815, 0.510, 0.955),  # Slot 1
    (0.520, 0.815, 0.705, 0.955),  # Slot 2
    (0.715, 0.815, 0.900, 0.955),  # Slot 3
]

# Next upcoming card slot (small preview on the bottom left)
NEXT_CARD_BOX: Tuple[float, float, float, float] = (0.015, 0.835, 0.115, 0.935)

# Elixir bar & numeric text readout
ELIXIR_BAR_BOX: Tuple[float, float, float, float] = (0.170, 0.960, 0.930, 0.995)
ELIXIR_TEXT_BOX: Tuple[float, float, float, float] = (0.095, 0.955, 0.175, 0.998)

# Match timer (top right)
TIMER_BOX: Tuple[float, float, float, float] = (0.750, 0.012, 0.950, 0.055)

# Tower regions (for HP bar detection / status)
TOWER_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    "enemy_king": (0.420, 0.110, 0.580, 0.190),
    "enemy_princess_left": (0.160, 0.200, 0.320, 0.280),
    "enemy_princess_right": (0.680, 0.200, 0.840, 0.280),
    "ally_king": (0.420, 0.690, 0.580, 0.770),
    "ally_princess_left": (0.160, 0.600, 0.320, 0.680),
    "ally_princess_right": (0.680, 0.600, 0.840, 0.680),
}

# ==============================================================================
# 3. Card Database: Metadata, Elixir Cost & Spell Classification
# ID 0 is strictly reserved for "Wait / No-Op / Empty"
# ==============================================================================

# v1 Default Deck: 2.6 Hog Cycle (10 classes: 0=Wait, 1..8=2.6 Hog Cards, 9=Unknown)
# Compatible with KataCR card_classification dataset
DEFAULT_DECK_CLASSES: List[Dict] = [
    {"id": 0, "name": "Wait", "cost": 0, "is_spell": False, "category": "none"},
    {"id": 1, "name": "Hog Rider", "cost": 4, "is_spell": False, "category": "ground"},
    {"id": 2, "name": "Cannon", "cost": 3, "is_spell": False, "category": "building"},
    {"id": 3, "name": "Musketeer", "cost": 4, "is_spell": False, "category": "ground"},
    {"id": 4, "name": "Ice Golem", "cost": 2, "is_spell": False, "category": "ground"},
    {"id": 5, "name": "Ice Spirit", "cost": 1, "is_spell": False, "category": "ground"},
    {"id": 6, "name": "Skeletons", "cost": 1, "is_spell": False, "category": "ground"},
    {"id": 7, "name": "Fireball", "cost": 4, "is_spell": True, "category": "spell"},
    {"id": 8, "name": "The Log", "cost": 2, "is_spell": True, "category": "spell"},
    {"id": 9, "name": "Unknown", "cost": 0, "is_spell": False, "category": "none"},
]

NUM_CARD_CLASSES_V1: int = len(DEFAULT_DECK_CLASSES)  # 10
STACKED_GRID_CHANNELS: int = 32  # 4 frames * 8 channels = 32
NUM_SCALARS: int = 12

DECK_ID_TO_COST: Dict[int, int] = {c["id"]: c["cost"] for c in DEFAULT_DECK_CLASSES}
DECK_ID_TO_IS_SPELL: Dict[int, bool] = {c["id"]: c["is_spell"] for c in DEFAULT_DECK_CLASSES}
DECK_NAME_TO_ID: Dict[str, int] = {c["name"].lower(): c["id"] for c in DEFAULT_DECK_CLASSES}

CARD_DATABASE: Dict[int, Dict] = {
    0: {"name": "Wait", "cost": 0, "is_spell": False, "category": "none"},
    1: {"name": "Arrows", "cost": 3, "is_spell": True, "category": "spell"},
    2: {"name": "Fireball", "cost": 4, "is_spell": True, "category": "spell"},
    3: {"name": "Zap", "cost": 2, "is_spell": True, "category": "spell"},
    4: {"name": "The Log", "cost": 2, "is_spell": True, "category": "spell"},
    5: {"name": "Lightning", "cost": 6, "is_spell": True, "category": "spell"},
    6: {"name": "Rocket", "cost": 6, "is_spell": True, "category": "spell"},
    7: {"name": "Poison", "cost": 4, "is_spell": True, "category": "spell"},
    8: {"name": "Freeze", "cost": 4, "is_spell": True, "category": "spell"},
    9: {"name": "Tornado", "cost": 3, "is_spell": True, "category": "spell"},
    10: {"name": "Earthquake", "cost": 3, "is_spell": True, "category": "spell"},
    11: {"name": "Barbarian Barrel", "cost": 2, "is_spell": True, "category": "spell"},
    12: {"name": "Goblin Barrel", "cost": 3, "is_spell": True, "category": "spell"},
    13: {"name": "Giant Snowball", "cost": 2, "is_spell": True, "category": "spell"},
    14: {"name": "Royal Delivery", "cost": 3, "is_spell": True, "category": "spell"},
    15: {"name": "Void", "cost": 3, "is_spell": True, "category": "spell"},
    16: {"name": "Graveyard", "cost": 5, "is_spell": True, "category": "spell"},
    17: {"name": "Clone", "cost": 3, "is_spell": True, "category": "spell"},
    18: {"name": "Mirror", "cost": 1, "is_spell": False, "category": "spell"},
    19: {"name": "Rage", "cost": 2, "is_spell": True, "category": "spell"},
    20: {"name": "Knight", "cost": 3, "is_spell": False, "category": "ground"},
    21: {"name": "Archers", "cost": 3, "is_spell": False, "category": "ground"},
    22: {"name": "Goblins", "cost": 2, "is_spell": False, "category": "ground"},
    23: {"name": "Spear Goblins", "cost": 2, "is_spell": False, "category": "ground"},
    24: {"name": "Giant", "cost": 5, "is_spell": False, "category": "ground"},
    25: {"name": "P.E.K.K.A", "cost": 7, "is_spell": False, "category": "ground"},
    26: {"name": "Mini P.E.K.K.A", "cost": 4, "is_spell": False, "category": "ground"},
    27: {"name": "Musketeer", "cost": 4, "is_spell": False, "category": "ground"},
    28: {"name": "Three Musketeers", "cost": 9, "is_spell": False, "category": "ground"},
    29: {"name": "Baby Dragon", "cost": 4, "is_spell": False, "category": "air"},
    30: {"name": "Skeletons", "cost": 1, "is_spell": False, "category": "ground"},
    31: {"name": "Skeleton Army", "cost": 3, "is_spell": False, "category": "ground"},
    32: {"name": "Witch", "cost": 5, "is_spell": False, "category": "ground"},
    33: {"name": "Valkyrie", "cost": 4, "is_spell": False, "category": "ground"},
    34: {"name": "Bomber", "cost": 2, "is_spell": False, "category": "ground"},
    35: {"name": "Cannon", "cost": 3, "is_spell": False, "category": "building"},
    36: {"name": "Barbarians", "cost": 5, "is_spell": False, "category": "ground"},
    37: {"name": "Golem", "cost": 8, "is_spell": False, "category": "ground"},
    38: {"name": "Tombstone", "cost": 3, "is_spell": False, "category": "building"},
    39: {"name": "Bomb Tower", "cost": 4, "is_spell": False, "category": "building"},
    40: {"name": "Goblin Hut", "cost": 5, "is_spell": False, "category": "building"},
    41: {"name": "Barbarian Hut", "cost": 6, "is_spell": False, "category": "building"},
    42: {"name": "Inferno Tower", "cost": 5, "is_spell": False, "category": "building"},
    43: {"name": "Minions", "cost": 3, "is_spell": False, "category": "air"},
    44: {"name": "Minion Horde", "cost": 5, "is_spell": False, "category": "air"},
    45: {"name": "Elixir Collector", "cost": 6, "is_spell": False, "category": "building"},
    46: {"name": "Balloon", "cost": 5, "is_spell": False, "category": "air"},
    47: {"name": "Giant Skeleton", "cost": 6, "is_spell": False, "category": "ground"},
    48: {"name": "Guards", "cost": 3, "is_spell": False, "category": "ground"},
    49: {"name": "Princess", "cost": 3, "is_spell": False, "category": "ground"},
    50: {"name": "Ice Wizard", "cost": 3, "is_spell": False, "category": "ground"},
    51: {"name": "Hog Rider", "cost": 4, "is_spell": False, "category": "ground"},
    52: {"name": "Dark Prince", "cost": 4, "is_spell": False, "category": "ground"},
    53: {"name": "Prince", "cost": 5, "is_spell": False, "category": "ground"},
    54: {"name": "Lava Hound", "cost": 7, "is_spell": False, "category": "air"},
    55: {"name": "Miner", "cost": 3, "is_spell": True, "category": "ground"},  # Placed anywhere like a spell!
    56: {"name": "Sparky", "cost": 6, "is_spell": False, "category": "ground"},
    57: {"name": "Bowler", "cost": 5, "is_spell": False, "category": "ground"},
    58: {"name": "Lumberjack", "cost": 4, "is_spell": False, "category": "ground"},
    59: {"name": "Battle Ram", "cost": 4, "is_spell": False, "category": "ground"},
    60: {"name": "Inferno Dragon", "cost": 4, "is_spell": False, "category": "air"},
    61: {"name": "Ice Golem", "cost": 2, "is_spell": False, "category": "ground"},
    62: {"name": "Mega Minion", "cost": 3, "is_spell": False, "category": "air"},
    63: {"name": "Dart Goblin", "cost": 3, "is_spell": False, "category": "ground"},
    64: {"name": "Goblin Gang", "cost": 3, "is_spell": False, "category": "ground"},
    65: {"name": "Electro Wizard", "cost": 4, "is_spell": False, "category": "ground"},
    66: {"name": "Elite Barbarians", "cost": 6, "is_spell": False, "category": "ground"},
    67: {"name": "Hunter", "cost": 4, "is_spell": False, "category": "ground"},
    68: {"name": "Executioner", "cost": 5, "is_spell": False, "category": "ground"},
    69: {"name": "Bandit", "cost": 3, "is_spell": False, "category": "ground"},
    70: {"name": "Royal Recruits", "cost": 7, "is_spell": False, "category": "ground"},
    71: {"name": "Night Witch", "cost": 4, "is_spell": False, "category": "ground"},
    72: {"name": "Bats", "cost": 2, "is_spell": False, "category": "air"},
    73: {"name": "Royal Ghost", "cost": 3, "is_spell": False, "category": "ground"},
    74: {"name": "Ram Rider", "cost": 5, "is_spell": False, "category": "ground"},
    75: {"name": "Zappies", "cost": 4, "is_spell": False, "category": "ground"},
    76: {"name": "Rascals", "cost": 5, "is_spell": False, "category": "ground"},
    77: {"name": "Cannon Cart", "cost": 5, "is_spell": False, "category": "ground"},
    78: {"name": "Mega Knight", "cost": 7, "is_spell": False, "category": "ground"},
    79: {"name": "Skeleton Barrel", "cost": 3, "is_spell": False, "category": "air"},
    80: {"name": "Flying Machine", "cost": 4, "is_spell": False, "category": "air"},
    81: {"name": "Wall Breakers", "cost": 2, "is_spell": False, "category": "ground"},
    82: {"name": "Royal Hogs", "cost": 5, "is_spell": False, "category": "ground"},
    83: {"name": "Goblin Giant", "cost": 6, "is_spell": False, "category": "ground"},
    84: {"name": "Fisherman", "cost": 3, "is_spell": False, "category": "ground"},
    85: {"name": "Magic Archer", "cost": 4, "is_spell": False, "category": "ground"},
    86: {"name": "Electro Dragon", "cost": 5, "is_spell": False, "category": "air"},
    87: {"name": "Firecracker", "cost": 3, "is_spell": False, "category": "ground"},
    88: {"name": "Elixir Golem", "cost": 3, "is_spell": False, "category": "ground"},
    89: {"name": "Battle Healer", "cost": 4, "is_spell": False, "category": "ground"},
    90: {"name": "Skeleton Dragons", "cost": 4, "is_spell": False, "category": "air"},
    91: {"name": "Mother Witch", "cost": 4, "is_spell": False, "category": "ground"},
    92: {"name": "Electro Spirit", "cost": 1, "is_spell": False, "category": "ground"},
    93: {"name": "Electro Giant", "cost": 7, "is_spell": False, "category": "ground"},
    94: {"name": "Golden Knight", "cost": 4, "is_spell": False, "category": "ground"},
    95: {"name": "Archer Queen", "cost": 5, "is_spell": False, "category": "ground"},
    96: {"name": "Skeleton King", "cost": 4, "is_spell": False, "category": "ground"},
    97: {"name": "Mighty Miner", "cost": 4, "is_spell": False, "category": "ground"},
    98: {"name": "Monk", "cost": 5, "is_spell": False, "category": "ground"},
    99: {"name": "Phoenix", "cost": 4, "is_spell": False, "category": "air"},
    100: {"name": "Little Prince", "cost": 3, "is_spell": False, "category": "ground"},
    101: {"name": "Heal Spirit", "cost": 1, "is_spell": False, "category": "ground"},
    102: {"name": "Ice Spirit", "cost": 1, "is_spell": False, "category": "ground"},
    103: {"name": "Fire Spirit", "cost": 1, "is_spell": False, "category": "ground"},
    104: {"name": "Goblin Demolisher", "cost": 4, "is_spell": False, "category": "ground"},
    105: {"name": "Goblin Machine", "cost": 5, "is_spell": False, "category": "ground"},
    106: {"name": "Goblin Curse", "cost": 2, "is_spell": True, "category": "spell"},
    107: {"name": "Suspicious Bush", "cost": 2, "is_spell": False, "category": "ground"},
    108: {"name": "Royal Giant", "cost": 6, "is_spell": False, "category": "ground"},
    109: {"name": "Mortar", "cost": 4, "is_spell": False, "category": "building"},
    110: {"name": "X-Bow", "cost": 6, "is_spell": False, "category": "building"},
    111: {"name": "Furnace", "cost": 4, "is_spell": False, "category": "building"},
    112: {"name": "Goblin Cage", "cost": 4, "is_spell": False, "category": "building"},
    113: {"name": "Goblin Drill", "cost": 4, "is_spell": True, "category": "building"},  # Placed anywhere
}

# Total card vocabulary size for embedding layers
NUM_TOTAL_CARDS: int = 135

# Fast lookup dictionaries
CARD_ID_TO_COST: Dict[int, int] = {
    cid: meta["cost"] for cid, meta in CARD_DATABASE.items()
}
CARD_ID_TO_IS_SPELL: Dict[int, bool] = {
    cid: meta["is_spell"] for cid, meta in CARD_DATABASE.items()
}
CARD_ID_TO_CATEGORY: Dict[int, str] = {
    cid: meta["category"] for cid, meta in CARD_DATABASE.items()
}
CARD_NAME_TO_ID: Dict[str, int] = {
    meta["name"].lower(): cid for cid, meta in CARD_DATABASE.items()
}

# ==============================================================================
# 4. Coordinate Conversion Functions (Normalized, Grid & Pixel)
# ==============================================================================

def grid_to_normalized_arena(tile_y: int, tile_x: int) -> Tuple[float, float]:
    """
    Converts a grid tile (tile_y: 0..31, tile_x: 0..17) to normalized arena coordinates [0.0, 1.0].
    Returns (norm_x, norm_y) relative to the ARENA_BOX.
    """
    norm_x = (tile_x + 0.5) / ARENA_COLS
    norm_y = (tile_y + 0.5) / ARENA_ROWS
    return norm_x, norm_y


def normalized_arena_to_grid(norm_x: float, norm_y: float) -> Tuple[int, int]:
    """
    Converts normalized arena coordinates [0.0, 1.0] to grid (tile_y, tile_x).
    """
    tile_x = int(np.clip(np.floor(norm_x * ARENA_COLS), 0, ARENA_COLS - 1))
    tile_y = int(np.clip(np.floor(norm_y * ARENA_ROWS), 0, ARENA_ROWS - 1))
    return tile_y, tile_x


def screen_to_grid(screen_norm_x: float, screen_norm_y: float) -> Optional[Tuple[int, int]]:
    """
    Converts full-screen normalized coordinates (0..1) into arena grid (tile_y, tile_x).
    Returns None if outside the ARENA_BOX.
    """
    ax1, ay1, ax2, ay2 = ARENA_BOX
    if not (ax1 <= screen_norm_x <= ax2 and ay1 <= screen_norm_y <= ay2):
        return None
    rel_x = (screen_norm_x - ax1) / (ax2 - ax1)
    rel_y = (screen_norm_y - ay1) / (ay2 - ay1)
    return normalized_arena_to_grid(rel_x, rel_y)


def grid_to_screen_pixel(
    tile_y: int,
    tile_x: int,
    screen_width: int,
    screen_height: int
) -> Tuple[int, int]:
    """
    Converts grid (tile_y, tile_x) to absolute pixel tap coordinate on the screen.
    Compatible with BlueStacks 540x960, 720x1280, or arbitrary resolution.
    """
    ax1, ay1, ax2, ay2 = ARENA_BOX
    arena_px_w = (ax2 - ax1) * screen_width
    arena_px_h = (ay2 - ay1) * screen_height

    norm_x, norm_y = grid_to_normalized_arena(tile_y, tile_x)
    pixel_x = int(round(ax1 * screen_width + norm_x * arena_px_w))
    pixel_y = int(round(ay1 * screen_height + norm_y * arena_px_h))
    return pixel_x, pixel_y


def hand_slot_to_screen_pixel(
    slot_idx: int,
    screen_width: int,
    screen_height: int
) -> Tuple[int, int]:
    """
    Returns the center tap pixel coordinate for hand slot 0..3.
    """
    if not (0 <= slot_idx <= 3):
        raise ValueError(f"Hand slot must be in 0..3, got {slot_idx}")
    x1, y1, x2, y2 = HAND_SLOTS[slot_idx]
    center_x = int(round(((x1 + x2) / 2.0) * screen_width))
    center_y = int(round(((y1 + y2) / 2.0) * screen_height))
    return center_x, center_y


# ==============================================================================
# 5. ClashRoyaleBuildABot Coordinate Interop
# ClashRoyaleBuildABot uses a coordinate convention where:
# - Left Princess Tower is (-3000, -6500)
# - Right Princess Tower is (3000, -6500)
# - Enemy King Tower is (0, 8500)
# - Range: X: [-7000, 7000], Y: [-10000, 10000]
# ==============================================================================

def grid_to_clashroyalebuildabot(tile_y: int, tile_x: int) -> Tuple[int, int]:
    """
    Maps (tile_y: 0..31, tile_x: 0..17) to ClashRoyaleBuildABot millimeter scale.
    tile_y = 0 (top, enemy) -> Y = +9000
    tile_y = 31 (bottom, ally) -> Y = -9000
    tile_x = 0 (left) -> X = -6500
    tile_x = 17 (right) -> X = +6500
    """
    bot_x = int(np.interp(tile_x + 0.5, [0, ARENA_COLS], [-6500, 6500]))
    bot_y = int(np.interp(tile_y + 0.5, [0, ARENA_ROWS], [9000, -9000]))
    return bot_x, bot_y


def clashroyalebuildabot_to_grid(bot_x: int, bot_y: int) -> Tuple[int, int]:
    """
    Maps ClashRoyaleBuildABot coordinate (X: -6500..6500, Y: -9000..9000) to grid (tile_y, tile_x).
    """
    tile_x = int(np.clip(np.round(np.interp(bot_x, [-6500, 6500], [0, ARENA_COLS - 1])), 0, ARENA_COLS - 1))
    tile_y = int(np.clip(np.round(np.interp(bot_y, [-9000, 9000], [ARENA_ROWS - 1, 0])), 0, ARENA_ROWS - 1))
    return tile_y, tile_x


# ==============================================================================
# 6. Deployable Zone Mask Generator
# ==============================================================================

def get_default_deployable_mask(
    enemy_left_down: bool = False,
    enemy_right_down: bool = False
) -> np.ndarray:
    """
    Generates a boolean mask of shape (32, 18) indicating legally deployable tiles
    for regular non-spell troops.
    - Default: rows 16 to 31 (ally half)
    - If enemy left princess tower is destroyed: rows 11 to 15 on cols 0..8 become deployable
    - If enemy right princess tower is destroyed: rows 11 to 15 on cols 9..17 become deployable
    """
    mask = np.zeros(ARENA_GRID, dtype=bool)

    # Ally territory is rows 16 to 31 across all columns
    mask[RIVER_ROW_SOUTH:ARENA_ROWS, :] = True

    # Left pocket expansion if enemy left tower is destroyed
    if enemy_left_down:
        mask[11:RIVER_ROW_SOUTH, 0:9] = True

    # Right pocket expansion if enemy right tower is destroyed
    if enemy_right_down:
        mask[11:RIVER_ROW_SOUTH, 9:18] = True

    return mask
