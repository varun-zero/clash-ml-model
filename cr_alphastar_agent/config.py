"""
Configuration and Hardware Profiles for AlphaStar + OpenAI Five Clash Royale Agent.
Auto-detects Colab Tesla T4 (16GB VRAM) vs Local RTX 3050 (4-6GB VRAM).
"""

import os
import torch
from typing import Dict, List, Set

# Arena Grid Dimensions
ARENA_WIDTH = 18
ARENA_HEIGHT = 32
TOTAL_TILES = ARENA_WIDTH * ARENA_HEIGHT  # 576

# Game Simulation Constants
TICKS_PER_SEC = 10
STEP_DT = 1.0 / TICKS_PER_SEC  # 0.1 seconds per physics tick
ACTION_REPEAT = 4  # 4 physics ticks per agent step (~0.4s / 2.5Hz cadence)
REGULAR_TIME_SECS = 180  # 3 minutes (1800 physics ticks)
OVERTIME_SECS = 60       # 1 minute overtime
MAX_MATCH_STEPS = (REGULAR_TIME_SECS + OVERTIME_SECS) * TICKS_PER_SEC  # 2400 physics ticks

# Elixir Generation
ELIXIR_START = 5.0
ELIXIR_MAX = 10.0
ELIXIR_RATE_NORMAL = 0.35  # elixir per second (~2.8s per elixir)
ELIXIR_RATE_DOUBLE = 0.70  # double elixir in last minute & overtime

# Cards Catalog (16 cards for deck variability)
CARDS_CATALOG: Dict[int, dict] = {
    1: {"name": "Knight", "cost": 3, "hp": 1400, "dmg": 160, "range": 1.2, "speed": 1.2, "air": False, "targets": "ground", "count": 1, "spell": False},
    2: {"name": "Archers", "cost": 3, "hp": 250, "dmg": 85, "range": 5.0, "speed": 1.2, "air": False, "targets": "both", "count": 2, "spell": False},
    3: {"name": "Giant", "cost": 5, "hp": 3300, "dmg": 210, "range": 1.2, "speed": 0.9, "air": False, "targets": "building", "count": 1, "spell": False},
    4: {"name": "Musketeer", "cost": 4, "hp": 600, "dmg": 180, "range": 6.0, "speed": 1.2, "air": False, "targets": "both", "count": 1, "spell": False},
    5: {"name": "Mini PEKKA", "cost": 4, "hp": 1100, "dmg": 600, "range": 1.2, "speed": 1.8, "air": False, "targets": "ground", "count": 1, "spell": False},
    6: {"name": "Skeletons", "cost": 1, "hp": 70, "dmg": 70, "range": 1.0, "speed": 1.5, "air": False, "targets": "ground", "count": 3, "spell": False},
    7: {"name": "Minions", "cost": 3, "hp": 190, "dmg": 84, "range": 2.0, "speed": 1.5, "air": True, "targets": "both", "count": 3, "spell": False},
    8: {"name": "Fireball", "cost": 4, "hp": 0, "dmg": 570, "range": 2.5, "speed": 0, "air": True, "targets": "both", "count": 1, "spell": True},
    9: {"name": "Arrows", "cost": 3, "hp": 0, "dmg": 300, "range": 4.0, "speed": 0, "air": True, "targets": "both", "count": 1, "spell": True},
    10: {"name": "Valkyrie", "cost": 4, "hp": 1650, "dmg": 220, "range": 1.2, "speed": 1.2, "air": False, "targets": "ground", "count": 1, "spell": False, "splash": 2.0},
    11: {"name": "Hog Rider", "cost": 4, "hp": 1400, "dmg": 260, "range": 1.2, "speed": 2.1, "air": False, "targets": "building", "count": 1, "spell": False},
    12: {"name": "Baby Dragon", "cost": 4, "hp": 950, "dmg": 133, "range": 3.5, "speed": 1.5, "air": True, "targets": "both", "count": 1, "spell": False, "splash": 1.5},
    13: {"name": "Cannon", "cost": 3, "hp": 750, "dmg": 127, "range": 5.5, "speed": 0, "air": False, "targets": "ground", "count": 1, "spell": False, "building": True},
    14: {"name": "Goblin Barrel", "cost": 3, "hp": 160, "dmg": 95, "range": 1.0, "speed": 1.5, "air": False, "targets": "ground", "count": 3, "spell": True},
    15: {"name": "Mega Minion", "cost": 3, "hp": 690, "dmg": 250, "range": 1.6, "speed": 1.2, "air": True, "targets": "both", "count": 1, "spell": False},
    16: {"name": "Zap", "cost": 2, "hp": 0, "dmg": 160, "range": 2.5, "speed": 0, "air": True, "targets": "both", "count": 1, "spell": True, "stun": 0.5},
}

SPELL_CARD_IDS: Set[int] = {8, 9, 14, 16}
NUM_CARDS = 16
DEFAULT_DECK = ["HogRider", "Musketeer", "Cannon", "IceGolemite", "Skeletons", "IceSpirits", "Fireball", "Log"]

# Standard Competitive Archetypes for AlphaStar Staged Curriculum
POPULAR_DECKS: Dict[str, list] = {
    "PEKKA Bridge Spam": ["Pekka", "BattleRam", "DarkPrince", "Ghost", "Musketeer", "Minions", "Poison", "Zap"],
    "Classic Log Bait": ["GoblinBarrel", "Princess", "GoblinGang", "Knight", "Skeletons", "InfernoTower", "Log", "Fireball"],
    "Hog 2.6 Cycle": ["HogRider", "Musketeer", "Cannon", "IceGolemite", "Skeletons", "IceSpirits", "Fireball", "Log"],
    "Giant Beatdown": ["Giant", "Witch", "MiniPekka", "Musketeer", "MegaMinion", "Fireball", "Zap", "Minions"],
    "Double Dragon Beatdown": ["Golem", "BabyDragon", "InfernoDragon", "MegaMinion", "DarkWitch", "DarkPrince", "Poison", "Zap"],
    "Mega Knight Bait": ["MegaKnight", "InfernoDragon", "Princess", "GoblinGang", "Bats", "GoblinBarrel", "Zap", "Fireball"],
    "Lavaloon Air Beatdown": ["LavaHound", "Balloon", "MegaMinion", "Minions", "Fireball", "Zap", "InfernoDragon", "DarkPrince"],
    # Aliases
    "Bridge Spam": ["Pekka", "BattleRam", "DarkPrince", "Ghost", "Musketeer", "Minions", "Poison", "Zap"],
    "Spell Bait": ["GoblinBarrel", "Princess", "GoblinGang", "Knight", "Skeletons", "InfernoTower", "Log", "Fireball"],
    "Log Bait": ["GoblinBarrel", "Princess", "GoblinGang", "Knight", "Skeletons", "InfernoTower", "Log", "Fireball"],
    "Pekka Control": ["Pekka", "BattleRam", "DarkPrince", "Ghost", "Musketeer", "Minions", "Poison", "Zap"],
}

def sample_random_deck(rng=None) -> List[int]:
    """Uniform 8-card sample without replacement from the distinct policy IDs."""
    import numpy as np
    card_ids = list(CARDS_CATALOG.keys())
    if rng is not None and hasattr(rng, "sample"):
        return list(rng.sample(card_ids, 8))
    source = rng if rng is not None else np.random
    return list(source.choice(card_ids, size=8, replace=False))

# Curriculum Milestones
STAGE_1_END_UPDATE = 100  # Scripted Bot Sets (Fixed Archetypes)
STAGE_2_END_UPDATE = 300  # Randomized Decks Every Match vs Scripted Bot Sets
# Stage 3: Updates > 300 -> Pure AlphaStar Self-Play (Bot vs Itself, NOT Scripted)

# Neural Network Architecture Specs
MAX_ENTITIES = 40
D_MODEL = 128
ENTITY_FEATURES = 7  # [card_id_norm, side, x_norm, y_norm, hp_ratio, is_air, state]
SPATIAL_CHANNELS = 6  # [p_ground, p_air, opp_ground, opp_air, buildings_towers, valid_deployment]

# RL Hyperparameters (OpenAI Five & AlphaStar)
PPO_LR = 3e-4
GAMMA = 0.995  # Extended credit assignment horizon for 25s back-of-king slow-roll pushes (0.995^65 = 0.72x vs 0.52x)
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.02  # Tuned up from 0.01 to maintain healthy card exploration
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4

# Production Rollout Specs (4 parallel envs x 512 steps = 2048 samples per rollout)
NUM_ENVS = 4
ROLLOUT_STEPS = 512
EFFECTIVE_BATCH_SIZE = 256  # Invariant across GPU types

# Imitation / Warm-Start Supervised Hyperparameters
BC_MATCHES = 500
BC_EPOCHS = 5

# Self-Play Pool Hyperparameters (DeepMind AlphaStar League Structure)
PFSP_RUSH_EXPLOITER_RATIO = 0.15  # 15% vs Rush/Bridge-Spam Exploiter
PFSP_SCRIPTED_RATIO = 0.15        # 15% vs rule-based ScriptedBot
PFSP_HISTORICAL_RATIO = 0.40      # 40% vs prioritized frozen league snapshots
PFSP_CURRENT_RATIO = 0.30         # 30% vs current live model
SNAPSHOT_INTERVAL = 50            # Snapshot policy to pool every 50 PPO updates


def get_hardware_profile() -> dict:
    """
    Auto-detects VRAM and configures micro-batching and gradient accumulation
    so Colab T4 and Local RTX 3050 maintain identical mathematical optimization steps.
    """
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "device_name": "CPU",
            "vram_gb": 0.0,
            "micro_batch": 32,
            "accum_steps": 8,
            "fp16": False,
            "profile_name": "CPU Fallback"
        }

    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(0)

    if vram_gb >= 12.0:
        # Colab Tesla T4 (16GB VRAM) Profile: single pass batch
        profile = {
            "device": "cuda",
            "device_name": gpu_name,
            "vram_gb": vram_gb,
            "micro_batch": 256,
            "accum_steps": 1,
            "fp16": True,
            "profile_name": "Colab T4 (16GB)"
        }
    else:
        # Local RTX 3050 Laptop (4GB / 6GB VRAM) Profile: 32 x 8 accumulation
        profile = {
            "device": "cuda",
            "device_name": gpu_name,
            "vram_gb": vram_gb,
            "micro_batch": 32,
            "accum_steps": 8,
            "fp16": True,
            "profile_name": "Local RTX 3050 (<12GB)"
        }

    return profile
