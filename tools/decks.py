"""Deck presets and random deck generation for the ClashBot simulator."""

from __future__ import annotations

import random


# The random pool contains cards whose behavior is represented in this
# simulator. Presets may also use other supported cards from the logic data.
ALL_COMPETITIVE_CARDS = (
    "Pekka", "MiniPekka", "BattleRam", "DarkPrince", "Ghost", "Princess",
    "GoblinGang", "GoblinBarrel", "InfernoTower", "Cannon", "Log", "Poison",
    "Fireball", "Arrows", "Zap", "Giant", "Golem", "MegaKnight", "BabyDragon",
    "InfernoDragon", "MegaMinion", "Musketeer", "Witch", "DarkWitch",
    "IceGolemite", "IceSpirits", "Bats", "Skeletons", "Minions", "Knight",
    "Valkyrie", "HogRider", "Archer", "Balloon", "LavaHound",
)


# These are stable archetype references for the simulator's card catalog, not
# a claim that they are today's highest-performing ladder lists. Several cards
# are absent from the catalog, so those profiles use the closest supported
# roster and say so in the name.
DECK_PRESETS: dict[str, tuple[str, ...]] = {
    "Classic Log Bait": (
        "GoblinBarrel", "Princess", "GoblinGang", "Knight", "InfernoTower",
        "Log", "Rocket", "IceSpirits",
    ),
    "Golem Beatdown": (
        "Golem", "BabyDragon", "InfernoDragon", "MegaMinion", "DarkWitch",
        "DarkPrince", "Poison", "Zap",
    ),
    "Golem Double Dragon": (
        "Golem", "BabyDragon", "InfernoDragon", "MegaMinion", "DarkWitch",
        "DarkPrince", "Tornado", "Lightning",
    ),
    "Hog 2.6 Cycle · sim roster": (
        "HogRider", "Musketeer", "Cannon", "IceGolemite", "Skeletons",
        "IceSpirits", "Fireball", "Log",
    ),
    "PEKKA Bridge Spam · sim roster": (
        "Pekka", "BattleRam", "DarkPrince", "Ghost", "ElectroWizard",
        "Miner", "Poison", "Zap",
    ),
    "Giant Beatdown": (
        "Giant", "Witch", "MiniPekka", "Musketeer", "MegaMinion", "Fireball",
        "Zap", "Minions",
    ),
    "Mega Knight Bait": (
        "MegaKnight", "InfernoDragon", "Princess", "GoblinGang", "Bats",
        "GoblinBarrel", "Zap", "Fireball",
    ),
    "LavaLoon Air Beatdown": (
        "LavaHound", "Balloon", "MegaMinion", "Minions", "Fireball", "Zap",
        "InfernoDragon", "DarkPrince",
    ),
}


DECK_PRESET_NAMES = (
    "Randomized Deck",
    "Classic Log Bait",
    "Golem Beatdown",
    "Golem Double Dragon",
    "Hog 2.6 Cycle · sim roster",
    "PEKKA Bridge Spam · sim roster",
    "Giant Beatdown",
    "Mega Knight Bait",
    "LavaLoon Air Beatdown",
)


DECK_ALIASES = {
    "Log Bait": "Classic Log Bait",
    "Spell Bait": "Classic Log Bait",
    "Double Dragon Beatdown": "Golem Double Dragon",
    "Hog 2.6 Cycle": "Hog 2.6 Cycle · sim roster",
    "PEKKA Bridge Spam": "PEKKA Bridge Spam · sim roster",
    "Bridge Spam": "PEKKA Bridge Spam · sim roster",
    "Pekka Control": "PEKKA Bridge Spam · sim roster",
    "Lavaloon Air": "LavaLoon Air Beatdown",
}


def generate_random_deck(
    rng: random.Random | random.SystemRandom | None = None,
) -> list[str]:
    """Draw 8 distinct cards uniformly; default to OS-backed randomness."""
    source = rng if rng is not None else random.SystemRandom()
    return source.sample(ALL_COMPETITIVE_CARDS, 8)


def resolve_deck(
    raw_name: str | None,
    rng: random.Random | random.SystemRandom | None = None,
) -> tuple[str, list[str]]:
    """Resolve a saved preset name, accepting legacy labels and UI prefixes."""
    clean = (raw_name or "").strip()
    for prefix in ("vs ", "🔴 ", "🔵 ", "🎲 "):
        while clean.startswith(prefix):
            clean = clean[len(prefix):].strip()

    if not clean or "random" in clean.casefold():
        return "Randomized Deck", generate_random_deck(rng)

    by_casefold = {name.casefold(): name for name in DECK_PRESETS}
    canonical = by_casefold.get(clean.casefold())
    if canonical is None:
        alias = next(
            (target for alias, target in DECK_ALIASES.items()
             if alias.casefold() == clean.casefold()),
            None,
        )
        canonical = alias
    if canonical is None:
        raise ValueError(f"Unknown deck preset: {raw_name!r}")
    return canonical, list(DECK_PRESETS[canonical])
