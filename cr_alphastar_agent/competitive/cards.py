"""Card vocabulary and card metadata for the competitive (v2) agent.

The vocabulary is derived deterministically from ``data/logic.json`` so that a
checkpoint made on one machine can be loaded on another.  Rows 0 and 1 are
reserved for padding and "unknown"; every real card / character / building name
follows in sorted order.

This module deliberately contains no simulator state.  It is safe for both the
training env and the public-observation audit to import.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from crsim.engine import Battle, MAX_ELIXIR, load_model

PAD_ID = 0
UNKNOWN_ID = 1
N_SPECIAL = 2

# The local deck presets are part of this repository, so they are a reasonable
# public "meta prior" for the opponent belief.  Additional decks can be added
# through configs/competitive.yaml.
try:  # pragma: no cover - import shape differs when tools is a namespace package
    from tools.decks import ALL_COMPETITIVE_CARDS, DECK_PRESETS
except Exception:  # pragma: no cover
    ALL_COMPETITIVE_CARDS = (
        "Pekka", "MiniPekka", "BattleRam", "DarkPrince", "Ghost", "Princess",
        "GoblinGang", "GoblinBarrel", "InfernoTower", "Cannon", "Log", "Poison",
        "Fireball", "Arrows", "Zap", "Giant", "Golem", "MegaKnight", "BabyDragon",
        "InfernoDragon", "MegaMinion", "Musketeer", "Witch", "DarkWitch",
        "IceGolemite", "IceSpirits", "Bats", "Skeletons", "Minions", "Knight",
        "Valkyrie", "HogRider", "Archer", "Balloon", "LavaHound",
    )
    DECK_PRESETS = {}


# Common spellings used by public replay datasets (snake_case) and by the
# engine (CamelCase).  Normalisation alone handles most of these; the explicit
# table covers the historical names that do not normalise to the engine name.
CARD_ALIASES = {
    "thelog": "Log",
    "log": "Log",
    "xbow": "XBow",
    "icewizard": "IceWizard",
    "electrowizard": "ElectroWizard",
    "electrospirit": "ElectroSpirits",
    "icespirit": "IceSpirits",
    "icespirits": "IceSpirits",
    "firespirit": "FireSpirits",
    "firespirits": "FireSpirits",
    "healspirit": "HealSpirits",
    "healspirits": "HealSpirits",
    "darkprince": "DarkPrince",
    "darkwitch": "DarkWitch",
    "goblingang": "GoblinGang",
    "goblinbarrel": "GoblinBarrel",
    "goblincage": "GoblinCage",
    "goblinhut": "GoblinHut",
    "infernodragon": "InfernoDragon",
    "infernotower": "InfernoTower",
    "megaknight": "MegaKnight",
    "megaminion": "MegaMinion",
    "minipekka": "MiniPekka",
    "battleram": "BattleRam",
    "royalgiant": "RoyalGiant",
    "royalhogs": "RoyalHogs",
    "royalrecruits": "RoyalRecruits",
    "skeletonarmy": "SkeletonArmy",
    "skeletonbarrel": "SkeletonBarrel",
    "skeletondragons": "SkeletonDragons",
    "skeletonking": "SkeletonKing",
    "skarmy": "SkeletonArmy",
    "lavahound": "LavaHound",
    "balloon": "Balloon",
    "babydragon": "BabyDragon",
    "musketeer": "Musketeer",
    "hog_rider": "HogRider",
    "hogrider": "HogRider",
    "giant": "Giant",
    "golem": "Golem",
    "pekka": "Pekka",
    "knight": "Knight",
    "valkyrie": "Valkyrie",
    "witch": "Witch",
    "wizard": "Wizard",
    "zap": "Zap",
    "arrows": "Arrows",
    "fireball": "Fireball",
    "poison": "Poison",
    "rocket": "Rocket",
    "tornado": "Tornado",
    "earthquake": "Earthquake",
    "lightning": "Lightning",
    "freeze": "Freeze",
    "mirror": "Mirror",
    "rage": "Rage",
    "clone": "Clone",
}


def normalize_card_name(name: str) -> str:
    """Case/punctuation-insensitive key used for matching replay card names."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


@dataclass(frozen=True)
class CardInfo:
    name: str
    card_id: int
    cost: int
    is_unit: bool
    is_spell: bool
    is_building: bool
    can_deploy_enemy_side: bool
    summon_count: int
    is_evolution: bool
    attacks_air: bool
    flying: bool
    max_hp: float


class CardVocab:
    """Deterministic card/character/building vocabulary.

    ``names`` includes every engine spell name plus every character and
    building name (spawned units carry character/building names).
    """

    def __init__(self, model: dict | None = None):
        self.model = model if model is not None else load_model()
        names: set[str] = set()
        tables = self.model.get("tables", {}) or {}
        for key in ("spells_characters", "spells_buildings", "spells_other", "spells_evolved"):
            for row in tables.get(key) or []:
                n = row.get("Name")
                if isinstance(n, str) and n:
                    names.add(n)
        names.update(self.model.get("characters", {}) or {})
        names.update(self.model.get("buildings", {}) or {})
        # A few aliases resolve to names that the engine knows, but only keep
        # aliases whose target exists.
        ordered = sorted(names)
        self.names: list[str] = ["<pad>", "<unknown>", *ordered]
        self.name_to_id = {n: i for i, n in enumerate(self.names)}
        self._alias_to_id: dict[str, int] = {}
        for alias, target in CARD_ALIASES.items():
            tid = self.name_to_id.get(target)
            if tid is not None:
                self._alias_to_id[alias] = tid
        for n in ordered:
            self._alias_to_id.setdefault(normalize_card_name(n), self.name_to_id[n])
        self.n_cards = len(self.names)
        self._meta_cache: dict[str, dict] = {}
        self._info_cache: dict[str, CardInfo] = {}
        self._spell_cache: dict[str, bool] = {}

    # ------------------------------------------------------------------ lookup
    def id(self, name: str | None) -> int:
        if not name:
            return PAD_ID
        nid = self.name_to_id.get(name)
        if nid is not None:
            return nid
        key = normalize_card_name(name)
        nid = self._alias_to_id.get(key)
        if nid is not None:
            return nid
        # Try one-card names with common plural/suffix variants.
        for suffix in ("s", "es", "spirit", "spirits"):
            if key.endswith(suffix):
                nid = self._alias_to_id.get(key[: -len(suffix)])
                if nid is not None:
                    return nid
        return UNKNOWN_ID

    def name(self, card_id: int) -> str:
        if 0 <= int(card_id) < len(self.names):
            return self.names[int(card_id)]
        return "<unknown>"

    def card_meta(self, name: str) -> dict:
        if name in self._meta_cache:
            return self._meta_cache[name]
        tabs = self.model.get("tables", {}) or {}
        row: dict = {}
        for key in ("spells_characters", "spells_buildings", "spells_other", "spells_evolved"):
            row = next((r for r in (tabs.get(key) or []) if r.get("Name") == name), {})
            if row:
                break
        self._meta_cache[name] = row or {}
        return self._meta_cache[name]

    def is_spell(self, name: str) -> bool:
        if name in self._spell_cache:
            return self._spell_cache[name]
        meta = self.card_meta(name)
        summon = meta.get("SummonCharacter") or ""
        result = False
        if not summon:
            result = True
        else:
            bdef = (self.model.get("buildings", {}) or {}).get(summon) or {}
            # A "spell" that summons a stat-less area container (e.g. Poison).
            result = bool(summon in (self.model.get("buildings", {}) or {})
                          and not bdef.get("Hitpoints")
                          and bdef.get("DeathAreaEffect"))
        self._spell_cache[name] = result
        return result

    def info(self, name: str) -> CardInfo:
        if name in self._info_cache:
            return self._info_cache[name]
        meta = self.card_meta(name)
        summon = meta.get("SummonCharacter") or ""
        sdef = ((self.model.get("characters", {}) or {}).get(summon)
                or (self.model.get("buildings", {}) or {}).get(summon) or {})
        is_building = bool(summon and summon in (self.model.get("buildings", {}) or {}))
        is_spell = self.is_spell(name)
        is_unit = bool(summon) and not is_spell
        info = CardInfo(
            name=name,
            card_id=self.id(name),
            cost=int(meta.get("ManaCost") or 0),
            is_unit=is_unit,
            is_spell=is_spell,
            is_building=is_building,
            can_deploy_enemy_side=bool(meta.get("CanDeployOnEnemySide")),
            summon_count=int(meta.get("SummonNumber") or 1),
            is_evolution=("Evolve" in name or "Evolution" in name or bool(meta.get("Evolve"))),
            attacks_air=bool(sdef.get("AttacksAir", False)),
            flying=bool("FlyingHeight" in sdef),
            max_hp=float(sdef.get("Hitpoints") or 0),
        )
        self._info_cache[name] = info
        return info

    def features(self, name: str) -> list[float]:
        """Static card features used by the global token (not by unit tokens)."""
        i = self.info(name)
        return [
            i.cost / 10.0,
            float(i.is_unit),
            float(i.is_spell),
            float(i.is_building),
            float(i.can_deploy_enemy_side),
            min(i.summon_count, 12) / 12.0,
            float(i.is_evolution),
            float(i.attacks_air),
            float(i.flying),
            min(i.max_hp / 6000.0, 1.0),
        ]

    def deck_ids(self, deck: Sequence[str]) -> list[int]:
        return [self.id(c) for c in deck]

    def deck_card_names(self) -> list[str]:
        """Every name that can legally appear as a deck card (spell-table rows)."""
        out: set[str] = set()
        tabs = self.model.get("tables", {}) or {}
        for key in ("spells_characters", "spells_buildings", "spells_other", "spells_evolved"):
            for row in tabs.get(key) or []:
                n = row.get("Name")
                if isinstance(n, str) and n and row.get("ManaCost") is not None:
                    out.add(n)
        return sorted(out)

    def playable_deck_pool(self) -> list[list[str]]:
        """Preset decks that are fully supported by this engine."""
        out: list[list[str]] = []
        for deck in DECK_PRESETS.values():
            if len(deck) == 8 and len(set(deck)) == 8 and all(self.card_meta(c) for c in deck):
                out.append(list(deck))
        return out

    def random_deck(self, rng: random.Random | random.SystemRandom | None = None) -> list[str]:
        rng = rng or random.SystemRandom()
        candidates = [c for c in ALL_COMPETITIVE_CARDS if self.card_meta(c)]
        # De-duplicate while preserving order.
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) < 8:
            raise ValueError("not enough supported cards for a random deck")
        return rng.sample(candidates, 8)


@lru_cache(maxsize=1)
def default_vocab() -> CardVocab:
    return CardVocab()


def vocab_from_checkpoint(payload: dict) -> CardVocab:
    """Rebuild a vocabulary from a checkpoint; falls back to the local engine.

    The names list is stored in every competitive checkpoint so inference does
    not depend on the exact logic.json revision that produced it.
    """
    names = payload.get("vocab_names")
    if not names:
        return default_vocab()
    v = CardVocab.__new__(CardVocab)
    v.model = load_model()
    v.names = list(names)
    v.name_to_id = {n: i for i, n in enumerate(v.names)}
    v._alias_to_id = {}
    for alias, target in CARD_ALIASES.items():
        tid = v.name_to_id.get(target)
        if tid is not None:
            v._alias_to_id[alias] = tid
    for n in v.names[2:]:
        v._alias_to_id.setdefault(normalize_card_name(n), v.name_to_id[n])
    v.n_cards = len(v.names)
    v._meta_cache = {}
    v._info_cache = {}
    v._spell_cache = {}
    return v


def deck_pool_from_config(cfg: dict, vocab: CardVocab, seed: int = 0) -> list[list[str]]:
    """Build the opponent deck pool from config, presets and optional random decks."""
    rng = random.Random(seed)
    pool: list[list[str]] = []
    for deck in cfg.get("decks") or []:
        deck = list(deck)
        if len(deck) == 8 and len(set(deck)) == 8 and all(vocab.card_meta(c) for c in deck):
            pool.append(deck)
    if cfg.get("include_presets", True):
        pool.extend(vocab.playable_deck_pool())
    n_random = int(cfg.get("random_decks", 0) or 0)
    for _ in range(n_random):
        try:
            pool.append(vocab.random_deck(rng))
        except ValueError:
            break
    # Deduplicate by sorted card-name tuple.
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for deck in pool:
        key = tuple(sorted(deck))
        if key not in seen:
            seen.add(key)
            unique.append(deck)
    return unique or vocab.playable_deck_pool()
