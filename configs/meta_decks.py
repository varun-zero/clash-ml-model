"""
Clash Royale Competitive Meta Deck Library & Archetype Classifier.
Identifies opponent deck archetypes, predicts unrevealed cards, and provides counter-strategies.
"""
from typing import Dict, List, Set, Optional, Tuple, Any
from configs.constants import CARD_NAME_TO_ID, CARD_ID_TO_NAME, CARD_ID_TO_COST


def _cid(name: str) -> int:
    return CARD_NAME_TO_ID[name.lower()]


META_DECKS: List[Dict[str, Any]] = [
    {
        "id": "hog_2_6",
        "name": "2.6 Hog Cycle",
        "archetype": "Fast Cycle",
        "avg_elixir": 2.6,
        "cards": [
            _cid("Hog Rider"), _cid("Musketeer"), _cid("Ice Golem"), _cid("Ice Spirit"),
            _cid("Skeletons"), _cid("Fireball"), _cid("The Log"), _cid("Cannon")
        ],
        "key_indicators": {_cid("Hog Rider"), _cid("Ice Golem"), _cid("Cannon"), _cid("Musketeer")},
        "counter_strategy": "Save Mini P.E.K.K.A or heavy damage for Hog Rider. Do not give Fireball value near Princess Towers.",
    },
    {
        "id": "classic_log_bait",
        "name": "Classic Log Bait",
        "archetype": "Spell Bait",
        "avg_elixir": 3.3,
        "cards": [
            _cid("Goblin Barrel"), _cid("Princess"), _cid("Knight"), _cid("Goblin Gang"),
            _cid("Rocket"), _cid("The Log"), _cid("Inferno Tower"), _cid("Ice Spirit")
        ],
        "key_indicators": {_cid("Goblin Barrel"), _cid("Princess"), _cid("Goblin Gang")},
        "counter_strategy": "SAVE ARROWS STRICTLY for Goblin Barrel! Counter Princess at bridge with Knight or Archers.",
    },
    {
        "id": "golem_beatdown",
        "name": "Golem Beatdown",
        "archetype": "Heavy Beatdown",
        "avg_elixir": 4.3,
        "cards": [
            _cid("Golem"), _cid("Night Witch"), _cid("Baby Dragon"), _cid("Mega Minion"),
            _cid("Lightning"), _cid("Tornado"), _cid("Barbarian Barrel"), _cid("Lumberjack")
        ],
        "key_indicators": {_cid("Golem"), _cid("Night Witch"), _cid("Lightning")},
        "counter_strategy": "Opponent will drop Golem at 10 elixir in back. Immediately push opposite lane with Giant/troops!",
    },
    {
        "id": "pekka_bridge_spam",
        "name": "P.E.K.K.A Bridge Spam",
        "archetype": "Bridge Spam",
        "avg_elixir": 3.9,
        "cards": [
            _cid("P.E.K.K.A"), _cid("Battle Ram"), _cid("Bandit"), _cid("Royal Ghost"),
            _cid("Electro Wizard"), _cid("Poison"), _cid("Zap"), _cid("Magic Archer")
        ],
        "key_indicators": {_cid("P.E.K.K.A"), _cid("Battle Ram"), _cid("Bandit"), _cid("Royal Ghost")},
        "counter_strategy": "Distract P.E.K.K.A with cheap swarm/Knight into the kill zone. Defend dual-lane Battle Ram/Bandit rushes.",
    },
    {
        "id": "lavaloon",
        "name": "LavaLoon Beatdown",
        "archetype": "Air Beatdown",
        "avg_elixir": 4.1,
        "cards": [
            _cid("Lava Hound"), _cid("Balloon"), _cid("Mega Minion"), _cid("Minions"),
            _cid("Fireball"), _cid("Zap"), _cid("Tombstone"), _cid("Guards")
        ],
        "key_indicators": {_cid("Lava Hound"), _cid("Balloon"), _cid("Tombstone")},
        "counter_strategy": "Preserve Musketeer and Archers for Balloon. Target Minions with predictive Arrows.",
    },
    {
        "id": "xbow_cycle",
        "name": "X-Bow 3.0 Cycle",
        "archetype": "Siege",
        "avg_elixir": 3.0,
        "cards": [
            _cid("X-Bow"), _cid("Tesla"), _cid("Archers"), _cid("Knight"),
            _cid("Ice Spirit"), _cid("Skeletons"), _cid("Fireball"), _cid("The Log")
        ],
        "key_indicators": {_cid("X-Bow"), _cid("Tesla")},
        "counter_strategy": "Drop Giant or Mini P.E.K.K.A in front of offensive X-Bow to absorb lock. Support with Fireball.",
    },
    {
        "id": "mortar_miner_bait",
        "name": "Mortar Miner Bait",
        "archetype": "Siege / Bait",
        "avg_elixir": 3.1,
        "cards": [
            _cid("Mortar"), _cid("Miner"), _cid("Skeleton King"), _cid("Firecracker"),
            _cid("Dart Goblin"), _cid("Goblin Gang"), _cid("The Log"), _cid("Poison")
        ],
        "key_indicators": {_cid("Mortar"), _cid("Miner"), _cid("Firecracker")},
        "counter_strategy": "Do not let Mortar lock onto Princess Tower. Cleanse Dart Goblin and Firecracker before swarm arrives.",
    },
    {
        "id": "giant_graveyard",
        "name": "Giant Graveyard",
        "archetype": "Beatdown / Control",
        "avg_elixir": 3.8,
        "cards": [
            _cid("Giant"), _cid("Graveyard"), _cid("Bowler"), _cid("Musketeer"),
            _cid("Arrows"), _cid("Giant Snowball"), _cid("Bats"), _cid("Skeleton Army")
        ],
        "key_indicators": {_cid("Giant"), _cid("Graveyard"), _cid("Bowler")},
        "counter_strategy": "Kill tank (Giant/Bowler) first with Mini P.E.K.K.A; deploy Archers/Minions on Graveyard skeletons.",
    },
    {
        "id": "royal_giant_fish",
        "name": "Royal Giant Fisherman",
        "archetype": "Control Beatdown",
        "avg_elixir": 3.4,
        "cards": [
            _cid("Royal Giant"), _cid("Fisherman"), _cid("Hunter"), _cid("Electro Spirit"),
            _cid("Fireball"), _cid("The Log"), _cid("Mega Minion"), _cid("Royal Ghost")
        ],
        "key_indicators": {_cid("Royal Giant"), _cid("Fisherman"), _cid("Hunter")},
        "counter_strategy": "Beware Fisherman hooking defenders away from Royal Giant. Place high DPS units out of Fisherman hook range.",
    },
    {
        "id": "egiant_lightning",
        "name": "Electro Giant Lightning",
        "archetype": "Heavy Beatdown",
        "avg_elixir": 4.4,
        "cards": [
            _cid("Electro Giant"), _cid("Lightning"), _cid("Tornado"), _cid("Dark Prince"),
            _cid("Baby Dragon"), _cid("Cannon"), _cid("Bomber"), _cid("Golden Knight")
        ],
        "key_indicators": {_cid("Electro Giant"), _cid("Lightning"), _cid("Tornado")},
        "counter_strategy": "Do NOT use swarms (zap backfire). Use Mini P.E.K.K.A or ranged Musketeer away from stun radius.",
    },
    {
        "id": "goblin_drill_cycle",
        "name": "Goblin Drill Fast Cycle",
        "archetype": "Fast Cycle",
        "avg_elixir": 2.9,
        "cards": [
            _cid("Goblin Drill"), _cid("Wall Breakers"), _cid("Firecracker"), _cid("Knight"),
            _cid("Tesla"), _cid("The Log"), _cid("Skeletons"), _cid("Poison")
        ],
        "key_indicators": {_cid("Goblin Drill"), _cid("Wall Breakers")},
        "counter_strategy": "Defend drill burrow immediately with splash or Knight. Catch Wall Breakers at the river.",
    },
    {
        "id": "starter_beatdown",
        "name": "Starter / Arena 1 Beatdown",
        "archetype": "Starter Beatdown",
        "avg_elixir": 3.6,
        "cards": [
            _cid("Giant"), _cid("Mini P.E.K.K.A"), _cid("Musketeer"), _cid("Knight"),
            _cid("Archers"), _cid("Arrows"), _cid("Fireball"), _cid("Minions")
        ],
        "key_indicators": {_cid("Giant"), _cid("Mini P.E.K.K.A"), _cid("Musketeer"), _cid("Archers")},
        "counter_strategy": "Standard beginner ladder matchup. Win elixir trades and build a Giant push with Musketeer backing.",
    },
]


def identify_meta_deck(seen_card_ids: List[int]) -> Dict[str, Any]:
    """
    Matches revealed opponent cards against known competitive Clash Royale meta decks.
    Returns matched deck, archetype, confidence, predicted remaining cards, and counter-strategy.
    """
    seen_set = set(int(c) for c in seen_card_ids if int(c) > 0)
    if not seen_set:
        return {
            "matched_deck_name": "Unknown",
            "archetype": "Unknown",
            "confidence": 0.0,
            "predicted_unrevealed_cards": [],
            "counter_strategy": "Observe opponent's first plays to identify archetype.",
        }

    best_deck = None
    best_score = -1.0

    for deck in META_DECKS:
        deck_cards = set(deck["cards"])
        overlap = seen_set.intersection(deck_cards)
        mismatches = seen_set.difference(deck_cards)

        if not overlap:
            continue

        # Precision: fraction of seen cards that belong to this deck
        precision = len(overlap) / float(len(seen_set))
        # Recall: fraction of this deck seen so far
        recall = len(overlap) / 8.0

        # Key indicator bonus
        key_hits = len(seen_set.intersection(deck["key_indicators"]))
        key_bonus = 0.25 * key_hits

        # Penalty for mismatched cards (not in this meta deck)
        mismatch_penalty = 0.20 * len(mismatches)

        score = (precision * 0.50 + recall * 0.35 + key_bonus) - mismatch_penalty

        if score > best_score:
            best_score = score
            best_deck = deck

    if best_deck is None or best_score < 0.15:
        return {
            "matched_deck_name": "Off-Meta / Homebrew Deck",
            "archetype": "Mid-Ladder Control",
            "confidence": 0.20,
            "predicted_unrevealed_cards": [],
            "counter_strategy": "Play reactive defense, exploit opponent's high elixir plays.",
        }

    confidence = float(np_clip(best_score, 0.0, 1.0))
    deck_card_set = set(best_deck["cards"])
    unrevealed = [cid for cid in best_deck["cards"] if cid not in seen_set]

    return {
        "matched_deck_name": best_deck["name"],
        "archetype": best_deck["archetype"],
        "confidence": round(confidence, 2),
        "predicted_unrevealed_cards": unrevealed,
        "predicted_unrevealed_names": [CARD_ID_TO_NAME.get(cid, "Card") for cid in unrevealed],
        "counter_strategy": best_deck["counter_strategy"],
        "avg_elixir": best_deck["avg_elixir"],
    }


def np_clip(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))
