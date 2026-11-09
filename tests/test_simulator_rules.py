import random

from crsim.engine import Battle, load_model
from tools.decks import (
    ALL_COMPETITIVE_CARDS,
    DECK_PRESETS,
    generate_random_deck,
    resolve_deck,
)


MODEL = load_model()


def make_battle(deck=None, seed=17):
    deck = deck or [
        "Knight", "Fireball", "Arrows", "Zap", "Giant", "Musketeer",
        "Skeletons", "Minions",
    ]
    return Battle([list(deck), list(deck)], model=MODEL, seed=seed)


def test_every_named_preset_is_a_valid_eight_card_deck():
    battle = make_battle()
    for name, cards in DECK_PRESETS.items():
        assert len(cards) == 8
        assert len(set(cards)) == 8
        assert all(battle._card_meta(card) for card in cards), name
        resolved_name, resolved_cards = resolve_deck(name)
        assert resolved_name == name
        assert resolved_cards == list(cards)


def test_random_decks_are_unique_and_use_supported_cards():
    deck = generate_random_deck(random.Random(23))
    battle = make_battle()
    assert len(deck) == 8
    assert len(set(deck)) == 8
    assert set(deck) <= set(ALL_COMPETITIVE_CARDS)
    assert all(battle._card_meta(card) for card in deck)


def test_same_match_seed_reproduces_both_opening_hands_and_cycles():
    first = make_battle(seed=91)
    second = make_battle(seed=91)
    assert first.hand == second.hand
    assert first.cycle == second.cycle


def test_both_player_card_cycles_expose_and_play_all_eight_cards():
    battle = make_battle(seed=19)
    deck_set = set(battle.decks[0])

    for team in (0, 1):
        expected_order = list(battle.hand[team]) + list(battle.cycle[team])
        assert set(expected_order) == deck_set
        assert len(expected_order) == 8
        played = []

        for _ in range(8):
            card_name = next(card for card in battle.hand[team] if card not in played)
            slot = battle.hand[team].index(card_name)
            battle.elixir[team] = 10.0
            battle.play_cd[team] = 0
            x, y = (8.0, 10.0) if team == 0 else (8.0, 22.0)
            assert battle.play(team, slot, x, y), (team, card_name)
            played.append(card_name)
            assert len(battle.hand[team]) == 4
            assert len(battle.cycle[team]) == 4
            assert set(battle.hand[team] + battle.cycle[team]) == deck_set
            assert len(set(battle.hand[team] + battle.cycle[team])) == 8

        assert set(played) == deck_set


def test_every_engine_catalog_card_can_be_deployed_from_a_hand_slot():
    starter = [
        "Knight", "Fireball", "Arrows", "Zap", "Giant", "Musketeer",
        "Skeletons", "Minions",
    ]
    for card in sorted(MODEL["spells"]):
        deck = [card] + [name for name in starter if name != card][:7]
        battle = Battle([deck, deck], model=MODEL, seed=3)
        battle.hand[0] = [card, *deck[1:4]]
        battle.cycle[0] = list(deck[4:])
        battle.elixir[0] = 10.0
        battle.play_cd[0] = 0
        if card == "Mirror":
            battle.last_played[0] = "Knight"

        assert battle.play(0, 0, 9.0, 8.0), card
        assert len(battle.hand[0]) == 4
        assert len(battle.cycle[0]) == 4
        assert set(battle.hand[0] + battle.cycle[0]) == set(deck)


def test_king_tower_stays_asleep_until_hit_or_princess_tower_falls():
    battle = make_battle()
    kings = [u for u in battle.units if u.kind == "KingTower"]
    assert len(kings) == 2
    assert all(not king.activated for king in kings)

    blue_king = next(king for king in kings if king.team == 0)
    battle._damage(blue_king, 0)
    assert not blue_king.activated
    battle._damage(blue_king, 1)
    assert blue_king.activated

    red_king = next(king for king in kings if king.team == 1)
    red_princess = next(
        tower for tower in battle.units
        if tower.team == 1 and tower.kind == "PrincessTower"
    )
    assert not red_king.activated
    battle._damage(red_princess, red_princess.hp)
    assert red_king.activated


def test_elixir_regeneration_uses_normal_double_and_triple_time_bands():
    battle = make_battle()
    assert battle._bar_ms() == 28_000

    battle.t = 120_000
    assert battle._bar_ms() == 14_000
    battle.t = 239_999
    assert battle._bar_ms() == 14_000

    battle.t = 240_000
    assert battle._bar_ms() == 9_300


def test_buff_percentages_apply_slow_and_rage_multipliers_correctly():
    battle = make_battle()

    assert battle._buff_effect_from_def("Poison").speed_mult == 0.85
    assert battle._buff_effect_from_def("Earthquake").speed_mult == 0.5
    assert battle._buff_effect_from_def("snowball_spell_ev1_hit").speed_mult == 0.65
    assert battle._buff_effect_from_def("IceWizardSlowDown").hitrate_mult == 0.7
    assert battle._buff_effect_from_def("Rage").speed_mult == 1.3
    assert battle._buff_effect_from_def("Rage").hitrate_mult == 1.3
    assert battle._buff_effect_from_def("Freeze").speed_mult == 0.0
