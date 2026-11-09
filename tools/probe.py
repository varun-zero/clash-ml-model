"""Probe why battles produce no action."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crsim.engine import load_model, Battle

M = load_model()
DECK = ["HogRider", "Knight", "Archer", "Cannon", "Fireball", "Arrows",
        "Skeletons", "Musketeer"]
b = Battle([DECK, DECK], seed=1)
print("hand:", b.hand[0], "cycle:", b.cycle[0])
print("elixir:", b.elixir, "cards:", [(c, b.card_cost(c)) for c in b.hand[0]])
print("meta Knight:", b._card_meta("Knight"))
print("meta Fireball:", b._card_meta("Fireball"))
print("legal(0,0,9,14):", b.legal_action(0, 0, 9, 14))
ok = b.play(0, 0, 9, 14)
print("play ok:", ok, "units:", [(u.name, u.hp) for u in b.units])
print("chars has Knight:", "Knight" in M["characters"], "HogRider:", "HogRider" in M["characters"])
print("chars has Skeletons:", "Skeletons" in M["characters"], "Archers:", "Archers" in M["characters"])
print("bld has Cannon:", "Cannon" in M["buildings"])
for t in range(600):
    b.step([])
print("after 60s: units=", [(u.name, u.team, u.hp, round(u.x,1), round(u.y,1)) for u in b.units])
for t in range(600):
    b.step([])
print("after 120s: units=", [(u.name, u.team, u.hp, round(u.x,1), round(u.y,1)) for u in b.units])
