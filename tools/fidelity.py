"""Targeted card-behavior fidelity checks against the data-driven engine.

Each check builds a controlled board, drives the engine, and compares the
observed outcome with the value implied by the decoded logic tables.
Run:  python tools/fidelity.py
"""
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from crsim.engine import Battle, Tower, _growth  # noqa: E402

TICK = 0.1
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def new_battle(seed=7):
    deck = ["Knight", "Fireball", "Arrows", "Zap", "Giant", "Musketeer", "Skeletons", "Minions"]
    return Battle([deck, deck], seed=seed)


def clear(b):
    b.units.clear()
    b.fields.clear()
    b.projectiles.clear()


def first_hit(b, target, max_ticks=80):
    for _ in range(max_ticks):
        b.step([])
        d = target.maxhp - target.hp
        if d:
            return d
    return target.maxhp - target.hp


def place(b, name, team, x, y):
    u = b._place(name, team, x, y)
    assert u is not None, f"could not place {name}"
    return u


def step(b, ticks):
    for _ in range(ticks):
        if not b.done:
            b.step([])


def cast(b, card, team, x, y):
    cm = b._card_meta(card)
    assert cm, f"no card meta for {card}"
    b._cast_spell(cm, team, x, y)


def base(card, team=0, x=9.0):
    b = new_battle()
    clear(b)
    return b


# ---------------------------------------------------------------- Princess
def probe_princess():
    b = base("Princess")
    p = place(b, "Princess", 0, 9.0, 10.0)
    k = place(b, "Knight", 1, 9.0, 12.0)
    expected = _growth(66, b.card_level, b.card_dmg_pct)
    step(b, 45)
    dmg = k.maxhp - k.hp
    check("Princess single bolt 66", dmg == expected, f"dmg={dmg} exp={expected}")
    check("Princess projectile range", p.projectile_speed_t > 0 and p.splash_radius_t >= 1.9,
          f"speed={p.projectile_speed_t:.1f} splash={p.splash_radius_t}")


# ---------------------------------------------------------------- Berserker
def probe_berserker():
    b = base("Berserker")
    z = place(b, "Berserker", 0, 9.0, 14.0)
    k = place(b, "Giant", 1, 9.0, 15.8)
    expected_hit = _growth(120, b.card_level, b.card_dmg_pct)
    check("Berserker combo damage 3x40", z.damage == expected_hit,
          f"dmg={z.damage} exp={expected_hit}")
    dmg = first_hit(b, k)
    check("Berserker lands combo", dmg == expected_hit, f"dmg={dmg}")


# ---------------------------------------------------------------- Firecracker
def probe_firecracker():
    b = base("Firecracker")
    pdef = b.model["projectiles"]["FirecrackerProjectile"]
    b._projectile_payload(pdef, 0, 9.0, 16.0, 0.4, [])
    shards = [p for p in b.projectiles if p.name == "FirecrackerExplosion"]
    check("Firecracker 5 shards", len(shards) == 5, f"n={len(shards)}")
    check("Firecracker shard damage 25", all(
        p.damage == _growth(25, b.card_level, b.card_dmg_pct) for p in shards))
    # firecracker shoots and shards hit a target behind the impact
    b2 = base("Firecracker")
    fc = place(b2, "Firecracker", 0, 9.0, 12.0)
    k = place(b2, "Knight", 1, 9.0, 16.0)
    step(b2, 50)
    dmg = k.maxhp - k.hp
    check("Firecracker damages target", dmg > 0, f"dmg={dmg} (fc hp={fc.hp})")


# ---------------------------------------------------------------- Barb barrel
def probe_barb_log():
    b = base("BarbLog")
    k = place(b, "Knight", 1, 9.0, 11.0)
    cast(b, "BarbLog", 0, 9.0, 7.0)
    rolls = [p for p in b.projectiles if p.name == "BarbLogProjectileRolling"]
    check("BarbLog rolls forward", len(rolls) == 1
          and abs(rolls[0].ty - rolls[0].y) > 0.1,
          f"n={len(rolls)}")
    if rolls:
        r = rolls[0]
        check("BarbLog pierces + carries barbarian",
              r.pierce and r.spawn_character == "Barbarian",
              f"pierce={r.pierce} spawn={r.spawn_character}")
    step(b, 25)
    dmg = k.maxhp - k.hp
    check("BarbLog hits troop 84", dmg == _growth(84, b.card_level, b.card_dmg_pct),
          f"dmg={dmg}")
    step(b, 15)
    barbs = [u for u in b.units if u.name == "Barbarian" and u.team == 0]
    check("BarbLog drops Barbarian", len(barbs) == 1, f"n={len(barbs)}")


# ---------------------------------------------------------------- Zap evolution
def probe_zap_evo():
    b = base("Zap_EV1")
    k = place(b, "Knight", 1, 9.0, 16.0)
    expected = _growth(75, b.card_level, b.card_dmg_pct)
    cast(b, "Zap_EV1", 0, 9.0, 16.0)
    step(b, 3)  # ~0.3s: first strike
    first = k.maxhp - k.hp
    stun = "ZapFreeze" in k.buffs and any(
        bd.stun for bd in k.buffs.values())
    step(b, 18)  # ~2.0s: delayed second strike at 1.45s
    total = k.maxhp - k.hp
    check("Zap Evo first strike", first == expected, f"dmg={first} exp={expected}")
    check("Zap Evo second strike @1.45s", total == 2 * expected,
          f"total={total} exp={2 * expected}")
    check("Zap Evo stun buff", stun)


# ---------------------------------------------------------------- Graveyard
def probe_graveyard():
    b = base("Graveyard")
    ids = set()

    def collect():
        for u in b.units:
            if u.name == "Graveyard_rework_Skeleton" and u.team == 0:
                ids.add(u.id)

    cast(b, "Graveyard", 0, 9.0, 16.0)
    for _ in range(95):
        if not b.done:
            b.step([])
        collect()
    check("Graveyard spawns 12 skeletons", len(ids) == 12, f"n={len(ids)}")


# ---------------------------------------------------------------- Poison
def probe_poison():
    b = base("Poison")
    k = place(b, "Knight", 1, 9.0, 16.0)
    cast(b, "Poison", 0, 9.0, 16.0)
    step(b, 35)
    dmg = k.maxhp - k.hp
    per_s = _growth(36, b.card_level, b.card_dmg_pct)
    check("Poison ticks DPS", dmg >= per_s, f"dmg={dmg} per_s={per_s}")
    check("Poison slows by 15%", b._speed_mult(k) == 0.85,
          f"mult={b._speed_mult(k):.2f}")


# ---------------------------------------------------------------- Heal Spirit
def probe_heal_spirit():
    b = base("HealSpirit")
    friend = place(b, "Knight", 0, 9.0, 11.5)
    friend.hp = 500
    place(b, "ElixirCollector", 1, 9.0, 13.5)
    spirit = place(b, "HealSpirit", 0, 9.0, 12.3)
    hp0 = friend.hp
    step(b, 40)
    healed = friend.hp - hp0
    check("Heal Spirit heals allies", healed > 0,
          f"healed={healed} spirit_alive={spirit.hp > 0}")


# ---------------------------------------------------------------- Earthquake
def probe_earthquake():
    b = base("Earthquake")
    troops = place(b, "Knight", 1, 9.0, 16.0)
    bld = place(b, "Cannon", 0, 11.0, 16.0)
    bld.team = 1  # enemy building
    cast(b, "Earthquake", 0, 9.0, 16.0)
    step(b, 25)
    slow = b._speed_mult(troops)
    step(b, 10)
    t_dmg = troops.maxhp - troops.hp
    b_dmg = bld.maxhp - bld.hp
    check("Earthquake slows troops by 50%", slow == 0.5, f"mult={slow:.2f}")
    check("Earthquake damages buildings 112/x3", b_dmg >= 2 * _growth(112, b.card_level, b.card_dmg_pct),
          f"bld_dmg={b_dmg}")
    check("Earthquake damages troops less", 0 < t_dmg < b_dmg,
          f"troop={t_dmg} bld={b_dmg}")


# ---------------------------------------------------------------- Goblin Curse
def probe_goblin_curse():
    b = base("GoblinCurse")
    k = place(b, "Knight", 1, 9.0, 16.0)
    cast(b, "GoblinCurse", 0, 9.0, 16.0)
    step(b, 3)
    check("Goblin Curse applies death spawn", any(
        bd.death_spawn == "GoblinCurseGoblin" for bd in k.buffs.values()),
        f"buffs={list(k.buffs)}")
    b._damage(k, 999999)
    goblins = [u for u in b.units if u.name == "GoblinCurseGoblin"]
    check("Goblin Curse transforms victim", len(goblins) >= 1
          and goblins[0].team == 0, f"n={len(goblins)}")


# ---------------------------------------------------------------- Clone
def probe_clone():
    b = base("Clone")
    k = place(b, "Knight", 0, 9.0, 14.0)
    cast(b, "Clone", 0, 9.0, 14.0)
    step(b, 2)
    clones = [u for u in b.units if u.is_clone]
    check("Clone creates 1hp copies", len(clones) == 1
          and clones[0].hp == 1 and clones[0].maxhp == 1,
          f"n={len(clones)} hp={[c.hp for c in clones]}")


# ---------------------------------------------------------------- Lightning
def probe_lightning():
    b = base("Lightning")
    units = [place(b, n, 1, x, 16.0) for n, x in
             (("Knight", 9.0), ("Wizard", 8.0), ("Musketeer", 10.0))]
    step(b, 1)  # let deploy/load finish? they spawn with load_ms
    for u in units:
        u.load_ms = 0
    cast(b, "Lightning", 0, 9.0, 16.0)
    step(b, 20)
    hits = sum(1 for u in units if u.hp < u.maxhp)
    expected = _growth(413, b.card_level, b.card_dmg_pct)
    check("Lightning 3 strikes hit 3 targets", hits == 3, f"hits={hits}")
    check("Lightning strike damage 413", all(
        u.maxhp - u.hp == expected for u in units),
        f"dmgs={[u.maxhp - u.hp for u in units]} exp={expected}")


# ---------------------------------------------------------------- Arrows
def probe_arrows():
    b = base("Arrows")
    units = [place(b, "Archer", 1, x, 16.0) for x in (8.0, 9.0, 10.0)]
    for u in units:
        u.load_ms = 0
    cast(b, "Arrows", 0, 9.0, 16.0)
    step(b, 12)
    expected = 3 * _growth(48, b.card_level, b.card_dmg_pct)
    dmgs = [u.maxhp - u.hp for u in units]
    check("Arrows 3 waves x48", dmgs and all(d == expected for d in dmgs),
          f"dmgs={dmgs} exp={expected}")


# ---------------------------------------------------------------- Snowball
def probe_snowball():
    b = base("Snowball")
    k = place(b, "Knight", 1, 9.0, 16.0)
    cast(b, "Snowball", 0, 9.0, 16.0)
    step(b, 3)
    dmg = k.maxhp - k.hp
    check("Snowball damage 70 + 30% slow",
          dmg == _growth(70, b.card_level, b.card_dmg_pct)
          and b._speed_mult(k) == 0.7,
          f"dmg={dmg} mult={b._speed_mult(k):.2f}")


# ---------------------------------------------------------------- Tornado
def probe_tornado():
    b = base("Tornado")
    k = place(b, "Knight", 1, 13.0, 16.0)
    cast(b, "Tornado", 0, 9.0, 16.0)
    x0 = k.x
    step(b, 8)
    check("Tornado pulls", k.x < x0, f"x {x0:.1f}->{k.x:.1f}")
    check("Tornado deals DPS", k.hp < k.maxhp, f"dmg={k.maxhp - k.hp}")


# ---------------------------------------------------------------- Freeze
def probe_freeze():
    b = base("Freeze")
    k = place(b, "Knight", 1, 9.0, 16.0)
    cast(b, "Freeze", 0, 9.0, 16.0)
    step(b, 4)
    check("Freeze roots unit", b._speed_mult(k) == 0.0 and
          any(bd.stun for bd in k.buffs.values()), f"mult={b._speed_mult(k)}")
    step(b, 40)
    check("Freeze expires ~3.5s", b._speed_mult(k) > 0.9, f"mult={b._speed_mult(k):.2f}")


# ---------------------------------------------------------------- Rage
def probe_rage():
    b = base("Rage")
    k = place(b, "Knight", 0, 9.0, 14.0)
    cast(b, "Rage", 0, 9.0, 14.0)
    step(b, 3)
    check("Rage speeds friendly troops", b._speed_mult(k) > 1.0,
          f"mult={b._speed_mult(k):.2f}")


# ---------------------------------------------------------------- Royal Delivery
def probe_royal_delivery():
    b = base("RoyalDelivery")
    cast(b, "RoyalDelivery", 0, 9.0, 16.0)
    recruits = []
    for _ in range(40):
        b.step([])
        recruits = [u for u in b.units if u.name == "DeliveryRecruit" and u.team == 0]
        if recruits:
            break
    check("Royal Delivery drops recruit", len(recruits) == 1, f"n={len(recruits)}")


# ---------------------------------------------------------------- Ice Spirit
def probe_ice_spirit():
    b = base("IceSpirits")
    k = place(b, "Knight", 1, 9.0, 13.0)
    place(b, "IceSpirits", 0, 9.0, 11.0)
    froze = False
    for _ in range(30):
        b.step([])
        if b._speed_mult(k) == 0.0:
            froze = True
    check("Ice Spirit freezes", froze,
          f"mult={b._speed_mult(k):.2f} dmg={k.maxhp - k.hp}")


# ---------------------------------------------------------------- Electro Spirit
def probe_electro_spirit():
    b = base("ElectroSpirit")
    tgt = [place(b, "Knight", 1, x, 13.0) for x in (8.0, 9.0, 10.0)]
    place(b, "ElectroSpirit", 0, 9.0, 11.0)
    step(b, 30)
    hit = [u for u in tgt if u.hp < u.maxhp]
    check("Electro Spirit chains", len(hit) >= 2, f"chained={len(hit)}")


# ---------------------------------------------------------------- Skeleton Army
def probe_skeleton_army():
    b = base("SkeletonArmy")
    cm = b._card_meta("SkeletonArmy")
    b._deploy_card(0, cm, 9.0, 16.0)
    skels = [u for u in b.units if u.team == 0 and "Skeleton" in u.name]
    check("Skeleton Army summons 15", len(skels) >= 15, f"n={len(skels)}")


# ---------------------------------------------------------------- Firecracker Evo
def probe_firecracker_evo():
    b = base("Firecracker_EV1")
    pdef = b.model["projectiles"]["FirecrackerProjectile_EV1"]
    b._projectile_payload(pdef, 0, 9.0, 16.0, 0.4, [])
    shards = [p for p in b.projectiles if p.name == "FirecrackerExplosion_EV1"]
    fields = [f for f in b.fields if f.name == "FirecrackerFireworksBig_EV1"]
    check("Firecracker Evo 5 shards", len(shards) == 5, f"n={len(shards)}")
    check("Firecracker Evo firework field", len(fields) == 1,
          f"fields={[f.name for f in b.fields]}")
    check("Firecracker Evo shard damage 25", all(
        p.damage == _growth(25, b.card_level, b.card_dmg_pct) for p in shards))


# ---------------------------------------------------------------- Goblin Drill
def probe_goblin_drill():
    b = base("GoblinDrill")
    d = place(b, "GoblinDrill", 0, 9.0, 12.0)
    k = place(b, "Knight", 1, 9.0, 13.8)
    step(b, 1)
    expected = _growth(33, b.card_level, b.card_dmg_pct)
    check("Goblin Drill emerge damage 33", k.maxhp - k.hp == expected,
          f"dmg={k.maxhp - k.hp} exp={expected}")
    step(b, 14)
    goblins = [u for u in b.units if u.name == "Goblin" and u.team == 0 and u.hp > 0]
    check("Goblin Drill spawns Goblin", len(goblins) >= 1, f"n={len(goblins)}")
    b._damage(d, 999999)
    step(b, 1)
    goblins = [u for u in b.units if u.name == "Goblin" and u.team == 0 and u.hp > 0]
    check("Goblin Drill death spawns 2", len(goblins) >= 2, f"n={len(goblins)}")


def probe_goblin_drill_evo():
    b = base("GoblinDrill_EV1")
    d = place(b, "GoblinDrill_EV1", 0, 9.0, 12.0)
    step(b, 2)
    y0 = d.y
    d.hp = int(d.maxhp * 0.60)
    step(b, 12)
    goblins = [u for u in b.units if u.name == "Goblin" and u.team == 0 and u.hp > 0]
    check("Goblin Drill Evo relocates", d.y > y0 + 1.0,
          f"y {y0:.1f}->{d.y:.1f}")
    check("Goblin Drill Evo leaves goblins", len(goblins) >= 2,
          f"n={len(goblins)}")


# ---------------------------------------------------------------- Skeleton Balloon Evo
def probe_skeleton_balloon_evo():
    b = base("SkeletonBalloon_EV1")
    u = place(b, "SkeletonBalloon_EV1", 0, 9.0, 12.0)
    u.load_ms = 0
    u.hp = int(u.maxhp * 0.70)
    step(b, 2)
    fields = [f for f in b.fields if f.spawn_on_end]
    check("Skeleton Balloon Evo drops barrel at 75%", len(fields) == 1,
          f"fields={[f.name for f in fields]}")
    step(b, 8)
    n1 = len([x for x in b.units if x.name == "Skeleton"])
    check("Skeleton Balloon Evo barrel spawns 7", n1 == 7, f"n={n1}")
    b._damage(u, 999999)
    step(b, 10)
    n2 = len([x for x in b.units if x.name == "Skeleton"])
    check("Skeleton Balloon Evo death drops second barrel", n2 == 14, f"n={n2}")


# ---------------------------------------------------------------- AxeMan Evo
def probe_axe_evo():
    b = base("AxeMan_EV1")
    u = place(b, "AxeMan_EV1", 0, 9.0, 10.0)
    u.load_ms = 0
    k = place(b, "Knight", 1, 9.0, 11.2)
    k.load_ms = 0
    strong = _growth(94, b.card_level, b.card_dmg_pct)
    normal = _growth(70, b.card_level, b.card_dmg_pct)
    dmg1 = first_hit(b, k, 40)
    hp0 = k.hp
    dmg2 = 0
    for _ in range(8):
        b.step([])
        dmg2 = hp0 - k.hp
        if dmg2:
            break
    check("AxeMan Evo close hit 94", dmg1 == strong, f"dmg={dmg1} exp={strong}")
    check("AxeMan Evo return hit 70", dmg2 == normal, f"dmg={dmg2} exp={normal}")


def main():
    for fn in (probe_princess, probe_berserker, probe_firecracker, probe_barb_log,
               probe_zap_evo, probe_graveyard, probe_poison, probe_heal_spirit,
               probe_earthquake, probe_goblin_curse, probe_clone,
               probe_lightning, probe_arrows, probe_snowball, probe_tornado,
               probe_freeze, probe_rage, probe_royal_delivery, probe_ice_spirit,
               probe_electro_spirit, probe_skeleton_army,
               probe_firecracker_evo, probe_goblin_drill, probe_goblin_drill_evo,
               probe_skeleton_balloon_evo, probe_axe_evo):
        try:
            fn()
        except Exception as e:
            check(fn.__name__ + " (exception)", False, f"{type(e).__name__}: {e}")
    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
