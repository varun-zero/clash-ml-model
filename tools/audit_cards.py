"""Audit every deployable card row against the engine's behavior coverage.

    python tools/audit_cards.py [--csv data/card_audit.csv]

For each card row (ManaCost 1..10, not NotInUse) this resolves the unit /
spell definition and reports:
  * how the engine deploys it (troop / building / spell-AEO / spell-proj)
  * which data-driven behaviors it has and whether the engine handles them
  * any definition gaps (missing summon, missing projectile/AEO/buff, ...)
  * unique mechanics that are still stubs (invisibility, attach, scripts)

Exit code 0 always; the report is the worklist.
"""
from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from crsim.engine import load_model  # noqa: E402

M = load_model()
CH = M["characters"]
BLD = M["buildings"]
PROJ = M["projectiles"]
AEO = dict(M.get("area_effects") or {})
AEO.update(M["tables"].get("area_effect_objects") or {})
BUFFS = M["tables"].get("character_buffs") or {}

CARD_TABLES = ("spells_characters", "spells_buildings", "spells_other",
               "spells_evolved", "spells_hero_form")


def card_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for key in CARD_TABLES:
        for r in M["tables"].get(key) or []:
            n = r.get("Name")
            if not n:
                continue
            if not isinstance(r.get("ManaCost"), int):
                continue
            if not (1 <= r["ManaCost"] <= 10):
                continue
            if r.get("NotInUse"):
                continue
            r = dict(r)
            r["_table"] = key
            rows.setdefault(n, r)
    return rows


def unit_def(name: str) -> tuple[dict, str]:
    if name in BLD:
        return BLD[name], "building"
    if name in CH:
        return CH[name], "troop"
    return {}, ""


def kind_of(row: dict) -> str:
    if row.get("Name") == "Mirror":
        return "mirror"
    s = row.get("SummonCharacter") or ""
    d, k = unit_def(s)
    if k == "building" and d.get("Hitpoints"):
        return "building"
    if k == "troop":
        return "troop"
    if s:
        return "summon-misc"          # bottle / spell-only summon
    if row.get("AreaEffectObject"):
        return "spell-aeo"
    if row.get("Projectile"):
        return "spell-proj"
    if row.get("InstantDamage") or row.get("InstantHeal") or row.get("HealPerSecond"):
        return "spell-instant"
    return "unknown"


def projectile_of(d: dict) -> dict:
    """Resolve the damage-carrying projectile, following spawn chains."""
    name = d.get("CustomFirstProjectile") or d.get("Projectile") or ""
    p = PROJ.get(name) or {}
    seen: set[str] = set()
    while (p and not (p.get("Damage") or p.get("SpawnCharacter")
                      or p.get("Buff") or p.get("TargetBuff"))
           and isinstance(p.get("SpawnProjectile"), str)
           and p["SpawnProjectile"] not in seen):
        seen.add(p["SpawnProjectile"])
        p = PROJ.get(p["SpawnProjectile"]) or {}
    return p


def aeo_has_payload(name: str, depth: int = 0) -> bool:
    a = AEO.get(name) or {}
    if not a or depth > 4:
        return False
    if a.get("Damage") or a.get("SpawnCharacter") or a.get("Projectile") \
            or a.get("Clone") or a.get("SpawnSchedule") or a.get("SubAeos"):
        return True
    for bname in ([a.get("Buff")] if a.get("Buff") else []) + list(a.get("Buffs") or []):
        bd = BUFFS.get(bname or "") or {}
        if bd.get("DamagePerSecond") or bd.get("HealPerSecond"):
            return True
    for sub in a.get("SubAeos") or []:
        an = sub.get("AEO") if isinstance(sub, dict) else sub
        if isinstance(an, str) and aeo_has_payload(an, depth + 1):
            return True
    if isinstance(a.get("SpawnAreaEffectObject"), str) \
            and aeo_has_payload(a["SpawnAreaEffectObject"], depth + 1):
        return True
    return False


def has_damage(d: dict) -> bool:
    p = projectile_of(d)
    if (d.get("Damage") or p.get("Damage") or d.get("InstantDamage")):
        return True
    seq = d.get("AttackSequenceList")
    if isinstance(seq, list) and any(
            isinstance(x, dict) and (x.get("Damage") or 0) for x in seq):
        return True
    for key in ("DeployAeos", "DeathAeos"):
        for an in d.get(key) or []:
            name = an.get("AEO") if isinstance(an, dict) else an
            if isinstance(name, str) and aeo_has_payload(name):
                return True
    if isinstance(p.get("SpawnAreaEffectObject"), str) \
            and aeo_has_payload(p["SpawnAreaEffectObject"]):
        return True
    if d.get("DeathAreaEffect") and AEO.get(d["DeathAreaEffect"], {}).get("Damage"):
        return True
    if d.get("DeathSpawnCharacter"):
        return True
    if d.get("DeathAreaEffect"):
        return True
    if d.get("SpawnCharacter") and d.get("SpawnNumber"):
        return True          # spawner buildings / huts
    if d.get("ManaCollectAmount"):
        return True          # elixir support
    if d.get("Kamikaze") and (p.get("Damage") or d.get("Damage")
                              or d.get("DeathAreaEffect")):
        return True
    return False


def behavior_tags(d: dict, row: dict) -> list[str]:
    """Features present in data.  '!' marks a behavior the engine stubs."""
    p = projectile_of(d)
    t: list[str] = []
    if d.get("FlyingHeight"):
        t.append("flying")
    if d.get("AttacksAir"):
        t.append("air")
    if d.get("AttacksGround") is False:
        t.append("ground-only")
    if d.get("TargetOnlyBuildings"):
        t.append("buildings-only")
    if d.get("AreaDamageRadius") or p.get("Radius"):
        t.append("splash")
    if d.get("Projectile"):
        t.append("proj" if p else "proj-missing!")
    if d.get("ChargeRange"):
        t.append("charge")
    if d.get("DashMaxRange") or d.get("DashDamage"):
        t.append("dash")
    if d.get("Kamikaze"):
        t.append("kamikaze")
    if d.get("ShieldHitpoints"):
        t.append("shield")
    if d.get("MinimumRange"):
        t.append("min-range")
    if d.get("VariableDamage2") or d.get("VariableDamage3"):
        t.append("ramp")
    if d.get("SpawnCharacter") and d.get("SpawnNumber") and (
            d.get("SpawnPauseTime") or d.get("SpawnStartTime")):
        t.append("spawner")
    if d.get("DeathSpawnCharacter"):
        t.append("death-spawn")
    if d.get("DeathAreaEffect"):
        t.append("death-aeo")
    if d.get("ManaCollectAmount"):
        t.append("elixir")
    if d.get("AttackPushBack"):
        t.append("knockback")
    if d.get("JumpEnabled"):
        t.append("jump")
    if d.get("Invisible") or "Invisibility" in str(d.get("BuffWhenNotAttacking") or ""):
        t.append("invisible!")
    if d.get("AttachedCharacter"):
        t.append("attach!")
    if d.get("OnAttackAction") or d.get("OnHitTargetAction") or row.get("Effect"):
        t.append("script?")
    if d.get("BuffWhenNotAttacking"):
        t.append("passive-buff")
    if d.get("DamageReduction") or d.get("BuffOnDamage"):
        t.append("dmg-reduction!")
    if row.get("SummonCharacterSecond"):
        t.append("second-unit!")
    if row.get("MultipleProjectiles") or row.get("ProjectileWaves"):
        t.append("multi-proj!")
    if row.get("AreaEffectObject"):
        t.append("aeo")
    if row.get("InstantHeal") or row.get("HealPerSecond"):
        t.append("heal!")
    return t


def check_problems(name: str, row: dict) -> list[str]:
    probs: list[str] = []
    k = kind_of(row)
    s = row.get("SummonCharacter") or ""
    if k in ("troop", "building"):
        d, _ = unit_def(s)
        if not d:
            probs.append(f"missing summon def {s!r}")
            return probs
        if not d.get("Hitpoints"):
            probs.append("no Hitpoints")
        pname = d.get("CustomFirstProjectile") or d.get("Projectile")
        if pname and pname not in PROJ and not d.get("Damage"):
            probs.append(f"missing projectile {pname!r}")
        p = projectile_of(d)
        if not has_damage(d):
            probs.append("no damage")
        attacks = bool(d.get("Damage") or p.get("Damage")
                       or d.get("AttackSequenceList"))
        if attacks and not d.get("HitSpeed"):
            probs.append("no HitSpeed")
        if k == "troop" and not d.get("Speed") and not d.get("JumpEnabled") \
                and not d.get("DashMaxRange") and not d.get("ChargeRange"):
            probs.append("no Speed")
        if attacks and not d.get("Range"):
            probs.append("no Range")
        if d.get("DeathAreaEffect") and d["DeathAreaEffect"] not in AEO:
            probs.append(f"missing death AEO {d['DeathAreaEffect']!r}")
        if d.get("SpawnCharacter") and d.get("SpawnNumber"):
            sd, _ = unit_def(d["SpawnCharacter"])
            if not sd:
                probs.append(f"missing spawn unit {d['SpawnCharacter']!r}")
        if d.get("BuffWhenNotAttacking") and d["BuffWhenNotAttacking"] not in BUFFS:
            probs.append(f"missing buff {d['BuffWhenNotAttacking']!r}")
    elif k == "mirror":
        pass
    elif k == "summon-misc":
        d, _ = unit_def(s)
        if not d:
            probs.append(f"missing summon def {s!r}")
        elif d.get("DeathAreaEffect") and d["DeathAreaEffect"] not in AEO:
            probs.append(f"missing death AEO {d['DeathAreaEffect']!r}")
    elif k == "spell-aeo":
        a = AEO.get(row["AreaEffectObject"])
        if not a:
            probs.append(f"missing AEO {row['AreaEffectObject']!r}")
        else:
            if a.get("Buff") and a["Buff"] not in BUFFS:
                probs.append(f"missing buff {a['Buff']!r}")
            if a.get("SpawnCharacter"):
                sd, _ = unit_def(a["SpawnCharacter"])
                if not sd:
                    probs.append(f"missing spawn unit {a['SpawnCharacter']!r}")
            meaningful = (a.get("Damage") or a.get("Buff") or a.get("Buffs")
                          or a.get("SpawnCharacter") or a.get("SpawnSchedule")
                          or a.get("Clone") or a.get("Projectile")
                          or a.get("SubAeos") or a.get("SpawnAreaEffectObject"))
            if not meaningful:
                probs.append("AEO has no damage/buff/spawn")
    elif k == "spell-proj":
        p = PROJ.get(row["Projectile"])
        if not p:
            probs.append(f"missing projectile {row['Projectile']!r}")
        else:
            p2 = projectile_of({"Projectile": row["Projectile"]})
            if not (p.get("Damage") or p2.get("Damage") or p.get("Buff")
                    or p.get("SpawnCharacter") or p2.get("SpawnCharacter")
                    or p.get("SpawnAreaEffectObject")
                    or p.get("ChainedHitCount")):
                probs.append("projectile has no damage/spawn")
    elif k == "spell-instant":
        if not (row.get("InstantDamage") or row.get("InstantHeal") or row.get("HealPerSecond")):
            probs.append("no instant effect")
    else:
        probs.append("unresolved card kind")
    return probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--all", action="store_true", help="include hero/event variants")
    a = ap.parse_args()

    rows = card_rows()
    report = []
    for name in sorted(rows):
        row = rows[name]
        if not a.all:
            if row.get("NotVisible"):
                continue
            if name.endswith("_hero") or name.startswith("Global") \
                    or name in ("AngryBarbarians", "AngryBarbarians_EV1",
                                "AngryBarbarians_hero"):
                continue
        k = kind_of(row)
        s = row.get("SummonCharacter") or ""
        d, _ = unit_def(s)
        tags = behavior_tags(d, row)
        probs = check_problems(name, row)
        report.append({"card": name, "cost": row.get("ManaCost"),
                       "table": row["_table"], "kind": k,
                       "summon": s, "tags": ",".join(tags),
                       "problems": "; ".join(probs)})

    by_kind = collections.Counter(r["kind"] for r in report)
    bad = [r for r in report if r["problems"]]
    stubs = [r for r in report if "!" in r["tags"] or "script?" in r["tags"]]
    print(f"cards audited: {len(report)}  by kind: {dict(by_kind)}")
    print(f"definition problems: {len(bad)}")
    for r in bad:
        print(f"  {r['card']:22} {r['kind']:13} {r['problems']}")
    print(f"\nstubbed/unique mechanics: {len(stubs)}")
    for r in stubs:
        print(f"  {r['card']:22} {r['kind']:13} {r['tags']}")
    print("\n-- supported behavior tags --")
    tag_counts = collections.Counter(
        t.rstrip("!").rstrip("?")
        for r in report for t in r["tags"].split(",") if t)
    for t, n in tag_counts.most_common():
        print(f"  {t:16} {n}")

    if a.csv:
        out = pathlib.Path(a.csv)
        if not out.is_absolute():
            out = pathlib.Path(__file__).resolve().parent.parent / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
            w.writeheader()
            w.writerows(report)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
