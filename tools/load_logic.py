"""Load decoded Supercell logic tables into one merged logic.json.

Input : decoded *.toml.dec / *.csv.dec under a directory supplied by the user
Output: repository\\data\\logic.json  (generated data file, keep local)

Usage:
    python tools\\load_logic.py --src <logic_dir> [--out <logic.json>]
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import pathlib
import sys
import tomllib


def parse_csv(path: pathlib.Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = list(csv.reader(text.splitlines()))
    if len(rows) < 3:
        return []
    names = rows[0]
    types = rows[1] + [None] * (len(rows[0]) - len(rows[1]))

    def cast(v: str, t: str | None):
        if v is None or v.strip() == "":
            return None
        t = (t or "string").lower()
        try:
            if t == "int": return int(v.replace(",", ""))
            if t == "float": return float(v)
            if t == "boolean": return v.strip().lower() in ("true", "1")
            if t.startswith("int_array"):
                return [int(x) for x in v.split(",") if x.strip() != ""]
            if t.startswith("string_array"):
                return [x for x in v.split(",") if x.strip() != ""]
        except ValueError:
            pass
        return v

    out = []
    for r in rows[2:]:
        if not any(x.strip() for x in r):
            continue
        out.append({
            n: cast(r[i] if i < len(r) else "", t)
            for i, (n, t) in enumerate(zip(names, types))
        })
    return out


def sheet_key(p: pathlib.Path) -> str:
    return p.name[: -len(".dec")].replace(".toml", "").replace(".csv", "")


# ---- Supercell TOML dialect sanitizer -------------------------------------
# Supercell's exporter deviates from strict TOML: files start with a blank
# line, end with a trailing NUL byte, may contain zero-width or whitespace
# section headers, duplicate keys, and stray NUL characters. Sanitize once
# here, then strict-parse.
def sanitize_supercell_toml(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\0", " ")
    lines: list[str] = []
    for ln in text.split("\n"):
        s = ln.lstrip(" \t")
        if s == "":  # blank or whitespace-only line
            lines.append("")
            continue
        # header like "[SpaceOnly]"? if the inside has no identifier chars, drop it
        if s.startswith("[") and s.endswith("]") and not any(k.isalnum() or k in "._-" for k in s[1:-1]):
            lines.append("")
            continue
        lines.append(f"{' ' * (len(ln) - len(s))}{s}")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def parse_sc_toml(path: pathlib.Path) -> dict:
    text = sanitize_supercell_toml(path.read_text(encoding="utf-8", errors="replace"))
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        # Supercell exports repeat section headers and keys. Drop the later
        # occurrence of each duplicated key/section (first definition wins)
        # so the file still parses as strict TOML.
        fixed: list[str] = []
        seen_sections: set[str] = set()
        seen_keys: dict[str, set] = {}
        section = ""
        skipping = False
        for ln in text.split("\n"):
            s = ln.strip()
            if s.startswith("["):
                is_array = s.startswith("[[")
                name = s.strip("[]").strip().lower() if not is_array else s[2:].split("]]", 1)[0].strip().lower()
                if not is_array:
                    if name in seen_sections:
                        skipping = True   # duplicate section: ignore body
                        continue
                    seen_sections.add(name)
                    skipping = False
                    section = name
                fixed.append(ln)
                continue
            if skipping:
                continue
            if "=" in s:
                key = s.split("=", 1)[0].strip().strip('"').strip("'").lower()
                bucket = seen_keys.setdefault(section, set())
                if key in bucket:
                    continue              # duplicate key: first wins
                bucket.add(key)
            fixed.append(ln)
        return tomllib.loads("\n".join(fixed))


def parse_toml_path(p: pathlib.Path) -> dict:
    return parse_sc_toml(p)


# ---- section routing -------------------------------------------------------
# Supercell TOMLs come in two shapes:
#   1. namespaced single units:  [CHARACTER.Knight] / [BUILDING.KingTower] /
#      [PROJECTILE.KingProjectile]  -> parsed as nested dict {"CHARACTER": {...}}
#   2. collection files:         [GoblinHut] / [Knight] / [BatProjectile]
#      -> parsed as one top-level dict per entry
# In both cases tomllib collapses dotted headers into their first component,
# so the effective namespace is always the top key's first component.
SKIP_NS = ("ACTION", "STATS", "SHEETSTATS")
CHAR_HEADS = ("CHARACTER", "EVO", "EXT")
AEO_HEADS = ("AEO", "AREA_EFFECT", "REGION")


def ingest(data: dict, model: dict, merged: dict, base_chars: dict,
           into_chars: bool, default_bucket: str | None,
           ext_sink: dict | None = None):
    """Route every top-level section of a parsed TOML into model buckets."""
    for ns, tree in data.items():
        if not isinstance(tree, dict):
            continue  # stray top-level scalar
        head = ns.split(".")[0].upper()
        if head == "ACTION":
            acts = model.setdefault("actions", {})
            for k, v in tree.items():
                if isinstance(v, dict):
                    acts[k] = v
            continue
        if head in SKIP_NS and head != "ACTION":
            continue
        if head in ("EXT", "EVO"):
            if ext_sink is not None:
                for k, v in tree.items():
                    if isinstance(v, dict):
                        ext_sink[k] = v
            continue
        if head == "BUFF":
            # buffs are declared inline in many character/spell files
            buffs = model["tables"].setdefault("character_buffs", {})
            for k, v in tree.items():
                if isinstance(v, dict):
                    buffs[k] = v
            continue
        if head == "DAMAGE_TYPE":
            dt = model.setdefault("damage_types", {})
            for k, v in tree.items():
                if isinstance(v, dict):
                    dt[k] = v
            continue
        if head == "ACTION":
            acts = model.setdefault("actions", {})
            for k, v in tree.items():
                if isinstance(v, dict):
                    acts[k] = v
            continue
        if head in ("CLIENT_ACTION", "VARIABLE",
                    "STATS", "SHEETSTATS", "SPELL_CHARACTER",
                    "SPELL_OTHER", "SPELL_EVOLVED", "SPELL_HERO",
                    "ABILITY", "PREDEFINEDDECK", "FILTER"):
            continue
        has_scalars = any(not isinstance(v, dict) for v in tree.values())
        # dotted header form: [PROJECTILE.X] -> {"PROJECTILE": {"X": {...}}}
        if head == "PROJECTILE":
            for k, v in tree.items():
                if isinstance(v, dict):
                    model["projectiles"][k] = v
        elif head in AEO_HEADS:
            for k, v in tree.items():
                if isinstance(v, dict):
                    model["area_effects"][k] = v
        elif head == "BUILDING":
            for k, v in tree.items():
                if isinstance(v, dict):
                    model["buildings"][k] = v
        elif into_chars and head in CHAR_HEADS:
            for k, v in tree.items():
                # unit def = dict containing scalar values (guards against
                # header-only tables and scalar sheets like Size="Medium")
                if isinstance(v, dict) and any(not isinstance(x, dict) for x in v.values()):
                    merged[k] = {**(base_chars.get(k) or {}), **v}
        elif default_bucket:
            model[default_bucket][ns] = tree
        # else: unknown namespace in a characters-dir file -> ignore


def eval_stat(base, val):
    """Evaluate Supercell derived-stat expressions like ["%",120] / ["=",4700]."""
    if (isinstance(val, list) and len(val) == 2
            and isinstance(val[0], str) and isinstance(val[1], (int, float))
            and not isinstance(val[1], bool)):
        op, x = val
        numeric = isinstance(base, (int, float)) and not isinstance(base, bool)
        if op == "=":
            return x
        if not numeric:
            return x
        if op == "%":
            return base * x / 100.0
        if op == "+":
            return base + x
        if op == "-":
            return base - x
        if op == "*":
            return base * x
        if op == "/" and x:
            return base / x
    return val


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default=os.environ.get("CRSIM_LOGIC_SRC"),
        help="decoded logic directory (or set CRSIM_LOGIC_SRC)",
    )
    ap.add_argument("--out", default=str(here / "data" / "logic.json"))
    a = ap.parse_args()
    if not a.src:
        ap.error("pass --src or set CRSIM_LOGIC_SRC to the decoded logic directory")
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)

    model = {"characters": {}, "buildings": {}, "spells": {},
             "projectiles": {}, "area_effects": {}, "hero_forms": {},
             "tables": {}, "damage_types": {}, "actions": {}}
    merged: dict[str, dict] = {}
    ext_sink: dict[str, dict] = {}

    base_path = src / "characters_base.toml.dec"
    base = parse_toml_path(base_path) if base_path.exists() else {}
    base_chars = base.get("CHARACTER", base)

    parsed: dict[pathlib.Path, dict] = {}
    failures: list[str] = []

    def read(p: pathlib.Path) -> dict:
        if p not in parsed:
            try:
                parsed[p] = parse_toml_path(p)
            except Exception as e:               # noqa: BLE001
                failures.append(f"{p.name}: {e}")
                parsed[p] = {}
        return parsed[p]

    def take(p: pathlib.Path, *, into_chars=True, bucket=None):
        ingest(read(p), model, merged, base_chars, into_chars, bucket,
               ext_sink=ext_sink)

    # 1) character library (base + evolutions, including hero-form subdir)
    char_dir = src / "characters"
    if char_dir.is_dir():
        for p in sorted(char_dir.glob("*.toml.dec")):
            take(p)
        for sub in sorted(char_dir.iterdir()):
            if not sub.is_dir():
                continue
            for p in sorted(sub.glob("*.toml.dec")):
                take(p)
                model["hero_forms"][f"{sub.name}/{sheet_key(p)}"] = read(p)

    # 2) buildings / spells / projectiles (root collection files)
    for p in [src / "buildings.toml.dec", src / "buildings_evo.toml.dec"]:
        if p.exists():
            take(p)
    for p in [src / "spells_characters.toml.dec", src / "spells_buildings.toml.dec",
              src / "spells_other.toml.dec", src / "spells_evolved.toml.dec",
              src / "spells_heroes.toml.dec"]:
        if p.exists():
            take(p, into_chars=False, bucket="spells")
    for p in [src / "projectiles.toml.dec", src / "projectiles_evo.toml.dec"]:
        if p.exists():
            take(p, into_chars=False, bucket="projectiles")

    # 3) every remaining TOML anywhere in the tree (events, game modes, ...)
    taken = {"characters_base.toml.dec", "buildings.toml.dec", "buildings_evo.toml.dec",
             "spells_characters.toml.dec", "spells_buildings.toml.dec",
             "spells_other.toml.dec", "spells_evolved.toml.dec",
             "spells_heroes.toml.dec", "projectiles.toml.dec",
             "projectiles_evo.toml.dec"}
    root_extra: list[pathlib.Path] = []
    for p in sorted(src.rglob("*.toml.dec")):
        if p.name in taken or (p.parent == src and p.name in taken):
            continue
        if p.parent == src:
            root_extra.append(p)
        else:
            take(p)                       # event / mode / nested definitions
    for p in root_extra:
        take(p)
        key = sheet_key(p)
        d = read(p)
        if isinstance(model["tables"].get(key), dict) and isinstance(d, dict):
            model["tables"][key].update(d)
        else:
            model["tables"][key] = d

    model["characters"] = merged

    # ---- resolve the generic [EXT.*] inheritance graph ----------------------
    ns_bucket = {
        "CHARACTER": merged,
        "BUILDING": model["buildings"],
        "PROJECTILE": model["projectiles"],
        "AEO": model["area_effects"],
        "AREA_EFFECT": model["area_effects"],
        "REGION": model["area_effects"],
        "BUFF": model["tables"].setdefault("character_buffs", {}),
        "ACTION": model["actions"],
    }

    def bucket_for(ns: str) -> dict:
        return ns_bucket.get(ns.upper(), {})

    def resolve_ext(name: str, depth: int = 0) -> dict:
        ext = ext_sink.get(name) or {}
        if not ext or depth > 10:
            return dict(ext)
        ref = ext.get("Base")
        if not isinstance(ref, str) or not ref:
            return dict(ext)
        ns, _, bname = ref.partition(".")
        if not bname:
            ns, bname = "", ref
        base_def = bucket_for(ns).get(bname) or {}
        if not base_def and bname in ext_sink:
            base_def = resolve_ext(bname, depth + 1)
        if not base_def:
            base_def = (merged.get(bname) or model["buildings"].get(bname)
                        or model["projectiles"].get(bname)
                        or model["area_effects"].get(bname) or {})
        if not isinstance(base_def, dict) or not base_def:
            return {}
        out = dict(base_def)
        for k, v in ext.items():
            if k == "Base":
                continue
            out[k] = eval_stat(base_def.get(k), v)
        return out

    for name in sorted(ext_sink):
        ref = ext_sink[name].get("Base")
        if not isinstance(ref, str) or not ref:
            continue
        ns = ref.split(".", 1)[0].upper()
        if ns in ("FILTER", "DAMAGE_TYPE", "PREDEFINEDDECK"):
            continue
        d = resolve_ext(name)
        if not d:
            continue
        if ns == "ACTION":
            target = model["actions"]
        elif ns == "BUILDING" or d.get("IsBuilding"):
            target = model["buildings"]
        elif ns == "PROJECTILE":
            target = model["projectiles"]
        elif ns in ("AEO", "AREA_EFFECT", "REGION"):
            target = model["area_effects"]
        elif ns == "BUFF":
            target = model["tables"]["character_buffs"]
        else:
            target = merged
        if target is not merged:
            merged.pop(name, None)
        target[name] = d

    # move any IsBuilding entries out of the troop library
    for name in list(merged):
        if merged[name].get("IsBuilding"):
            model["buildings"][name] = merged.pop(name)

    # ---- resolve string damage references (DAMAGE_TYPE.*) --------------------
    def fix_damage(d: dict):
        dv = d.get("Damage")
        if isinstance(dv, str):
            ref = model["damage_types"].get(dv)
            if ref:
                dd = dict(ref)
                if "BaseDamage" not in dd:
                    dd["BaseDamage"] = dd.get("Damage", 0)
                d["Damage"] = dd
    for bucket in (model["characters"], model["buildings"], model["projectiles"],
                   model["area_effects"]):
        for d in bucket.values():
            if isinstance(d, dict):
                fix_damage(d)

    # ---- card rows (CSV) -----------------------------------------------------
    for p in sorted(src.glob("*.csv.dec")):
        model["tables"][sheet_key(p)] = parse_csv(p)
    for p in sorted(src.glob("*/*.csv.dec")):
        model["tables"][f"{p.parent.name}/{sheet_key(p)}"] = parse_csv(p)

    # Evolution CSVs also carry "card info" placeholder rows that reuse the
    # Knight template (TID_SPELL_KNIGHT, NotInUse) for every non-evolved card.
    # Keep only rows that actually describe an evolution.
    ev_rows = model["tables"].get("spells_evolved")
    if isinstance(ev_rows, list):
        model["tables"]["spells_evolved"] = [
            r for r in ev_rows
            if not str(r.get("Name") or "").endswith("_EV1")
            or "_EVO_" in str(r.get("TID_INFO") or "")
            or str(r.get("HighresPlaceholderAsset") or "").endswith(
                "card_info_fx_evolution")
        ]

    # ---- action flattening for AEO spells ------------------------------------
    actions = model["actions"]

    def action_obj(a):
        if isinstance(a, str):
            return actions.get(a) or {}
        return a if isinstance(a, dict) else {}

    def _aeo_name(sd) -> str:
        if isinstance(sd, str):
            return sd
        if isinstance(sd, dict) and sd.get("Name"):
            model["area_effects"].setdefault(sd["Name"], sd)
            return sd["Name"]
        return ""

    def action_nodes(a, depth=0):
        """Yield every concrete action in a named/grouped action tree."""
        a = action_obj(a)
        if not a or depth > 8:
            return
        ct = a.get("ClassType", "")
        if ct in ("ActionGroup", "ActionSelect"):
            for s in a.get("SubActions") or []:
                yield from action_nodes(s, depth + 1)
            for key in ("OnActivateAction", "NextAction"):
                if a.get(key):
                    yield from action_nodes(a.get(key), depth + 1)
            return
        yield a

    def walk_action(a, t_ms, acc, depth=0):
        a = action_obj(a)
        if not a or depth > 8:
            return
        ct = a.get("ClassType", "")
        if ct in ("ActionGroup", "ActionSelect"):
            subs = a.get("SubActions") or []
            delays = a.get("SubActionsDelay") or []
            for i, s in enumerate(subs):
                delay = delays[i] if (i < len(delays)
                                      and isinstance(delays[i], (int, float))) else 0
                walk_action(s, t_ms + int(delay), acc, depth + 1)
        elif ct == "ActionSpawn":
            st = a.get("SpawnType")
            sd = a.get("SpawnData")
            stime = int(a.get("SpawnTime") or 0)
            if st == "CharacterType" and isinstance(sd, str):
                acc["spawns"].append((t_ms + stime, sd))
            elif st == "AreaEffectType":
                name = _aeo_name(sd)
                if name:
                    acc["aeos"].append((t_ms + stime, name))
            elif st == "BuffType":
                if isinstance(sd, str):
                    acc["buffs"].append(sd)
                elif isinstance(sd, dict) and sd.get("Name"):
                    model["tables"]["character_buffs"][sd["Name"]] = sd
                    acc["buffs"].append(sd["Name"])
        elif ct == "ActionSpawnToLocation":
            st = a.get("SpawnType")
            sd = a.get("SpawnData")
            stime = int(a.get("SpawnTime") or 0)
            if st == "CharacterType" and isinstance(sd, str):
                acc["spawns"].append((t_ms + stime, sd))
            elif st == "AreaEffectType":
                name = _aeo_name(sd)
                if name:
                    acc["aeos"].append((t_ms + stime, name))
            elif st == "BuffType":
                if isinstance(sd, str):
                    acc["buffs"].append(sd)
        elif ct == "ActionTakeDamage":
            dm = a.get("Damage")
            if isinstance(dm, dict):
                acc["damage"].append(dm)
        elif ct == "ActionClone":
            acc["clone"] = True
        elif ct == "ActionLaserBall":
            for det in a.get("OnDetectedUnitActionList") or []:
                walk_action(det, t_ms, acc, depth + 1)
        for nxt in (a.get("NextAction"),):
            if isinstance(nxt, dict):
                walk_action(nxt, t_ms, acc, depth + 1)

    def enrich_aeo(aeo: dict, depth=0) -> dict:
        out = dict(aeo)
        acc = {"spawns": [], "buffs": [], "damage": [], "aeos": [], "clone": False}
        for key in ("OnStartingAction", "OnHitAction"):
            if out.get(key):
                walk_action(out[key], 0, acc)
        # one-shot spawn when the field expires (skeleton-balloon barrels)
        end = action_obj(out.get("OnLifeTimeEndAction"))
        if end and end.get("ClassType") in ("ActionSpawn", "ActionSpawnToLocation") \
                and end.get("SpawnType") == "CharacterType" \
                and isinstance(end.get("SpawnData"), str):
            out["SpawnOnEnd"] = {
                "Character": end["SpawnData"],
                "Count": int(end.get("Count") or end.get("SpawnNumber") or 1),
                "Radius": float(end.get("SpawnRadius")
                                or end.get("SpawnAreaRadius") or 0) / 1000.0,
                "DeployTime": int(end.get("DeployTime")
                                  or end.get("SpawnTime") or 500),
            }
        for sub_t, sub_name in acc["aeos"]:
            sub = model["area_effects"].get(sub_name)
            if not isinstance(sub, dict):
                continue
            if depth < 4:
                merged_sub = enrich_aeo(sub, depth + 1)
                if sub_t <= 0:
                    if not out.get("Damage") and merged_sub.get("Damage"):
                        out["Damage"] = merged_sub["Damage"]
                    for k in ("Buff", "BuffTime", "HitSpeed", "LifeDuration",
                              "Radius", "SpawnCharacter", "SpawnNumber",
                              "SpawnInterval", "SpawnInitialDelay", "SpawnSchedule",
                              "Buffs", "Clone", "BuildingDamagePercent",
                              "CrownTowerDamagePercent"):
                        if k not in out and k in merged_sub:
                            out[k] = merged_sub[k]
                    if merged_sub.get("Buffs"):
                        out.setdefault("Buffs", [])
                        out["Buffs"] += [b for b in merged_sub["Buffs"]
                                         if b not in out["Buffs"]]
                    if "Building" in str(merged_sub.get("Filter") or ""):
                        out["DamageBuildingsOnly"] = True
                else:
                    out.setdefault("SubAeos", [])
                    out["SubAeos"].append({"t": sub_t, "AEO": sub_name})
        parent_filter = str(out.get("Filter") or "")
        if ("MainTargets" in parent_filter or "Characters" in parent_filter
                or "Character" in parent_filter or "troops" in parent_filter.lower()):
            out["BuffTroopsOnly"] = True
        if acc["damage"] and not out.get("Damage"):
            out["Damage"] = dict(acc["damage"][0])
        if acc["spawns"]:
            out["SpawnSchedule"] = [{"t": t, "Character": c}
                                    for t, c in sorted(acc["spawns"])]
            names = [c for _, c in acc["spawns"]]
            if not out.get("SpawnCharacter") and names:
                out["SpawnCharacter"] = max(set(names), key=names.count)
            out.setdefault("SpawnInitialDelay", acc["spawns"][0][0])
            if len(acc["spawns"]) > 1:
                diffs = [b[0] - a[0] for a, b in zip(acc["spawns"], acc["spawns"][1:])
                         if b[0] > a[0]]
                if diffs:
                    out.setdefault("SpawnInterval", sorted(diffs)[len(diffs) // 2])
            out.setdefault("SpawnLimit", len(acc["spawns"]))
        if acc["buffs"]:
            # prefer an explicit *_Base buff (Vines) over size variants
            base = next((b for b in acc["buffs"] if b.endswith("_Base")), None)
            pick = [base] if base else list(dict.fromkeys(acc["buffs"]))
            out.setdefault("Buffs", [])
            for b in pick:
                if b and b not in out["Buffs"]:
                    out["Buffs"].append(b)
        if acc["clone"]:
            out["Clone"] = True
        return out

    for name, aeo in list(model["area_effects"].items()):
        if not isinstance(aeo, dict):
            continue
        if aeo.get("OnStartingAction") or aeo.get("OnHitAction") \
                or aeo.get("OnLifeTimeEndAction"):
            en = enrich_aeo(aeo)
            # keep internal action definitions out of the runtime data
            en.pop("OnStartingAction", None)
            en.pop("OnHitAction", None)
            en.pop("OnLifeTimeEndAction", None)
            model["area_effects"][name] = en

    # ---- card-row fixes / hardcoded group cards ------------------------------
    all_rows = {}
    for key in ("spells_characters", "spells_buildings", "spells_other",
                "spells_evolved", "spells_hero_form"):
        for r in model["tables"].get(key) or []:
            if r.get("Name"):
                all_rows[r["Name"]] = r
    if "ThreeMusketeers" in all_rows:
        r = all_rows["ThreeMusketeers"]
        if not r.get("SummonCharacter"):
            r["SummonCharacter"] = "Musketeer"
        r.setdefault("SummonNumber", 3)
        r.setdefault("SummonWidth", 3)
        r.setdefault("SummonDeployDelay", 100)
    if "MergeMaiden" in all_rows:
        r = all_rows["MergeMaiden"]
        if not r.get("SummonCharacter"):
            r["SummonCharacter"] = "MergeMaiden_Mounted"
        r.setdefault("SummonNumber", 1)

    # ---- bind card-row action scripts to runtime unit data -------------------
    def _aeo_meaningful(an: str) -> bool:
        ae = model["area_effects"].get(an) or {}
        return bool(ae.get("Damage") or ae.get("Buff") or ae.get("Buffs")
                    or ae.get("SpawnCharacter") or ae.get("Projectile")
                    or ae.get("SubAeos") or ae.get("Clone"))

    def _spawn_aeo_name(act: dict) -> str:
        if act.get("SpawnType") != "AreaEffectType":
            return ""
        an = _aeo_name(act.get("SpawnData"))
        return an if an and _aeo_meaningful(an) else ""

    for store in ("characters", "buildings"):
        for uname, udef in list(model[store].items()):
            if not isinstance(udef, dict):
                continue
            used_pop: set[str] = set()

            def add_deploy(act):
                an = _spawn_aeo_name(act)
                if an:
                    udef.setdefault("DeployAeos", [])
                    if an not in udef["DeployAeos"]:
                        udef["DeployAeos"].append(an)

            for act in action_nodes(udef.get("OnStartingAction")):
                ct = act.get("ClassType", "")
                if ct in ("ActionSpawn", "ActionSpawnToLocation"):
                    add_deploy(act)
                elif ct == "ActionSkeletonBarrelPopBalloon":
                    names = act.get("ContainerAeoList") or []
                    pcts = act.get("DropBalloonAtHpList") or []
                    xs = act.get("OffsetXList") or []
                    ys = act.get("OffsetYList") or []
                    for i, pct in enumerate(pcts):
                        if i >= len(names) or names[i] in used_pop:
                            continue
                        used_pop.add(names[i])
                        udef.setdefault("HpDrops", []).append({
                            "Pct": pct,
                            "AEO": names[i],
                            "DX": (xs[i] if i < len(xs) else 0) / 1000.0,
                            "DY": (ys[i] if i < len(ys) else 0) / 1000.0,
                        })
                elif ct == "ActionGoblinDrillEvoRelocate":
                    # emerge hit (also used for each reappearance)
                    for sub in action_nodes(act.get("FirstAppearAction")):
                        add_deploy(sub)
                    spawn_lists = []
                    for hidden in act.get("HideActions") or []:
                        names = [a2.get("SpawnData") for a2 in action_nodes(hidden)
                                 if a2.get("ClassType") in
                                 ("ActionSpawn", "ActionSpawnToLocation")
                                 and a2.get("SpawnType") == "CharacterType"
                                 and isinstance(a2.get("SpawnData"), str)]
                        if names:
                            spawn_lists.append(names)
                    udef["HpRelocate"] = {
                        "Thresholds": act.get("HideHpThresholds") or [],
                        "Spawns": spawn_lists,
                        "HideMs": int(act.get("HideTime") or 0),
                        "AEO": next((an for sub in
                                     action_nodes(act.get("FirstAppearAction"))
                                     for an in [_spawn_aeo_name(sub)] if an), ""),
                    }

            for act in action_nodes(udef.get("OnDeathAction")):
                ct = act.get("ClassType", "")
                if ct == "ActionSkeletonBarrelPopBalloon":
                    names = act.get("ContainerAeoList") or []
                    xs = act.get("OffsetXList") or []
                    ys = act.get("OffsetYList") or []
                    for i, an in enumerate(names):
                        if an in used_pop:
                            continue
                        used_pop.add(an)
                        udef.setdefault("DeathAeos", []).append({
                            "AEO": an,
                            "DX": (xs[i] if i < len(xs) else 0) / 1000.0,
                            "DY": (ys[i] if i < len(ys) else 0) / 1000.0,
                        })
                elif ct in ("ActionSpawn", "ActionSpawnToLocation") \
                        and act.get("SpawnType") == "CharacterType" \
                        and isinstance(act.get("SpawnData"), str):
                    if not udef.get("DeathSpawnCharacter"):
                        udef["DeathSpawnCharacter"] = act["SpawnData"]
                        udef["DeathSpawnCount"] = int(
                            act.get("Count") or act.get("SpawnNumber") or 1)

    # executioner-style projectile controllers carry their damage in an action
    for aname, act in list(model["actions"].items()):
        if not isinstance(act, dict):
            continue
        if act.get("ClassType") != "ActionExecutionerEvoProjectile":
            continue
        owner = aname.split("_Projectile_Controller")[0]
        udef = (model["characters"].get(owner)
                or model["buildings"].get(owner) or {})
        if not udef:
            continue
        if act.get("Damage"):
            udef["Damage"] = act["Damage"]
        for src_key, dst_key in (("StrongDamage", "StrongDamage"),
                                 ("StrongDamageRange", "StrongDamageRange"),
                                 ("FirstStrongHitPushback", "FirstStrongHitPushback")):
            if act.get(src_key):
                udef[dst_key] = act[src_key]

    # drilling units morph into their combat building at the target
    for r in all_rows.values():
        s = r.get("SummonCharacter") or ""
        sdef = (model["characters"].get(s)
                or model["buildings"].get(s) or {})
        morph = sdef.get("SpawnPathfindMorph")
        if isinstance(morph, str) and morph and (
                morph in model["characters"] or morph in model["buildings"]):
            r["MorphFrom"] = s
            r["SummonCharacter"] = morph

    # AEO alias resolution (case-insensitive / suffix variants)
    aeo_lower = {k.lower(): k for k in model["area_effects"]}
    for r in all_rows.values():
        a = r.get("AreaEffectObject")
        if isinstance(a, str) and a not in model["area_effects"]:
            cand = (aeo_lower.get(a.lower())
                    or aeo_lower.get((a + "_AeO").lower())
                    or aeo_lower.get((a + "AeO").lower())
                    or aeo_lower.get((a + "AOE").lower()))
            if cand:
                r["AreaEffectObject"] = cand

    # projectiles referenced by cards that only exist as SpawnProjectile chains
    def projectile_chain(name: str, depth: int = 0) -> dict:
        p = model["projectiles"].get(name) or {}
        if (not p.get("Damage") and not p.get("SpawnCharacter")
                and isinstance(p.get("SpawnProjectile"), str) and depth < 4):
            return projectile_chain(p["SpawnProjectile"], depth + 1)
        return p
    model["projectile_chain"] = {
        n: projectile_chain(n) for n in list(model["projectiles"])[:0]
    }  # placeholder kept empty; engine follows chains at runtime

    # ---- per-level scaling (card / tower growth from globals.csv) ------------
    gtab = {r.get("Name"): r for r in (model["tables"].get("globals") or [])
            if r.get("Name")}
    def g(key):  # globals.csv rows: Name, NumberValue, BooleanValue, TextValue, ...
        r = gtab.get(key) or {}
        v = r.get("NumberValue")
        return v if isinstance(v, (int, float)) else None

    def scaling(dmg_key, hp_key):  # both *_AFTER_TOURNAMENTCAP variants exist
        return {"damage_pct_per_level": g(dmg_key), "hitpoint_pct_per_level": g(hp_key),
                "damage_pct_after_tournament_cap": g(f"{dmg_key}_AFTER_TOURNAMENTCAP"),
                "hitpoint_pct_after_tournament_cap": g(f"{hp_key}_AFTER_TOURNAMENTCAP")}
    spell = scaling("DAMAGE_INCREASE_PERCENT_PER_SPELL_LEVEL", "HITPOINT_INCREASE_PERCENT_PER_SPELL_LEVEL")
    king = scaling("DAMAGE_INCREASE_PERCENT_PER_KING_LEVEL", "HITPOINT_INCREASE_PERCENT_PER_KING_LEVEL")
    tower = scaling("DAMAGE_INCREASE_PERCENT_PER_TOWER_LEVEL", "HITPOINT_INCREASE_PERCENT_PER_TOWER_LEVEL")
    exp_rows = model["tables"].get("exp_levels") or []
    model["level_scaling"] = {
        "card": spell,           # troops/units/buildings/spells all use "spell" scaling
        "king_tower": king,
        "princess_tower": tower,
        "tower_scaling_start_exp_level": g("TOWER_SCALING_START_EXP_LEVEL"),
        "tournament_max_exp_level": g("TOURNAMENT_MAX_EXP_LEVEL"),
        "exp_levels": [{"exp": r.get("Name"), "summoner": r.get("SummonerLevel"),
                        "tower": r.get("TowerLevel"), "troop": r.get("TroopLevel")}
                       for r in exp_rows if isinstance(r.get("SummonerLevel"), int)],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")

    # ---- summary ----
    n_units = len(model["characters"])
    n_bld = len(model["buildings"])
    n_spel = len(model["spells"])
    n_proj = len(model["projectiles"])
    n_tabs = len(model["tables"])
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    print(f"characters={n_units} buildings={n_bld} spells={n_spel} projectiles={n_proj} tables={n_tabs}")
    for u in ("Knight", "HogRider", "TowerPrincess"):
        c = model["characters"].get(u)
        if c:
            print(f"  char {u}: hp={c.get('Hitpoints')} dmg={c.get('Damage')} hs={c.get('HitSpeed')} range={c.get('Range')} speed={c.get('Speed')}")
    for b in ("KingTower", "PrincessTower", "Cannon", "Tesla"):
        c = model["buildings"].get(b)
        if c:
            proj = model["projectiles"].get(c.get("Projectile") or "")
            print(f"  bld  {b}: hp={c.get('Hitpoints')} hs={c.get('HitSpeed')} range={c.get('Range')} "
                  f"proj={c.get('Projectile')} projDmg={proj and proj.get('Damage')}")
    ls = model["level_scaling"]
    print(f"  level_scaling card={ls['card']['damage_pct_per_level']}% dmg "
          f"king={ls['king_tower']['hitpoint_pct_per_level']}%hp tower={ls['princess_tower']['hitpoint_pct_per_level']}%hp")
    tl = model["tables"].get("battle_timelines") or []
    if tl:
        print(f"  battle_timelines rows={len(tl)}")
    print(f"  spawn_groups keys={list((model['tables'].get('spawn_groups') or {}).keys())[:6]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
