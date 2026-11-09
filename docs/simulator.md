# Simulator notes

`crsim/engine.py` is a Python battle simulator driven by the local parameter
model at `data/logic.json`. It advances in 100 ms ticks on an 18 × 32 arena and
models towers, unit movement and targeting, elixir, card rotation, spells,
buildings, and match outcomes. `crsim/env.py` and
`cr_alphastar_agent/env/crsim_replica_env.py` expose agent observations and
actions.

The engine and policy are research approximations. Some card-specific behavior,
collision, timing, and live-game edge cases are simplified; a valid card entry
does not imply exact parity with the current game.

## Local parameter data

`data/logic.json` is local generated data and is intentionally excluded from
version control. To rebuild it, provide the decoded logic directory with
`--src` or `CRSIM_LOGIC_SRC`:

```powershell
python tools/load_logic.py --src C:\path\to\decoded\logic
```

The loader writes `data/logic.json` by default. Do not publish the decoded
parameter file or extracted game assets.

Run the regression checks from the repository root with `python -m pytest` and
the focused mechanics probes with `python tools/fidelity.py`.
