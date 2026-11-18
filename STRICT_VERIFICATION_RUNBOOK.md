# STRICT_VERIFICATION_RUNBOOK.md

**Mandatory operating procedure for every live MuMu test run.**

This repository previously produced *fabricated* match results (`Arrows x8`
deployments in a 17-card match, "LOSS 2 - 8" crown scores, trophy counts that
did not add up).  Every rule below exists because a real failure slipped
through the old, comment-driven process.

> **Ground truth hierarchy (highest first)**
> 1. The **in-game Clash Royale Battle Log** (authoritative for result *and*
>    crown score).
> 2. The **post-match end screen**, read with
>    `PerceptionAdapter.detect_match_end_result()`.
> 3. The **match video** in `match_recordings/`.
> 4. `trophy_climb_history.json` - a *log*, never a source of truth.
>
> A commit message, a markdown note, a telemetry line or a JSON history entry
> is **never** evidence.

---

## 0. Pre-flight (before MuMu is even started)

Run the offline gates first. If any of them fails, **do not** start a live
match.

```powershell
python tools/verify_ground_truth.py
python tools/diagnose_truth.py
python -m pytest tests/ -q
```

Requirements:

* `tools/verify_ground_truth.py` prints `RESULT: ALL <N> CHECKS PASSED`.
* `pytest tests/` is 100 % green.
* `git status` is clean, or every change is explained in the run notes.

If the offline gates fail, the live run is invalid by definition.  Fix the
offline failure first.

### 0.1 ADB transport (read this before the first live run)

MuMu exposes the same guest on several transports and they are **not**
equally reliable:

| Transport | Survives an adb-server restart? | Use |
| :--- | :--- | :--- |
| `emulator-XXXX` (emulator console) | **yes** - the emulator re-registers it | preferred |
| `127.0.0.1:16384` (raw TCP) | **no** - silently dropped | fallback only |

`find_adb_serial()` now probes candidates with a real command
(`adb -s <serial> shell echo`) and prefers the console transport.  It also
re-connects the TCP transport when needed.  **Do not pin `--adb-device
127.0.0.1:16384`** unless you know the server is stable - a dropped TCP
transport used to kill the bridge with an endless
"[ADB] Shell pipe disconnected" loop.

Use the emulator's own `adb.exe` (it is the binary the emulator ships and
the one whose server the emulator manages):

```powershell
$env:ADB = 'C:\Program Files\Netease\MuMuPlayer\nx_main\adb.exe'
```

### 0.2 Console encoding

The bridge forces UTF-8 on stdout/stderr (`_force_utf8_stdio()`).  Without
it a single emoji in a log line (`⚔`, `🎥`, `🏁`) raised
`UnicodeEncodeError` on the default Windows `cp1252` console and **killed
the bridge in the middle of a live match** - and it silently disabled the
match video recorder, because the recorder prints its status line before it
starts recording.  If you ever see that traceback again, the UTF-8
reconfiguration has been removed.

### 0.3 Screen calibration (measured on a real 900x1600 in-battle frame)

```
elixir drop icon        x 0.208 .. 0.242   (NEVER elixir)
elixir counter ("7")    x 0.243 .. 0.309   (white glyph, drawn over pip 0)
ten elixir pips          x 0.269 .. 0.967   (10 segments, ~62.8 px each)
elixir bar rows         y 0.965 .. 0.987
hand cards              x 0.230 .. 0.945   (4 cards, ~148 px wide, ~165 px pitch)
hand card centres       x 0.312 / 0.494 / 0.680 / 0.864
```

The elixir reader samples `x 0.258 .. 0.955` and masks the white counter
glyph, so the counter cannot hide a filled pip.

---

## 1. `read_elixir` misreads the bar

### Symptom

* The HUD elixir number disagrees with `perception.current_elixir`.
* The bot tries to play a card it cannot afford (`[REJECTED TAP]` with
  `elixir_before` lower than the card cost).
* `read_elixir` returns a value that never changes, or always returns 0 / 10.

### Diagnosis

1. Capture the elixir bar region:

   ```powershell
   python -c "import cv2; from deploy_mumu import PerceptionAdapter; import torch; a=PerceptionAdapter(device=torch.device('cpu')); f=cv2.imread('.tmp/current_screen.png'); print(a.save_elixir_debug(f))"
   ```

   This writes **`.tmp/elixir_debug.png`** - the pip band of the bar with the
   *detected* pip boundaries drawn as cyan vertical lines.

2. Open `.tmp/elixir_debug.png` and check:

   | Observation | Meaning | Action |
   | :--- | :--- | :--- |
   | The purple **drop icon** is still inside the crop | the icon is not excluded | increase `ELIXIR_ICON_MAX_X` in `deploy_mumu.py` until the icon is outside the crop |
   | The cyan lines do not line up with the real pip separators | the bar does not start/end where we think | adjust `ELIXIR_PIP_X0` / `ELIXIR_PIP_X1` |
   | The **white counter** ("7", "10") covers a pip | the glyph hides a filled pip | confirm the reader still reports the correct count; if not, extend the glyph masking in `read_elixir` |
   | The whole crop is uniform magenta | the frame is not an in-battle frame | re-capture |

3. Re-run the offline gate after every coordinate change:

   ```powershell
   python tools/verify_ground_truth.py
   python -m pytest tests/test_deploy_mumu.py -q
   ```

   `tests/test_deploy_mumu.py::test_read_elixir_reads_ten_discrete_pips` must
   still read every pip count 0..10 exactly, and
   `test_elixir_reader_on_the_real_captured_frame` must still read 10.0 on
   `.tmp/diag_live.png`.

4. **Never** fall back to the old continuous fill ratio.  The drop icon is
   always "full", so a continuous ratio permanently over-reports the elixir
   by roughly one pip.

### Reference geometry (measured on a real 900x1600 MuMu capture)

```
x 0.080 .. 0.263   purple drop icon + white elixir counter  (NEVER elixir)
x 0.274 .. 0.966   the ten elixir pips
y 0.965 .. 0.987   the bar itself
```

---

## 2. `[REJECTED TAP]` fires more than twice in a match

### Symptom

The console prints

```
[REJECTED TAP] <Card> (Slot N) -> Tile (x, y) was NOT deployed: {...}
```

more than twice in a single match.

A rejected tap is **not** a bug by itself - it means the verification frame
proved the card never left the hand.  But repeated rejections mean the bot is
not playing at all, so the match is worthless.

### Diagnosis

1. Open **`.tmp/rejected_tap.png`** (the verification frame that was captured
   0.55 s after the tap).

   | Observation | Meaning | Action |
   | :--- | :--- | :--- |
   | The card is still in its hand slot | the selection never happened | increase the dwell time between the hand tap and the arena tap (`deploy_card`, currently `0.16 s`) |
   | The card is **lifted** (raised above the other slots) | the card was selected but the arena tap missed | the arena tap landed outside the arena - check the target tile |
   | The elixir did not change **and** the slot is empty | the card was played but the elixir reader misread | go to section 1 |
   | A **modal** (chest, level-up, offer) covers the screen | the taps hit the modal | dismiss the modal before the battle starts, see section 4 |
   | The tap target is on the top HUD | the arena mapping is wrong | check `perception.arena_box`; targets are clamped to `tile_x in [1..16]`, `tile_y in [1..28]` |

2. The diagnostics printed with the rejection tell you which rule fired:

   | `reason` | Meaning |
   | :--- | :--- |
   | `elixir dropped` | verified - this is *not* a rejection |
   | `card left the hand slot` | verified - this is *not* a rejection |
   | `no evidence the card was deployed` | the real rejection reason |
   | `no verification frame` | the capture failed; the emulator was probably busy |

3. Adjustments (edit `deploy_mumu.py`):

   * **Timing** - `deploy_card()` performs three taps with host-side sleeps:
     dead-zone `0.06 s`, hand `0.16 s`.  If the card is selected but not
     dropped, increase the hand dwell to `0.20 s`.
   * **Dead zone** - the first tap always goes to the neutral dead zone
     `(140, 1585)` (scaled to the guest resolution).  This deselects a card
     that is still lifted from a previous failed attempt.  **Never remove it**:
     without it the next hand-slot tap *deselects* the lifted card and the
     arena tap drops nothing on empty grass.  That bug is what produced
     `Arrows x8` deployments.
   * **Coordinates** - if the arena reticle is systematically off, re-check
     `perception.arena_box` and `perception.hand_xs` against a fresh
     screenshot; the tap target is clamped so a spell can never hit the top
     HUD banners.

4. After any change:

   ```powershell
   python tools/verify_ground_truth.py
   python -m pytest tests/ -q
   ```

### Note on the 4-card cycle

A card must not reappear in hand for the next three plays.  If the HUD shows
the same card in the same slot over and over, the cycle bookkeeping
(`perception.recent_deployed_cards`, `maxlen=3`) is broken - check that
`read_hand()` passes `exclude=set(self.recent_deployed_cards)` to the
classifier and that `update_hand_on_play()` is only called for **verified**
deployments.

---

## 3. `detect_match_end_result()` returns `"UNVERIFIED"`

### Symptom

```
[END SCREEN] Outcome UNVERIFIED - saved .tmp/unverified_endscreen.png
```

The end screen did not show unambiguous evidence.  This is the **correct**
behaviour - the bot refuses to invent a result.  It must be resolved by hand
before the match may be recorded.

### Mandatory procedure (do not skip)

1. **Inspect the saved frame**

   Open `.tmp/unverified_endscreen.png`.

   The detector is **layout independent** - the post-match screen does *not*
   use fixed vertical bands, because the winner's side is drawn at the top
   and the loser's at the bottom and the two halves have different heights:

   | screen | opponent (pink) ribbon | player (blue) ribbon | winner's crowns |
   | :--- | :--- | :--- | :--- |
   | LOSS | `y 0.188 .. 0.334` | `y 0.490 .. 0.627` | `y 0.151 .. 0.231` (opponent) |
   | LOSS (2 crowns) | `y 0.188 .. 0.334` | `y 0.490 .. 0.627` | `y 0.151 .. 0.231` (opponent) |
   | WIN | `y 0.419 .. 0.488` | `y 0.694 .. 0.750` | `y 0.569 .. 0.663` (player) |

   The detector finds the pink and the blue ribbon by their **full-width row
   coverage** and then assigns every large gold blob to the **nearest**
   ribbon.  The blue `OK` button covers only ~22 % of the width and is
   therefore never mistaken for the player ribbon.

   If the frame is *not* the post-match screen at all (a chest popup, the
   home screen, the battle log), the end screen was captured too late or too
   early.  Fix the capture timing in `main()`
   (`detect_post_match_buttons` / `battle_ended_by_timeout`).

2. **Open the in-game Battle Log - this is the mandatory proof step**

   1. On the device, go to the **Battle Log** (main menu -> *Battle Log* ->
      *Battles*).  The log lists every recent match with the word
      `VICTORY` / `DEFEAT`, the crown score and the trophy counter.
   2. Find the row for the match that just finished (match opponent name and
      the crown score).
   3. **Capture it** and save it as `.tmp/battle_log_proof.png`:

      ```powershell
      adb -s 127.0.0.1:16384 exec-out screencap -p > .tmp/battle_log_proof.png
      ```

      (or `adb shell screencap /sdcard/bl.png` + `adb pull`).
   4. **Verify the ground-truth banner** in the captured image:
      * the left-hand word is `VICTORY` (blue) or `DEFEAT` (red),
      * the crown score `X - Y` is legible (blue crown = us, pink crown = the
        opponent),
      * the row you read belongs to the match you just played (check the
        opponent name and the relative time, e.g. *8min ago*).

   4. **Read the absolute trophy counter** on the Clash Royale home screen
      (the player badge in the top-left).  The *difference* between the
      counter before and after the match is the **measured** trophy delta and
      it overrules the model in `trophy_delta_for()`.  Capture it:

      ```powershell
      python -c "import subprocess; open('.tmp/post_match_home.png','wb').write(subprocess.run([r'C:\Program Files\Netease\MuMuPlayer\nx_main\adb.exe','-s','emulator-5554','exec-out','screencap','-p'],capture_output=True).stdout)"
      ```

3. **Adjudicate the match**

   | Battle Log says | Record |
   | :--- | :--- |
   | `VICTORY X - Y` | `result = "WIN"`, `score = "X - Y"`, `delta = +30` |
   | `DEFEAT X - Y` | `result = "LOSS"`, `score = "X - Y"`, `delta = -2` |
   | the row is not there | the match is **not** recorded at all |

   Write the adjudication into `trophy_climb_history.json` together with the
   evidence path:

   ```json
   {
     "match": 23,
     "opponent": "Natsu",
     "result": "WIN",
     "score": "3 - 1",
     "delta": 30,
     "trophies": 214,
     "cards_deployed": 17,
     "verified": true,
     "evidence": ".tmp/battle_log_proof.png",
     "notes": "Adjudicated from the in-game Battle Log.",
     "timestamp": "2026-11-15 15:02:11"
   }
   ```

   * `verified` must be `true` only when a Battle Log screenshot exists.
   * `cards_deployed` may only contain **verified** deployments (see section 5).
   * Never write a result that was not read off a screen.

4. **Never** "fix" an `UNVERIFIED` result by defaulting to `WIN`.  The
   detector is written so that it physically cannot do that.

---

## 4. Pre-match hygiene (prevents most rejections)

Before the autopilot taps *Battle*:

* No modal may be open (chest, level-up, offer, "Trophy Road").
  `deploy_mumu.py` dismisses them, but verify by eye.
* The hand tray must be visible and the four cards must be in the resting
  position (not lifted).
* The elixir bar must be fully visible - a modal that covers it makes
  `read_elixir` return 0 and the bot will never play.

### 4.1 The agent must never get lost in a menu

The Pass Royale reward screen opens **by itself** after a match.  The old
`dismiss_post_match_popups()` tapped the screen centre `(450, 800)` six times
to "collect rewards" - that coordinate is the middle of the arena and it
opened the *card upgrade screen*, from which the agent could not recover.
The account ended up stuck in a menu instead of on the Home screen.

Rules that are now enforced in code:

* only **proven UI elements** are tapped (the modal OK button, the bottom
  navigation "Battle" tab) - never the screen centre;
* the Android **BACK** key leaves any menu that is still open;
* the BACK key is only pressed after the screen has been without a Home
  button for `STUCK_SCREEN_S` (45 s), because BACK during the matchmaking
  would cancel the match;
* `detect_home_battle_button()` only accepts a **button shaped** blob
  (0.26-0.45 of the screen width, 0.045-0.20 of the height, aspect 1.3-3.6,
  fill > 0.55).  A flat orange band - which is what the Pass Royale reward
  screen looks like - is rejected instead of being mistaken for the Battle
  button.

If the agent ever ends up in a menu again, check these three things first:
`detect_home_battle_button()`, the `STUCK_SCREEN_S` recovery branch and
`dismiss_post_match_popups()`.

---

## 5. Deployment accounting rules

`cards_deployed` counts **verified deployments only**.  A deployment is
verified when, 0.55 s after the tap, either

* the elixir dropped by at least 1.0, or
* the template correlation of the tapped hand slot fell below 0.80 (the card
  left the slot).

Anything else is a `[REJECTED TAP]` and must **not** be counted.

Physical sanity limits (use these to sanity-check the history file):

* A 180 s match yields at most about **28** plays (elixir income is
  ~85 elixir, the cheapest cards cost 3).
* With an 8-card deck and a 4-card cycle, any single card can be played at
  most ``ceil(plays / 4)`` times - so `Arrows x8` in 17 plays is
  **physically impossible** and always indicates phantom deployments.

If the history file contains an entry that violates these limits, the entry
is fabricated and must be removed (see section 6).

### Trophy deltas (measured, not modelled)

| Outcome | Delta | Evidence |
| :--- | :--- | :--- |
| WIN | `+30` | Battle Log: 154 -> 184 (3-1 win) and 184 -> 214 (1-0 win) |
| WIN | `+31` | live match, end screen `.tmp/endscreen_win_1crown.png`; device counter 212 -> 243 |
| WIN | `+29` | live overtime match, end screen `.tmp/endscreen_win_2crowns_ot.png`; device counter 243 -> 272 |
| WIN | `+32` | live 3-crown match, end screen `.tmp/endscreen_win_3crowns.png` |
| LOSS, 1 opponent crown | `-9` | end screen `.tmp/diag_live2.png` (ribbon reads `-9`); device counter 223 -> 212 |
| LOSS, 2 opponent crowns | `-11` | end screen `.tmp/endscreen_loss_2crowns.png` (ribbon reads `-11`) |
| LOSS, 3 opponent crowns | `-4` | end screen saved as `.tmp/unverified_endscreen.png` (ribbon reads `-4`); device counter 304 -> 300 |

So **neither bonus nor penalty is a constant**: wins were seen at +29 .. +32 and
losses at -4, -9 and -11 - a 3-crown loss even cost *less* than a 1-crown loss.
The three loss observations do **not** fit a formula, so the model in
`trophy_delta_for()` returns a conservative estimate and every history entry
it produces carries `"delta_source": "model"`.

**The absolute trophy counter on the Home screen is the only authoritative
delta.**  Read it before and after every match; the difference overrules the
model.  **If a new end screen shows a different value, that observation wins
- update the history entry, do not fudge the code.**

### The HUD is not always there

Clash Royale removes the HUD during a match:

* tapping in the arena without a card selected replaces the **whole elixir
  bar** with a red `No card selected` hint,
* a tower destruction / overtime camera pull-back hides the **card tray as
  well**.

A live capture showed the bar hidden for more than six seconds, which made
the old agent declare the match over while the arena was still running (the
match actually continued for another 46 seconds and was won 3-0).

`is_in_battle()` accepts **two** independent markers:

1. the elixir bar has >= 30 purple columns,
2. the **hand tray** shows at least three card frames.

The arena floor was tried as a third marker and **removed**: the Pass Royale
"Bone Pit" screen shows a big sand arena graphic and was mistaken for a live
battle, which made the agent tap "Battle" on a menu.

The end-of-match timeout is `MATCH_OVER_TIMEOUT_S` (120 s), not 6 s, and the
agent refuses to tap anything while the elixir bar is invisible - otherwise
the stale elixir value makes the game reject the play with
"Not enough Elixir!" / "No card selected".

### The celebration frame is not the result

When a tower is destroyed (especially the third crown) Clash Royale first
shows a short celebration frame with the HUD hidden, and only afterwards the
real post-match summary with the pink/blue ribbons.  Reading the celebration
frame produced `UNVERIFIED` even for clear wins.

The match-end branch now polls for up to `END_SCREEN_WAIT_TICKS` (30 × 0.5 s)
until `detect_match_end_result()` finds the ribbons, and only then writes the
result.  A final live match was detected **WIN 3-0** this way and the
post-match screen independently showed "+30" and the device trophy counter
went 330 -> 360.

### Deployment verification thresholds

A deployment is verified when the elixir dropped by `>= 1.0` **or** the
template correlation of the tapped hand slot fell by `>= 0.15`.

The absolute correlation threshold of `0.80` is **unreachable** with the
shipped templates: they were cut from the deck screen while the live hand
tray shows the same art at a different scale.  Measured on a real in-battle
frame the resting correlation of a *correctly identified* card is
**0.37 - 0.60**.  The relative drop is therefore the mechanism that actually
works, and the absolute rule is only a secondary fallback.

---

## 5b. Live board understanding and card identity

The bridge now keeps a **stateful board model** (`env/board_tracker.py`)
instead of treating every frame independently:

* enemy detections are matched to persistent tracks using velocity prediction
  (two troops one tile apart must not swap identities);
* each track carries a smoothed velocity, a 3-second destination, and a
  confidence;
* the tracker answers "which tower is in danger", "is the opponent building
  a push", and "where should I defend";
* a detected push tightens troop placement to the threatened lane before the
  model chooses a tile.

`data/logic.json` (the shipped game data, 145 playable cards) is converted to
a counter table by `utils/card_intel.py`.  Every score comes from real stats
(cost, HP, DPS, range, splash, air/ground, summoned count) and carries an
explanation.  When a card is completely useless against the threat (e.g. a
ground-only troop against a lone Balloon) it is masked out.  With a strong
counter in hand, clearly bad trades are masked as well — at least one card and
Wait always stay legal.

**Enemy card identity is learned from the game itself**: every verified
deployment teaches `env/unit_identity.py` what that card looks like on the
battlefield.  Templates persist in `data/unit_templates/`.  Identification
is only accepted above 0.72 correlation and its confidence is recorded in the
telemetry; unknown units remain `card_id = 5` (the honest fallback from the
original perception layer).

### Post-match learning

`training/live_learner.py` runs after every WIN/LOSS:

* only decisions with `weight = 1` (verified deployments) are targets;
* `UNVERIFIED` matches produce **no gradient**;
* at least 12 verified decisions are required, learning rate 1e-5, one epoch;
* the update is rejected if validation loss regresses > 5% or the policy
  drifts more than 0.02 KL from the frozen reference;
* the old checkpoint is backed up to `*.pre_live.bak` before every write.

Every attempt is written to `checkpoints/live_learning_log.jsonl`.
Use `--no-learn` to disable.  The checkpoint keeps all trainer-compatible
keys, so training can be resumed normally afterwards.

---

## 5c. Elite upgrades

Offline checks for the new subsystems live in
`tests/test_opponent_and_full_hp.py` and `tests/test_pro_geometry.py`:

* opponent Elixir regeneration (1x/2x/3x), exact card cost subtraction,
  idle resync to 10, 4-play cycle lockout and the punish window;
* background subtraction finds a full-HP unit against a static plate and
  excludes active unit boxes from background updates;
* kinematic typing separates air swarms from ground tanks;
* kite tiles cross the centre, King activation coordinates are exact,
  spell leads track a moving unit, and lethal spell cycles require a low
  tower in the final 30 s / overtime.

The observation shape changed to 7 scalars and 9 entity features. Existing
checkpoints are padded with zero-weight input columns by
`adapt_legacy_state_dict()`, so upgrading is behaviour-preserving until the
new columns are trained. Always verify `python tools/diagnose_truth.py` after
a checkpoint change.

---

## 6. History file hygiene

`trophy_climb_history.json` is a log, not a source of truth.

* Only entries with a captured Battle Log screenshot may carry
  `result: "WIN"` / `"LOSS"` and a `delta`.
* Entries that cannot be proven must use `"UNVERIFIED"` with `delta: 0` and
  `verified: false`.
* The `trophies` column must be internally consistent
  (`trophies[n] == trophies[n-1] + delta[n]`).
* Fabricated entries (impossible `cards_deployed`, impossible crown scores,
  inconsistent trophy chains) must be **removed**, not edited.

---

## 7. End-of-run checklist

```powershell
python tools/verify_ground_truth.py     # ALL CHECKS PASSED
python tools/diagnose_truth.py           # no FAIL in the tactical diagnostics
python -m pytest tests/ -q              # all green
```

Then verify manually:

* [ ] every new history entry has `verified: true` **and** an
      `.tmp/battle_log_proof.png` (or an explicit `UNVERIFIED`)
* [ ] the trophy column adds up
* [ ] `cards_deployed` is within the physical limits of section 5
* [ ] no `[REJECTED TAP]` spam (more than two in a match is a bug, see
      section 2)
* [ ] the end-screen detector never printed `UNVERIFIED` without a
      Battle Log adjudication
* [ ] the match video exists in `match_recordings/`

Report the exact console output of the three commands above.  Do not
summarise them.
