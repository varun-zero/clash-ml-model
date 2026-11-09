#!/usr/bin/env python3
"""Interactive Web UI for the Headless Clash Royale Simulator (crsim).

Serves a single-page HTML5 canvas client and REST API for real-time human play,
bot vs bot observation, pause/step, speed control, and card deployment.
Supports playing live against the trained AlphaStar RL model (checkpoints/latest.pt).

Usage:
    python tools/ui.py [--port 8080] [--host 127.0.0.1]
"""

import argparse
import datetime
import json
import os
import secrets
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure crsim and cr_alphastar_agent are importable
CRSIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CRSIM_ROOT not in sys.path:
    sys.path.insert(0, CRSIM_ROOT)

from crsim.engine import Battle, load_model, ARENA_W, ARENA_H, RIVER_Y, RIVER_HALF, BRIDGES_X
from cr_alphastar_agent.env.crsim_replica_env import CrSimReplicaEnv, CRSIM_NAME_TO_ID, ID_TO_CRSIM_NAME
from cr_alphastar_agent.env.scripted_bot import TacticalProBot
from tools.decks import (
    ALL_COMPETITIVE_CARDS,
    DECK_PRESETS as POPULAR_DECKS,
    DECK_PRESET_NAMES,
    generate_random_deck,
    resolve_deck,
)

try:
    import torch
    from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MODEL = load_model()
DEFAULT_BLUE_DECK = POPULAR_DECKS["PEKKA Bridge Spam · sim roster"]
DEFAULT_RED_DECK = POPULAR_DECKS["Classic Log Bait"]


class GameSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.blue_deck_name = "PEKKA Bridge Spam · sim roster"
        self.red_deck_name = "Classic Log Bait"
        self.blue_deck = list(DEFAULT_BLUE_DECK)
        self.red_deck = list(DEFAULT_RED_DECK)
        self.env_adapter = None
        self.battle = None
        self.paused = False
        self.speed = 1.0
        self.blue_ai_mode = "alphastar"  # "alphastar", "heuristic", "human"
        self.red_ai_mode = "heuristic"   # "alphastar", "heuristic", "off"
        self.ai_mode = "heuristic"       # alias for red_ai_mode
        self.running = True
        self.alphastar_net = None
        self._last_ckpt_mtime = 0
        self.latest_ckpt_step = 0
        self.latest_ckpt_update = 0

        # Match Replay & Event Logging
        self.match_id = f"match_{int(time.time())}"
        self.replay_events = []          # Live events in current match
        self.last_completed_replay = None # Last completed match replay
        self.playback_mode = False       # True if actively playing back a replay
        self.playback_events = []        # Events to execute
        self.playback_cursor = 0         # Next event to execute
        self.current_replay_data = None  # Full replay dictionary for seeking
        self.saved_ai_modes = None       # To restore after playback
        self.value_history = [0.0]       # Rolling 120-step value critic history
        self.blue_elixir_history = [5.0] # Rolling 120-step blue elixir history
        self.red_elixir_history = [5.0]  # Rolling 120-step red elixir history

        # Try to load AlphaStar agent
        self._init_alphastar()
        self.reset()

    def _init_alphastar(self):
        if not HAS_TORCH:
            self.blue_ai_mode = "heuristic"
            self.red_ai_mode = "heuristic"
            self.ai_mode = "heuristic"
            return
        ckpt_path = os.path.join(CRSIM_ROOT, "checkpoints", "latest.pt")
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                net = AlphaStarCRNet()
                net.load_state_dict(ckpt["model_state_dict"])
                net.eval()
                self.alphastar_net = net
                self.blue_ai_mode = "alphastar"
                self.red_ai_mode = "heuristic"
                self.ai_mode = "heuristic"
                self._last_ckpt_mtime = os.path.getmtime(ckpt_path)
                self.latest_ckpt_step = ckpt.get('global_step', 0)
                self.latest_ckpt_update = ckpt.get('update_count', 0)
                print(f"[UI] Loaded AlphaStar policy (Step {self.latest_ckpt_step}, Update {self.latest_ckpt_update})")
            except Exception as e:
                print(f"[UI] Warning: Could not load AlphaStar policy: {e}")
                self.blue_ai_mode = "heuristic"
                self.red_ai_mode = "heuristic"
                self.ai_mode = "heuristic"
        else:
            self.blue_ai_mode = "heuristic"
            self.red_ai_mode = "heuristic"
            self.ai_mode = "heuristic"

    def _reload_alphastar_if_changed(self):
        if not HAS_TORCH:
            return
        ckpt_path = os.path.join(CRSIM_ROOT, "checkpoints", "latest.pt")
        if not os.path.exists(ckpt_path):
            return
        try:
            mtime = os.path.getmtime(ckpt_path)
            if mtime > getattr(self, "_last_ckpt_mtime", 0):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                net = AlphaStarCRNet()
                net.load_state_dict(ckpt["model_state_dict"])
                net.eval()
                self.alphastar_net = net
                self._last_ckpt_mtime = mtime
                self.latest_ckpt_step = ckpt.get("global_step", 0)
                self.latest_ckpt_update = ckpt.get("update_count", 0)
                print(f"[UI] Hot-reloaded AlphaStar policy (Step {self.latest_ckpt_step}, Update {self.latest_ckpt_update})")
        except Exception:
            pass

    def reset(self, blue_deck_name=None, red_deck_name=None, *, deck_overrides=None):
        with self.lock:
            # Check for newer checkpoints and hot-reload policy weights
            self._reload_alphastar_if_changed()

            # Archive previous match events if present
            if getattr(self, "battle", None) is not None and len(getattr(self, "replay_events", [])) > 0 and not getattr(self, "playback_mode", False):
                self.last_completed_replay = {
                    "id": getattr(self, "match_id", f"match_{int(time.time())}"),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "blue_deck_name": self.blue_deck_name,
                    "red_deck_name": self.red_deck_name,
                    "blue_deck": list(self.blue_deck),
                    "red_deck": list(self.red_deck),
                    "match_seed": getattr(self, "match_seed", None),
                    "winner": self.battle.result if self.battle.done else "Incomplete",
                    "crowns": list(self.battle.crowns),
                    "duration_ms": self.battle.t,
                    "events": list(self.replay_events)
                }

            # Resolve named presets. "Custom Deck" retains its already
            # validated cards across manual resets and match auto-loops.
            if deck_overrides is not None:
                if 0 in deck_overrides:
                    self.blue_deck_name = "Custom Deck"
                    self.blue_deck = list(deck_overrides[0])
                if 1 in deck_overrides:
                    self.red_deck_name = "Custom Deck"
                    self.red_deck = list(deck_overrides[1])
            elif blue_deck_name and blue_deck_name != "Custom Deck":
                self.blue_deck_name, self.blue_deck = resolve_deck(blue_deck_name)
            elif not blue_deck_name and getattr(self, "blue_deck_name", "") == "Randomized Deck":
                self.blue_deck = generate_random_deck()
            elif not hasattr(self, "blue_deck") or not self.blue_deck:
                self.blue_deck_name, self.blue_deck = resolve_deck("PEKKA Bridge Spam · sim roster")

            # Resolve Red Deck
            if deck_overrides is None and red_deck_name and red_deck_name != "Custom Deck":
                self.red_deck_name, self.red_deck = resolve_deck(red_deck_name)
            elif deck_overrides is None and not red_deck_name and getattr(self, "red_deck_name", "") == "Randomized Deck":
                self.red_deck = generate_random_deck()
            elif deck_overrides is None and (not hasattr(self, "red_deck") or not self.red_deck):
                self.red_deck_name, self.red_deck = resolve_deck("Classic Log Bait")

            print(f"[UI] Match reset: Blue '{self.blue_deck_name}' ({self.blue_deck[:3]}...) vs Red '{self.red_deck_name}' ({self.red_deck[:3]}...)")

            self.match_seed = secrets.randbits(63)
            self.env_adapter = CrSimReplicaEnv(
                blue_deck=self.blue_deck,
                red_deck=self.red_deck,
                seed=self.match_seed,
                model=MODEL,
            )
            self.battle = self.env_adapter.battle
            self.tactical_blue_bot = TacticalProBot(player_id=1)
            self.tactical_red_bot = TacticalProBot(player_id=2)
            self.match_id = f"match_{int(time.time())}"
            self.replay_events = []
            self.value_history = [0.0]
            self.blue_elixir_history = [float(self.battle.elixir[0])]
            self.red_elixir_history = [float(self.battle.elixir[1])]
            self.match_ended_time = 0
            self._tick_counter = 0
            if not getattr(self, "playback_mode", False):
                self.playback_events = []
                self.playback_cursor = 0

    def _record_deploy(self, team: int, hand_idx: int, card_name: str, x: float, y: float):
        if not getattr(self, "playback_mode", False) and getattr(self, "battle", None) is not None:
            event = {
                "t_ms": self.battle.t,
                "team": team,
                "card": card_name,
                "hand_idx": hand_idx,
                "x": round(float(x), 2),
                "y": round(float(y), 2)
            }
            self.replay_events.append(event)

    def get_all_playable_cards(self):
        if not hasattr(self, "_cached_cards"):
            cards = []
            default_deck = ['Knight', 'Fireball', 'Arrows', 'Zap', 'Giant', 'Musketeer', 'Skeletons', 'Minions']
            b = Battle([default_deck, default_deck], model=MODEL)
            for s in sorted(list(MODEL["spells"].keys())):
                cm = b._card_meta(s)
                if cm and b.card_cost(s) is not None:
                    cards.append({
                        "name": s,
                        "cost": b.card_cost(s),
                        "is_evo": "_EV1" in s or "Evo" in s
                    })
            self._cached_cards = cards
        return self._cached_cards

    def set_custom_deck(self, team: int, card_names: list[str]) -> tuple[bool, str]:
        with self.lock:
            if team not in (0, 1):
                return False, "Team must be 0 (blue) or 1 (red)"
            if not isinstance(card_names, list):
                return False, "Deck must be a list of card names"
            if len(card_names) != 8:
                return False, "Deck must contain exactly 8 cards"
            if any(not isinstance(name, str) for name in card_names):
                return False, "Deck entries must be card names"
            if len(set(card_names)) != 8:
                return False, "Deck must contain 8 unique cards"
            default_deck = ['Knight', 'Fireball', 'Arrows', 'Zap', 'Giant', 'Musketeer', 'Skeletons', 'Minions']
            b = Battle([default_deck, default_deck], model=MODEL)
            for c in card_names:
                if not b._card_meta(c):
                    return False, f"Unknown card: {c}"
            self.reset(deck_overrides={team: list(card_names)})
            return True, "Custom deck equipped successfully"

    def step(self):
        with self.lock:
            b = self.battle
            if b is None:
                return

            if b.done:
                if not getattr(self, "playback_mode", False):
                    if not getattr(self, "match_ended_time", 0):
                        self.match_ended_time = time.time()
                    elif time.time() - self.match_ended_time > 4.0:
                        # Auto-loop into next match so the UI stays continuously alive
                        self.match_ended_time = 0
                        self.reset(self.blue_deck_name, self.red_deck_name)
                return

            if getattr(self, "playback_mode", False):
                # Replay playback mode: execute events scheduled at or before b.t
                while self.playback_cursor < len(self.playback_events):
                    ev = self.playback_events[self.playback_cursor]
                    if ev.get("t_ms", 0) <= b.t:
                        team = ev.get("team", 0)
                        card = ev.get("card", "")
                        x = float(ev.get("x", 0.0))
                        y = float(ev.get("y", 0.0))
                        hand = b.hand[team]
                        slot = None
                        if card in hand:
                            slot = hand.index(card)
                        if slot is not None and b.legal_action(team, slot, x, y):
                            b.play(team, slot, x, y)
                        self.playback_cursor += 1
                    else:
                        break
            else:
                self._tick_counter += 1
                # AI decision cadence: 2.5Hz (every 4 physics ticks = 400ms) matching training action repeat
                if self._tick_counter % 4 == 0:
                    # Blue AI Logic (if enabled)
                    if self.blue_ai_mode != "human" and b.play_cd[0] <= 0:
                        self._run_blue_ai()

                    # Red AI Logic (if enabled)
                    if self.red_ai_mode != "off" and b.play_cd[1] <= 0:
                        self._run_red_ai()

            # Advance 1 tick (100ms)
            b.step([])

    def _run_blue_ai(self):
        b = self.battle
        if b is None or b.done:
            return

        # 1. AlphaStar Neural Network AI
        if self.blue_ai_mode == "alphastar" and self.alphastar_net is not None:
            try:
                obs1 = self.env_adapter.get_obs(1)
                batch_obs = {
                    k: torch.from_numpy(v).unsqueeze(0) for k, v in obs1.items()
                }
                with torch.no_grad():
                    actions, _, _, _ = self.alphastar_net.get_action_and_value(batch_obs)
                slot = actions[0, 0].item()
                tile = actions[0, 1].item()
                # Slot 4 is the explicit "Wait / Hold Elixir" action from the policy head
                if slot == 4:
                    return

                if 0 <= slot < len(b.hand[0]):
                    # Player 1 is canonical bottom-up (tx, ty)
                    tx = float(tile % 18)
                    ty = float(tile // 18)
                    if b.legal_action(0, slot, tx, ty):
                        card = b.hand[0][slot]
                        if b.play(0, slot, tx, ty):
                            self._record_deploy(0, slot, card, tx, ty)
                return
            except Exception as e:
                return

        # 2. Tactical Pro Bot (when blue_ai_mode is heuristic)
        if self.blue_ai_mode == "heuristic" and getattr(self, "tactical_blue_bot", None) is not None:
            try:
                obs1 = self.env_adapter.get_obs(1)
                action = self.tactical_blue_bot.act(obs1)
                slot = int(action[0])
                tile = int(action[1])
                if slot == 4:
                    return
                if 0 <= slot < len(b.hand[0]):
                    tx = float(tile % 18)
                    ty = float(tile // 18)
                    if b.legal_action(0, slot, tx, ty):
                        card = b.hand[0][slot]
                        if b.play(0, slot, tx, ty):
                            self._record_deploy(0, slot, card, tx, ty)
            except Exception:
                pass

    def _run_red_ai(self):
        b = self.battle
        if b is None or b.done:
            return

        # 1. AlphaStar Neural Network AI
        if self.red_ai_mode == "alphastar" and self.alphastar_net is not None:
            try:
                obs2 = self.env_adapter.get_obs(2)
                batch_obs = {
                    k: torch.from_numpy(v).unsqueeze(0) for k, v in obs2.items()
                }
                with torch.no_grad():
                    actions, _, _, _ = self.alphastar_net.get_action_and_value(batch_obs)
                slot = actions[0, 0].item()
                tile = actions[0, 1].item()
                if slot == 4:
                    return

                if 0 <= slot < len(b.hand[1]):
                    # Canonical Y-mirror back to engine coordinates
                    tx = float(tile % 18)
                    ty = 31.0 - float(tile // 18)
                    if b.legal_action(1, slot, tx, ty):
                        card = b.hand[1][slot]
                        if b.play(1, slot, tx, ty):
                            self._record_deploy(1, slot, card, tx, ty)
                return
            except Exception as e:
                return

        # 2. Tactical Pro Bot for Red (heuristic / scripted / pro)
        if self.red_ai_mode in ("heuristic", "scripted", "pro") and getattr(self, "tactical_red_bot", None) is not None:
            try:
                obs2 = self.env_adapter.get_obs(2)
                action = self.tactical_red_bot.act(obs2)
                slot = int(action[0])
                tile = int(action[1])
                if slot == 4:
                    return
                if 0 <= slot < len(b.hand[1]):
                    tx = float(tile % 18)
                    ty = 31.0 - float(tile // 18)
                    if b.legal_action(1, slot, tx, ty):
                        card = b.hand[1][slot]
                        if b.play(1, slot, tx, ty):
                            self._record_deploy(1, slot, card, tx, ty)
            except Exception:
                pass

    def play_blue(self, hand_idx: int, x: float, y: float) -> tuple[bool, str]:
        with self.lock:
            b = self.battle
            if b is None:
                return False, "Battle not initialized"
            if b.done:
                return False, "Battle already finished"
            if not (0 <= hand_idx < len(b.hand[0])):
                return False, "Invalid hand slot"

            card = b.hand[0][hand_idx]
            cost = b.card_cost(card)
            if b.elixir[0] < cost:
                return False, f"Need {cost} elixir (have {b.elixir[0]:.1f})"
            if b.play_cd[0] > 0:
                return False, f"Card deployment on cooldown ({b.play_cd[0]}ms)"
            if not b.legal_action(0, hand_idx, x, y):
                return False, "Illegal deployment zone"

            success = b.play(0, hand_idx, x, y)
            if success:
                self._record_deploy(0, hand_idx, card, x, y)
            return success, "Deployed" if success else "Failed to deploy"

    def get_current_replay(self) -> dict:
        with self.lock:
            b = self.battle
            events = list(getattr(self, "replay_events", []))
            if not events and getattr(self, "last_completed_replay", None):
                return self.last_completed_replay
            return {
                "id": getattr(self, "match_id", f"match_{int(time.time())}"),
                "timestamp": datetime.datetime.now().isoformat(),
                "blue_deck_name": self.blue_deck_name,
                "red_deck_name": self.red_deck_name,
                "blue_deck": list(self.blue_deck),
                "red_deck": list(self.red_deck),
                "match_seed": getattr(self, "match_seed", None),
                "winner": b.result if b and b.done else "In Progress",
                "crowns": list(b.crowns) if b else [0, 0],
                "duration_ms": b.t if b else 0,
                "events": events
            }

    def save_replay_to_disk(self, filename: str = None) -> tuple[bool, str]:
        with self.lock:
            replay_data = self.get_current_replay()
            if not replay_data.get("events"):
                return False, "No events in current replay to save"
            replays_dir = os.path.join(CRSIM_ROOT, "replays")
            os.makedirs(replays_dir, exist_ok=True)
            if not filename:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                b_tag = replay_data['blue_deck_name'].replace(' ', '_')
                r_tag = replay_data['red_deck_name'].replace(' ', '_')
                filename = f"replay_{ts}_{b_tag}_vs_{r_tag}.json"
            if not filename.endswith(".json"):
                filename += ".json"
            clean_fn = "".join(c for c in filename if c.isalnum() or c in "._- ")
            filepath = os.path.join(replays_dir, clean_fn)
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(replay_data, f, indent=2)
                return True, clean_fn
            except Exception as e:
                return False, str(e)

    def list_replays(self) -> list[dict]:
        replays_dir = os.path.join(CRSIM_ROOT, "replays")
        if not os.path.exists(replays_dir):
            return []
        items = []
        for fn in sorted(os.listdir(replays_dir), reverse=True):
            if fn.endswith(".json"):
                fp = os.path.join(replays_dir, fn)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    items.append({
                        "filename": fn,
                        "timestamp": data.get("timestamp", ""),
                        "blue_deck_name": data.get("blue_deck_name", ""),
                        "red_deck_name": data.get("red_deck_name", ""),
                        "winner": data.get("winner", ""),
                        "crowns": data.get("crowns", [0, 0]),
                        "event_count": len(data.get("events", [])),
                        "duration_sec": round(data.get("duration_ms", 0) / 1000, 1)
                    })
                except Exception:
                    pass
        return items

    def load_replay(self, replay_data: dict) -> tuple[bool, str]:
        with self.lock:
            events = replay_data.get("events", [])
            if not events:
                return False, "Replay has no events"
            b_deck = replay_data.get("blue_deck", DEFAULT_BLUE_DECK)
            r_deck = replay_data.get("red_deck", DEFAULT_RED_DECK)
            b_name = replay_data.get("blue_deck_name", "Replay Blue")
            r_name = replay_data.get("red_deck_name", "Replay Red")

            if not getattr(self, "playback_mode", False):
                self.saved_ai_modes = (self.blue_ai_mode, self.red_ai_mode)

            self.playback_mode = True
            self.playback_events = sorted(events, key=lambda e: e.get("t_ms", 0))
            self.playback_cursor = 0
            self.current_replay_data = dict(replay_data)
            self.blue_ai_mode = "playback"
            self.red_ai_mode = "playback"

            self.blue_deck = list(b_deck)
            self.red_deck = list(r_deck)
            self.blue_deck_name = b_name
            self.red_deck_name = r_name

            # Replay decks retain their actual simulator card names. Converting
            # through representative policy IDs can collapse distinct cards.
            replay_seed = replay_data.get("match_seed")
            self.match_seed = int(replay_seed) if isinstance(replay_seed, int) else None
            self.env_adapter = CrSimReplicaEnv(
                blue_deck=self.blue_deck,
                red_deck=self.red_deck,
                seed=self.match_seed,
                model=MODEL,
            )
            self.battle = self.env_adapter.battle
            self.paused = False
            return True, "Replay playback loaded"

    def seek_replay(self, target_t_ms: int) -> tuple[bool, str]:
        with self.lock:
            if not getattr(self, "playback_mode", False) or not getattr(self, "current_replay_data", None):
                return False, "Not in replay playback mode"

            target_t_ms = max(0, int(target_t_ms))
            b = self.battle

            # If seeking backward or restart, re-initialize to t=0
            if target_t_ms < b.t:
                b_deck = self.current_replay_data.get("blue_deck", DEFAULT_BLUE_DECK)
                r_deck = self.current_replay_data.get("red_deck", DEFAULT_RED_DECK)
                replay_seed = self.current_replay_data.get("match_seed")
                self.env_adapter = CrSimReplicaEnv(
                    blue_deck=b_deck,
                    red_deck=r_deck,
                    seed=int(replay_seed) if isinstance(replay_seed, int) else None,
                    model=MODEL,
                )
                self.battle = self.env_adapter.battle
                self.playback_cursor = 0
                b = self.battle

            # Fast-forward simulation to target_t_ms
            while b.t < target_t_ms and not b.done:
                while self.playback_cursor < len(self.playback_events):
                    ev = self.playback_events[self.playback_cursor]
                    if ev.get("t_ms", 0) <= b.t:
                        team = int(ev.get("team", 0))
                        card = ev.get("card", "")
                        x = float(ev.get("x", 0.0))
                        y = float(ev.get("y", 0.0))
                        hand = b.hand[team]
                        slot = None
                        if card in hand:
                            slot = hand.index(card)
                        if slot is not None and b.legal_action(team, slot, x, y):
                            b.play(team, slot, x, y)
                        self.playback_cursor += 1
                    else:
                        break
                b.step([])

            return True, f"Seeked to {b.t}ms"

    def exit_replay(self):
        with self.lock:
            self.playback_mode = False
            self.playback_events = []
            self.playback_cursor = 0
            self.current_replay_data = None
            if getattr(self, "saved_ai_modes", None):
                self.blue_ai_mode, self.red_ai_mode = self.saved_ai_modes
                self.ai_mode = self.red_ai_mode
            else:
                self.blue_ai_mode = "alphastar"
                self.red_ai_mode = "heuristic"
                self.ai_mode = "heuristic"
            self.reset(self.blue_deck_name, self.red_deck_name)

    def _get_neural_diagnostics(self):
        if not HAS_TORCH or self.alphastar_net is None or self.env_adapter is None:
            return None
        try:
            b = self.battle
            eval_pid = 2 if self.red_ai_mode == "alphastar" and self.blue_ai_mode != "alphastar" else 1
            obs = self.env_adapter.get_obs(eval_pid)
            batch = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                core = self.alphastar_net.forward_core(
                    batch["scalars"], batch["hand_cards"], batch["entities"],
                    batch["entity_mask"], batch["arena_grid"]
                )
                
                # 1. Bipolar Value Critic V(s) with Terminal Outcome Anchoring
                if b is not None and b.done:
                    # Terminal state: Blue win = +1.0, Red win = -1.0, Draw = 0.0
                    if b.result == 0:
                        signed_val = 1.0
                    elif b.result == 1:
                        signed_val = -1.0
                    else:
                        signed_val = 0.0
                else:
                    raw_val = float(self.alphastar_net.value_head(core).item())
                    # Blue is Player 1 (+1.0), Red is Player 2 (-1.0)
                    signed_val = -raw_val if eval_pid == 2 else raw_val
                    signed_val = max(-1.0, min(1.0, signed_val))

                if not hasattr(self, "value_history"):
                    self.value_history = []
                self.value_history.append(round(signed_val, 3))
                if len(self.value_history) > 120:
                    self.value_history.pop(0)

                # Track Elixir Economy Series
                if b is not None:
                    if not hasattr(self, "blue_elixir_history"):
                        self.blue_elixir_history = []
                    if not hasattr(self, "red_elixir_history"):
                        self.red_elixir_history = []
                    self.blue_elixir_history.append(round(b.elixir[0], 1))
                    self.red_elixir_history.append(round(b.elixir[1], 1))
                    if len(self.blue_elixir_history) > 120:
                        self.blue_elixir_history.pop(0)
                    if len(self.red_elixir_history) > 120:
                        self.red_elixir_history.pop(0)

                # 2. Autoregressive Card Policy Logits
                card_logits = self.alphastar_net.card_head(core).float()
                card_mask = batch["card_mask"].bool()
                masked = card_logits.masked_fill(~card_mask, -1e9)
                card_probs = torch.softmax(masked, dim=-1).squeeze(0)

                best_slot = card_probs.argmax().item()
                best_prob = float(card_probs[best_slot].item())
                is_waiting = (best_slot == 4)
                top_tiles = []
                top_vals = []
                if not is_waiting:
                    eval_tensor = torch.tensor([best_slot], dtype=torch.long)

                    # 3. Spatial Tile Policy Logits. No placement is inferred for Wait.
                    chosen_emb = self.alphastar_net.chosen_action_emb(eval_tensor)
                    tile_input = torch.cat([core, chosen_emb], dim=-1)
                    tile_logits = self.alphastar_net.tile_head(tile_input).float()

                    if "card_tile_mask" in batch:
                        slots = eval_tensor.view(1, 1, 1).expand(-1, 1, batch["tile_mask"].shape[-1])
                        eff_mask = torch.gather(batch["card_tile_mask"].bool(), 1, slots).squeeze(1)
                    else:
                        clamped_slot = eval_tensor.clamp(0, 3).unsqueeze(-1)
                        chosen_id = torch.gather(batch["hand_cards"][:, :4], 1, clamped_slot).squeeze(-1)
                        is_spell = torch.isin(chosen_id, self.alphastar_net.spell_ids)
                        full_mask = torch.ones_like(batch["tile_mask"], dtype=torch.bool)
                        eff_mask = torch.where(is_spell.unsqueeze(-1), full_mask, batch["tile_mask"].bool())
                    masked_tile = tile_logits.masked_fill(~eff_mask, -1e9)
                    tile_probs = torch.softmax(masked_tile, dim=-1).squeeze(0)

                    top_vals, top_idx = torch.topk(tile_probs, 8)
                    for v, t in zip(top_vals, top_idx):
                        cx = int(t.item() % 18)
                        cy = int(t.item() // 18)
                        ey = (31 - cy) if eval_pid == 2 else cy
                        top_tiles.append({"x": cx, "y": ey, "p": round(float(v.item()), 3)})

                # Resolve card name & tactical intent
                target_card_name = "Wait (Hold Elixir)"
                if best_slot < 4 and b is not None and (eval_pid - 1) < len(b.hand) and best_slot < len(b.hand[eval_pid - 1]):
                    target_card_name = b.hand[eval_pid - 1][best_slot]

                elixir_val = round(float(batch["scalars"][0, 0].item() * 10.0), 1)
                entity_count = int((~batch["entity_mask"]).sum().item())
                core_energy = round(float(core.norm().item()), 2)

                # Gate Intent on Slot 4 (Wait)
                if is_waiting:
                    action_type = "wait"
                    if elixir_val < 5.0:
                        intent_summary = f"Saving elixir · {elixir_val} / 10"
                    else:
                        intent_summary = f"Waiting · {elixir_val} elixir"
                else:
                    if top_tiles:
                        tx, ty = top_tiles[0]["x"], top_tiles[0]["y"]
                        is_bridge = ty in (13, 14, 15, 16, 17, 18)
                        is_defense = (eval_pid == 1 and ty < 13) or (eval_pid == 2 and ty > 18)
                        if is_bridge:
                            action_type = "bridge"
                            intent_summary = f"Bridge pressure · {target_card_name} · ({tx}, {ty})"
                        elif is_defense:
                            action_type = "defense"
                            intent_summary = f"Defend · {target_card_name} · ({tx}, {ty})"
                        else:
                            action_type = "buildup"
                            intent_summary = f"Build lane · {target_card_name} · ({tx}, {ty})"
                    else:
                        action_type = "play"
                        intent_summary = f"Deploy · {target_card_name}"

                # 4. Multi-Layer Activation Histograms (5 layers x 8 mini-bars)
                ent_bars = [round(float(torch.norm(batch["entities"][0, i]).item()) / 2.0, 2) for i in range(min(8, len(batch["entities"][0])))]
                while len(ent_bars) < 8:
                    ent_bars.append(0.12)

                grid_slice = batch["arena_grid"][0].view(6, -1).mean(dim=-1).tolist()
                cnn_bars = [round(float(x), 2) for x in grid_slice]
                while len(cnn_bars) < 8:
                    cnn_bars.append(0.22)

                core_128 = core[0]
                core_bars = [round(float(core_128[i*16:(i+1)*16].norm().item()) / 4.0, 2) for i in range(8)]

                card_bars = [round(float(p.item()), 3) for p in card_probs]
                while len(card_bars) < 8:
                    card_bars.append(0.04)

                tile_bars = [round(float(v.item()), 3) for v in top_vals[:8]]
                while len(tile_bars) < 8:
                    tile_bars.append(0.02)

                layer_strip_data = [
                    {"name": "ENTITY_TF", "dim": "40×128", "bars": ent_bars},
                    {"name": "SPATIAL_CNN", "dim": "6×32×18", "bars": cnn_bars},
                    {"name": "FUSION_CORE", "dim": "128-D", "bars": core_bars},
                    {"name": "HEAD1_CARD", "dim": "5 LOGITS", "bars": card_bars},
                    {"name": "HEAD2_TILE", "dim": "576 TILES", "bars": tile_bars}
                ]

                # 5. 6-Axis Tactical Posture Radar
                friendly_team = eval_pid - 1
                friendly_units = [
                    u for u in b.units
                    if u.team == friendly_team and u.hp > 0
                    and not getattr(u, "is_building", False)
                ] if b else []
                n_friendly = max(1, len(friendly_units))
                aggro_cnt = sum(
                    1 for u in friendly_units
                    if (u.y >= 15.0 if friendly_team == 0 else u.y <= 15.0)
                )
                def_cnt = len(friendly_units) - aggro_cnt
                left_cnt = sum(1 for u in friendly_units if u.x <= 8.5)
                right_cnt = sum(1 for u in friendly_units if u.x > 9.5)
                spell_cnt = sum(
                    1 for c in b.hand[friendly_team]
                    if b._card_meta(c).get("CanDeployOnEnemySide")
                    or not b._is_unit_card(b._card_meta(c))
                ) if b else 1

                tactical_radar = {
                    "aggro": round(min(1.0, max(0.18, aggro_cnt / n_friendly + (0.35 if not is_waiting and best_slot != 4 else 0.0))), 2),
                    "bank": round(min(1.0, max(0.15, elixir_val / 10.0)), 2),
                    "left": round(min(1.0, max(0.2, (left_cnt / n_friendly) if friendly_units else 0.5)), 2),
                    "right": round(min(1.0, max(0.2, (right_cnt / n_friendly) if friendly_units else 0.5)), 2),
                    "spell": round(min(1.0, max(0.2, spell_cnt / 4.0)), 2),
                    "def": round(min(1.0, max(0.2, (def_cnt / n_friendly) if friendly_units else 0.75)), 2)
                }

            return {
                "eval_pid": eval_pid,
                "value": round(signed_val, 2),
                "value_history": list(self.value_history),
                "blue_elixir_history": list(getattr(self, "blue_elixir_history", [])),
                "red_elixir_history": list(getattr(self, "red_elixir_history", [])),
                "probs": [round(float(p.item()), 3) for p in card_probs],
                "top_tiles": top_tiles,
                "target_card_slot": best_slot,
                "target_card_name": target_card_name,
                "team": eval_pid - 1,
                "is_waiting": is_waiting,
                "tactical_intent": {
                    "action_type": action_type,
                    "summary": intent_summary,
                    "confidence": round(best_prob, 2),
                    "best_slot": best_slot
                },
                "layers": layer_strip_data,
                "tactical_radar": tactical_radar,
                "layer_signals": {
                    "entity_count": entity_count,
                    "core_energy": core_energy,
                    "elixir": elixir_val
                }
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if getattr(self, "_last_neural_diagnostics_error", None) != error:
                print(f"[UI] Neural diagnostics unavailable: {error}")
                self._last_neural_diagnostics_error = error
            return None

    def get_state(self) -> dict:
        with self.lock:
            b = self.battle
            if b is None:
                return {"initialized": False}

            # Units
            units_data = []
            for u in b.units:
                if u.hp <= 0:
                    continue
                units_data.append({
                    "id": u.id,
                    "name": u.name,
                    "team": u.team,
                    "x": round(u.x, 2),
                    "y": round(u.y, 2),
                    "hp": u.hp,
                    "maxhp": u.maxhp,
                    "hp_pct": round(u.hp / max(1, u.maxhp), 3),
                    "flying": u.flying,
                    "building": u.is_building or getattr(u, "kind", "") in ("KingTower", "PrincessTower"),
                    "tower": getattr(u, "kind", "") in ("KingTower", "PrincessTower"),
                    "kind": getattr(u, "kind", ""),
                    "is_king": getattr(u, "kind", "") == "KingTower",
                    "target_id": getattr(u, "target_id", -1),
                    "range": round(getattr(u, "range_t", 1.2), 2),
                    "col_t": round(getattr(u, "col_t", 0.3), 2),
                    "activated": getattr(u, "activated", False)
                })

            # Projectiles
            proj_data = []
            for p in b.projectiles:
                proj_data.append({
                    "id": p.id,
                    "name": p.name,
                    "team": p.team,
                    "x": round(p.x, 2),
                    "y": round(p.y, 2),
                    "tx": round(p.tx, 2),
                    "ty": round(p.ty, 2),
                })

            # Area Fields
            fields_data = []
            for f in b.fields:
                fields_data.append({
                    "id": f.id,
                    "name": f.name,
                    "team": f.team,
                    "x": round(f.x, 2),
                    "y": round(f.y, 2),
                    "radius": round(f.radius_t, 2),
                })

            # Hand
            hand_cards = []
            for i, c in enumerate(b.hand[0]):
                cm = b._card_meta(c)
                cost = b.card_cost(c)
                is_unit = b._is_unit_card(cm)
                can_enemy = bool(cm.get("CanDeployOnEnemySide"))
                hand_cards.append({
                    "idx": i,
                    "name": c,
                    "cost": cost,
                    "is_spell": not is_unit or can_enemy,
                    "affordable": b.elixir[0] >= cost and b.play_cd[0] <= 0
                })

            red_hand_cards = []
            for i, c in enumerate(b.hand[1]):
                cm = b._card_meta(c)
                cost = b.card_cost(c)
                is_unit = b._is_unit_card(cm)
                can_enemy = bool(cm.get("CanDeployOnEnemySide"))
                red_hand_cards.append({
                    "idx": i,
                    "name": c,
                    "cost": cost,
                    "is_spell": not is_unit or can_enemy,
                    "affordable": b.elixir[1] >= cost and b.play_cd[1] <= 0
                })

            next_card = b.cycle[0][0] if (b.cycle and len(b.cycle[0]) > 0) else "Log"
            time_left_sec = max(0, int((180_000 - b.t) / 1000))
            is_ot = b.t >= 180_000
            if is_ot:
                time_left_sec = max(0, int((300_000 - b.t) / 1000))

            delta_e = round(float(b.elixir[0] - b.elixir[1]), 1)

            return {
                "initialized": True,
                "t_ms": b.t,
                "time_left_sec": time_left_sec,
                "is_overtime": is_ot,
                "crowns": b.crowns,
                "elixir": [round(b.elixir[0], 1), round(b.elixir[1], 1)],
                "delta_e": delta_e,
                "next_card": next_card,
                "play_cd": [b.play_cd[0], b.play_cd[1]],
                "hand": hand_cards,
                "red_hand": red_hand_cards,
                "units": units_data,
                "projectiles": proj_data,
                "fields": fields_data,
                "paused": self.paused,
                "speed": self.speed,
                "blue_ai_mode": self.blue_ai_mode,
                "red_ai_mode": self.red_ai_mode,
                "ai_mode": self.red_ai_mode,
                "has_alphastar": self.alphastar_net is not None,
                "latest_ckpt_step": getattr(self, "latest_ckpt_step", 0),
                "latest_ckpt_update": getattr(self, "latest_ckpt_update", 0),
                "blue_deck_name": self.blue_deck_name,
                "red_deck_name": self.red_deck_name,
                "blue_deck": list(self.blue_deck),
                "red_deck": list(self.red_deck),
                "match_seed": getattr(self, "match_seed", None),
                "cycle": [list(b.cycle[0]), list(b.cycle[1])] if b else [[], []],
                "popular_decks": DECK_PRESET_NAMES,
                "done": b.done,
                "winner": b.result,
                "arena_w": ARENA_W,
                "arena_h": ARENA_H,
                "neural_hud": self._get_neural_diagnostics(),
                "playback_mode": getattr(self, "playback_mode", False),
                "playback_cursor": getattr(self, "playback_cursor", 0),
                "playback_total": len(getattr(self, "playback_events", [])),
                "playback_duration_ms": (
                    self.current_replay_data.get("duration_ms", 180000)
                    if getattr(self, "playback_mode", False) and getattr(self, "current_replay_data", None)
                    else 0
                ),
                "recent_events": (
                    self.playback_events[max(0, self.playback_cursor-10):self.playback_cursor]
                    if getattr(self, "playback_mode", False)
                    else getattr(self, "replay_events", [])[-12:]
                ),
                "total_events": (
                    len(getattr(self, "playback_events", []))
                    if getattr(self, "playback_mode", False)
                    else len(getattr(self, "replay_events", []))
                ),
            }


SESSION = GameSession()


def simulation_worker():
    """Background simulation thread."""
    while SESSION.running:
        t0 = time.time()
        if not SESSION.paused:
            SESSION.step()
        dt = 0.10 / max(0.1, SESSION.speed)
        elapsed = time.time() - t0
        sleep_time = max(0.002, dt - elapsed)
        time.sleep(sleep_time)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clash Royale Simulator & AlphaStar Agent UI</title>
<style>
  :root {
    --bg-dark: #0f141c;
    --panel-bg: #1a2230;
    --border-color: #2e3a4e;
    --blue-team: #2979ff;
    --red-team: #ff3d00;
    --elixir-color: #d500f9;
    --text-main: #eceff1;
    --text-muted: #90a4ae;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg-dark); color: var(--text-main); display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 12px; }
  header { text-align: center; margin-bottom: 12px; }
  header h1 { font-size: 22px; color: #ffca28; letter-spacing: 0.5px; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
  .subtitle { font-size: 13px; color: var(--text-muted); }

  .main-container { display: flex; gap: 20px; align-items: flex-start; justify-content: center; max-width: 1040px; width: 100%; }
  
  /* Arena Container */
  .arena-wrapper {
    position: relative;
    background: #2b392a;
    border: 4px solid #4a3c28;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    overflow: hidden;
  }
  canvas { display: block; background: #608848; cursor: crosshair; }

  /* Side Panel */
  .side-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
    width: 340px;
    background: var(--panel-bg);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
  }
  .panel-section { display: flex; flex-direction: column; gap: 6px; }
  .panel-title { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); font-weight: bold; }

  /* Status Bar */
  .scoreboard {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #111822;
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: bold;
  }
  .team-blue { color: var(--blue-team); display: flex; align-items: center; gap: 6px; }
  .team-red { color: var(--red-team); display: flex; align-items: center; gap: 6px; }
  .timer-box { font-size: 16px; color: #fff; background: #263238; padding: 4px 10px; border-radius: 6px; }
  .overtime-badge { background: #b71c1c; color: #ffeb3b; font-size: 11px; padding: 2px 6px; border-radius: 4px; font-weight: bold; margin-left: 4px; }

  /* Elixir Bar */
  .elixir-bar-container {
    background: #212121;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #424242;
    position: relative;
  }
  .elixir-fill {
    height: 18px;
    background: linear-gradient(90deg, #aa00ff, #ea80fc);
    border-radius: 5px;
    transition: width 0.1s linear;
  }
  .elixir-text {
    position: absolute;
    width: 100%;
    text-align: center;
    top: 3px;
    font-size: 12px;
    font-weight: bold;
    color: #fff;
    text-shadow: 0 1px 2px #000;
  }

  /* Hand Cards */
  .hand-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }
  .card-slot {
    background: #263238;
    border: 2px solid #37474f;
    border-radius: 8px;
    padding: 8px 4px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    min-height: 80px;
    cursor: pointer;
    transition: all 0.15s ease;
    user-select: none;
    position: relative;
  }
  .card-slot:hover { border-color: #64b5f6; transform: translateY(-2px); }
  .card-slot.selected { border-color: #ffeb3b; background: #37474f; box-shadow: 0 0 10px rgba(255,235,59,0.5); }
  .card-slot.disabled { opacity: 0.45; cursor: not-allowed; }
  .card-name { font-size: 11px; font-weight: 600; text-align: center; line-height: 1.2; word-break: break-word; }
  .card-cost {
    background: var(--elixir-color);
    color: #fff;
    font-size: 11px;
    font-weight: bold;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4);
    margin-top: 4px;
  }

  /* Buttons & Controls */
  .btn-row { display: flex; gap: 6px; }
  button, select {
    background: #37474f;
    color: #fff;
    border: none;
    padding: 7px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
    flex: 1;
  }
  button:hover, select:hover { background: #455a64; }
  button.active { background: #0288d1; }
  button.danger { background: #d32f2f; }
  button.danger:hover { background: #f44336; }

  /* Info & Status */
  .status-toast {
    font-size: 12px;
    padding: 8px;
    border-radius: 6px;
    background: #263238;
    color: #ffb74d;
    min-height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
  }
  .winner-banner {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0,0,0,0.85);
    border: 3px solid #ffca28;
    border-radius: 12px;
    padding: 20px 30px;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8);
    display: none;
    z-index: 100;
  }
  .winner-banner h2 { font-size: 28px; margin-bottom: 8px; }

  /* Replay & Action Feed */
  .event-feed-box {
    background: #111822;
    border-radius: 8px;
    padding: 8px;
    max-height: 120px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 11px;
    font-family: monospace;
    border: 1px solid #263238;
  }
  .event-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2px 4px;
    border-radius: 4px;
    background: #1e293b;
  }
  .event-blue { color: #64b5f6; border-left: 3px solid var(--blue-team); }
  .event-red { color: #ff8a80; border-left: 3px solid var(--red-team); }
  .playback-banner {
    background: #004d40;
    border: 1px solid #00bfa5;
    color: #a7ffeb;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 0 10px rgba(0,191,165,0.3);
  }
</style>
</head>
<body>

<header>
  <h1>⚔️ Clash Royale Simulator & AlphaStar Agent</h1>
  <div class="subtitle">100% Parameter-Driven &bull; Full-Card Fidelity Engine &bull; Live Neural Agent Play</div>
</header>

<div class="main-container">
  <!-- Arena Canvas -->
  <div class="arena-wrapper">
    <canvas id="arenaCanvas" width="360" height="640"></canvas>
    <div id="winnerBanner" class="winner-banner">
      <h2 id="winnerText">Victory!</h2>
      <button onclick="resetGame()" style="margin-top:10px; padding:8px 20px;">Play Again</button>
    </div>
  </div>

  <!-- Side Control Panel -->
  <div class="side-panel">
    <!-- Score & Timer -->
    <div class="scoreboard">
      <div class="team-blue"><span>👑</span> <span id="blueCrowns">0</span></div>
      <div class="timer-box">
        <span id="matchTimer">3:00</span>
        <span id="otBadge" class="overtime-badge" style="display:none;">OT</span>
      </div>
      <div class="team-red"><span id="redCrowns">0</span> <span>👑</span></div>
    </div>

    <!-- Blue Elixir -->
    <div class="panel-section">
      <div class="panel-title">Your Elixir (Blue)</div>
      <div class="elixir-bar-container">
        <div id="elixirFill" class="elixir-fill" style="width: 50%;"></div>
        <div id="elixirText" class="elixir-text">5.0 / 10</div>
      </div>
    </div>

    <!-- Hand Cards -->
    <div class="panel-section">
      <div class="panel-title">Deploy Cards (Click card then click arena)</div>
      <div id="handGrid" class="hand-grid"></div>
    </div>

    <!-- Feedback Message -->
    <div id="statusToast" class="status-toast">Ready for battle. Click a card to deploy!</div>

    <!-- Match Controls -->
    <div class="panel-section">
      <div class="panel-title">Match Controls</div>
      <div class="btn-row">
        <button id="pauseBtn" onclick="togglePause()">Pause</button>
        <button id="stepBtn" onclick="stepOnce()">Step (100ms)</button>
      </div>
      <div class="btn-row" style="margin-top:4px;">
        <button id="speed1" class="active" onclick="setSpeed(1.0)">1x</button>
        <button id="speed2" onclick="setSpeed(2.0)">2x</button>
        <button id="speed4" onclick="setSpeed(4.0)">4x</button>
        <button id="speed10" onclick="setSpeed(10.0)">10x</button>
        <button id="speed20" onclick="setSpeed(20.0)">20x</button>
      </div>
    </div>

    <!-- Blue Player Mode -->
    <div class="panel-section">
      <div class="panel-title">Blue Player (Bottom)</div>
      <select id="blueAiModeSelect" onchange="changeBlueAiMode(this.value)">
        <option value="alphastar">🤖 AlphaStar RL Agent (Auto-Play)</option>
        <option value="heuristic">🧠 Scripted Bot (Auto-Play)</option>
        <option value="human">👤 Human Player (Manual Click to Deploy)</option>
      </select>
    </div>

    <!-- Red Opponent Mode -->
    <div class="panel-section">
      <div class="panel-title">Red Opponent (Top)</div>
      <select id="redAiModeSelect" onchange="changeRedAiMode(this.value)">
        <option value="heuristic">🧠 Scripted Bot (Auto-Play)</option>
        <option value="alphastar">🤖 AlphaStar RL Agent (checkpoints/latest.pt)</option>
        <option value="off">👤 Manual Sandboxing (AI Disabled)</option>
      </select>
    </div>

    <!-- Deck Selection -->
    <div class="panel-section">
      <div class="panel-title">Match Decks</div>
      <div class="btn-row">
        <select id="blueDeckSelect" title="Blue Deck" onchange="resetWithSelectedDecks()"></select>
        <select id="redDeckSelect" title="Red Deck" onchange="resetWithSelectedDecks()"></select>
      </div>
      <button class="danger" style="margin-top:4px;" onclick="resetGame()">New Match</button>
    </div>

    <!-- Custom 8-Card Deck Builder -->
    <details class="panel-section" id="deckBuilderDetails">
      <summary class="panel-title" style="cursor:pointer; color:#64b5f6; user-select:none;">🃏 Custom Deck Builder (Click to Build)</summary>
      <div style="background:#111822; padding:10px; border-radius:8px; display:flex; flex-direction:column; gap:8px; margin-top:4px;">
        <div style="display:flex; justify-content:space-between; font-size:12px;">
          <span>Selected (<span id="customCount" style="color:#64b5f6; font-weight:bold;">0</span>/8):</span>
          <span id="customAvgElixir" style="color:#d500f9; font-weight:bold;">Avg: 0.0</span>
        </div>
        <div id="customDeckSlots" style="display:grid; grid-template-columns:repeat(4, 1fr); gap:4px;"></div>
        <div style="display:flex; gap:4px; margin-top:4px;">
          <input type="text" id="cardSearchInput" placeholder="Filter 133 cards..." style="flex:1; background:#263238; border:1px solid #37474f; color:#fff; padding:5px 8px; border-radius:4px; font-size:11px;" oninput="filterCardCatalog(this.value)">
        </div>
        <div id="cardCatalogGrid" style="max-height:160px; overflow-y:auto; display:grid; grid-template-columns:repeat(3, 1fr); gap:4px; padding-right:2px;"></div>
        <button onclick="applyCustomDeck()" style="background:#00c853; margin-top:4px; font-size:12px;">⚔️ Equip Custom Deck</button>
      </div>
    </details>

    <!-- Opponent Elixir -->
    <div class="panel-section">
      <div class="panel-title">Opponent Elixir (Red)</div>
      <div class="elixir-bar-container">
        <div id="redElixirFill" class="elixir-fill" style="background:#e53935; width: 50%;"></div>
        <div id="redElixirText" class="elixir-text">5.0 / 10</div>
      </div>
    </div>

    <!-- AlphaStar Neural Policy HUD -->
    <div id="neuralHudSection" class="panel-section" style="display:none;">
      <div class="panel-title">🧠 AlphaStar Policy & Value HUD <span id="ckptStepBadge" style="color:#ffca28; float:right; text-transform:none; font-weight:normal; font-size:10px;"></span></div>
      <div style="background:#111822; padding:10px; border-radius:8px; display:flex; flex-direction:column; gap:8px;">
        <div style="display:flex; justify-content:space-between; font-size:12px;">
          <span>Value Advantage:</span>
          <span id="valueEstimate" style="font-weight:bold; color:#4caf50;">+0.00</span>
        </div>
        <div style="height:6px; background:#37474f; border-radius:3px; overflow:hidden; position:relative;">
          <div id="valueMeter" style="height:100%; width:50%; background:#2979ff; transition:width 0.2s;"></div>
        </div>
        <div style="font-size:11px; color:#90a4ae; margin-top:2px;">Live Action Intent:</div>
        <div id="actionProbs" style="display:flex; flex-direction:column; gap:4px; font-size:11px;"></div>
      </div>
    </div>
    <!-- Replay Playback Active Banner -->
    <div id="playbackBanner" class="playback-banner" style="display:none;">
      <span>▶️ REPLAY PLAYBACK (<span id="playbackProg">0/0</span>)</span>
      <button onclick="exitReplay()" style="background:#d32f2f; font-size:10px; padding:3px 8px; flex:none;">Exit Replay</button>
    </div>

    <!-- Match Replay & Action Feed -->
    <details class="panel-section" id="replaySection" open>
      <summary class="panel-title" style="cursor:pointer; color:#00bfa5; user-select:none;">📹 Match Replay & Action Feed (<span id="eventCountBadge">0</span>)</summary>
      <div style="background:#111822; padding:8px; border-radius:8px; display:flex; flex-direction:column; gap:6px; margin-top:4px;">
        <div id="eventFeed" class="event-feed-box">
          <div style="color:#78909c; text-align:center; padding:10px 0;">No deployments yet</div>
        </div>
        <div class="btn-row" style="margin-top:2px;">
          <button onclick="exportReplayJson()" style="background:#00897b; font-size:11px;" title="Export current match to JSON file">📥 Export JSON</button>
          <button onclick="saveReplayToServer()" style="background:#00acc1; font-size:11px;" title="Save current match to server disk">💾 Save Match</button>
        </div>
        <div style="display:flex; gap:4px; margin-top:2px;">
          <select id="savedReplaySelect" style="flex:1; font-size:11px; padding:4px 6px;">
            <option value="">(Select Saved Replay)</option>
          </select>
          <button onclick="loadSelectedReplay()" style="background:#3949ab; flex:none; font-size:11px; padding:4px 8px;">▶️ Watch</button>
        </div>
      </div>
    </details>
  </div>
</div>

<script>
const canvas = document.getElementById('arenaCanvas');
const ctx = canvas.getContext('2d');

const TILE_PX = 20;
const ARENA_W = 18;
const ARENA_H = 32;

let gameState = null;
let selectedCardIdx = null;
let mouseX = -1;
let mouseY = -1;
let decksInitialized = false;
let allCardCatalog = [];
let customSelectedCards = [];

async function loadCardCatalog() {
  try {
    const res = await fetch('/api/all_cards');
    if (res.ok) {
      allCardCatalog = await res.json();
      renderCardCatalog(allCardCatalog);
      updateCustomDeckUI();
    }
  } catch (err) {
    console.error('Error loading card catalog:', err);
  }
}

function filterCardCatalog(q) {
  const query = (q || '').toLowerCase();
  const filtered = allCardCatalog.filter(c => c.name.toLowerCase().includes(query));
  renderCardCatalog(filtered);
}

function renderCardCatalog(cards) {
  const grid = document.getElementById('cardCatalogGrid');
  if (!grid) return;
  grid.innerHTML = '';
  cards.forEach(c => {
    const btn = document.createElement('div');
    const isSelected = customSelectedCards.includes(c.name);
    btn.style.cssText = `background:${isSelected ? '#1565c0' : '#263238'}; border:1px solid ${isSelected ? '#64b5f6' : '#37474f'}; border-radius:4px; padding:4px 2px; text-align:center; font-size:10px; cursor:pointer; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;`;
    btn.innerHTML = `<span style="color:#fff;">${c.name.replace(/_EV1/g, '★')}</span> <span style="color:#d500f9; font-weight:bold;">${c.cost}</span>`;
    btn.onclick = () => toggleCustomCard(c.name);
    grid.appendChild(btn);
  });
}

function toggleCustomCard(name) {
  const idx = customSelectedCards.indexOf(name);
  if (idx >= 0) {
    customSelectedCards.splice(idx, 1);
  } else {
    if (customSelectedCards.length >= 8) {
      showToast('Deck already has 8 cards! Remove one first.');
      return;
    }
    customSelectedCards.push(name);
  }
  updateCustomDeckUI();
}

function updateCustomDeckUI() {
  const countEl = document.getElementById('customCount');
  if (countEl) countEl.textContent = customSelectedCards.length;
  let totalCost = 0;
  customSelectedCards.forEach(n => {
    const card = allCardCatalog.find(c => c.name === n);
    if (card) totalCost += card.cost;
  });
  const avg = customSelectedCards.length > 0 ? (totalCost / customSelectedCards.length).toFixed(1) : '0.0';
  const avgEl = document.getElementById('customAvgElixir');
  if (avgEl) avgEl.textContent = `Avg: ${avg}`;

  const slots = document.getElementById('customDeckSlots');
  if (slots) {
    slots.innerHTML = '';
    for (let i = 0; i < 8; i++) {
      const cardName = customSelectedCards[i];
      const slot = document.createElement('div');
      slot.style.cssText = 'background:#1a2230; border:1px dashed #37474f; border-radius:4px; padding:4px 2px; text-align:center; font-size:10px; min-height:28px; display:flex; align-items:center; justify-content:center; cursor:pointer;';
      if (cardName) {
        slot.style.border = '1px solid #64b5f6';
        slot.style.background = '#263238';
        slot.innerHTML = `<span style="color:#eceff1;">${cardName.replace(/_EV1/g, '★').slice(0, 8)}</span>`;
        slot.onclick = () => toggleCustomCard(cardName);
      } else {
        slot.innerHTML = '<span style="color:#546e7a;">Empty</span>';
      }
      slots.appendChild(slot);
    }
  }
  filterCardCatalog(document.getElementById('cardSearchInput') ? document.getElementById('cardSearchInput').value : '');
}

async function applyCustomDeck() {
  if (customSelectedCards.length !== 8) {
    showToast(`Need 8 cards for a deck (currently have ${customSelectedCards.length})!`);
    return;
  }
  try {
    const res = await fetch('/api/set_custom_deck', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ team: 0, deck: customSelectedCards })
    });
    const result = await res.json();
    if (result.ok) {
      showToast('⚔️ Custom 8-card deck equipped successfully!');
      pollState();
    } else {
      showToast(`Error: ${result.message}`);
    }
  } catch (err) {
    showToast('Failed to equip custom deck');
  }
}

function toCanvasX(x) { return x * TILE_PX; }
function toCanvasY(y) { return (ARENA_H - y) * TILE_PX; }
function toEngineX(cx) { return cx / TILE_PX; }
function toEngineY(cy) { return ARENA_H - (cy / TILE_PX); }

async function pollState() {
  try {
    const res = await fetch('/api/state');
    if (res.ok) {
      gameState = await res.json();
      if (!decksInitialized) {
        loadCardCatalog();
        refreshSavedReplays();
      }
      updateUI();
      renderArena();
    }
  } catch (err) {
    console.error('Polling error:', err);
  }
}
setInterval(pollState, 100);

function updateUI() {
  if (!gameState || !gameState.initialized) return;

  // Replay playback banner
  const pbBanner = document.getElementById('playbackBanner');
  if (gameState.playback_mode) {
    pbBanner.style.display = 'flex';
    document.getElementById('playbackProg').textContent = `${gameState.playback_cursor}/${gameState.playback_total}`;
  } else {
    pbBanner.style.display = 'none';
  }

  // Action feed
  document.getElementById('eventCountBadge').textContent = gameState.total_events || 0;
  const feed = document.getElementById('eventFeed');
  if (gameState.recent_events && gameState.recent_events.length > 0) {
    feed.innerHTML = '';
    gameState.recent_events.forEach(ev => {
      const row = document.createElement('div');
      const isBlue = ev.team === 0;
      row.className = `event-row ${isBlue ? 'event-blue' : 'event-red'}`;
      const sec = Math.floor(ev.t_ms / 1000);
      const m = Math.floor(sec / 60);
      const s = sec % 60;
      const tStr = `${m}:${s < 10 ? '0' : ''}${s}`;
      const teamIcon = isBlue ? '🔵' : '🔴';
      row.innerHTML = `<span>${teamIcon} <b>${ev.card}</b> <span style="color:#90a4ae;">(${ev.x.toFixed(1)}, ${ev.y.toFixed(1)})</span></span><span style="color:#78909c;">${tStr}</span>`;
      feed.appendChild(row);
    });
  } else {
    feed.innerHTML = '<div style="color:#78909c; text-align:center; padding:10px 0;">No deployments yet</div>';
  }

  // Crowns
  document.getElementById('blueCrowns').textContent = gameState.crowns[0];
  document.getElementById('redCrowns').textContent = gameState.crowns[1];

  // Timer
  const m = Math.floor(gameState.time_left_sec / 60);
  const s = gameState.time_left_sec % 60;
  document.getElementById('matchTimer').textContent = `${m}:${s < 10 ? '0' : ''}${s}`;
  document.getElementById('otBadge').style.display = gameState.is_overtime ? 'inline-block' : 'none';

  // Elixir
  const blueE = Math.min(10.0, gameState.elixir[0]);
  document.getElementById('elixirFill').style.width = `${(blueE / 10) * 100}%`;
  document.getElementById('elixirText').textContent = `${blueE.toFixed(1)} / 10`;

  const redE = Math.min(10.0, gameState.elixir[1]);
  document.getElementById('redElixirFill').style.width = `${(redE / 10) * 100}%`;
  document.getElementById('redElixirText').textContent = `${redE.toFixed(1)} / 10`;

  // Pause
  document.getElementById('pauseBtn').textContent = gameState.paused ? 'Resume' : 'Pause';

  // AI Modes
  const blueSel = document.getElementById('blueAiModeSelect');
  if (blueSel && gameState.blue_ai_mode && blueSel.value !== gameState.blue_ai_mode) {
    blueSel.value = gameState.blue_ai_mode;
  }
  const redSel = document.getElementById('redAiModeSelect');
  if (redSel && gameState.red_ai_mode && redSel.value !== gameState.red_ai_mode) {
    redSel.value = gameState.red_ai_mode;
  }

  // Populate Deck Selectors
  if (!decksInitialized && gameState.popular_decks) {
    const bSel = document.getElementById('blueDeckSelect');
    const rSel = document.getElementById('redDeckSelect');
    bSel.innerHTML = '';
    rSel.innerHTML = '';
    gameState.popular_decks.forEach(d => {
      const optB = document.createElement('option');
      optB.value = d; optB.textContent = `🔵 ${d}`;
      if (d === gameState.blue_deck_name) optB.selected = true;
      bSel.appendChild(optB);

      const optR = document.createElement('option');
      optR.value = d; optR.textContent = `🔴 ${d}`;
      if (d === gameState.red_deck_name) optR.selected = true;
      rSel.appendChild(optR);
    });
    decksInitialized = true;
  }

  // Winner Banner
  const banner = document.getElementById('winnerBanner');
  const winText = document.getElementById('winnerText');
  if (gameState.done) {
    banner.style.display = 'block';
    if (gameState.winner === 0) {
      winText.textContent = '🎉 Blue Victory!';
      winText.style.color = '#42a5f5';
    } else if (gameState.winner === 1) {
      winText.textContent = '💀 Red Victory!';
      winText.style.color = '#ef5350';
    } else {
      winText.textContent = '🤝 Match Draw!';
      winText.style.color = '#ffca28';
    }
  } else {
    banner.style.display = 'none';
  }

  // Render Hand Cards
  const grid = document.getElementById('handGrid');
  grid.innerHTML = '';
  gameState.hand.forEach((card) => {
    const slot = document.createElement('div');
    slot.className = `card-slot ${card.idx === selectedCardIdx ? 'selected' : ''} ${card.affordable ? '' : 'disabled'}`;
    slot.onclick = () => selectCard(card.idx);

    const name = document.createElement('div');
    name.className = 'card-name';
    name.textContent = card.name;

    const cost = document.createElement('div');
    cost.className = 'card-cost';
    cost.textContent = card.cost;

    slot.appendChild(name);
    slot.appendChild(cost);
    grid.appendChild(slot);
  });

  // AlphaStar Neural HUD Update
  const hudSec = document.getElementById('neuralHudSection');
  if (gameState.neural_hud) {
    hudSec.style.display = 'flex';
    if (gameState.latest_ckpt_step) {
      document.getElementById('ckptStepBadge').textContent = `Step ${gameState.latest_ckpt_step.toLocaleString()}`;
    }
    const val = gameState.neural_hud.value;
    const valText = document.getElementById('valueEstimate');
    valText.textContent = (val >= 0 ? '+' : '') + val.toFixed(2);
    valText.style.color = val >= 0 ? '#4caf50' : '#ef5350';
    const pct = Math.max(0, Math.min(100, ((val + 3.0) / 6.0) * 100));
    document.getElementById('valueMeter').style.width = pct + '%';

    const probsDiv = document.getElementById('actionProbs');
    probsDiv.innerHTML = '';
    const labels = [...gameState.hand.map(c => c.name), 'Wait (Bank Elixir)'];
    const probs = gameState.neural_hud.probs || [];
    probs.forEach((p, idx) => {
      const row = document.createElement('div');
      row.style.display = 'flex';
      row.style.justifyContent = 'space-between';
      row.style.alignItems = 'center';
      row.style.padding = '2px 0';
      const label = labels[idx] || `Slot ${idx}`;
      const pctStr = (p * 100).toFixed(1);
      row.innerHTML = `<span style="color:${idx === 4 ? '#90a4ae' : '#eceff1'};">${label}</span><span style="color:#ffca28; font-weight:bold;">${pctStr}%</span>`;
      probsDiv.appendChild(row);
    });
  } else {
    hudSec.style.display = 'none';
  }
}

function selectCard(idx) {
  if (selectedCardIdx === idx) {
    selectedCardIdx = null;
    showToast('Selection cleared');
  } else {
    selectedCardIdx = idx;
    const card = gameState.hand.find(c => c.idx === idx);
    showToast(`Selected ${card ? card.name : ''}. Click arena to deploy!`);
  }
}

function showToast(msg) {
  document.getElementById('statusToast').textContent = msg;
}

function renderArena() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Arena Grass
  ctx.fillStyle = '#5c8a42';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let x = 0; x <= ARENA_W; x++) {
    ctx.beginPath();
    ctx.moveTo(x * TILE_PX, 0);
    ctx.lineTo(x * TILE_PX, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y <= ARENA_H; y++) {
    ctx.beginPath();
    ctx.moveTo(0, y * TILE_PX);
    ctx.lineTo(canvas.width, y * TILE_PX);
    ctx.stroke();
  }

  // River & Bridges
  const riverTop = toCanvasY(16.0 + 0.6);
  const riverHeight = 1.2 * TILE_PX;
  ctx.fillStyle = '#29b6f6';
  ctx.fillRect(0, riverTop, canvas.width, riverHeight);

  ctx.fillStyle = '#8d6e63';
  ctx.fillRect(0, riverTop - 3, canvas.width, 3);
  ctx.fillRect(0, riverTop + riverHeight, canvas.width, 3);

  const drawBridge = (bx) => {
    const px = toCanvasX(bx - 1.1);
    const pw = 2.2 * TILE_PX;
    ctx.fillStyle = '#8d6e63';
    ctx.fillRect(px, riverTop - 2, pw, riverHeight + 4);
    ctx.fillStyle = '#a1887f';
    ctx.fillRect(px + 3, riverTop, pw - 6, riverHeight);
  };
  drawBridge(3.5);
  drawBridge(14.5);

  // Deployment Highlight
  if (selectedCardIdx !== null && gameState) {
    const card = gameState.hand.find(c => c.idx === selectedCardIdx);
    if (card) {
      if (card.is_spell) {
        ctx.fillStyle = 'rgba(76, 175, 80, 0.18)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      } else {
        const yTop = toCanvasY(15.4);
        ctx.fillStyle = 'rgba(76, 175, 80, 0.22)';
        ctx.fillRect(0, yTop, canvas.width, canvas.height - yTop);
      }
    }
  }

  if (!gameState) return;

  // Area Fields
  gameState.fields.forEach(f => {
    const cx = toCanvasX(f.x);
    const cy = toCanvasY(f.y);
    const r = f.radius * TILE_PX;

    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    if (f.name.includes('Poison')) {
      ctx.fillStyle = 'rgba(230, 81, 0, 0.28)';
      ctx.strokeStyle = '#e65100';
    } else if (f.name.includes('Freeze')) {
      ctx.fillStyle = 'rgba(3, 169, 244, 0.35)';
      ctx.strokeStyle = '#03a9f4';
    } else if (f.name.includes('Rage')) {
      ctx.fillStyle = 'rgba(156, 39, 176, 0.28)';
      ctx.strokeStyle = '#ab47bc';
    } else {
      ctx.fillStyle = 'rgba(255, 235, 59, 0.25)';
      ctx.strokeStyle = '#ffeb3b';
    }
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  // Units & Towers
  gameState.units.forEach(u => {
    const cx = toCanvasX(u.x);
    const cy = toCanvasY(u.y);
    const isBlue = u.team === 0;

    if (u.tower) {
      const isKing = u.name.includes('King');
      const tw = isKing ? 46 : 38;
      const th = isKing ? 46 : 38;

      ctx.fillStyle = isBlue ? '#1565c0' : '#c62828';
      ctx.beginPath();
      ctx.roundRect(cx - tw/2, cy - th/2, tw, th, 8);
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = isBlue ? '#64b5f6' : '#ef5350';
      ctx.stroke();

      drawHealthBar(cx, cy - th/2 - 10, tw + 8, u.hp, u.maxhp, isBlue);

      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(isKing ? '👑 KING' : 'TOWER', cx, cy + 4);
    } else {
      const r = u.flying ? 13 : (u.building ? 16 : 11);

      if (u.flying) {
        ctx.beginPath();
        ctx.ellipse(cx, cy + 10, r, r * 0.5, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        ctx.fill();
      }

      ctx.beginPath();
      if (u.building) {
        ctx.roundRect(cx - r, cy - r, r * 2, r * 2, 4);
      } else {
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
      }
      ctx.fillStyle = isBlue ? '#1e88e5' : '#e53935';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#fff';
      ctx.stroke();

      drawHealthBar(cx, cy - r - 7, 26, u.hp, u.maxhp, isBlue);

      ctx.fillStyle = '#fff';
      ctx.font = 'bold 9px sans-serif';
      ctx.textAlign = 'center';
      const initials = u.name.replace(/_EV1/g, '★').slice(0, 4);
      ctx.fillText(initials, cx, cy + 3);
    }
  });

  // Projectiles
  gameState.projectiles.forEach(p => {
    const cx = toCanvasX(p.x);
    const cy = toCanvasY(p.y);
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#ffeb3b';
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#ff6f00';
    ctx.stroke();
  });

  // AlphaStar AI Spatial Attention Target
  if (gameState.neural_hud && gameState.neural_hud.top_tiles) {
    gameState.neural_hud.top_tiles.forEach((t, i) => {
      const tx = toCanvasX(t.x + 0.5);
      const ty = toCanvasY(t.y + 0.5);
      ctx.beginPath();
      ctx.arc(tx, ty, 10 - i * 2, 0, Math.PI * 2);
      ctx.strokeStyle = i === 0 ? 'rgba(255, 235, 59, 0.9)' : 'rgba(255, 152, 0, 0.5)';
      ctx.lineWidth = i === 0 ? 2 : 1;
      ctx.stroke();

      if (i === 0) {
        ctx.fillStyle = '#ffeb3b';
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`🎯 AI Target (${(t.p * 100).toFixed(0)}%)`, tx, ty - 12);
      }
    });
  }

  // Placement Crosshair
  if (selectedCardIdx !== null && mouseX >= 0 && mouseY >= 0) {
    const ex = toEngineX(mouseX);
    const ey = toEngineY(mouseY);
    ctx.beginPath();
    ctx.arc(mouseX, mouseY, 14, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffeb3b';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#fff';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`(${ex.toFixed(1)}, ${ey.toFixed(1)})`, mouseX, mouseY - 18);
  }
}

function drawHealthBar(x, y, w, hp, maxhp, isBlue) {
  const h = 5;
  const pct = Math.max(0, Math.min(1.0, hp / maxhp));
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(x - w/2, y, w, h);
  ctx.fillStyle = pct > 0.5 ? '#4caf50' : (pct > 0.2 ? '#ff9800' : '#f44336');
  ctx.fillRect(x - w/2, y, w * pct, h);

  if (w > 30) {
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`${hp}`, x, y - 2);
  }
}

canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  mouseX = e.clientX - rect.left;
  mouseY = e.clientY - rect.top;
});

canvas.addEventListener('mouseleave', () => {
  mouseX = -1;
  mouseY = -1;
});

canvas.addEventListener('click', async (e) => {
  if (selectedCardIdx === null) {
    showToast('Select a card from your hand first!');
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const x = toEngineX(cx);
  const y = toEngineY(cy);

  try {
    const res = await fetch('/api/deploy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hand_idx: selectedCardIdx, x: x, y: y })
    });
    const result = await res.json();
    if (result.ok) {
      showToast('Card deployed successfully!');
      selectedCardIdx = null;
      pollState();
    } else {
      showToast(`Cannot deploy: ${result.reason}`);
    }
  } catch (err) {
    showToast('Error communicating with server');
  }
});

async function togglePause() {
  await fetch('/api/pause', { method: 'POST' });
  pollState();
}

async function stepOnce() {
  await fetch('/api/step', { method: 'POST' });
  pollState();
}

async function setSpeed(s) {
  await fetch('/api/speed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ speed: s })
  });
  ['speed1', 'speed2', 'speed4', 'speed10', 'speed20'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.className = '';
  });
  if (s === 1.0 && document.getElementById('speed1')) document.getElementById('speed1').className = 'active';
  if (s === 2.0 && document.getElementById('speed2')) document.getElementById('speed2').className = 'active';
  if (s === 4.0 && document.getElementById('speed4')) document.getElementById('speed4').className = 'active';
  if (s === 10.0 && document.getElementById('speed10')) document.getElementById('speed10').className = 'active';
  if (s === 20.0 && document.getElementById('speed20')) document.getElementById('speed20').className = 'active';
}

async function changeBlueAiMode(mode) {
  await fetch('/api/set_blue_ai_mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ blue_ai_mode: mode })
  });
  if (mode === 'human') {
    showToast('Blue: 👤 Human mode. Click a card in hand below, then click arena grass to deploy!');
  } else if (mode === 'alphastar') {
    showToast('Blue: 🤖 AlphaStar RL Agent active & auto-deploying.');
  } else {
    showToast('Blue: 🧠 Scripted Bot active & auto-deploying.');
  }
  pollState();
}

async function changeRedAiMode(mode) {
  await fetch('/api/set_red_ai_mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ red_ai_mode: mode })
  });
  pollState();
}

async function changeAiMode(mode) {
  return changeRedAiMode(mode);
}

async function resetWithSelectedDecks() {
  const b = document.getElementById('blueDeckSelect').value;
  const r = document.getElementById('redDeckSelect').value;
  await fetch('/api/reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ blue_deck_name: b, red_deck_name: r })
  });
  selectedCardIdx = null;
  showToast(`Match reset with ${b} vs ${r}`);
  pollState();
}

async function resetGame() {
  await resetWithSelectedDecks();
}

async function exportReplayJson() {
  try {
    const res = await fetch('/api/export_replay');
    if (!res.ok) throw new Error("Failed to export replay");
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${data.id || 'replay'}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Replay exported (${data.events ? data.events.length : 0} events)`);
  } catch (err) {
    showToast(`Export error: ${err.message}`);
  }
}

async function saveReplayToServer() {
  try {
    const res = await fetch('/api/save_replay', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    const result = await res.json();
    if (result.ok) {
      showToast(`Saved replay: ${result.filename}`);
      refreshSavedReplays();
    } else {
      showToast(`Save failed: ${result.error || result.message}`);
    }
  } catch (err) {
    showToast(`Save error: ${err.message}`);
  }
}

async function refreshSavedReplays() {
  try {
    const res = await fetch('/api/list_replays');
    if (!res.ok) return;
    const replays = await res.json();
    const sel = document.getElementById('savedReplaySelect');
    if (!sel) return;
    sel.innerHTML = '<option value="">(Select Saved Replay)</option>';
    replays.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.filename;
      opt.textContent = `${r.filename.substring(0, 24)}... (${r.event_count} evts)`;
      sel.appendChild(opt);
    });
  } catch (err) {
    console.error(err);
  }
}

async function loadSelectedReplay() {
  const sel = document.getElementById('savedReplaySelect');
  if (!sel || !sel.value) {
    showToast('Select a replay from dropdown first');
    return;
  }
  try {
    const res = await fetch('/api/load_replay', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: sel.value })
    });
    const result = await res.json();
    if (result.ok) {
      showToast(`▶️ Playing replay: ${sel.value}`);
      pollState();
    } else {
      showToast(`Load failed: ${result.message}`);
    }
  } catch (err) {
    showToast(`Load error: ${err.message}`);
  }
}

async function exitReplay() {
  try {
    await fetch('/api/exit_replay', { method: 'POST' });
    showToast('Exited replay mode. Normal match restored.');
    pollState();
  } catch (err) {
    showToast(`Exit error: ${err.message}`);
  }
}

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') {
    e.preventDefault();
    togglePause();
  } else if (e.key >= '1' && e.key <= '4') {
    const idx = parseInt(e.key) - 1;
    selectCard(idx);
  }
});
</script>

</body>
</html>
"""


TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "ui_template.html")


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html_content = INDEX_HTML
            if os.path.exists(TEMPLATE_PATH):
                try:
                    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
                        html_content = f.read()
                except Exception:
                    pass
            body = html_content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/state":
            self._send_json(SESSION.get_state())
        elif parsed.path == "/api/all_cards":
            self._send_json(SESSION.get_all_playable_cards())
        elif parsed.path == "/api/export_replay":
            self._send_json(SESSION.get_current_replay())
        elif parsed.path == "/api/list_replays":
            self._send_json(SESSION.list_replays())
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            body = {}

        if parsed.path == "/api/deploy":
            idx = int(body.get("hand_idx", -1))
            x = float(body.get("x", 0.0))
            y = float(body.get("y", 0.0))
            ok, reason = SESSION.play_blue(idx, x, y)
            self._send_json({"ok": ok, "reason": reason})

        elif parsed.path == "/api/pause":
            SESSION.paused = not SESSION.paused
            self._send_json({"ok": True, "paused": SESSION.paused})

        elif parsed.path == "/api/step":
            SESSION.step()
            self._send_json({"ok": True})

        elif parsed.path == "/api/speed":
            speed = float(body.get("speed", 1.0))
            SESSION.speed = max(0.1, min(20.0, speed))
            self._send_json({"ok": True, "speed": SESSION.speed})

        elif parsed.path == "/api/set_blue_ai_mode":
            mode = body.get("blue_ai_mode", "alphastar")
            SESSION.blue_ai_mode = mode
            self._send_json({"ok": True, "blue_ai_mode": SESSION.blue_ai_mode})

        elif parsed.path == "/api/set_red_ai_mode":
            mode = body.get("red_ai_mode", "heuristic")
            SESSION.red_ai_mode = mode
            SESSION.ai_mode = mode
            self._send_json({"ok": True, "red_ai_mode": SESSION.red_ai_mode})

        elif parsed.path == "/api/set_ai_mode":
            mode = body.get("ai_mode", "alphastar")
            SESSION.red_ai_mode = mode
            SESSION.ai_mode = mode
            self._send_json({"ok": True, "ai_mode": SESSION.ai_mode})

        elif parsed.path == "/api/reset":
            b_name = body.get("blue_deck_name")
            r_name = body.get("red_deck_name")
            try:
                SESSION.reset(b_name, r_name)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)})
                return
            self._send_json({"ok": True})

        elif parsed.path == "/api/set_custom_deck":
            team = int(body.get("team", 0))
            deck = body.get("deck", [])
            ok, msg = SESSION.set_custom_deck(team, deck)
            self._send_json({"ok": ok, "message": msg})

        elif parsed.path == "/api/save_replay":
            filename = body.get("filename")
            ok, res = SESSION.save_replay_to_disk(filename)
            self._send_json({"ok": ok, "filename": res if ok else None, "error": None if ok else res})

        elif parsed.path == "/api/load_replay":
            replay_data = body.get("replay")
            filename = body.get("filename")
            if not replay_data and filename:
                replays_dir = os.path.join(CRSIM_ROOT, "replays")
                fp = os.path.join(replays_dir, filename)
                if os.path.exists(fp):
                    try:
                        with open(fp, "r", encoding="utf-8") as f:
                            replay_data = json.load(f)
                    except Exception as e:
                        self._send_json({"ok": False, "message": f"Failed reading replay: {e}"})
                        return
                else:
                    self._send_json({"ok": False, "message": "Replay file not found"})
                    return
            if replay_data:
                ok, msg = SESSION.load_replay(replay_data)
                self._send_json({"ok": ok, "message": msg})
            else:
                self._send_json({"ok": False, "message": "No replay payload provided"})

        elif parsed.path == "/api/seek_replay":
            target_t_ms = body.get("t_ms", 0)
            ok, msg = SESSION.seek_replay(target_t_ms)
            self._send_json({"ok": ok, "message": msg})

        elif parsed.path == "/api/exit_replay":
            SESSION.exit_replay()
            self._send_json({"ok": True, "message": "Exited replay mode"})

        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Clash Royale Headless Simulator UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    args = parser.parse_args()

    sim_thread = threading.Thread(target=simulation_worker, daemon=True)
    sim_thread.start()

    HTTPServer.allow_reuse_address = True
    server = HTTPServer((args.host, args.port), RequestHandler)
    url = f"http://{args.host}:{args.port}"
    print("=" * 75)
    print("      Clash Royale Headless Simulator & AlphaStar Agent - Web UI")
    print("=" * 75)
    print(f"[Server Running] {url}")
    print(f"[Blue Mode] {SESSION.blue_ai_mode.upper()} ({'AlphaStar Agent' if SESSION.blue_ai_mode == 'alphastar' else SESSION.blue_ai_mode})")
    print(f"[Red Mode]  {SESSION.red_ai_mode.upper()} ({'checkpoints/latest.pt' if SESSION.red_ai_mode == 'alphastar' else SESSION.red_ai_mode})")
    print(f"[Instructions] Open {url} in your browser to watch or play!")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI server...")
    finally:
        SESSION.running = False
        server.server_close()


if __name__ == "__main__":
    main()
