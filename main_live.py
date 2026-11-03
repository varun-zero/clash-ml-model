"""
CR-Zero Real-Time Live Windows Agent.
Captures BlueStacks 5 window client area, parses game state via YOLO & templates,
evaluates either DeckConditionedPolicy (neural) or RuleBotPolicy (heuristic rule engine),
and executes non-blocking two-tap actions via safety-hardened ADBHarness.
"""
from typing import Optional, Tuple, Dict, Any, List
import os
import sys
import time
from collections import deque
import cv2
import numpy as np
import torch

from configs.constants import (
    ARENA_COLS,
    ARENA_ROWS,
    TOTAL_TILES,
    CARD_DATABASE,
    CARD_ID_TO_COST,
    CARD_ID_TO_IS_SPELL,
    CARD_ID_TO_NAME,
    NUM_TOTAL_CARDS,
    STACKED_GRID_CHANNELS,
    grid_to_screen_pixel,
)
from perception.detector import ClashRoyaleDetector
from perception.state_encoder import StateEncoder, stack_spatial_frames
from policy.model import DeckConditionedPolicy
from policy.rule_bot import RuleBotPolicy
from controller.screen_capture import WindowCapture
from controller.adb_harness import ADBHarness
from configs.observation_schema import SCHEMA_VERSION


class CRZeroLiveAgent:
    """
    Modular real-time agent coordinating vision capture, perception, decision policy,
    and device actuation with safety guardrails.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        weights: Optional[str] = None,
        use_rule_bot: bool = False,
        device_address: str = "127.0.0.1:5555",
        adb_path: Optional[str] = None,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        target_hz: float = 5.0,
        temperature: float = 0.7,
        play_thresh: float = 0.35,
        max_actions: Optional[int] = None,
        action_cooldown_sec: float = 0.800,
        dry_run: bool = False,
        log_dir: Optional[str] = None,
        show_debug_window: bool = True,
        use_fp16: bool = True,
    ):
        self.use_rule_bot = use_rule_bot
        self.dry_run = dry_run
        self.target_dt = 1.0 / max(1.0, target_hz)
        self.temperature = temperature
        self.play_thresh = play_thresh
        self.show_debug_window = show_debug_window
        self.log_dir = log_dir
        self.running = False

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"[CR-Zero] Match logging active -> {self.log_dir}")

        # Compute device
        self.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[CR-Zero] Initializing (Device: {self.torch_device}, RuleBot: {use_rule_bot}, DryRun: {dry_run})...")

        # 1. Screen Capture Engine (Client Area + Direct ADB Socket Fallback)
        self.capture = WindowCapture(
            window_title_keyword="BlueStacks",
            target_width=screen_width or 540,
            target_height=screen_height or 960,
            adb_device=device_address,
        )

        # 2. Vision Perception (YOLO / Templates / OCR)
        # Require weights if not in dry_run and not rule_bot
        self.detector = ClashRoyaleDetector(
            weights=weights,
            conf_threshold=0.35,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_fp16=use_fp16,
            require_weights=False,  # Checked at CLI entrypoint
        )
        self.encoder = StateEncoder()

        # 3. Decision Policy Model (RuleBot or DeckConditionedPolicy)
        if self.use_rule_bot:
            print("[CR-Zero] Running with RuleBot heuristic strategy engine.")
            self.rule_bot = RuleBotPolicy()
            self.policy = None
        else:
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                raise ValueError(
                    f"Policy checkpoint required: '{checkpoint_path}' does not exist.\n"
                    f"Train a policy or specify --rule-bot to use the heuristic rule engine."
                )
            self.policy = DeckConditionedPolicy(
                num_card_classes=NUM_TOTAL_CARDS,
                card_emb_dim=16,
                num_scalars=12,
                in_channels=STACKED_GRID_CHANNELS,
            ).to(self.torch_device)
            self._load_policy_weights(checkpoint_path)
            self.policy.eval()

        # 4. ADB Tap Harness
        self.adb = ADBHarness(
            device_address=device_address,
            adb_path=adb_path,
            screen_width=screen_width,
            screen_height=screen_height,
            tap_delay_ms=40,
            action_cooldown_sec=action_cooldown_sec,
            max_actions=max_actions,
            dry_run=dry_run,
        )

        # Match trajectory buffer for --log-dir
        self.match_buffer = {
            "arena_grid": [],
            "scalars": [],
            "hand_ids": [],
            "legal_card_mask": [],
            "legal_tile_mask": [],
            "action_card": [],
            "action_tile": [],
            "timestamps": [],
        }

    def _load_policy_weights(self, checkpoint_path: str):
        """Loads trained behavior cloning weights."""
        print(f"[CR-Zero] Loading policy weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.torch_device, weights_only=True)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.policy.load_state_dict(state_dict)
        print("[CR-Zero] Policy loaded successfully.")

    def run(self):
        """Main real-time control loop."""
        self.running = True
        print("\n" + "=" * 60)
        print(f"CR-Zero Live Loop Active (Target: {1.0/self.target_dt:.1f} Hz, Temp: {self.temperature}).")
        print("Safety: Press F12 or Ctrl+C to emergency stop.")
        print("=" * 60 + "\n")

        fps_timer = time.time()
        frame_counter = 0
        fps = 0.0

        grid_history = deque(maxlen=4)
        last_action = (0, 0)
        pending_plays: Dict[int, Dict[str, Any]] = {}

        try:
            while self.running:
                if self.adb.is_panicked():
                    print("[CR-Zero] Panic flag detected. Halting loop.")
                    break

                loop_start = time.perf_counter()

                # Step 1: Capture client frame
                t_cap0 = time.perf_counter()
                frame = self.capture.grab_frame()
                dt_cap = (time.perf_counter() - t_cap0) * 1000.0

                # Step 2: Vision & Observation parsing
                t_det0 = time.perf_counter()
                obs = self.detector.parse_frame(
                    frame,
                    est_enemy_elixir=0.5,
                    last_action=last_action,
                )
                dt_det = (time.perf_counter() - t_det0) * 1000.0

                screen_state = obs["screen_state"]

                # Handle Result Screen & Match Shard Logging
                if screen_state == "result":
                    # Drain any pending plays that finished right as the battle concluded
                    if self.log_dir:
                        for ack in self.adb.get_completed_actions():
                            aid = ack.get("action_id", 0)
                            if aid in pending_plays:
                                play = pending_plays.pop(aid)
                                if ack.get("success", False):
                                    self.match_buffer["arena_grid"].append(play["grid"])
                                    self.match_buffer["scalars"].append(play["scalars"])
                                    self.match_buffer["hand_ids"].append(play["hand_ids"])
                                    self.match_buffer["legal_card_mask"].append(play["legal_card_mask"])
                                    self.match_buffer["legal_tile_mask"].append(play["played_tile_mask"])
                                    self.match_buffer["action_card"].append(play["card_slot"])
                                    self.match_buffer["action_tile"].append(play["tile"])
                                    self.match_buffer["timestamps"].append(play["timestamp"])
                        pending_plays.clear()

                    if self.log_dir and len(self.match_buffer["action_card"]) > 10:
                        ts = int(time.time())
                        outcome = result_meta.get("outcome", "unknown")
                        shard_name = f"live_match_{ts}_{outcome}.npz"
                        shard_path = os.path.join(self.log_dir, shard_name)
                        np.savez_compressed(
                            shard_path,
                            arena_grid=np.array(self.match_buffer["arena_grid"], dtype=np.float32),
                            scalars=np.array(self.match_buffer["scalars"], dtype=np.float32),
                            hand_ids=np.array(self.match_buffer["hand_ids"], dtype=np.int64),
                            legal_card_mask=np.array(self.match_buffer["legal_card_mask"], dtype=bool),
                            legal_tile_mask=np.array(self.match_buffer["legal_tile_mask"], dtype=bool),
                            action_card=np.array(self.match_buffer["action_card"], dtype=np.int64),
                            action_tile=np.array(self.match_buffer["action_tile"], dtype=np.int64),
                            timestamps=np.array(self.match_buffer["timestamps"], dtype=np.float32),
                            schema_version=np.int32(SCHEMA_VERSION),
                            match_outcome=1 if outcome == "victory" else (-1 if outcome == "defeat" else 0),
                            ally_crowns=result_meta["ally_crowns"],
                            enemy_crowns=result_meta["enemy_crowns"],
                        )
                        print(f"[Match Logger] Saved match shard to {shard_path} (Outcome: {outcome})")

                    if self.log_dir:
                        self.match_buffer = {k: [] for k in self.match_buffer}

                    action_desc = "[RESULT] Match Concluded. Waiting for Lobby..."
                    grid_history.clear(); last_action = (0, 0); pending_plays.clear()   # new match must not inherit old state
                    time.sleep(0.5)
                    continue

                if screen_state != "battle":
                    grid_history.clear(); last_action = (0, 0); pending_plays.clear()
                    # Throttle idle logging to once every 3 seconds
                    now = time.time()
                    if not hasattr(self, "_last_idle_log") or now - self._last_idle_log >= 3.0:
                        self._last_idle_log = now
                        print(f"[{screen_state.upper()}] Idle: Waiting for active battle in BlueStacks... (F12 to stop)", flush=True)
                    time.sleep(0.2)
                    continue

                # In Battle: process observation
                arena_grid = obs["arena_grid"]
                grid_history.append(arena_grid.copy())
                stacked_grid = stack_spatial_frames(list(grid_history), target_frames=4)

                scalars = obs["scalars"]
                legal_card_mask = obs["legal_card_mask"].copy()
                hand_ids = obs["hand_ids"]
                elixir = obs["elixir"]

                # No-Wait guard at elixir == 10: force playing a card so elixir is not wasted
                if elixir >= 9.8 and legal_card_mask[1:].any():
                    legal_card_mask[0] = False

                # Step 3: Policy / RuleBot Evaluation
                t_pol0 = time.perf_counter()
                chosen_slot = 0
                chosen_tile = 0
                card_conf = 1.0
                tile_conf = 1.0

                if self.use_rule_bot:
                    chosen_slot, chosen_tile, card_conf, tile_conf = self.rule_bot.predict_action(
                        arena_grid=stacked_grid,
                        scalars=scalars,
                        hand_card_ids=hand_ids,
                        legal_card_mask=legal_card_mask,
                        detections=obs.get("detections"),
                    )
                else:
                    # Neural policy evaluation
                    grid_tensor = torch.from_numpy(stacked_grid).unsqueeze(0).to(self.torch_device)
                    scalar_tensor = torch.from_numpy(scalars).unsqueeze(0).to(self.torch_device)
                    hand_tensor = torch.tensor([hand_ids], dtype=torch.long, device=self.torch_device)
                    cmask_tensor = torch.from_numpy(legal_card_mask).unsqueeze(0).to(self.torch_device)

                    # 1. Sample card slot
                    chosen_slot, card_conf = self.policy.predict_card(
                        arena_grid=grid_tensor,
                        scalars=scalar_tensor,
                        hand_card_ids=hand_tensor,
                        legal_card_mask=cmask_tensor,
                        temperature=self.temperature,
                    )

                    # Play threshold check
                    if chosen_slot > 0 and card_conf < self.play_thresh and legal_card_mask[0]:
                        chosen_slot = 0

                    # 2. Condition tile prediction on chosen card slot
                    if chosen_slot == 0:
                        chosen_tile = 0
                        tile_conf = 1.0
                    else:
                        played_cid = hand_ids[chosen_slot - 1]
                        tile_mask = self.encoder.get_legal_tile_mask(
                            played_cid, tower_hps=obs.get("tower_hps")
                        )
                        tmask_tensor = torch.from_numpy(tile_mask).unsqueeze(0).to(self.torch_device)
                        chosen_tile, tile_conf = self.policy.predict_tile(
                            arena_grid=grid_tensor,
                            scalars=scalar_tensor,
                            hand_card_ids=hand_tensor,
                            legal_tile_mask=tmask_tensor,
                            card_slot=chosen_slot,
                            temperature=self.temperature,
                        )

                dt_pol = (time.perf_counter() - t_pol0) * 1000.0

                # Step 4: Dispatch action via ADB
                t_act0 = time.perf_counter()
                tile_y = chosen_tile // ARENA_COLS
                tile_x = chosen_tile % ARENA_COLS

                # Drain completed ADB tap acknowledgements
                if self.log_dir:
                    for ack in self.adb.get_completed_actions():
                        aid = ack.get("action_id", 0)
                        if aid in pending_plays:
                            play = pending_plays.pop(aid)
                            if ack.get("success", False):
                                # Tap confirmed physically executed; commit to trajectory
                                self.match_buffer["arena_grid"].append(play["grid"])
                                self.match_buffer["scalars"].append(play["scalars"])
                                self.match_buffer["hand_ids"].append(play["hand_ids"])
                                self.match_buffer["legal_card_mask"].append(play["legal_card_mask"])
                                self.match_buffer["legal_tile_mask"].append(play["played_tile_mask"])
                                self.match_buffer["action_card"].append(play["card_slot"])
                                self.match_buffer["action_tile"].append(play["tile"])
                                self.match_buffer["timestamps"].append(play["timestamp"])

                played_tile_mask = None
                if chosen_slot > 0:
                    slot_idx = chosen_slot - 1
                    played_card_id = hand_ids[slot_idx]
                    card_meta = CARD_DATABASE.get(played_card_id, {"name": f"Card #{played_card_id}"})
                    sent = self.adb.play_card(slot_idx, tile_y, tile_x, non_blocking=True)
                    if sent:
                        aid = self.adb.get_last_action_id()
                        played_tile_mask = self.encoder.get_legal_tile_mask(played_card_id, tower_hps=obs.get("tower_hps"))
                        action_desc = f"PLAY [{card_meta['name']}] (Slot {slot_idx}) -> Tile ({tile_y:02d}, {tile_x:02d}) [Conf: {card_conf*100:.1f}%]"
                        last_action = (chosen_slot, chosen_tile)   # persists until the next accepted play
                        if self.log_dir:
                            pending_plays[aid] = {
                                "card_slot": chosen_slot,
                                "tile": chosen_tile,
                                "played_tile_mask": played_tile_mask,
                                "grid": stacked_grid,
                                "scalars": scalars,
                                "hand_ids": np.array(hand_ids, dtype=np.int64),
                                "legal_card_mask": legal_card_mask,
                                "timestamp": time.time(),
                            }
                    else:
                        # harness rejected it (cooldown / limit / panic): record a Wait, never a phantom play
                        chosen_slot, chosen_tile = 0, 0
                        action_desc = "REJECTED by ADB harness -> logged as WAIT"
                        if self.log_dir:
                            self.match_buffer["arena_grid"].append(stacked_grid)
                            self.match_buffer["scalars"].append(scalars)
                            self.match_buffer["hand_ids"].append(np.array(hand_ids, dtype=np.int64))
                            self.match_buffer["legal_card_mask"].append(legal_card_mask)
                            self.match_buffer["legal_tile_mask"].append(self.encoder.get_legal_tile_mask(0))
                            self.match_buffer["action_card"].append(0)
                            self.match_buffer["action_tile"].append(0)
                            self.match_buffer["timestamps"].append(time.time())
                else:
                    action_desc = f"WAIT (Elixir: {elixir:.1f}/10) [Conf: {card_conf*100:.1f}%]"
                    if self.log_dir:
                        self.match_buffer["arena_grid"].append(stacked_grid)
                        self.match_buffer["scalars"].append(scalars)
                        self.match_buffer["hand_ids"].append(np.array(hand_ids, dtype=np.int64))
                        self.match_buffer["legal_card_mask"].append(legal_card_mask)
                        self.match_buffer["legal_tile_mask"].append(self.encoder.get_legal_tile_mask(0))
                        self.match_buffer["action_card"].append(0)
                        self.match_buffer["action_tile"].append(0)
                        self.match_buffer["timestamps"].append(time.time())

                dt_act = (time.perf_counter() - t_act0) * 1000.0

                # FPS and latency tracking
                frame_counter += 1
                if time.time() - fps_timer >= 1.0:
                    fps = frame_counter / (time.time() - fps_timer)
                    frame_counter = 0
                    fps_timer = time.time()

                total_latency = (time.perf_counter() - loop_start) * 1000.0
                print(
                    f"[{fps:4.1f} FPS] {action_desc} | "
                    f"Latency: {total_latency:.1f}ms (Cap: {dt_cap:.1f}ms, Det: {dt_det:.1f}ms, Pol: {dt_pol:.1f}ms)",
                    flush=True
                )

                # Sleep to maintain frequency (5 Hz)
                loop_elapsed = time.perf_counter() - loop_start
                sleep_sec = max(0.0, self.target_dt - loop_elapsed)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

        except KeyboardInterrupt:
            print("\n[CR-Zero] Interrupted by user.")
        finally:
            self.stop()

    def stop(self):
        """Cleans up capture, ADB, and display resources."""
        self.running = False
        self.capture.close()
        self.adb.close()
        print("[CR-Zero] Shutdown complete.")


def run_dummy_tap(
    device_address: str = "127.0.0.1:5555",
    screen_width: int = 540,
    screen_height: int = 960,
    adb_path: Optional[str] = None,
    dry_run: bool = True,
):
    """Executes a single test tap sequence (slot 0 -> tile 20, 9) and exits cleanly."""
    print(f"\n[CR-Zero Dummy] Initiating ADB tap test on {device_address} (dry_run: {dry_run})...")
    adb = ADBHarness(
        device_address=device_address,
        adb_path=adb_path,
        screen_width=screen_width,
        screen_height=screen_height,
        dry_run=dry_run,
    )
    try:
        print("[CR-Zero Dummy] Dispatching fixed tap: Slot 0 -> Arena Tile (20, 9)...")
        ok = adb.play_card(slot_idx=0, tile_y=20, tile_x=9, non_blocking=False)
        time.sleep(0.2)
        print(f"[CR-Zero Dummy] Test tap dispatch result: {ok}")
    finally:
        adb.close()
    print("[CR-Zero Dummy] Dummy test completed cleanly.\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CR-Zero Real-Time Live Windows Agent")
    parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint (.pt)")
    parser.add_argument("--weights", "--yolo", dest="weights", type=str, default=None, help="YOLO model path (.pt or .onnx)")
    parser.add_argument("--rule-bot", action="store_true", help="Run with RuleBot heuristic strategy engine instead of neural policy")
    parser.add_argument("--device", type=str, default="127.0.0.1:5555", help="ADB device host:port")
    parser.add_argument("--adb-path", type=str, default=None, help="Explicit path to adb binary")
    parser.add_argument("--width", type=int, default=None, help="Screen width override")
    parser.add_argument("--height", type=int, default=None, help="Screen height override")
    parser.add_argument("--hz", type=float, default=5.0, help="Inference frequency (Hz)")
    parser.add_argument("--temp", "--temperature", dest="temp", type=float, default=0.7, help="Temperature for action sampling")
    parser.add_argument("--play-thresh", type=float, default=0.35, help="Confidence threshold to play a card vs wait")
    parser.add_argument("--max-actions", type=int, default=None, help="Maximum actions before halting")
    parser.add_argument("--cooldown", type=float, default=0.8, help="Action cooldown in seconds (default: 0.8)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run: log actions without sending taps to emulator")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory to save match trajectory shards (.npz)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug visual window")
    parser.add_argument("--dummy", action="store_true", help="Run dummy mode: tap fixed card/tile to test ADB and exit")
    args = parser.parse_args()

    if args.dummy:
        run_dummy_tap(
            device_address=args.device,
            adb_path=args.adb_path,
            screen_width=args.width or 540,
            screen_height=args.height or 960,
            dry_run=args.dry_run,
        )
        return

    # Safety: Require weights and checkpoint unless running rule-bot
    if not args.rule_bot:
        if not args.weights or not os.path.exists(args.weights):
            raise ValueError(
                f"--weights path is required for live detection: '{args.weights}' not found.\n"
                f"Please provide trained YOLO weights (.pt / .onnx) or run with --rule-bot."
            )
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            raise ValueError(
                f"--checkpoint path is required for neural policy: '{args.checkpoint}' not found.\n"
                f"Please provide trained checkpoint (.pt) or run with --rule-bot."
            )

    agent = CRZeroLiveAgent(
        checkpoint_path=args.checkpoint,
        weights=args.weights,
        use_rule_bot=args.rule_bot,
        device_address=args.device,
        adb_path=args.adb_path,
        screen_width=args.width,
        screen_height=args.height,
        target_hz=args.hz,
        temperature=args.temp,
        play_thresh=args.play_thresh,
        max_actions=args.max_actions,
        action_cooldown_sec=args.cooldown,
        dry_run=args.dry_run,
        log_dir=args.log_dir,
        show_debug_window=not args.no_debug,
    )
    agent.run()


if __name__ == "__main__":
    main()
