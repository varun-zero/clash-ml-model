"""
CR-Zero Real-Time Inference Execution Engine (Windows 11 Runtime).
Operates at 5-10 Hz on RTX 3050 6GB GPU:
DXCam -> YOLO Detection (FP16) -> StateEncoder -> DeckConditionedPolicy -> ADB Harness.
"""
from typing import Optional, List, Tuple
import os
import sys
import time
import signal
import cv2
import numpy as np
import torch
from collections import deque

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.constants import (
    ARENA_COLS,
    ARENA_ROWS,
    DEFAULT_DECK_CLASSES,
    NUM_CARD_CLASSES_V1,
    STACKED_GRID_CHANNELS,
    NUM_SCALARS,
    CARD_DATABASE,
    grid_to_screen_pixel,
    hand_slot_to_screen_pixel,
)
from perception.detector import ClashRoyaleDetector
from perception.state_encoder import StateEncoder, stack_spatial_frames
from policy.model import DeckConditionedPolicy
from controller.adb_harness import ADBHarness
from controller.screen_capture import WindowCapture


class CRZeroLiveAgent:
    """
    Modular real-time agent coordinating vision capture, perception, decision policy,
    and device actuation.
    """
    def __init__(
        self,
        checkpoint_path: str = "./checkpoints/cr_zero_policy_best.pt",
        weights: Optional[str] = None,
        device_address: str = "127.0.0.1:5555",
        screen_width: int = 540,
        screen_height: int = 960,
        target_hz: float = 5.0,
        temperature: float = 0.7,
        play_thresh: float = 0.35,
        show_debug_window: bool = True,
        use_fp16: bool = True,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.target_dt = 1.0 / max(1.0, target_hz)
        self.temperature = temperature
        self.play_thresh = play_thresh
        self.show_debug_window = show_debug_window
        self.running = False

        # Compute device
        self.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[CR-Zero] Initializing inference on {self.torch_device} (FP16: {use_fp16})...")

        # 1. Screen Capture Engine (DirectX Desktop Duplication with mss fallback)
        self.capture = WindowCapture(
            window_title_keyword="BlueStacks",
            target_width=screen_width,
            target_height=screen_height,
        )

        # 2. Vision Perception (YOLO / Template / Heuristics)
        self.detector = ClashRoyaleDetector(
            weights=weights,
            conf_threshold=0.35,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_fp16=use_fp16,
            card_classes=DEFAULT_DECK_CLASSES,
        )
        self.encoder = StateEncoder()

        # 3. Decision Policy Model (Stacked 32 channels, 12 scalars, 10 card classes)
        self.policy = DeckConditionedPolicy(
            num_card_classes=NUM_CARD_CLASSES_V1,
            card_emb_dim=16,
            num_scalars=12,
            in_channels=STACKED_GRID_CHANNELS,
        ).to(self.torch_device)
        self._load_policy_weights(checkpoint_path)
        self.policy.eval()

        # 4. ADB Tap Harness (Persistent interactive shell pipe)
        self.adb = ADBHarness(
            device_address=device_address,
            screen_width=screen_width,
            screen_height=screen_height,
            tap_delay_ms=40,
        )

    def _load_policy_weights(self, checkpoint_path: str):
        """Loads trained behavior cloning weights."""
        if os.path.exists(checkpoint_path):
            print(f"[CR-Zero] Loading policy weights from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=self.torch_device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.policy.load_state_dict(state_dict)
            print("[CR-Zero] Policy loaded successfully.")
        else:
            print(f"[CR-Zero] Warning: Checkpoint {checkpoint_path} not found. Running with initialized policy weights.")

    def run(self):
        """Main real-time control loop."""
        self.running = True
        print("\n" + "=" * 60)
        print(f"CR-Zero Live Loop Active (Target: {1.0/self.target_dt:.1f} Hz, Temp: {self.temperature}, Thresh: {self.play_thresh}).")
        print("Press Ctrl+C or 'q' to stop.")
        print("=" * 60 + "\n")

        fps_timer = time.time()
        frame_counter = 0
        fps = 0.0

        grid_history = deque(maxlen=4)
        last_action = (0, 0)

        try:
            while self.running:
                loop_start = time.perf_counter()

                # Step 1: Capture frame
                t_cap0 = time.perf_counter()
                frame = self.capture.grab_frame()
                dt_cap = (time.perf_counter() - t_cap0) * 1000.0

                # Step 2: Vision & Observation parsing
                t_det0 = time.perf_counter()
                obs = self.detector.parse_frame(
                    frame,
                    time_left=180.0,
                    est_enemy_elixir=0.5,
                    last_action=last_action,
                )
                dt_det = (time.perf_counter() - t_det0) * 1000.0

                arena_grid = obs["arena_grid"]
                grid_history.append(arena_grid.copy())
                # Stack last 4 frames into (32, 32, 18)
                stacked_grid = stack_spatial_frames(list(grid_history), target_frames=4)

                scalars = obs["scalars"]
                legal_card_mask = obs["legal_card_mask"].copy()
                hand_ids = obs["hand_ids"]
                elixir = obs["elixir"]

                # No-Wait guard at elixir == 10: force playing a card so elixir is never burned
                if elixir >= 9.8 and legal_card_mask[1:].any():
                    legal_card_mask[0] = False

                # Step 3: Policy evaluation
                t_pol0 = time.perf_counter()
                # Prepare tensors on GPU
                grid_tensor = torch.from_numpy(stacked_grid).unsqueeze(0).to(self.torch_device)
                scalar_tensor = torch.from_numpy(scalars).unsqueeze(0).to(self.torch_device)
                hand_tensor = torch.tensor([hand_ids], dtype=torch.long, device=self.torch_device)
                cmask_tensor = torch.from_numpy(legal_card_mask).unsqueeze(0).to(self.torch_device)

                # Default tile mask for card 0 / general
                default_tile_mask = self.encoder.get_legal_tile_mask(card_id=0)
                tmask_tensor = torch.from_numpy(default_tile_mask).unsqueeze(0).to(self.torch_device)

                # Forward pass with temperature sampling
                chosen_slot, chosen_tile, card_conf, tile_conf = self.policy.predict_action(
                    arena_grid=grid_tensor,
                    scalars=scalar_tensor,
                    hand_card_ids=hand_tensor,
                    legal_card_mask=cmask_tensor,
                    legal_tile_mask=tmask_tensor,
                    temperature=self.temperature,
                    deterministic=False,
                )

                # Play threshold check: if card confidence is low and waiting is legal, revert to Wait
                if chosen_slot > 0 and card_conf < self.play_thresh and legal_card_mask[0]:
                    chosen_slot = 0
                    chosen_tile = 0

                # If card slot chosen > 0 (i.e. play a card 1..4), recalculate tile with true card legality
                if chosen_slot > 0:
                    played_card_id = hand_ids[chosen_slot - 1]
                    specific_tile_mask = self.encoder.get_legal_tile_mask(played_card_id)
                    tmask_tensor = torch.from_numpy(specific_tile_mask).unsqueeze(0).to(self.torch_device)

                    # Re-evaluate tile head with specific legal tile mask
                    _, chosen_tile, _, tile_conf = self.policy.predict_action(
                        arena_grid=grid_tensor,
                        scalars=scalar_tensor,
                        hand_card_ids=hand_tensor,
                        legal_card_mask=cmask_tensor,
                        legal_tile_mask=tmask_tensor,
                        temperature=self.temperature,
                    )
                dt_pol = (time.perf_counter() - t_pol0) * 1000.0

                # Step 4: Dispatch action via ADB
                t_act0 = time.perf_counter()
                tile_y = chosen_tile // ARENA_COLS
                tile_x = chosen_tile % ARENA_COLS

                if chosen_slot > 0:
                    slot_idx = chosen_slot - 1
                    played_card_id = hand_ids[slot_idx]
                    card_meta = CARD_DATABASE.get(played_card_id, {"name": f"Card #{played_card_id}"})
                    self.adb.play_card(slot_idx, tile_y, tile_x, non_blocking=True)
                    action_desc = f"PLAY [{card_meta['name']}] (Slot {slot_idx}) -> Tile ({tile_y:02d}, {tile_x:02d}) [Conf: {card_conf*100:.1f}%]"
                    last_action = (chosen_slot, chosen_tile)
                else:
                    action_desc = f"WAIT (Elixir: {elixir:.1f}/10) [Conf: {card_conf*100:.1f}%]"
                    last_action = (0, 0)
                dt_act = (time.perf_counter() - t_act0) * 1000.0

                # FPS and Latency bookkeeping
                frame_counter += 1
                if time.time() - fps_timer >= 1.0:
                    fps = frame_counter / (time.time() - fps_timer)
                    frame_counter = 0
                    fps_timer = time.time()

                total_latency = (time.perf_counter() - loop_start) * 1000.0
                print(
                    f"[{fps:4.1f} FPS] {action_desc} | "
                    f"Latency: {total_latency:.1f}ms (Cap: {dt_cap:.1f}ms, Det: {dt_det:.1f}ms, Pol: {dt_pol:.1f}ms)"
                )

                # Step 5: Optional debug overlay
                if self.show_debug_window:
                    self._render_overlay(
                        frame=frame,
                        action_desc=action_desc,
                        fps=fps,
                        chosen_slot=chosen_slot,
                        tile_y=tile_y,
                        tile_x=tile_x,
                        hand_ids=hand_ids,
                        elixir=elixir,
                        detections=obs["detections"],
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[CR-Zero] Exit key 'q' pressed.")
                        break

                # Sleep to maintain target loop frequency (e.g. 5 Hz = 200ms)
                loop_elapsed = time.perf_counter() - loop_start
                sleep_sec = max(0.0, self.target_dt - loop_elapsed)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

        except KeyboardInterrupt:
            print("\n[CR-Zero] Interrupted by user.")
        finally:
            self.stop()

    def _render_overlay(
        self,
        frame: np.ndarray,
        action_desc: str,
        fps: float,
        chosen_slot: int,
        tile_y: int,
        tile_x: int,
        hand_ids: list,
        elixir: float,
        detections: list,
    ):
        """Draws HUD overlay with detections, elixir, and target crosshair."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw detected units
        for d in detections:
            x1, y1 = int(d.x1 * w), int(d.y1 * h)
            x2, y2 = int(d.x2 * w), int(d.y2 * h)
            color = (0, 255, 0) if d.is_ally else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{d.category}:{d.confidence:.2f}", (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw target placement crosshair if action is a play
        if chosen_slot > 0:
            px, py = grid_to_screen_pixel(tile_y, tile_x, w, h)
            cv2.circle(vis, (px, py), 12, (0, 255, 255), 3)
            cv2.drawMarker(vis, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(vis, f"DEPLOY ({tile_y},{tile_x})", (px + 15, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # HUD Top Banner
        cv2.rectangle(vis, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.putText(vis, f"CR-Zero | {fps:.1f} FPS | Elixir: {elixir:.1f}/10", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, action_desc, (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

        # Display window (scaled down if height > 800)
        disp_h = 720
        disp_w = int(round(w * (disp_h / h)))
        resized_disp = cv2.resize(vis, (disp_w, disp_h))
        cv2.imshow("CR-Zero Live Agent HUD", resized_disp)

    def stop(self):
        """Cleans up capture, ADB, and display windows."""
        self.running = False
        self.capture.close()
        self.adb.close()
        if self.show_debug_window:
            cv2.destroyAllWindows()
        print("[CR-Zero] Shutdown complete.")


def run_dummy_tap(
    device_address: str = "127.0.0.1:5555",
    screen_width: int = 540,
    screen_height: int = 960,
):
    """
    Executes a single fixed card tap sequence (slot 0 -> tile 20, 9) to test ADB actuation and exits.
    """
    print(f"\n[CR-Zero Dummy] Initiating dummy ADB tap mode on {device_address}...")
    adb = ADBHarness(
        device_address=device_address,
        screen_width=screen_width,
        screen_height=screen_height,
        tap_delay_ms=40,
    )
    try:
        print("[CR-Zero Dummy] Dispatching fixed tap: Slot 0 -> Arena Tile (20, 9)...")
        adb.play_card(slot_idx=0, tile_y=20, tile_x=9, non_blocking=False)
        time.sleep(0.5)
        print("[CR-Zero Dummy] Fixed test tap successfully executed.")
    except Exception as e:
        print(f"[CR-Zero Dummy] Notice: Tap dispatch encountered: {e}")
    finally:
        adb.close()
    print("[CR-Zero Dummy] Dummy test completed cleanly.\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CR-Zero Real-Time Live Windows Agent")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/cr_zero_policy_best.pt", help="Policy checkpoint")
    parser.add_argument("--weights", "--yolo", dest="weights", type=str, default=None, help="YOLO model path (.pt or .onnx)")
    parser.add_argument("--device", type=str, default="127.0.0.1:5555", help="ADB device host:port")
    parser.add_argument("--hz", type=float, default=5.0, help="Inference frequency (Hz)")
    parser.add_argument("--temp", "--temperature", dest="temp", type=float, default=0.7, help="Temperature for action sampling (default: 0.7)")
    parser.add_argument("--play-thresh", type=float, default=0.35, help="Confidence threshold to play a card vs wait (default: 0.35)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug visual window")
    parser.add_argument("--dummy", action="store_true", help="Run dummy mode: tap fixed card/tile to test ADB and exit")
    args = parser.parse_args()

    if args.dummy:
        run_dummy_tap(device_address=args.device)
        return

    agent = CRZeroLiveAgent(
        checkpoint_path=args.checkpoint,
        weights=args.weights,
        device_address=args.device,
        target_hz=args.hz,
        temperature=args.temp,
        play_thresh=args.play_thresh,
        show_debug_window=not args.no_debug,
    )
    agent.run()


if __name__ == "__main__":
    main()

