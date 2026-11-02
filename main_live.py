"""
CR-Zero Real-Time Inference Execution Engine (Windows 11 Runtime).
Operates at 5-10 Hz on RTX 3050 6GB GPU:
DXCam -> YOLO Detection (FP16) -> StateEncoder -> DeckConditionedPolicy -> ADB Harness.
"""
from typing import Optional
import os
import sys
import time
import signal
import cv2
import numpy as np
import torch

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.constants import (
    ARENA_COLS,
    ARENA_ROWS,
    CARD_DATABASE,
    grid_to_screen_pixel,
    hand_slot_to_screen_pixel,
)
from perception.detector import ClashRoyaleDetector
from perception.state_encoder import StateEncoder
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
        yolo_path: Optional[str] = None,
        device_address: str = "127.0.0.1:5555",
        screen_width: int = 540,
        screen_height: int = 960,
        target_hz: float = 5.0,
        show_debug_window: bool = True,
        use_fp16: bool = True,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.target_dt = 1.0 / max(1.0, target_hz)
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
            model_path=yolo_path,
            conf_threshold=0.35,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_fp16=use_fp16,
        )
        self.encoder = StateEncoder()

        # 3. Decision Policy Model
        self.policy = DeckConditionedPolicy(num_card_classes=135, card_emb_dim=16, num_scalars=12).to(self.torch_device)
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
        print("CR-Zero Live Loop Active (Target: 5 Hz). Press Ctrl+C or 'q' to stop.")
        print("=" * 60 + "\n")

        fps_timer = time.time()
        frame_counter = 0
        fps = 0.0

        try:
            while self.running:
                loop_start = time.perf_counter()

                # Step 1: Capture frame
                t_cap0 = time.perf_counter()
                frame = self.capture.grab_frame()
                dt_cap = (time.perf_counter() - t_cap0) * 1000.0

                # Step 2: Vision & Observation parsing
                t_det0 = time.perf_counter()
                obs = self.detector.parse_frame(frame)
                dt_det = (time.perf_counter() - t_det0) * 1000.0

                arena_grid = obs["arena_grid"]
                scalars = obs["scalars"]
                legal_card_mask = obs["legal_card_mask"]
                hand_ids = obs["hand_ids"]
                elixir = obs["elixir"]

                # Step 3: Policy evaluation
                t_pol0 = time.perf_counter()
                # Prepare tensors on GPU
                grid_tensor = torch.from_numpy(arena_grid).unsqueeze(0).to(self.torch_device)
                scalar_tensor = torch.from_numpy(scalars).unsqueeze(0).to(self.torch_device)
                hand_tensor = torch.tensor([hand_ids], dtype=torch.long, device=self.torch_device)
                cmask_tensor = torch.from_numpy(legal_card_mask).unsqueeze(0).to(self.torch_device)

                # Default tile mask for card 0 / general
                default_tile_mask = self.encoder.get_legal_tile_mask(card_id=0)
                tmask_tensor = torch.from_numpy(default_tile_mask).unsqueeze(0).to(self.torch_device)

                # Forward pass
                chosen_slot, chosen_tile, card_conf, tile_conf = self.policy.predict_action(
                    arena_grid=grid_tensor,
                    scalars=scalar_tensor,
                    hand_card_ids=hand_tensor,
                    legal_card_mask=cmask_tensor,
                    legal_tile_mask=tmask_tensor,
                    temperature=0.8,
                    deterministic=False,
                )

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
                        temperature=0.8,
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
                else:
                    action_desc = f"WAIT (Elixir: {elixir:.1f}/10) [Conf: {card_conf*100:.1f}%]"
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CR-Zero Real-Time Live Windows Agent")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/cr_zero_policy_best.pt", help="Policy checkpoint")
    parser.add_argument("--yolo", type=str, default=None, help="YOLO model path (.pt or .onnx)")
    parser.add_argument("--device", type=str, default="127.0.0.1:5555", help="ADB device host:port")
    parser.add_argument("--hz", type=float, default=5.0, help="Inference frequency (Hz)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug visual window")
    args = parser.parse_args()

    agent = CRZeroLiveAgent(
        checkpoint_path=args.checkpoint,
        yolo_path=args.yolo,
        device_address=args.device,
        target_hz=args.hz,
        show_debug_window=not args.no_debug,
    )
    agent.run()


if __name__ == "__main__":
    main()

