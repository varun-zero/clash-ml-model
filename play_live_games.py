"""
AlphaStar Clash Royale - Automated 5 Live Games Runner (Google Play Games PC)

Connects to the official Clash Royale window via Windows Native Mouse,
detects match lifecycle states (Home / Matchmaking / Live Battle / Post-Match / Rewards),
deploys cards with AlphaStarCRNet at 2.5 Hz cadence, and plays 5 full consecutive matches.
"""

import os
import sys
import time
import argparse
import ctypes
import numpy as np
import cv2
import torch
import pyautogui

from cr_alphastar_agent.config import (
    ARENA_WIDTH, ARENA_HEIGHT, TOTAL_TILES, CARDS_CATALOG,
    DEFAULT_DECK, get_hardware_profile
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.env.crsim_replica_env import CRSIM_NAME_TO_ID
from deploy_bluestacks import (
    WindowCapture, PerceptionAdapter, WindowsMouseController, draw_hud
)
from evaluate_visual import classify_tactical_zone


def attach_desktop():
    try:
        user32 = ctypes.windll.user32
        hdesk = user32.OpenDesktopW("Default", 0, False, 0x10000000)
        if hdesk:
            user32.SetThreadDesktop(hdesk)
    except Exception:
        pass


def detect_game_state(frame: np.ndarray, capture: WindowCapture):
    """
    Analyzes the captured frame to identify the game screen:
    - 'POST_MATCH_PLAY_AGAIN': (x, y) screen coords of 'Play Again' button
    - 'POST_MATCH_OK': (x, y) screen coords of 'OK' button
    - 'POST_MATCH_BONUS': (x, y) screen coords of Bonus collect button
    - 'HOME_BATTLE': (x, y) screen coords of yellow 'Battle' button
    - 'SEARCHING': Waiting for opponent
    - 'IN_BATTLE': Live arena match in progress
    """
    h, w = frame.shape[:2]
    left = capture.bbox["left"]
    top = capture.bbox["top"]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. Check for Play Again (yellow) and OK (blue) buttons at bottom third
    bottom_hsv = hsv[int(h * 0.75):, :]
    yellow_mask = cv2.inRange(bottom_hsv, np.array([15, 170, 170]), np.array([35, 255, 255]))
    blue_mask = cv2.inRange(bottom_hsv, np.array([95, 140, 140]), np.array([125, 255, 255]))

    # Search for Play Again (yellow)
    y_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in y_contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > 50 and bh > 25:
            cx = left + bx + bw // 2
            cy = top + int(h * 0.75) + by + bh // 2
            # Check if this is the bottom 'Play Again' button (by > h*0.1) or bonus chest
            if by > int(h * 0.12):
                return "POST_MATCH_PLAY_AGAIN", (cx, cy)
            else:
                return "POST_MATCH_BONUS", (cx, cy)

    # Search for OK (blue)
    b_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in b_contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > 50 and bh > 25:
            cx = left + bx + bw // 2
            cy = top + int(h * 0.75) + by + bh // 2
            return "POST_MATCH_OK", (cx, cy)

    # 2. Check for Home Screen 'Battle' button (prominent yellow button at bottom center)
    btn_crop = hsv[int(h * 0.88):int(h * 0.99), int(w * 0.35):int(w * 0.75)]
    if btn_crop.size > 0:
        home_y_mask = cv2.inRange(btn_crop, np.array([15, 150, 150]), np.array([35, 255, 255]))
        if np.sum(home_y_mask > 0) / home_y_mask.size > 0.30:
            cx = left + int(w * 0.55)
            cy = top + int(h * 0.94)
            return "HOME_BATTLE", (cx, cy)

    # 3. Check for 'Searching for opponent...' or loading screen (Cancel button: red in middle)
    mid_hsv = hsv[int(h * 0.60):int(h * 0.80), int(w * 0.25):int(w * 0.75)]
    if mid_hsv.size > 0:
        red_mask1 = cv2.inRange(mid_hsv, np.array([0, 150, 150]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(mid_hsv, np.array([170, 150, 150]), np.array([180, 255, 255]))
        red_ratio = (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)) / mid_hsv.size
        if red_ratio > 0.15:
            return "SEARCHING", None

    # 4. Default to IN_BATTLE
    return "IN_BATTLE", None


def run_5_games(checkpoint: str = "checkpoints/latest.pt", target_matches: int = 5, device_str: str = None):
    attach_desktop()

    if device_str is None:
        hw = get_hardware_profile()
        device = torch.device(hw["device"])
    else:
        device = torch.device(device_str)

    print("=" * 75)
    print("      AlphaStar Clash Royale - Automated 5 Live Matches Autopilot")
    print("=" * 75)
    print(f"[Device]         {device}")
    print(f"[Checkpoint]     {os.path.abspath(checkpoint)}")
    print(f"[Target Matches] {target_matches}")
    print(f"[Target Window]  Google Play Games PC ('Clash Royale')\n")

    # Load Model
    model = AlphaStarCRNet().to(device)
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            step = ckpt.get("global_step", 0)
            upd = ckpt.get("update_count", 0)
            print(f"[Model Loaded] Checkpoint Step {step} | Update {upd}")
        else:
            model.load_state_dict(ckpt)
    model.eval()

    capture = WindowCapture()
    if not capture.hwnd:
        print("[Error] Clash Royale window not found! Please ensure Google Play Games is running.")
        return

    controller = WindowsMouseController(hwnd=capture.hwnd, bbox=capture.bbox, dry_run=False)
    perception = PerceptionAdapter(device=device)

    completed_matches = 0
    match_stats = []
    current_match_deployments = []
    current_match_start = None
    in_active_battle = False
    last_action_time = 0.0
    action_interval = 0.40  # 2.5 Hz cadence

    print(f"[Autopilot Ready] Hooked window HWND {capture.hwnd}. Starting 5-match loop...\n")

    while completed_matches < target_matches:
        attach_desktop()
        frame = capture.capture_frame()
        if frame is None:
            time.sleep(0.2)
            continue

        h, w = frame.shape[:2]
        controller.set_screen_dimensions(w, h)
        if capture.bbox:
            controller.update_bbox(capture.bbox)

        # Detect current game state
        state, coords = detect_game_state(frame, capture)

        if state == "POST_MATCH_BONUS":
            print(f"[State] Bonus Reward popup detected. Dismissing at {coords}...")
            pyautogui.click(coords[0], coords[1])
            time.sleep(1.0)
            continue

        elif state == "POST_MATCH_PLAY_AGAIN":
            if in_active_battle:
                # Conclude previous match
                match_duration = time.time() - (current_match_start or time.time())
                completed_matches += 1
                match_info = {
                    "match": completed_matches,
                    "duration": match_duration,
                    "deployments": len(current_match_deployments),
                    "cards": list(current_match_deployments)
                }
                match_stats.append(match_info)
                print(f"\n=======================================================")
                print(f"🏁 MATCH {completed_matches}/{target_matches} FINISHED!")
                print(f"   Duration: {match_duration:.1f}s | Cards Deployed: {len(current_match_deployments)}")
                print(f"=======================================================\n")
                in_active_battle = False
                current_match_deployments = []
                current_match_start = None

            if completed_matches < target_matches:
                print(f"[Autopilot] Clicking 'Play Again' at {coords} to start Match {completed_matches + 1}...")
                pyautogui.click(coords[0], coords[1])
                time.sleep(2.0)
            else:
                print(f"[Autopilot] All {target_matches} matches complete! Clicking OK to return to Home.")
                # Fall through to click OK
                pass

        elif state == "POST_MATCH_OK":
            if in_active_battle:
                match_duration = time.time() - (current_match_start or time.time())
                completed_matches += 1
                match_info = {
                    "match": completed_matches,
                    "duration": match_duration,
                    "deployments": len(current_match_deployments),
                    "cards": list(current_match_deployments)
                }
                match_stats.append(match_info)
                print(f"\n=======================================================")
                print(f"🏁 MATCH {completed_matches}/{target_matches} FINISHED!")
                print(f"   Duration: {match_duration:.1f}s | Cards Deployed: {len(current_match_deployments)}")
                print(f"=======================================================\n")
                in_active_battle = False
                current_match_deployments = []
                current_match_start = None

            if completed_matches < target_matches:
                print(f"[Autopilot] On OK screen, clicking at {coords} to return to Home...")
                pyautogui.click(coords[0], coords[1])
                time.sleep(2.0)
            else:
                print(f"[Autopilot] All {target_matches} matches finished successfully.")
                break

        elif state == "HOME_BATTLE":
            if not in_active_battle and completed_matches < target_matches:
                print(f"[Autopilot] On Home Screen. Clicking 'Battle' at {coords} to start Match {completed_matches + 1}...")
                pyautogui.click(coords[0], coords[1])
                time.sleep(2.5)

        elif state == "SEARCHING":
            print(f"[State] Matchmaking: Searching for opponent...", end="\r", flush=True)
            time.sleep(1.0)

        elif state == "IN_BATTLE":
            if not in_active_battle:
                in_active_battle = True
                current_match_start = time.time()
                current_match_deployments = []
                # Calibrate column for this match
                perception.calibrate_game_column(frame)
                print(f"\n⚔️  LIVE BATTLE {completed_matches + 1}/{target_matches} IN PROGRESS!")
                print(f"   Calibrated Arena bounds: Left {perception.arena_box['left']:.2f}, Right {perception.arena_box['right']:.2f}")

            now = time.time()
            if now - last_action_time >= action_interval:
                last_action_time = now

                # Perception & Observation tensor building
                obs = perception.frame_to_obs(frame)

                # Neural Network Decision
                with torch.no_grad():
                    actions, _, _, values = model.get_action_and_value(obs)
                    chosen_card = actions[0, 0].item()
                    chosen_tile = actions[0, 1].item()
                    val = values[0].item() if values.dim() == 1 else values[0, 0].item()

                tile_x = chosen_tile % ARENA_WIDTH
                tile_y = chosen_tile // ARENA_WIDTH

                # Execute action via direct OS mouse clicks
                if chosen_card != 4:
                    card_name = perception.tracked_hand[chosen_card] if chosen_card < len(perception.tracked_hand) else f"Card {chosen_card}"
                    zone = classify_tactical_zone(float(tile_x), float(tile_y))
                    deployed = controller.deploy_card(
                        card_slot=chosen_card,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        arena_box=perception.arena_box,
                        hand_xs=perception.hand_xs,
                        hand_box_y=perception.hand_box_y
                    )
                    perception.update_hand_on_play(chosen_card)
                    current_match_deployments.append((card_name, (tile_x, tile_y), zone, val))
                    print(f"  [Action] {card_name:<12} -> Tile ({tile_x:2d}, {tile_y:2d}) [{zone:<24}] | V(s): {val:+.2f} | Elixir: {perception.current_elixir:.1f}")
                else:
                    # Bank Elixir / Wait tick
                    pass

            time.sleep(0.05)

    print("\n" + "=" * 75)
    print("                 LIVE 5-GAME SESSION COMPLETED")
    print("=" * 75)
    print(f"Total Matches Played:    {completed_matches}")
    for m in match_stats:
        print(f"  Match {m['match']}: Duration {m['duration']:.1f}s | Deployments: {m['deployments']}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaStar Clash Royale - Automated 5 Live Games")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--matches", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_5_games(checkpoint=args.checkpoint, target_matches=args.matches, device_str=args.device)
