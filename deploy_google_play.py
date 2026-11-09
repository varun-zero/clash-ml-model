"""
AlphaStar Clash Royale - Google Play Games PC Deployment Engine
Hardware-level Win32 SendInput controller with:
1. Inverted Y-Axis (tile_y=0 -> Friendly King Tower, tile_y=31 -> Enemy King Tower)
2. Auto-Calculated Centered 9:16 Mobile Viewport (handles 16:9 widescreen pillarboxing)
3. VirtIO-Safe Touch Timings & Smooth Hardware Drag-and-Drop
4. Visual Test-Placement Mode with full 18x32 Grid & Card Slot overlay
"""

import os
import sys
import time
import argparse
import ctypes
from ctypes import wintypes
from typing import Tuple, List, Optional, Dict

import numpy as np
import cv2
import mss
import torch

try:
    import win32gui
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

from cr_alphastar_agent.config import (
    ARENA_WIDTH, ARENA_HEIGHT, TOTAL_TILES, CARDS_CATALOG,
    DEFAULT_DECK, SPELL_CARD_IDS, get_hardware_profile
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.env.crsim_replica_env import CRSIM_NAME_TO_ID
from evaluate_visual import classify_tactical_zone


# --- 1. Windows DPI Awareness & Process Setup ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def attach_desktop():
    """Attaches background thread to the user's interactive Windows desktop."""
    try:
        user32 = ctypes.windll.user32
        hdesk = user32.OpenDesktopW("Default", 0, False, 0x10000000)
        if hdesk:
            user32.SetThreadDesktop(hdesk)
    except Exception:
        pass


# --- 2. Win32 Hardware SendInput Definitions ---
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("mi", MOUSEINPUT),
    ]


# --- 3. Google Play Games Controller ---
class GooglePlayController:
    """
    Hardware-level Win32 SendInput controller for Google Play Games on PC.
    Solves the 'card clicked but not placed' bug via:
    1. Canonical Y-Axis Inversion (tile_y=0 maps to bottom friendly King, not top enemy King)
    2. Centered 9:16 Viewport Box Calculation (ignores Google Play Games 16:9 sidebars)
    3. VirtIO Touch Dwell & Smooth Drag-and-Drop injection
    """

    def __init__(self, window_title: str = "Clash Royale", use_drag: bool = False):
        self.window_title = window_title
        self.use_drag = use_drag
        self.screen_w = max(1, ctypes.windll.user32.GetSystemMetrics(0))
        self.screen_h = max(1, ctypes.windll.user32.GetSystemMetrics(1))
        self.hwnd = self._find_window()
        self.viewport = self._get_mobile_viewport()

        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("[WARNING] Terminal is NOT running as Administrator!")
            print("          Google Play Games may block mouse clicks via Windows UIPI.")
            print("          If clicks are not received, launch PowerShell / VS Code as Administrator.\n")
        else:
            print("[Admin Check] Running with elevated Administrator privileges.")

    def _find_window(self) -> int:
        attach_desktop()
        candidates = []

        def enum_cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).strip()
                # Exclude IDE/tools titled ClashBot
                if "clashbot" in title.lower():
                    return
                if self.window_title.lower() in title.lower():
                    rect = win32gui.GetWindowRect(hwnd)
                    w = rect[2] - rect[0]
                    h = rect[3] - rect[1]
                    if w > 300 and h > 400:
                        candidates.append((hwnd, title, 10))  # High priority
                elif "google play games" in title.lower():
                    rect = win32gui.GetWindowRect(hwnd)
                    w = rect[2] - rect[0]
                    h = rect[3] - rect[1]
                    if w > 300 and h > 400:
                        candidates.append((hwnd, title, 5))   # Lower priority

        win32gui.EnumWindows(enum_cb, None)
        if not candidates:
            raise RuntimeError(f"Could not find '{self.window_title}' window. Is Google Play Games open?")

        candidates.sort(key=lambda x: x[2], reverse=True)
        target_hwnd, title, _ = candidates[0]
        print(f"[GooglePlayController] Hooked window: '{title}' (HWND: {target_hwnd})")
        return target_hwnd

    def _get_mobile_viewport(self) -> Dict:
        """
        Calculates the exact 9:16 phone screen inside Google Play Games' 16:9 window,
        plus the inner 18x32 arena grass bounding box and 4 hand card centers.
        """
        attach_desktop()
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        pt_tl = win32gui.ClientToScreen(self.hwnd, (left, top))
        pt_br = win32gui.ClientToScreen(self.hwnd, (right, bottom))

        win_w = max(100, pt_br[0] - pt_tl[0])
        win_h = max(100, pt_br[1] - pt_tl[1])

        # Clash Royale runs in a centered 9:16 portrait aspect ratio inside the window
        mobile_h = win_h
        mobile_w = int(mobile_h * (9.0 / 16.0))
        if mobile_w > win_w:
            mobile_w = win_w
            mobile_h = int(mobile_w * (16.0 / 9.0))

        mob_left = pt_tl[0] + (win_w - mobile_w) // 2
        mob_top = pt_tl[1] + (win_h - mobile_h) // 2

        # Normalized Clash Royale 9:16 layout bounds:
        # Arena grass spans X: [6.5% .. 93.5%] of mobile_w, Y: [11.5% .. 76.5%] of mobile_h
        arena_left = int(mob_left + mobile_w * 0.065)
        arena_right = int(mob_left + mobile_w * 0.935)
        arena_top = int(mob_top + mobile_h * 0.115)     # Enemy back wall (tile_y = 31)
        arena_bottom = int(mob_top + mobile_h * 0.765)  # Friendly back wall (tile_y = 0)

        # 4 Hand Card Slots sit at Y = 88.5% of mobile_h, X = [27%, 45%, 63%, 81%]
        card_y = int(mob_top + mobile_h * 0.885)
        card_x_ratios = [0.27, 0.45, 0.63, 0.81]
        card_centers = [(int(mob_left + mobile_w * rx), card_y) for rx in card_x_ratios]

        return {
            "win_left": pt_tl[0], "win_top": pt_tl[1],
            "win_w": win_w, "win_h": win_h,
            "mob_left": mob_left, "mob_top": mob_top,
            "mob_w": mobile_w, "mob_h": mobile_h,
            "arena_left": arena_left, "arena_right": arena_right,
            "arena_top": arena_top, "arena_bottom": arena_bottom,
            "arena_w": arena_right - arena_left,
            "arena_h": arena_bottom - arena_top,
            "card_centers": card_centers,
        }

    def tile_to_screen_xy(self, tile_x: int, tile_y: int, is_spell: bool = False) -> Tuple[int, int]:
        """
        Converts crsim canonical (tile_x in 0..17, tile_y in 0..31) to Windows screen pixels.
        CRITICAL FIX: In crsim, tile_y=0 is Friendly Bottom and tile_y=31 is Enemy Top.
        On screen, arena_top is Enemy Top and arena_bottom is Friendly Bottom.
        Inverting norm_y = 1.0 - ((tile_y + 0.5) / 32.0) ensures friendly troop deployments
        click the bottom half of the arena instead of illegal enemy territory!
        """
        vp = self.viewport
        tile_x = int(np.clip(tile_x, 0, 17))

        # Clamp non-spell troops to friendly half (0..14) so border jitter never blocks placement
        if not is_spell:
            tile_y = int(np.clip(tile_y, 0, 14))
        else:
            tile_y = int(np.clip(tile_y, 0, 31))

        # X: 0 = Left Lane, 17 = Right Lane
        norm_x = (tile_x + 0.5) / 18.0
        # Y: INVERT so tile_y=0 -> bottom of arena (Friendly back), tile_y=31 -> top (Enemy back)
        norm_y = 1.0 - ((tile_y + 0.5) / 32.0)

        px = int(vp["arena_left"] + norm_x * vp["arena_w"])
        py = int(vp["arena_top"] + norm_y * vp["arena_h"])
        return px, py

    def _send_mouse(self, x: int, y: int, flags: int):
        """Sends absolute hardware mouse event via Windows SendInput."""
        abs_x = int((x * 65535) / max(1, self.screen_w - 1))
        abs_y = int((y * 65535) / max(1, self.screen_h - 1))
        mi = MOUSEINPUT(
            dx=abs_x, dy=abs_y,
            mouseData=0,
            dwFlags=flags | MOUSEEVENTF_ABSOLUTE,
            time=0,
            dwExtraInfo=None
        )
        inp = INPUT(type=INPUT_MOUSE, mi=mi)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def ensure_foreground(self):
        """Ensures Google Play Games window has foreground keyboard/mouse focus."""
        if self.hwnd and HAS_WIN32:
            try:
                fg = win32gui.GetForegroundWindow()
                if fg != self.hwnd:
                    win32gui.ShowWindow(self.hwnd, 5)  # SW_SHOW
                    win32gui.SetForegroundWindow(self.hwnd)
                    time.sleep(0.05)
            except Exception:
                pass

    def hardware_tap(self, x: int, y: int, move_delay: float = 0.035, hold_delay: float = 0.045):
        """
        VirtIO-safe tap:
        1. Move cursor to target first and wait 35ms so Google Play Games updates hover coordinates
        2. Press Left Button Down and hold for 45ms so Android registers touch down
        3. Release Left Button Up
        """
        self._send_mouse(x, y, MOUSEEVENTF_MOVE)
        time.sleep(move_delay)
        self._send_mouse(x, y, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTDOWN)
        time.sleep(hold_delay)
        self._send_mouse(x, y, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTUP)

    def hardware_drag(self, start_xy: Tuple[int, int], end_xy: Tuple[int, int], steps: int = 8, duration: float = 0.14):
        """
        Alternative Drag-and-Drop placement (drags card directly from hand onto arena tile).
        """
        sx, sy = start_xy
        ex, ey = end_xy
        self._send_mouse(sx, sy, MOUSEEVENTF_MOVE)
        time.sleep(0.03)
        self._send_mouse(sx, sy, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.04)

        for i in range(1, steps + 1):
            t = i / float(steps)
            cx = int(sx + (ex - sx) * t)
            cy = int(sy + (ey - sy) * t)
            self._send_mouse(cx, cy, MOUSEEVENTF_MOVE)
            time.sleep(duration / steps)

        time.sleep(0.03)
        self._send_mouse(ex, ey, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTUP)

    def deploy_card(self, card_slot: int, tile_x: int, tile_y: int, is_spell: bool = False) -> bool:
        """
        Deploys card from slot 0..3 onto arena tile (tile_x, tile_y).
        Uses dwell delays or drag-and-drop to guarantee placement.
        """
        if card_slot == 4 or not (0 <= card_slot <= 3):
            return False

        self.ensure_foreground()
        self.viewport = self._get_mobile_viewport()

        card_px, card_py = self.viewport["card_centers"][card_slot]
        tile_px, tile_py = self.tile_to_screen_xy(tile_x, tile_y, is_spell=is_spell)

        action_desc = f"Slot {card_slot} -> Tile ({tile_x:2d}, {tile_y:2d}) | Screen: ({card_px}, {card_py}) -> ({tile_px}, {tile_py})"

        if self.use_drag:
            self.hardware_drag((card_px, card_py), (tile_px, tile_py))
            print(f"[HARDWARE DRAG] {action_desc}")
        else:
            # Step 1: Click card in hand
            self.hardware_tap(card_px, card_py)
            # Step 2: Wait 95ms for Clash Royale card-select UI animation to complete
            time.sleep(0.095)
            # Step 3: Click target arena tile
            self.hardware_tap(tile_px, tile_py)
            print(f"[HARDWARE TAP]  {action_desc}")

        return True


# --- 4. Frame Capture & Perception ---
class GooglePlayCapture:
    """Captures low-latency frames from Google Play Games PC window via MSS."""

    def __init__(self, hwnd: int):
        self.hwnd = hwnd
        self.sct = mss.MSS()

    def capture_frame(self) -> Optional[np.ndarray]:
        attach_desktop()
        try:
            left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
            pt_tl = win32gui.ClientToScreen(self.hwnd, (left, top))
            pt_br = win32gui.ClientToScreen(self.hwnd, (right, bottom))
            bbox = {
                "left": pt_tl[0],
                "top": pt_tl[1],
                "width": max(100, pt_br[0] - pt_tl[0]),
                "height": max(100, pt_br[1] - pt_tl[1]),
            }
            raw = self.sct.grab(bbox)
            frame = np.array(raw)[:, :, :3]
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
        except Exception as e:
            return None


# --- 5. Visual Grid Overlay & Placement Verification Mode ---
def run_test_placement(controller: GooglePlayController, capture: GooglePlayCapture, output_dir: str = "."):
    """
    CLI Mode: --test-placement
    Captures the Google Play Games window, overlays:
    - Centered 9:16 mobile border (Yellow)
    - 18x32 Canonical Arena Grid (Green)
    - River & Bridge separator line (Cyan)
    - Friendly vs Enemy zone markers
    - 4 Hand Card Slot circles (Magenta)
    - Test target points: (9, 3) Friendly King Backline & (3, 14) Left Bridge
    Then executes placement test of Card Slot 0 at both targets!
    """
    print("\n" + "=" * 75)
    print("      Google Play Games PC - Grid Calibration & Placement Test")
    print("=" * 75)

    controller.ensure_foreground()
    time.sleep(0.2)

    frame = capture.capture_frame()
    if frame is None:
        print("[Error] Failed to capture window frame!")
        return

    vp = controller.viewport
    win_left = vp["win_left"]
    win_top = vp["win_top"]

    # Frame-relative coordinates
    f_mob_l = vp["mob_left"] - win_left
    f_mob_t = vp["mob_top"] - win_top
    f_mob_w = vp["mob_w"]
    f_mob_h = vp["mob_h"]

    f_arena_l = vp["arena_left"] - win_left
    f_arena_r = vp["arena_right"] - win_left
    f_arena_t = vp["arena_top"] - win_top
    f_arena_b = vp["arena_bottom"] - win_top
    f_arena_w = vp["arena_w"]
    f_arena_h = vp["arena_h"]

    overlay = frame.copy()

    # 1. Draw 9:16 Mobile Viewport Border
    cv2.rectangle(overlay, (f_mob_l, f_mob_t), (f_mob_l + f_mob_w, f_mob_t + f_mob_h), (0, 255, 255), 2)
    cv2.putText(overlay, "9:16 Viewport", (f_mob_l + 10, f_mob_t + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 2. Draw 18x32 Arena Grid
    for gx in range(19):
        x = int(f_arena_l + (gx / 18.0) * f_arena_w)
        cv2.line(overlay, (x, f_arena_t), (x, f_arena_b), (0, 180, 0), 1)

    for gy in range(33):
        # gy=0 is top on screen (Enemy back), gy=32 is bottom (Friendly back)
        y = int(f_arena_t + (gy / 32.0) * f_arena_h)
        # Highlight river boundary (tile_y = 15/16)
        if gy == 16:
            cv2.line(overlay, (f_arena_l, y), (f_arena_r, y), (255, 255, 0), 2)
            cv2.putText(overlay, "--- RIVER (y=15/16) ---", (f_arena_l + 20, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.line(overlay, (f_arena_l, y), (f_arena_r, y), (0, 180, 0), 1)

    # 3. Draw 4 Card Hand Slots
    for i, (cx, cy) in enumerate(vp["card_centers"]):
        fcx = cx - win_left
        fcy = cy - win_top
        cv2.circle(overlay, (fcx, fcy), 22, (255, 0, 255), 2)
        cv2.putText(overlay, f"Slot {i}", (fcx - 22, fcy - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # 4. Draw Test Target Points
    test_targets = [
        (0, 9, 3, "Friendly King Pocket (9, 3)"),
        (0, 3, 14, "Left Bridge Lane (3, 14)"),
    ]

    for slot, tx, ty, desc in test_targets:
        sx, sy = controller.tile_to_screen_xy(tx, ty, is_spell=False)
        fsx = sx - win_left
        fsy = sy - win_top
        cv2.circle(overlay, (fsx, fsy), 14, (0, 0, 255), -1)
        cv2.putText(overlay, f"{desc}: ({tx},{ty})", (fsx + 18, fsy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save visualization artifact
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "google_play_calibration_grid.png")
    cv2.imwrite(out_path, overlay)
    print(f"[Visualization Saved] {os.path.abspath(out_path)}")

    # 5. Execute Live Placement Test
    print("\n--- Executing Live Test Placements on Screen ---")
    print("Testing Placement 1: Slot 0 -> Friendly King Pocket (9, 3)...")
    controller.deploy_card(card_slot=0, tile_x=9, tile_y=3, is_spell=False)
    time.sleep(1.0)

    print("Testing Placement 2: Slot 0 -> Left Bridge (3, 14)...")
    controller.deploy_card(card_slot=0, tile_x=3, tile_y=14, is_spell=False)
    time.sleep(0.5)

    print("\n[Test-Placement Complete] Verify the card was dropped on the arena grass!")


# --- 6. Full Autonomous Live Battle Engine ---
def run_live_matches(checkpoint: str, matches: int, use_drag: bool, device_str: Optional[str] = None):
    """Runs autonomous Clash Royale matches using the hardware-level SendInput engine."""
    attach_desktop()

    device = torch.device(device_str if device_str else get_hardware_profile()["device"])
    print("=" * 75)
    print("      AlphaStar Clash Royale - Google Play Games PC Live Runner")
    print("=" * 75)
    print(f"[Device]         {device}")
    print(f"[Checkpoint]     {os.path.abspath(checkpoint)}")
    print(f"[Target Matches] {matches}")
    print(f"[Input Method]   {'Hardware Drag-and-Drop' if use_drag else 'Hardware Dwell-Tap'}\n")

    # Load Model
    model = AlphaStarCRNet().to(device)
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"[Model Loaded] Checkpoint loaded successfully from {checkpoint}")
    model.eval()

    controller = GooglePlayController(window_title="Clash Royale", use_drag=use_drag)
    capture = GooglePlayCapture(hwnd=controller.hwnd)

    completed = 0
    in_battle = False
    last_action_time = 0.0
    action_interval = 0.40  # 2.5 Hz cadence

    tracked_hand = list(DEFAULT_DECK[:4])
    tracked_next = DEFAULT_DECK[4] if len(DEFAULT_DECK) > 4 else DEFAULT_DECK[0]
    current_elixir = 5.0
    match_deployments = []

    print("[Runner Ready] Entering autonomous match loop...\n")

    while completed < matches:
        attach_desktop()
        frame = capture.capture_frame()
        if frame is None:
            time.sleep(0.2)
            continue

        h, w = frame.shape[:2]
        vp = controller.viewport
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Detect Post-Match Play Again (Yellow) and OK (Blue) in bottom 25% of viewport
        bot_crop = hsv[int(h * 0.78):, :]
        yellow_mask = cv2.inRange(bot_crop, np.array([15, 170, 170]), np.array([35, 255, 255]))
        blue_mask = cv2.inRange(bot_crop, np.array([95, 140, 140]), np.array([125, 255, 255]))

        # Play Again Check
        y_cnts, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found_play_again = False
        for c in y_cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw > 60 and bh > 25:
                # Play Again button sits in bottom center-left
                cx = vp["win_left"] + bx + bw // 2
                cy = vp["win_top"] + int(h * 0.78) + by + bh // 2
                if by > int(h * 0.08):
                    found_play_again = True
                    if in_battle:
                        completed += 1
                        print(f"\n🏁 MATCH {completed}/{matches} FINISHED! Cards deployed: {len(match_deployments)}")
                        in_battle = False
                        match_deployments = []

                    if completed < matches:
                        print(f"[Autopilot] Clicking 'Play Again' at ({cx}, {cy}) for Match {completed + 1}...")
                        controller.hardware_tap(cx, cy)
                        time.sleep(2.5)
                    break

        if found_play_again:
            continue

        # 2. Live Battle Loop
        if not in_battle:
            in_battle = True
            current_elixir = 5.0
            tracked_hand = list(DEFAULT_DECK[:4])
            match_deployments = []
            print(f"\n⚔️  LIVE BATTLE {completed + 1}/{matches} STARTED!")

        now = time.time()
        if now - last_action_time >= action_interval:
            last_action_time = now
            current_elixir = min(10.0, current_elixir + 0.14)

            # Build simple observation tensors for policy
            scalars = torch.tensor([[current_elixir / 10.0, 0.5, 0.8, 0.0]], dtype=torch.float32, device=device)
            hand_ids = [CRSIM_NAME_TO_ID.get(c, 1) for c in tracked_hand] + [CRSIM_NAME_TO_ID.get(tracked_next, 1)]
            hand_tensor = torch.tensor([hand_ids], dtype=torch.int64, device=device)
            entities = torch.zeros((1, 32, 7), dtype=torch.float32, device=device)
            entity_mask = torch.ones((1, 32), dtype=torch.bool, device=device)
            # Unmask crown towers (first 6 slots)
            entity_mask[0, :6] = False
            arena_grid = torch.zeros((1, 6, ARENA_HEIGHT, ARENA_WIDTH), dtype=torch.float32, device=device)

            card_mask = torch.zeros((1, 5), dtype=torch.bool, device=device)
            for i, cid in enumerate(hand_ids[:4]):
                cost = CARDS_CATALOG.get(cid, {}).get("cost", 3)
                if current_elixir >= cost:
                    card_mask[0, i] = True
            card_mask[0, 4] = True  # Wait is legal

            tile_mask = torch.zeros((1, TOTAL_TILES), dtype=torch.bool, device=device)
            for y in range(15):
                for x in range(ARENA_WIDTH):
                    tile_mask[0, y * ARENA_WIDTH + x] = True

            obs = {
                "scalars": scalars,
                "hand_cards": hand_tensor,
                "entities": entities,
                "entity_mask": entity_mask,
                "arena_grid": arena_grid,
                "card_mask": card_mask,
                "tile_mask": tile_mask,
            }

            with torch.no_grad():
                actions, _, _, values = model.get_action_and_value(obs)
                chosen_card = actions[0, 0].item()
                chosen_tile = actions[0, 1].item()
                val = values[0].item() if values.dim() == 1 else values[0, 0].item()

            tile_x = chosen_tile % ARENA_WIDTH
            tile_y = chosen_tile // ARENA_WIDTH

            # Elixir Banking Override: Never sit at max elixir doing nothing
            if chosen_card == 4 and current_elixir >= 6.5:
                for slot in range(4):
                    cid = hand_ids[slot]
                    cost = CARDS_CATALOG.get(cid, {}).get("cost", 3)
                    if current_elixir >= cost:
                        chosen_card = slot
                        tile_y = 12
                        tile_x = 4 if tile_x < 9 else 13
                        break

            if chosen_card != 4:
                card_name = tracked_hand[chosen_card]
                cid = CRSIM_NAME_TO_ID.get(card_name, 1)
                is_spell = cid in SPELL_CARD_IDS

                deployed = controller.deploy_card(
                    card_slot=chosen_card,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    is_spell=is_spell
                )
                if deployed:
                    cost = CARDS_CATALOG.get(cid, {}).get("cost", 3)
                    current_elixir = max(0.0, current_elixir - cost)
                    # Rotate hand
                    tracked_hand[chosen_card] = tracked_next
                    cycle = [c for c in DEFAULT_DECK if c not in tracked_hand]
                    tracked_next = cycle[0] if cycle else DEFAULT_DECK[0]

                    zone = classify_tactical_zone(float(tile_x), float(tile_y))
                    match_deployments.append((card_name, (tile_x, tile_y)))
                    print(f"  [Play] {card_name:<12} -> ({tile_x:2d}, {tile_y:2d}) [{zone:<20}] | V(s): {val:+.2f} | Elixir: {current_elixir:.1f}")

        time.sleep(0.05)

    print("\n" + "=" * 75)
    print("                 LIVE 5-GAME SESSION COMPLETED")
    print("=" * 75)


# --- 7. Main CLI Entrypoint ---
def main():
    parser = argparse.ArgumentParser(description="AlphaStar Clash Royale - Google Play Games PC Engine")
    parser.add_argument("--test-placement", action="store_true", help="Draw grid overlay and test placing card slot 0")
    parser.add_argument("--live", action="store_true", help="Run live autonomous battle session")
    parser.add_argument("--matches", type=int, default=5, help="Number of matches to play (default: 5)")
    parser.add_argument("--drag", action="store_true", help="Use hardware drag-and-drop instead of dwell tap")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt", help="Model checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda / cpu)")
    parser.add_argument("--window", type=str, default="Clash Royale", help="Window title substring")
    args = parser.parse_args()

    try:
        controller = GooglePlayController(window_title=args.window, use_drag=args.drag)
        capture = GooglePlayCapture(hwnd=controller.hwnd)
    except Exception as e:
        print(f"[Error] Failed to initialize Google Play Games controller: {e}")
        return

    if args.test_placement:
        brain_dir = r"C:\Users\varun\.gemini\antigravity\brain\42ce9bd6-0ed7-406b-aa50-7c85b510c081"
        run_test_placement(controller, capture, output_dir=brain_dir)
    elif args.live:
        run_live_matches(checkpoint=args.checkpoint, matches=args.matches, use_drag=args.drag, device_str=args.device)
    else:
        # Default: Run test placement to verify calibration
        brain_dir = r"C:\Users\varun\.gemini\antigravity\brain\42ce9bd6-0ed7-406b-aa50-7c85b510c081"
        run_test_placement(controller, capture, output_dir=brain_dir)


if __name__ == "__main__":
    main()
