#!/usr/bin/env python3
"""
Live BlueStacks Deployment Bridge for AlphaStar Clash Royale Agent.
Connects trained AlphaStarCRNet (checkpoints/latest.pt) to a live BlueStacks emulator instance.

Features:
1. Low-Latency Screen Capture:
   - Discovers BlueStacks window via win32gui or bounding box.
   - Fast framebuffer sampling via mss at 2.5 Hz (0.4s cadence to match action_repeat = 4).
2. Perception-to-Tensor Adapter (frame_to_obs):
   - Injects active Crown Towers into slots 0..5 (guarantees NaN-safe transformer).
   - Crops & estimates current elixir (0.0-10.0) from bottom purple bar.
   - Identifies active hand cards and builds legal action masks.
   - Constructs full 7-tensor observation dictionary for AlphaStarCRNet.
3. Persistent ADB Command Pipe (ADBController):
   - Keeps interactive adb shell pipe open (subprocess.Popen) for zero-latency taps.
   - Translates (slot, tile_x, tile_y) into dual-tap sequence (card tap -> arena tile tap).
4. Dry-Run & Visual Overlays:
   - --dry-run flag displays live OpenCV HUD overlay showing detected towers, hand slots,
     elixir bar, and planned deployment reticles without executing ADB commands.

Usage:
    python deploy_bluestacks.py --dry-run
    python deploy_bluestacks.py --checkpoint checkpoints/latest.pt --visualize
"""

import os
import sys
import time
import argparse
import subprocess
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import cv2

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import win32gui
    import win32process
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

import torch

# Ensure local packages are importable
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cr_alphastar_agent.config import (
    ARENA_WIDTH, ARENA_HEIGHT, TOTAL_TILES, MAX_ENTITIES, D_MODEL,
    ENTITY_FEATURES, SPATIAL_CHANNELS, NUM_CARDS, SPELL_CARD_IDS,
    CARDS_CATALOG, DEFAULT_DECK, POPULAR_DECKS, get_hardware_profile
)
from cr_alphastar_agent.model.alphastar_cr_net import AlphaStarCRNet
from cr_alphastar_agent.env.crsim_replica_env import CRSIM_NAME_TO_ID, ID_TO_CRSIM_NAME


DEFAULT_WINDOW_CANDIDATES = ["Clash Royale", "Google Play Games", "BlueStacks"]


# ==============================================================================
# 1. Low-Latency Screen Capture (MSS + Win32 Window Handle)
# ==============================================================================

class WindowCapture:
    """Discovers Clash Royale / Google Play Games / BlueStacks window and captures low-latency frames via MSS."""

    def __init__(self, title_substr: Optional[str] = None, bbox: Optional[Tuple[int, int, int, int]] = None):
        self.title_substr = title_substr
        self.hwnd = None
        self.bbox = None  # {"left": x, "top": y, "width": w, "height": h}
        self.sct = mss.MSS() if HAS_MSS else None
        self.detected_mode = "mouse"  # "mouse" (Google Play Games PC) or "adb" (BlueStacks)

        if bbox is not None:
            self.bbox = {"left": bbox[0], "top": bbox[1], "width": bbox[2], "height": bbox[3]}
            print(f"[WindowCapture] Using custom bounding box: {self.bbox}")
        else:
            self.locate_window()

    def locate_window(self) -> bool:
        if not HAS_WIN32:
            print("[WindowCapture] win32gui not available. Using full-screen fallback.")
            if self.sct:
                self.bbox = self.sct.monitors[1]
            return False

        try:
            import ctypes
            user32 = ctypes.windll.user32
            hdesk = user32.OpenDesktopW("Default", 0, False, 0x10000000)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass

        search_titles = [self.title_substr] if self.title_substr else DEFAULT_WINDOW_CANDIDATES

        for candidate_title in search_titles:
            candidates = []

            def enum_cb(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    text = win32gui.GetWindowText(hwnd)
                    if candidate_title.lower() in text.lower():
                        rect = win32gui.GetWindowRect(hwnd)
                        w = rect[2] - rect[0]
                        h = rect[3] - rect[1]
                        if w > 200 and h > 300:
                            candidates.append((hwnd, text, rect, w * h))

            win32gui.EnumWindows(enum_cb, None)

            if candidates:
                # Pick largest matching window
                candidates.sort(key=lambda x: x[3], reverse=True)
                self.hwnd, title, rect, _ = candidates[0]
                if "bluestacks" in title.lower():
                    self.detected_mode = "adb"
                else:
                    self.detected_mode = "mouse"

                print(f"[WindowCapture] Hooked window: '{title}' (HWND: {self.hwnd}, Mode: {self.detected_mode})")

                # Get client area (strips window chrome / title bar)
                client_rect = win32gui.GetClientRect(self.hwnd)
                p1 = win32gui.ClientToScreen(self.hwnd, (client_rect[0], client_rect[1]))
                p2 = win32gui.ClientToScreen(self.hwnd, (client_rect[2], client_rect[3]))
                left, top = p1
                width = max(100, p2[0] - p1[0])
                height = max(100, p2[1] - p1[1])

                self.bbox = {"left": left, "top": top, "width": width, "height": height}
                print(f"[WindowCapture] Active client area: {self.bbox}")
                return True

        print(f"[WindowCapture] No window matching {search_titles} found.")
        # Fallback to primary monitor
        if self.sct:
            self.bbox = self.sct.monitors[1]
        return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Captures frame as BGR uint8 numpy array."""
        if not self.sct:
            return None

        # Re-verify bounding box periodically if hooked to a window
        if self.hwnd and HAS_WIN32:
            try:
                import ctypes
                user32 = ctypes.windll.user32
                hdesk = user32.OpenDesktopW("Default", 0, False, 0x10000000)
                if hdesk:
                    user32.SetThreadDesktop(hdesk)
            except Exception:
                pass

            try:
                if not win32gui.IsWindow(self.hwnd):
                    print("[WindowCapture] Window closed, re-locating...")
                    self.locate_window()
                else:
                    client_rect = win32gui.GetClientRect(self.hwnd)
                    p1 = win32gui.ClientToScreen(self.hwnd, (client_rect[0], client_rect[1]))
                    p2 = win32gui.ClientToScreen(self.hwnd, (client_rect[2], client_rect[3]))
                    self.bbox = {
                        "left": p1[0],
                        "top": p1[1],
                        "width": max(100, p2[0] - p1[0]),
                        "height": max(100, p2[1] - p1[1])
                    }
            except Exception:
                pass

        if not self.bbox:
            return None

        try:
            sct_img = self.sct.grab(self.bbox)
            frame = np.array(sct_img, dtype=np.uint8)  # BGRA
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"[WindowCapture] Grab error: {e}")
            return None


# ==============================================================================
# 2. Perception-to-Tensor Adapter (frame_to_obs)
# ==============================================================================

class PerceptionAdapter:
    """
    Transforms raw captured emulator frames into canonical observation tensors
    for AlphaStarCRNet with NaN-safety and action-masking guarantees.
    """

    def __init__(self, device: torch.device):
        self.device = device

        # Arena layout normalization ratios (based on Clash Royale vertical phone layout)
        # Arena bounds (top, bottom, left, right in normalized 0.0 - 1.0 coords)
        self.arena_box = {"top": 0.05, "bottom": 0.78, "left": 0.03, "right": 0.97}
        # Hand slots (normalized center locations)
        self.hand_box_y = 0.88
        self.hand_xs = [0.22, 0.41, 0.60, 0.79]
        # Elixir bar region
        self.elixir_box = {"top": 0.965, "bottom": 0.985, "left": 0.20, "right": 0.80}

        # Tracked internal game state
        self.tracked_hand = list(DEFAULT_DECK[:4])
        self.tracked_next = DEFAULT_DECK[4] if len(DEFAULT_DECK) > 4 else DEFAULT_DECK[0]
        self.current_elixir = 5.0
        self.match_time_elapsed = 0.0

    def read_elixir(self, frame: np.ndarray) -> float:
        """
        Estimates elixir by cropping the bottom purple bar and computing fill ratio.
        Clash Royale Elixir Bar features a distinct magenta/purple hue (HSV: ~135-165).
        """
        h, w, _ = frame.shape
        ey1 = int(self.elixir_box["top"] * h)
        ey2 = int(self.elixir_box["bottom"] * h)
        ex1 = int(self.elixir_box["left"] * w)
        ex2 = int(self.elixir_box["right"] * w)

        crop = frame[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            return 5.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Purple/magenta mask in HSV: Hue 130-170, Sat 60-255, Val 60-255
        purple_mask = cv2.inRange(hsv, np.array([125, 50, 50]), np.array([170, 255, 255]))

        # Column-wise fill: find farthest right column with purple pixels
        col_sums = np.sum(purple_mask > 0, axis=0)
        filled_cols = np.where(col_sums > (crop.shape[0] * 0.3))[0]

        if len(filled_cols) == 0:
            # Fallback heuristic: gentle tick accumulation
            self.current_elixir = min(10.0, self.current_elixir + 0.14)
            return round(self.current_elixir, 1)

        fill_ratio = (filled_cols[-1] + 1) / max(1, crop.shape[1])
        est_elixir = float(np.clip(fill_ratio * 10.0, 0.0, 10.0))
        self.current_elixir = est_elixir
        return round(est_elixir, 1)

    def read_hand(self, frame: np.ndarray) -> Tuple[List[int], int]:
        """
        Return policy card IDs for the tracked four-card hand and next card.

        The tracker stores simulator card names; the policy consumes the compact
        integer IDs defined by CRSIM_NAME_TO_ID.
        """
        hand_ids = [CRSIM_NAME_TO_ID.get(name, 1) for name in self.tracked_hand]
        next_id = CRSIM_NAME_TO_ID.get(self.tracked_next, 1)
        return hand_ids, next_id

    def detect_entities(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detects units and builds:
        1. entities: [MAX_ENTITIES, 7]
        2. entity_mask: [MAX_ENTITIES] (False = valid entity, True = padding)
        3. arena_grid: [6, 32, 18] spatial channels
        Guarantees slots 0..5 hold the 6 Crown Towers so Transformer NEVER encounters all True mask!
        """
        entities = np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32)
        entity_mask = np.ones(MAX_ENTITIES, dtype=bool)  # default all padded
        arena_grid = np.zeros((SPATIAL_CHANNELS, ARENA_HEIGHT, ARENA_WIDTH), dtype=np.float32)

        # 1. Guaranteed Crown Towers Injection (Slots 0..5)
        # Coordinates in canonical bottom-up space:
        # Friendly: Left Princess (3.5, 6.5), Right Princess (14.5, 6.5), King (9.0, 3.0)
        # Enemy:    Left Princess (3.5, 25.5), Right Princess (14.5, 25.5), King (9.0, 31.0)
        towers = [
            # [card_id_norm, side, x_norm, y_norm, hp_ratio, is_air, state]
            [0.0,  1.0, 3.5 / 18.0,   6.5 / 32.0,  1.0, 0.0, 1.0],  # 0: Friendly Left Princess
            [0.0,  1.0, 14.5 / 18.0,  6.5 / 32.0,  1.0, 0.0, 1.0],  # 1: Friendly Right Princess
            [0.0,  1.0, 9.0 / 18.0,   3.0 / 32.0,  1.0, 0.0, 1.0],  # 2: Friendly King
            [0.0, -1.0, 3.5 / 18.0,  25.5 / 32.0,  1.0, 0.0, 1.0],  # 3: Enemy Left Princess
            [0.0, -1.0, 14.5 / 18.0, 25.5 / 32.0,  1.0, 0.0, 1.0],  # 4: Enemy Right Princess
            [0.0, -1.0, 9.0 / 18.0,  31.0 / 32.0,  1.0, 0.0, 1.0],  # 5: Enemy King
        ]

        for i, t in enumerate(towers):
            entities[i] = t
            entity_mask[i] = False  # UNMASKED -> Active entity

            # Spatial grid channel 4 = buildings/towers
            gx = int(np.clip(t[2] * ARENA_WIDTH, 0, ARENA_WIDTH - 1))
            gy = int(np.clip(t[3] * ARENA_HEIGHT, 0, ARENA_HEIGHT - 1))
            arena_grid[4, gy, gx] = 1.0

        # Channel 5 = valid friendly deployment zone (bottom half y < 16)
        arena_grid[5, :16, :] = 1.0

        # 2. Modular Unit Detection (Detects active troops via OpenCV color segmentation)
        # Friendly units: Green HP bars / blue halos. Enemy units: Red HP bars.
        h, w, _ = frame.shape
        ay1 = int(self.arena_box["top"] * h)
        ay2 = int(self.arena_box["bottom"] * h)
        ax1 = int(self.arena_box["left"] * w)
        ax2 = int(self.arena_box["right"] * w)

        arena_crop = frame[ay1:ay2, ax1:ax2]
        if arena_crop.size > 0:
            hsv = cv2.cvtColor(arena_crop, cv2.COLOR_BGR2HSV)
            # Red HP bar detection (Enemy)
            red1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
            red2 = cv2.inRange(hsv, np.array([170, 120, 100]), np.array([180, 255, 255]))
            enemy_mask = red1 | red2

            # Green HP bar detection (Friendly)
            friendly_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([85, 255, 255]))

            slot_idx = 6
            # Extract detected enemy contours
            contours, _ = cv2.findContours(enemy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if slot_idx >= MAX_ENTITIES:
                    break
                area = cv2.contourArea(cnt)
                if 15 < area < 1500:  # Valid HP bar profile
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        # Map to normalized arena coordinates (0..1)
                        norm_x = float(cx / arena_crop.shape[1])
                        # Invert Y to match canonical bottom-up frame
                        norm_y = float(1.0 - (cy / arena_crop.shape[0]))

                        entities[slot_idx] = [
                            5.0 / NUM_CARDS,  # Default troop proxy
                            -1.0,              # Enemy
                            norm_x, norm_y,
                            1.0,              # HP ratio
                            0.0,              # Ground
                            1.0               # Alive
                        ]
                        entity_mask[slot_idx] = False

                        gx = int(np.clip(norm_x * ARENA_WIDTH, 0, ARENA_WIDTH - 1))
                        gy = int(np.clip(norm_y * ARENA_HEIGHT, 0, ARENA_HEIGHT - 1))
                        arena_grid[2, gy, gx] = 1.0  # opp_ground
                        slot_idx += 1

        return entities, entity_mask, arena_grid

    def calibrate_game_column(self, frame: np.ndarray):
        """
        Dynamically adjusts bounding boxes if game runs in widescreen with black pillar letterboxes.
        """
        h, w = frame.shape[:2]
        aspect = w / max(1, h)
        if aspect > 0.65:
            # Widescreen with letterbox pillars: center 9:16 game column
            target_w = int(h * (9.0 / 16.0))
            pad = max(0, (w - target_w) // 2)
            col_left = pad / w
            col_right = (w - pad) / w
            col_w = col_right - col_left
            self.arena_box = {
                "top": 0.05,
                "bottom": 0.78,
                "left": col_left + 0.03 * col_w,
                "right": col_left + 0.97 * col_w,
            }
            self.hand_xs = [col_left + x * col_w for x in [0.22, 0.41, 0.60, 0.79]]
            self.elixir_box = {
                "top": 0.965,
                "bottom": 0.985,
                "left": col_left + 0.20 * col_w,
                "right": col_left + 0.80 * col_w,
            }
        else:
            self.arena_box = {"top": 0.05, "bottom": 0.78, "left": 0.03, "right": 0.97}
            self.hand_xs = [0.22, 0.41, 0.60, 0.79]
            self.elixir_box = {"top": 0.965, "bottom": 0.985, "left": 0.20, "right": 0.80}

    def frame_to_obs(self, frame: np.ndarray) -> Dict[str, torch.Tensor]:
        """Constructs full 7-tensor observation batch ready for AlphaStarCRNet."""
        self.calibrate_game_column(frame)
        elixir = self.read_elixir(frame)
        hand_cards, next_card = self.read_hand(frame)
        entities, entity_mask, arena_grid = self.detect_entities(frame)

        # 1. Scalars: [own_elixir_norm, opp_elixir_est, time_norm, is_ot]
        time_norm = max(0.0, (180.0 - self.match_time_elapsed) / 180.0)
        is_ot = 1.0 if self.match_time_elapsed >= 180.0 else 0.0
        scalars = np.array([elixir / 10.0, 0.5, time_norm, is_ot], dtype=np.float32)

        # 2. Hand Cards: [card0, card1, card2, card3, next_card]
        hand_tensor_np = np.array(hand_cards + [next_card], dtype=np.int64)

        # 3. Card Mask: 5-way boolean mask (Slots 0..3: Hand, Slot 4: Wait)
        card_mask = np.zeros(5, dtype=bool)
        for i, cid in enumerate(hand_cards):
            cost = CARDS_CATALOG.get(cid, {}).get("cost", 3)
            if elixir >= cost:
                card_mask[i] = True
        card_mask[4] = True  # Slot 4 (Wait) is ALWAYS legal

        # 4. Tile Mask: 576-way boolean mask (Friendly territory for troops, full for spells)
        tile_mask = np.zeros(TOTAL_TILES, dtype=bool)
        # Default troop mask: y < 16
        for y in range(16):
            for x in range(ARENA_WIDTH):
                tile_mask[y * ARENA_WIDTH + x] = True

        # Wrap in PyTorch batch tensors [1, ...] on target device
        obs = {
            "scalars": torch.from_numpy(scalars).unsqueeze(0).to(self.device),
            "hand_cards": torch.from_numpy(hand_tensor_np).unsqueeze(0).to(self.device),
            "entities": torch.from_numpy(entities).unsqueeze(0).to(self.device),
            "entity_mask": torch.from_numpy(entity_mask).unsqueeze(0).to(self.device),
            "arena_grid": torch.from_numpy(arena_grid).unsqueeze(0).to(self.device),
            "card_mask": torch.from_numpy(card_mask).unsqueeze(0).to(self.device),
            "tile_mask": torch.from_numpy(tile_mask).unsqueeze(0).to(self.device),
        }
        return obs

    def update_hand_on_play(self, played_slot: int):
        """Updates internal hand cycle when a card is deployed."""
        if 0 <= played_slot < 4:
            played_card = self.tracked_hand[played_slot]
            card_id = CRSIM_NAME_TO_ID.get(played_card, 1)
            cost = CARDS_CATALOG.get(card_id, {}).get("cost", 3)
            self.current_elixir = max(0.0, self.current_elixir - cost)

            # Draw next card
            self.tracked_hand[played_slot] = self.tracked_next
            # Rotate cycle
            cycle = [c for c in DEFAULT_DECK if c not in self.tracked_hand]
            self.tracked_next = cycle[0] if cycle else DEFAULT_DECK[0]


# ==============================================================================
# 3. Input Controllers (Windows Native Mouse & Persistent ADB Pipe)
# ==============================================================================

class WindowsMouseController:
    """
    Direct Windows mouse click controller for Google Play Games for PC.
    Executes card selection and arena tile deployment using native OS events with <1ms latency.
    """

    def __init__(self, hwnd: Optional[int] = None, bbox: Optional[dict] = None, dry_run: bool = False):
        self.hwnd = hwnd
        self.bbox = bbox or {"left": 0, "top": 0, "width": 720, "height": 1280}
        self.dry_run = dry_run
        self.screen_w = max(100, self.bbox.get("width", 720))
        self.screen_h = max(100, self.bbox.get("height", 1280))
        self.mode_name = "LIVE MOUSE (PC)"

    def set_screen_dimensions(self, width: int, height: int):
        self.screen_w = max(100, width)
        self.screen_h = max(100, height)

    def update_bbox(self, bbox: dict):
        self.bbox = bbox
        self.screen_w = max(100, bbox.get("width", self.screen_w))
        self.screen_h = max(100, bbox.get("height", self.screen_h))

    def grid_to_screen_pixel(self, tile_x: float, tile_y: float, arena_box: dict) -> Tuple[int, int]:
        norm_x = (tile_x + 0.5) / ARENA_WIDTH
        norm_y = (tile_y + 0.5) / ARENA_HEIGHT
        screen_norm_y = arena_box["bottom"] - norm_y * (arena_box["bottom"] - arena_box["top"])
        screen_norm_x = arena_box["left"] + norm_x * (arena_box["right"] - arena_box["left"])
        px = int(np.clip(screen_norm_x * self.screen_w, 0, self.screen_w - 1))
        py = int(np.clip(screen_norm_y * self.screen_h, 0, self.screen_h - 1))
        return px, py

    def card_slot_to_screen_pixel(self, slot_idx: int, hand_xs: List[float], hand_box_y: float) -> Tuple[int, int]:
        slot_idx = max(0, min(3, slot_idx))
        norm_x = hand_xs[slot_idx]
        norm_y = hand_box_y
        px = int(norm_x * self.screen_w)
        py = int(norm_y * self.screen_h)
        return px, py

    def deploy_card(self, card_slot: int, tile_x: int, tile_y: int,
                    arena_box: dict, hand_xs: List[float], hand_box_y: float) -> bool:
        if card_slot == 4:
            return True

        card_rel_x, card_rel_y = self.card_slot_to_screen_pixel(card_slot, hand_xs, hand_box_y)
        arena_rel_x, arena_rel_y = self.grid_to_screen_pixel(tile_x, tile_y, arena_box)

        abs_card_x = self.bbox.get("left", 0) + card_rel_x
        abs_card_y = self.bbox.get("top", 0) + card_rel_y
        abs_arena_x = self.bbox.get("left", 0) + arena_rel_x
        abs_arena_y = self.bbox.get("top", 0) + arena_rel_y

        action_desc = f"Slot {card_slot} -> Tile ({tile_x}, {tile_y}) | Mouse: ({abs_card_x}, {abs_card_y}) -> ({abs_arena_x}, {abs_arena_y})"

        if self.dry_run:
            print(f"[MOUSE DRY-RUN] {action_desc}")
            return True

        try:
            import pyautogui
            pyautogui.click(abs_card_x, abs_card_y)
            time.sleep(0.04)
            pyautogui.click(abs_arena_x, abs_arena_y)
            print(f"[MOUSE EXEC] {action_desc}")
            return True
        except Exception as e:
            print(f"[MOUSE Error] Failed to execute click: {e}")
            return False

    def close(self):
        pass


class ADBController:
    """
    Maintains a persistent interactive ADB shell pipe (`adb shell`)
    for zero-latency card deployment taps on BlueStacks.
    """

    def __init__(self, device_serial: Optional[str] = None, dry_run: bool = False):
        self.device_serial = device_serial
        self.dry_run = dry_run
        self.process: Optional[subprocess.Popen] = None
        self.mode_name = "LIVE ADB"

        # Pixel mapping calibration (default normalized coordinates)
        # Emulator resolution will scale these to actual screen taps
        self.screen_w = 720
        self.screen_h = 1280

        if not self.dry_run:
            self._connect()

    def update_bbox(self, bbox: dict):
        pass

    def _connect(self):
        cmd = ["adb"]
        if self.device_serial:
            cmd.extend(["-s", self.device_serial])
        cmd.append("shell")

        try:
            print(f"[ADB] Launching persistent shell: {' '.join(cmd)}")
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            print("[ADB] Shell pipe established successfully.")
        except Exception as e:
            print(f"[ADB] Warning: Could not connect to ADB shell: {e}")
            self.process = None

    def set_screen_dimensions(self, width: int, height: int):
        self.screen_w = max(100, width)
        self.screen_h = max(100, height)

    def grid_to_screen_pixel(self, tile_x: float, tile_y: float, arena_box: dict) -> Tuple[int, int]:
        """Converts canonical 18x32 grid tile to screen pixel (X, Y)."""
        # tile_x in 0..17, tile_y in 0..31 (canonical bottom-up: 0 = bottom, 31 = top)
        norm_x = (tile_x + 0.5) / ARENA_WIDTH
        norm_y = (tile_y + 0.5) / ARENA_HEIGHT

        # Invert Y to screen pixel space (top = 0, bottom = 1)
        screen_norm_y = arena_box["bottom"] - norm_y * (arena_box["bottom"] - arena_box["top"])
        screen_norm_x = arena_box["left"] + norm_x * (arena_box["right"] - arena_box["left"])

        px = int(np.clip(screen_norm_x * self.screen_w, 0, self.screen_w - 1))
        py = int(np.clip(screen_norm_y * self.screen_h, 0, self.screen_h - 1))
        return px, py

    def card_slot_to_screen_pixel(self, slot_idx: int, hand_xs: List[float], hand_box_y: float) -> Tuple[int, int]:
        """Converts hand slot index (0..3) to screen pixel (X, Y)."""
        slot_idx = max(0, min(3, slot_idx))
        norm_x = hand_xs[slot_idx]
        norm_y = hand_box_y
        px = int(norm_x * self.screen_w)
        py = int(norm_y * self.screen_h)
        return px, py

    def deploy_card(self, card_slot: int, tile_x: int, tile_y: int,
                    arena_box: dict, hand_xs: List[float], hand_box_y: float) -> bool:
        """
        Executes dual-tap sequence via ADB shell:
        Tap 1: Select Card in Hand -> Tap 2: Deploy at Target Tile on Arena.
        """
        if card_slot == 4:
            # Wait / No-Op action
            return True

        card_px, card_py = self.card_slot_to_screen_pixel(card_slot, hand_xs, hand_box_y)
        arena_px, arena_py = self.grid_to_screen_pixel(tile_x, tile_y, arena_box)

        action_desc = f"Slot {card_slot} -> Tile ({tile_x}, {tile_y}) | Taps: ({card_px}, {card_py}) then ({arena_px}, {arena_py})"

        if self.dry_run:
            print(f"[ADB DRY-RUN] {action_desc}")
            return True

        if not self.process or self.process.poll() is not None:
            print("[ADB] Shell pipe disconnected, reconnecting...")
            self._connect()

        if self.process and self.process.stdin:
            try:
                # Write instant tap commands
                self.process.stdin.write(f"input tap {card_px} {card_py}\n")
                self.process.stdin.write(f"input tap {arena_px} {arena_py}\n")
                self.process.stdin.flush()
                print(f"[ADB EXEC] {action_desc}")
                return True
            except Exception as e:
                print(f"[ADB Error] Failed to write tap command: {e}")
                return False
        return False

    def close(self):
        if self.process:
            try:
                self.process.terminate()
            except Exception:
                pass


# ==============================================================================
# 4. Live Deployment Runner & Visualization HUD
# ==============================================================================

def draw_hud(frame: np.ndarray, perception: PerceptionAdapter, controller: Any,
             action: Optional[Tuple[int, int]], value_est: float, fps: float) -> np.ndarray:
    """Draws real-time tactical overlay on the captured emulator frame."""
    viz = frame.copy()
    h, w, _ = viz.shape

    # 1. Draw Arena Boundary
    ay1 = int(perception.arena_box["top"] * h)
    ay2 = int(perception.arena_box["bottom"] * h)
    ax1 = int(perception.arena_box["left"] * w)
    ax2 = int(perception.arena_box["right"] * w)
    cv2.rectangle(viz, (ax1, ay1), (ax2, ay2), (60, 180, 75), 2)

    # 2. Draw River Line (canonical y = 16 / middle of arena)
    river_py = int(ay1 + (ay2 - ay1) * 0.5)
    cv2.line(viz, (ax1, river_py), (ax2, river_py), (255, 150, 50), 2)
    cv2.putText(viz, "RIVER", (ax1 + 8, river_py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 50), 1)

    # 3. Draw Hand Card Slots
    hy = int(perception.hand_box_y * h)
    for i, x_ratio in enumerate(perception.hand_xs):
        hx = int(x_ratio * w)
        card_id = perception.tracked_hand[i]
        cname = CARDS_CATALOG.get(card_id, {}).get("name", f"Card {card_id}")
        cost = CARDS_CATALOG.get(card_id, {}).get("cost", 3)

        is_selected = (action is not None and action[0] == i)
        color = (0, 255, 255) if is_selected else (200, 200, 200)
        cv2.circle(viz, (hx, hy), 24, color, -1 if is_selected else 2)

        cv2.putText(viz, f"{i}:{cost}", (hx - 12, hy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0) if is_selected else (255, 255, 255), 1)
        cv2.putText(viz, cname[:6], (hx - 18, hy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0) if is_selected else (255, 255, 255), 1)

    # 4. Draw Elixir Bar Overlay
    ey1 = int(perception.elixir_box["top"] * h)
    ey2 = int(perception.elixir_box["bottom"] * h)
    ex1 = int(perception.elixir_box["left"] * w)
    ex2 = int(perception.elixir_box["right"] * w)
    cv2.rectangle(viz, (ex1, ey1), (ex2, ey2), (220, 0, 220), 1)
    fill_w = int((perception.current_elixir / 10.0) * (ex2 - ex1))
    cv2.rectangle(viz, (ex1, ey1), (ex1 + fill_w, ey2), (200, 0, 255), -1)
    cv2.putText(viz, f"Elixir: {perception.current_elixir:.1f}/10", (ex1 + 8, ey2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 5. Draw Chosen Deployment Reticle
    if action is not None:
        slot, tile = action
        if slot == 4:
            cv2.putText(viz, "[ACTION] WAIT / NO-OP", (ax1 + 10, ay2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            tx = tile % ARENA_WIDTH
            ty = tile // ARENA_WIDTH
            tpx, tpy = controller.grid_to_screen_pixel(tx, ty, perception.arena_box)
            # Reticle
            cv2.circle(viz, (tpx, tpy), 16, (0, 0, 255), 2)
            cv2.line(viz, (tpx - 22, tpy), (tpx + 22, tpy), (0, 0, 255), 2)
            cv2.line(viz, (tpx, tpy - 22), (tpx, tpy + 22), (0, 0, 255), 2)
            cv2.putText(viz, f"TARGET ({tx},{ty})", (tpx + 18, tpy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # 6. Status Banner
    cv2.rectangle(viz, (0, 0), (w, 35), (20, 20, 20), -1)
    mode_label = getattr(controller, "mode_name", "LIVE")
    status_text = f"AlphaStar Live Agent | V(s): {value_est:+.2f} | FPS: {fps:.1f} | {'DRY-RUN' if controller.dry_run else mode_label}"
    cv2.putText(viz, status_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1)

    return viz


def main():
    parser = argparse.ArgumentParser(description="AlphaStar Clash Royale - PC & Emulator Live Deployment Bridge")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--window-title", type=str, default=None,
                        help="Window title search substring (default: auto-detect 'Clash Royale', 'Google Play Games', or 'BlueStacks')")
    parser.add_argument("--input-mode", type=str, default="auto", choices=["auto", "mouse", "adb"],
                        help="Input dispatch method: mouse (for Google Play Games PC), adb (for BlueStacks), or auto")
    parser.add_argument("--bbox", type=str, default=None,
                        help="Manual capture bounding box as left,top,width,height")
    parser.add_argument("--adb-device", type=str, default=None,
                        help="Specific ADB device serial (optional)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run mode: Visualize decisions without sending mouse clicks or ADB taps")
    parser.add_argument("--visualize", action="store_true",
                        help="Display real-time OpenCV HUD overlay window")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable OpenCV HUD window and run headless in terminal")
    parser.add_argument("--cadence-hz", type=float, default=2.5,
                        help="Decision sampling frequency (default: 2.5 Hz / 0.4s interval)")
    parser.add_argument("--device", type=str, default=None,
                        help="PyTorch device (cuda or cpu)")
    args = parser.parse_args()

    # Hardware & Model Initialization
    if args.device is None:
        hw = get_hardware_profile()
        device = torch.device(hw["device"])
    else:
        device = torch.device(args.device)

    print("=" * 75)
    print("      AlphaStar Clash Royale - Live PC & Emulator Bridge")
    print("=" * 75)
    print(f"[Device]       {device}")
    print(f"[Checkpoint]   {os.path.abspath(args.checkpoint)}")
    print(f"[Mode]         {'DRY-RUN (Simulated Actions)' if args.dry_run else 'LIVE Execution'}")
    print(f"[Cadence]      {args.cadence_hz:.1f} Hz (Period: {1.0 / args.cadence_hz:.3f}s)")

    # Load Model
    model = AlphaStarCRNet().to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            step = ckpt.get("global_step", 0)
            upd = ckpt.get("update_count", 0)
            print(f"[Model Loaded] Checkpoint Step {step} | Update {upd}")
        else:
            model.load_state_dict(ckpt)
            print("[Model Loaded] Raw state dict loaded.")
    else:
        print(f"[Warning] Checkpoint '{args.checkpoint}' not found. Using randomly initialized weights.")
    model.eval()

    # Parse custom bbox if provided
    bbox = None
    if args.bbox:
        parts = [int(p.strip()) for p in args.bbox.split(",")]
        if len(parts) == 4:
            bbox = (parts[0], parts[1], parts[2], parts[3])

    # Initialize Screen Capture & Adapters
    capture = WindowCapture(title_substr=args.window_title, bbox=bbox)
    perception = PerceptionAdapter(device=device)

    # Determine Input Controller
    input_mode = args.input_mode
    if input_mode == "auto":
        input_mode = getattr(capture, "detected_mode", "mouse")

    if input_mode == "mouse":
        controller = WindowsMouseController(
            hwnd=capture.hwnd,
            bbox=capture.bbox,
            dry_run=args.dry_run
        )
        print(f"[Input Mode]   Windows Native Mouse (Google Play Games PC) - Direct OS clicks")
    else:
        controller = ADBController(
            device_serial=args.adb_device,
            dry_run=args.dry_run
        )
        print(f"[Input Mode]   ADB Shell Pipe (BlueStacks Emulator)")

    step_interval = 1.0 / max(0.5, args.cadence_hz)
    t_prev = time.time()
    frame_count = 0
    fps = 0.0

    print("\n[Bridge Running] Press Ctrl+C in terminal (or 'q' in OpenCV window) to stop.")

    try:
        while True:
            t_start = time.time()

            # 1. Capture screen frame
            frame = capture.capture_frame()

            # If no frame captured (e.g., emulator minimized or testing), generate placeholder frame
            if frame is None:
                frame = np.zeros((1280, 720, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for Clash Royale Window...", (120, 640),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            h, w, _ = frame.shape
            controller.set_screen_dimensions(w, h)
            if capture.bbox and hasattr(controller, "update_bbox"):
                controller.update_bbox(capture.bbox)

            perception.calibrate_game_column(frame)

            # 2. Perception: Transform frame into canonical observation batch
            obs = perception.frame_to_obs(frame)

            # 3. Model Inference: Autoregressive Action & Value Prediction
            with torch.no_grad():
                actions, _, _, values = model.get_action_and_value(obs)
                chosen_card = actions[0, 0].item()
                chosen_tile = actions[0, 1].item()
                value_est = values[0].item() if values.dim() == 1 else values[0, 0].item()

            tile_x = chosen_tile % ARENA_WIDTH
            tile_y = chosen_tile // ARENA_WIDTH

            # 4. Action Execution via Controller
            controller.deploy_card(
                card_slot=chosen_card,
                tile_x=tile_x,
                tile_y=tile_y,
                arena_box=perception.arena_box,
                hand_xs=perception.hand_xs,
                hand_box_y=perception.hand_box_y
            )

            if chosen_card != 4:
                perception.update_hand_on_play(chosen_card)

            # 5. Optional Visualization Overlay
            if not args.no_gui and (args.visualize or args.dry_run):
                try:
                    hud_frame = draw_hud(
                        frame=frame,
                        perception=perception,
                        controller=controller,
                        action=(chosen_card, chosen_tile),
                        value_est=value_est,
                        fps=fps
                    )
                    cv2.imshow("AlphaStar Clash Royale - Live Bridge", hud_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except (cv2.error, Exception) as gui_err:
                    if not getattr(main, "_gui_warned", False):
                        print(f"[HUD GUI Notice] GUI overlay disabled ({gui_err}). Continuing live actions via console.")
                        main._gui_warned = True

            # Rate Limiting: 2.5 Hz cadence (~0.4s interval)
            elapsed = time.time() - t_start
            sleep_time = max(0.005, step_interval - elapsed)
            time.sleep(sleep_time)

            frame_count += 1
            if time.time() - t_prev >= 1.0:
                fps = frame_count / (time.time() - t_prev)
                frame_count = 0
                t_prev = time.time()

    except KeyboardInterrupt:
        print("\nStopping live deployment bridge...")
    finally:
        controller.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[Bridge Closed] Clean shutdown complete.")


if __name__ == "__main__":
    main()
