"""
Ultra-fast Windows 11 Desktop / Window Capture for BlueStacks 5.
Uses DirectX Desktop Duplication (dxcam) with automatic mss fallback.
Captures locked 540x960 or 720x1280 9:16 viewport at 30+ FPS.
"""
from typing import Tuple, Optional
import os
import sys
import time
import ctypes
import numpy as np
import cv2

# Windows user32 API for finding window handle and bounds
if os.name == "nt":
    user32 = ctypes.windll.user32
    from ctypes import wintypes


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class WindowCapture:
    """
    DirectX / mss Screen Capture engine for BlueStacks 5.
    """
    def __init__(
        self,
        window_title_keyword: str = "BlueStacks",
        target_width: int = 540,
        target_height: int = 960,
        preferred_backend: str = "dxcam",  # 'dxcam' or 'mss'
    ):
        self.window_keyword = window_title_keyword
        self.target_width = target_width
        self.target_height = target_height
        self.backend_name = preferred_backend

        self.hwnd: Optional[int] = None
        self.window_region: Optional[Tuple[int, int, int, int]] = None  # (left, top, right, bottom)
        self.dxcam_camera = None
        self.mss_instance = None

        self._find_window()
        self._init_backend()

    def _find_window(self):
        """Finds BlueStacks HWND matching keyword in window title."""
        if os.name != "nt":
            print("[Capture] Non-Windows OS detected. Screen capture will use virtual dummy frames.")
            return

        found_hwnds = []

        def enum_windows_callback(hwnd, extra):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buff, length + 1)
                    if self.window_keyword.lower() in buff.value.lower():
                        found_hwnds.append((hwnd, buff.value))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_windows_callback), 0)

        if found_hwnds:
            self.hwnd, title = found_hwnds[0]
            print(f"[Capture] Found target window: '{title}' (HWND: {self.hwnd})")
            self.update_window_bounds()
        else:
            print(f"[Capture] Warning: Window containing '{self.window_keyword}' not found. Defaulting to full primary monitor.")

    def update_window_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """Updates window bounding box."""
        if not self.hwnd or os.name != "nt":
            return None

        rect = RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top

        if w > 50 and h > 50:
            self.window_region = (rect.left, rect.top, rect.right, rect.bottom)
            return self.window_region
        return None

    def _init_backend(self):
        """Initializes dxcam or falls back to mss."""
        if self.backend_name == "dxcam":
            try:
                import dxcam
                self.dxcam_camera = dxcam.create(output_color="BGR")
                print("[Capture] DXCam DirectX Desktop Duplication initialized successfully.")
                return
            except Exception as e:
                print(f"[Capture] dxcam initialization failed ({e}). Falling back to mss.")

        # Fallback to mss
        try:
            import mss
            self.mss_instance = mss.mss()
            self.backend_name = "mss"
            print("[Capture] mss Screen Capture initialized successfully.")
        except Exception as e:
            print(f"[Capture] Warning: mss initialization failed ({e}).")

    def grab_frame(self) -> np.ndarray:
        """
        Grabs latest frame from BlueStacks window.
        Returns BGR numpy array of shape (target_height, target_width, 3).
        """
        # 1. DXCam capture
        if self.dxcam_camera is not None:
            region = self.update_window_bounds()
            try:
                if region:
                    left, top, right, bottom = region
                    frame = self.dxcam_camera.grab(region=(left, top, right, bottom))
                else:
                    frame = self.dxcam_camera.grab()

                if frame is not None:
                    if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                        frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
                    return frame
            except Exception:
                pass

        # 2. MSS capture fallback
        if self.mss_instance is not None:
            region = self.update_window_bounds()
            if region:
                left, top, right, bottom = region
                bbox = {"left": left, "top": top, "width": right - left, "height": bottom - top}
            else:
                bbox = self.mss_instance.monitors[1]

            sct_img = self.mss_instance.grab(bbox)
            frame = np.array(sct_img)[:, :, :3]  # BGRA -> BGR
            if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
            return frame

        # 3. Virtual dummy frame if running headless or without display
        dummy = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        cv2.putText(dummy, "Virtual Display", (50, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return dummy

    def close(self):
        """Releases capture resources."""
        if self.dxcam_camera is not None:
            try:
                del self.dxcam_camera
            except Exception:
                pass
        if self.mss_instance is not None:
            try:
                self.mss_instance.close()
            except Exception:
                pass

