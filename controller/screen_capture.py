"""
Ultra-fast Windows 11 Desktop / Window Capture for BlueStacks 5.
Uses DirectX Desktop Duplication (dxcam) with automatic mss fallback.
Captures the Win32 client area (excluding title bar and window margins),
verifies 9:16 aspect ratio, and detects minimized windows.
"""
from typing import Tuple, Optional
import os
import sys
import time
import ctypes
import numpy as np
import cv2

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


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_long),
        ("y", ctypes.c_long),
    ]


class WindowCapture:
    """
    DirectX / mss Screen Capture engine for BlueStacks 5.
    Captures only the inner client viewport, ensuring accurate alignment with normalized boxes.
    """
    def __init__(
        self,
        window_title_keyword: str = "BlueStacks",
        target_width: int = 540,
        target_height: int = 960,
        preferred_backend: str = "dxcam",  # 'dxcam' or 'mss'
        require_window: bool = False,
        adb_device: str = "127.0.0.1:5556",
    ):
        self.window_keyword = window_title_keyword
        self.target_width = target_width
        self.target_height = target_height
        self.backend_name = preferred_backend
        self.require_window = require_window
        self.adb_device = adb_device
        if self.adb_device == "127.0.0.1:5555" and os.name == "nt":
            import re
            conf_paths = [
                r"C:\ProgramData\BlueStacks_nxt\bluestacks.conf",
                os.path.expandvars(r"%ALLUSERSPROFILE%\BlueStacks_nxt\bluestacks.conf"),
            ]
            for cp in conf_paths:
                if os.path.exists(cp):
                    try:
                        with open(cp, "r", encoding="utf-8", errors="ignore") as f:
                            m = re.search(r'status\.adb_port="(\d+)"', f.read())
                            if m and m.group(1) != "5555":
                                self.adb_device = f"127.0.0.1:{m.group(1)}"
                                break
                    except Exception:
                        pass

        self.hwnd: Optional[int] = None
        self.window_region: Optional[Tuple[int, int, int, int]] = None  # (left, top, right, bottom)
        self.dxcam_camera = None
        self.mss_instance = None
        self.adb_bin = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe" if os.path.exists(r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe") else "adb"
        import tempfile
        self.temp_cap_file = os.path.join(tempfile.gettempdir(), "cr_zero_live_cap.png")

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
            if self.require_window:
                raise RuntimeError(
                    f"Required window matching '{self.window_keyword}' not found. "
                    f"Make sure BlueStacks 5 is running and visible."
                )
            print(f"[Capture] Warning: Window containing '{self.window_keyword}' not found. Defaulting to full monitor or dummy.")

    def update_window_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Updates window client area bounding box.
        Fails if window is minimized. Verifies 9:16 aspect ratio.
        """
        if not self.hwnd or os.name != "nt":
            return None

        # Fail loudly if window is minimized (Iconic)
        if user32.IsIconic(self.hwnd):
            raise RuntimeError(
                f"[Capture] Target window '{self.window_keyword}' (HWND {self.hwnd}) is minimized! "
                f"BlueStacks must be restored and visible on screen for accurate capture."
            )

        # Use GetClientRect + ClientToScreen to exclude window chrome / title bar
        client_rect = RECT()
        if not user32.GetClientRect(self.hwnd, ctypes.byref(client_rect)):
            return None

        origin = POINT(0, 0)
        user32.ClientToScreen(self.hwnd, ctypes.byref(origin))

        left = origin.x
        top = origin.y
        w = client_rect.right - client_rect.left
        h = client_rect.bottom - client_rect.top
        right = left + w
        bottom = top + h

        if w > 50 and h > 50:
            # Verify 9:16 aspect ratio (expected ~0.5625)
            aspect = w / float(h)
            expected_aspect = 9.0 / 16.0
            if abs(aspect - expected_aspect) > 0.08:
                print(
                    f"[Capture Warning] Client area aspect ratio {aspect:.3f} deviates from 9:16 ({expected_aspect:.3f}). "
                    f"Client dimensions: {w}x{h}. Ensure BlueStacks is locked to portrait 540x960 or 720x1280."
                )

            self.window_region = (left, top, right, bottom)
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
        Grabs latest frame from BlueStacks client area.
        Returns BGR numpy array of shape (target_height, target_width, 3).
        """
        region = self.update_window_bounds()

        # 1. DXCam capture
        if self.dxcam_camera is not None:
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

        # 2. MSS capture fallback (only if window region is confirmed)
        if self.mss_instance is not None and region is not None:
            left, top, right, bottom = region
            bbox = {"left": left, "top": top, "width": right - left, "height": bottom - top}
            try:
                sct_img = self.mss_instance.grab(bbox)
                frame = np.array(sct_img)[:, :, :3]  # BGRA -> BGR
                if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                    frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
                return frame
            except Exception:
                pass

        # 3. Direct ADB screencap pull (pixel-perfect 540x960 directly from BlueStacks emulator)
        adb_frame = self._grab_adb_pull()
        if adb_frame is not None:
            return adb_frame

        # 4. Virtual dummy frame if running completely disconnected
        dummy = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        cv2.putText(dummy, "Virtual Display (Client Area)", (30, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return dummy

    def _grab_adb_pull(self) -> Optional[np.ndarray]:
        """Captures frame directly from BlueStacks emulator via screencap and file pull."""
        try:
            import subprocess
            cmd_cap = f'"{self.adb_bin}" -s {self.adb_device} shell screencap -p /sdcard/sc_live.png'
            subprocess.run(cmd_cap, shell=True, capture_output=True, timeout=2.0)
            pull_cmd = f'"{self.adb_bin}" -s {self.adb_device} pull /sdcard/sc_live.png "{self.temp_cap_file}"'
            subprocess.run(pull_cmd, shell=True, capture_output=True, timeout=2.0)
            if os.path.exists(self.temp_cap_file):
                img = cv2.imread(self.temp_cap_file)
                if img is not None:
                    if img.shape[1] != self.target_width or img.shape[0] != self.target_height:
                        img = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
                    return img
        except Exception:
            return None
        return None

    def _grab_adb_socket(self) -> Optional[np.ndarray]:
        """Captures raw uncompressed frame directly from BlueStacks ADB daemon via TCP socket."""
        try:
            import socket
            import struct
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(("127.0.0.1", 5037))

            def _adb_cmd(cmd: str):
                msg = f"{len(cmd):04x}{cmd}".encode("ascii")
                s.sendall(msg)
                resp = s.recv(4)
                if resp != b"OKAY":
                    raise RuntimeError(f"ADB error: {resp}")

            _adb_cmd(f"host:transport:{self.adb_device}")
            _adb_cmd("exec:screencap")

            header = b""
            while len(header) < 12:
                chunk = s.recv(12 - len(header))
                if not chunk:
                    break
                header += chunk

            if len(header) < 12:
                s.close()
                return None

            w, h, fmt = struct.unpack("<III", header)
            total_bytes = w * h * 4
            raw_data = bytearray(total_bytes)
            view = memoryview(raw_data)
            bytes_read = 0

            while bytes_read < total_bytes:
                n = s.recv_into(view[bytes_read:], total_bytes - bytes_read)
                if n == 0:
                    break
                bytes_read += n

            s.close()
            if bytes_read == total_bytes:
                rgba = np.frombuffer(raw_data, dtype=np.uint8).reshape((h, w, 4))
                bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                if bgr.shape[1] != self.target_width or bgr.shape[0] != self.target_height:
                    bgr = cv2.resize(bgr, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
                return bgr
        except Exception:
            return None
        return None

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
