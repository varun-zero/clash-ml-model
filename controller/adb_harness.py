"""
High-Performance, Safety-Hardened ADB Harness for BlueStacks 5 & Android Emulators.
Enforces:
1. Strict binary resolution (--adb-path / ADB_PATH / shutil.which) with loud failure.
2. Verified device connectivity via `adb devices`.
3. Device resolution querying via `wm size` and accurate pixel scaling.
4. Real execution success reporting (never reports "Deployed" on failure).
5. Safety controls: --dry-run mode, max-actions limit, 800ms action cooldown, and F12 panic stop.
"""
from typing import Tuple, Optional, Dict
import os
import sys
import time
import shutil
import subprocess
import threading
import queue
import re

from configs.constants import (
    grid_to_screen_pixel,
    hand_slot_to_screen_pixel,
    clashroyalebuildabot_to_grid,
    ARENA_COLS,
    ARENA_ROWS,
)


class ADBHarness:
    """
    Manages ADB connection and fast non-blocking tap dispatch with strict safety rails.
    """
    def __init__(
        self,
        device_address: str = "127.0.0.1:5555",
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        adb_path: Optional[str] = None,
        tap_delay_ms: int = 40,
        action_cooldown_sec: float = 0.800,  # 800ms minimum interval between plays
        max_actions: Optional[int] = None,   # Halt after N actions
        dry_run: bool = False,
    ):
        self.device_address = device_address
        self.dry_run = dry_run
        self.tap_delay_sec = tap_delay_ms / 1000.0
        self.action_cooldown_sec = action_cooldown_sec
        self.max_actions = max_actions
        self.actions_performed = 0
        self.last_action_time = 0.0

        self.action_queue: queue.Queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.panic_event = threading.Event()
        self.shell_proc: Optional[subprocess.Popen] = None
        self._pipe_lock = threading.Lock()

        # Auto-detect BlueStacks ADB port if using default address
        if self.device_address == "127.0.0.1:5555" and os.name == "nt":
            detected_port = self._detect_bluestacks_port()
            if detected_port and detected_port != "5555":
                print(f"[ADB] Detected active BlueStacks ADB port: {detected_port}")
                self.device_address = f"127.0.0.1:{detected_port}"

        # Step 1: Resolve ADB executable
        self.adb_binary = self._resolve_adb_binary(adb_path)

        # Step 2: Validate device connection (skipped in dry_run if binary absent)
        if not self.dry_run:
            self._verify_device_connection()

        # Step 3: Query device resolution via `wm size`
        self.device_width, self.device_height = self._determine_resolution(
            screen_width, screen_height
        )

        # Step 4: Initialize persistent shell pipe if not dry run
        if not self.dry_run:
            self._init_persistent_shell()

        # Step 5: Start panic listener (F12 key)
        self._start_panic_listener()

        # Step 6: Start non-blocking execution worker
        self.worker_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.worker_thread.start()

    @staticmethod
    def _detect_bluestacks_port() -> Optional[str]:
        """Detects active BlueStacks ADB port from bluestacks.conf on Windows."""
        conf_paths = [
            r"C:\ProgramData\BlueStacks_nxt\bluestacks.conf",
            os.path.expandvars(r"%ALLUSERSPROFILE%\BlueStacks_nxt\bluestacks.conf"),
        ]
        for cp in conf_paths:
            if os.path.exists(cp):
                try:
                    with open(cp, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    match = re.search(r'status\.adb_port="(\d+)"', content)
                    if match:
                        return match.group(1)
                except Exception:
                    pass
        return None

    def _resolve_adb_binary(self, explicit_path: Optional[str] = None) -> str:
        """Resolves ADB executable via explicit arg, ADB_PATH env var, or shutil.which."""
        candidates = []
        if explicit_path:
            candidates.append(explicit_path)
        if os.environ.get("ADB_PATH"):
            candidates.append(os.environ["ADB_PATH"])
        which_adb = shutil.which("adb")
        if which_adb:
            candidates.append(which_adb)

        # Standard BlueStacks / Android SDK paths on Windows
        default_paths = [
            r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
            r"C:\Program Files (x86)\BlueStacks_nxt\HD-Adb.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"),
        ]
        for p in default_paths:
            if os.path.exists(p):
                candidates.append(p)

        for candidate in candidates:
            if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
                print(f"[ADB] Resolved executable: {candidate}")
                return candidate

        if self.dry_run:
            print("[ADB] Notice: No 'adb' binary found, but continuing in --dry-run mode.")
            return "adb_mock"

        raise FileNotFoundError(
            "Could not locate 'adb' executable on system PATH or standard emulator locations.\n"
            "Please specify --adb-path, set the ADB_PATH environment variable, or run with --dry-run."
        )

    def _verify_device_connection(self):
        """Validates that target device is listed and in 'device' state."""
        print(f"[ADB] Verifying connectivity to {self.device_address}...")
        # First attempt connect
        try:
            conn_res = subprocess.run(
                [self.adb_binary, "connect", self.device_address],
                capture_output=True,
                text=True,
                timeout=10,
            )
            print(f"[ADB] connect: {conn_res.stdout.strip()}")
        except Exception as e:
            print(f"[ADB] Connect attempt notice: {e}")

        # Now query `adb devices`
        try:
            dev_res = subprocess.run(
                [self.adb_binary, "devices"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to execute 'adb devices': {e}") from e

        output = dev_res.stdout
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        connected_devices = {}
        for line in lines[1:]:  # skip 'List of devices attached'
            parts = line.split()
            if len(parts) >= 2:
                connected_devices[parts[0]] = parts[1]

        print(f"[ADB] Active devices found: {connected_devices}")
        if self.device_address not in connected_devices:
            raise ConnectionError(
                f"Target device '{self.device_address}' not found in 'adb devices'. "
                f"Connected devices: {list(connected_devices.keys())}"
            )

        state = connected_devices[self.device_address]
        if state != "device":
            raise ConnectionError(
                f"Device '{self.device_address}' is in '{state}' state (must be 'device'). "
                f"Please authorize USB debugging or restart emulator."
            )

    def _determine_resolution(
        self, width_override: Optional[int], height_override: Optional[int]
    ) -> Tuple[int, int]:
        """Queries `wm size` from device and applies overrides."""
        if width_override is not None and height_override is not None:
            print(f"[ADB] Using manual resolution override: {width_override}x{height_override}")
            return int(width_override), int(height_override)

        if self.dry_run or self.adb_binary == "adb_mock":
            w = width_override or 540
            h = height_override or 960
            print(f"[ADB Dry-Run] Using default screen resolution: {w}x{h}")
            return w, h

        try:
            res = subprocess.run(
                [self.adb_binary, "-s", self.device_address, "shell", "wm", "size"],
                capture_output=True,
                text=True,
                timeout=4,
            )
            # Output format: "Physical size: 1080x1920" or "Override size: 720x1280"
            matches = re.findall(r"(\d+)x(\d+)", res.stdout)
            if matches:
                # Last match is override size if present, otherwise physical size
                w, h = int(matches[-1][0]), int(matches[-1][1])
                print(f"[ADB] Detected device resolution via `wm size`: {w}x{h}")
                return w, h
        except Exception as e:
            print(f"[ADB] Warning: Could not query wm size ({e}). Falling back to 540x960.")

        return width_override or 540, height_override or 960

    def _init_persistent_shell(self):
        """Initializes a persistent interactive ADB shell pipe."""
        try:
            cmd = [self.adb_binary, "-s", self.device_address, "shell"]
            self.shell_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            print(f"[ADB] Persistent shell pipe established on {self.device_address}.")
        except Exception as e:
            print(f"[ADB] Warning: Failed to spawn persistent shell ({e}). Using direct subprocess fallback.")
            self.shell_proc = None

    def _start_panic_listener(self):
        """Spawns background thread listening for F12 panic stop."""
        def listener():
            # Windows API GetAsyncKeyState(0x7B) for VK_F12
            try:
                import ctypes
                has_user32 = hasattr(ctypes, "windll") and hasattr(ctypes.windll, "user32")
            except Exception:
                has_user32 = False

            while not self.stop_event.is_set():
                if has_user32:
                    # 0x7B is VK_F12; highest bit (0x8000) indicates key is down
                    if ctypes.windll.user32.GetAsyncKeyState(0x7B) & 0x8000:
                        print("\n" + "!" * 60)
                        print("[ADB PANIC] F12 pressed! Triggering immediate emergency shutdown...")
                        print("!" * 60 + "\n")
                        self.panic_event.set()
                        self.stop_event.set()
                        break
                time.sleep(0.05)

        t = threading.Thread(target=listener, daemon=True)
        t.start()

    def is_panicked(self) -> bool:
        """Returns True if panic stop was triggered."""
        return self.panic_event.is_set()

    def _send_shell_cmd(self, cmd_str: str) -> bool:
        """Sends command through persistent stdin pipe or direct subprocess."""
        if self.dry_run:
            return True

        with self._pipe_lock:
            if self.shell_proc and self.shell_proc.poll() is None and self.shell_proc.stdin:
                try:
                    self.shell_proc.stdin.write(cmd_str + "\n")
                    self.shell_proc.stdin.flush()
                    return True
                except Exception as e:
                    print(f"[ADB] Pipe write failed ({e}), falling back to direct subprocess...")
                    self._init_persistent_shell()

        # Fallback to direct subprocess call
        try:
            res = subprocess.run(
                [self.adb_binary, "-s", self.device_address, "shell", cmd_str],
                capture_output=True,
                timeout=2,
            )
            return res.returncode == 0
        except Exception as e:
            print(f"[ADB] Command execution failed: {e}")
            return False

    def _dispatch_loop(self):
        """Background worker thread consuming tap tasks without blocking inference."""
        while not self.stop_event.is_set():
            try:
                task = self.action_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if task is None:
                break

            slot_idx, tile_y, tile_x = task
            self._execute_two_tap(slot_idx, tile_y, tile_x)
            self.action_queue.task_done()

    def _execute_two_tap(self, slot_idx: int, tile_y: int, tile_x: int) -> bool:
        """
        Executes hand card tap -> sleep -> arena tile tap.
        Returns True on confirmed dispatch, False on failure.
        Never prints 'Deployed' on failure.
        """
        if self.panic_event.is_set():
            print("[ADB] Tap rejected: Panic stop active.")
            return False

        hx, hy = hand_slot_to_screen_pixel(slot_idx, self.device_width, self.device_height)
        ax, ay = grid_to_screen_pixel(tile_y, tile_x, self.device_width, self.device_height)

        if self.dry_run:
            print(
                f"[ADB Dry-Run] Card Slot {slot_idx} ({hx}, {hy}) -> "
                f"Arena Tile ({tile_y}, {tile_x}) at ({ax}, {ay}) [Target res: {self.device_width}x{self.device_height}]"
            )
            self.actions_performed += 1
            self.last_action_time = time.perf_counter()
            return True

        t0 = time.perf_counter()
        # Tap 1: Select card in hand
        ok1 = self._send_shell_cmd(f"input tap {hx} {hy}")
        if not ok1:
            print(f"[ADB Tap Error] Failed to tap hand slot {slot_idx} at ({hx}, {hy})")
            return False

        # Sleep rapid delay
        time.sleep(self.tap_delay_sec)

        # Tap 2: Place card on arena grid
        ok2 = self._send_shell_cmd(f"input tap {ax} {ay}")
        if not ok2:
            print(f"[ADB Tap Error] Failed to tap arena tile ({tile_y}, {tile_x}) at ({ax}, {ay})")
            return False

        elapsed = (time.perf_counter() - t0) * 1000.0
        self.actions_performed += 1
        self.last_action_time = time.perf_counter()
        print(f"[ADB Tap] Deployed Slot {slot_idx} -> Tile ({tile_y}, {tile_x}) at ({ax}, {ay}) in {elapsed:.1f}ms")
        return True

    def play_card(
        self,
        slot_idx: int,
        tile_y: int,
        tile_x: int,
        non_blocking: bool = True
    ) -> bool:
        """
        Public dispatch API with cooldown and action limits.
        Returns True if action was accepted/executed, False if rejected.
        """
        if self.panic_event.is_set():
            print("[ADB] Action rejected: Panic shutdown active.")
            return False

        if not (0 <= slot_idx <= 3):
            raise ValueError(f"Invalid hand slot index: {slot_idx}")
        if not (0 <= tile_y < ARENA_ROWS and 0 <= tile_x < ARENA_COLS):
            raise ValueError(f"Invalid tile coordinates: ({tile_y}, {tile_x})")

        # Action limit check
        if self.max_actions is not None and self.actions_performed >= self.max_actions:
            print(f"[ADB Limit] Reached max actions ({self.max_actions}). Halting dispatch.")
            self.stop_event.set()
            return False

        # 800ms Cooldown check
        now = time.perf_counter()
        elapsed = now - self.last_action_time
        if self.last_action_time > 0 and elapsed < self.action_cooldown_sec:
            print(f"[ADB Cooldown] Ignored play ({elapsed*1000:.0f}ms < {self.action_cooldown_sec*1000:.0f}ms).")
            return False

        if non_blocking:
            # Drop older pending tap if a newer action is generated
            if self.action_queue.full():
                try:
                    self.action_queue.get_nowait()
                except queue.Empty:
                    pass
            self.action_queue.put((slot_idx, tile_y, tile_x))
            return True
        else:
            return self._execute_two_tap(slot_idx, tile_y, tile_x)

    def play_card_buildabot_coords(self, slot_idx: int, bot_x: int, bot_y: int) -> bool:
        """Dispatches card using ClashRoyaleBuildABot coordinate conventions."""
        tile_y, tile_x = clashroyalebuildabot_to_grid(bot_x, bot_y)
        return self.play_card(slot_idx, tile_y, tile_x)

    def close(self):
        """Cleanly terminates the worker thread and persistent shell process."""
        self.stop_event.set()
        try:
            self.action_queue.put(None, timeout=0.1)
        except Exception:
            pass
        with self._pipe_lock:
            if self.shell_proc:
                try:
                    self.shell_proc.terminate()
                    self.shell_proc.wait(timeout=1)
                except Exception:
                    pass
                self.shell_proc = None
        print("[ADB] Harness closed cleanly.")
