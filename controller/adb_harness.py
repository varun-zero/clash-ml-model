"""
High-Performance ADB Harness for BlueStacks 5 & Android Emulators.
Dispatches non-blocking clicks using persistent shell pipes or fast worker threads.
Converts (slot_idx, tile_y, tile_x) into accurate two-tap deployments.
"""
from typing import Tuple, Optional
import os
import sys
import time
import subprocess
import threading
import queue

from configs.constants import (
    grid_to_screen_pixel,
    hand_slot_to_screen_pixel,
    clashroyalebuildabot_to_grid,
    ARENA_COLS,
    ARENA_ROWS,
)


class ADBHarness:
    """
    Manages ADB connection and fast non-blocking tap dispatch.
    Uses persistent adb shell stdin pipe for sub-millisecond dispatch overhead.
    """
    def __init__(
        self,
        device_address: str = "127.0.0.1:5555",
        screen_width: int = 540,
        screen_height: int = 960,
        adb_binary: str = "adb",
        tap_delay_ms: int = 40,
    ):
        self.device_address = device_address
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.adb_binary = adb_binary
        self.tap_delay_sec = tap_delay_ms / 1000.0

        self.action_queue: queue.Queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.shell_proc: Optional[subprocess.Popen] = None
        self._pipe_lock = threading.Lock()

        # Connect to ADB device
        self._connect()
        self._init_persistent_shell()

        # Start non-blocking execution worker
        self.worker_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.worker_thread.start()

    def _connect(self):
        """Ensures device is connected via ADB."""
        print(f"[ADB] Connecting to {self.device_address}...")
        try:
            res = subprocess.run(
                [self.adb_binary, "connect", self.device_address],
                capture_output=True,
                text=True,
                timeout=5
            )
            print(f"[ADB] {res.stdout.strip()}")
        except Exception as e:
            print(f"[ADB] Connection error: {e}. Will attempt commands anyway.")

    def _init_persistent_shell(self):
        """Initializes a persistent ADB shell pipe."""
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
            print(f"[ADB] Warning: Failed to spawn persistent shell ({e}). Using subprocess fallback.")
            self.shell_proc = None

    def _send_shell_cmd(self, cmd_str: str):
        """Sends command directly through persistent stdin pipe or fallback."""
        with self._pipe_lock:
            if self.shell_proc and self.shell_proc.poll() is None and self.shell_proc.stdin:
                try:
                    self.shell_proc.stdin.write(cmd_str + "\n")
                    self.shell_proc.stdin.flush()
                    return
                except Exception as e:
                    print(f"[ADB] Pipe write failed ({e}), restarting shell...")
                    self._init_persistent_shell()

        # Fallback to direct subprocess call
        try:
            subprocess.run(
                [self.adb_binary, "-s", self.device_address, "shell", cmd_str],
                capture_output=True,
                timeout=2
            )
        except FileNotFoundError:
            print(f"[ADB] Warning: '{self.adb_binary}' executable not found in PATH.")
        except Exception as e:
            print(f"[ADB] Command execution error: {e}")

    def _dispatch_loop(self):
        """Background worker thread consuming tap commands without blocking inference."""
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

    def _execute_two_tap(self, slot_idx: int, tile_y: int, tile_x: int):
        """Performs hand card tap -> sleep -> arena tile tap."""
        hx, hy = hand_slot_to_screen_pixel(slot_idx, self.screen_width, self.screen_height)
        ax, ay = grid_to_screen_pixel(tile_y, tile_x, self.screen_width, self.screen_height)

        t0 = time.perf_counter()
        # Tap 1: Select card in hand
        self._send_shell_cmd(f"input tap {hx} {hy}")

        # Rapid delay (40ms)
        time.sleep(self.tap_delay_sec)

        # Tap 2: Place card on arena grid
        self._send_shell_cmd(f"input tap {ax} {ay}")
        elapsed = (time.perf_counter() - t0) * 1000.0

        print(f"[ADB Tap] Deployed Slot {slot_idx} -> Tile ({tile_y}, {tile_x}) at ({ax}, {ay}) in {elapsed:.1f}ms")

    def play_card(self, slot_idx: int, tile_y: int, tile_x: int, non_blocking: bool = True):
        """
        Public dispatch API.
        Args:
            slot_idx: Hand slot 0..3
            tile_y: Arena row 0..31
            tile_x: Arena col 0..17
            non_blocking: If True, queues command instantly and returns without waiting.
        """
        if not (0 <= slot_idx <= 3):
            raise ValueError(f"Invalid hand slot index: {slot_idx}")
        if not (0 <= tile_y < ARENA_ROWS and 0 <= tile_x < ARENA_COLS):
            raise ValueError(f"Invalid tile coordinates: ({tile_y}, {tile_x})")

        if non_blocking:
            # Drop older pending tap if bot is generating a newer action
            if self.action_queue.full():
                try:
                    self.action_queue.get_nowait()
                except queue.Empty:
                    pass
            self.action_queue.put((slot_idx, tile_y, tile_x))
        else:
            self._execute_two_tap(slot_idx, tile_y, tile_x)

    def play_card_buildabot_coords(self, slot_idx: int, bot_x: int, bot_y: int):
        """Dispatches card using ClashRoyaleBuildABot coordinate conventions."""
        tile_y, tile_x = clashroyalebuildabot_to_grid(bot_x, bot_y)
        self.play_card(slot_idx, tile_y, tile_x)

    def close(self):
        """Cleanly terminates the worker thread and adb shell process."""
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

