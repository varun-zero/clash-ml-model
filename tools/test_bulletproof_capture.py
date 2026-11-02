import socket
import struct
import subprocess
import time
import os
import cv2
import numpy as np

adb_bin = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
device = "127.0.0.1:5556"

def ensure_connected():
    # Make sure adb server is running and device is connected
    try:
        subprocess.run([adb_bin, "start-server"], capture_output=True, timeout=5)
        subprocess.run([adb_bin, "connect", device], capture_output=True, timeout=5)
    except Exception as e:
        print("[ADB Init Notice]", e)

def grab_frame_socket(target_device="127.0.0.1:5556", timeout=5.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(("127.0.0.1", 5037))

    def _adb_cmd(cmd: str):
        msg = f"{len(cmd):04x}{cmd}".encode("ascii")
        s.sendall(msg)
        resp = s.recv(4)
        if resp != b"OKAY":
            err_len_raw = s.recv(4)
            err_len = int(err_len_raw.decode("ascii"), 16) if err_len_raw else 0
            err_msg = s.recv(err_len).decode("ascii") if err_len else ""
            raise RuntimeError(f"ADB command '{cmd}' failed: {err_msg}")

    _adb_cmd(f"host:transport:{target_device}")
    _adb_cmd("exec:screencap")

    header = b""
    while len(header) < 12:
        chunk = s.recv(12 - len(header))
        if not chunk:
            break
        header += chunk

    if len(header) < 12:
        s.close()
        raise RuntimeError("Failed to read screencap header")

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
    if bytes_read != total_bytes:
        raise RuntimeError(f"Incomplete frame: read {bytes_read}/{total_bytes} bytes")

    rgba = np.frombuffer(raw_data, dtype=np.uint8).reshape((h, w, 4))
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    return bgr

if __name__ == "__main__":
    ensure_connected()
    print("Testing grab_frame_socket...")
    t0 = time.perf_counter()
    frame = grab_frame_socket()
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"Captured frame shape: {frame.shape} in {dt:.1f}ms")
    cv2.imwrite("bulletproof_live.png", frame)
    print("Saved bulletproof_live.png successfully!")

