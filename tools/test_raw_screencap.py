import socket
import struct
import time
import numpy as np
import cv2

# ADB protocol: connect to ADB server on 127.0.0.1:5037
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 5037))

def adb_command(cmd: str):
    msg = f"{len(cmd):04x}{cmd}".encode("ascii")
    s.sendall(msg)
    resp = s.recv(4)
    if resp != b"OKAY":
        err_len = int(s.recv(4).decode("ascii"), 16)
        err = s.recv(err_len).decode("ascii")
        raise RuntimeError(f"ADB error: {err}")

# Switch target to 127.0.0.1:5556
adb_command("host:transport:127.0.0.1:5556")

# Send exec:screencap
t0 = time.perf_counter()
adb_command("exec:screencap")

# Read header: width(4), height(4), format(4)
header = b""
while len(header) < 12:
    chunk = s.recv(12 - len(header))
    if not chunk:
        break
    header += chunk

w, h, fmt = struct.unpack("<III", header)
print(f"Frame header: {w}x{h}, format: {fmt}")

total_bytes = w * h * 4
raw_data = bytearray(total_bytes)
view = memoryview(raw_data)
bytes_read = 0

while bytes_read < total_bytes:
    n = s.recv_into(view[bytes_read:], total_bytes - bytes_read)
    if n == 0:
        break
    bytes_read += n

dt = (time.perf_counter() - t0) * 1000.0
print(f"Read {bytes_read}/{total_bytes} bytes in {dt:.1f}ms")

s.close()

if bytes_read == total_bytes:
    img = np.frombuffer(raw_data, dtype=np.uint8).reshape((h, w, 4))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite("fast_live_screen.png", bgr)
    print(f"[SUCCESS] Ultra-fast direct ADB socket capture saved: {bgr.shape} in {dt:.1f}ms!")

