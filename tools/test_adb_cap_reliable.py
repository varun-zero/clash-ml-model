import subprocess
import os
import cv2
import time

adb_bin = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
device = "127.0.0.1:5556"

# Ensure server and connection
subprocess.run([adb_bin, "start-server"], capture_output=True)
subprocess.run([adb_bin, "connect", device], capture_output=True)

# Method 1: shell screencap to /sdcard/frame.png then pull
t0 = time.perf_counter()
cmd_cap = f'"{adb_bin}" -s {device} shell screencap -p /sdcard/frame.png'
cmd_pull = f'"{adb_bin}" -s {device} pull /sdcard/frame.png frame_out.png'

os.system(cmd_cap)
os.system(cmd_pull)
dt = (time.perf_counter() - t0) * 1000.0

if os.path.exists("frame_out.png"):
    img = cv2.imread("frame_out.png")
    if img is not None:
        print(f"[SUCCESS] Frame captured: {img.shape} in {dt:.1f}ms")
    else:
        print("[FAIL] Image read failed")
else:
    print("[FAIL] File does not exist")

