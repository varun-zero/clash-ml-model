import subprocess
import cv2
import numpy as np
import os

adb_bin = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
device = "127.0.0.1:5556"

# Ensure connected
subprocess.run([adb_bin, "connect", device], capture_output=True, timeout=10)

# Capture screenshot via adb screencap to device /sdcard/sc.png then pull, or raw screencap
cmd_cap = [adb_bin, "-s", device, "shell", "screencap", "-p", "/sdcard/sc.png"]
subprocess.run(cmd_cap, capture_output=True, timeout=10)

cmd_pull = [adb_bin, "-s", device, "pull", "/sdcard/sc.png", "current_screen.png"]
res_pull = subprocess.run(cmd_pull, capture_output=True, timeout=10)

if os.path.exists("current_screen.png"):
    img = cv2.imread("current_screen.png")
    if img is not None:
        print(f"[OK] Screencap successful! Resolution: {img.shape[1]}x{img.shape[0]}")
    else:
        print("[FAIL] File exists but cv2 failed to read image.")
else:
    print(f"[FAIL] Pull failed: {res_pull.stderr.decode('utf-8', errors='ignore')}")

