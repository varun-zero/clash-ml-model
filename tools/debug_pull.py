import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller.screen_capture import WindowCapture
import subprocess

cap = WindowCapture()
print('adb_bin:', cap.adb_bin)
print('adb_device:', cap.adb_device)
print('temp_cap_file:', cap.temp_cap_file)

cmd_cap = f'"{cap.adb_bin}" -s {cap.adb_device} shell screencap -p /sdcard/sc_live.png'
res1 = subprocess.run(cmd_cap, shell=True, capture_output=True, timeout=5.0)
print('res1 stdout:', res1.stdout.decode('utf-8', errors='ignore'))
print('res1 stderr:', res1.stderr.decode('utf-8', errors='ignore'))

pull_cmd = f'"{cap.adb_bin}" -s {cap.adb_device} pull /sdcard/sc_live.png "{cap.temp_cap_file}"'
res2 = subprocess.run(pull_cmd, shell=True, capture_output=True, timeout=5.0)
print('res2 stdout:', res2.stdout.decode('utf-8', errors='ignore'))
print('res2 stderr:', res2.stderr.decode('utf-8', errors='ignore'))
print('File exists:', os.path.exists(cap.temp_cap_file))
