import ctypes
from ctypes import wintypes
import psutil

user32 = ctypes.windll.user32

# Find HD-Player PID
pids = [p.pid for p in psutil.process_iter(['name']) if 'hd-player' in p.info['name'].lower()]
print("HD-Player PIDs:", pids)

found = []
def enum_cb(hwnd, extra):
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value in pids:
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        class_buff = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buff, 256)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        found.append((hwnd, buff.value, class_buff.value, bool(user32.IsWindowVisible(hwnd)), w, h))
    return True

WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

for h, title, cls, vis, w, h_dim in found:
    print(f"HWND: {h} | Title: '{title}' | Class: '{cls}' | Visible: {vis} | Size: {w}x{h_dim}")

