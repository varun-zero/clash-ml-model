"""
Quick verification script to test BlueStacks window capture, elixir reading,
and hand card detection directly from your active desktop terminal.
"""
import sys
import os
import time
import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from controller.screen_capture import WindowCapture
from perception.detector import ClashRoyaleDetector
from configs.constants import CARD_DATABASE

def main():
    print("=" * 60)
    print("CR-Zero Vision & Window Diagnostics")
    print("=" * 60)

    cap = WindowCapture(window_title_keyword="BlueStacks", target_width=540, target_height=960)
    det = ClashRoyaleDetector(weights="weights/yolo_units.pt")

    print(f"\n[Capture] Window HWND: {cap.hwnd}")
    print(f"[Capture] Window Region: {cap.window_region}")
    print(f"[Capture] Backend: {cap.backend_name}")

    print("\nCapturing 3 live frames from BlueStacks...")
    for i in range(1, 4):
        frame = cap.grab_frame()
        obs = det.parse_frame(frame)
        hand_names = [CARD_DATABASE.get(cid, {}).get("name", f"Unknown#{cid}") for cid in obs["hand_ids"]]
        
        print(f"Frame {i}:")
        print(f"  -> Screen State: {obs['screen_state'].upper()}")
        print(f"  -> Elixir Level:  {obs['elixir']:.1f} / 10.0")
        print(f"  -> Hand Cards:   {obs['hand_ids']} ({hand_names})")
        print(f"  -> Towers HP:    {[round(h, 2) for h in obs['tower_hps']]}")
        time.sleep(1.0)

    cv2.imwrite("diagnostics_capture.png", frame)
    print("\nSaved diagnostics capture to 'diagnostics_capture.png'.")
    cap.close()

if __name__ == "__main__":
    main()
