"""Generates sample test frames in samples/ directory."""
import os
import cv2
import numpy as np

os.makedirs("./samples", exist_ok=True)

# 1. Generate 540x960 sample frame
frame1 = np.zeros((960, 540, 3), dtype=np.uint8)
# Arena grass background (greenish)
frame1[96:750, 21:518] = [34, 139, 34]
# River (blueish horizontal band across rows 15-16)
frame1[400:430, 21:518] = [180, 105, 30]
# Left Bridge (brown)
frame1[395:435, 100:145] = [42, 42, 165]
# Right Bridge (brown)
frame1[395:435, 395:440] = [42, 42, 165]
# Ally King tower (blueish)
frame1[660:740, 226:314] = [200, 50, 50]
# Enemy King tower (reddish)
frame1[105:185, 226:314] = [50, 50, 200]
# Hand slots (bottom dark bar)
frame1[780:920, 10:530] = [30, 30, 30]
# Elixir pink bar
frame1[925:955, 90:500] = [180, 20, 200]

cv2.imwrite("./samples/sample_match_frame.png", frame1)
print("Saved sample frame to ./samples/sample_match_frame.png")
