"""
Generates sample match frames in samples/ directory (real_01.png)
with accurate 540x960 9:16 game viewport, river, bridges, towers, health bars,
hand cards, pink elixir bar, and battle units.
"""
import os
import cv2
import numpy as np

os.makedirs("./samples", exist_ok=True)

# Generate 540x960 match frame
frame = np.zeros((960, 540, 3), dtype=np.uint8)

# Arena grass background (greenish)
frame[96:750, 21:518] = [34, 139, 34]

# River (blueish horizontal band across rows 15-16)
frame[400:430, 21:518] = [180, 105, 30]

# Left Bridge (brown)
frame[395:435, 100:145] = [42, 42, 165]

# Right Bridge (brown)
frame[395:435, 395:440] = [42, 42, 165]

# Ally King tower (blueish)
frame[660:740, 226:314] = [200, 50, 50]
# Enemy King tower (reddish)
frame[105:185, 226:314] = [50, 50, 200]

# Ally Princess Towers
frame[580:650, 95:165] = [180, 60, 60]
frame[580:650, 375:445] = [180, 60, 60]

# Enemy Princess Towers
frame[190:260, 95:165] = [60, 60, 180]
frame[190:260, 375:445] = [60, 60, 180]

# Tower HP Bars: [enemy_king, enemy_left, enemy_right, ally_king, ally_left, ally_right]
# Enemy Left princess health bar (red)
frame[238:255, 97:162] = [40, 40, 220]
# Enemy Right princess destroyed (no health bar)
# Ally Left princess health bar (green/blue)
frame[622:639, 97:162] = [40, 220, 40]
# Ally Right princess health bar (green/blue)
frame[622:639, 378:443] = [40, 220, 40]

# Units on field:
# Enemy Knight (red border) at row 22, col 5
frame[500:540, 120:150] = [30, 30, 200]
# Ally Archers (blue border) at row 24, col 6
frame[550:585, 160:190] = [200, 50, 50]

# Hand slots (bottom dark bar)
frame[780:920, 10:530] = [30, 30, 30]
# Card 0 (Knight)
frame[800:900, 30:120] = [60, 80, 120]
# Card 1 (Archers)
frame[800:900, 140:230] = [80, 120, 60]
# Card 2 (Fireball)
frame[800:900, 250:340] = [40, 60, 180]
# Card 3 (Arrows)
frame[800:900, 360:450] = [120, 140, 60]

# Elixir pink bar (7.5 / 10 full)
frame[925:955, 90:390] = [180, 20, 200]
frame[925:955, 390:500] = [20, 20, 20]

# Timer text (white digits on dark pill)
frame[28:62, 388:513] = [30, 30, 30]
cv2.putText(frame, "2:15", (400, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.imwrite("./samples/real_01.png", frame)
print("Saved sample frame to ./samples/real_01.png")
