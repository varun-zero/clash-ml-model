"""Static-background subtraction for full-HP unit detection.

HP-bar colour segmentation cannot see a troop at 100% HP: the bar only
appears once the unit is damaged.  Units also disappear on stone bridges and
tower bases because the "dark border" heuristic fails there.

The Clash Royale arena itself is static — only the river shimmers.  This
module keeps an exponential moving average of the arena background, updates
it *only where no tracked unit is present*, and exposes moving regions via
``cv2.absdiff``.  The result is fused with the existing HP-bar detector so a
unit is visible the instant it spawns behind the enemy King Tower.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cr_alphastar_agent.config import ARENA_HEIGHT, ARENA_WIDTH

DEFAULT_ALPHA = 0.02
DIFF_THRESHOLD = 26
MIN_BLOB_AREA = 90
MAX_BLOB_AREA = 9000
MIN_ASPECT = 0.35
MAX_ASPECT = 2.8


@dataclass
class MotionBlob:
    px: int                 # centre in arena-crop pixels
    py: int
    w: int
    h: int
    area: float
    x_tile: float = 0.0
    y_tile: float = 0.0

    @property
    def aspect(self) -> float:
        return self.w / float(max(1, self.h))


class ArenaMonitor:
    """Maintain a clean background plate and detect moving foreground blobs."""

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = float(alpha)
        self.background: Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ plate
    def _ensure(self, crop: np.ndarray) -> None:
        if self.background is None or self._shape != crop.shape[:2]:
            self.background = crop.astype(np.float32).copy()
            self._shape = crop.shape[:2]

    def suppress(self, crop: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]],
                 alpha: Optional[float] = None) -> None:
        """Update the background everywhere except inside active unit boxes."""
        self._ensure(crop)
        rate = self.alpha if alpha is None else float(alpha)
        mask = np.ones(crop.shape[:2], dtype=np.float32)
        for bx, by, bw, bh in boxes:
            x0 = max(0, int(bx) - 4)
            y0 = max(0, int(by) - 4)
            x1 = min(crop.shape[1], int(bx + bw) + 4)
            y1 = min(crop.shape[0], int(by + bh) + 4)
            mask[y0:y1, x0:x1] = 0.0
        current = crop.astype(np.float32)
        self.background = (self.background * (1.0 - rate * mask[..., None])
                           + current * (rate * mask[..., None]))

    # ---------------------------------------------------------------- detect
    def detect(self, crop: np.ndarray, arena_box: dict,
               frame_shape: Tuple[int, int]) -> List[MotionBlob]:
        """Return moving blobs converted to canonical tile coordinates."""
        self._ensure(crop)
        current = crop.astype(np.int16)
        diff = cv2.absdiff(current, self.background.astype(np.int16))
        grey = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grey, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        # Do NOT blank the river band here.  The EMA background already
        # absorbs the static river and its shimmer; masking rows 0.43..0.58
        # split any troop standing on a bridge into thin strips and recreated
        # the very bridge blindness this detector is meant to fix.
        h, w = mask.shape[:2]
        mask[:, :int(0.05 * w)] = 0
        mask[:, int(0.95 * w):] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs: List[MotionBlob] = []
        for cnt in cnts:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = float(cv2.contourArea(cnt))
            if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
                continue
            aspect = bw / float(max(1, bh))
            if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
                continue
            px, py = x + bw // 2, y + bh // 2
            # arena crop -> full frame -> canonical tile
            fx = arena_box["left"] + (px / float(w)) * (
                arena_box["right"] - arena_box["left"])
            fy = arena_box["top"] + (py / float(h)) * (
                arena_box["bottom"] - arena_box["top"])
            frame_h, frame_w = frame_shape[:2]
            nx = fx
            ny = 1.0 - (fy - arena_box["top"]) / max(
                1e-6, (arena_box["bottom"] - arena_box["top"]))
            blobs.append(MotionBlob(
                px=px, py=py, w=bw, h=bh, area=area,
                x_tile=float(np.clip(nx * ARENA_WIDTH, 0, ARENA_WIDTH - 1)),
                y_tile=float(np.clip(ny * ARENA_HEIGHT, 0, ARENA_HEIGHT - 1)),
            ))

        self.suppress(crop, [(b.px - b.w // 2, b.py - b.h // 2, b.w, b.h)
                             for b in blobs])
        return blobs

    def reset(self) -> None:
        self.background = None
        self._shape = None


__all__ = ["ArenaMonitor", "MotionBlob"]
