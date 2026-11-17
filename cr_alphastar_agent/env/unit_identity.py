"""Self-improving visual identity for units on the arena.

The shipped ``card_templates/`` are crops of the *deck screen*; they do not
look like the 3D units on the battlefield.  This module learns unit templates
from the game itself:

* every deployment we can verify on a follow-up frame gives us a labelled
  screenshot: we know the card name and the tile it was played on;
* a tight crop around that tile is stored as a template for the card;
* on later ticks, enemy units are compared against the learned templates and
  identified when the correlation is high enough.

The memory grows with every match, survives restarts, and is the only part of
the system that can identify an enemy card that is not in our own deck (it
learns those templates when the opponent plays them and we later observe the
same sprite).  Confidence is reported so downstream code can ignore guesses.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CROP_W = 56
CROP_H = 72
MIN_CORRELATION = 0.72
MAX_TEMPLATES_PER_CARD = 24


@dataclass
class Identification:
    card_name: str
    confidence: float
    card_id: int = 5


class UnitIdentityMemory:
    """Persistent, self-growing library of battlefield unit templates."""

    def __init__(self, root: str = "data/unit_templates") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.templates: Dict[str, List[np.ndarray]] = {}
        self.counts: Dict[str, int] = {}
        self._load()

    # ---------------------------------------------------------------- storage
    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for name, files in (index.get("templates") or {}).items():
            loaded = []
            for rel in files:
                path = self.root / rel
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is not None and image.shape[:2] == (CROP_H, CROP_W):
                    loaded.append(image)
            if loaded:
                self.templates[name] = loaded
        self.counts = dict(index.get("counts") or {})

    def _save(self) -> None:
        index = {"templates": {}, "counts": self.counts}
        for name, images in self.templates.items():
            files = []
            for i, image in enumerate(images):
                rel = f"{name}_{i:03d}.png"
                cv2.imwrite(str(self.root / rel), image)
                files.append(rel)
            index["templates"][name] = files
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index, indent=2), encoding="utf-8")
        os.replace(tmp, self.index_path)

    # ------------------------------------------------------------------ learn
    def learn(self, card_name: str, frame: np.ndarray,
              px: int, py: int) -> bool:
        """Store a labelled crop of a unit we know we just deployed."""
        crop = self._crop(frame, px, py)
        if crop is None:
            return False
        images = self.templates.setdefault(card_name, [])
        # Avoid storing near-duplicates of the same frame.
        if images:
            best = max(float(cv2.matchTemplate(crop, t, cv2.TM_CCOEFF_NORMED).max())
                       for t in images[-6:])
            if best >= 0.97:
                return False
        images.append(crop)
        if len(images) > MAX_TEMPLATES_PER_CARD:
            images.pop(0)
        self.counts[card_name] = int(self.counts.get(card_name, 0)) + 1
        self._save()
        return True

    # --------------------------------------------------------------- identify
    @staticmethod
    def _crop(frame: np.ndarray, px: int, py: int) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x0 = int(np.clip(px - CROP_W // 2, 0, max(0, w - CROP_W)))
        y0 = int(np.clip(py - CROP_H // 2, 0, max(0, h - CROP_H)))
        if x0 + CROP_W > w or y0 + CROP_H > h:
            return None
        crop = frame[y0:y0 + CROP_H, x0:x0 + CROP_W].copy()
        if crop.shape[:2] != (CROP_H, CROP_W):
            return None
        return crop

    def identify(self, frame: np.ndarray, px: int, py: int) -> Optional[Identification]:
        """Identify the unit at a screen position, or ``None`` if uncertain."""
        crop = self._crop(frame, px, py)
        if crop is None or not self.templates:
            return None

        best_name = None
        best_score = 0.0
        for name, images in self.templates.items():
            for template in images:
                res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
                score = float(np.max(res))
                if score > best_score:
                    best_score = score
                    best_name = name
        if best_name is None or best_score < MIN_CORRELATION:
            return None
        return Identification(best_name, round(best_score, 3))

    def identify_many(self, frame: np.ndarray,
                      positions: Sequence[Tuple[int, int]]) -> List[Identification]:
        out = []
        for px, py in positions:
            found = self.identify(frame, px, py)
            if found is not None:
                out.append(found)
        return out

    def __len__(self) -> int:
        return sum(len(v) for v in self.templates.values())

    def learned_cards(self) -> List[str]:
        return sorted(self.templates)
