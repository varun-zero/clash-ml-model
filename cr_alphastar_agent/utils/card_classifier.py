"""
Card Classifier for Clash Royale Hand Slots.
Combines visual template matching with normalized HSV color/edge signatures
to reliably identify which card is in each hand slot every frame.
"""
import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

# Card aliases and canonical names
DECK_CARDS = [
    "Knight", "MiniPekka", "Archers", "Musketeer",
    "Arrows", "Fireball", "Giant", "Minions"
]

# Baseline HSV color signatures for the 8 deck cards
# [H_min, H_max, S_min, S_max, V_min, V_max, weight]
COLOR_PROFILES = {
    "Fireball": {
        "h_range": (5, 25), "s_min": 130, "v_min": 140,
        "desc": "Fiery orange/yellow sphere"
    },
    "Arrows": {
        "h_range": (90, 118), "s_min": 80, "v_min": 130,
        "desc": "Bright cyan/sky-blue background"
    },
    "Minions": {
        "h_range": (115, 140), "s_min": 70, "v_max": 145,
        "desc": "Dark purple/indigo gargoyles"
    },
    "Archers": {
        "h_range": (145, 172), "s_min": 90, "v_min": 100,
        "desc": "Vibrant magenta/pink hair"
    },
    "Archer": {
        "h_range": (145, 172), "s_min": 90, "v_min": 100,
        "desc": "Vibrant magenta/pink hair"
    },
    "MiniPekka": {
        "h_range": (85, 105), "s_max": 95, "v_range": (50, 160),
        "desc": "Dark steel grey with cyan eye accent"
    },
    "Mini PEKKA": {
        "h_range": (85, 105), "s_max": 95, "v_range": (50, 160),
        "desc": "Dark steel grey with cyan eye accent"
    },
    "Mini P.E.K.K.A": {
        "h_range": (85, 105), "s_max": 95, "v_range": (50, 160),
        "desc": "Dark steel grey with cyan eye accent"
    },
    "Giant": {
        "h_range": (12, 32), "s_range": (80, 180), "v_range": (90, 180),
        "desc": "Warm brown/tan tunic and orange beard"
    },
    "Knight": {
        "h_range": (20, 45), "s_range": (70, 210), "v_min": 130,
        "desc": "Golden yellow hair/armor"
    },
    "Musketeer": {
        "h_range": (130, 155), "s_range": (60, 180), "v_range": (80, 200),
        "desc": "Purple hair with steel helmet"
    }
}

class VisualCardClassifier:
    """Classifies hand card slots into canonical card names from live frames."""
    
    def __init__(self, template_dir: str = "card_templates", active_deck: Optional[List[str]] = None):
        self.template_dir = template_dir
        if active_deck is not None:
            self.active_deck = list(active_deck)
        else:
            self.active_deck = ["Knight", "Mini P.E.K.K.A", "Archers", "Musketeer", "Arrows", "Fireball", "Giant", "Minions"]
        self.templates: Dict[str, np.ndarray] = {}
        self.load_templates()
        
    @staticmethod
    def _canonical(name: str) -> str:
        """Normalizes card names for robust comparison."""
        return name.replace(" ", "").replace(".", "").lower()

    def load_templates(self):
        """Loads reference card templates from disk if available, or auto-extracts from deck screenshot."""
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir, exist_ok=True)

        # Check if templates need to be auto-extracted from deck screenshot
        required_templates = ["Knight", "MiniPekka", "Minions", "Arrows", "Fireball", "Giant", "Musketeer", "Archers"]
        missing = [c for c in required_templates if not os.path.exists(os.path.join(self.template_dir, f"{c}.png"))]
        
        if missing:
            # Search candidate deck screenshots
            candidate_paths = [
                r"C:/Users/varun/.gemini/antigravity/brain/42ce9bd6-0ed7-406b-aa50-7c85b510c081/deck_screen.png",
                "deck_screen.png",
                "scratch/deck_screen.png"
            ]
            deck_img = None
            for p in candidate_paths:
                if os.path.exists(p):
                    deck_img = cv2.imread(p)
                    if deck_img is not None:
                        break
            
            if deck_img is not None:
                card_coords = {
                    "Knight": (60, 415, 150, 210),
                    "Archers": (275, 415, 150, 210),
                    "Minions": (490, 415, 150, 210),
                    "Arrows": (700, 415, 150, 210),
                    "Fireball": (60, 760, 150, 210),
                    "Giant": (275, 760, 150, 210),
                    "Musketeer": (490, 760, 150, 210),
                    "MiniPekka": (700, 760, 150, 210)
                }
                for cname, (x, y, w, h) in card_coords.items():
                    crop = deck_img[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join(self.template_dir, f"{cname}.png"), crop)

        for fname in os.listdir(self.template_dir):
            if fname.endswith(".png") or fname.endswith(".jpg"):
                cname = os.path.splitext(fname)[0]
                p = os.path.join(self.template_dir, fname)
                img = cv2.imread(p)
                if img is not None:
                    self.templates[cname] = img
                    # Also register alias keys
                    self.templates[self._canonical(cname)] = img
                    if cname == "MiniPekka":
                        self.templates["Mini PEKKA"] = img
                        self.templates["Mini P.E.K.K.A"] = img
                    elif cname == "Archers":
                        self.templates["Archer"] = img

    def crop_slots(self, frame: np.ndarray,
                   y_range: Tuple[float, float] = (0.835, 0.945),
                   x_slots: Optional[List[Tuple[float, float]]] = None) -> List[np.ndarray]:
        """Crops the 4 hand card slots from a vertical frame."""
        h, w = frame.shape[:2]
        if x_slots is None:
            x_slots = [(0.24, 0.38), (0.43, 0.57), (0.62, 0.76), (0.81, 0.95)]
            
        y1, y2 = int(y_range[0] * h), int(y_range[1] * h)
        crops = []
        for x1_r, x2_r in x_slots:
            x1, x2 = int(x1_r * w), int(x2_r * w)
            crop = frame[y1:y2, x1:x2]
            crops.append(crop)
        return crops

    def score_card_crop(self, crop: np.ndarray, cname: str) -> float:
        """Computes matching similarity score [0.0 .. 1.0] for a given card candidate."""
        if crop is None or crop.size == 0:
            return 0.0
            
        # 1. Template match score if reference image exists
        template_score = 0.0
        tmpl = self.templates.get(cname)
        if tmpl is None:
            tmpl = self.templates.get(self._canonical(cname))
        if tmpl is not None:
            # Resize template to match crop size
            resized_tmpl = cv2.resize(tmpl, (crop.shape[1], crop.shape[0]))
            res = cv2.matchTemplate(crop, resized_tmpl, cv2.TM_CCOEFF_NORMED)
            template_score = float(max(0.0, res[0, 0]))
            
        # 2. HSV Color Profile Score
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        prof = COLOR_PROFILES.get(cname)
        if prof is None:
            prof = COLOR_PROFILES.get(self._canonical(cname), {})
        
        # Compute matching pixel mask for this card profile
        h_min, h_max = prof.get("h_range", (0, 180))
        h_mask = (h >= h_min) & (h <= h_max)
        
        s_min = prof.get("s_min", 0)
        s_max = prof.get("s_max", 255)
        s_range = prof.get("s_range")
        if s_range:
            s_min, s_max = s_range
        s_mask = (s >= s_min) & (s <= s_max)
        
        v_min = prof.get("v_min", 0)
        v_max = prof.get("v_max", 255)
        v_range = prof.get("v_range")
        if v_range:
            v_min, v_max = v_range
        v_mask = (v >= v_min) & (v <= v_max)
        
        match_pixels = np.sum(h_mask & s_mask & v_mask)
        color_score = float(match_pixels / max(1, crop.shape[0] * crop.shape[1]))
        
        # Combined score: if template exists, 85% template + 15% color; else 100% color
        if tmpl is not None:
            return 0.85 * template_score + 0.15 * min(1.0, color_score * 2.0)
        else:
            return float(np.clip(color_score * 3.5, 0.0, 1.0))

    def classify_hand(self, frame: np.ndarray,
                      candidate_deck: Optional[List[str]] = None,
                      return_scores: bool = False):
        """Classifies the 4 slots and ensures 4 distinct cards strictly from active_deck are chosen."""
        # Strictly restrict candidate deck to active_deck
        allowed_deck = list(self.active_deck)
        if candidate_deck is not None:
            allowed_canon = {self._canonical(c): c for c in allowed_deck}
            filtered = []
            for c in candidate_deck:
                cn = self._canonical(c)
                if cn in allowed_canon:
                    filtered.append(allowed_canon[cn])
            if len(filtered) == len(allowed_deck):
                candidate_deck = filtered
            else:
                candidate_deck = allowed_deck
        else:
            candidate_deck = allowed_deck
            
        crops = self.crop_slots(frame)
        if len(crops) != 4:
            if return_scores:
                return candidate_deck[:4], [1.0] * 4
            return candidate_deck[:4]
            
        # Build score matrix: [4 slots, len(candidate_deck)]
        scores = np.zeros((4, len(candidate_deck)), dtype=np.float32)
        for s_idx, crop in enumerate(crops):
            for c_idx, cname in enumerate(candidate_deck):
                scores[s_idx, c_idx] = self.score_card_crop(crop, cname)
                
        # Greedy assignment to enforce 4 distinct cards
        chosen = [None] * 4
        chosen_scores = [0.0] * 4
        used_cards = set()
        
        # Sort all (score, slot, card) tuples descending
        pairs = []
        for s in range(4):
            for c in range(len(candidate_deck)):
                pairs.append((scores[s, c], s, candidate_deck[c]))
        pairs.sort(key=lambda x: -x[0])
        
        for score, s, cname in pairs:
            if chosen[s] is None and cname not in used_cards:
                chosen[s] = cname
                chosen_scores[s] = float(score)
                used_cards.add(cname)
                
        # Fill any unassigned slots
        rem = [c for c in candidate_deck if c not in used_cards]
        for s in range(4):
            if chosen[s] is None:
                chosen[s] = rem.pop(0) if rem else candidate_deck[s]
                chosen_scores[s] = 0.5
                
        self.last_scores = chosen_scores
        if return_scores:
            return chosen, chosen_scores
        return chosen
