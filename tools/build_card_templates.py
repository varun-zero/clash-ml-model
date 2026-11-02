"""
Card Template Builder and Perceptual Hash Clustering Tool for Clash Royale.
Crops hand card slots from samples/real_*.png, computes perceptual difference hashes (dHash),
clusters similar card crops, prompts/prints clusters for naming, and saves templates
to templates/<global_id>/<hash>.png for template-matching recognition.
"""
from typing import List, Dict, Tuple, Optional
import os
import sys
import glob
import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.constants import HAND_SLOTS, CARD_DATABASE, CARD_NAME_TO_ID
from configs.unlocked_cards import load_unlocked_cards


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """
    Computes 64-bit difference hash (dHash) for an image crop.
    Fast, robust against slight lighting / compression variations.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize to (hash_size + 1, hash_size)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)

    # Compute difference between adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # Convert binary matrix to 64-bit integer
    decimal_val = 0
    for bit in diff.flatten():
        decimal_val = (decimal_val << 1) | int(bit)
    return decimal_val


def hamming_distance(hash1: int, hash2: int) -> int:
    """Computes bitwise Hamming distance between two 64-bit hashes."""
    return bin(hash1 ^ hash2).count("1")


def cluster_card_crops(
    crops: List[Tuple[str, int, np.ndarray, int]],
    threshold: int = 12
) -> List[List[Tuple[str, int, np.ndarray, int]]]:
    """
    Clusters crops where crop = (source_file, slot_idx, crop_img, dhash_val).
    Hamming distance <= threshold groups them into the same card cluster.
    """
    clusters: List[List[Tuple[str, int, np.ndarray, int]]] = []

    for item in crops:
        src, slot, img, h = item
        matched_cluster = None

        for cluster in clusters:
            # Compare to first item in cluster (centroid/representative)
            rep_hash = cluster[0][3]
            if hamming_distance(h, rep_hash) <= threshold:
                matched_cluster = cluster
                break

        if matched_cluster is not None:
            matched_cluster.append(item)
        else:
            clusters.append([item])

    return clusters


def build_templates(
    samples_dir: str = "./samples",
    output_templates_dir: str = "./templates",
    hamming_thresh: int = 12,
    interactive: bool = False,
):
    """
    Scans samples_dir for real_*.png, extracts hand slots 0..3,
    clusters crops, and stores templates under templates/<global_id>/.
    """
    pattern = os.path.join(samples_dir, "real_*.png")
    sample_files = sorted(glob.glob(pattern))

    if not sample_files:
        print(f"[TemplateBuilder] No samples found matching '{pattern}'.")
        print("[TemplateBuilder] Place real match screenshots (real_01.png, real_02.png) in ./samples/")
        return

    print(f"[TemplateBuilder] Found {len(sample_files)} sample frame(s). Extracting hand card slots...")

    extracted_crops: List[Tuple[str, int, np.ndarray, int]] = []

    for filepath in sample_files:
        frame = cv2.imread(filepath)
        if frame is None:
            print(f"[TemplateBuilder] Warning: Failed to read {filepath}")
            continue

        h, w = frame.shape[:2]
        for slot_idx, box in enumerate(HAND_SLOTS):
            bx1, by1, bx2, by2 = box
            x1, y1 = int(bx1 * w), int(by1 * h)
            x2, y2 = int(bx2 * w), int(by2 * h)

            # Crop inner 80% to avoid slot border frames
            crop_w = x2 - x1
            crop_h = y2 - y1
            pad_x = int(crop_w * 0.10)
            pad_y = int(crop_h * 0.10)
            card_crop = frame[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]

            if card_crop.size == 0:
                continue

            # Standardize template crop size to 64x64
            std_crop = cv2.resize(card_crop, (64, 64), interpolation=cv2.INTER_AREA)
            dhash_val = compute_dhash(std_crop)
            extracted_crops.append((filepath, slot_idx, std_crop, dhash_val))

    print(f"[TemplateBuilder] Extracted {len(extracted_crops)} hand card crops.")
    clusters = cluster_card_crops(extracted_crops, threshold=hamming_thresh)
    print(f"[TemplateBuilder] Identified {len(clusters)} unique card cluster(s) via perceptual hashing.\n")

    unlocked_ids = load_unlocked_cards()
    print("Available starter/unlocked cards for assignment:")
    for cid in unlocked_ids:
        meta = CARD_DATABASE.get(cid, {})
        print(f"  ID {cid:02d}: {meta.get('name', 'Unknown')} (Cost: {meta.get('cost', 0)})")
    print()

    # Process and save clusters
    for c_idx, cluster in enumerate(clusters):
        rep_src, rep_slot, rep_img, rep_hash = cluster[0]
        print(f"--- Cluster #{c_idx + 1} ({len(cluster)} instances, rep: {os.path.basename(rep_src)} slot {rep_slot}) ---")

        assigned_id = None
        if interactive:
            val = input(f"Enter global card ID or card name for Cluster #{c_idx + 1} (or Enter to skip): ").strip()
            if val.isdigit() and int(val) in CARD_DATABASE:
                assigned_id = int(val)
            elif val.lower() in CARD_NAME_TO_ID:
                assigned_id = CARD_NAME_TO_ID[val.lower()]
        else:
            # In non-interactive mode, match against unlocked cards sequentially or print info
            print(f"  Cluster {c_idx + 1} representative hash: 0x{rep_hash:016x}")

        if assigned_id is not None:
            save_dir = os.path.join(output_templates_dir, str(assigned_id))
            os.makedirs(save_dir, exist_ok=True)
            for i, (src, slot, img, h) in enumerate(cluster):
                dst = os.path.join(save_dir, f"template_{h:016x}_{i:02d}.png")
                cv2.imwrite(dst, img)
            print(f"  -> Saved {len(cluster)} template(s) to {save_dir}")

    print("\n[TemplateBuilder] Complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build and cluster Clash Royale card templates")
    parser.add_argument("--samples", type=str, default="./samples", help="Samples directory")
    parser.add_argument("--templates", type=str, default="./templates", help="Output templates directory")
    parser.add_argument("--threshold", type=int, default=12, help="Hamming distance threshold")
    parser.add_argument("--interactive", action="store_true", help="Prompt user to assign card IDs to clusters")
    args = parser.parse_args()

    build_templates(
        samples_dir=args.samples,
        output_templates_dir=args.templates,
        hamming_thresh=args.threshold,
        interactive=args.interactive,
    )

