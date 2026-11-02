"""
Updates and enriches configs/card_features.py using Hugging Face dataset:
Nitesh-04/Clash-Royale-Cards-Data (clash_royale_cards_1.json)
Link: https://huggingface.co/datasets/Nitesh-04/Clash-Royale-Cards-Data
"""
import os
import sys
import json
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.constants import CARD_DATABASE, NUM_TOTAL_CARDS


def fetch_card_metadata():
    """Fetches card data from huggingface_hub cache or direct download."""
    try:
        from huggingface_hub import hf_hub_download
        cached_file = hf_hub_download(
            repo_id="Nitesh-04/Clash-Royale-Cards-Data",
            filename="clash_royale_cards_1.json",
            repo_type="dataset",
        )
    except Exception as e:
        print(f"[Cards Dataset] HF Hub download fallback: {e}")
        # Check standard cache directory
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--Nitesh-04--Clash-Royale-Cards-Data")
        candidates = []
        for root, _, files in os.walk(cache_dir):
            for f in files:
                if f == "clash_royale_cards_1.json":
                    candidates.append(os.path.join(root, f))
        if not candidates:
            raise FileNotFoundError("Could not locate clash_royale_cards_1.json")
        cached_file = candidates[0]

    with open(cached_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Clean illegal trailing commas
    cleaned = re.sub(r",\s*([\]\}])", r"\1", text)
    data = json.loads(cleaned)
    items = data.get("items", [])
    print(f"[Cards Dataset] Loaded {len(items)} cards from {cached_file}")
    return items


def generate_feature_maps(items):
    """Parses attributes and maps them to CR-Zero card vocabulary."""
    air_target_cards = set()
    ground_target_cards = set()
    swarm_cards = set()
    card_types = {}
    mobilities = {}

    for item in items:
        name = item.get("name", "").strip().lower()
        targets = item.get("targets", "")
        ctype = item.get("type", "troop")
        group_card = item.get("groupCard", False)
        mobility = item.get("mobility", "ground")

        card_types[name] = ctype
        mobilities[name] = mobility

        if targets in ("both", "air"):
            air_target_cards.add(name)
        if targets in ("both", "ground") or ctype == "spell":
            ground_target_cards.add(name)
        if group_card:
            swarm_cards.add(name)

    print(f"[Features] Air targeting cards count: {len(air_target_cards)}")
    print(f"[Features] Swarm cards count: {len(swarm_cards)}")
    return air_target_cards, ground_target_cards, swarm_cards, card_types, mobilities


def main():
    items = fetch_card_metadata()
    air_targeting, ground_targeting, swarms, card_types, mobilities = generate_feature_maps(items)

    print("\n[Card Coverage in CR-Zero Database]:")
    matched = 0
    for card_id, info in CARD_DATABASE.items():
        if card_id == 0:
            continue
        cname = info["name"].lower()
        if cname in card_types:
            matched += 1
    print(f"Matched {matched}/{len(CARD_DATABASE)-1} cards in CARD_DATABASE with official metadata.")


if __name__ == "__main__":
    main()

