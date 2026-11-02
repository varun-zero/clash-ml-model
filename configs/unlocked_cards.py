"""Load and validate the editable unlocked-card list.

The account's unlocked cards are global ``CARD_DATABASE`` IDs. Card templates
and the hand classifier are restricted to this list, so the detector never
guesses a card the account cannot own.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional, Sequence

from configs.constants import CARD_DATABASE, DEFAULT_UNLOCKED_CARD_IDS

DEFAULT_UNLOCKED_PATH = os.path.join(os.path.dirname(__file__), "unlocked_cards.json")


def _normalize_ids(raw_ids: Iterable[object]) -> List[int]:
    normalized: List[int] = []
    for raw in raw_ids:
        try:
            card_id = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unlocked card IDs must be integers, got {raw!r}") from exc
        if card_id == 0:
            raise ValueError("Card ID 0 is Wait and must not be in unlocked_cards.json")
        if card_id not in CARD_DATABASE:
            raise ValueError(
                f"Unlocked card ID {card_id} is not in the global CARD_DATABASE"
            )
        if card_id not in normalized:
            normalized.append(card_id)
    if not normalized:
        raise ValueError("unlocked_cards.json must contain at least one card ID")
    return normalized


def load_unlocked_cards(path: Optional[str] = None) -> List[int]:
    """Return validated global card IDs for this account.

    A missing file falls back to the starter list. A malformed file raises, so
    a typo cannot silently shrink the classifier's vocabulary.
    """
    resolved = path or DEFAULT_UNLOCKED_PATH
    if not os.path.exists(resolved):
        return list(DEFAULT_UNLOCKED_CARD_IDS)

    with open(resolved, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        raw_ids = data.get("unlocked_ids", data.get("cards"))
        if raw_ids is None:
            raise ValueError(f"{resolved} must contain an 'unlocked_ids' list")
    elif isinstance(data, list):
        raw_ids = data
    else:
        raise ValueError(f"{resolved} must contain a JSON list or object")

    if not isinstance(raw_ids, list):
        raise ValueError(f"'unlocked_ids' must be a JSON list in {resolved}")

    return _normalize_ids(raw_ids)


def filter_unlocked(card_ids: Sequence[int], unlocked_ids: Optional[Sequence[int]] = None) -> List[int]:
    """Keep only IDs present in the account's unlocked set."""
    allowed = set(unlocked_ids if unlocked_ids is not None else load_unlocked_cards())
    return [int(cid) for cid in card_ids if int(cid) in allowed]
