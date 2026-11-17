"""Tests for the self-growing battlefield unit identity memory."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cr_alphastar_agent.env.unit_identity import (  # noqa: E402
    CROP_H, CROP_W, UnitIdentityMemory,
)


def make_frame(marker=(30, 180, 220)):
    frame = np.full((1600, 900, 3), 70, dtype=np.uint8)
    cv2.rectangle(frame, (300 - 18, 800 - 24), (300 + 18, 800 + 24), marker, -1)
    return frame


def test_learn_and_identify(tmp_path):
    memory = UnitIdentityMemory(str(tmp_path / "templates"))
    frame = make_frame()
    assert memory.learn("MiniPekka", frame, 300, 800)
    assert len(memory) == 1

    found = memory.identify(frame, 300, 800)
    assert found is not None
    assert found.card_name == "MiniPekka"
    assert found.confidence >= 0.72


def test_near_duplicate_is_not_stored(tmp_path):
    memory = UnitIdentityMemory(str(tmp_path / "templates"))
    frame = make_frame()
    assert memory.learn("MiniPekka", frame, 300, 800)
    assert not memory.learn("MiniPekka", frame, 300, 800)
    assert len(memory) == 1


def test_different_card_is_not_confidently_matched(tmp_path):
    memory = UnitIdentityMemory(str(tmp_path / "templates"))
    memory.learn("MiniPekka", make_frame((30, 180, 220)), 300, 800)
    other = make_frame((220, 30, 30))
    found = memory.identify(other, 300, 800)
    assert found is None or found.confidence < 0.72


def test_memory_survives_restart(tmp_path):
    root = str(tmp_path / "templates")
    frame = make_frame()
    first = UnitIdentityMemory(root)
    first.learn("Knight", frame, 300, 800)
    assert "counts" in json.loads((Path(root) / "index.json").read_text())

    second = UnitIdentityMemory(root)
    assert len(second) == 1
    assert second.learned_cards() == ["Knight"]
    assert second.identify(frame, 300, 800).card_name == "Knight"


def test_multiple_templates_per_card(tmp_path):
    memory = UnitIdentityMemory(str(tmp_path / "templates"))
    for marker in [(30, 180, 220), (220, 30, 30), (30, 220, 30)]:
        memory.learn("Musketeer", make_frame(marker), 300, 800)
    assert len(memory) == 3
    assert memory.counts["Musketeer"] == 3
