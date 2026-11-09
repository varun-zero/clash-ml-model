"""Focused tests for the BlueStacks perception adapter and dry-run input."""

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from deploy_bluestacks import ADBController, PerceptionAdapter  # noqa: E402
from cr_alphastar_agent.config import CARDS_CATALOG  # noqa: E402
from cr_alphastar_agent.env.crsim_replica_env import CRSIM_NAME_TO_ID  # noqa: E402


def make_frame(adapter: PerceptionAdapter) -> np.ndarray:
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    frame[64:1000, 20:700] = [60, 140, 60]

    y1 = int(adapter.elixir_box["top"] * frame.shape[0])
    y2 = int(adapter.elixir_box["bottom"] * frame.shape[0])
    x1 = int(adapter.elixir_box["left"] * frame.shape[1])
    x2 = int(adapter.elixir_box["right"] * frame.shape[1])
    frame[y1:y2, x1:int(x1 + (x2 - x1) * 0.6)] = [180, 0, 220]
    return frame


def test_frame_to_obs_encodes_tracked_card_names_as_policy_ids():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    frame = make_frame(adapter)

    hand_ids, next_id = adapter.read_hand(frame)
    assert hand_ids == [
        CRSIM_NAME_TO_ID[name] for name in adapter.tracked_hand
    ]
    assert next_id == CRSIM_NAME_TO_ID[adapter.tracked_next]

    obs = adapter.frame_to_obs(frame)
    assert obs["scalars"].shape == (1, 4)
    assert obs["hand_cards"].shape == (1, 5)
    assert obs["hand_cards"].dtype == torch.int64
    assert obs["hand_cards"].tolist() == [[*hand_ids, next_id]]
    assert obs["entities"].shape == (1, 40, 7)
    assert obs["entity_mask"].shape == (1, 40)
    assert obs["arena_grid"].shape == (1, 6, 32, 18)
    assert obs["card_mask"].shape == (1, 5)
    assert obs["tile_mask"].shape == (1, 576)
    assert all(not obs["entity_mask"][0, i] for i in range(6))


def test_hand_rotation_deducts_the_actual_card_cost():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    played_name = adapter.tracked_hand[0]
    played_id = CRSIM_NAME_TO_ID[played_name]
    expected_cost = CARDS_CATALOG[played_id]["cost"]
    adapter.current_elixir = 10.0

    adapter.update_hand_on_play(0)

    assert adapter.current_elixir == 10.0 - expected_cost
    assert adapter.tracked_hand[0] == "Skeletons"
    assert adapter.tracked_next == played_name


def test_adb_controller_dry_run_never_needs_a_device():
    adb = ADBController(dry_run=True)
    try:
        adb.set_screen_dimensions(720, 1280)
        assert adb.deploy_card(
            card_slot=0,
            tile_x=8,
            tile_y=8,
            arena_box={"top": 0.05, "bottom": 0.78, "left": 0.03, "right": 0.97},
            hand_xs=[0.22, 0.41, 0.60, 0.79],
            hand_box_y=0.88,
        )
    finally:
        adb.close()
