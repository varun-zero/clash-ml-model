"""Focused tests for the MuMu perception adapter and background ADB input."""

from pathlib import Path
import struct
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from deploy_mumu import (  # noqa: E402
    ADBController,
    BackgroundCapture,
    PerceptionAdapter,
    WindowsMouseController,
    draw_hud,
    update_artifact_log,
)
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
    incoming_card = adapter.tracked_next
    adapter.current_elixir = 10.0

    adapter.update_hand_on_play(0)

    assert adapter.current_elixir == 10.0 - expected_cost
    assert adapter.tracked_hand[0] == incoming_card
    assert adapter.tracked_next != incoming_card


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


def test_mouse_controller_dry_run_never_needs_a_window():
    mouse = WindowsMouseController(bbox={"left": 100, "top": 50, "width": 720, "height": 1280}, dry_run=True)
    try:
        mouse.set_screen_dimensions(720, 1280)
        assert mouse.deploy_card(
            card_slot=1,
            tile_x=4,
            tile_y=12,
            arena_box={"top": 0.05, "bottom": 0.78, "left": 0.03, "right": 0.97},
            hand_xs=[0.22, 0.41, 0.60, 0.79],
            hand_box_y=0.88,
        )
        # Test wait action
        assert mouse.deploy_card(
            card_slot=4,
            tile_x=0,
            tile_y=0,
            arena_box={"top": 0.05, "bottom": 0.78, "left": 0.03, "right": 0.97},
            hand_xs=[0.22, 0.41, 0.60, 0.79],
            hand_box_y=0.88,
        )
    finally:
        mouse.close()


def test_mouse_controller_refuses_live_host_input():
    mouse = WindowsMouseController(bbox={"left": 0, "top": 0, "width": 720, "height": 1280}, dry_run=False)
    with pytest.raises(RuntimeError, match="cannot inject host input"):
        mouse.deploy_card(
            card_slot=1,
            tile_x=4,
            tile_y=12,
            arena_box={"top": 0.05, "bottom": 0.78, "left": 0.03, "right": 0.97},
            hand_xs=[0.22, 0.41, 0.60, 0.79],
            hand_box_y=0.88,
        )


def test_decode_raw_screencap_rgba():
    width, height = 120, 120
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[..., 0] = 10   # R
    pixels[..., 1] = 20   # G
    pixels[..., 2] = 30   # B
    pixels[..., 3] = 255  # A
    payload = struct.pack("<III", width, height, 1) + pixels.tobytes()

    frame = BackgroundCapture._decode_raw_screencap(payload)
    assert frame is not None
    assert frame.shape == (height, width, 3)
    assert tuple(int(v) for v in frame[0, 0]) == (30, 20, 10)  # BGR


def test_perception_calibrates_widescreen_letterboxing():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    # Widescreen frame 1920x1080 (aspect ratio 1.77 > 0.65)
    widescreen_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    adapter.calibrate_game_column(widescreen_frame)

    # In 1080p, 9:16 column is target_w = 1080 * 9/16 = 607 pixels
    # Padding on each side is (1920 - 607) / 2 = 656 pixels
    # Left ratio ~ 656/1920 = 0.34
    assert adapter.arena_box["left"] > 0.20
    assert adapter.arena_box["right"] < 0.80


def test_draw_hud_renders_without_error():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    mouse = WindowsMouseController(bbox={"left": 0, "top": 0, "width": 544, "height": 1016}, dry_run=True)
    frame = np.zeros((1016, 544, 3), dtype=np.uint8)
    hud = draw_hud(
        frame=frame,
        perception=adapter,
        controller=mouse,
        action=(0, 100),
        value_est=0.42,
        fps=60.0
    )
    assert hud is not None
    assert hud.shape == frame.shape
    assert hud.dtype == np.uint8


def test_adb_controller_supports_swipe_and_tap_modes():
    adb_swipe = ADBController(dry_run=True, use_swipe=True)
    adb_tap = ADBController(dry_run=True, use_swipe=False)
    try:
        assert adb_swipe.tap_pixel(100, 200) is True
        assert adb_swipe.deploy_card(
            card_slot=0,
            tile_x=4,
            tile_y=6,
            arena_box={"top": 0.05, "bottom": 0.78, "left": 0.03, "right": 0.97},
            hand_xs=[0.22, 0.41, 0.60, 0.79],
            hand_box_y=0.88,
        ) is True

        assert adb_tap.deploy_card(
            card_slot=1,
            tile_x=12,
            tile_y=14,
            arena_box={"top": 0.05, "bottom": 0.78, "left": 0.03, "right": 0.97},
            hand_xs=[0.22, 0.41, 0.60, 0.79],
            hand_box_y=0.88,
        ) is True
    finally:
        adb_swipe.close()
        adb_tap.close()


def test_perception_detects_buttons_on_synthetic_frames():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    # Create 900x1600 frame
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)

    # 1. Draw yellow button at Home Screen Battle location: (450, 1200), w=200, h=80
    # Yellow in BGR: B=0, G=200, R=240 -> HSV hue ~25
    frame[1160:1240, 350:550] = [0, 200, 240]
    btn = adapter.detect_home_battle_button(frame)
    assert btn is not None
    assert 400 <= btn[0] <= 500
    assert 1150 <= btn[1] <= 1250

    # 2. Draw blue button at post-match OK location: (450, 1450), w=160, h=60
    # Cyan/blue in BGR: B=240, G=180, R=0 -> HSV hue ~105
    frame_post = np.zeros((1600, 900, 3), dtype=np.uint8)
    frame_post[1420:1480, 370:530] = [240, 180, 0]
    post_btn = adapter.detect_post_match_buttons(frame_post)
    assert post_btn is not None
    assert post_btn[0] == "ok"
    assert 400 <= post_btn[1] <= 500
    assert 1400 <= post_btn[2] <= 1500


def test_score_match_and_artifact_logging(tmp_path):
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    # Synthetic frame with a player gold crown in player region (yr ~ 0.45)
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)
    # Gold in HSV is Hue 15-35, Sat 140-255, Val 140-255. In BGR ~ [0, 200, 240]
    # Place a 90x55 crown at y=720 (yr = 720/1600 = 0.45)
    frame[700:755, 400:495] = [0, 215, 255]

    result, p_crowns, o_crowns, delta = adapter.score_match(frame)
    assert result == "WIN"
    assert p_crowns == 1
    assert o_crowns == 0
    assert delta == 30

    # Test update_artifact_log
    history = [{
        "match": 1,
        "opponent": "TestOpponent",
        "result": "WIN",
        "score": "1 - 0",
        "delta": 30,
        "trophies": 89,
        "duration_s": 180.0,
        "cards_deployed": 25,
        "notes": "Verified swipe deployment",
        "timestamp": "2026-11-12 19:00:00"
    }]
    artifact_file = tmp_path / "test_climb_log.md"
    update_artifact_log(history, current_trophies=89, target_trophies=500, artifact_path=str(artifact_file))
    assert artifact_file.exists()
    content = artifact_file.read_text(encoding="utf-8")
    assert "89" in content
    assert "500" in content
    assert "TestOpponent" in content


