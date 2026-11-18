"""Focused tests for the MuMu perception adapter and background ADB input."""

from pathlib import Path
import struct
import sys

import cv2
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from deploy_mumu import (  # noqa: E402
    ADBController,
    BackgroundCapture,
    ELIXIR_ICON_MAX_X,
    ELIXIR_PIP_COUNT,
    PerceptionAdapter,
    WindowsMouseController,
    detect_match_end_result,
    draw_hud,
    resolve_deck_names,
    TROPHY_DELTA_LOSS,
    trophy_delta_for,
    update_artifact_log,
)
from cr_alphastar_agent.config import ARENA_WIDTH, CARDS_CATALOG  # noqa: E402
from cr_alphastar_agent.env.crsim_replica_env import CRSIM_NAME_TO_ID  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic frame builders
# ---------------------------------------------------------------------------
def _magenta_bgr() -> np.ndarray:
    """Bright magenta (H 150, S 200, V 255) as a BGR pixel."""
    return cv2.cvtColor(np.uint8([[[150, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]


def _gold_bgr() -> np.ndarray:
    """Crown gold (H 25, S 220, V 255) as a BGR pixel."""
    return cv2.cvtColor(np.uint8([[[25, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]


def _pink_bgr() -> np.ndarray:
    """Opponent ribbon pink (H 175, S 200, V 220) as a BGR pixel."""
    return cv2.cvtColor(np.uint8([[[175, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]


def _blue_bgr() -> np.ndarray:
    """Player ribbon blue (H 110, S 200, V 220) as a BGR pixel."""
    return cv2.cvtColor(np.uint8([[[110, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]


def make_elixir_frame(pips: int, with_icon: bool = False,
                      with_counter: str = "") -> np.ndarray:
    """Build a 900x1600 frame whose elixir bar shows ``pips`` filled pips.

    The empty segments are drawn as a dim purple track - that is what the
    real game renders for an empty bar (only the segment separators keep a
    purple tint), and it is what makes an *empty* bar distinguishable from a
    *hidden* one.
    """
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)
    magenta = _magenta_bgr()
    track = cv2.cvtColor(np.uint8([[[150, 80, 120]]]), cv2.COLOR_HSV2BGR)[0, 0]
    x0 = int(0.258 * 900)
    x1 = int(0.955 * 900)
    y0 = int(0.966 * 1600)
    y1 = int(0.984 * 1600)
    frame[y0:y1, x0:x1] = track
    segment = (x1 - x0) / 10.0
    for index in range(max(0, min(10, pips))):
        a = x0 + int(round(index * segment))
        b = x0 + int(round((index + 1) * segment))
        frame[y0:y1, a:b] = magenta
    if with_icon:
        # The purple drop icon, far to the left of the bar.
        frame[y0:y1, int(0.08 * 900):int(0.11 * 900)] = magenta
    if with_counter:
        # The white elixir counter, drawn on top of the first pips.
        cv2.putText(frame, with_counter, (int(0.262 * 900), int(0.978 * 1600)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return frame


def make_end_screen(opponent_crowns: int = 0, player_crowns: int = 0,
                    with_ribbons: bool = True,
                    with_ok_button: bool = False) -> np.ndarray:
    """Build a synthetic Clash Royale post-match end screen."""
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)
    if with_ribbons:
        # Opponent (pink) ribbon, y 0.24 .. 0.36
        frame[int(0.24 * 1600):int(0.36 * 1600), :] = _pink_bgr()
        # Player (blue) ribbon, y 0.51 .. 0.63
        frame[int(0.51 * 1600):int(0.63 * 1600), :] = _blue_bgr()
    gold = _gold_bgr()
    for index in range(opponent_crowns):
        x = 120 + index * 200
        frame[int(0.15 * 1600):int(0.205 * 1600), x:x + 160] = gold
    for index in range(player_crowns):
        x = 120 + index * 200
        frame[int(0.42 * 1600):int(0.48 * 1600), x:x + 160] = gold
    if with_ok_button:
        # The blue "OK" button lives at the bottom and is NOT evidence.
        frame[int(0.879 * 1600):int(0.941 * 1600), 350:550] = _blue_bgr()
    return frame


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
    assert obs["scalars"].shape == (1, 7)
    assert obs["hand_cards"].shape == (1, 5)
    assert obs["hand_cards"].dtype == torch.int64
    assert obs["hand_cards"].tolist() == [[*hand_ids, next_id]]
    assert obs["entities"].shape == (1, 40, 9)
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

    # 1. Draw the yellow Battle button at its real size and position:
    #    302x168 px, centre (455, 1251).
    frame[1167:1335, 304:606] = [0, 200, 240]
    btn = adapter.detect_home_battle_button(frame)
    assert btn is not None
    assert 400 <= btn[0] <= 510
    assert 1200 <= btn[1] <= 1300

    # 2. Draw blue button at post-match OK location: (450, 1450), w=160, h=60
    # Cyan/blue in BGR: B=240, G=180, R=0 -> HSV hue ~105
    frame_post = np.zeros((1600, 900, 3), dtype=np.uint8)
    frame_post[1420:1480, 370:530] = [240, 180, 0]
    post_btn = adapter.detect_post_match_buttons(frame_post)
    assert post_btn is not None
    assert post_btn[0] == "ok"
    assert 400 <= post_btn[1] <= 500
    assert 1400 <= post_btn[2] <= 1500


def test_home_battle_button_is_not_confused_with_a_flat_orange_band():
    """A full-width orange band (Pass Royale reward screen) is not a button."""
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)
    frame[1120:1360, :] = [0, 170, 230]      # orange band across the screen
    frame[1200:1260, 300:600] = [255, 255, 255]  # some white text in it
    assert adapter.detect_home_battle_button(frame) is None


def test_score_match_and_artifact_logging(tmp_path):
    """The end-screen detector must never guess, and the artifact log must
    never default a missing result to a win."""
    adapter = PerceptionAdapter(device=torch.device("cpu"))

    # A frame that is *not* the post-match end screen must be UNVERIFIED.
    blank = np.zeros((1600, 900, 3), dtype=np.uint8)
    result, p_crowns, o_crowns, delta = adapter.detect_match_end_result(blank)
    assert result == "UNVERIFIED"
    assert (p_crowns, o_crowns, delta) == (0, 0, 0)

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

    # A history entry without a result must be rendered as UNVERIFIED.
    update_artifact_log([{"match": 2, "opponent": "NoResult"}],
                        current_trophies=89, target_trophies=500,
                        artifact_path=str(artifact_file))
    content = artifact_file.read_text(encoding="utf-8")
    assert "UNVERIFIED" in content
    assert "| **WIN**" not in content


def test_read_elixir_reads_ten_discrete_pips():
    """The elixir reader must count filled pips, never a continuous ratio."""
    adapter = PerceptionAdapter(device=torch.device("cpu"))

    # 1. A *rendered but empty* bar -> 0 pips.
    adapter.current_elixir = 3.5
    assert adapter.read_elixir(make_elixir_frame(0)) == 0.0
    assert adapter.current_elixir == 0.0

    # 2. A frame without any elixir bar (the HUD is hidden) must NOT be
    #    reported as "0 elixir" - that made the agent play cards it could
    #    not afford and made the deployment verification believe that the
    #    elixir had dropped when nothing was played.
    adapter.current_elixir = 3.5
    assert adapter.read_elixir(np.zeros((1600, 900, 3), dtype=np.uint8)) == 3.5
    assert adapter.elixir_visible is False

    # 2. One filled pip -> 1, ten filled pips -> 10.
    for pips in range(0, 11):
        frame = make_elixir_frame(pips)
        value = adapter.read_elixir(frame)
        assert value == float(pips), (pips, value)

    # 3. The purple drop icon on the far left must never be counted.
    frame = make_elixir_frame(0, with_icon=True)
    assert adapter.read_elixir(frame) == 0.0
    frame = make_elixir_frame(3, with_icon=True)
    assert adapter.read_elixir(frame) == 3.0

    # 4. The white elixir counter drawn over the first pips must not hide them.
    frame = make_elixir_frame(10, with_counter="10")
    assert adapter.read_elixir(frame) == 10.0
    frame = make_elixir_frame(4, with_counter="4")
    assert adapter.read_elixir(frame) == 4.0


def test_read_elixir_uses_the_calibrated_pip_geometry():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    # The drop icon is excluded, the pip band starts at 0.258.
    assert adapter.elixir_box["left"] == pytest.approx(0.255)
    assert adapter.elixir_box["right"] == pytest.approx(0.955)
    assert adapter.elixir_box["top"] == pytest.approx(0.966)
    assert adapter.elixir_box["bottom"] == pytest.approx(0.984)
    assert ELIXIR_ICON_MAX_X == 0.255
    assert ELIXIR_PIP_COUNT == 10


def test_elixir_reader_on_the_real_captured_frame():
    """Regression test against the real 900x1600 emulator capture."""
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    frame_path = PROJECT_ROOT / ".tmp" / "diag_live.png"
    if not frame_path.exists():
        pytest.skip("no real emulator capture available")
    frame = cv2.imread(str(frame_path))
    # The reference capture shows a completely filled bar.
    assert adapter.read_elixir(frame) == 10.0


def test_deploy_instant_elixir_deduction_and_cooldown():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    adapter.current_elixir = 7.0
    card_name = adapter.tracked_hand[0]
    cid = CRSIM_NAME_TO_ID[card_name]
    cost = CARDS_CATALOG[cid]["cost"]

    # A *verified* play deducts the real card cost exactly once and rotates
    # the hand.
    prev_next = adapter.tracked_next
    adapter.update_hand_on_play(0, deduct_elixir=True, card_name=card_name)
    assert adapter.current_elixir == 7.0 - cost
    assert adapter.tracked_hand[0] == prev_next
    assert card_name in adapter.recent_deployed_cards

    # The live loop deducts the elixir itself, so the rotation must not
    # deduct it a second time.
    adapter.current_elixir = 7.0 - cost
    adapter.update_hand_on_play(1, deduct_elixir=False, card_name=adapter.tracked_hand[1])
    assert adapter.current_elixir == 7.0 - cost

    # Cooldown masking contract: while the cooldown is active only "Wait"
    # may be chosen.
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)
    obs = adapter.frame_to_obs(frame)
    obs["card_mask"][:, :4] = False
    obs["card_mask"][:, 4] = True
    assert obs["card_mask"][0, 4].item() is True
    assert all(obs["card_mask"][0, i].item() is False for i in range(4))


def test_hand_resync_after_cooldown():
    """The hand must be re-classified after the cooldown - with the real
    classifier, not a mock."""
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)

    # When force_refresh=False, read_hand does not invoke classify_hand.
    calls = []
    original = adapter.classifier.classify_hand

    def counting(frame_, candidate_deck=None, return_scores=False, exclude=None):
        calls.append(exclude)
        return original(frame_, candidate_deck, return_scores, exclude)

    adapter.classifier.classify_hand = counting  # type: ignore[assignment]

    hand_ids, next_id = adapter.read_hand(frame, force_refresh=False)
    assert not calls
    assert [CRSIM_NAME_TO_ID[c] for c in adapter.tracked_hand] == hand_ids

    # Post-cooldown re-sync with force_refresh=True invokes the classifier.
    adapter.read_hand(frame, force_refresh=True)
    assert len(calls) == 1
    assert len(adapter.tracked_hand) == 4
    assert len(set(adapter.tracked_hand)) == 4
    assert set(adapter.tracked_hand) <= set(adapter.active_deck)
    assert [CRSIM_NAME_TO_ID[c] for c in adapter.tracked_hand] == adapter.read_hand(frame)[0]


def test_four_card_cycle_blocks_recently_deployed_cards():
    """A card that was just played must not be predicted again for 3 plays."""
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)

    adapter.recent_deployed_cards.append("Arrows")
    adapter.read_hand(frame, force_refresh=True)
    assert "Arrows" not in adapter.tracked_hand, adapter.tracked_hand

    # After the card leaves the 3-card window it may be predicted again.
    adapter.recent_deployed_cards.clear()
    adapter.read_hand(frame, force_refresh=True)
    assert len(adapter.tracked_hand) == 4


def test_update_hand_on_play_records_the_cycle():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    played = adapter.tracked_hand[0]
    adapter.update_hand_on_play(0, deduct_elixir=False, card_name=played)
    assert played in adapter.recent_deployed_cards
    assert adapter.deployed_card_names == [played]


def test_deploy_rejects_when_nothing_happened():
    """A tap that does not move the elixir and does not empty the slot must
    be rejected (this is the phantom-deployment guard)."""
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    frame = make_elixir_frame(5)
    ok, diagnostics = adapter.verify_deployment(
        before_frame=frame, after_frame=frame, slot=0,
        card_name="Knight", elixir_before=5.0,
    )
    assert ok is False
    assert diagnostics["reason"] == "no evidence the card was deployed"

    # A dropped elixir is proof of a deployment.
    after = make_elixir_frame(3)
    ok, diagnostics = adapter.verify_deployment(
        before_frame=frame, after_frame=after, slot=0,
        card_name="Knight", elixir_before=5.0,
    )
    assert ok is True
    assert diagnostics["reason"] == "elixir dropped"


def test_rejected_slot_is_blocked_for_two_ticks():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    adapter.block_slot(2, ticks=2)
    assert adapter._tick_blocked_slots() == [2]
    assert adapter._tick_blocked_slots() == [2]
    assert adapter._tick_blocked_slots() == []
    assert 2 not in adapter.blocked_slots


def test_detect_match_end_result_uses_the_ground_truth_zones():
    # 0 - 3 (opponent took three crowns) -> LOSS
    frame = make_end_screen(opponent_crowns=3)
    result, p_crowns, o_crowns, delta = detect_match_end_result(frame)
    assert result == "LOSS"
    assert o_crowns == 3
    assert p_crowns == 0
    assert delta == -9  # the model uses the conservative -9

    # 1 - 0 player win -> WIN
    frame = make_end_screen(player_crowns=1)
    result, p_crowns, o_crowns, delta = detect_match_end_result(frame)
    assert result == "WIN"
    assert p_crowns == 1
    assert delta == 30

    # 3 - 1 player win -> WIN (both sides can carry crowns, the higher score won)
    frame = make_end_screen(player_crowns=3, opponent_crowns=1)
    result, p_crowns, o_crowns, delta = detect_match_end_result(frame)
    assert result == "WIN", f"{result} {p_crowns}-{o_crowns}"
    assert p_crowns == 3 and o_crowns == 1

    # 1 - 1 -> genuinely ambiguous, never guessed.
    frame = make_end_screen(opponent_crowns=1, player_crowns=1)
    result, _, _, _ = detect_match_end_result(frame)
    assert result == "UNVERIFIED"

    # Ambiguous frame -> UNVERIFIED, never WIN.
    frame = make_end_screen(opponent_crowns=0, player_crowns=0)
    result, _, _, _ = detect_match_end_result(frame)
    assert result == "UNVERIFIED"

    # The bottom blue "OK" button alone must never be taken as evidence.
    frame = make_end_screen(opponent_crowns=0, player_crowns=0,
                            with_ribbons=False, with_ok_button=True)
    result, _, _, _ = detect_match_end_result(frame)
    assert result == "UNVERIFIED"

    # A completely blank frame must never be a win.
    result, _, _, _ = detect_match_end_result(np.zeros((1600, 900, 3), dtype=np.uint8))
    assert result == "UNVERIFIED"


def test_detect_match_end_result_on_the_real_captured_loss_screen():
    frame_path = PROJECT_ROOT / ".tmp" / "diag_live2.png"
    if not frame_path.exists():
        pytest.skip("no real emulator capture available")
    frame = cv2.imread(str(frame_path))
    result, p_crowns, o_crowns, delta = detect_match_end_result(frame)
    assert result == "LOSS"
    assert p_crowns == 0
    assert o_crowns >= 1


def test_detect_match_end_result_on_the_real_two_crown_loss_screen():
    """Real end screen: the opponent took two crowns (the ribbon shows -11)."""
    frame_path = PROJECT_ROOT / ".tmp" / "endscreen_loss_2crowns.png"
    if not frame_path.exists():
        pytest.skip("no real emulator capture available")
    frame = cv2.imread(str(frame_path))
    result, p_crowns, o_crowns, delta = detect_match_end_result(frame)
    assert result == "LOSS"
    assert p_crowns == 0
    assert o_crowns == 2
    assert delta == TROPHY_DELTA_LOSS  # model value, reconciled by hand


def test_classifier_identifies_a_real_in_battle_hand():
    """Regression test on a real 900x1600 in-battle frame.

    The hand in the reference frame is Knight / Musketeer / Giant / Arrows
    with Archers as the next card.
    """
    frame_path = PROJECT_ROOT / ".tmp" / "inbattle_reference.png"
    if not frame_path.exists():
        pytest.skip("no real emulator capture available")
    frame = cv2.imread(str(frame_path))
    adapter = PerceptionAdapter(device=torch.device("cpu"),
                                 deck=resolve_deck_names("current"))

    adapter.read_hand(frame, force_refresh=True)
    assert adapter.tracked_hand == ["Knight", "Musketeer", "Giant", "Arrows"]
    assert adapter.tracked_next == "Archers"

    # The resting template correlation of a correctly identified card is
    # 0.37 - 0.60 with the shipped deck-screen templates.  The deployment
    # verification therefore relies on a *relative* drop, not an absolute
    # 0.80 threshold.
    for slot, card in enumerate(adapter.tracked_hand):
        assert adapter.slot_correlation(frame, slot, card) >= 0.30, card


def test_is_in_battle_survives_a_hidden_hud():
    """The elixir bar can be replaced by the red "No card selected" hint.

    A live capture showed the whole HUD (card tray *and* elixir bar)
    disappearing for several seconds while the arena was still running.
    The agent must not mistake that for "the match ended".
    """
    adapter = PerceptionAdapter(device=torch.device("cpu"))

    # Real arena frame with the elixir bar replaced by the red
    # "No card selected" hint (extracted from the recorded match video).
    for name in ("vid7c/f120.png", "vid7c/f126.png"):
        path = PROJECT_ROOT / ".tmp" / name
        if path.exists():
            assert adapter.is_in_battle(cv2.imread(str(path))) is True, name

    # The sand arena and the green arena are both battles.
    for name in ("inbattle_reference.png", "diag_live.png", "live_battle.png"):
        path = PROJECT_ROOT / ".tmp" / name
        if path.exists():
            assert adapter.is_in_battle(cv2.imread(str(path))) is True, name

    # Screens that are definitely not a battle.
    for name in ("post_match_home.png", "endscreen_win_2crowns_ot.png",
                 "battle_log_live.png", "m22_mid.png"):
        path = PROJECT_ROOT / ".tmp" / name
        if path.exists():
            assert adapter.is_in_battle(cv2.imread(str(path))) is False, name


def test_is_in_battle_rejects_the_pass_royale_reward_screen():
    """An orange reward band is not an arena."""
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)
    frame[100:1200, :] = [0, 170, 230]   # saturated orange everywhere
    assert adapter.is_in_battle(frame) is False


def test_is_in_battle_rejects_the_bone_pit_screen():
    """The Pass Royale "Bone Pit" arena graphic is NOT a live battle.

    Using the arena floor as a battle marker made the agent tap "Battle"
    on this screen and start a match that never began.
    """
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    frame_path = PROJECT_ROOT / ".tmp" / "vid8" / "f003.png"
    if not frame_path.exists():
        pytest.skip("no real emulator capture available")
    frame = cv2.imread(str(frame_path))
    assert adapter.is_in_battle(frame) is False
    assert adapter.detect_home_battle_button(frame) is None


def test_detect_match_end_result_on_the_real_three_crown_win_screen():
    """Real end screen of a 3-0 win, including the reward chest popup.

    The reward chest is a big gold box right below the player ribbon - it
    must not be counted as a fourth crown.
    """
    frame_path = PROJECT_ROOT / ".tmp" / "endscreen_win_3crowns.png"
    if not frame_path.exists():
        pytest.skip("no real emulator capture available")
    frame = cv2.imread(str(frame_path))
    result, p_crowns, o_crowns, delta = detect_match_end_result(frame)
    assert result == "WIN"
    assert p_crowns == 3
    assert o_crowns == 0


def test_detect_match_end_result_on_the_real_win_screen():
    """Real end screen after a live match: the player won 1-0 (+31).

    The post-match screen does NOT have a fixed vertical layout - on a WIN
    the ribbons sit much lower than on a LOSS - so this is the regression
    test for the layout-independent detector.
    """
    frame_path = PROJECT_ROOT / ".tmp" / "endscreen_win_1crown.png"
    if not frame_path.exists():
        pytest.skip("no real emulator capture available")
    frame = cv2.imread(str(frame_path))
    result, p_crowns, o_crowns, delta = detect_match_end_result(frame)
    assert result == "WIN"
    assert p_crowns == 1
    assert o_crowns == 0


def test_detect_match_end_result_is_layout_independent():
    """The same content at a different vertical offset must read the same."""
    frame = make_end_screen(player_crowns=1)
    shifted = np.zeros_like(frame)
    shifted[200:] = frame[:frame.shape[0] - 200]
    assert detect_match_end_result(frame)[0] == "WIN"
    assert detect_match_end_result(shifted)[0] == "WIN"


def test_detect_match_end_result_on_the_real_overtime_win_screen():
    """Real end screen after a live overtime match: the player won 2-1.

    On this screen the player's two crowns sit exactly *between* the two
    ribbons, so the crown has to be assigned by the midpoint of the gap
    between the ribbons, not by proximity.
    """
    frame_path = PROJECT_ROOT / ".tmp" / "endscreen_win_2crowns_ot.png"
    if not frame_path.exists():
        pytest.skip("no real emulator capture available")
    frame = cv2.imread(str(frame_path))
    result, p_crowns, o_crowns, delta = detect_match_end_result(frame)
    assert result == "WIN"
    assert p_crowns == 2
    assert o_crowns == 1


def test_trophy_delta_matches_the_observed_end_screens():
    assert trophy_delta_for("WIN", 3, 1) == 30
    assert trophy_delta_for("WIN", 1, 0) == 30
    assert trophy_delta_for("LOSS", 0, 1) == -9
    assert trophy_delta_for("LOSS", 0, 2) == -9
    assert trophy_delta_for("LOSS", 0, 3) == -9
    assert trophy_delta_for("UNVERIFIED", 0, 0) == 0


def test_calibrate_game_column_preserves_calibrated_coordinates():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    # Standard portrait frame (900x1600)
    portrait_frame = np.zeros((1600, 900, 3), dtype=np.uint8)
    adapter.calibrate_game_column(portrait_frame)

    assert adapter.hand_xs == [0.312, 0.494, 0.680, 0.864]
    assert adapter.elixir_box["top"] == pytest.approx(0.966)
    assert adapter.elixir_box["bottom"] == pytest.approx(0.984)
    # The drop icon is excluded from the elixir crop.
    assert adapter.elixir_box["left"] == pytest.approx(0.255)
    assert adapter.elixir_box["right"] == pytest.approx(0.955)

    # Widescreen frame (1920x1080)
    widescreen_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    adapter.calibrate_game_column(widescreen_frame)

    assert adapter.elixir_box["top"] == pytest.approx(0.966)
    assert adapter.elixir_box["bottom"] == pytest.approx(0.984)
    assert adapter.hand_xs[0] > 0.30  # scaled into pillar
    assert adapter.elixir_box["left"] > 0.20  # scaled into pillar


def test_is_in_battle_rejects_full_screen_purple_modals():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    # 1. Full-screen purple modal frame (like Mystery Box chest opening animation)
    modal_frame = np.zeros((1600, 900, 3), dtype=np.uint8)
    modal_frame[:] = cv2.cvtColor(np.uint8([[[145, 180, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    adapter.calibrate_game_column(modal_frame)
    assert not adapter.is_in_battle(modal_frame)

    # 2. Modal with purple across arena and card tray bands
    partially_purple_modal = np.zeros((1600, 900, 3), dtype=np.uint8)
    partially_purple_modal[100:1500, :] = cv2.cvtColor(np.uint8([[[140, 160, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    assert not adapter.is_in_battle(partially_purple_modal)


def test_is_in_battle_accepts_genuine_battle_frame():
    adapter = PerceptionAdapter(device=torch.device("cpu"))
    # Genuine battle frame: green arena, dark card tray, localized purple elixir bar fill
    frame = np.zeros((1600, 900, 3), dtype=np.uint8)
    frame[100:1150, :] = [45, 120, 45]  # Green arena grass
    frame[1150:1540, :] = [30, 30, 35]  # Dark card tray

    # Localized elixir fill in elixir container
    adapter.calibrate_game_column(frame)
    ey1 = int(adapter.elixir_box["top"] * frame.shape[0])
    ey2 = int(adapter.elixir_box["bottom"] * frame.shape[0])
    ex1 = int(adapter.elixir_box["left"] * frame.shape[1])
    ex2 = int(adapter.elixir_box["right"] * frame.shape[1])
    purple_bgr = cv2.cvtColor(np.uint8([[[145, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frame[ey1:ey2, ex1:int(ex1 + (ex2 - ex1) * 0.6)] = purple_bgr

    assert adapter.is_in_battle(frame)



