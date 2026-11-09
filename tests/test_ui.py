"""Isolated UI API integration tests on an ephemeral localhost port."""

from http.server import HTTPServer
import json
from threading import Thread
from urllib.request import Request, urlopen

import pytest

from tools import ui


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get(self, path: str):
        with urlopen(self.base_url + path, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def post(self, path: str, payload: dict | None = None):
        request = Request(
            self.base_url + path,
            data=json.dumps(payload or {}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "CRSIM_ROOT", str(tmp_path))
    session = ui.SESSION
    old_state = (
        session.blue_ai_mode,
        session.red_ai_mode,
        session.paused,
        session.running,
    )
    session.playback_mode = False
    session.current_replay_data = None
    session.blue_ai_mode = "human"
    session.red_ai_mode = "off"
    session.paused = False
    session.reset("PEKKA Bridge Spam · sim roster", "Classic Log Bait")

    server = HTTPServer(("127.0.0.1", 0), ui.RequestHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ApiClient(f"http://127.0.0.1:{server.server_address[1]}")
    try:
        yield client
    finally:
        if session.playback_mode:
            session.exit_replay()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        session.blue_ai_mode, session.red_ai_mode, session.paused, session.running = old_state


def test_state_reports_both_four_card_hands_and_eight_card_cycles(api):
    state = api.get("/api/state")

    assert state["initialized"]
    assert len([unit for unit in state["units"] if unit["tower"]]) == 6
    assert len(state["hand"]) == len(state["red_hand"]) == 4
    assert len(state["cycle"][0]) == len(state["cycle"][1]) == 4
    assert set(card["name"] for card in state["hand"]) | set(state["cycle"][0]) == set(state["blue_deck"])
    assert set(card["name"] for card in state["red_hand"]) | set(state["cycle"][1]) == set(state["red_deck"])


def test_random_and_custom_deck_endpoints_are_validated_and_persist(api):
    catalog = api.get("/api/all_cards")
    assert len(catalog) == len({card["name"] for card in catalog}) == 133

    rolled = api.post("/api/reset", {
        "blue_deck_name": "Randomized Deck",
        "red_deck_name": "Randomized Deck",
    })
    assert rolled["ok"]
    state = api.get("/api/state")
    for deck in (state["blue_deck"], state["red_deck"]):
        assert len(deck) == len(set(deck)) == 8

    invalid = api.post("/api/reset", {
        "blue_deck_name": "Unknown Deck",
        "red_deck_name": "Classic Log Bait",
    })
    assert invalid["ok"] is False

    custom = ["Knight", "Fireball", "Arrows", "Zap", "Giant", "Musketeer", "Skeletons", "Minions"]
    result = api.post("/api/set_custom_deck", {"team": 0, "deck": custom})
    assert result["ok"]
    state = api.get("/api/state")
    assert state["blue_deck_name"] == "Custom Deck"
    assert state["blue_deck"] == custom

    api.post("/api/reset", {})
    assert api.get("/api/state")["blue_deck"] == custom


def test_replay_save_load_seek_reuses_the_match_seed(api):
    state = api.get("/api/state")
    opening_hand = [card["name"] for card in state["hand"]]
    seed = state["match_seed"]
    ui.SESSION.battle.elixir[0] = 10.0

    deployed = api.post("/api/deploy", {"hand_idx": 0, "x": 8.5, "y": 8.5})
    assert deployed["ok"]
    replay = api.get("/api/export_replay")
    assert replay["match_seed"] == seed
    assert len(replay["events"]) == 1

    saved = api.post("/api/save_replay")
    assert saved["ok"]
    loaded = api.post("/api/load_replay", {"filename": saved["filename"]})
    assert loaded["ok"]
    playback = api.get("/api/state")
    assert playback["playback_mode"]
    assert [card["name"] for card in playback["hand"]] == opening_hand

    seek = api.post("/api/seek_replay", {"t_ms": 15000})
    assert seek["ok"]
    assert api.get("/api/state")["t_ms"] >= 15000
    assert api.post("/api/exit_replay")["ok"]
    assert api.get("/api/state")["playback_mode"] is False
