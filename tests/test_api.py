"""API contract, including the loading state the UI depends on."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from fantasypicker import api as api_module
from fantasypicker.service import PickerService

from .test_sleeper import stub_transport


@pytest.fixture
def client(monkeypatch):
    """A fresh service per test, with Sleeper stubbed and warm-up disabled."""
    service = PickerService()
    monkeypatch.setattr(api_module, "service", service)

    async def fake_enter(self):
        self._client = httpx.AsyncClient(transport=stub_transport())
        self._owned = True
        return self

    monkeypatch.setattr("fantasypicker.sleeper.client.SleeperClient.__aenter__", fake_enter)
    monkeypatch.setattr(PickerService, "start_warmup", lambda self, **kw: None)
    monkeypatch.setattr(
        "fantasypicker.data.crosswalk.load_crosswalk", lambda players=None: _StubCrosswalk()
    )
    monkeypatch.setattr(
        "fantasypicker.service.load_crosswalk", lambda players=None: _StubCrosswalk()
    )
    monkeypatch.setattr("fantasypicker.service.load_expert_ranks", lambda *a, **k: None)
    return TestClient(api_module.app), service


class _StubCrosswalk:
    sleeper_to_gsis: dict = {}
    gsis_to_sleeper: dict = {}
    sleeper_to_fp: dict = {}
    fp_to_sleeper: dict = {}
    name_to_sleeper: dict = {}

    def gsis(self, sleeper_id):
        return None

    def by_name(self, name, position):
        return None


def test_status_before_connecting(client):
    http, _ = client
    payload = http.get("/api/status").json()
    assert payload["league"]["connected"] is False
    assert payload["status"]["ready"] is False
    # Remembered leagues are offered even when nothing is connected — that is
    # what the front page needs in order to show "pick up where you left off".
    assert payload["league"]["known_leagues"] == []


def test_remembered_leagues_are_offered_on_a_cold_start(client):
    http, service = client
    http.post("/api/connect", json={"league_id": "999", "username": "alice"})

    # A brand new process, same machine: the state file is what carries over.
    from fantasypicker.service import PickerService

    fresh = PickerService()
    known = fresh.known_leagues()
    assert [lg["league_id"] for lg in known] == ["999"]
    assert known[0]["my_team"] == "Alice's Aces"


def test_forget_removes_a_remembered_league(client):
    http, _ = client
    http.post("/api/connect", json={"league_id": "999", "username": "alice"})
    response = http.delete("/api/known/999")
    assert response.status_code == 200
    assert response.json()["forgotten"] is True
    assert http.get("/api/known").json()["leagues"] == []


def test_connect_returns_the_league_shape(client):
    http, _ = client
    response = http.post("/api/connect", json={"league_id": "999", "username": "alice"})
    assert response.status_code == 200
    league = response.json()["league"]
    assert league["name"] == "Sunday Sickos"
    assert league["teams"] == 12
    assert league["scoring"].startswith("half PPR")
    assert league["my_roster_id"] == 1
    assert [t["label"] for t in league["teams_list"]] == ["Alice's Aces", "bob"]


def test_connect_to_a_missing_league_is_a_404(client):
    http, _ = client
    response = http.post("/api/connect", json={"league_id": "does-not-exist"})
    assert response.status_code == 404


def test_projection_routes_report_loading_rather_than_hanging(client):
    """A UI needs a fast, meaningful answer while the model trains."""
    http, _ = client
    http.post("/api/connect", json={"league_id": "999", "username": "alice"})
    for path in (
        "/api/matchup",
        "/api/draft",
        "/api/board",
        "/api/waivers",
        "/api/trades",
    ):
        response = http.get(path)
        assert response.status_code == 425, path
        body = response.json()
        assert "loading" in body["error"].lower()
        assert "status" in body


def test_selecting_a_team_updates_the_league(client):
    http, service = client
    http.post("/api/connect", json={"league_id": "999"})
    assert service.league.my_roster_id is None
    response = http.post("/api/team", json={"roster_id": 2})
    assert response.status_code == 200
    assert response.json()["league"]["my_roster_id"] == 2


def test_setting_a_team_before_connecting_is_rejected(client):
    http, _ = client
    assert http.post("/api/team", json={"roster_id": 1}).status_code == 409


def test_model_card_reports_untrained_state(client):
    http, _ = client
    payload = http.get("/api/model").json()
    assert payload["trained"] is False


def test_index_page_is_served(client):
    http, _ = client
    response = http.get("/")
    assert response.status_code == 200
    assert "FantasyPicker" in response.text


# ---------------------------------------------------------------------- espn


@pytest.fixture
def espn_client(client, monkeypatch):
    """The API fixture, with ESPN stubbed as well."""
    from fantasypicker.data.crosswalk import Crosswalk
    from fantasypicker.espn.client import EspnClient

    from .test_espn import LEAGUE, stub_transport as espn_stub

    def make(*args, **kwargs):
        kwargs.pop("client", None)
        return EspnClient(
            client=httpx.AsyncClient(transport=espn_stub({"mRoster": LEAGUE})), **kwargs
        )

    crosswalk = Crosswalk(espn_to_sleeper={"3139477": "4034", "4241457": "6794"})
    monkeypatch.setattr("fantasypicker.platforms.EspnClient", make)
    monkeypatch.setattr("fantasypicker.platforms._crosswalk", lambda: crosswalk)
    monkeypatch.setattr("fantasypicker.espn.league.load_crosswalk", lambda *a, **k: crosswalk)
    return client


def test_connecting_to_an_espn_league(espn_client):
    api, service = espn_client
    response = api.post("/api/connect/espn", json={"league_id": "999", "season": 2026})
    assert response.status_code == 200
    league = response.json()["league"]
    assert league["platform"] == "espn"
    assert league["name"] == "Test ESPN League"


def test_a_private_espn_league_answers_401_with_instructions(client, monkeypatch):
    """The UI opens the cookie fields off the back of this, so it must be a 401."""
    from fantasypicker.espn.client import EspnAuthRequired

    async def refuse(self, league_id, season, *, fresh=False):
        raise EspnAuthRequired(league_id, had_cookies=False)

    monkeypatch.setattr("fantasypicker.espn.client.EspnClient.league", refuse)
    api, _ = client
    response = api.post("/api/connect/espn", json={"league_id": "999"})
    assert response.status_code == 401
    body = response.json()
    assert body["needs_cookies"] is True
    assert "espn_s2" in body["error"]


def test_espn_cookies_are_never_echoed_back(espn_client):
    api, _ = espn_client
    response = api.post(
        "/api/connect/espn",
        json={
            "league_id": "999",
            "season": 2026,
            "espn_s2": "super-secret-value",
            "swid": "{ABC}",
        },
    )
    assert response.status_code == 200
    assert "super-secret-value" not in response.text
