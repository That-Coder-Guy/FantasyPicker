"""Remembered leagues: what gets kept, what gets merged, what survives a bad file."""

from __future__ import annotations

import json

import httpx
import pytest

from fantasypicker.config import get_settings
from fantasypicker.service import PickerService
from fantasypicker.sleeper.client import SleeperClient
from fantasypicker.store import (
    SCHEMA_VERSION,
    AppState,
    MAX_REMEMBERED,
    RememberedLeague,
    load_state,
    save_state,
)

from .test_sleeper import stub_transport


def league(league_id: str, **kw) -> RememberedLeague:
    return RememberedLeague(league_id=league_id, **kw)


# ------------------------------------------------------------------- in memory


def test_remembering_sets_the_active_league():
    state = AppState()
    state.remember(league("1", name="First"))
    assert state.active_league_id == "1"
    assert state.active.name == "First"


def test_reconnecting_without_a_username_keeps_the_chosen_team():
    """The bug this guards: reconnecting by league ID wiping your team."""
    state = AppState()
    state.remember(league("1", name="League", username="alice", my_roster_id=4))
    state.remember(league("1", name="League"))  # a bare reconnect
    assert state.get("1").my_roster_id == 4
    assert state.get("1").username == "alice"


def test_new_values_do_overwrite_old_ones():
    state = AppState()
    state.remember(league("1", my_roster_id=4))
    state.remember(league("1", my_roster_id=7))
    assert state.get("1").my_roster_id == 7


def test_leagues_are_kept_separate():
    state = AppState()
    state.remember(league("1", name="Home", my_roster_id=2))
    state.remember(league("2", name="Work", my_roster_id=9))
    assert state.get("1").my_roster_id == 2
    assert state.get("2").my_roster_id == 9
    assert state.active_league_id == "2"


def test_recent_is_ordered_by_last_use():
    state = AppState()
    state.remember(league("1"))
    state.remember(league("2"))
    state.remember(league("1"))  # reopened
    assert [lg.league_id for lg in state.recent()] == ["1", "2"]


def test_forgetting_the_active_league_falls_back_to_another():
    state = AppState()
    state.remember(league("1"))
    state.remember(league("2"))
    assert state.forget("2") is True
    assert state.active_league_id == "1"


def test_forgetting_the_last_league_leaves_nothing_active():
    state = AppState()
    state.remember(league("1"))
    state.forget("1")
    assert state.active_league_id is None
    assert state.recent() == []


def test_the_list_is_trimmed_but_never_drops_the_active_one():
    state = AppState()
    for i in range(MAX_REMEMBERED + 5):
        state.remember(league(str(i)))
    assert len(state.leagues) <= MAX_REMEMBERED
    assert state.active_league_id in state.leagues


# ---------------------------------------------------------------- persistence


def test_state_survives_a_round_trip_to_disk():
    state = AppState()
    state.remember(league("1", name="Sunday Sickos", my_roster_id=3, username="alice"))
    save_state(state)

    reloaded = load_state()
    assert reloaded.active_league_id == "1"
    assert reloaded.get("1").name == "Sunday Sickos"
    assert reloaded.get("1").my_roster_id == 3
    assert reloaded.get("1").username == "alice"


def test_missing_file_is_an_empty_state():
    assert load_state().leagues == {}


def test_a_corrupt_file_does_not_break_startup():
    get_settings().state_file.write_text("{not json at all", encoding="utf-8")
    assert load_state().leagues == {}


def test_an_older_schema_is_discarded_rather_than_misread():
    get_settings().state_file.write_text(
        json.dumps({"version": 0, "leagues": {"1": {"league_id": "1"}}}), encoding="utf-8"
    )
    assert load_state().leagues == {}


def test_unknown_fields_in_the_file_are_ignored():
    """A file written by a newer version must not crash an older one."""
    get_settings().state_file.write_text(
        json.dumps(
            {
                "version": SCHEMA_VERSION,
                "active_league_id": "1",
                "leagues": {"1": {"league_id": "1", "name": "X", "future_field": 42}},
            }
        ),
        encoding="utf-8",
    )
    state = load_state()
    assert state.get("1").name == "X"


def test_an_active_id_pointing_nowhere_is_dropped():
    get_settings().state_file.write_text(
        json.dumps({"version": SCHEMA_VERSION, "active_league_id": "gone", "leagues": {}}),
        encoding="utf-8",
    )
    assert load_state().active_league_id is None


# -------------------------------------------------------------------- service


@pytest.fixture
def service(monkeypatch):
    async def fake_enter(self):
        self._client = httpx.AsyncClient(transport=stub_transport())
        self._owned = True
        return self

    monkeypatch.setattr(SleeperClient, "__aenter__", fake_enter)
    monkeypatch.setattr(
        "fantasypicker.service.load_crosswalk", lambda players=None: _StubCrosswalk()
    )
    monkeypatch.setattr("fantasypicker.service.load_expert_ranks", lambda *a, **k: None)
    return PickerService()


class _StubCrosswalk:
    sleeper_to_gsis: dict = {}
    gsis_to_sleeper: dict = {}

    def gsis(self, sleeper_id):
        return None


@pytest.mark.asyncio
async def test_connecting_remembers_the_league(service):
    await service.connect("999", "alice")
    assert load_state().active_league_id == "999"
    remembered = load_state().get("999")
    assert remembered.name == "Sunday Sickos"
    assert remembered.my_roster_id == 1  # resolved from the username
    assert remembered.scoring.startswith("half PPR")


@pytest.mark.asyncio
async def test_choosing_a_team_is_remembered(service):
    await service.connect("999")
    assert service.league.my_roster_id is None
    service.set_my_team(2)
    assert load_state().get("999").my_roster_id == 2


@pytest.mark.asyncio
async def test_a_remembered_team_is_restored_on_reconnect(service):
    """Connect without a username, pick a team, come back — team still yours."""
    await service.connect("999")
    service.set_my_team(2)

    fresh = PickerService()
    await fresh.connect("999")
    assert fresh.league.my_roster_id == 2


@pytest.mark.asyncio
async def test_sleeper_wins_over_a_stale_remembered_team(service):
    """If the username resolves a team, that is authoritative."""
    await service.connect("999")
    service.set_my_team(2)  # remembered as roster 2

    fresh = PickerService()
    await fresh.connect("999", "alice")  # alice actually owns roster 1
    assert fresh.league.my_roster_id == 1


@pytest.mark.asyncio
async def test_reopen_last_restores_the_previous_session(service):
    await service.connect("999", "alice")

    fresh = PickerService()
    summary = await fresh.reopen_last()
    assert summary is not None
    assert fresh.league_id == "999"
    assert fresh.league.my_roster_id == 1


@pytest.mark.asyncio
async def test_reopen_last_with_nothing_remembered_returns_none():
    assert await PickerService().reopen_last() is None


@pytest.mark.asyncio
async def test_reopen_last_survives_a_league_that_no_longer_exists(monkeypatch):
    state = AppState()
    state.remember(league("404", name="Deleted"))
    save_state(state)

    async def fake_enter(self):
        self._client = httpx.AsyncClient(
            transport=stub_transport({"/v1/league/404": None})
        )
        self._owned = True
        return self

    monkeypatch.setattr(SleeperClient, "__aenter__", fake_enter)
    fresh = PickerService()
    assert await fresh.reopen_last() is None
    assert fresh.league is None


@pytest.mark.asyncio
async def test_forgetting_a_league_removes_it_from_the_list(service):
    await service.connect("999", "alice")
    assert any(lg["league_id"] == "999" for lg in service.known_leagues())
    assert service.forget_league("999") is True
    assert service.known_leagues() == []
    assert load_state().leagues == {}


@pytest.mark.asyncio
async def test_known_leagues_flags_the_active_one(service):
    await service.connect("999", "alice")
    rows = service.known_leagues()
    assert rows[0]["league_id"] == "999"
    assert rows[0]["is_active"] is True
    # No model has been trained in this temp home.
    assert rows[0]["model_ready"] is False


@pytest.mark.asyncio
async def test_reconnecting_to_the_same_scoring_keeps_the_loaded_model(service):
    """A re-warm on every reconnect would make the startup reopen pointless."""

    class FakeModel:
        scoring_key = None

    await service.connect("999", "alice")
    from fantasypicker.model.train import scoring_key

    fake = FakeModel()
    fake.scoring_key = scoring_key(service.league.scoring)
    service.model = fake
    service.panel = object()
    service.season_projections = object()

    await service.connect("999", "alice")
    assert service.model is fake
    assert service.status.stage == "ready"


@pytest.mark.asyncio
async def test_different_scoring_discards_the_loaded_model(service):
    class FakeModel:
        scoring_key = "a-different-league-entirely"

    await service.connect("999", "alice")
    service.model = FakeModel()
    service.panel = object()
    service.season_projections = object()

    await service.connect("999", "alice")
    assert service.model is None
    assert service.status.stage == "connected"


@pytest.mark.asyncio
async def test_a_half_loaded_state_still_warms_up(service):
    """Model present but projections missing is not 'ready'."""

    class FakeModel:
        scoring_key = None

    await service.connect("999", "alice")
    from fantasypicker.model.train import scoring_key

    fake = FakeModel()
    fake.scoring_key = scoring_key(service.league.scoring)
    service.model = fake
    service.panel = None  # warm-up never finished
    service.season_projections = None

    await service.connect("999", "alice")
    assert service.model is None
    assert service.status.stage == "connected"
