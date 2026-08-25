"""Getting connected, and staying current once you are.

These cover the two things that actually go wrong in practice: a league ID that
Sleeper does not recognise, and state that silently stops matching the league.
"""

from __future__ import annotations

import httpx
import pytest

from fantasypicker.model.availability import AvailabilityModel
from fantasypicker.model.predict import ProjectionSet, apply_availability
from fantasypicker.service import _clean_league_id
from fantasypicker.sleeper.client import SleeperClient
from fantasypicker.sleeper.league import (
    LeagueNotFound,
    build_teams,
    load_league,
    refresh_teams,
)

from .conftest import make_projection_frame
from .test_sleeper import ROSTERS, USERS, stub_transport

QUANTILES = (0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95)


# ------------------------------------------------------------------- league id


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1048273661924872192", "1048273661924872192"),
        ("  1048273661924872192 ", "1048273661924872192"),
        ("https://sleeper.com/leagues/1048273661924872192/team", "1048273661924872192"),
        ("https://sleeper.app/leagues/1048273661924872192", "1048273661924872192"),
        ("sleeper.com/leagues/1048273661924872192/matchup", "1048273661924872192"),
        ("https://sleeper.com/draft/nfl/1048273661924872192", "1048273661924872192"),
    ],
)
def test_pasted_urls_are_accepted_as_league_ids(raw, expected):
    """People paste the address bar far more often than the number inside it."""
    assert _clean_league_id(raw) == expected


def test_non_numeric_input_is_passed_through_for_the_error_message():
    assert _clean_league_id("my-league") == "my-league"


@pytest.mark.asyncio
async def test_unknown_league_raises_an_actionable_error():
    client = SleeperClient(
        httpx.AsyncClient(transport=stub_transport({"/v1/league/999": None}))
    )
    with pytest.raises(LeagueNotFound) as caught:
        await load_league(client, "999")
    message = str(caught.value)
    assert "999" in message
    assert "username" in message.lower()  # points at the way out
    assert "draft" in message.lower()  # names the common mix-up


# ---------------------------------------------------------------- refreshing


@pytest.mark.asyncio
async def test_refresh_picks_up_a_roster_change():
    """A waiver claim between page loads has to reach the app."""
    rosters = [dict(row) for row in ROSTERS]
    routes = {"/v1/league/999/rosters": rosters}
    client = SleeperClient(httpx.AsyncClient(transport=stub_transport(routes)))
    league = await load_league(client, "999", username="alice")
    assert "999999" not in league.teams[1].players

    rosters[0] = {**rosters[0], "players": [*rosters[0]["players"], "999999"]}
    routes["/v1/league/999/rosters"] = rosters
    client2 = SleeperClient(httpx.AsyncClient(transport=stub_transport(routes)))
    # fresh=True is what an explicit refresh does: without it this call is
    # legitimately served from the 60-second disk cache and sees nothing.
    changed = await refresh_teams(client2, league, fresh=True)

    assert changed is True
    assert "999999" in league.teams[1].players


@pytest.mark.asyncio
async def test_a_throttled_refresh_is_allowed_to_use_the_cache():
    """Background polling should not re-download rosters every few seconds."""
    rosters = [dict(row) for row in ROSTERS]
    routes = {"/v1/league/999/rosters": rosters}
    client = SleeperClient(httpx.AsyncClient(transport=stub_transport(routes)))
    league = await load_league(client, "999", username="alice")

    rosters[0] = {**rosters[0], "players": [*rosters[0]["players"], "999999"]}
    routes["/v1/league/999/rosters"] = rosters
    client2 = SleeperClient(httpx.AsyncClient(transport=stub_transport(routes)))
    assert await refresh_teams(client2, league, fresh=False) is False


@pytest.mark.asyncio
async def test_refresh_reports_no_change_when_nothing_moved():
    client = SleeperClient(httpx.AsyncClient(transport=stub_transport()))
    league = await load_league(client, "999", username="alice")
    assert await refresh_teams(client, league, fresh=True) is False


@pytest.mark.asyncio
async def test_refresh_invalidates_the_matchup_cache():
    client = SleeperClient(httpx.AsyncClient(transport=stub_transport()))
    league = await load_league(client, "999", username="alice")
    await league.load_matchups(client, 4)
    assert 4 in league._matchup_cache
    await refresh_teams(client, league, fresh=True)
    assert 4 not in league._matchup_cache


def test_build_teams_survives_a_roster_with_no_owner():
    """Orphan teams are real in Sleeper — a manager leaves mid-season."""
    teams = build_teams([{"roster_id": 3, "owner_id": None, "players": ["1"]}], USERS)
    assert teams[3].label == "Team 3"
    assert teams[3].players == ["1"]


# --------------------------------------------------------------- availability


def _projections(status_players):
    frame = make_projection_frame(
        [{"id": pid, "position": "RB", "mean": 12.0} for pid in status_players]
    )
    return ProjectionSet(frame, QUANTILES, season=2026, week=4)


def test_availability_can_be_reapplied_without_reprojecting():
    """An injury downgrade must not require rebuilding the model."""
    projections = _projections(["a", "b"])
    availability = AvailabilityModel()
    healthy = {"a": {"injury_status": None}, "b": {"injury_status": None}}
    apply_availability(projections, availability, healthy)
    before = projections.frame.set_index("sleeper_id")["exp_points"]["a"]

    ruled_out = {"a": {"injury_status": "Out"}, "b": {"injury_status": None}}
    changed = apply_availability(projections, availability, ruled_out)

    after = projections.frame.set_index("sleeper_id")
    assert changed is True
    assert after["p_play"]["a"] == 0.0
    assert after["exp_points"]["a"] == 0.0
    assert after["exp_points"]["a"] < before
    # The projection itself — how he scores if he plays — is untouched.
    assert after["proj_mean"]["a"] == pytest.approx(12.0)
    assert after["exp_points"]["b"] > 0


def test_reapplying_the_same_status_reports_no_change():
    projections = _projections(["a"])
    availability = AvailabilityModel()
    players = {"a": {"injury_status": "Questionable"}}
    apply_availability(projections, availability, players)
    assert apply_availability(projections, availability, players) is False


def test_availability_on_an_empty_projection_set_is_a_no_op():
    import pandas as pd

    empty = ProjectionSet(pd.DataFrame(), QUANTILES, season=2026, week=1)
    assert apply_availability(empty, AvailabilityModel(), {}) is False
