"""League parsing and the offline path through the Sleeper client.

The client is exercised against a stub transport rather than the live API, so
these tests assert the shapes we depend on — the ones documented at
docs.sleeper.com — without needing the network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from fantasypicker.sleeper.client import SleeperClient
from fantasypicker.sleeper.league import (
    SLOT_ELIGIBILITY,
    RosterSlot,
    load_league,
)

from .conftest import HALF_PPR, STANDARD_SLOTS, SUPERFLEX_SLOTS

LEAGUE = {
    "league_id": "999",
    "name": "Sunday Sickos",
    "season": "2026",
    "total_rosters": 12,
    "roster_positions": STANDARD_SLOTS,
    "scoring_settings": HALF_PPR,
    "settings": {"type": 0},
}

USERS = [
    {"user_id": "u1", "display_name": "alice", "metadata": {"team_name": "Alice's Aces"}},
    {"user_id": "u2", "display_name": "bob", "metadata": {}},
]

ROSTERS = [
    {
        "roster_id": 1,
        "owner_id": "u1",
        "players": ["100", "101", "102", "103"],
        "starters": ["100", "101"],
        "reserve": ["103"],
        "settings": {"wins": 3, "losses": 1, "fpts": 412, "fpts_decimal": 55},
    },
    {
        "roster_id": 2,
        "owner_id": "u2",
        "players": ["200", "201"],
        "starters": ["200"],
        "settings": {"wins": 2, "losses": 2, "fpts": 380, "fpts_decimal": 10},
    },
]

MATCHUPS = [
    {"roster_id": 1, "matchup_id": 1, "starters": ["100", "101"], "points": 88.5},
    {"roster_id": 2, "matchup_id": 1, "starters": ["200"], "points": 79.2},
]


def stub_transport(overrides: dict[str, object] | None = None) -> httpx.MockTransport:
    routes: dict[str, object] = {
        "/v1/state/nfl": {"season": "2026", "week": 4, "season_type": "regular"},
        "/v1/league/999": LEAGUE,
        "/v1/league/999/users": USERS,
        "/v1/league/999/rosters": ROSTERS,
        "/v1/league/999/matchups/4": MATCHUPS,
        "/v1/user/alice": {"user_id": "u1", "display_name": "alice"},
        "/v1/players/nfl": {"100": {"full_name": "A Back", "position": "RB", "team": "SF"}},
    }
    routes.update(overrides or {})

    def handler(request: httpx.Request) -> httpx.Response:
        payload = routes.get(request.url.path)
        if payload is None:
            return httpx.Response(404, json=None)
        return httpx.Response(200, content=json.dumps(payload))

    return httpx.MockTransport(handler)


@pytest.fixture
def client():
    return SleeperClient(httpx.AsyncClient(transport=stub_transport()))


@pytest.mark.asyncio
async def test_load_league_builds_teams_and_scoring(client):
    league = await load_league(client, "999", username="alice")
    assert league.name == "Sunday Sickos"
    assert league.team_count == 12
    assert league.my_roster_id == 1
    assert league.my_team.label == "Alice's Aces"
    assert league.teams[2].label == "bob"
    assert league.teams[1].record == "3-1"
    assert league.teams[1].points_for == pytest.approx(412.55)
    assert league.scoring.ppr == 0.5


@pytest.mark.asyncio
async def test_reserve_players_are_not_startable(client):
    league = await load_league(client, "999", username="alice")
    assert "103" in league.teams[1].players
    assert "103" not in league.teams[1].active_players


@pytest.mark.asyncio
async def test_matchup_finds_the_opponent_without_manual_entry(client):
    """The whole point: the opponent's roster arrives from the league ID alone."""
    league = await load_league(client, "999", username="alice")
    matchup = await league.matchup_for(client, 4, roster_id=1)
    assert matchup is not None
    assert matchup.away is not None
    assert matchup.away.roster_id == 2
    assert matchup.away_starters == ["200"]
    assert matchup.away.players == ["200", "201"]


@pytest.mark.asyncio
async def test_missing_league_raises_a_clear_error():
    client = SleeperClient(httpx.AsyncClient(transport=stub_transport({"/v1/league/999": None})))
    with pytest.raises(ValueError, match="no league"):
        await load_league(client, "999")


def test_slot_eligibility_covers_flex_variants():
    assert RosterSlot(0, "FLEX").accepts("RB")
    assert not RosterSlot(0, "FLEX").accepts("QB")
    assert RosterSlot(0, "SUPER_FLEX").accepts("QB")
    assert RosterSlot(0, "REC_FLEX").accepts("TE")
    assert not RosterSlot(0, "REC_FLEX").accepts("RB")
    assert RosterSlot(0, "DEF").accepts("DST")
    assert "DST" in SLOT_ELIGIBILITY["DEF"]


def test_starters_needed_splits_flex_fractionally(scoring):
    from .conftest import make_league

    league = make_league(scoring)
    # Two dedicated RB slots plus a third of the flex.
    assert league.starters_needed("RB") == pytest.approx(2 + 1 / 3)
    assert league.starters_needed("QB") == 1.0

    superflex = make_league(scoring, slots=SUPERFLEX_SLOTS)
    assert superflex.starters_needed("QB") == pytest.approx(1.25)


def test_bench_slots_are_excluded_from_the_lineup(scoring):
    from .conftest import make_league

    league = make_league(scoring)
    assert len(league.slots) == 9
    assert league.bench_size == 6
    assert league.roster_size == 15
