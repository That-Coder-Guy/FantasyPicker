"""The command line, which is mostly a diagnostic surface.

`doctor` exists because the one failure people actually hit — teams showing as
"Team 4" instead of their names — has several possible causes that look
identical from the outside. It has to survive every one of them without
crashing, since a diagnostic that raises tells you nothing.
"""

from __future__ import annotations

import httpx
import pytest

from fantasypicker import cli
from fantasypicker.sleeper import client as client_module
from fantasypicker.sleeper.client import SleeperClient

from .test_sleeper import LEAGUE, stub_transport


@pytest.fixture
def stub_sleeper(monkeypatch):
    """Point `doctor` at a mock transport instead of api.sleeper.app."""

    def install(overrides):
        transport = stub_transport(overrides)

        def factory(*args, **kwargs):
            return SleeperClient(httpx.AsyncClient(transport=transport))

        monkeypatch.setattr(client_module, "SleeperClient", factory)

    return install


def test_doctor_prints_the_user_to_roster_join(stub_sleeper, capsys):
    stub_sleeper(
        {
            "/v1/league/999/users": [
                {
                    "user_id": "u1",
                    "username": "alice99",
                    "display_name": "alice",
                    "metadata": {"team_name": "Aces"},
                }
            ],
            "/v1/league/999/rosters": [{"roster_id": 1, "owner_id": "u1"}],
        }
    )
    assert cli.main(["doctor", "999"]) == 0
    out = capsys.readouterr().out
    assert LEAGUE["name"] in out
    assert "alice99" in out  # the username, which the label chain now uses
    assert "'Aces'" in out
    assert "owner_id matches a user: 1" in out


def test_doctor_flags_a_roster_whose_owner_is_missing(stub_sleeper, capsys):
    """The exact shape that produces a numeric team name."""
    stub_sleeper(
        {
            "/v1/league/999/users": [{"user_id": "u1", "display_name": "alice"}],
            "/v1/league/999/rosters": [
                {"roster_id": 5, "owner_id": "departed", "players": ["1"]}
            ],
        }
    )
    assert cli.main(["doctor", "999"]) == 0
    captured = capsys.readouterr()
    assert "no manager" in captured.out
    assert "'Team 5'" in captured.out
    assert "abandoned" in captured.err


def test_doctor_survives_users_with_missing_fields(stub_sleeper, capsys):
    """A malformed row must not turn the diagnostic itself into the bug report."""
    stub_sleeper(
        {
            "/v1/league/999/users": [{"display_name": None, "metadata": None}],
            "/v1/league/999/rosters": [{"roster_id": 1, "owner_id": None}],
        }
    )
    assert cli.main(["doctor", "999"]) == 0
    assert "no user_id" in capsys.readouterr().out


def test_doctor_explains_a_league_that_is_still_filling_up(capsys, stub_sleeper):
    """The case that sent a real user hunting for a bug that was not there."""
    stub_sleeper(
        {
            "/v1/league/999/users": [{"user_id": "u1", "display_name": "alice"}],
            "/v1/league/999/rosters": [
                {"roster_id": 1, "owner_id": "u1"},
                {"roster_id": 2, "owner_id": None},
                {"roster_id": 3, "owner_id": None},
            ],
        }
    )
    assert cli.main(["doctor", "999"]) == 0
    out = capsys.readouterr().out
    assert "nobody has joined this seat" in out
    assert "2 of 3 seats have no manager yet" in out
    assert "nothing to type in" in out


def test_doctor_on_an_unknown_league_explains_rather_than_traces(stub_sleeper, capsys):
    stub_sleeper({"/v1/league/999": None})
    assert cli.main(["doctor", "999"]) == 1
    assert "no league with ID 999" in capsys.readouterr().err
