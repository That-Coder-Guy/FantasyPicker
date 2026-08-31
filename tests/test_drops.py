"""The drop finder.

The premise: the right player to drop depends on who you are adding. Every
scenario here is built so a human reading the rosters knows the answer — a
buried fourth tight end, a startable free agent at a position of need — and the
assertions check the engine reaches it, refuses the swaps a human would refuse,
and never suggests cutting someone who is actually playing.
"""

from __future__ import annotations

import pytest

from fantasypicker.engine.drops import DEAD_WEIGHT, MIN_UPGRADE, find_drops
from fantasypicker.model.predict import ProjectionSet
from fantasypicker.sleeper.league import Team

from .conftest import make_league, make_projection_frame

QUANTILES = (0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95)


def projections(players: list[dict]) -> ProjectionSet:
    frame = make_projection_frame([{**p, "spread": 10.0} for p in players])
    return ProjectionSet(frame, QUANTILES, season=2026, week=None)


def team(roster_id: int, name: str, players: list[str]) -> Team:
    return Team(
        roster_id=roster_id,
        owner_id=f"u{roster_id}",
        display_name=name,
        team_name=name,
        players=players,
    )


def _staples(prefix: str) -> list[dict]:
    return [
        {"id": f"{prefix}_qb", "position": "QB", "mean": 300.0},
        {"id": f"{prefix}_k", "position": "K", "mean": 110.0},
        {"id": f"{prefix}_dst", "position": "DST", "mean": 105.0},
    ]


def _ids(rows: list[dict]) -> list[str]:
    return [str(r["id"]) for r in rows]


def build(mine: list[dict], pool: list[dict], *, others: list[dict] | None = None):
    """A one-team league plus an open pool, so 'unowned' is unambiguous."""
    others = others or []
    teams = {1: team(1, "Me", _ids(mine))}
    if others:
        teams[2] = team(2, "Them", _ids(others))
    league = make_league(scoring_fixture, teams=teams, team_count=len(teams))
    return league, projections(mine + pool + others)


scoring_fixture = None  # replaced per-test by the ``scoring`` fixture


@pytest.fixture(autouse=True)
def _bind_scoring(scoring):
    global scoring_fixture
    scoring_fixture = scoring
    yield


# ------------------------------------------------------------------ upgrades


def test_a_startable_free_agent_replaces_the_player_he_beats(scoring):
    """The obvious case: my WR2 is terrible and a good one is unowned."""
    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_wr1", "position": "WR", "mean": 180.0},
        {"id": "m_wr_bad", "position": "WR", "mean": 20.0},
        {"id": "m_te", "position": "TE", "mean": 120.0},
    ]
    pool = [{"id": "fa_wr", "position": "WR", "mean": 160.0}]
    league, projs = build(mine, pool)

    report = find_drops(league, projs, my_roster_id=1)
    assert report.upgrades, "the obvious WR upgrade was not found"
    top = report.upgrades[0]
    assert top.drop == "m_wr_bad"
    assert top.add == "fa_wr"
    assert top.gain > MIN_UPGRADE
    assert "rest-of-season" in top.reason


def test_the_drop_matches_the_position_being_added(scoring):
    """Adding a tight end should cost a tight end, not the worst player overall.

    This is the bug in pairing every add with your single worst player: here
    the worst player by projection is a running back who is still starting.
    """
    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 150.0},
        {"id": "m_rb2", "position": "RB", "mean": 140.0},
        {"id": "m_wr1", "position": "WR", "mean": 160.0},
        {"id": "m_wr2", "position": "WR", "mean": 155.0},
        {"id": "m_te1", "position": "TE", "mean": 130.0},
        {"id": "m_te_spare", "position": "TE", "mean": 30.0},
    ]
    pool = [{"id": "fa_te", "position": "TE", "mean": 145.0}]
    league, projs = build(mine, pool)

    report = find_drops(league, projs, my_roster_id=1)
    assert report.upgrades
    top = report.upgrades[0]
    assert top.add == "fa_te"
    # The spare tight end goes, not a starting running back.
    assert top.drop == "m_te_spare"


def test_a_starter_is_never_dropped_for_a_worse_free_agent(scoring):
    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_wr1", "position": "WR", "mean": 180.0},
        {"id": "m_wr2", "position": "WR", "mean": 170.0},
        {"id": "m_te", "position": "TE", "mean": 120.0},
    ]
    pool = [{"id": f"fa{i}", "position": "WR", "mean": 20.0} for i in range(5)]
    league, projs = build(mine, pool)

    report = find_drops(league, projs, my_roster_id=1)
    assert report.upgrades == []
    assert any("improves your lineup" in n for n in report.notes)


def test_a_marginal_gain_is_not_worth_a_transaction(scoring):
    """Inside the model's error bars, a swap is noise dressed as advice."""
    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_wr1", "position": "WR", "mean": 180.0},
        {"id": "m_wr2", "position": "WR", "mean": 100.0},
        {"id": "m_te", "position": "TE", "mean": 120.0},
    ]
    # Beats the WR2 by a fraction of a point.
    pool = [{"id": "fa_wr", "position": "WR", "mean": 100.4}]
    league, projs = build(mine, pool)

    report = find_drops(league, projs, my_roster_id=1)
    assert report.upgrades == []


def test_one_free_agent_is_not_recommended_for_several_drops(scoring):
    """The same add answering three drops is one idea printed three ways."""
    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_wr1", "position": "WR", "mean": 180.0},
        {"id": "m_wr_bad1", "position": "WR", "mean": 10.0},
        {"id": "m_wr_bad2", "position": "WR", "mean": 11.0},
        {"id": "m_wr_bad3", "position": "WR", "mean": 12.0},
        {"id": "m_te", "position": "TE", "mean": 120.0},
    ]
    pool = [{"id": "fa_wr", "position": "WR", "mean": 170.0}]
    league, projs = build(mine, pool)

    report = find_drops(league, projs, my_roster_id=1)
    adds = [u.add for u in report.upgrades]
    assert len(adds) == len(set(adds))


# --------------------------------------------------------------- dead weight


def test_a_buried_player_is_flagged_as_free_to_cut(scoring):
    """Nobody in the pool beats him, but he never plays and is not even depth
    worth keeping — exactly what you need to know when a bye forces a spot.

    Note the roster is deep enough that the FLEX is filled by someone else and
    three better players sit ahead of him on the bench; a genuine handcuff to a
    starter is a different thing and is deliberately not flagged.
    """
    mine = _staples("m") + [
        {"id": "m_te1", "position": "TE", "mean": 140.0},
        {"id": "m_te2", "position": "TE", "mean": 8.0},
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_rb3", "position": "RB", "mean": 160.0},
        {"id": "m_rb4", "position": "RB", "mean": 150.0},
        {"id": "m_wr1", "position": "WR", "mean": 180.0},
        {"id": "m_wr2", "position": "WR", "mean": 170.0},
        {"id": "m_wr3", "position": "WR", "mean": 145.0},
    ]
    pool = [{"id": "fa_te", "position": "TE", "mean": 5.0}]
    league, projs = build(mine, pool)

    report = find_drops(league, projs, my_roster_id=1)
    dead = {d.drop for d in report.dead_weight}
    assert "m_te2" in dead
    entry = next(d for d in report.dead_weight if d.drop == "m_te2")
    assert entry.cost < DEAD_WEIGHT
    assert "m_te1" in entry.blocked_by
    assert "costs nothing" in entry.reason or "never reaches" in entry.reason


def test_a_real_handcuff_is_not_offered_up_as_free(scoring):
    """A backup worth real points behind a starter costs something to cut, and
    saying otherwise would talk someone out of their insurance."""
    mine = _staples("m") + [
        {"id": "m_te1", "position": "TE", "mean": 140.0},
        {"id": "m_te2", "position": "TE", "mean": 90.0},
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_rb3", "position": "RB", "mean": 160.0},
        {"id": "m_wr1", "position": "WR", "mean": 180.0},
        {"id": "m_wr2", "position": "WR", "mean": 170.0},
    ]
    league, projs = build(mine, [{"id": "fa_te", "position": "TE", "mean": 5.0}])

    report = find_drops(league, projs, my_roster_id=1)
    assert all(d.drop != "m_te2" for d in report.dead_weight)


def test_a_starter_is_never_called_dead_weight(scoring):
    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_wr1", "position": "WR", "mean": 180.0},
        {"id": "m_wr2", "position": "WR", "mean": 170.0},
        {"id": "m_te", "position": "TE", "mean": 120.0},
    ]
    league, projs = build(mine, [{"id": "fa", "position": "WR", "mean": 5.0}])

    report = find_drops(league, projs, my_roster_id=1)
    dead = {d.drop for d in report.dead_weight}
    for starter in ("m_rb1", "m_wr1", "m_te", "m_qb"):
        assert starter not in dead


# ---------------------------------------------------------------- edge cases


def test_players_owned_by_other_teams_are_not_offered(scoring):
    """"Open pool" means unowned — suggesting someone else's stud is useless."""
    mine = _staples("m") + [
        {"id": "m_wr1", "position": "WR", "mean": 180.0},
        {"id": "m_wr_bad", "position": "WR", "mean": 10.0},
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_te", "position": "TE", "mean": 120.0},
    ]
    others = _staples("t") + [{"id": "t_wr_stud", "position": "WR", "mean": 250.0}]
    league, projs = build(mine, [], others=others)

    report = find_drops(league, projs, my_roster_id=1)
    for entry in report.upgrades:
        assert entry.add != "t_wr_stud"


def test_an_empty_roster_says_so_plainly(scoring):
    league = make_league(scoring, teams={1: team(1, "Me", [])}, team_count=1)
    report = find_drops(
        league,
        projections([{"id": "x", "position": "RB", "mean": 100.0}]),
        my_roster_id=1,
    )
    assert report.upgrades == []
    assert any("empty" in n for n in report.notes)


def test_an_unknown_team_asks_for_one(scoring):
    league, projs = build(_staples("m"), [])
    report = find_drops(league, projs, my_roster_id=99)
    assert any("Pick your team" in n for n in report.notes)


def test_injured_reserve_players_are_left_out(scoring):
    """An IR player holds no active roster spot, so he is not a drop candidate
    and is not blocking anyone."""
    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_wr1", "position": "WR", "mean": 180.0},
        {"id": "m_wr_hurt", "position": "WR", "mean": 5.0},
        {"id": "m_te", "position": "TE", "mean": 120.0},
    ]
    league, projs = build(mine, [{"id": "fa_wr", "position": "WR", "mean": 150.0}])
    league.teams[1].reserve = ["m_wr_hurt"]

    report = find_drops(league, projs, my_roster_id=1)
    assert all(u.drop != "m_wr_hurt" for u in report.upgrades)
    assert all(d.drop != "m_wr_hurt" for d in report.dead_weight)


def test_unprojected_players_are_noted_not_silently_skipped(scoring):
    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_wr1", "position": "WR", "mean": 180.0},
    ]
    league, projs = build(mine, [{"id": "fa", "position": "WR", "mean": 20.0}])
    league.teams[1].players.append("mystery_rookie")

    report = find_drops(league, projs, my_roster_id=1)
    assert any("no projection" in n for n in report.notes)


# ------------------------------------------------------------------- service


@pytest.mark.asyncio
async def test_the_service_decorates_ids_with_names(scoring, monkeypatch):
    """The page must never show a raw sleeper_id."""
    from fantasypicker.service import PickerService

    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_wr1", "position": "WR", "mean": 180.0},
        {"id": "m_wr_bad", "position": "WR", "mean": 20.0},
        {"id": "m_te", "position": "TE", "mean": 120.0},
    ]
    league, projs = build(mine, [{"id": "fa_wr", "position": "WR", "mean": 160.0}])
    league.my_roster_id = 1

    service = PickerService()
    service.league = league
    service.season_projections = projs
    service.model = object()  # readiness gate looks only at presence
    service.panel = object()

    async def no_refresh(self, *, force=False):
        return {"rosters": False, "players": False}

    monkeypatch.setattr(PickerService, "refresh_live", no_refresh)
    payload = await service.drops()

    assert payload["my_roster_id"] == 1
    assert payload["upgrades"]
    for row in payload["upgrades"] + payload["dead_weight"]:
        for pid in [row["drop"], row["add"], *row["blocked_by"]]:
            if pid is None:
                continue
            assert pid in payload["players"], pid
            assert payload["players"][pid]["name"]
            assert payload["players"][pid]["position"]


@pytest.mark.asyncio
async def test_the_service_requires_a_team(scoring, monkeypatch):
    from fantasypicker.service import PickerService

    league, projs = build(_staples("m"), [])
    league.my_roster_id = None

    service = PickerService()
    service.league = league
    service.season_projections = projs
    service.model = object()
    service.panel = object()

    async def no_refresh(self, *, force=False):
        return {"rosters": False, "players": False}

    monkeypatch.setattr(PickerService, "refresh_live", no_refresh)
    with pytest.raises(ValueError, match="Pick your team"):
        await service.drops()
