"""Free-agent evaluation: gains measured against your own roster, not in a vacuum."""

from __future__ import annotations

from fantasypicker.engine.waivers import find_targets, replacement_baseline
from fantasypicker.model.predict import ProjectionSet
from fantasypicker.sleeper.league import Team

from .conftest import make_league, make_projection_frame

QUANTILES = (0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95)


def build(scoring, my_players, other_players, pool):
    players = my_players + other_players + pool
    projections = ProjectionSet(make_projection_frame(players), QUANTILES, season=2026)
    teams = {
        1: Team(roster_id=1, owner_id="me", display_name="Me", team_name="Mine",
                players=[p["id"] for p in my_players]),
        2: Team(roster_id=2, owner_id="you", display_name="You", team_name="Yours",
                players=[p["id"] for p in other_players]),
    }
    league = make_league(scoring, teams=teams, team_count=2)
    league.my_roster_id = 1
    return league, projections


def starting_roster(prefix: str, level: float) -> list[dict]:
    shape = [("QB", 1), ("RB", 3), ("WR", 3), ("TE", 1), ("K", 1), ("DST", 1)]
    players = []
    for position, count in shape:
        for i in range(count):
            players.append(
                {
                    "id": f"{prefix}-{position}{i}",
                    "position": position,
                    "name": f"{prefix} {position}{i}",
                    "mean": level - i * 12,
                }
            )
    return players


def test_a_clear_upgrade_is_surfaced(scoring):
    mine = starting_roster("me", 150)
    theirs = starting_roster("you", 150)
    pool = [{"id": "fa-star", "position": "WR", "name": "Breakout", "mean": 260}]
    league, projections = build(scoring, mine, theirs, pool)

    report = find_targets(league, [p["id"] for p in mine], projections)
    assert report.targets
    assert report.targets[0].sleeper_id == "fa-star"
    assert report.targets[0].roster_gain > 50


def test_rostered_players_are_never_offered(scoring):
    mine = starting_roster("me", 150)
    theirs = starting_roster("you", 400)  # their roster is much better than mine
    league, projections = build(scoring, mine, theirs, [])

    report = find_targets(league, [p["id"] for p in mine], projections)
    offered = {target.sleeper_id for target in report.targets}
    assert not offered & {p["id"] for p in theirs}


def test_nothing_is_recommended_when_the_wire_is_worse(scoring):
    mine = starting_roster("me", 300)
    theirs = starting_roster("you", 300)
    pool = [{"id": f"scrub{i}", "position": "WR", "name": f"Scrub {i}", "mean": 5} for i in range(5)]
    league, projections = build(scoring, mine, theirs, pool)

    report = find_targets(league, [p["id"] for p in mine], projections)
    assert report.targets == []


def test_gain_accounts_for_the_player_being_dropped(scoring):
    mine = starting_roster("me", 200)
    theirs = starting_roster("you", 200)
    pool = [{"id": "fa", "position": "RB", "name": "Free Back", "mean": 210}]
    league, projections = build(scoring, mine, theirs, pool)

    report = find_targets(league, [p["id"] for p in mine], projections)
    target = report.targets[0]
    assert target.drop_candidate is not None
    assert target.drop_candidate in {p["id"] for p in mine}
    # The drop candidate should be the worst player on the roster.
    worst = min(mine, key=lambda p: p["mean"])["id"]
    assert target.drop_candidate == worst


def test_replacement_baseline_is_the_best_free_agent(scoring):
    mine = starting_roster("me", 200)
    theirs = starting_roster("you", 200)
    pool = [
        {"id": "fa1", "position": "TE", "name": "TE A", "mean": 90},
        {"id": "fa2", "position": "TE", "name": "TE B", "mean": 130},
    ]
    league, projections = build(scoring, mine, theirs, pool)
    assert replacement_baseline(league, projections, "TE") == 130.0


def test_trending_counts_are_reported_but_do_not_drive_the_ranking(scoring):
    mine = starting_roster("me", 150)
    theirs = starting_roster("you", 150)
    pool = [
        # Both beat the worst player on the roster, so both get listed.
        {"id": "hype", "position": "WR", "name": "Hyped", "mean": 200},
        {"id": "quiet", "position": "WR", "name": "Quiet", "mean": 250},
    ]
    league, projections = build(scoring, mine, theirs, pool)

    report = find_targets(
        league, [p["id"] for p in mine], projections, trending={"hype": 90_000}
    )
    assert report.targets[0].sleeper_id == "quiet"
    hyped = next(t for t in report.targets if t.sleeper_id == "hype")
    assert hyped.trending_adds == 90_000
