"""The league-wide view: every team, its lineup, and how they compare."""

from __future__ import annotations

import pytest

from fantasypicker.engine.league_view import build_league_view, league_averages
from fantasypicker.model.predict import ProjectionSet
from fantasypicker.sleeper.league import Team
from fantasypicker.sleeper.scoring import ScoringRules

from .conftest import HALF_PPR, make_league, make_projection_frame

QUANTILES = (0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95)


def roster(prefix: str, level: float, *, extra: list[dict] | None = None) -> list[dict]:
    """A full legal roster: QB, 3 RB, 3 WR, TE, K, DST, declining in value."""
    shape = [("QB", 1), ("RB", 3), ("WR", 3), ("TE", 1), ("K", 1), ("DST", 1)]
    players = []
    for position, count in shape:
        for i in range(count):
            players.append(
                {
                    "id": f"{prefix}-{position}{i}",
                    "position": position,
                    "name": f"{prefix} {position}{i}",
                    "mean": level - i * 3,
                }
            )
    return players + (extra or [])


def build(rosters: dict[int, list[dict]], *, my_roster_id=1, matchups=None, starters=None):
    players = [p for group in rosters.values() for p in group]
    projections = ProjectionSet(make_projection_frame(players), QUANTILES, season=2026, week=4)
    teams = {
        rid: Team(
            roster_id=rid,
            owner_id=f"u{rid}",
            display_name=f"owner{rid}",
            team_name=f"Team {rid}",
            players=[p["id"] for p in group],
            starters=(starters or {}).get(rid, []),
        )
        for rid, group in rosters.items()
    }
    # Projections are supplied directly, so only the roster slots matter here.
    league = make_league(
        ScoringRules.from_league({"scoring_settings": HALF_PPR}),
        teams=teams,
        team_count=len(rosters),
    )
    league.my_roster_id = my_roster_id
    return league, projections, matchups


def test_every_team_gets_a_view():
    league, projections, _ = build({1: roster("a", 15), 2: roster("b", 12), 3: roster("c", 10)})
    views = build_league_view(league, projections)
    assert len(views) == 3
    assert {v.roster_id for v in views} == {1, 2, 3}


def test_teams_are_ranked_by_best_possible_lineup():
    league, projections, _ = build({1: roster("a", 10), 2: roster("b", 18), 3: roster("c", 14)})
    views = build_league_view(league, projections)
    assert [v.roster_id for v in views] == [2, 3, 1]
    assert views[0].projected_points > views[1].projected_points > views[2].projected_points


def test_my_team_is_flagged():
    league, projections, _ = build({1: roster("a", 10), 2: roster("b", 18)}, my_roster_id=2)
    views = build_league_view(league, projections)
    assert [v.is_me for v in views] == [True, False]


def test_lineups_fill_every_slot_and_respect_eligibility():
    league, projections, _ = build({1: roster("a", 15)})
    view = build_league_view(league, projections)[0]
    slots = [p["slot"] for p in view.starters]
    assert len(view.starters) == len(league.slots)
    assert slots.count("RB") == 2 and slots.count("WR") == 2
    flex = next(p for p in view.starters if p["slot"] == "FLEX")
    assert flex["position"] in {"RB", "WR", "TE"}


def test_bench_holds_everyone_not_starting():
    """A roster with a spare receiver should show him on the bench."""
    extra = [{"id": "a-WR9", "position": "WR", "name": "Spare WR", "mean": 4.0}]
    league, projections, _ = build({1: roster("a", 15, extra=extra)})
    view = build_league_view(league, projections)[0]
    started = {p["sleeper_id"] for p in view.starters}
    benched = {p["sleeper_id"] for p in view.bench}
    assert "a-WR9" in benched
    assert not (started & benched)
    assert len(started) + len(benched) == 11


def test_bench_is_ordered_best_first():
    extra = [
        {"id": "a-WR8", "position": "WR", "name": "Good spare", "mean": 9.0},
        {"id": "a-WR9", "position": "WR", "name": "Bad spare", "mean": 2.0},
    ]
    league, projections, _ = build({1: roster("a", 15, extra=extra)})
    view = build_league_view(league, projections)[0]
    projections_out = [p["projection"] for p in view.bench]
    assert projections_out == sorted(projections_out, reverse=True)


def test_a_suboptimal_declared_lineup_is_quantified():
    """The interesting case: a manager starting their worst players."""
    players = roster("a", 20)
    # Deliberately start the weakest legal lineup.
    bad = ["a-QB0", "a-RB2", "a-RB1", "a-WR2", "a-WR1", "a-TE0", "a-RB0", "a-K0", "a-DST0"]
    league, projections, _ = build({1: players, 2: roster("b", 10)}, starters={1: bad})
    views = build_league_view(league, projections)
    mine = next(v for v in views if v.roster_id == 1)
    assert mine.declared_points < mine.projected_points
    assert mine.points_left_on_bench > 0
    assert any("bench" in note for note in mine.notes)


def test_a_team_with_no_lineup_set_is_reported_not_scored_as_zero():
    league, projections, _ = build({1: roster("a", 15)}, starters={1: []})
    view = build_league_view(league, projections)[0]
    assert view.declared_points == view.projected_points
    assert any("No lineup set" in note for note in view.notes)
    assert view.points_left_on_bench == 0


def test_a_roster_too_thin_to_fill_the_lineup_says_so():
    thin = [
        {"id": "t-QB0", "position": "QB", "name": "Only QB", "mean": 18.0},
        {"id": "t-RB0", "position": "RB", "name": "Only RB", "mean": 12.0},
    ]
    league, projections, _ = build({1: thin})
    view = build_league_view(league, projections)[0]
    assert len(view.starters) == 2
    assert any("cannot be filled" in note for note in view.notes)


def test_matchup_pairings_are_resolved_both_ways():
    league, projections, _ = build({1: roster("a", 15), 2: roster("b", 12)})
    rows = [
        {"roster_id": 1, "matchup_id": 7, "starters": []},
        {"roster_id": 2, "matchup_id": 7, "starters": []},
    ]
    views = {v.roster_id: v for v in build_league_view(league, projections, matchup_rows=rows)}
    assert views[1].opponent_roster_id == 2
    assert views[2].opponent_roster_id == 1
    assert views[1].opponent_label == "Team 2"
    assert views[2].opponent_label == "Team 1"


def test_declared_starters_come_from_the_matchup_feed_when_present():
    """Sleeper's weekly feed is fresher than the roster object's starters."""
    players = roster("a", 20)
    stale = ["a-QB0", "a-RB0", "a-RB1", "a-WR0", "a-WR1", "a-TE0", "a-RB2", "a-K0", "a-DST0"]
    fresh = ["a-QB0", "a-RB2", "a-RB1", "a-WR2", "a-WR1", "a-TE0", "a-RB0", "a-K0", "a-DST0"]
    league, projections, _ = build({1: players}, starters={1: stale})
    rows = [{"roster_id": 1, "matchup_id": 1, "starters": fresh}]
    view = build_league_view(league, projections, matchup_rows=rows)[0]
    assert {p["sleeper_id"] for p in view.declared} == set(fresh)


def test_a_team_on_bye_at_a_position_is_counted():
    """A player with no opponent this week has no game."""
    extra = [{"id": "a-WR7", "position": "WR", "name": "Bye WR", "mean": 9.0, "opponent": ""}]
    league, projections, _ = build({1: roster("a", 15, extra=extra)})
    view = build_league_view(league, projections)[0]
    assert view.byes.get("WR") == 1


def test_position_strength_counts_only_startable_depth():
    """Six mediocre receivers must not out-rank two good ones."""
    hoarder = roster("h", 6) + [
        {"id": f"h-WRx{i}", "position": "WR", "name": f"Spare {i}", "mean": 6.0}
        for i in range(5)
    ]
    sharp = roster("s", 6)
    sharp[4]["mean"] = 22.0  # WR0
    sharp[5]["mean"] = 20.0  # WR1
    league, projections, _ = build({1: hoarder, 2: sharp})
    views = {v.roster_id: v for v in build_league_view(league, projections)}
    assert views[2].position_strength["WR"] > views[1].position_strength["WR"]


def test_league_averages_sit_between_the_extremes():
    league, projections, _ = build({1: roster("a", 20), 2: roster("b", 10)})
    views = build_league_view(league, projections)
    averages = league_averages(views)
    assert views[1].projected_points < averages["projected_points"] < views[0].projected_points
    assert "strength_RB" in averages


def test_league_averages_of_nothing_is_empty():
    assert league_averages([]) == {}


def test_players_with_no_projection_are_skipped_not_faked():
    league, projections, _ = build({1: roster("a", 15)})
    league.teams[1].players.append("ghost-player")
    view = build_league_view(league, projections)[0]
    ids = {p["sleeper_id"] for p in view.starters} | {p["sleeper_id"] for p in view.bench}
    assert "ghost-player" not in ids


def test_reserve_players_are_not_startable():
    league, projections, _ = build({1: roster("a", 15)})
    league.teams[1].reserve = ["a-RB0"]
    view = build_league_view(league, projections)[0]
    assert "a-RB0" not in {p["sleeper_id"] for p in view.starters}


def test_as_dict_is_json_safe():
    league, projections, _ = build({1: roster("a", 15), 2: roster("b", 12)})
    import json

    payload = [v.as_dict() for v in build_league_view(league, projections)]
    json.dumps(payload)  # raises if a numpy scalar leaked through
    assert payload[0]["projected_points"] == pytest.approx(
        payload[0]["projected_points"], abs=0.05
    )


def test_unprojectable_starters_are_not_scored_as_zero():
    """A just-signed call-up we have no projection for must not read as a
    manager benching their entire team."""
    league, projections, _ = build({1: roster("a", 15)})
    rows = [{"roster_id": 1, "matchup_id": 1, "starters": ["nobody-1", "nobody-2"]}]
    view = build_league_view(league, projections, matchup_rows=rows)[0]
    assert view.declared == []
    assert view.declared_points == view.projected_points
    assert view.points_left_on_bench == 0
    assert any("could not be read" in note for note in view.notes)
    assert not any("on the bench" in note for note in view.notes)


def test_a_partially_unprojectable_lineup_is_flagged_as_a_floor():
    players = roster("a", 20)
    mixed = ["a-QB0", "a-RB0", "a-RB1", "a-WR0", "a-WR1", "a-TE0", "a-K0", "ghost"]
    league, projections, _ = build({1: players})
    rows = [{"roster_id": 1, "matchup_id": 1, "starters": mixed}]
    view = build_league_view(league, projections, matchup_rows=rows)[0]
    assert view.declared_points > 0
    assert any("no projection" in note for note in view.notes)
    # The gap is partly our ignorance, so no accusation of benching points.
    assert not any("on the bench" in note for note in view.notes)
