"""Draft mechanics: pick numbering, scarcity, replacement level, and value."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasypicker.engine.draft import (
    DraftState,
    RosterValuer,
    assign_tiers,
    build_board,
    expected_best_available,
    overall_pick_number,
    parse_draft_state,
    recommend,
    replacement_levels,
    survival_probability,
)
from fantasypicker.model.predict import ProjectionSet

from .conftest import SUPERFLEX_SLOTS, make_league, make_projection_frame

QUANTILES = (0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95)


# --------------------------------------------------------------------- picks


def test_snake_reverses_even_rounds():
    assert overall_pick_number("snake", 12, 1, 1) == 1
    assert overall_pick_number("snake", 12, 1, 12) == 12
    assert overall_pick_number("snake", 12, 2, 12) == 13
    assert overall_pick_number("snake", 12, 2, 1) == 24
    assert overall_pick_number("snake", 12, 3, 1) == 25


def test_linear_never_reverses():
    assert overall_pick_number("linear", 10, 2, 1) == 11
    assert overall_pick_number("linear", 10, 3, 10) == 30


def test_third_round_reversal_flips_from_round_three():
    # Rounds 2 and 3 both run backwards, so slot 1 picks last twice in a row.
    assert overall_pick_number("snake_3rr", 10, 2, 1) == 20
    assert overall_pick_number("snake_3rr", 10, 3, 1) == 30
    assert overall_pick_number("snake_3rr", 10, 4, 1) == 31


def test_upcoming_picks_are_ordered_and_future_only():
    state = DraftState("d", rounds=4, teams=10, pick_type="snake", my_slot=3, picks_made=12)
    assert state.current_pick == 13
    assert state.my_upcoming_picks(3) == [18, 23, 38]
    assert state.my_next_pick == 18
    assert state.my_following_pick == 23


def test_parse_draft_state_reads_sleepers_shapes():
    draft = {
        "draft_id": "abc",
        "type": "snake",
        "settings": {"teams": 10, "rounds": 15, "reversal_round": 3},
        "draft_order": {"user-1": 4},
    }
    picks = [
        {"player_id": "111", "pick_no": 1, "draft_slot": 1},
        {"player_id": "222", "pick_no": 2, "draft_slot": 2},
    ]
    state = parse_draft_state(draft, picks, my_user_id="user-1")
    assert state.pick_type == "snake_3rr"
    assert state.my_slot == 4
    assert state.picks_made == 2
    assert state.current_pick == 3
    assert state.on_the_clock_slot == 3
    assert state.drafted == {"111": 1, "222": 2}


# ----------------------------------------------------------------- scarcity


def test_survival_falls_as_the_pick_gets_later():
    early = survival_probability(10, 4, pick=15)
    late = survival_probability(10, 4, pick=30)
    assert early > late
    assert survival_probability(10, 4, pick=10) == pytest.approx(0.5, abs=1e-6)


def test_survival_respects_consensus_disagreement():
    """Two players with the same rank but different spreads are not equally safe."""
    agreed = survival_probability(20, 1.0, pick=28)
    disputed = survival_probability(20, 12.0, pick=28)
    assert disputed > agreed


def test_survival_without_a_rank_is_a_coin_flip():
    assert survival_probability(None, None, pick=40) == 0.5


def test_expected_best_available_sits_between_the_options():
    group = pd.DataFrame(
        {
            "sleeper_id": ["a", "b", "c"],
            "vor": [50.0, 40.0, 30.0],
            "ecr": [5.0, 25.0, 60.0],
            "adp_sd": [2.0, 5.0, 10.0],
        }
    )
    expected = expected_best_available(group, pick=30)
    assert 30.0 < expected < 50.0


def test_excluding_a_player_lowers_the_expected_best():
    group = pd.DataFrame(
        {
            "sleeper_id": ["a", "b"],
            "vor": [50.0, 20.0],
            "ecr": [40.0, 41.0],
            "adp_sd": [3.0, 3.0],
        }
    )
    with_a = expected_best_available(group, pick=35)
    without_a = expected_best_available(group, pick=35, exclude="a")
    assert without_a < with_a


# -------------------------------------------------------------------- tiers


def test_tiers_break_at_the_cliffs():
    values = np.array([100.0, 98.0, 96.0, 60.0, 58.0, 20.0])
    tiers = assign_tiers(values)
    assert tiers[0] == tiers[1] == tiers[2]
    assert tiers[3] == tiers[4]
    assert tiers[0] < tiers[3] < tiers[5]


def test_a_flat_position_gets_one_tier():
    tiers = assign_tiers(np.array([50.0, 49.5, 49.0, 48.5, 48.0]))
    assert len(set(tiers.tolist())) == 1


# ------------------------------------------------------- replacement & value


def board_for(league, players) -> pd.DataFrame:
    frame = make_projection_frame(players)
    projections = ProjectionSet(frame, QUANTILES, season=2026)
    ranks = pd.DataFrame(
        {
            "sleeper_id": [p["id"] for p in players],
            "ecr": [p.get("ecr", 100.0) for p in players],
            "sd": [p.get("sd", 8.0) for p in players],
            "bye": [p.get("bye", 9) for p in players],
            "overall_rank": range(1, len(players) + 1),
        }
    )
    return build_board(league, projections, ranks)


def deep_pool(count_per_position=60):
    players = []
    for position, top in (("QB", 320), ("RB", 290), ("WR", 280), ("TE", 230), ("K", 150), ("DST", 150)):
        for i in range(count_per_position):
            players.append(
                {
                    "id": f"{position}{i}",
                    "position": position,
                    "name": f"{position} {i}",
                    "mean": top - i * 4.0,
                    "spread": 30.0,
                    "ecr": 1 + i * 2 + {"QB": 20, "RB": 0, "WR": 3, "TE": 30, "K": 200, "DST": 190}[position],
                    "sd": 6.0,
                }
            )
    return players


def test_replacement_level_tracks_league_shape(scoring):
    players = deep_pool()
    shallow = make_league(scoring, team_count=8)
    deep = make_league(scoring, team_count=14)
    shallow_level = replacement_levels(shallow, board_for(shallow, players))["RB"]
    deep_level = replacement_levels(deep, board_for(deep, players))["RB"]
    # More teams means digging further down the same board.
    assert deep_level < shallow_level


def test_superflex_raises_quarterback_value(scoring):
    players = deep_pool()
    one_qb = make_league(scoring)
    superflex = make_league(scoring, slots=SUPERFLEX_SLOTS)
    assert superflex.is_superflex
    assert superflex.starters_needed("QB") > one_qb.starters_needed("QB")
    levels_one = replacement_levels(one_qb, board_for(one_qb, players))
    levels_sf = replacement_levels(superflex, board_for(superflex, players))
    assert levels_sf["QB"] < levels_one["QB"]  # deeper into the pool


def test_marginal_value_equals_value_over_replacement_on_an_empty_roster(scoring, league):
    players = deep_pool()
    board = board_for(league, players)
    valuer = RosterValuer(league.slots, board)
    row = board.iloc[0]
    marginal = valuer.marginal([], row["sleeper_id"])
    assert marginal == pytest.approx(row["vor"], abs=0.01)


def test_marginal_value_collapses_once_a_position_is_full(scoring, league):
    players = deep_pool()
    board = board_for(league, players)
    valuer = RosterValuer(league.slots, board)
    quarterbacks = board[board["position"] == "QB"]["sleeper_id"].tolist()
    first = valuer.marginal([], quarterbacks[1])
    # One QB slot and no superflex: the second quarterback only helps the bench.
    second = valuer.marginal([quarterbacks[0]], quarterbacks[1])
    assert first > second
    assert second < first * 0.5


def test_recommendations_prefer_the_position_about_to_run_dry(scoring, league):
    """Two positions, equal value now, but one is about to be picked clean."""
    players = [
        {"id": "scarce1", "position": "RB", "name": "Scarce RB", "mean": 200, "ecr": 6, "sd": 1.0},
        {"id": "scarce2", "position": "RB", "name": "Next RB", "mean": 120, "ecr": 8, "sd": 1.0},
        {"id": "deep1", "position": "WR", "name": "Deep WR", "mean": 200, "ecr": 60, "sd": 1.0},
        {"id": "deep2", "position": "WR", "name": "Next WR", "mean": 198, "ecr": 62, "sd": 1.0},
    ]
    for i in range(40):
        players.append(
            {"id": f"fill{i}", "position": "TE", "name": f"TE {i}", "mean": 100 - i, "ecr": 90 + i}
        )
    board = board_for(league, players)
    state = DraftState("d", rounds=15, teams=12, pick_type="snake", my_slot=1, picks_made=0)
    advice = recommend(league, board, state, [], top_n=4)
    assert advice.recommendations[0].sleeper_id == "scarce1"
    assert advice.positional_runs["RB"] > advice.positional_runs["WR"]


def test_drafted_players_are_excluded(scoring, league):
    players = deep_pool(count_per_position=20)
    frame = make_projection_frame(players)
    projections = ProjectionSet(frame, QUANTILES, season=2026)
    ranks = pd.DataFrame(
        {
            "sleeper_id": [p["id"] for p in players],
            "ecr": [p["ecr"] for p in players],
            "sd": [p["sd"] for p in players],
            "bye": 9,
            "overall_rank": range(1, len(players) + 1),
        }
    )
    board = build_board(league, projections, ranks, drafted={"RB0", "RB1"})
    state = DraftState("d", rounds=15, teams=12, pick_type="snake", my_slot=1, picks_made=2)
    advice = recommend(league, board, state, [], top_n=6)
    taken = {c.sleeper_id for c in advice.recommendations}
    assert "RB0" not in taken and "RB1" not in taken


def test_needs_account_for_flex_eligibility(scoring, league):
    players = deep_pool(count_per_position=20)
    board = board_for(league, players)
    state = DraftState("d", rounds=15, teams=12, pick_type="snake", my_slot=1, picks_made=0)
    # Three running backs fill RB, RB, and FLEX.
    advice = recommend(league, board, state, ["RB0", "RB1", "RB2"], top_n=3)
    assert "RB" not in advice.needs
    assert "FLEX" not in advice.needs
    assert "WR" in advice.needs
