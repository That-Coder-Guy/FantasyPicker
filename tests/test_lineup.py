"""Lineup construction, including the cases greedy filling gets wrong."""

from __future__ import annotations

import numpy as np

from fantasypicker.engine.lineup import (
    optimize_points,
    optimize_win_probability,
    solve_assignment,
    swap_impacts,
)
from fantasypicker.sleeper.league import RosterSlot


def slots(*names: str) -> list[RosterSlot]:
    return [RosterSlot(i, name) for i, name in enumerate(names)]


def test_flex_takes_the_best_eligible_leftover():
    lineup = slots("RB", "WR", "FLEX")
    solution = optimize_points(
        lineup,
        ["rb1", "rb2", "wr1", "te1"],
        ["RB", "RB", "WR", "TE"],
        np.array([20.0, 14.0, 18.0, 9.0]),
    )
    assert solution.assignment[0] == "rb1"
    assert solution.assignment[1] == "wr1"
    assert solution.assignment[2] == "rb2"  # 14 beats the tight end's 9
    assert solution.objective == 52.0


def test_assignment_beats_greedy_when_a_slot_would_be_stranded():
    """Greedy puts the tight end in FLEX and leaves TE unfillable.

    With one tight end on the roster, taking him for the flex costs the whole
    TE slot. The exact solver sees that; a "best available at each slot in
    order" pass does not.
    """
    lineup = slots("FLEX", "TE")
    solution = optimize_points(
        lineup,
        ["te1", "wr1"],
        ["TE", "WR"],
        np.array([15.0, 12.0]),
    )
    assert solution.assignment[0] == "wr1"
    assert solution.assignment[1] == "te1"
    assert solution.objective == 27.0


def test_superflex_can_take_a_quarterback():
    lineup = slots("QB", "SUPER_FLEX")
    solution = optimize_points(
        lineup,
        ["qb1", "qb2", "rb1"],
        ["QB", "QB", "RB"],
        np.array([24.0, 21.0, 15.0]),
    )
    assert set(solution.assignment.values()) == {"qb1", "qb2"}


def test_unfillable_slots_are_left_empty_rather_than_filled_illegally():
    lineup = slots("QB", "K", "DEF")
    solution = solve_assignment(lineup, ["qb1"], ["QB"], np.array([20.0]))
    assert solution.assignment == {0: "qb1"}
    assert solution.bench == []


def test_empty_roster_is_handled():
    solution = optimize_points(slots("QB"), [], [], np.array([]))
    assert solution.assignment == {}
    assert solution.objective == 0.0


def _samples(rng, means, sds, n=4000):
    return np.column_stack(
        [rng.normal(mean, sd, n) for mean, sd in zip(means, sds)]
    )


def test_underdog_prefers_the_volatile_player():
    """Behind by a mile, the steady player cannot win the week."""
    rng = np.random.default_rng(0)
    lineup = slots("FLEX")
    ids = ["steady", "volatile"]
    positions = ["RB", "RB"]
    samples = _samples(rng, [12.0, 11.0], [2.0, 12.0])
    opponent = np.full(samples.shape[0], 30.0)

    points_lineup = optimize_points(lineup, ids, positions, samples.mean(axis=0))
    assert points_lineup.assignment[0] == "steady"

    win_lineup = optimize_win_probability(
        lineup, ids, positions, samples, opponent, start=points_lineup
    )
    assert win_lineup.assignment[0] == "volatile"


def test_favourite_prefers_the_steady_player():
    rng = np.random.default_rng(1)
    lineup = slots("FLEX")
    ids = ["steady", "volatile"]
    positions = ["RB", "RB"]
    samples = _samples(rng, [12.0, 12.5], [2.0, 12.0])
    opponent = np.full(samples.shape[0], 2.0)

    points_lineup = optimize_points(lineup, ids, positions, samples.mean(axis=0))
    assert points_lineup.assignment[0] == "volatile"  # higher mean

    win_lineup = optimize_win_probability(
        lineup, ids, positions, samples, opponent, start=points_lineup
    )
    assert win_lineup.assignment[0] == "steady"


def test_swap_impacts_rank_by_win_probability():
    rng = np.random.default_rng(2)
    lineup = slots("FLEX")
    ids = ["starter", "better", "worse"]
    positions = ["RB", "RB", "RB"]
    samples = _samples(rng, [10.0, 16.0, 4.0], [3.0, 3.0, 3.0])
    opponent = rng.normal(12.0, 3.0, samples.shape[0])
    current = solve_assignment(lineup, ids, positions, np.array([10.0, 0.0, 0.0]))

    impacts = swap_impacts(lineup, ids, positions, samples, opponent, current)
    assert impacts[0]["in"] == "better"
    assert impacts[0]["win_prob_delta"] > 0
    assert impacts[-1]["in"] == "worse"
    assert impacts[-1]["win_prob_delta"] < 0
