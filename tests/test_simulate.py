"""The simulator must reproduce the marginals it was given, and the correlations."""

from __future__ import annotations

import numpy as np
import pytest

from fantasypicker.engine.correlations import CorrelationModel, _nearest_positive_definite
from fantasypicker.engine.simulate import Simulator, _inverse_cdf
from fantasypicker.model.predict import ProjectionSet

from .conftest import make_projection_frame

QUANTILES = (0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95)


def make_set(players) -> ProjectionSet:
    return ProjectionSet(make_projection_frame(players), QUANTILES, season=2026, week=1)


def test_inverse_cdf_reproduces_the_fitted_quantiles():
    levels = np.array(QUANTILES)
    values = np.array([1.0, 3.0, 6.0, 9.0, 11.0, 13.0, 17.0, 24.0, 29.0])
    drawn = _inverse_cdf(levels, values, levels)
    assert np.allclose(drawn, values, atol=1e-6)


def test_inverse_cdf_extends_beyond_the_fitted_range():
    levels = np.array(QUANTILES)
    values = np.array([1.0, 3.0, 6.0, 9.0, 11.0, 13.0, 17.0, 24.0, 29.0])
    assert _inverse_cdf(levels, values, np.array([0.999]))[0] > 29.0
    assert _inverse_cdf(levels, values, np.array([0.001]))[0] < 1.0


def test_inverse_cdf_repairs_crossed_quantiles():
    levels = np.array(QUANTILES)
    crossed = np.array([1.0, 3.0, 6.0, 12.0, 11.0, 13.0, 17.0, 24.0, 29.0])
    drawn = _inverse_cdf(levels, crossed, np.array([0.4, 0.5]))
    assert drawn[0] <= drawn[1]


def test_simulated_marginals_match_the_projection():
    projections = make_set(
        [{"id": "a", "position": "WR", "mean": 12.0, "spread": 5.0}]
    )
    simulator = Simulator(projections, n_sims=40_000, seed=11)
    sampled = simulator.sample(["a"])
    scores = sampled.scores[:, 0]
    # The median of the draws should land on the fitted median.
    assert np.median(scores) == pytest.approx(12.0, abs=0.3)
    assert np.percentile(scores, 90) == pytest.approx(12.0 + 1.28 * 5.0, abs=0.6)


def test_availability_produces_real_zeros():
    projections = make_set(
        [{"id": "a", "position": "RB", "mean": 14.0, "spread": 4.0, "p_play": 0.5}]
    )
    sampled = Simulator(projections, n_sims=20_000, seed=3).sample(["a"])
    zeros = np.mean(sampled.scores[:, 0] == 0.0)
    assert zeros == pytest.approx(0.5, abs=0.02)


def test_correlated_teammates_move_together():
    projections = make_set(
        [
            {"id": "qb", "position": "QB", "mean": 20.0, "team": "KC", "opponent": "DEN"},
            {"id": "wr", "position": "WR", "mean": 14.0, "team": "KC", "opponent": "DEN"},
            {"id": "far", "position": "WR", "mean": 14.0, "team": "SF", "opponent": "LA"},
        ]
    )
    correlations = CorrelationModel({("team", "QB", "WR"): 0.45})
    sampled = Simulator(projections, correlations, n_sims=30_000, seed=5).sample(
        ["qb", "wr", "far"]
    )
    stacked = np.corrcoef(sampled.scores[:, 0], sampled.scores[:, 1])[0, 1]
    unrelated = np.corrcoef(sampled.scores[:, 0], sampled.scores[:, 2])[0, 1]
    assert stacked > 0.3
    assert abs(unrelated) < 0.05


def test_opposing_defense_moves_against_the_offence():
    projections = make_set(
        [
            {"id": "qb", "position": "QB", "mean": 20.0, "team": "KC", "opponent": "DEN"},
            {"id": "dst", "position": "DST", "mean": 7.0, "team": "DEN", "opponent": "KC"},
        ]
    )
    correlations = CorrelationModel({("opp", "DST", "QB"): -0.35})
    sampled = Simulator(projections, correlations, n_sims=30_000, seed=7).sample(["qb", "dst"])
    assert np.corrcoef(sampled.scores[:, 0], sampled.scores[:, 1])[0, 1] < -0.2


def test_stacking_widens_the_total_distribution():
    """A stacked lineup should be more volatile than an uncorrelated one.

    This is the whole reason the copula exists: two identical rosters with the
    same expected points do not have the same chance of beating a strong
    opponent.
    """
    players = [
        {"id": "qb", "position": "QB", "mean": 20.0, "team": "KC", "opponent": "DEN"},
        {"id": "wr", "position": "WR", "mean": 14.0, "team": "KC", "opponent": "DEN"},
    ]
    projections = make_set(players)
    stacked = Simulator(
        projections, CorrelationModel({("team", "QB", "WR"): 0.5}), n_sims=30_000, seed=9
    ).sample(["qb", "wr"])
    independent = Simulator(
        projections, CorrelationModel({("team", "QB", "WR"): 0.0}), n_sims=30_000, seed=9
    ).sample(["qb", "wr"])
    assert stacked.scores.sum(axis=1).std() > independent.scores.sum(axis=1).std() * 1.1


def test_unknown_players_are_dropped_not_invented():
    projections = make_set([{"id": "a", "position": "WR", "mean": 10.0}])
    sampled = Simulator(projections, n_sims=100, seed=1).sample(["a", "ghost"])
    assert sampled.ids == ["a"]
    assert sampled.scores.shape == (100, 1)


def test_inconsistent_correlations_are_repaired_into_a_usable_matrix():
    # A cycle of strong positive correlations that cannot all hold at once.
    broken = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    repaired = _nearest_positive_definite(broken)
    np.linalg.cholesky(repaired)  # raises if the repair failed
    assert np.allclose(np.diag(repaired), 1.0)
