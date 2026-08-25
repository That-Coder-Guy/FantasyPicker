"""Model plumbing: leakage guards, calibration, availability, and projections."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasypicker.model.availability import AvailabilityModel, fit_availability
from fantasypicker.model.dataset import (
    USAGE_COLUMNS,
    _add_availability_features,
    _add_prior_season,
    _lagged_rolling,
    feature_columns,
)
from fantasypicker.model.train import QUANTILES, recalibrate


def panel_for(points: list[float], season: int = 2025, future: int = 0) -> pd.DataFrame:
    rows = [
        {
            "gsis_id": "p1",
            "season": season,
            "week": i + 1,
            "played": 1,
            "fantasy_points": value,
            "targets": value / 2,
        }
        for i, value in enumerate(points)
    ]
    for j in range(future):
        rows.append(
            {
                "gsis_id": "p1",
                "season": season + 1,
                "week": j + 1,
                "played": 0,
                "fantasy_points": np.nan,
                "targets": np.nan,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ leakage


def test_rolling_features_never_see_the_current_game():
    panel = panel_for([10.0, 20.0, 30.0, 40.0])
    rolled = _lagged_rolling(panel, ["fantasy_points"])
    # Week 1 has no history at all; week 2 sees only week 1.
    assert pd.isna(rolled["fantasy_points_r3"].iloc[0])
    assert rolled["fantasy_points_r3"].iloc[1] == 10.0
    assert rolled["fantasy_points_r3"].iloc[3] == pytest.approx(20.0)  # 10, 20, 30


def test_future_rows_inherit_the_last_played_window():
    """Rolling past the end of a player's history must not go blank.

    Without the carry-forward, the fourth future week and beyond would have no
    non-null values left inside the window and every form feature would be NaN
    exactly where projections are needed most.
    """
    panel = panel_for([10.0, 20.0, 30.0], future=6)
    rolled = _lagged_rolling(panel, ["fantasy_points"])
    future = rolled["fantasy_points_r3"].iloc[3:]
    assert future.notna().all()
    assert future.iloc[0] == pytest.approx(20.0)  # mean of 10, 20, 30
    assert future.nunique() == 1  # nothing new is learned between future weeks


def test_feature_columns_exclude_current_game_statistics():
    panel = panel_for([10.0, 20.0])
    panel["implied_total"] = 24.0
    panel["receiving_yards"] = 55.0
    columns = feature_columns(panel)
    assert "implied_total" in columns
    assert "fantasy_points" not in columns
    assert "receiving_yards" not in columns
    for leaky in USAGE_COLUMNS:
        assert leaky not in columns


def test_prior_season_averages_ignore_the_current_season():
    frames = [panel_for([10.0, 10.0, 10.0], season=2024), panel_for([30.0, 30.0], season=2025)]
    panel = pd.concat(frames, ignore_index=True)
    out = _add_prior_season(panel)
    season_2025 = out[out["season"] == 2025]
    assert (season_2025["prior_fantasy_points_pg"] == 10.0).all()
    assert (season_2025["prior_season_games"] == 3).all()
    season_2024 = out[out["season"] == 2024]
    assert season_2024["prior_fantasy_points_pg"].isna().all()


def test_prior_season_rescues_a_player_who_rested_in_week_18():
    """The failure this feature exists for.

    Sixteen good weeks then two rested ones leaves the three-game window at 4
    points a game. The prior-season average still says 20, which is the truth.
    """
    points = [20.0] * 16 + [4.0, 4.0]
    panel = panel_for(points, season=2025, future=1)
    rolled = pd.concat([panel, _lagged_rolling(panel, ["fantasy_points"])], axis=1)
    enriched = _add_prior_season(rolled)
    week1 = enriched[(enriched["season"] == 2026) & (enriched["week"] == 1)].iloc[0]
    assert week1["fantasy_points_r3"] == pytest.approx(9.33, abs=0.1)
    assert week1["prior_fantasy_points_pg"] == pytest.approx(18.22, abs=0.1)


def test_weeks_since_last_game_counts_across_a_season_break():
    panel = panel_for([10.0], season=2025, future=1)
    out = _add_availability_features(panel)
    gap = out["weeks_since_last_game"].iloc[1]
    assert 1 <= gap <= 22  # a season boundary, not an 80-week jump


# -------------------------------------------------------------- calibration


def test_recalibration_shifts_an_over_predicting_curve_down():
    nominal = np.asarray(QUANTILES)
    # Measured coverage above nominal means the curve sits too high.
    measured = np.asarray([0.08, 0.16, 0.34, 0.48, 0.57, 0.65, 0.79, 0.91, 0.96])
    curve = np.asarray([[1.0, 3.0, 6.0, 9.0, 11.0, 13.0, 17.0, 24.0, 29.0]])
    corrected = recalibrate(curve, nominal, measured)
    assert (corrected[0] <= curve[0] + 1e-9).all()
    assert corrected[0][4] < curve[0][4]
    assert np.all(np.diff(corrected[0]) >= -1e-9)  # still monotone


def test_recalibration_is_a_no_op_when_calibration_is_already_good():
    nominal = np.asarray(QUANTILES)
    curve = np.asarray([[1.0, 3.0, 6.0, 9.0, 11.0, 13.0, 17.0, 24.0, 29.0]])
    assert np.allclose(recalibrate(curve, nominal, nominal), curve)


# ------------------------------------------------------------- availability


def test_availability_defaults_are_sane():
    model = AvailabilityModel()
    assert model.probability("OUT") == 0.0
    assert model.probability("") > 0.9
    assert 0.4 < model.probability("QUESTIONABLE") < 0.9


def test_live_sleeper_status_overrides_the_weekly_report():
    model = AvailabilityModel()
    assert model.probability("QUESTIONABLE", sleeper_status="IR") == 0.0
    assert model.probability("", sleeper_status="PUP") == 0.0


def test_practice_participation_refines_questionable():
    model = AvailabilityModel()
    limited = model.probability("QUESTIONABLE", practice_limitation=1.0)
    absent = model.probability("QUESTIONABLE", practice_limitation=2.0)
    assert absent < limited


def test_fit_availability_measures_rates_from_appearances():
    played = pd.DataFrame(
        [
            {"gsis_id": f"p{i}", "season": 2024, "week": w, "played": 1, "position": "WR"}
            for i in range(60)
            for w in range(1, 11)
        ]
    )
    model = fit_availability(played, (2024,))
    # No injury file in the test cache, so the fallbacks stand — and must be sane.
    assert model.probability("OUT") == 0.0
    assert model.probability("") > 0.9
