"""Model plumbing: leakage guards, calibration, availability, and projections."""

from __future__ import annotations

import time

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


# -------------------------------------------------------------- panel cache


def _fake_panel() -> pd.DataFrame:
    """A panel with the dtype mix the real one has: floats, ints, and strings."""
    return pd.DataFrame(
        {
            "gsis_id": ["00-001", "00-002", "LAR"],
            "player_display_name": ["A Back", "A Receiver", "LA D/ST"],
            "position": ["RB", "WR", "DST"],
            "season": [2025, 2025, 2025],
            "week": [1, 1, 1],
            "played": [1, 1, 1],
            "fantasy_points": [12.5, np.nan, 7.0],
            "fantasy_points_r3": [np.nan, 8.25, 6.0],
            "implied_total": [24.5, 24.5, 19.0],
        }
    )


def test_panel_cache_round_trips_exactly(scoring, tmp_path):
    from fantasypicker.model.dataset import load_or_build_panel, panel_path

    seasons = (2024, 2025)
    built = _fake_panel()
    calls = {"n": 0}

    def fake_build(*args, **kwargs):
        calls["n"] += 1
        return built

    import fantasypicker.model.dataset as ds

    original = ds.build_panel
    ds.build_panel = fake_build
    try:
        first = load_or_build_panel(scoring, seasons)
        second = load_or_build_panel(scoring, seasons)
    finally:
        ds.build_panel = original

    assert calls["n"] == 1  # the second call came off disk
    assert panel_path(scoring, seasons).exists()
    # NaN placement and values must survive, or every rolling feature shifts.
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True), check_dtype=False
    )


def test_panel_cache_is_keyed_by_scoring(scoring):
    """Two leagues that score differently must not share a labelled panel."""
    from fantasypicker.model.dataset import panel_path
    from fantasypicker.sleeper.scoring import ScoringRules

    from .conftest import HALF_PPR

    full_ppr = ScoringRules.from_league({"scoring_settings": {**HALF_PPR, "rec": 1.0}})
    assert panel_path(scoring, (2025,)) != panel_path(full_ppr, (2025,))
    # ...and identical scoring must share one, which is the point.
    same = ScoringRules.from_league({"scoring_settings": HALF_PPR})
    assert panel_path(scoring, (2025,)) == panel_path(same, (2025,))


def test_panel_cache_is_keyed_by_season_range(scoring):
    from fantasypicker.model.dataset import panel_path

    assert panel_path(scoring, (2024, 2025)) != panel_path(scoring, (2025,))


def test_a_stale_panel_cache_is_rebuilt(scoring):
    import os

    import fantasypicker.model.dataset as ds

    calls = {"n": 0}

    def fake_build(*args, **kwargs):
        calls["n"] += 1
        return _fake_panel()

    original = ds.build_panel
    ds.build_panel = fake_build
    try:
        ds.load_or_build_panel(scoring, (2025,))
        path = ds.panel_path(scoring, (2025,))
        old = time.time() - 48 * 3600
        os.utime(path, (old, old))
        ds.load_or_build_panel(scoring, (2025,), max_age_hours=12.0)
    finally:
        ds.build_panel = original

    assert calls["n"] == 2


def test_force_rebuilds_even_a_fresh_cache(scoring):
    import fantasypicker.model.dataset as ds

    calls = {"n": 0}

    def fake_build(*args, **kwargs):
        calls["n"] += 1
        return _fake_panel()

    original = ds.build_panel
    ds.build_panel = fake_build
    try:
        ds.load_or_build_panel(scoring, (2025,))
        ds.load_or_build_panel(scoring, (2025,), force=True)
    finally:
        ds.build_panel = original

    assert calls["n"] == 2


def test_a_corrupt_panel_cache_falls_back_to_rebuilding(scoring):
    import fantasypicker.model.dataset as ds

    path = ds.panel_path(scoring, (2025,))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a parquet file")

    original = ds.build_panel
    ds.build_panel = lambda *a, **k: _fake_panel()
    try:
        panel = ds.load_or_build_panel(scoring, (2025,))
    finally:
        ds.build_panel = original
    assert len(panel) == 3


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


# ----------------------------------------------------- empty projection sets


def test_an_empty_projection_set_answers_lookups_instead_of_raising():
    """Preseason, and any week off the schedule, produce an empty set.

    Every accessor has to survive that: the frame has no columns at all, so
    naive indexing raises KeyError and takes a whole page down with it.
    """
    from fantasypicker.model.predict import ProjectionSet

    empty = ProjectionSet(pd.DataFrame(), QUANTILES, season=2026, week=1)
    assert empty.is_empty is True
    assert len(empty) == 0
    assert empty.by_id("anyone") is None
    assert empty.subset(["anyone"]).empty
    quantiles, p_play, ids = empty.matrix(["anyone"])
    assert quantiles.shape == (0, len(QUANTILES))
    assert p_play.shape == (0,)
    assert ids == []


def test_a_populated_projection_set_is_not_reported_empty():
    from fantasypicker.model.predict import ProjectionSet

    from .conftest import make_projection_frame

    populated = ProjectionSet(
        make_projection_frame([{"id": "a", "position": "RB", "mean": 10.0}]),
        QUANTILES,
        season=2026,
        week=1,
    )
    assert populated.is_empty is False
    assert populated.by_id("a") is not None
    assert populated.by_id("missing") is None


# ------------------------------------------------------- who gets projected

# A player's rolling form features carry his last real games forward for ever,
# so a retired quarterback draws a perfectly credible projection unless he is
# kept out of the panel. The symptom is him turning up on the waiver and drops
# pages as someone worth adding.


def _retirement_fixture(monkeypatch, *, active_players):
    import fantasypicker.model.dataset as ds

    schedule = pd.DataFrame(
        [
            {"season": 2025, "week": 1, "team": "PIT", "opponent_team": "CLE",
             "team_score": 20.0},
            {"season": 2026, "week": 1, "team": "PIT", "opponent_team": "CLE",
             "team_score": np.nan},
            {"season": 2026, "week": 2, "team": "PIT", "opponent_team": "BAL",
             "team_score": np.nan},
        ]
    )
    monkeypatch.setattr(ds, "team_schedule", lambda seasons: schedule)
    monkeypatch.setattr(ds, "_depth_chart_rank", lambda seasons: pd.DataFrame())

    history = pd.DataFrame(
        [
            {"gsis_id": "00-RETIRED", "player_display_name": "Retired Veteran",
             "position": "QB", "team": "PIT", "season": 2021, "week": 1,
             "played": 1, "fantasy_points": 22.0},
            {"gsis_id": "00-CURRENT", "player_display_name": "Current Starter",
             "position": "QB", "team": "PIT", "season": 2025, "week": 1,
             "played": 1, "fantasy_points": 19.0},
        ]
    )
    rows = ds._future_rows(
        (2021, 2025, 2026), history, None, None, active_players
    )
    return set(rows["player_display_name"]) if not rows.empty else set()


def test_a_retired_player_gets_no_future_rows(monkeypatch):
    """He has not taken a snap in five seasons. He is not a free agent target."""
    names = _retirement_fixture(monkeypatch, active_players=None)
    assert "Retired Veteran" not in names
    assert "Current Starter" in names


def test_recency_still_applies_when_a_roster_list_says_otherwise(monkeypatch):
    """The guard that does not depend on anyone else's metadata.

    Sleeper's player file keeps every player it has ever known, and a retired
    one can still carry the team he retired from — so an "on a roster" list is
    a narrowing signal, never a licence to skip the recency check.
    """
    names = _retirement_fixture(
        monkeypatch, active_players={"00-RETIRED", "00-CURRENT"}
    )
    assert "Retired Veteran" not in names
    assert "Current Starter" in names


def test_a_roster_list_narrows_the_recent_set(monkeypatch):
    """Both guards bite: recent *and* still rostered."""
    names = _retirement_fixture(monkeypatch, active_players={"00-RETIRED"})
    assert names == set()


def test_recency_is_measured_against_the_season_being_projected(monkeypatch):
    """Not against whichever seasons happen to be in the data.

    The panel here holds 2021 and 2025 only. "The last two seasons present"
    would call 2021 recent; counting back from the 2026 being projected does
    not.
    """
    import fantasypicker.model.dataset as ds

    assert ds.RECENT_SEASONS == 2
    names = _retirement_fixture(monkeypatch, active_players=None)
    assert "Retired Veteran" not in names


def test_a_player_who_missed_last_season_is_still_projected(monkeypatch):
    """A year lost to injury is not retirement."""
    import fantasypicker.model.dataset as ds

    schedule = pd.DataFrame(
        [
            {"season": 2026, "week": 1, "team": "PIT", "opponent_team": "CLE",
             "team_score": np.nan},
        ]
    )
    monkeypatch.setattr(ds, "team_schedule", lambda seasons: schedule)
    monkeypatch.setattr(ds, "_depth_chart_rank", lambda seasons: pd.DataFrame())
    history = pd.DataFrame(
        [
            {"gsis_id": "00-HURT", "player_display_name": "Missed A Year",
             "position": "RB", "team": "PIT", "season": 2024, "week": 1,
             "played": 1, "fantasy_points": 15.0},
        ]
    )
    rows = ds._future_rows((2024, 2026), history, None, None, {"00-HURT"})
    assert set(rows["player_display_name"]) == {"Missed A Year"}


def test_the_panel_cache_key_changes_with_the_build_version(scoring):
    """A fix nobody sees until tomorrow reads like a fix that did not work."""
    import fantasypicker.model.dataset as ds

    seasons = (2024, 2025)
    before = ds.panel_path(scoring, seasons)
    original = ds.PANEL_VERSION
    try:
        ds.PANEL_VERSION = original + 1
        assert ds.panel_path(scoring, seasons) != before
    finally:
        ds.PANEL_VERSION = original


def test_sleeper_marking_a_player_inactive_keeps_him_off_the_board():
    """A team is not proof of anything — a retired player keeps the one he
    retired from."""
    from fantasypicker.service import PickerService

    class _Crosswalk:
        def gsis(self, sleeper_id):
            return f"00-{sleeper_id}"

    service = PickerService()
    service.crosswalk = _Crosswalk()
    service.players = {
        "playing": {"team": "PIT", "active": True},
        "retired": {"team": "PIT", "active": False},
        "unknown": {"team": "PIT"},  # no flag either way: keep him
        "no_team": {"team": None, "active": True},
    }
    active = service._active_gsis_ids()
    assert "00-playing" in active
    assert "00-unknown" in active
    assert "00-retired" not in active
    assert "00-no_team" not in active
