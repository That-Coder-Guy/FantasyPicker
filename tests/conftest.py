"""Shared fixtures.

Tests never touch the network. The nflverse cache is pointed at a temp
directory, and Sleeper responses come from hand-built fixtures that mirror the
shapes documented at https://docs.sleeper.com.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasypicker.config import Settings, set_settings
from fantasypicker.sleeper.league import LeagueContext, Team, _parse_slots
from fantasypicker.sleeper.scoring import ScoringRules

HALF_PPR = {
    "pass_yd": 0.04,
    "pass_td": 4,
    "pass_int": -2,
    "rush_yd": 0.1,
    "rush_td": 6,
    "rec": 0.5,
    "rec_yd": 0.1,
    "rec_td": 6,
    "fum_lost": -2,
    "bonus_rec_te": 0.5,
    "sack": 1,
    "int": 2,
    "fum_rec": 2,
    "safe": 2,
    "def_td": 6,
    "pts_allow_0": 10,
    "pts_allow_1_6": 7,
    "pts_allow_7_13": 4,
    "pts_allow_14_20": 1,
    "pts_allow_28_34": -1,
    "pts_allow_35p": -4,
    "fgm_0_19": 3,
    "fgm_20_29": 3,
    "fgm_30_39": 3,
    "fgm_40_49": 4,
    "fgm_50p": 5,
    "fgmiss": -1,
    "xpm": 1,
}

STANDARD_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"] + ["BN"] * 6
SUPERFLEX_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX"] + ["BN"] * 7


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep every test's cache and models inside its own temp directory."""
    set_settings(Settings(home=tmp_path / "fp", train_seasons=(2023, 2024)))
    yield
    set_settings(Settings())


@pytest.fixture
def scoring() -> ScoringRules:
    return ScoringRules.from_league({"scoring_settings": HALF_PPR})


def make_league(
    scoring: ScoringRules,
    *,
    slots: list[str] | None = None,
    teams: dict[int, Team] | None = None,
    team_count: int = 12,
) -> LeagueContext:
    parsed, bench = _parse_slots(slots or STANDARD_SLOTS)
    return LeagueContext(
        league_id="test",
        raw={
            "name": "Test League",
            "season": "2026",
            "total_rosters": team_count,
            "roster_positions": slots or STANDARD_SLOTS,
        },
        scoring=scoring,
        slots=parsed,
        bench_size=bench,
        teams=teams or {},
        season=2026,
        current_week=1,
    )


@pytest.fixture
def league(scoring) -> LeagueContext:
    return make_league(scoring)


def make_projection_frame(players: list[dict]) -> pd.DataFrame:
    """Build a projection frame with a plausible quantile curve per player.

    ``mean`` and ``spread`` are turned into a symmetric-ish curve so simulator
    tests get realistic marginals without needing a trained model.
    """
    levels = np.array([0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95])
    z = np.array([-1.64, -1.28, -0.67, -0.25, 0.0, 0.25, 0.67, 1.28, 1.64])
    rows = []
    for player in players:
        mean = float(player["mean"])
        spread = float(player.get("spread", max(mean * 0.55, 2.0)))
        curve = np.maximum(mean + z * spread, -3.0)
        row = {
            "sleeper_id": str(player["id"]),
            "name": player.get("name", f"Player {player['id']}"),
            "position": player["position"],
            "team": player.get("team", "NE"),
            "opponent": player.get("opponent", "BUF"),
            "proj_mean": mean,
            "median": curve[4],
            "floor": curve[1],
            "ceiling": curve[7],
            "spread": curve[7] - curve[1],
            "p_play": float(player.get("p_play", 1.0)),
            "exp_points": mean * float(player.get("p_play", 1.0)),
        }
        for level, value in zip(levels, curve):
            row[f"q{int(round(level * 100)):02d}"] = value
        rows.append(row)
    return pd.DataFrame(rows)
