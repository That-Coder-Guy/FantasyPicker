"""Weekly team-defense frames.

Fantasy team defenses score off things that live in three different places:
turnovers and sacks in the team box score, points allowed in the game result,
and yards allowed in the *opponent's* box score. This module joins those into
one row per team-week with ``dst_*`` columns matching
:data:`fantasypicker.sleeper.scoring.DST_TERMS`, so the same
:class:`~fantasypicker.sleeper.scoring.ScoringRules` object scores defenses and
skill players alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .nflverse import load_team_weekly, team_schedule


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def build_dst_weekly(seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    """One row per team-week of defensive fantasy production."""
    team = load_team_weekly(seasons)
    if team.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "season": team["season"].astype(int),
            "week": team["week"].astype(int),
            "team": team["team"],
            "opponent_team": team["opponent_team"],
            "dst_sacks": _num(team, "def_sacks"),
            "dst_interceptions": _num(team, "def_interceptions"),
            "dst_fumbles_forced": _num(team, "def_fumbles_forced"),
            "dst_fumble_recoveries": _num(team, "fumble_recovery_opp"),
            "dst_safeties": _num(team, "def_safeties"),
            "dst_def_tds": _num(team, "def_tds"),
            "dst_st_tds": _num(team, "special_teams_tds"),
            "dst_blocked_kicks": _num(team, "def_punt_blocks")
            + _num(team, "def_pat_blocks")
            + _num(team, "def_fg_blocks"),
            "dst_def_2pt": _num(team, "def_2pt_made"),
            "dst_tackles_for_loss": _num(team, "def_tackles_for_loss"),
            # Offensive production of the defense's own team, kept because it
            # predicts game script (a good offense means more defensive snaps
            # with a lead, which means more sacks and fewer garbage-time drives).
            "own_offense_yards": _num(team, "passing_yards") + _num(team, "rushing_yards"),
        }
    )

    # Yards allowed = the opponent's offensive output that week.
    offense = pd.DataFrame(
        {
            "season": team["season"].astype(int),
            "week": team["week"].astype(int),
            "opponent_team": team["team"],
            "dst_yards_allowed": _num(team, "passing_yards") + _num(team, "rushing_yards"),
            "dst_pass_yards_allowed": _num(team, "passing_yards"),
            "dst_rush_yards_allowed": _num(team, "rushing_yards"),
            "dst_takeaway_opportunities": _num(team, "attempts") + _num(team, "carries"),
        }
    )
    out = out.merge(offense, on=["season", "week", "opponent_team"], how="left")

    # Points allowed = the opponent's final score.
    sched = team_schedule(seasons)[
        [
            "season",
            "week",
            "team",
            "opponent_team",
            "opponent_score",
            "team_score",
            "is_home",
            "implied_total",
            "opponent_implied_total",
            "total_line",
            "team_spread",
            "rest_days",
            "is_dome",
            "temp",
            "wind",
            "div_game",
        ]
    ]
    out = out.merge(sched, on=["season", "week", "team", "opponent_team"], how="left")
    out = out.rename(columns={"opponent_score": "dst_points_allowed"})
    out["dst_points_allowed"] = out["dst_points_allowed"].fillna(0.0)
    out["position"] = "DST"
    out["player_id"] = out["team"]  # Sleeper keys defenses by team abbreviation
    out["player_display_name"] = out["team"] + " D/ST"
    return out


def dst_defense_allowed(dst: pd.DataFrame) -> pd.DataFrame:
    """Rolling "how generous is this defense" table, for opponent adjustments."""
    if dst.empty:
        return pd.DataFrame()
    df = dst.sort_values(["team", "season", "week"]).copy()
    for col, out_col in (
        ("dst_points_allowed", "def_pts_allowed_r5"),
        ("dst_yards_allowed", "def_yds_allowed_r5"),
        ("dst_pass_yards_allowed", "def_pass_yds_allowed_r5"),
        ("dst_rush_yards_allowed", "def_rush_yds_allowed_r5"),
    ):
        df[out_col] = (
            df.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            .astype(float)
        )
    return df[
        [
            "season",
            "week",
            "team",
            "def_pts_allowed_r5",
            "def_yds_allowed_r5",
            "def_pass_yds_allowed_r5",
            "def_rush_yds_allowed_r5",
        ]
    ].replace([np.inf, -np.inf], np.nan)
