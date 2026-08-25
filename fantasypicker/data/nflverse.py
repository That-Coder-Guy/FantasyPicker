"""Loaders for the nflverse public data releases.

nflverse publishes one GitHub release per dataset, with a per-season asset
inside it. Parquet is preferred (typed and roughly a fifth the size); CSV is the
fallback for anything not published as parquet.

The season in progress is a moving target: its files are rewritten every week,
and at the start of a season they do not exist at all. Loaders therefore use a
shorter TTL for the current season and treat a 404 as "not published yet"
rather than an error — in August, ``load_weekly_stats`` for the current season
legitimately returns nothing.
"""

from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from ..cache import fetch_file
from ..config import NFLDATA_RAW, NFLVERSE_RELEASE, get_settings

log = logging.getLogger(__name__)


def current_nfl_season(today: dt.date | None = None) -> int:
    """The season a given date belongs to (a season rolls over in March)."""
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def _read(path, **kwargs) -> pd.DataFrame:
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def _season_ttl(season: int) -> int:
    settings = get_settings()
    if season >= current_nfl_season():
        return settings.ttl_current_season
    return 30 * 24 * 3600  # a finished season never changes


def _load_release(
    tag: str, name: str, season: int | None = None, *, prefer_parquet: bool = True
) -> pd.DataFrame:
    """Fetch one nflverse release asset as a DataFrame (empty if unpublished)."""
    stem = f"{name}_{season}" if season is not None else name
    ttl = _season_ttl(season) if season is not None else get_settings().ttl_static
    exts = [".parquet", ".csv"] if prefer_parquet else [".csv"]
    for ext in exts:
        url = f"{NFLVERSE_RELEASE}/{tag}/{stem}{ext}"
        try:
            path = fetch_file(url, ttl, ext=ext, required=False)
        except Exception as exc:  # network problem on one format; try the other
            log.debug("fetch failed for %s: %s", url, exc)
            continue
        if path is None:
            continue
        try:
            return _read(path)
        except Exception as exc:
            log.warning("could not parse %s: %s", path, exc)
            continue
    log.info("nflverse asset not available: %s/%s", tag, stem)
    return pd.DataFrame()


def _concat_seasons(tag: str, name: str, seasons: tuple[int, ...]) -> pd.DataFrame:
    frames = [_load_release(tag, name, s) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #


def load_weekly_stats(seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Player box scores by week — the backbone of both labels and features."""
    seasons = seasons or get_settings().train_seasons
    df = _concat_seasons("stats_player", "stats_player_week", seasons)
    if df.empty:
        return df
    df = df[df["season_type"].astype(str).str.upper() == "REG"].copy()
    df["position"] = df["position"].astype(str).str.upper()
    df["team"] = df["team"].astype(str).str.upper()
    df["opponent_team"] = df["opponent_team"].astype(str).str.upper()
    return df


def load_team_weekly(seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Team box scores by week, offense and defense — feeds DST scoring."""
    seasons = seasons or get_settings().train_seasons
    df = _concat_seasons("stats_team", "stats_team_week", seasons)
    if df.empty:
        return df
    df = df[df["season_type"].astype(str).str.upper() == "REG"].copy()
    df["team"] = df["team"].astype(str).str.upper()
    df["opponent_team"] = df["opponent_team"].astype(str).str.upper()
    return df


def load_snap_counts(seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    seasons = seasons or get_settings().train_seasons
    df = _concat_seasons("snap_counts", "snap_counts", seasons)
    if df.empty:
        return df
    df = df[df["game_type"].astype(str).str.upper() == "REG"].copy()
    return df


def load_depth_charts(seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Depth charts normalised across nflverse's two schemas.

    Through 2024 the file was one row per player per week with a ``depth_team``
    rank. From 2025 it became a timestamped scrape — many snapshots per week,
    ranked by ``pos_rank`` within ``pos_abb`` — with no season or week column at
    all. Both are folded into ``(gsis_id, season, week, position, depth_rank)``
    here so that callers never have to care which era a season belongs to.
    """
    seasons = seasons or get_settings().train_seasons
    frames = []
    for season in seasons:
        raw = _load_release("depth_charts", "depth_charts", season)
        if raw.empty:
            continue
        frames.append(_normalize_depth_chart(raw, season))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["gsis_id", "season", "week", "position", "depth_rank"])
    return pd.concat(frames, ignore_index=True)


_SKILL_DEPTH_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}


def _normalize_depth_chart(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    if "depth_team" in raw.columns:  # legacy weekly format
        df = raw[raw["gsis_id"].notna()].copy()
        df["position"] = df["position"].astype(str).str.upper()
        df = df[df["position"].isin(_SKILL_DEPTH_POSITIONS)]
        return pd.DataFrame(
            {
                "gsis_id": df["gsis_id"].astype(str),
                "season": pd.to_numeric(df["season"], errors="coerce").fillna(season).astype(int),
                "week": pd.to_numeric(df["week"], errors="coerce").fillna(1).astype(int),
                "position": df["position"],
                "depth_rank": pd.to_numeric(df["depth_team"], errors="coerce"),
            }
        )

    if "pos_rank" not in raw.columns:
        return pd.DataFrame()

    df = raw[raw["gsis_id"].notna()].copy()
    df["position"] = df["pos_abb"].astype(str).str.upper().replace({"FB": "RB"})
    df = df[df["position"].isin(_SKILL_DEPTH_POSITIONS)]
    if df.empty:
        return pd.DataFrame()
    snapshot = pd.to_datetime(df["dt"], errors="coerce", utc=True).dt.tz_localize(None)
    df["week"] = _date_to_week(snapshot, season)
    df["snapshot"] = snapshot
    # Several scrapes land in the same week; the last one is the live chart.
    df = df.sort_values("snapshot").drop_duplicates(["gsis_id", "week"], keep="last")
    return pd.DataFrame(
        {
            "gsis_id": df["gsis_id"].astype(str),
            "season": season,
            "week": df["week"].astype(int),
            "position": df["position"],
            "depth_rank": pd.to_numeric(df["pos_rank"], errors="coerce"),
        }
    )


def _date_to_week(dates: pd.Series, season: int) -> pd.Series:
    """Map calendar dates to the NFL week they fall in (offseason -> week 1)."""
    games = load_schedule()
    weeks = games[(games["season"] == season) & (games["game_type"].astype(str).str.upper() == "REG")]
    if weeks.empty:
        return pd.Series(1, index=dates.index)
    starts = (
        weeks.assign(gameday=pd.to_datetime(weeks["gameday"], errors="coerce"))
        .groupby("week")["gameday"]
        .min()
        .sort_index()
    )
    # A scrape between two weeks' first kickoffs describes the later week — on
    # the Tuesday after week 3 the chart being published is week 4's. A scrape
    # on a kickoff day belongs to that week, hence side="left".
    bounds = starts.to_numpy()
    filled = dates.fillna(pd.Timestamp(f"{season}-09-01"))
    idx = np.searchsorted(bounds, filled.to_numpy(), side="left")
    week = pd.Series(
        starts.index.to_numpy()[np.clip(idx, 0, len(bounds) - 1)], index=dates.index
    )
    return week.astype(int)


def load_injuries(seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    seasons = seasons or get_settings().train_seasons
    return _concat_seasons("injuries", "injuries", seasons)


@lru_cache(maxsize=1)
def load_players() -> pd.DataFrame:
    """nflverse player master: birth date, draft capital, experience."""
    return _load_release("players", "players")


@lru_cache(maxsize=1)
def load_schedule() -> pd.DataFrame:
    """Every game since 1999 with betting lines, rest days, and weather.

    The betting market is the single best public estimate of how many points a
    team will score, so ``spread_line`` and ``total_line`` are load-bearing
    features rather than decoration. This file also carries future games, which
    is how projections get made for weeks that have not happened yet.
    """
    path = fetch_file(f"{NFLDATA_RAW}/games.csv", get_settings().ttl_current_season)
    df = pd.read_csv(path, low_memory=False)
    df["team_home"] = df["home_team"].astype(str).str.upper()
    df["team_away"] = df["away_team"].astype(str).str.upper()
    return df


def team_schedule(seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    """One row per team-game: opponent, home/away, Vegas view of the game.

    ``implied_total`` is the market's expected points for this team.
    ``spread_line`` in nfldata is the *home* team's line stated as points the
    home team is favoured by, so the away team's spread is its negation.
    """
    games = load_schedule()
    if seasons:
        games = games[games["season"].isin(seasons)]
    games = games[games["game_type"].astype(str).str.upper() == "REG"]

    common = [
        "game_id",
        "season",
        "week",
        "gameday",
        "gametime",
        "total_line",
        "spread_line",
        "roof",
        "surface",
        "temp",
        "wind",
        "div_game",
    ]
    home = games[common + ["team_home", "team_away", "home_rest", "home_score", "away_score"]].copy()
    home = home.rename(
        columns={
            "team_home": "team",
            "team_away": "opponent_team",
            "home_rest": "rest_days",
            "home_score": "team_score",
            "away_score": "opponent_score",
        }
    )
    home["is_home"] = 1
    home["team_spread"] = home["spread_line"]

    away = games[common + ["team_away", "team_home", "away_rest", "away_score", "home_score"]].copy()
    away = away.rename(
        columns={
            "team_away": "team",
            "team_home": "opponent_team",
            "away_rest": "rest_days",
            "away_score": "team_score",
            "home_score": "opponent_score",
        }
    )
    away["is_home"] = 0
    away["team_spread"] = -away["spread_line"]

    out = pd.concat([home, away], ignore_index=True)
    # Favourite by N in a game totalling T is expected to score (T + N) / 2.
    out["implied_total"] = (out["total_line"] + out["team_spread"]) / 2.0
    out["opponent_implied_total"] = (out["total_line"] - out["team_spread"]) / 2.0
    out["is_dome"] = out["roof"].astype(str).str.lower().isin(["dome", "closed"]).astype(int)
    out["team"] = out["team"].astype(str).str.upper()
    out["opponent_team"] = out["opponent_team"].astype(str).str.upper()
    return out
