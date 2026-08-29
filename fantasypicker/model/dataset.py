"""Build the player-week panel that both training and prediction run on.

The central idea: rows for games that have *already* happened and rows for games
that have *not yet* happened live in the same table. Historical rows carry a
label; future rows do not. Every feature is computed from information available
before kickoff — rolling windows are shifted by one game, opponent strength uses
only prior weeks, and game context comes from the betting market, which is
published days ahead. Because the future rows sit at the end of each player's
history, the same rolling code fills them in with no special cases, and there is
no way for a future row to see its own outcome.

Positional note: the panel keys players by nflverse ``gsis_id`` for skill
players and by team abbreviation for defenses. Translation to Sleeper IDs
happens at the edges, in :mod:`fantasypicker.model.predict`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.dst import build_dst_weekly
from ..data.nflverse import (
    load_depth_charts,
    load_injuries,
    load_players,
    load_snap_counts,
    load_weekly_stats,
    team_schedule,
)
from ..config import get_settings
from ..sleeper.scoring import ScoringRules

log = logging.getLogger(__name__)

SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "K")
ALL_POSITIONS = SKILL_POSITIONS + ("DST",)

#: Raw weekly stats carried through the panel and used to build rolling form.
USAGE_COLUMNS = [
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_air_yards",
    "passing_first_downs",
    "passing_epa",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",
    "rushing_epa",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_first_downs",
    "receiving_epa",
    "target_share",
    "air_yards_share",
    "wopr",
    "racr",
    "fg_att",
    "fg_made",
    "pat_att",
    "fumbles_lost_total",
]

#: Rolling windows, in games. Three is "hot hand", eight is "this is who he is",
#: and the expanding career mean anchors small samples.
WINDOWS = (3, 8)

#: Which usage columns get the full rolling treatment (the rest get one window).
CORE_USAGE = [
    "fantasy_points",
    "targets",
    "target_share",
    "air_yards_share",
    "wopr",
    "carries",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "passing_yards",
    "attempts",
    "fg_att",
    "offense_pct",
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


# --------------------------------------------------------------------------- #
# base rows
# --------------------------------------------------------------------------- #


def _offense_rows(scoring: ScoringRules, seasons: tuple[int, ...]) -> pd.DataFrame:
    weekly = load_weekly_stats(seasons)
    if weekly.empty:
        return pd.DataFrame()
    weekly = weekly[weekly["position"].isin(SKILL_POSITIONS)].copy()
    weekly["fantasy_points"] = scoring.score_offense(weekly)
    keep = [
        "player_id",
        "player_display_name",
        "position",
        "season",
        "week",
        "team",
        "opponent_team",
        "fantasy_points",
    ] + [c for c in USAGE_COLUMNS if c in weekly.columns]
    out = weekly[keep].rename(columns={"player_id": "gsis_id"})
    out["played"] = 1
    return out


def _dst_rows(scoring: ScoringRules, seasons: tuple[int, ...]) -> pd.DataFrame:
    dst = build_dst_weekly(seasons)
    if dst.empty:
        return pd.DataFrame()
    dst = dst.copy()
    dst["fantasy_points"] = scoring.score_dst(dst)
    out = pd.DataFrame(
        {
            "gsis_id": dst["team"],
            "player_display_name": dst["player_display_name"],
            "position": "DST",
            "season": dst["season"],
            "week": dst["week"],
            "team": dst["team"],
            "opponent_team": dst["opponent_team"],
            "fantasy_points": dst["fantasy_points"],
            "played": 1,
            "dst_sacks": dst["dst_sacks"],
            "dst_interceptions": dst["dst_interceptions"],
            "dst_points_allowed": dst["dst_points_allowed"],
            "dst_yards_allowed": dst["dst_yards_allowed"],
            "dst_def_tds": dst["dst_def_tds"],
        }
    )
    return out


def _snap_rows(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Snap shares, keyed to gsis_id via the nflverse player master.

    Snaps matter twice over: as a feature (a back-up who just took 70% of the
    snaps is about to matter) and as a universe correction — a player can take
    snaps and record no stat at all, and those genuine zeros belong in the
    training data or every low quantile comes out too high.
    """
    snaps = load_snap_counts(seasons)
    if snaps.empty:
        return pd.DataFrame()
    players = load_players()
    if players.empty or "pfr_id" not in players.columns:
        return pd.DataFrame()
    id_map = (
        players[["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    )
    snaps = snaps.merge(
        id_map, left_on="pfr_player_id", right_on="pfr_id", how="left"
    )
    snaps = snaps[snaps["gsis_id"].notna()]
    snaps = snaps[snaps["position"].astype(str).str.upper().isin(SKILL_POSITIONS)]
    out = pd.DataFrame(
        {
            "gsis_id": snaps["gsis_id"],
            "player_display_name": snaps["player"],
            "position": snaps["position"].astype(str).str.upper(),
            "season": snaps["season"].astype(int),
            "week": snaps["week"].astype(int),
            "team": snaps["team"].astype(str).str.upper(),
            "opponent_team": snaps["opponent"].astype(str).str.upper(),
            "offense_pct": pd.to_numeric(snaps["offense_pct"], errors="coerce"),
            "st_pct": pd.to_numeric(snaps["st_pct"], errors="coerce"),
            "offense_snaps": pd.to_numeric(snaps["offense_snaps"], errors="coerce"),
        }
    )
    return out.drop_duplicates(["gsis_id", "season", "week"])


def _depth_chart_rank(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Best depth-chart position per player-week (1 = starter)."""
    charts = load_depth_charts(seasons)
    if charts.empty or "depth_rank" not in charts.columns:
        return pd.DataFrame()
    return (
        charts.dropna(subset=["gsis_id"])
        .groupby(["gsis_id", "season", "week"], as_index=False)["depth_rank"]
        .min()
    )


def _injury_status(seasons: tuple[int, ...]) -> pd.DataFrame:
    injuries = load_injuries(seasons)
    if injuries.empty or "gsis_id" not in injuries.columns:
        return pd.DataFrame()
    inj = injuries[injuries["gsis_id"].notna()].copy()
    severity = {
        "OUT": 3.0,
        "DOUBTFUL": 2.0,
        "QUESTIONABLE": 1.0,
    }
    inj["injury_severity"] = (
        inj["report_status"].astype(str).str.upper().map(severity).fillna(0.0)
    )
    practice = {
        "DID NOT PARTICIPATE IN PRACTICE": 2.0,
        "LIMITED PARTICIPATION IN PRACTICE": 1.0,
        "FULL PARTICIPATION IN PRACTICE": 0.0,
    }
    inj["practice_limitation"] = (
        inj["practice_status"].astype(str).str.upper().map(practice).fillna(0.0)
    )
    return (
        inj.groupby(["gsis_id", "season", "week"], as_index=False)[
            ["injury_severity", "practice_limitation"]
        ]
        .max()
    )


def _future_rows(
    seasons: tuple[int, ...],
    history: pd.DataFrame,
    team_overrides: dict[str, str] | None,
    through_week: int | None,
    active_players: set[str] | None = None,
) -> pd.DataFrame:
    """Rows for scheduled-but-unplayed games, one per active player per game.

    A player's team for a future game is taken from ``team_overrides`` (which
    the caller fills from Sleeper's live player data, so an August trade is
    reflected immediately) and otherwise from the last team he appeared for.

    ``active_players`` — gsis IDs Sleeper still lists as on a roster — is what
    keeps last season's since-retired quarterback out of this season's board.
    Without it, the fallback is "appeared in one of the last two seasons",
    narrowed by the current season's depth charts when those exist.
    """
    sched = team_schedule(seasons)
    if sched.empty or history.empty:
        return pd.DataFrame()
    played = history[history["played"] == 1]
    if played.empty:
        return pd.DataFrame()

    # Games that have not been played: no score recorded yet.
    future_games = sched[sched["team_score"].isna()].copy()
    if through_week is not None:
        future_games = future_games[future_games["week"] <= through_week]
    if future_games.empty:
        return pd.DataFrame()

    latest = (
        played.sort_values(["season", "week"])
        .groupby("gsis_id", as_index=False)
        .last()[["gsis_id", "player_display_name", "position", "team"]]
    )
    if team_overrides:
        latest["team"] = (
            latest["gsis_id"].map(team_overrides).fillna(latest["team"])
        )

    if active_players:
        latest = latest[latest["gsis_id"].isin(active_players)]
    else:
        # Only carry forward players who appeared recently; a 2018 practice-squad
        # tight end does not need a row for every 2026 game.
        recent_seasons = sorted(played["season"].unique())[-2:]
        recent = set(played[played["season"].isin(recent_seasons)]["gsis_id"])
        upcoming = int(future_games["season"].min())
        charts = _depth_chart_rank((upcoming,))
        if not charts.empty:
            listed = set(charts["gsis_id"].astype(str))
            # Kickers and defenses are absent from offensive depth charts, so
            # filtering on the chart alone would silently drop two positions.
            exempt = set(
                played[played["position"].isin(["K", "DST"])]["gsis_id"].astype(str)
            )
            filtered = {p for p in recent if str(p) in listed or str(p) in exempt}
            if filtered:
                recent = filtered
        latest = latest[latest["gsis_id"].isin(recent)]

    rows = latest.merge(
        future_games[["season", "week", "team", "opponent_team"]], on="team", how="inner"
    )
    rows["fantasy_points"] = np.nan
    rows["played"] = 0
    return rows


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #


def _lagged_rolling(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rolling summaries of a player's *previous games*, per player.

    Two cases have to come out right:

    * A **played** row must not see its own box score, so the window is taken
      over games strictly before it (``shift(1)``).
    * A **future** row has no box score at all, and neither do the future rows
      before it. Rolling over the raw column would walk off the end of the
      player's real history and return NaN from the fourth future week on. So
      the windows are computed over played rows only, unshifted, and then
      carried forward — a player's week-12 features are the same form numbers
      as his week-4 features, because that is genuinely all we know today.
    """
    played = df["played"] == 1
    gid = df["gsis_id"]
    sub = df.loc[played]
    sub_gid = sub["gsis_id"]

    def _expand(values: pd.Series) -> pd.Series:
        """Place a played-rows-only series back on the full index, then carry."""
        full = pd.Series(np.nan, index=df.index, dtype=float)
        full.loc[played] = values
        return full

    out: dict[str, pd.Series] = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(sub[col], errors="coerce")
        grouped = series.groupby(sub_gid, sort=False)

        specs: list[tuple[str, pd.Series, pd.Series]] = []
        for window in WINDOWS:
            specs.append(
                (
                    f"{col}_r{window}",
                    grouped.transform(lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()),
                    grouped.transform(lambda s, w=window: s.rolling(w, min_periods=1).mean()),
                )
            )
        specs.append(
            (
                f"{col}_career",
                grouped.transform(lambda s: s.shift(1).expanding(min_periods=1).mean()),
                grouped.transform(lambda s: s.expanding(min_periods=1).mean()),
            )
        )
        specs.append(
            (
                f"{col}_r3_std",
                grouped.transform(lambda s: s.shift(1).rolling(5, min_periods=2).std()),
                grouped.transform(lambda s: s.rolling(5, min_periods=2).std()),
            )
        )

        for name, lagged, current in specs:
            values = _expand(lagged)
            # Future rows inherit the value as of the player's last played game.
            carry = _expand(current).groupby(gid, sort=False).ffill()
            out[name] = values.where(played, carry)

    return pd.DataFrame(out, index=df.index)


def _add_availability_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_idx"] = df["season"] * 100 + df["week"]
    # A continuous week counter, so an offseason gap reads as a few weeks rather
    # than as the 80-odd that a season*100 index would produce.
    df["abs_week"] = (df["season"] - int(df["season"].min())) * 22 + df["week"]
    grouped = df.groupby("gsis_id", sort=False)
    df["games_played_to_date"] = grouped["played"].transform(
        lambda s: s.shift(1).fillna(0).cumsum()
    )
    df["season_game_number"] = (
        df.groupby(["gsis_id", "season"], sort=False).cumcount() + 1
    )
    # How much evidence *this* season has produced so far. In week 1 it is zero,
    # which is the model's cue to lean on the prior-season averages instead of
    # the three-game window.
    df["season_games_played"] = df.groupby(["gsis_id", "season"], sort=False)[
        "played"
    ].transform(lambda s: s.shift(1).fillna(0).cumsum())
    # Weeks since the last game a player actually appeared in: a proxy for
    # coming back off an injury, and for a bye.
    appeared_idx = df["abs_week"].where(df["played"] == 1)
    last_played_idx = appeared_idx.groupby(df["gsis_id"], sort=False).transform(
        lambda s: s.shift(1).ffill()
    )
    df["weeks_since_last_game"] = (
        (df["abs_week"] - last_played_idx).clip(lower=0).fillna(99)
    )
    df.loc[df["weeks_since_last_game"] > 20, "weeks_since_last_game"] = 20
    return df


#: Per-game rates carried over from a player's previous seasons.
_PRIOR_SEASON_COLUMNS = (
    "fantasy_points",
    "targets",
    "carries",
    "receiving_yards",
    "rushing_yards",
    "passing_yards",
    "offense_pct",
    "target_share",
)


def _add_prior_season(df: pd.DataFrame) -> pd.DataFrame:
    """Per-game averages from the previous two seasons.

    Rolling three- and eight-game windows are the right memory *within* a
    season, and badly wrong at the start of one. A team resting its starters in
    week 18 leaves its quarterback with a three-game average built from two
    series of garbage time — so the last thing the model sees about a
    twenty-point-a-game passer is ten points a game. Come week 1, that is the
    freshest evidence there is, and it is worthless.

    A full prior season is immune to that: it averages over the rested weeks and
    the blowouts alike. Both are provided, and the model decides how to weigh
    them — which is why ``season_game_number`` is a feature too.
    """
    played = df[df["played"] == 1]
    if played.empty:
        for column in _PRIOR_SEASON_COLUMNS:
            df[f"prior_{column}_pg"] = np.nan
            df[f"prior2_{column}_pg"] = np.nan
        df["prior_season_games"] = 0.0
        return df

    available = [c for c in _PRIOR_SEASON_COLUMNS if c in played.columns]
    aggregated = (
        played.groupby(["gsis_id", "season"])[available]
        .mean()
        .reset_index()
        .rename(columns={c: f"prior_{c}_pg" for c in available})
    )
    counts = (
        played.groupby(["gsis_id", "season"])["played"]
        .size()
        .reset_index(name="prior_season_games")
    )
    aggregated = aggregated.merge(counts, on=["gsis_id", "season"], how="left")

    out = df.copy()
    for offset, prefix in ((1, "prior"), (2, "prior2")):
        shifted = aggregated.copy()
        shifted["season"] = shifted["season"] + offset
        columns = {f"prior_{c}_pg": f"{prefix}_{c}_pg" for c in available}
        columns["prior_season_games"] = f"{prefix}_season_games"
        shifted = shifted.rename(columns=columns)
        out = out.merge(
            shifted[["gsis_id", "season"] + list(columns.values())],
            on=["gsis_id", "season"],
            how="left",
        )
    out["prior_season_games"] = out["prior_season_games"].fillna(0.0)
    out["prior2_season_games"] = out["prior2_season_games"].fillna(0.0)
    return out


def _add_opponent_defense(df: pd.DataFrame) -> pd.DataFrame:
    """How many fantasy points each defense has been allowing, by position.

    Computed against the league's own scoring, expressed relative to that
    week's league average so that it means the same thing in a 4-point and a
    6-point passing-touchdown league.
    """
    played = df[df["played"] == 1]
    if played.empty:
        df["opp_fp_allowed_rel"] = np.nan
        return df

    allowed = (
        played.groupby(["season", "week", "opponent_team", "position"], as_index=False)[
            "fantasy_points"
        ]
        .sum()
        .rename(columns={"opponent_team": "defense", "fantasy_points": "fp_allowed"})
    )
    league_avg = (
        allowed.groupby(["season", "week", "position"], as_index=False)["fp_allowed"]
        .mean()
        .rename(columns={"fp_allowed": "league_fp"})
    )
    allowed = allowed.merge(league_avg, on=["season", "week", "position"], how="left")
    allowed["rel"] = allowed["fp_allowed"] / allowed["league_fp"].replace(0, np.nan)

    allowed = allowed.sort_values(["defense", "position", "season", "week"])
    allowed["opp_fp_allowed_rel"] = (
        allowed.groupby(["defense", "position"])["rel"]
        .transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
    )
    # Carry the last known value forward so week 1 of a season, and future
    # weeks, inherit the defense's most recent form rather than going blank.
    allowed["opp_fp_allowed_rel"] = allowed.groupby(["defense", "position"])[
        "opp_fp_allowed_rel"
    ].ffill()

    lookup = allowed[
        ["season", "week", "defense", "position", "opp_fp_allowed_rel"]
    ].rename(columns={"defense": "opponent_team"})

    merged = df.merge(
        lookup, on=["season", "week", "opponent_team", "position"], how="left"
    )
    # Future weeks have no row in ``lookup``; fill from the defense's latest value.
    latest = (
        allowed.sort_values(["season", "week"])
        .groupby(["defense", "position"], as_index=False)
        .last()[["defense", "position", "opp_fp_allowed_rel"]]
        .rename(
            columns={
                "defense": "opponent_team",
                "opp_fp_allowed_rel": "opp_fp_allowed_latest",
            }
        )
    )
    merged = merged.merge(latest, on=["opponent_team", "position"], how="left")
    merged["opp_fp_allowed_rel"] = merged["opp_fp_allowed_rel"].fillna(
        merged["opp_fp_allowed_latest"]
    )
    return merged.drop(columns=["opp_fp_allowed_latest"])


def _add_player_attributes(df: pd.DataFrame) -> pd.DataFrame:
    players = load_players()
    if players.empty:
        for col in ("age", "years_exp", "draft_pick", "height", "weight"):
            df[col] = np.nan
        return df
    attrs = players[
        [
            "gsis_id",
            "birth_date",
            "draft_year",
            "draft_round",
            "draft_pick",
            "height",
            "weight",
            "rookie_season",
        ]
    ].drop_duplicates("gsis_id")
    out = df.merge(attrs, on="gsis_id", how="left")
    birth = pd.to_datetime(out["birth_date"], errors="coerce")
    # Age on 1 September of the season: precise enough, and stable within a year.
    season_start = pd.to_datetime(out["season"].astype(str) + "-09-01", errors="coerce")
    out["age"] = (season_start - birth).dt.days / 365.25
    out["years_exp"] = out["season"] - pd.to_numeric(out["rookie_season"], errors="coerce")
    out["draft_pick"] = pd.to_numeric(out["draft_pick"], errors="coerce").fillna(300)
    out["draft_round"] = pd.to_numeric(out["draft_round"], errors="coerce").fillna(8)
    out["height"] = pd.to_numeric(out["height"], errors="coerce")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    return out.drop(columns=["birth_date", "draft_year", "rookie_season"])


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #


def build_panel(
    scoring: ScoringRules,
    seasons: tuple[int, ...],
    *,
    include_future: bool = True,
    team_overrides: dict[str, str] | None = None,
    through_week: int | None = None,
    active_players: set[str] | None = None,
) -> pd.DataFrame:
    """Assemble the full player-week panel with features and labels."""
    frames = [_offense_rows(scoring, seasons), _dst_rows(scoring, seasons)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError(
            "No nflverse weekly stats available for seasons "
            f"{seasons}. Check network access to github.com."
        )
    history = pd.concat(frames, ignore_index=True)

    snaps = _snap_rows(seasons)
    if not snaps.empty:
        # Players who took snaps but recorded no stat line are real zeros.
        key = ["gsis_id", "season", "week"]
        known = set(map(tuple, history[key].astype(str).itertuples(index=False)))
        extra_mask = ~snaps[key].astype(str).apply(tuple, axis=1).isin(known)
        extra = snaps[extra_mask & (snaps["offense_snaps"].fillna(0) > 0)].copy()
        if not extra.empty:
            extra["fantasy_points"] = 0.0
            extra["played"] = 1
            history = pd.concat([history, extra], ignore_index=True)
        history = history.merge(
            snaps[key + ["offense_pct", "st_pct", "offense_snaps"]],
            on=key,
            how="left",
            suffixes=("", "_snap"),
        )
        for col in ("offense_pct", "st_pct", "offense_snaps"):
            if f"{col}_snap" in history.columns:
                history[col] = history[col].fillna(history[f"{col}_snap"])
                history = history.drop(columns=[f"{col}_snap"])

    if include_future:
        future = _future_rows(
            seasons, history, team_overrides, through_week, active_players
        )
        if not future.empty:
            history = pd.concat([history, future], ignore_index=True)

    panel = history.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)
    panel = panel.drop_duplicates(["gsis_id", "season", "week"], keep="first")
    panel = panel.reset_index(drop=True)

    # Weekly side inputs that are published before kickoff.
    depth = _depth_chart_rank(seasons)
    if not depth.empty:
        panel = panel.merge(depth, on=["gsis_id", "season", "week"], how="left")
        # Depth charts are published irregularly — the modern feed is a series of
        # snapshots, and in the offseason every snapshot maps to week 1. Carrying
        # the last known rank forward keeps a starter marked as a starter for the
        # rest of the season instead of going blank after week 1.
        panel["depth_rank"] = panel.groupby("gsis_id", sort=False)["depth_rank"].ffill()
    else:
        panel["depth_rank"] = np.nan
    injuries = _injury_status(seasons)
    if not injuries.empty:
        panel = panel.merge(injuries, on=["gsis_id", "season", "week"], how="left")
    else:
        panel["injury_severity"] = 0.0
        panel["practice_limitation"] = 0.0
    panel["injury_severity"] = panel["injury_severity"].fillna(0.0)
    panel["practice_limitation"] = panel["practice_limitation"].fillna(0.0)

    panel = _add_availability_features(panel)

    roll_cols = [c for c in CORE_USAGE if c in panel.columns]
    secondary = [
        c
        for c in USAGE_COLUMNS + ["dst_sacks", "dst_points_allowed", "dst_yards_allowed"]
        if c in panel.columns and c not in roll_cols
    ]
    rolling = _lagged_rolling(panel, roll_cols)
    panel = pd.concat([panel, rolling], axis=1)
    if secondary:
        extra = _lagged_rolling(panel, secondary)
        keep = [c for c in extra.columns if c.endswith("_r8") or c.endswith("_career")]
        panel = pd.concat([panel, extra[keep]], axis=1)

    context = team_schedule(seasons)[
        [
            "season",
            "week",
            "team",
            "opponent_team",
            "is_home",
            "rest_days",
            "implied_total",
            "opponent_implied_total",
            "total_line",
            "team_spread",
            "is_dome",
            "temp",
            "wind",
            "div_game",
        ]
    ]
    panel = panel.merge(
        context, on=["season", "week", "team", "opponent_team"], how="left"
    )

    panel = _add_prior_season(panel)
    panel = _add_opponent_defense(panel)
    panel = _add_player_attributes(panel)

    panel["is_favored"] = (panel["team_spread"] > 0).astype(float)
    panel["abs_spread"] = panel["team_spread"].abs()
    panel["bad_weather"] = (
        (panel["is_dome"] == 0)
        & ((panel["wind"].fillna(0) >= 15) | (panel["temp"].fillna(60) <= 25))
    ).astype(float)
    panel["week_num"] = panel["week"].astype(float)

    log.info(
        "panel: %d rows (%d labelled) across %d players",
        len(panel),
        int(panel["fantasy_points"].notna().sum()),
        panel["gsis_id"].nunique(),
    )
    return panel


def panel_path(scoring: ScoringRules, seasons: tuple[int, ...]) -> Path:
    """Where a built panel is cached.

    Keyed by the scoring fingerprint and the season range, because the labels —
    and the rolling features derived from them — are league-specific. Two
    leagues that score identically share one file, which is the common case for
    anyone running more than one half-PPR league.
    """
    digest = hashlib.sha1(
        (json.dumps(scoring.settings, sort_keys=True) + repr(seasons)).encode()
    ).hexdigest()[:12]
    return get_settings().model_dir / f"panel_{digest}.parquet"


def load_or_build_panel(
    scoring: ScoringRules,
    seasons: tuple[int, ...],
    *,
    force: bool = False,
    max_age_hours: float = 12.0,
    **kwargs,
) -> pd.DataFrame:
    """Build the panel, or reuse a recent one from disk.

    Assembling eleven seasons takes a couple of minutes, which is fine once and
    grating every time the app is opened. The cached copy is used when it is
    fresher than ``max_age_hours`` — long enough to cover a session, short
    enough that a week's new results are picked up the next day.
    """
    path = panel_path(scoring, seasons)
    if not force and path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        if age_hours < max_age_hours:
            try:
                panel = pd.read_parquet(path)
                log.info("reusing cached panel from %s (%.1fh old)", path, age_hours)
                return panel
            except Exception as exc:
                log.warning("could not read cached panel %s: %s", path, exc)

    panel = build_panel(scoring, seasons, **kwargs)
    try:
        panel.to_parquet(path, index=False)
        log.info("cached panel to %s", path)
    except Exception as exc:  # a cache miss is survivable; a crash is not
        log.warning("could not cache panel to %s: %s", path, exc)
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Every numeric column that is safe to feed the model.

    Anything derived from the current game's outcome is excluded by name: raw
    stat columns describe the game being predicted and would leak the label.
    """
    leaky = set(USAGE_COLUMNS) | {
        "fantasy_points",
        "played",
        "season",
        "week",
        "time_idx",
        "abs_week",
        "dst_sacks",
        "dst_interceptions",
        "dst_points_allowed",
        "dst_yards_allowed",
        "dst_def_tds",
        "offense_pct",
        "st_pct",
        "offense_snaps",
    }
    cols = []
    for col in panel.columns:
        if col in leaky or panel[col].dtype == object:
            continue
        if pd.api.types.is_numeric_dtype(panel[col]):
            cols.append(col)
    return sorted(cols)


#: Filled in on first use by :func:`build_training_matrix`; exported for the
#: model card and for feature-importance reporting.
FEATURE_COLUMNS: list[str] = []


def build_training_matrix(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split the panel into (features, label, metadata) for labelled rows."""
    global FEATURE_COLUMNS
    labelled = panel[panel["fantasy_points"].notna() & (panel["played"] == 1)].copy()
    FEATURE_COLUMNS = feature_columns(panel)
    X = labelled[FEATURE_COLUMNS].astype(float)
    y = labelled["fantasy_points"].astype(float)
    meta = labelled[
        ["gsis_id", "player_display_name", "position", "season", "week", "team", "opponent_team"]
    ]
    return X, y, meta
