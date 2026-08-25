"""Turn a Sleeper league's ``scoring_settings`` into fantasy points.

Why not just use nflverse's ``fantasy_points_ppr`` column? Because it is full
PPR with default touchdown values and nothing else. Real leagues have half PPR,
TE premium, first-down scoring, 100-yard bonuses, per-position reception values,
6-point passing touchdowns, and kicker scoring by distance bucket. A projection
trained on the wrong target is wrong everywhere, so points are recomputed from
box-score components under *this* league's rules — for the training labels and
for the projections alike.

Every supported Sleeper key maps to an expression over nflverse
``stats_player_week`` columns. Keys we cannot derive from weekly box scores
(e.g. ``pass_td_50p``, which needs play-by-play) are collected in
:attr:`ScoringRules.unsupported` so the app can say so out loud rather than
silently scoring them as zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

Column = Callable[[pd.DataFrame], pd.Series]


def _col(name: str) -> Column:
    def get(df: pd.DataFrame) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    return get


def _sum(*names: str) -> Column:
    def get(df: pd.DataFrame) -> pd.Series:
        total = pd.Series(0.0, index=df.index)
        for n in names:
            total = total + _col(n)(df)
        return total

    return get


def _at_least(name: str, threshold: float) -> Column:
    """Indicator for milestone bonuses (100-yard games, 300-yard passing games)."""

    def get(df: pd.DataFrame) -> pd.Series:
        return (_col(name)(df) >= threshold).astype(float)

    return get


def _sum_at_least(names: tuple[str, ...], threshold: float) -> Column:
    def get(df: pd.DataFrame) -> pd.Series:
        return (_sum(*names)(df) >= threshold).astype(float)

    return get


def _diff(a: str, b: str) -> Column:
    def get(df: pd.DataFrame) -> pd.Series:
        return (_col(a)(df) - _col(b)(df)).clip(lower=0)

    return get


def _pos_only(name: str, positions: tuple[str, ...]) -> Column:
    """A stat that only counts for certain positions (TE premium, RB-only PPR)."""

    def get(df: pd.DataFrame) -> pd.Series:
        base = _col(name)(df)
        if "position" not in df.columns:
            return base
        mask = df["position"].astype(str).str.upper().isin(positions)
        return base.where(mask, 0.0)

    return get


#: Sleeper scoring key -> expression over nflverse weekly stats.
OFFENSE_TERMS: dict[str, Column] = {
    # passing
    "pass_att": _col("attempts"),
    "pass_cmp": _col("completions"),
    "pass_inc": _diff("attempts", "completions"),
    "pass_yd": _col("passing_yards"),
    "pass_td": _col("passing_tds"),
    "pass_int": _col("passing_interceptions"),
    "pass_2pt": _col("passing_2pt_conversions"),
    "pass_fd": _col("passing_first_downs"),
    "pass_sack": _col("sacks_suffered"),
    "pass_cmp_40p": _col("passing_40"),
    "bonus_pass_yd_300": _at_least("passing_yards", 300),
    "bonus_pass_yd_400": _at_least("passing_yards", 400),
    "bonus_pass_cmp_25": _at_least("completions", 25),
    # rushing
    "rush_att": _col("carries"),
    "rush_yd": _col("rushing_yards"),
    "rush_td": _col("rushing_tds"),
    "rush_2pt": _col("rushing_2pt_conversions"),
    "rush_fd": _col("rushing_first_downs"),
    "rush_40p": _col("rushing_40"),
    "bonus_rush_yd_100": _at_least("rushing_yards", 100),
    "bonus_rush_yd_200": _at_least("rushing_yards", 200),
    "bonus_rush_att_20": _at_least("carries", 20),
    # receiving
    "rec": _col("receptions"),
    "rec_yd": _col("receiving_yards"),
    "rec_td": _col("receiving_tds"),
    "rec_2pt": _col("receiving_2pt_conversions"),
    "rec_fd": _col("receiving_first_downs"),
    "rec_tgt": _col("targets"),
    "rec_40p": _col("receiving_40"),
    "bonus_rec_yd_100": _at_least("receiving_yards", 100),
    "bonus_rec_yd_200": _at_least("receiving_yards", 200),
    # per-position reception values (TE premium and friends)
    "bonus_rec_rb": _pos_only("receptions", ("RB",)),
    "bonus_rec_wr": _pos_only("receptions", ("WR",)),
    "bonus_rec_te": _pos_only("receptions", ("TE",)),
    "rec_rb": _pos_only("receptions", ("RB",)),
    "rec_wr": _pos_only("receptions", ("WR",)),
    "rec_te": _pos_only("receptions", ("TE",)),
    # combined yardage bonuses
    "bonus_rush_rec_yd_100": _sum_at_least(("rushing_yards", "receiving_yards"), 100),
    "bonus_rush_rec_yd_200": _sum_at_least(("rushing_yards", "receiving_yards"), 200),
    # turnovers / misc
    "fum": _col("fumbles_total"),
    "fum_lost": _col("fumbles_lost_total"),
    "fum_rec_td": _col("fumble_recovery_tds"),
    "st_td": _col("special_teams_tds"),
    "st_ff": _col("def_fumbles_forced"),
    "st_fum_rec": _col("fumble_recovery_opp"),
    # kicking
    "fgm": _col("fg_made"),
    "fgm_0_19": _col("fg_made_0_19"),
    "fgm_20_29": _col("fg_made_20_29"),
    "fgm_30_39": _col("fg_made_30_39"),
    "fgm_40_49": _col("fg_made_40_49"),
    "fgm_50p": _sum("fg_made_50_59", "fg_made_60_"),
    "fgm_50_59": _col("fg_made_50_59"),
    "fgm_60p": _col("fg_made_60_"),
    "fgmiss": _col("fg_missed"),
    "fgmiss_0_19": _col("fg_missed_0_19"),
    "fgmiss_20_29": _col("fg_missed_20_29"),
    "fgmiss_30_39": _col("fg_missed_30_39"),
    "fgmiss_40_49": _col("fg_missed_40_49"),
    "fgmiss_50p": _sum("fg_missed_50_59", "fg_missed_60_"),
    "xpm": _col("pat_made"),
    "xpmiss": _col("pat_missed"),
    # individual defensive players (IDP leagues)
    "idp_tkl": _col("def_tackles_solo"),
    "idp_tkl_solo": _col("def_tackles_solo"),
    "idp_tkl_ast": _col("def_tackle_assists"),
    "idp_tkl_loss": _col("def_tackles_for_loss"),
    "idp_sack": _col("def_sacks"),
    "idp_qb_hit": _col("def_qb_hits"),
    "idp_int": _col("def_interceptions"),
    "idp_int_ret_yd": _col("def_interception_yards"),
    "idp_pass_def": _col("def_pass_defended"),
    "idp_ff": _col("def_fumbles_forced"),
    "idp_fum_rec": _col("fumble_recovery_opp"),
    "idp_def_td": _col("def_tds"),
    "idp_td": _col("def_tds"),
    "idp_safe": _col("def_safeties"),
    "idp_blk_kick": _sum("def_punt_blocks", "def_pat_blocks", "def_fg_blocks"),
}

#: Team-defense keys. Evaluated against the DST frame built in ``data.dst``.
DST_TERMS: dict[str, Column] = {
    "sack": _col("dst_sacks"),
    "int": _col("dst_interceptions"),
    "ff": _col("dst_fumbles_forced"),
    "fum_rec": _col("dst_fumble_recoveries"),
    "safe": _col("dst_safeties"),
    "def_td": _col("dst_def_tds"),
    "def_st_td": _col("dst_st_tds"),
    "st_td": _col("dst_st_tds"),
    "blk_kick": _col("dst_blocked_kicks"),
    "def_2pt": _col("dst_def_2pt"),
    "pts_allow": _col("dst_points_allowed"),
    "tkl_loss": _col("dst_tackles_for_loss"),
}

#: Points-allowed and yards-allowed buckets: key -> (low, high) inclusive range.
PTS_ALLOW_BUCKETS: dict[str, tuple[float, float]] = {
    "pts_allow_0": (0, 0),
    "pts_allow_1_6": (1, 6),
    "pts_allow_7_13": (7, 13),
    "pts_allow_14_20": (14, 20),
    "pts_allow_21_27": (21, 27),
    "pts_allow_28_34": (28, 34),
    "pts_allow_35p": (35, np.inf),
}

YDS_ALLOW_BUCKETS: dict[str, tuple[float, float]] = {
    "yds_allow_0_100": (0, 99.999),
    "yds_allow_100_199": (100, 199.999),
    "yds_allow_200_299": (200, 299.999),
    "yds_allow_300_349": (300, 349.999),
    "yds_allow_350_399": (350, 399.999),
    "yds_allow_400_449": (400, 449.999),
    "yds_allow_450_499": (450, 499.999),
    "yds_allow_500_549": (500, 549.999),
    "yds_allow_550p": (550, np.inf),
}

#: Real Sleeper keys that weekly box scores cannot reconstruct. Listed so the
#: UI can warn instead of silently treating them as zero.
KNOWN_UNSUPPORTED = {
    "pass_td_40p",
    "pass_td_50p",
    "rush_td_40p",
    "rush_td_50p",
    "rec_td_40p",
    "rec_td_50p",
    "pass_int_td",
    "blk_kick_ret_yd",
    "fgm_yds",
    "fgm_yds_over_30",
    "pr_yd",
    "kr_yd",
    "int_ret_yd",
    "fum_ret_yd",
    "sack_yd",
    # Drive-level defensive scoring needs play-by-play, not box scores.
    "def_3_and_out",
    "def_4_and_stop",
    "def_forced_punts",
    # Return touchdowns are only published as a combined special-teams total,
    # which ``st_td`` / ``def_st_td`` already cover; splitting them would
    # double-count.
    "def_pr_td",
    "def_kr_td",
}

#: Sleeper's defaults, used when a league omits a key (it usually omits the
#: ones set to zero, but a missing ``rec`` really does mean standard scoring).
STANDARD_DEFAULTS = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "fum_lost": -2.0,
}


@dataclass
class ScoringRules:
    """A league's scoring settings, applied to box-score frames."""

    settings: dict[str, float]
    unsupported: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_league(cls, league: dict | None) -> "ScoringRules":
        raw = dict((league or {}).get("scoring_settings") or {})
        clean: dict[str, float] = {}
        for key, value in raw.items():
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if val != 0.0:
                clean[key] = val
        if not clean:
            clean = dict(STANDARD_DEFAULTS)
        known = set(OFFENSE_TERMS) | set(DST_TERMS) | set(PTS_ALLOW_BUCKETS) | set(
            YDS_ALLOW_BUCKETS
        )
        unsupported = tuple(sorted(k for k in clean if k not in known))
        return cls(settings=clean, unsupported=unsupported)

    # -- descriptive helpers ---------------------------------------------- #

    @property
    def ppr(self) -> float:
        """Points per reception for a generic pass catcher."""
        return float(self.settings.get("rec", 0.0))

    @property
    def te_premium(self) -> float:
        return float(
            self.settings.get("bonus_rec_te", 0.0) or self.settings.get("rec_te", 0.0)
        )

    @property
    def passing_td_value(self) -> float:
        return float(self.settings.get("pass_td", 4.0))

    def describe(self) -> str:
        ppr = self.ppr
        label = {0.0: "standard", 0.5: "half PPR", 1.0: "full PPR"}.get(ppr)
        if label is None:
            label = f"{ppr:g} PPR"
        extras = []
        if self.te_premium:
            extras.append(f"TE +{self.te_premium:g}")
        if self.passing_td_value != 4.0:
            extras.append(f"{self.passing_td_value:g}pt pass TD")
        if self.settings.get("rec_fd") or self.settings.get("rush_fd"):
            extras.append("first downs")
        return label + (" · " + ", ".join(extras) if extras else "")

    # -- scoring ----------------------------------------------------------- #

    def score_offense(self, df: pd.DataFrame) -> pd.Series:
        """Fantasy points for every row of an nflverse weekly-stats frame."""
        if df.empty:
            return pd.Series(dtype=float)
        points = pd.Series(0.0, index=df.index)
        for key, value in self.settings.items():
            term = OFFENSE_TERMS.get(key)
            if term is not None:
                points = points + value * term(df)
        return points

    def score_dst(self, df: pd.DataFrame) -> pd.Series:
        """Fantasy points for a team-defense frame (see :mod:`data.dst`)."""
        if df.empty:
            return pd.Series(dtype=float)
        points = pd.Series(0.0, index=df.index)
        for key, value in self.settings.items():
            term = DST_TERMS.get(key)
            if term is not None:
                points = points + value * term(df)

        pts_allowed = _col("dst_points_allowed")(df)
        for key, (low, high) in PTS_ALLOW_BUCKETS.items():
            value = self.settings.get(key)
            if value:
                hit = (pts_allowed >= low) & (pts_allowed <= high)
                points = points + value * hit.astype(float)

        yds_allowed = _col("dst_yards_allowed")(df)
        for key, (low, high) in YDS_ALLOW_BUCKETS.items():
            value = self.settings.get(key)
            if value:
                hit = (yds_allowed >= low) & (yds_allowed <= high)
                points = points + value * hit.astype(float)
        return points

    def score_stat_dict(self, stats: dict[str, float] | None) -> float | None:
        """Score a Sleeper-shaped stat dictionary directly.

        Sleeper's projection endpoint reports stats under the same keys its
        scoring settings use (``pass_yd``, ``rec``, ``fgm_40_49`` …), so a
        projection can be converted to *this* league's points by pairing the two
        dictionaries — no column mapping and no assumption about which scoring
        system Sleeper had in mind.
        """
        if not stats:
            return None
        total = 0.0
        matched = False
        for key, value in self.settings.items():
            raw = stats.get(key)
            if raw is None:
                continue
            try:
                total += value * float(raw)
            except (TypeError, ValueError):
                continue
            matched = True
        return total if matched else None

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Score a mixed frame, routing DST rows to the team-defense rules."""
        if df.empty:
            return pd.Series(dtype=float)
        positions = df.get("position")
        if positions is None:
            return self.score_offense(df)
        is_dst = positions.astype(str).str.upper().isin({"DST", "DEF", "D/ST"})
        points = pd.Series(0.0, index=df.index)
        if (~is_dst).any():
            points.loc[~is_dst] = self.score_offense(df.loc[~is_dst])
        if is_dst.any():
            points.loc[is_dst] = self.score_dst(df.loc[is_dst])
        return points
