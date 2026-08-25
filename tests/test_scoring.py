"""Scoring is the foundation — if it is wrong, every projection is wrong."""

from __future__ import annotations

import pandas as pd

from fantasypicker.sleeper.scoring import ScoringRules

from .conftest import HALF_PPR


def frame(**stats) -> pd.DataFrame:
    return pd.DataFrame([{"position": "WR", **stats}])


def test_standard_receiving_line(scoring):
    points = scoring.score_offense(
        frame(position="WR", receptions=6, receiving_yards=84, receiving_tds=1)
    )
    # 6 catches × 0.5 + 84 × 0.1 + 6 = 17.4
    assert points.iloc[0] == 17.4


def test_te_premium_applies_only_to_tight_ends(scoring):
    line = {"receptions": 8, "receiving_yards": 80}
    te = scoring.score_offense(frame(position="TE", **line)).iloc[0]
    wr = scoring.score_offense(frame(position="WR", **line)).iloc[0]
    assert te - wr == 4.0  # 8 receptions × the 0.5 TE bonus


def test_passing_and_turnovers(scoring):
    points = scoring.score_offense(
        frame(
            position="QB",
            passing_yards=300,
            passing_tds=2,
            passing_interceptions=1,
            rushing_yards=25,
            fumbles_lost_total=1,
        )
    )
    # 12 + 8 - 2 + 2.5 - 2
    assert round(points.iloc[0], 2) == 18.5


def test_kicker_scoring_by_distance(scoring):
    points = scoring.score_offense(
        frame(
            position="K",
            fg_made_30_39=1,
            fg_made_40_49=1,
            fg_made_50_59=1,
            fg_missed=1,
            pat_made=3,
        )
    )
    # 3 + 4 + 5 (50-59 rolls into fgm_50p) - 1 + 3
    assert points.iloc[0] == 14.0


def test_dst_points_allowed_buckets(scoring):
    dst = pd.DataFrame(
        [
            {"position": "DST", "dst_points_allowed": 3, "dst_sacks": 4, "dst_interceptions": 2},
            {"position": "DST", "dst_points_allowed": 38, "dst_sacks": 1, "dst_interceptions": 0},
        ]
    )
    points = scoring.score_dst(dst)
    assert points.iloc[0] == 7 + 4 + 4  # 1-6 bucket, 4 sacks, 2 picks
    assert points.iloc[1] == -4 + 1  # 35+ bucket, 1 sack


def test_mixed_frame_routes_by_position(scoring):
    mixed = pd.DataFrame(
        [
            {"position": "WR", "receptions": 4, "receiving_yards": 50},
            {"position": "DST", "dst_points_allowed": 0, "dst_sacks": 2},
        ]
    )
    points = scoring.score(mixed)
    assert points.iloc[0] == 7.0
    assert points.iloc[1] == 12.0


def test_full_ppr_differs_from_half():
    half = ScoringRules.from_league({"scoring_settings": HALF_PPR})
    full = ScoringRules.from_league({"scoring_settings": {**HALF_PPR, "rec": 1.0}})
    line = frame(position="WR", receptions=10, receiving_yards=100)
    assert full.score_offense(line).iloc[0] - half.score_offense(line).iloc[0] == 5.0


def test_missing_scoring_settings_fall_back_to_standard():
    rules = ScoringRules.from_league({})
    points = rules.score_offense(frame(position="WR", receptions=5, receiving_yards=60))
    assert points.iloc[0] == 6.0  # no PPR in standard scoring
    assert rules.ppr == 0.0


def test_unsupported_keys_are_reported_not_silently_dropped():
    rules = ScoringRules.from_league(
        {"scoring_settings": {**HALF_PPR, "pass_td_50p": 2, "def_3_and_out": 1}}
    )
    assert "pass_td_50p" in rules.unsupported
    assert "def_3_and_out" in rules.unsupported
    assert "rec" not in rules.unsupported


def test_score_stat_dict_uses_league_rules():
    rules = ScoringRules.from_league({"scoring_settings": HALF_PPR})
    # A Sleeper-shaped projection: same keys as the scoring settings.
    assert rules.score_stat_dict({"rec": 5, "rec_yd": 70, "rec_td": 1}) == 15.5
    assert rules.score_stat_dict({}) is None
    assert rules.score_stat_dict(None) is None


def test_describe_is_human_readable(scoring):
    assert "half PPR" in scoring.describe()
    assert "TE +0.5" in scoring.describe()
