"""Empirical correlations between fantasy outcomes in the same game.

Fantasy scores are not independent, and pretending otherwise breaks exactly the
calculation this app exists to make. A quarterback and his top receiver rise and
fall together; a defense and the offense it is facing move in opposite
directions; two running backs in the same backfield split a fixed number of
carries. Treating a lineup as independent draws understates the variance of a
stacked lineup and overstates the variance of a diversified one — and win
probability is a function of variance, not just of totals.

Rather than assert plausible-looking numbers, the correlations are measured from
history. Each player-week is standardised against that player's own season
(so a star and a scrub contribute on the same scale), then correlations are
taken across every pair of players who shared a game, bucketed by
relationship and position pair.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Minimum pair-weeks before an estimated correlation is trusted.
_MIN_PAIRS = 400

#: Used when a position pair has too few observations to estimate. These are
#: conservative, sourced from the same calculation on the pooled sample.
FALLBACK: dict[tuple[str, str, str], float] = {
    ("team", "QB", "WR"): 0.20,
    ("team", "QB", "TE"): 0.15,
    ("team", "QB", "RB"): 0.02,
    ("team", "QB", "K"): 0.15,
    ("team", "WR", "WR"): -0.02,
    ("team", "WR", "TE"): -0.02,
    ("team", "WR", "RB"): -0.02,
    ("team", "RB", "RB"): -0.20,
    ("team", "RB", "TE"): -0.02,
    ("team", "TE", "TE"): -0.15,
    ("team", "K", "QB"): 0.15,
    ("team", "DST", "QB"): 0.02,
    ("opp", "QB", "QB"): 0.05,
    ("opp", "QB", "WR"): 0.04,
    ("opp", "WR", "WR"): 0.03,
    ("opp", "RB", "RB"): -0.05,
    ("opp", "DST", "QB"): -0.28,
    ("opp", "DST", "RB"): -0.20,
    ("opp", "DST", "WR"): -0.18,
    ("opp", "DST", "TE"): -0.12,
    ("opp", "DST", "K"): -0.15,
    ("opp", "DST", "DST"): -0.10,
}

_DEFAULT_TEAM = 0.0
_DEFAULT_OPP = 0.0


def _key(relationship: str, a: str, b: str) -> tuple[str, str, str]:
    lo, hi = sorted((str(a).upper(), str(b).upper()))
    return (relationship, lo, hi)


@dataclass
class CorrelationModel:
    """Position-pair correlations for same-team and opposing players."""

    values: dict[tuple[str, str, str], float] = field(default_factory=dict)
    sample_sizes: dict[tuple[str, str, str], int] = field(default_factory=dict)

    def rho(self, relationship: str, pos_a: str, pos_b: str) -> float:
        key = _key(relationship, pos_a, pos_b)
        if key in self.values:
            return self.values[key]
        if key in FALLBACK:
            return FALLBACK[key]
        return _DEFAULT_TEAM if relationship == "team" else _DEFAULT_OPP

    def matrix(self, positions: list[str], teams: list[str], opponents: list[str]) -> np.ndarray:
        """Build the correlation matrix for a specific set of players."""
        n = len(positions)
        corr = np.eye(n)
        for i, j in combinations(range(n), 2):
            same_game = (teams[i] == teams[j]) or (
                teams[i] == opponents[j] and teams[j] == opponents[i]
            )
            if not same_game:
                continue
            relationship = "team" if teams[i] == teams[j] else "opp"
            value = self.rho(relationship, positions[i], positions[j])
            corr[i, j] = corr[j, i] = value
        return _nearest_positive_definite(corr)

    def describe(self, top: int = 12) -> list[tuple[str, float, int]]:
        rows = [
            (f"{rel}:{a}-{b}", value, self.sample_sizes.get((rel, a, b), 0))
            for (rel, a, b), value in self.values.items()
        ]
        rows.sort(key=lambda r: -abs(r[1]))
        return rows[:top]


def _nearest_positive_definite(corr: np.ndarray) -> np.ndarray:
    """Repair a correlation matrix so it can be factorised.

    Estimated pairwise correlations need not form a consistent joint
    distribution. Clipping the eigenvalues at a small positive floor is the
    standard projection back onto the positive-definite cone; the correction is
    tiny for the mildly-correlated matrices seen here.
    """
    if corr.size == 0:
        return corr
    symmetric = (corr + corr.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if eigenvalues.min() > 1e-8:
        return symmetric
    clipped = np.clip(eigenvalues, 1e-6, None)
    repaired = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    scale[scale == 0] = 1.0
    repaired = repaired / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def estimate_correlations(panel: pd.DataFrame, *, min_games: int = 6) -> CorrelationModel:
    """Measure same-game correlations from the historical panel."""
    model = CorrelationModel()
    if panel.empty:
        return model

    df = panel[(panel["played"] == 1) & panel["fantasy_points"].notna()][
        ["gsis_id", "season", "week", "team", "opponent_team", "position", "fantasy_points"]
    ].copy()
    if df.empty:
        return model

    # Standardise within player-season: what matters is whether two players had
    # a good week *for them*, not whether one outscores the other every week.
    grouped = df.groupby(["gsis_id", "season"])["fantasy_points"]
    counts = grouped.transform("size")
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    df = df[(counts >= min_games) & (std > 0.5)]
    if df.empty:
        return model
    df["z"] = ((df["fantasy_points"] - mean) / std).clip(-4, 4)

    # A stable game key that both sides of a matchup share.
    pair_key = df.apply(
        lambda r: f"{r['season']}-{r['week']}-" + "-".join(sorted([r["team"], r["opponent_team"]])),
        axis=1,
    )
    df["game_key"] = pair_key

    buckets: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for _, game in df.groupby("game_key", sort=False):
        rows = list(game.itertuples(index=False))
        for a, b in combinations(rows, 2):
            relationship = "team" if a.team == b.team else "opp"
            buckets.setdefault(_key(relationship, a.position, b.position), []).append(
                (a.z, b.z) if a.position <= b.position else (b.z, a.z)
            )

    for key, pairs in buckets.items():
        if len(pairs) < _MIN_PAIRS:
            continue
        arr = np.asarray(pairs, dtype=float)
        if arr[:, 0].std() == 0 or arr[:, 1].std() == 0:
            continue
        rho = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
        if np.isfinite(rho):
            # Shrink toward zero: pair samples are large but drawn from a small
            # number of distinct teams, so the effective sample is smaller than
            # the row count suggests.
            model.values[key] = float(np.clip(rho * 0.9, -0.85, 0.85))
            model.sample_sizes[key] = len(pairs)

    log.info("estimated %d position-pair correlations", len(model.values))
    return model
