"""Monte Carlo simulation of fantasy weeks.

The projection model produces a quantile curve per player. This module turns
those curves into joint samples, which is what every downstream question
actually needs:

* *Will I win this week?* — compare two lineup totals across many simulated
  Sundays, not two point estimates.
* *Should I start the safe player or the volatile one?* — depends entirely on
  whether you are ahead or behind, which only shows up in the distribution.
* *How much is this player worth?* — the spread of his outcomes, not just the
  middle.

The sampling is a Gaussian copula: correlated normals give the *dependence*
between players (see :mod:`.correlations`), and each player's own quantile curve
gives the *shape* of his outcomes. That keeps the marginals exactly as the model
predicted them while still letting a quarterback and his receiver boom together.
Availability is drawn separately — a player who does not suit up scores zero,
which is a different thing from a bad game.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ..model.predict import ProjectionSet
from .correlations import CorrelationModel

DEFAULT_SIMS = 20_000
#: Fantasy scores have a floor; nothing realistic goes below this.
_MIN_POINTS = -10.0


@dataclass
class SampledPlayers:
    """Per-simulation scores for an ordered set of players."""

    ids: list[str]
    names: list[str]
    positions: list[str]
    #: shape ``(n_sims, n_players)``
    scores: np.ndarray

    def index_of(self, sleeper_id: str) -> int | None:
        try:
            return self.ids.index(str(sleeper_id))
        except ValueError:
            return None

    def totals(self, sleeper_ids: list[str]) -> np.ndarray:
        cols = [self.index_of(p) for p in sleeper_ids]
        cols = [c for c in cols if c is not None]
        if not cols:
            return np.zeros(self.scores.shape[0])
        return self.scores[:, cols].sum(axis=1)


def _inverse_cdf(quantile_levels: np.ndarray, values: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Invert one player's quantile curve at uniform draws ``u``.

    Between the fitted quantiles the curve is interpolated linearly. Outside
    them it is extended with an exponential tail whose scale is set by the gap
    between the two outermost fitted quantiles — so a player whose 90th and 95th
    percentiles are far apart also gets a long tail beyond the 95th, and a
    player with a tight range does not.
    """
    values = np.maximum.accumulate(values)  # guard against residual crossing
    out = np.interp(u, quantile_levels, values)

    lo_level, hi_level = quantile_levels[0], quantile_levels[-1]

    upper = u > hi_level
    if upper.any():
        gap = max(values[-1] - values[-2], 1e-6)
        # Exponential tail matched to the slope of the last fitted segment.
        scale = gap / max(np.log((1 - quantile_levels[-2]) / (1 - hi_level)), 1e-6)
        out[upper] = values[-1] + scale * np.log(
            (1 - hi_level) / np.clip(1 - u[upper], 1e-9, None)
        )

    lower = u < lo_level
    if lower.any():
        gap = max(values[1] - values[0], 1e-6)
        scale = gap / max(np.log(quantile_levels[1] / lo_level), 1e-6)
        out[lower] = values[0] - scale * np.log(
            lo_level / np.clip(u[lower], 1e-9, None)
        )

    return np.maximum(out, _MIN_POINTS)


class Simulator:
    """Draws correlated weekly outcomes for an arbitrary set of players."""

    def __init__(
        self,
        projections: ProjectionSet,
        correlations: CorrelationModel | None = None,
        *,
        n_sims: int = DEFAULT_SIMS,
        seed: int | None = None,
    ) -> None:
        self.projections = projections
        self.correlations = correlations or CorrelationModel()
        self.n_sims = int(n_sims)
        self.rng = np.random.default_rng(seed)
        self._levels = np.asarray(projections.quantiles, dtype=float)

    def sample(self, sleeper_ids: list[str]) -> SampledPlayers:
        """Simulate ``n_sims`` weeks for the given players, jointly."""
        rows = self.projections.subset([str(p) for p in sleeper_ids])
        if rows.empty:
            return SampledPlayers([], [], [], np.zeros((self.n_sims, 0)))

        positions = rows["position"].astype(str).tolist()
        teams = rows.get("team", pd.Series([""] * len(rows))).astype(str).tolist()
        opponents = rows.get("opponent", pd.Series([""] * len(rows))).astype(str).tolist()
        corr = self.correlations.matrix(positions, teams, opponents)

        chol = np.linalg.cholesky(corr)
        normals = self.rng.standard_normal((self.n_sims, len(rows))) @ chol.T
        uniforms = _standard_normal_cdf(normals)

        quantile_values = rows[self.projections.quantile_columns].to_numpy(dtype=float)
        scores = np.empty_like(uniforms)
        for j in range(len(rows)):
            scores[:, j] = _inverse_cdf(self._levels, quantile_values[j], uniforms[:, j])

        # Availability: an independent coin per player per simulated week. It is
        # deliberately not correlated with performance — the model's conditional
        # distribution already reflects playing hurt.
        p_play = rows["p_play"].to_numpy(dtype=float)
        active = self.rng.random((self.n_sims, len(rows))) < p_play
        scores = np.where(active, scores, 0.0)

        return SampledPlayers(
            ids=rows["sleeper_id"].astype(str).tolist(),
            names=rows["name"].astype(str).tolist(),
            positions=positions,
            scores=scores,
        )


def _standard_normal_cdf(x: np.ndarray) -> np.ndarray:
    """Φ(x), clipped away from 0 and 1 so the tail inversion stays finite."""
    return np.clip(ndtr(x), 1e-9, 1 - 1e-9)


def summarize(samples: np.ndarray) -> dict[str, float]:
    """Headline statistics for a simulated point total."""
    if samples.size == 0:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p10": float(np.percentile(samples, 10)),
        "p25": float(np.percentile(samples, 25)),
        "p75": float(np.percentile(samples, 75)),
        "p90": float(np.percentile(samples, 90)),
        "std": float(np.std(samples)),
    }
