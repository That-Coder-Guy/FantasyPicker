"""Train per-position quantile models for weekly fantasy points.

**Why quantiles rather than a single number.** Every decision this app makes —
who to start, whether to reach for a player, whether to chase variance against a
stronger opponent — is a decision under uncertainty. A point estimate throws
away exactly the information those decisions need. So each position gets a
LightGBM model per quantile, and the resulting curve *is* the projection: the
median is the headline number, the spread is the risk, and the simulator samples
from the curve directly instead of assuming a shape.

**Why per position.** A quarterback's points come from 35 pass attempts and are
close to normal; a tight end's come from 5 targets and are closer to a spike at
zero with a long tail. One model would have to spend its capacity learning the
position indicator before it could learn anything else.

**Guarding against leakage and drift.** Validation is walk-forward by season:
train on everything before season *S*, score season *S*, never the reverse.
Recent seasons are up-weighted, because the sport changes faster than the data
accumulates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import get_settings
from ..sleeper.scoring import ScoringRules
from .dataset import ALL_POSITIONS, build_panel, build_training_matrix, feature_columns

log = logging.getLogger(__name__)

#: The quantile grid. Dense in the middle where lineup decisions are decided,
#: with tails wide enough to price a boom/bust swing.
QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95)

#: Per-position hyperparameters. Positions with fewer rows get shallower trees
#: and more regularisation; there are only ~2,000 kicker-weeks per four seasons.
_BASE_PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "learning_rate": 0.045,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,
}

_POSITION_OVERRIDES: dict[str, dict[str, Any]] = {
    "QB": {"num_leaves": 24, "min_data_in_leaf": 30},
    "RB": {"num_leaves": 31},
    "WR": {"num_leaves": 40, "min_data_in_leaf": 50},
    "TE": {"num_leaves": 24, "min_data_in_leaf": 40},
    "K": {"num_leaves": 12, "min_data_in_leaf": 60, "feature_fraction": 0.6},
    "DST": {"num_leaves": 12, "min_data_in_leaf": 60, "feature_fraction": 0.6},
}

_MAX_ROUNDS = 1200
_EARLY_STOPPING = 60
#: Fallback when a position has too little data to hold out a season.
_DEFAULT_ROUNDS = 300
#: Each season back from the newest is worth this much less in training.
_SEASON_DECAY = 0.82


@dataclass
class PositionModel:
    """One position's mean model plus its quantile ladder."""

    position: str
    mean_booster: lgb.Booster
    quantile_boosters: dict[float, lgb.Booster]
    features: list[str]
    n_train: int
    metrics: dict[str, float] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, quantiles)`` where quantiles is ``(n_rows, n_quantiles)``.

        Independently fitted quantile models can cross — the 0.6 model
        occasionally predicting below the 0.5 model on a given row. Sorting each
        row restores monotonicity, which is the standard fix and costs nothing
        in accuracy.
        """
        matrix = X[self.features].astype(float)
        mean = self.mean_booster.predict(matrix)
        columns = [
            self.quantile_boosters[q].predict(matrix) for q in sorted(self.quantile_boosters)
        ]
        quantiles = np.sort(np.column_stack(columns), axis=1)
        # Fantasy points can go negative (a two-interception day, a defense that
        # allowed 40) but not far, so only the extreme tail is clipped.
        return np.asarray(mean), np.clip(quantiles, -15.0, None)


def recalibrate(
    quantiles: np.ndarray, nominal: np.ndarray, empirical: np.ndarray
) -> np.ndarray:
    """Correct a quantile curve using its measured coverage.

    If the fitted 50th percentile actually sat above 56% of outcomes, then the
    value the model called "the median" is really the 56th percentile. Reading
    the predicted curve off the *empirical* coverage axis instead of the nominal
    one puts each level back where it belongs. It is a single monotone remap per
    position, so it corrects systematic bias without touching the model's
    ranking of players.
    """
    if quantiles.size == 0:
        return quantiles
    axis = np.clip(np.maximum.accumulate(np.asarray(empirical, dtype=float)), 1e-4, 1 - 1e-4)
    if np.allclose(axis, nominal, atol=0.01):
        return quantiles
    out = np.empty_like(quantiles)
    for i in range(quantiles.shape[0]):
        out[i] = np.interp(nominal, axis, quantiles[i])
    return out


@dataclass
class ProjectionModel:
    """All positions, plus the metadata that says what it was trained for."""

    positions: dict[str, PositionModel]
    quantiles: tuple[float, ...]
    features: list[str]
    scoring_key: str
    seasons: tuple[int, ...]
    trained_at: float
    validation: dict[str, dict[str, float]] = field(default_factory=dict)
    #: position -> measured coverage of each nominal quantile, from validation.
    calibration: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def predict_frame(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Attach ``proj_mean`` and ``q05``…``q95`` columns to panel rows."""
        if panel.empty:
            return panel.assign(proj_mean=[], **{_qname(q): [] for q in self.quantiles})
        out = panel.copy()
        out["proj_mean"] = np.nan
        for q in self.quantiles:
            out[_qname(q)] = np.nan
        missing = [c for c in self.features if c not in out.columns]
        for col in missing:
            out[col] = np.nan
        if missing:
            log.warning("%d features absent at prediction time, filled as NaN", len(missing))

        nominal = np.asarray(self.quantiles, dtype=float)
        for position, model in self.positions.items():
            mask = out["position"] == position
            if not mask.any():
                continue
            mean, quantiles = model.predict(out.loc[mask])
            measured = self.calibration.get(position)
            if measured:
                corrected = recalibrate(quantiles, nominal, np.asarray(measured))
                # Keep the mean consistent with the corrected curve by moving it
                # the same distance the median moved.
                median_index = int(np.argmin(np.abs(nominal - 0.5)))
                mean = mean + (corrected[:, median_index] - quantiles[:, median_index])
                quantiles = corrected
            out.loc[mask, "proj_mean"] = mean
            for i, q in enumerate(sorted(model.quantile_boosters)):
                out.loc[mask, _qname(q)] = quantiles[:, i]

        unknown = out["proj_mean"].isna() & out["position"].notna()
        if unknown.any():
            log.info("%d rows had no model for their position", int(unknown.sum()))
        return out

    def feature_importance(self, position: str, top: int = 20) -> list[tuple[str, float]]:
        model = self.positions.get(position)
        if model is None:
            return []
        booster = model.quantile_boosters[0.50]
        gains = booster.feature_importance(importance_type="gain")
        pairs = sorted(zip(model.features, gains), key=lambda kv: kv[1], reverse=True)
        total = sum(g for _, g in pairs) or 1.0
        return [(name, float(gain) / total) for name, gain in pairs[:top]]


def _qname(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def scoring_key(scoring: ScoringRules) -> str:
    """A stable fingerprint of a league's scoring, so models are not reused across
    leagues that score differently."""
    payload = json.dumps(scoring.settings, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def model_path(scoring: ScoringRules) -> Path:
    return get_settings().model_dir / f"projection_{scoring_key(scoring)}.joblib"


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def _sample_weights(seasons: pd.Series) -> np.ndarray:
    newest = int(seasons.max())
    return np.power(_SEASON_DECAY, (newest - seasons.astype(int)).to_numpy())


def _best_rounds(
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    weights: np.ndarray,
) -> int:
    """Pick a boosting-round count by early stopping on the newest season.

    A random split would leak: two weeks of the same player's season are close
    to the same observation. Holding out the most recent season instead matches
    how the model is actually used — fit on the past, predict forward.
    """
    unique = sorted(seasons.unique())
    if len(unique) < 3:
        return _DEFAULT_ROUNDS
    inner_holdout = unique[-1]
    fit = seasons < inner_holdout
    val = seasons == inner_holdout
    if fit.sum() < 200 or val.sum() < 50:
        return _DEFAULT_ROUNDS
    train_set = lgb.Dataset(X[fit], label=y[fit], weight=weights[fit.to_numpy()])
    valid_set = lgb.Dataset(X[val], label=y[val], reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=_MAX_ROUNDS,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(_EARLY_STOPPING, verbose=False)],
    )
    return max(50, int(booster.best_iteration or _DEFAULT_ROUNDS))


def _train_position(
    position: str,
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    features: list[str],
) -> PositionModel:
    params = {**_BASE_PARAMS, **_POSITION_OVERRIDES.get(position, {})}
    weights = _sample_weights(seasons)
    matrix = X[features].astype(float)

    # Round counts are chosen once per objective family: the median model stands
    # in for the whole quantile ladder, which keeps training to two early-stopping
    # runs per position instead of ten.
    mean_params = {**params, "objective": "regression_l2", "metric": "l2"}
    median_params = {**params, "objective": "quantile", "alpha": 0.5, "metric": "quantile"}
    mean_rounds = _best_rounds(mean_params, matrix, y, seasons, weights)
    quantile_rounds = _best_rounds(median_params, matrix, y, seasons, weights)
    log.debug("%s rounds: mean=%d quantile=%d", position, mean_rounds, quantile_rounds)

    dataset = lgb.Dataset(matrix, label=y, weight=weights, free_raw_data=False)
    mean_booster = lgb.train(mean_params, dataset, num_boost_round=mean_rounds)

    boosters: dict[float, lgb.Booster] = {}
    for q in QUANTILES:
        boosters[q] = lgb.train(
            {**params, "objective": "quantile", "alpha": q},
            dataset,
            num_boost_round=quantile_rounds,
        )
    return PositionModel(
        position=position,
        mean_booster=mean_booster,
        quantile_boosters=boosters,
        features=features,
        n_train=len(X),
    )


def _validate(
    position: str,
    X: pd.DataFrame,
    y: pd.Series,
    seasons: pd.Series,
    features: list[str],
) -> dict[str, float]:
    """Walk-forward validation on the most recent complete season.

    Reports both accuracy (MAE, correlation) and calibration — the share of
    outcomes that actually fell below each predicted quantile. Calibration is
    the number that matters here, because the simulator trusts the spread.
    """
    unique_seasons = sorted(seasons.unique())
    if len(unique_seasons) < 2:
        return {}
    holdout = unique_seasons[-1]
    train_mask = seasons < holdout
    test_mask = seasons == holdout
    if train_mask.sum() < 200 or test_mask.sum() < 50:
        return {}

    model = _train_position(
        position,
        X[train_mask],
        y[train_mask],
        seasons[train_mask],
        features,
    )
    mean, quantiles = model.predict(X[test_mask])
    median = quantiles[:, QUANTILES.index(0.50)]
    actual = y[test_mask].to_numpy()

    # Each estimator is judged on the loss it optimises: MAE is minimised by the
    # median and RMSE by the mean, so scoring the mean model on MAE would
    # penalise it for correctly reflecting the right-skew of fantasy scoring.
    metrics = {
        "holdout_season": float(holdout),
        "n_test": float(test_mask.sum()),
        "mae": float(np.mean(np.abs(median - actual))),
        "rmse": float(np.sqrt(np.mean((mean - actual) ** 2))),
        "bias": float(np.mean(mean - actual)),
    }
    if np.std(mean) > 0 and np.std(actual) > 0:
        metrics["pearson"] = float(np.corrcoef(mean, actual)[0, 1])
        metrics["spearman"] = float(
            pd.Series(mean).corr(pd.Series(actual), method="spearman")
        )
    # Baseline: predicting each player's own recent average. If the model cannot
    # beat this, it is not earning its keep.
    if "fantasy_points_r8" in X.columns:
        base = X.loc[test_mask, "fantasy_points_r8"].fillna(float(np.mean(actual))).to_numpy()
        metrics["baseline_mae"] = float(np.mean(np.abs(base - actual)))
        metrics["baseline_rmse"] = float(np.sqrt(np.mean((base - actual) ** 2)))
        metrics["baseline_spearman"] = float(
            pd.Series(base).corr(pd.Series(actual), method="spearman")
        )
    for i, q in enumerate(QUANTILES):
        metrics[f"coverage_{_qname(q)}"] = float(np.mean(actual <= quantiles[:, i]))
    return metrics


def train_model(
    scoring: ScoringRules,
    *,
    seasons: tuple[int, ...] | None = None,
    panel: pd.DataFrame | None = None,
    validate: bool = True,
    save: bool = True,
) -> ProjectionModel:
    """Train the full projection model for one league's scoring rules."""
    settings = get_settings()
    seasons = seasons or settings.train_seasons
    started = time.time()

    if panel is None:
        panel = build_panel(scoring, seasons)
    X, y, meta = build_training_matrix(panel)
    features = feature_columns(panel)
    positions = meta["position"].to_numpy()
    season_series = meta["season"].astype(int)

    models: dict[str, PositionModel] = {}
    validation: dict[str, dict[str, float]] = {}
    calibration: dict[str, tuple[float, ...]] = {}
    for position in ALL_POSITIONS:
        mask = positions == position
        if mask.sum() < 300:
            log.warning("skipping %s: only %d training rows", position, int(mask.sum()))
            continue
        Xp, yp, sp = X[mask], y[mask], season_series[mask]
        if validate:
            metrics = _validate(position, Xp, yp, sp, features)
            if metrics:
                validation[position] = metrics
                calibration[position] = tuple(
                    float(metrics[f"coverage_{_qname(q)}"]) for q in QUANTILES
                )
                log.info(
                    "%-4s holdout %d: MAE %.2f (baseline %.2f) spearman %.3f",
                    position,
                    int(metrics["holdout_season"]),
                    metrics["mae"],
                    metrics.get("baseline_mae", float("nan")),
                    metrics.get("spearman", float("nan")),
                )
        models[position] = _train_position(position, Xp, yp, sp, features)
        log.info("%-4s trained on %d rows", position, int(mask.sum()))

    model = ProjectionModel(
        positions=models,
        quantiles=QUANTILES,
        features=features,
        scoring_key=scoring_key(scoring),
        seasons=tuple(seasons),
        trained_at=time.time(),
        validation=validation,
        calibration=calibration,
    )
    log.info("trained %d position models in %.1fs", len(models), time.time() - started)
    if save:
        path = model_path(scoring)
        joblib.dump(model, path)
        log.info("saved model to %s", path)
    return model


def load_model(scoring: ScoringRules, *, max_age_days: float = 7.0) -> ProjectionModel | None:
    """Load a cached model for this scoring configuration, if it is fresh enough."""
    path = model_path(scoring)
    if not path.exists():
        return None
    try:
        model: ProjectionModel = joblib.load(path)
    except Exception as exc:
        log.warning("could not load %s: %s", path, exc)
        return None
    age_days = (time.time() - model.trained_at) / 86400.0
    if age_days > max_age_days:
        log.info("cached model is %.1f days old; retraining", age_days)
        return None
    return model
