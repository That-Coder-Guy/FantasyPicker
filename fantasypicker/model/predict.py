"""Turn a trained model into projections keyed by Sleeper player ID.

Three things happen here that the raw model does not do:

* **Identity.** The model speaks ``gsis_id``; rosters speak Sleeper IDs.
* **Availability.** The model is trained on games players appeared in, so its
  output is conditional on playing. It gets multiplied by P(play) — and the
  simulator keeps the two separate so a 60%-to-play star reads as a real
  bimodal outcome rather than a 60%-sized average one.
* **Thin histories.** A rookie has no rolling features worth the name. Where our
  own evidence is thin the projection is shrunk toward the market — Sleeper's
  published projections, rescored under this league's rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.crosswalk import Crosswalk
from ..sleeper.scoring import ScoringRules
from .availability import AvailabilityModel
from .train import ProjectionModel, _qname

log = logging.getLogger(__name__)

#: Games of observed history at which the model's own projection gets half the
#: weight against the market prior. Small, because the rolling features stabilise
#: fast once a player has real usage.
_SHRINKAGE_GAMES = 6.0


@dataclass
class ProjectionSet:
    """Projections for one week (or an aggregate), indexed by Sleeper ID."""

    frame: pd.DataFrame
    quantiles: tuple[float, ...]
    season: int
    week: int | None = None
    scoring: ScoringRules | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def quantile_columns(self) -> list[str]:
        return [_qname(q) for q in self.quantiles]

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def is_empty(self) -> bool:
        """True when there is nothing to look up.

        A projection set is legitimately empty — a week the schedule does not
        cover, or the preseason before any games exist — and every accessor has
        to survive that rather than raising on a missing column.
        """
        return self.frame.empty or "sleeper_id" not in self.frame.columns

    def by_id(self, sleeper_id: str) -> pd.Series | None:
        if self.is_empty:
            return None
        rows = self.frame[self.frame["sleeper_id"] == str(sleeper_id)]
        return None if rows.empty else rows.iloc[0]

    def subset(self, sleeper_ids: list[str]) -> pd.DataFrame:
        """Rows for the given players, preserving the order asked for."""
        if self.is_empty:
            return self.frame
        wanted = [str(p) for p in sleeper_ids]
        indexed = self.frame.set_index("sleeper_id")
        present = [p for p in wanted if p in indexed.index]
        if not present:
            return self.frame.iloc[0:0]
        return indexed.loc[present].reset_index()

    def matrix(self, sleeper_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """``(quantiles, p_play, ids)`` for the simulator, in the given order."""
        rows = self.subset(sleeper_ids)
        if rows.empty:
            return np.zeros((0, len(self.quantiles))), np.zeros(0), []
        quantiles = rows[self.quantile_columns].to_numpy(dtype=float)
        return quantiles, rows["p_play"].to_numpy(dtype=float), rows["sleeper_id"].tolist()


def _attach_sleeper_ids(
    frame: pd.DataFrame, crosswalk: Crosswalk, sleeper_players: dict[str, dict] | None
) -> pd.DataFrame:
    out = frame.copy()
    out["sleeper_id"] = out["gsis_id"].astype(str).map(crosswalk.gsis_to_sleeper)
    # Defenses are keyed by team abbreviation on both sides.
    is_dst = out["position"] == "DST"
    out.loc[is_dst, "sleeper_id"] = out.loc[is_dst, "team"].astype(str)

    unresolved = out["sleeper_id"].isna()
    if unresolved.any() and sleeper_players:
        out.loc[unresolved, "sleeper_id"] = [
            crosswalk.by_name(name, position)
            for name, position in zip(
                out.loc[unresolved, "player_display_name"],
                out.loc[unresolved, "position"],
            )
        ]
    dropped = int(out["sleeper_id"].isna().sum())
    if dropped:
        log.info("%d projected players could not be matched to a Sleeper ID", dropped)
    return out[out["sleeper_id"].notna()].copy()


def _availability_column(
    frame: pd.DataFrame,
    availability: AvailabilityModel,
    sleeper_players: dict[str, dict] | None,
) -> pd.Series:
    statuses: list[str | None] = []
    for sleeper_id in frame["sleeper_id"]:
        meta = (sleeper_players or {}).get(str(sleeper_id)) or {}
        statuses.append(meta.get("injury_status") or meta.get("status"))
    severity_to_report = {3.0: "OUT", 2.0: "DOUBTFUL", 1.0: "QUESTIONABLE", 0.0: ""}
    reports = frame.get("injury_severity", pd.Series(0.0, index=frame.index)).map(
        lambda s: severity_to_report.get(float(s or 0.0), "")
    )
    practice = frame.get("practice_limitation", pd.Series(0.0, index=frame.index))
    return pd.Series(
        [
            availability.probability(
                report, sleeper_status=live, practice_limitation=limit
            )
            for report, live, limit in zip(reports, statuses, practice)
        ],
        index=frame.index,
    )


def _market_prior(
    frame: pd.DataFrame,
    scoring: ScoringRules,
    sleeper_projections: object,
) -> pd.Series:
    """Sleeper's own projections, rescored under this league's rules."""
    empty = pd.Series(np.nan, index=frame.index)
    if not sleeper_projections:
        return empty

    lookup: dict[str, dict] = {}
    rows = (
        sleeper_projections.values()
        if isinstance(sleeper_projections, dict)
        else sleeper_projections
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        player_id = str(row.get("player_id") or row.get("id") or "")
        stats = row.get("stats") if isinstance(row.get("stats"), dict) else row
        if player_id:
            lookup[player_id] = stats
    if not lookup:
        return empty

    values = [
        scoring.score_stat_dict(lookup.get(str(sleeper_id)))
        for sleeper_id in frame["sleeper_id"]
    ]
    return pd.Series(values, index=frame.index, dtype=float)


def _blend_with_prior(frame: pd.DataFrame, prior: pd.Series) -> pd.DataFrame:
    """Shrink thin-history projections toward the market, and shift the whole
    quantile curve by the same amount so the spread is preserved."""
    out = frame.copy()
    have_prior = prior.notna()
    if not have_prior.any():
        out["market_points"] = np.nan
        out["model_weight"] = 1.0
        return out

    games = out.get("games_played_to_date", pd.Series(0.0, index=out.index)).fillna(0.0)
    weight = (games / (games + _SHRINKAGE_GAMES)).clip(0.15, 1.0)
    weight = weight.where(have_prior, 1.0)

    blended = weight * out["proj_mean"] + (1 - weight) * prior.fillna(out["proj_mean"])
    shift = blended - out["proj_mean"]
    for col in [c for c in out.columns if c.startswith("q") and c[1:].isdigit()]:
        out[col] = out[col] + shift
    out["proj_mean"] = blended
    out["market_points"] = prior
    out["model_weight"] = weight
    return out


def project_week(
    model: ProjectionModel,
    panel: pd.DataFrame,
    *,
    season: int,
    week: int,
    crosswalk: Crosswalk,
    scoring: ScoringRules,
    availability: AvailabilityModel | None = None,
    sleeper_players: dict[str, dict] | None = None,
    sleeper_projections: object = None,
) -> ProjectionSet:
    """Projections for every player scheduled to play in one week."""
    rows = panel[(panel["season"] == season) & (panel["week"] == week)].copy()
    if rows.empty:
        log.warning("no panel rows for %s week %s", season, week)
        return ProjectionSet(pd.DataFrame(), model.quantiles, season, week, scoring)

    predicted = model.predict_frame(rows)
    predicted = predicted[predicted["proj_mean"].notna()]
    predicted = _attach_sleeper_ids(predicted, crosswalk, sleeper_players)

    availability = availability or AvailabilityModel()
    predicted["p_play"] = _availability_column(predicted, availability, sleeper_players)

    prior = _market_prior(predicted, scoring, sleeper_projections)
    predicted = _blend_with_prior(predicted, prior)

    quantile_cols = [_qname(q) for q in model.quantiles]
    predicted["median"] = predicted[_qname(0.50)]
    predicted["floor"] = predicted[_qname(0.10)]
    predicted["ceiling"] = predicted[_qname(0.90)]
    # Unconditional expectation: a player who does not suit up scores zero.
    predicted["exp_points"] = predicted["proj_mean"] * predicted["p_play"]
    predicted["spread"] = predicted["ceiling"] - predicted["floor"]

    keep = [
        "sleeper_id",
        "gsis_id",
        "player_display_name",
        "position",
        "team",
        "opponent_team",
        "season",
        "week",
        "is_home",
        "implied_total",
        "team_spread",
        "total_line",
        "opp_fp_allowed_rel",
        "depth_rank",
        "games_played_to_date",
        # Carried so availability can be recomputed later without re-running the
        # model — see :func:`apply_availability`.
        "injury_severity",
        "practice_limitation",
        "p_play",
        "proj_mean",
        "median",
        "floor",
        "ceiling",
        "spread",
        "exp_points",
        "market_points",
        "model_weight",
    ] + quantile_cols
    keep = [c for c in keep if c in predicted.columns]
    frame = (
        predicted[keep]
        .rename(columns={"player_display_name": "name", "opponent_team": "opponent"})
        .sort_values("exp_points", ascending=False)
        .reset_index(drop=True)
    )
    return ProjectionSet(frame, model.quantiles, season, week, scoring)


def apply_availability(
    projections: ProjectionSet,
    availability: AvailabilityModel,
    sleeper_players: dict[str, dict] | None,
) -> bool:
    """Recompute ``p_play`` from fresh Sleeper player data, in place.

    Injury designations move all week — a Wednesday questionable becomes a
    Sunday-morning out — while the model's view of how a player performs *given
    that he plays* does not. Splitting the two means a status change is a
    millisecond of arithmetic over a cached frame instead of a re-projection,
    so the app can pick it up on every request. Returns True if anything moved.
    """
    frame = projections.frame
    if frame.empty:
        return False
    updated = _availability_column(frame, availability, sleeper_players)
    previous = frame.get("p_play")
    if previous is not None and np.allclose(
        previous.to_numpy(dtype=float), updated.to_numpy(dtype=float), atol=1e-9
    ):
        return False
    frame["p_play"] = updated
    frame["exp_points"] = frame["proj_mean"] * frame["p_play"]
    return True


def project_season(
    model: ProjectionModel,
    panel: pd.DataFrame,
    *,
    season: int,
    from_week: int = 1,
    through_week: int = 18,
    crosswalk: Crosswalk,
    scoring: ScoringRules,
    availability: AvailabilityModel | None = None,
    sleeper_players: dict[str, dict] | None = None,
) -> ProjectionSet:
    """Rest-of-season totals: the currency of a draft board.

    Each remaining week is projected separately and summed, so a player's bye
    week and his actual remaining schedule are both accounted for — two backs
    with identical weekly projections are not worth the same if one has already
    had his bye.

    Weekly quantiles are *not* summed. Adding week-by-week 10th percentiles
    would describe a season where a player is bad every single week, which is
    far rarer than the sum implies. The season spread is instead widened from
    the weekly spread by the square root of the number of games, the standard
    independent-sum scaling.
    """
    weeks = [w for w in range(from_week, through_week + 1)]
    frames: list[pd.DataFrame] = []
    for week in weeks:
        weekly = project_week(
            model,
            panel,
            season=season,
            week=week,
            crosswalk=crosswalk,
            scoring=scoring,
            availability=availability,
            sleeper_players=sleeper_players,
        )
        if len(weekly):
            frames.append(weekly.frame.assign(week=week))
    if not frames:
        return ProjectionSet(pd.DataFrame(), model.quantiles, season, None, scoring)

    allweeks = pd.concat(frames, ignore_index=True)
    quantile_cols = [_qname(q) for q in model.quantiles]

    grouped = allweeks.groupby("sleeper_id", as_index=False).agg(
        name=("name", "first"),
        position=("position", "first"),
        team=("team", "first"),
        games=("week", "nunique"),
        exp_points=("exp_points", "sum"),
        proj_mean=("proj_mean", "sum"),
        weekly_mean=("proj_mean", "mean"),
        weekly_p_play=("p_play", "mean"),
        weekly_median=("median", "mean"),
        weekly_floor=("floor", "mean"),
        weekly_ceiling=("ceiling", "mean"),
    )
    played_weeks = allweeks.groupby("sleeper_id")["week"].apply(sorted)
    byes = {
        pid: sorted(set(weeks) - set(w)) for pid, w in played_weeks.items()
    }
    grouped["bye_weeks"] = grouped["sleeper_id"].map(
        lambda p: ",".join(str(w) for w in byes.get(p, []))
    )

    scale = np.sqrt(grouped["games"].clip(lower=1))
    weekly_spread = (grouped["weekly_ceiling"] - grouped["weekly_floor"]).clip(lower=0)
    grouped["floor"] = grouped["exp_points"] - 0.5 * weekly_spread * scale
    grouped["ceiling"] = grouped["exp_points"] + 0.5 * weekly_spread * scale
    grouped["median"] = grouped["exp_points"]
    grouped["p_play"] = grouped["weekly_p_play"]
    grouped["spread"] = grouped["ceiling"] - grouped["floor"]

    # Keep a representative weekly quantile curve for the simulator.
    weekly_q = allweeks.groupby("sleeper_id")[quantile_cols].mean()
    grouped = grouped.merge(weekly_q, on="sleeper_id", how="left")

    frame = grouped.sort_values("exp_points", ascending=False).reset_index(drop=True)
    return ProjectionSet(
        frame,
        model.quantiles,
        season,
        None,
        scoring,
        notes=[
            f"Rest-of-season totals for weeks {from_week}–{through_week}.",
            "Season floor/ceiling scale weekly spread by sqrt(games), not by summing weekly quantiles.",
        ],
    )
