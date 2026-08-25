"""How likely is this player to be on the field at all?

Separating *will he play* from *how well will he play* matters because the two
questions have different answers and different evidence. The projection model is
trained on games players actually appeared in, so its output is conditional on
playing. Multiplying that by an availability probability gives the unconditional
expectation, and the simulator uses the same split — a coin flip for
availability, then a draw from the conditional distribution.

The rates are measured from the data rather than asserted: every official injury
report since 2016 is checked against whether the player recorded a snap that
week. Hard-coded fallbacks are used only when a designation has too few
observations to estimate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ..data.nflverse import load_injuries

log = logging.getLogger(__name__)

#: Fallbacks, used when the empirical sample for a designation is too small.
FALLBACK_RATES: dict[str, float] = {
    "": 0.97,
    "QUESTIONABLE": 0.72,
    "DOUBTFUL": 0.07,
    "OUT": 0.0,
}

#: Sleeper's live ``injury_status`` and roster ``status`` values that mean the
#: player cannot suit up regardless of what the injury report says.
SLEEPER_INACTIVE = {
    "IR",
    "INJURED RESERVE",
    "PUP",
    "PHYSICALLY UNABLE TO PERFORM",
    "NON FOOTBALL INJURY",
    "SUS",
    "SUSPENDED",
    "DNR",
    "COV",
    "PRACTICE SQUAD",
    "INACTIVE",
}

_MIN_SAMPLE = 200


@dataclass
class AvailabilityModel:
    """Play rates by injury designation, estimated from historical reports."""

    rates: dict[str, float] = field(default_factory=lambda: dict(FALLBACK_RATES))
    sample_sizes: dict[str, int] = field(default_factory=dict)

    def probability(
        self,
        report_status: str | None = None,
        *,
        sleeper_status: str | None = None,
        practice_limitation: float | None = None,
    ) -> float:
        """P(this player appears in this week's game).

        Sleeper's live status wins when it says the player is unavailable: an
        IR designation is a fact, not a probability, and it is fresher than any
        weekly file.
        """
        live = (sleeper_status or "").strip().upper()
        if live in SLEEPER_INACTIVE:
            return 0.0
        status = (live or report_status or "").strip().upper()
        if status in {"NA", "NONE", "NAN", "ACTIVE"}:
            status = ""
        base = self.rates.get(status, self.rates.get("", 0.97))
        # A questionable player who did not practise all week is a worse bet
        # than one who was limited; nudge within the designation.
        if status == "QUESTIONABLE" and practice_limitation is not None:
            if practice_limitation >= 2:
                base *= 0.62
            elif practice_limitation >= 1:
                base *= 0.95
        return float(min(max(base, 0.0), 1.0))

    def describe(self) -> str:
        parts = [
            f"{k or 'no designation'}: {v:.0%} (n={self.sample_sizes.get(k, 0)})"
            for k, v in sorted(self.rates.items(), key=lambda kv: -kv[1])
        ]
        return "; ".join(parts)


def fit_availability(panel: pd.DataFrame, seasons: tuple[int, ...]) -> AvailabilityModel:
    """Estimate play rates by matching injury reports to actual appearances."""
    injuries = load_injuries(seasons)
    model = AvailabilityModel()
    if injuries.empty or panel.empty or "gsis_id" not in injuries.columns:
        log.info("no injury history available; using fallback availability rates")
        return model

    played = panel[panel["played"] == 1]
    appeared = played[["gsis_id", "season", "week"]].drop_duplicates().assign(appeared=1)
    # The injury report covers all 53 men; the panel covers fantasy positions.
    # Without this restriction every injured offensive lineman would count as a
    # player who "did not appear", and questionable would look far more damning
    # than it is. The denominator is players who suited up at least once that
    # season — a genuine roster member, not a practice-squad name.
    roster_members = {
        (str(g), int(s))
        for g, s in played[["gsis_id", "season"]].drop_duplicates().itertuples(index=False)
    }

    inj = injuries[injuries["gsis_id"].notna()].copy()
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce")
    inj["status"] = inj["report_status"].astype(str).str.upper().str.strip()
    inj.loc[inj["status"].isin(["NAN", "NA", "NONE"]), "status"] = ""
    inj = inj.dropna(subset=["season", "week"])
    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)
    inj = inj[
        [
            (str(g), int(s)) in roster_members
            for g, s in zip(inj["gsis_id"], inj["season"])
        ]
    ]
    if inj.empty:
        log.info("no fantasy-position injury rows matched; using fallback rates")
        return model

    merged = inj.merge(appeared, on=["gsis_id", "season", "week"], how="left")
    merged["appeared"] = merged["appeared"].fillna(0)

    grouped = merged.groupby("status")["appeared"].agg(["mean", "size"])
    for status, row in grouped.iterrows():
        if int(row["size"]) < _MIN_SAMPLE:
            continue
        model.rates[str(status)] = float(row["mean"])
        model.sample_sizes[str(status)] = int(row["size"])

    # Players who never appear on an injury report are the healthy majority; the
    # report file cannot measure them, so keep the fallback for "no designation".
    model.rates.setdefault("", FALLBACK_RATES[""])
    log.info("availability rates — %s", model.describe())
    return model
