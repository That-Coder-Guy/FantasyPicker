"""Free-agent evaluation.

A pickup is worth what it changes about your team, not what the player scores.
That means two numbers, and they answer different questions:

* **Rest of season** — how much better does your best lineup get over the
  remaining schedule if you add him and cut your worst player? This is the
  number that decides waiver priority and FAAB.
* **This week** — how much does adding him move your win probability in the
  matchup you are actually playing? A streaming defense with a great matchup can
  be worth more this week than a stash who is worth more in October.

Sleeper's trending-adds feed is reported alongside as a crowd signal, kept
clearly separate from the model's own view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..model.predict import ProjectionSet
from ..sleeper.league import LeagueContext
from .lineup import solve_assignment


@dataclass
class WaiverTarget:
    sleeper_id: str
    name: str
    position: str
    team: str | None
    projected_points: float
    roster_gain: float
    weekly_gain: float
    drop_candidate: str | None
    drop_candidate_name: str | None
    trending_adds: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "sleeper_id": self.sleeper_id,
            "name": self.name,
            "position": self.position,
            "team": self.team,
            "projected_points": round(self.projected_points, 1),
            "roster_gain": round(self.roster_gain, 1),
            "weekly_gain": round(self.weekly_gain, 2),
            "drop_candidate": self.drop_candidate,
            "drop_candidate_name": self.drop_candidate_name,
            "trending_adds": self.trending_adds,
            "note": self.note,
        }


@dataclass
class WaiverReport:
    targets: list[WaiverTarget]
    droppable: list[dict[str, object]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _lineup_total(league: LeagueContext, rows: pd.DataFrame, column: str) -> float:
    if rows.empty:
        return 0.0
    solution = solve_assignment(
        league.slots,
        rows["sleeper_id"].astype(str).tolist(),
        rows["position"].astype(str).tolist(),
        rows[column].to_numpy(dtype=float),
    )
    return solution.objective


def find_targets(
    league: LeagueContext,
    my_roster: list[str],
    season_projections: ProjectionSet,
    weekly_projections: ProjectionSet | None = None,
    *,
    trending: dict[str, int] | None = None,
    limit: int = 15,
    pool_size: int = 120,
) -> WaiverReport:
    """Rank available free agents by what they would add to this roster."""
    if season_projections.frame.empty:
        return WaiverReport([], [], ["No projections available."])

    rostered_anywhere = league.rostered_players()
    board = season_projections.frame.copy()
    board["sleeper_id"] = board["sleeper_id"].astype(str)
    value_column = "exp_points" if "exp_points" in board.columns else "proj_mean"

    mine = board[board["sleeper_id"].isin([str(p) for p in my_roster])]
    free_agents = board[~board["sleeper_id"].isin(rostered_anywhere)]
    if free_agents.empty:
        return WaiverReport([], [], ["Every projected player is rostered."])

    base_value = _lineup_total(league, mine, value_column)

    # The player you would cut: worst by season value, restricted to players who
    # are not holding down a starting slot on their own.
    if mine.empty:
        drop_id = drop_name = None
        after_drop = mine
    else:
        ranked = mine.sort_values(value_column)
        drop_row = ranked.iloc[0]
        drop_id = str(drop_row["sleeper_id"])
        drop_name = str(drop_row.get("name"))
        after_drop = mine[mine["sleeper_id"] != drop_id]

    weekly_lookup = None
    my_weekly_total = 0.0
    if weekly_projections is not None and not weekly_projections.frame.empty:
        weekly_lookup = weekly_projections.frame.set_index("sleeper_id")
        weekly_mine = weekly_projections.subset([str(p) for p in my_roster])
        my_weekly_total = _lineup_total(league, weekly_mine, "exp_points")

    targets: list[WaiverTarget] = []
    for row in free_agents.nlargest(pool_size, value_column).itertuples(index=False):
        candidate = pd.DataFrame(
            [
                {
                    "sleeper_id": str(row.sleeper_id),
                    "position": str(row.position),
                    value_column: float(getattr(row, value_column)),
                }
            ]
        )
        combined = pd.concat([after_drop[candidate.columns], candidate], ignore_index=True)
        gain = _lineup_total(league, combined, value_column) - base_value

        weekly_gain = 0.0
        if weekly_lookup is not None and str(row.sleeper_id) in weekly_lookup.index:
            weekly_candidate = weekly_lookup.loc[[str(row.sleeper_id)]].reset_index()
            weekly_after = weekly_projections.subset(
                [p for p in my_roster if str(p) != drop_id]
            )
            weekly_combined = pd.concat(
                [
                    weekly_after[["sleeper_id", "position", "exp_points"]],
                    weekly_candidate[["sleeper_id", "position", "exp_points"]],
                ],
                ignore_index=True,
            )
            weekly_gain = (
                _lineup_total(league, weekly_combined, "exp_points") - my_weekly_total
            )

        targets.append(
            WaiverTarget(
                sleeper_id=str(row.sleeper_id),
                name=str(row.name),
                position=str(row.position),
                team=str(row.team) if pd.notna(row.team) else None,
                projected_points=float(getattr(row, value_column)),
                roster_gain=float(gain),
                weekly_gain=float(weekly_gain),
                drop_candidate=drop_id,
                drop_candidate_name=drop_name,
                trending_adds=int((trending or {}).get(str(row.sleeper_id), 0)),
                note=_note(gain, weekly_gain),
            )
        )

    targets = [t for t in targets if t.roster_gain > 0 or t.weekly_gain > 0]
    targets.sort(key=lambda t: (-t.roster_gain, -t.weekly_gain))

    droppable = [
        {
            "sleeper_id": str(r.sleeper_id),
            "name": str(r.name),
            "position": str(r.position),
            "projected_points": round(float(getattr(r, value_column)), 1),
        }
        for r in mine.nsmallest(5, value_column).itertuples(index=False)
    ]

    notes = []
    if drop_name:
        notes.append(f"Gains are measured against dropping {drop_name}.")
    if weekly_lookup is None:
        notes.append("Weekly gain unavailable — no in-week projections loaded.")

    return WaiverReport(targets[:limit], droppable, notes)


def _note(roster_gain: float, weekly_gain: float) -> str:
    if roster_gain > 15:
        return "Roster upgrade — worth real FAAB."
    if roster_gain > 3:
        return "Marginal upgrade over your worst bench spot."
    if weekly_gain > 1:
        return "Streaming option for this week only."
    return "Depth/stash."


def replacement_baseline(
    league: LeagueContext, projections: ProjectionSet, position: str
) -> float:
    """Best free agent at a position — the true bar any rostered player must clear."""
    if projections.frame.empty:
        return 0.0
    rostered = league.rostered_players()
    pool = projections.frame[
        (projections.frame["position"] == position)
        & (~projections.frame["sleeper_id"].astype(str).isin(rostered))
    ]
    if pool.empty:
        return 0.0
    column = "exp_points" if "exp_points" in pool.columns else "proj_mean"
    return float(np.nanmax(pool[column].to_numpy(dtype=float)))
