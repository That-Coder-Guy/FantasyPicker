"""Every team in the league, side by side.

Scouting is half of the game away from your own lineup: who has running back
depth to trade for, who is about to be short at tight end because of a bye, who
is actually good versus who has been lucky. All of that needs the same
projections your own lineup uses, applied to everyone else's roster.

Two lineups get computed for each team, and the gap between them is itself
information:

* **Best possible** — what they *could* start. This is the honest measure of a
  roster's strength, and the right thing to rank teams by.
* **As set** — what they *will* start. The difference is points a manager is
  leaving on their bench, which tells you who is paying attention and, if they
  are your opponent this week, how much slack you actually have.

Points-for and record come from Sleeper and describe what already happened;
projected points describe what is about to. Both are shown, because a team with
a great record and a thin roster is a very different trade partner from one
with the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..model.predict import ProjectionSet
from ..sleeper.league import LeagueContext, RosterSlot, Team
from .lineup import LineupSolution, solve_assignment

#: Positions summarised in each team's strength breakdown.
SUMMARY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


@dataclass
class TeamView:
    roster_id: int
    label: str
    owner: str
    record: str
    points_for: float
    is_me: bool
    projected_points: float
    declared_points: float
    points_left_on_bench: float
    starters: list[dict[str, object]] = field(default_factory=list)
    declared: list[dict[str, object]] = field(default_factory=list)
    bench: list[dict[str, object]] = field(default_factory=list)
    position_strength: dict[str, float] = field(default_factory=dict)
    ros_points: float = 0.0
    opponent_roster_id: int | None = None
    opponent_label: str | None = None
    byes: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "roster_id": self.roster_id,
            "label": self.label,
            "owner": self.owner,
            "record": self.record,
            "points_for": round(self.points_for, 1),
            "is_me": self.is_me,
            "projected_points": round(self.projected_points, 1),
            "declared_points": round(self.declared_points, 1),
            "points_left_on_bench": round(self.points_left_on_bench, 1),
            "starters": self.starters,
            "declared": self.declared,
            "bench": self.bench,
            "position_strength": {k: round(v, 1) for k, v in self.position_strength.items()},
            "ros_points": round(self.ros_points, 1),
            "opponent_roster_id": self.opponent_roster_id,
            "opponent_label": self.opponent_label,
            "byes": self.byes,
            "notes": self.notes,
        }


def _player_rows(
    projections: ProjectionSet, ids: list[str], slots_by_id: dict[str, str] | None = None
) -> list[dict[str, object]]:
    rows = projections.subset([str(p) for p in ids])
    out: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        sleeper_id = str(row.sleeper_id)
        out.append(
            {
                "sleeper_id": sleeper_id,
                "name": getattr(row, "name", sleeper_id),
                "position": getattr(row, "position", None),
                "team": getattr(row, "team", None),
                "opponent": getattr(row, "opponent", None),
                "projection": round(float(getattr(row, "proj_mean", 0.0) or 0.0), 1),
                "floor": round(float(getattr(row, "floor", 0.0) or 0.0), 1),
                "ceiling": round(float(getattr(row, "ceiling", 0.0) or 0.0), 1),
                "p_play": round(float(getattr(row, "p_play", 1.0) or 0.0), 2),
                "slot": (slots_by_id or {}).get(sleeper_id),
            }
        )
    return out


def _solve(
    slots: list[RosterSlot], projections: ProjectionSet, ids: list[str]
) -> tuple[LineupSolution, pd.DataFrame]:
    rows = projections.subset([str(p) for p in ids])
    if rows.empty:
        return LineupSolution({}, slots, [], 0.0), rows
    solution = solve_assignment(
        slots,
        rows["sleeper_id"].astype(str).tolist(),
        rows["position"].astype(str).tolist(),
        rows["exp_points"].to_numpy(dtype=float),
    )
    return solution, rows


def _bye_counts(rows: pd.DataFrame, projections: ProjectionSet) -> dict[str, int]:
    """Players with no game this week, by position — the usual reason a team
    suddenly cannot fill its lineup."""
    del projections
    if rows.empty or "opponent" not in rows.columns:
        return {}
    idle = rows[rows["opponent"].isna() | (rows["opponent"].astype(str) == "")]
    if idle.empty:
        return {}
    return {str(k): int(v) for k, v in idle["position"].value_counts().items()}


def build_league_view(
    league: LeagueContext,
    projections: ProjectionSet,
    *,
    season_projections: ProjectionSet | None = None,
    matchup_rows: list[dict] | None = None,
) -> list[TeamView]:
    """One :class:`TeamView` per team, ranked by best-possible projected points."""
    slots = league.slots

    # Sleeper pairs teams by a shared matchup_id for the week.
    opponent_of: dict[int, int] = {}
    by_matchup: dict[object, list[int]] = {}
    declared_by_roster: dict[int, list[str]] = {}
    for row in matchup_rows or []:
        roster_id = row.get("roster_id")
        if roster_id is None:
            continue
        roster_id = int(roster_id)
        declared_by_roster[roster_id] = [
            str(p) for p in (row.get("starters") or []) if p and p != "0"
        ]
        matchup_id = row.get("matchup_id")
        if matchup_id is not None:
            by_matchup.setdefault(matchup_id, []).append(roster_id)
    for members in by_matchup.values():
        if len(members) == 2:
            opponent_of[members[0]] = members[1]
            opponent_of[members[1]] = members[0]

    ros_lookup: dict[str, float] = {}
    if season_projections is not None and not season_projections.frame.empty:
        frame = season_projections.frame
        column = "exp_points" if "exp_points" in frame.columns else "proj_mean"
        ros_lookup = dict(
            zip(frame["sleeper_id"].astype(str), frame[column].astype(float))
        )

    views: list[TeamView] = []
    for roster_id, team in sorted(league.teams.items()):
        view = _build_team_view(
            league,
            team,
            projections,
            slots,
            declared=declared_by_roster.get(roster_id, list(team.starters)),
            opponent_id=opponent_of.get(roster_id),
            ros_lookup=ros_lookup,
        )
        views.append(view)

    labels = {v.roster_id: v.label for v in views}
    for view in views:
        if view.opponent_roster_id is not None:
            view.opponent_label = labels.get(view.opponent_roster_id)

    views.sort(key=lambda v: -v.projected_points)
    return views


def _build_team_view(
    league: LeagueContext,
    team: Team,
    projections: ProjectionSet,
    slots: list[RosterSlot],
    *,
    declared: list[str],
    opponent_id: int | None,
    ros_lookup: dict[str, float],
) -> TeamView:
    best, rows = _solve(slots, projections, team.active_players)
    best_ids = best.starters
    slots_by_id = {pid: best.slot_name(i) for i, pid in best.assignment.items()}

    declared_ids = [p for p in declared if p]
    declared_solution, declared_rows = _solve(slots, projections, declared_ids)
    declared_slots = {
        pid: declared_solution.slot_name(i)
        for i, pid in declared_solution.assignment.items()
    }

    projected = float(best.objective)
    declared_points = float(declared_solution.objective)

    # A declared starter we have no projection for (a just-signed call-up, a
    # rookie the ID crosswalk has not caught up with) contributes nothing to the
    # total. Left alone that reads as a manager benching their whole team, so
    # unresolved starters are counted and reported rather than scored as zero.
    unresolved = len(declared_ids) - len(declared_rows)

    notes: list[str] = []
    trust_declared = True
    if not declared_ids:
        notes.append("No lineup set yet.")
        trust_declared = False
    elif len(declared_rows) == 0:
        notes.append("Their lineup could not be read — no projections for anyone in it.")
        trust_declared = False
    elif unresolved:
        notes.append(
            f"{unresolved} of their starters {'has' if unresolved == 1 else 'have'} "
            "no projection, so the set-lineup total is a floor."
        )

    if not trust_declared:
        declared_points = projected

    unfilled = len(slots) - len(best.assignment)
    if unfilled > 0:
        notes.append(
            f"{unfilled} starting slot{'s' if unfilled > 1 else ''} cannot be filled "
            "from this roster."
        )

    left_on_bench = max(0.0, projected - declared_points) if trust_declared else 0.0
    # With unresolved starters the gap is partly our ignorance, not their choice.
    if left_on_bench >= 3 and unresolved == 0:
        notes.append(
            f"Leaving {left_on_bench:.1f} projected points on the bench as set."
        )

    strength: dict[str, float] = {}
    if not rows.empty:
        for position in SUMMARY_POSITIONS:
            at_position = rows[rows["position"] == position]
            if at_position.empty:
                continue
            # Count only as many as could realistically start, so a team hoarding
            # six mediocre receivers does not out-rank one with two good ones.
            keep = max(1, int(round(league.starters_needed(position))))
            strength[position] = float(
                at_position["exp_points"].nlargest(keep).sum()
            )

    ros = sum(ros_lookup.get(str(p), 0.0) for p in team.active_players)

    return TeamView(
        roster_id=team.roster_id,
        label=team.label,
        owner=team.display_name,
        record=team.record,
        points_for=team.points_for,
        is_me=team.roster_id == league.my_roster_id,
        projected_points=projected,
        declared_points=declared_points,
        points_left_on_bench=left_on_bench,
        starters=_player_rows(projections, best_ids, slots_by_id),
        declared=_player_rows(projections, declared_ids, declared_slots),
        bench=sorted(
            _player_rows(projections, best.bench),
            key=lambda r: -float(r["projection"] or 0.0),
        ),
        position_strength=strength,
        ros_points=float(ros),
        opponent_roster_id=opponent_id,
        byes=_bye_counts(rows, projections),
        notes=notes,
    )


def league_averages(views: list[TeamView]) -> dict[str, float]:
    """League-wide means, so a single team's numbers have something to sit against."""
    if not views:
        return {}
    projected = np.array([v.projected_points for v in views], dtype=float)
    out = {
        "projected_points": float(projected.mean()),
        "projected_points_median": float(np.median(projected)),
        "ros_points": float(np.mean([v.ros_points for v in views])),
    }
    for position in SUMMARY_POSITIONS:
        values = [v.position_strength.get(position) for v in views]
        values = [v for v in values if v is not None]
        if values:
            out[f"strength_{position}"] = float(np.mean(values))
    return out
