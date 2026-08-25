"""Weekly matchup analysis: win probability, start/sit, and strategy.

Everything here follows from one simulation. Both rosters are sampled *jointly*,
so a defense facing one of your receivers pulls against him in every simulated
week, and a stacked quarterback-receiver pair rises together. Splitting the
sample afterwards gives two correlated point totals, and the whole matchup —
win probability, the value of a swap, whether to chase variance — is read off
those.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model.predict import ProjectionSet
from ..sleeper.league import LeagueContext, RosterSlot, Team
from .correlations import CorrelationModel
from .lineup import (
    LineupSolution,
    optimize_points,
    optimize_win_probability,
    swap_impacts,
)
from .simulate import Simulator, summarize

#: Below/above these win probabilities, chasing or suppressing variance is worth
#: more than chasing expected points.
UNDERDOG_THRESHOLD = 0.42
FAVOURITE_THRESHOLD = 0.62


@dataclass
class MatchupAnalysis:
    week: int
    my_team: str
    opponent_team: str | None
    win_probability: float
    my_lineup: LineupSolution
    opponent_lineup: LineupSolution | None
    my_distribution: dict[str, float]
    opponent_distribution: dict[str, float]
    margin_distribution: dict[str, float]
    swaps: list[dict[str, object]] = field(default_factory=list)
    leverage_lineup: LineupSolution | None = None
    leverage_gain: float = 0.0
    strategy: str = ""
    player_rows: list[dict[str, object]] = field(default_factory=list)
    opponent_rows: list[dict[str, object]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _roster_inputs(
    projections: ProjectionSet, player_ids: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Filter to players we have a projection for, keeping ids/positions aligned."""
    rows = projections.subset([str(p) for p in player_ids])
    if rows.empty:
        return [], [], []
    return (
        rows["sleeper_id"].astype(str).tolist(),
        rows["position"].astype(str).tolist(),
        rows["name"].astype(str).tolist(),
    )


def _describe_strategy(win_probability: float, gain: float) -> str:
    if win_probability < UNDERDOG_THRESHOLD:
        base = (
            "You are the underdog. The lineup that scores the most on average is "
            "not the one most likely to win — you need outcomes in the top of your "
            "range, so higher-variance starters are worth more than their averages."
        )
    elif win_probability > FAVOURITE_THRESHOLD:
        base = (
            "You are the favourite. Protect the lead: prefer the steadier player "
            "when two projections are close, because a bust costs you more than a "
            "boom gains you."
        )
    else:
        base = (
            "This is a close matchup, so maximising expected points is also the "
            "best way to maximise win probability."
        )
    if gain > 0.005:
        base += f" Switching to the win-probability lineup is worth {gain:+.1%}."
    return base


def analyze_matchup(
    league: LeagueContext,
    projections: ProjectionSet,
    *,
    week: int,
    my_team: Team,
    opponent: Team | None,
    my_starters: list[str] | None = None,
    opponent_starters: list[str] | None = None,
    correlations: CorrelationModel | None = None,
    n_sims: int = 20_000,
    seed: int | None = None,
    opponent_mode: str = "auto",
) -> MatchupAnalysis:
    """Simulate the week and return everything the matchup view needs.

    ``opponent_mode`` decides which opponent lineup to play against:
    ``"declared"`` uses whatever they have set right now, ``"optimal"`` assumes
    they will fix it before kickoff, and ``"auto"`` (the default) uses their
    declared lineup when they have set one and their optimal lineup otherwise —
    which is the safer assumption early in the week.
    """
    slots: list[RosterSlot] = league.slots
    my_ids, my_positions, my_names = _roster_inputs(projections, my_team.active_players)
    opponent_ids: list[str] = []
    opponent_positions: list[str] = []
    if opponent is not None:
        opponent_ids, opponent_positions, _ = _roster_inputs(
            projections, opponent.active_players
        )

    everyone = my_ids + [p for p in opponent_ids if p not in my_ids]
    simulator = Simulator(projections, correlations, n_sims=n_sims, seed=seed)
    sampled = simulator.sample(everyone)
    if not sampled.ids:
        raise ValueError("no projections available for either roster this week")

    column_of = {pid: i for i, pid in enumerate(sampled.ids)}
    my_ids, my_positions = _align(my_ids, my_positions, column_of)
    my_samples = sampled.scores[:, [column_of[p] for p in my_ids]]

    notes: list[str] = []

    # -- opponent ---------------------------------------------------------- #
    opponent_ids, opponent_positions = _align(opponent_ids, opponent_positions, column_of)
    if opponent_ids:
        opponent_samples = sampled.scores[:, [column_of[p] for p in opponent_ids]]
        declared = [p for p in (opponent_starters or []) if p in column_of]
        use_declared = opponent_mode == "declared" or (
            opponent_mode == "auto" and len(declared) >= max(1, len(slots) - 1)
        )
        if use_declared:
            opponent_lineup = _lineup_from_starters(slots, declared, projections)
            notes.append("Opponent modelled with the lineup they have currently set.")
        else:
            opponent_lineup = optimize_points(
                slots,
                opponent_ids,
                opponent_positions,
                opponent_samples.mean(axis=0),
            )
            notes.append(
                "Opponent modelled with their best possible lineup — the "
                "conservative assumption before lineups lock."
            )
        opponent_cols = [
            column_of[p] for p in opponent_lineup.starters if p in column_of
        ]
        opponent_totals = (
            sampled.scores[:, opponent_cols].sum(axis=1)
            if opponent_cols
            else np.zeros(n_sims)
        )
    else:
        opponent_lineup = None
        opponent_totals = np.zeros(n_sims)
        notes.append("No opponent found for this week (bye or playoff structure).")

    # -- my lineup ---------------------------------------------------------- #
    points_lineup = optimize_points(
        slots, my_ids, my_positions, my_samples.mean(axis=0)
    )
    if my_starters:
        current = [p for p in my_starters if p in column_of]
        active_lineup = (
            _lineup_from_starters(slots, current, projections)
            if len(current) >= max(1, len(slots) - 2)
            else points_lineup
        )
    else:
        active_lineup = points_lineup

    def win_probability(lineup: LineupSolution) -> tuple[float, np.ndarray]:
        cols = [column_of[p] for p in lineup.starters if p in column_of]
        totals = sampled.scores[:, cols].sum(axis=1) if cols else np.zeros(n_sims)
        prob = float(
            np.mean(totals > opponent_totals) + 0.5 * np.mean(totals == opponent_totals)
        )
        return prob, totals

    base_prob, my_totals = win_probability(points_lineup)

    leverage_lineup = optimize_win_probability(
        slots,
        my_ids,
        my_positions,
        my_samples,
        opponent_totals,
        start=points_lineup,
    )
    leverage_prob, _ = win_probability(leverage_lineup)
    gain = leverage_prob - base_prob
    if gain <= 0.002:
        leverage_lineup = None
        gain = 0.0

    swaps = swap_impacts(
        slots,
        my_ids,
        my_positions,
        my_samples,
        opponent_totals,
        active_lineup,
    )

    margin = my_totals - opponent_totals
    rows = _player_rows(projections, my_ids, points_lineup, active_lineup, my_samples)
    opponent_rows = (
        _player_rows(
            projections,
            opponent_ids,
            opponent_lineup or LineupSolution({}, slots, [], 0.0),
            opponent_lineup or LineupSolution({}, slots, [], 0.0),
            opponent_samples,
        )
        if opponent_ids
        else []
    )

    return MatchupAnalysis(
        week=week,
        my_team=my_team.label,
        opponent_team=opponent.label if opponent else None,
        win_probability=base_prob,
        my_lineup=points_lineup,
        opponent_lineup=opponent_lineup,
        my_distribution=summarize(my_totals),
        opponent_distribution=summarize(opponent_totals),
        margin_distribution=summarize(margin),
        swaps=swaps,
        leverage_lineup=leverage_lineup,
        leverage_gain=gain,
        strategy=_describe_strategy(base_prob, gain),
        player_rows=rows,
        opponent_rows=opponent_rows,
        notes=notes,
    )


def _align(
    ids: list[str], positions: list[str], column_of: dict[str, int]
) -> tuple[list[str], list[str]]:
    """Drop ids with no simulated column, keeping positions in step."""
    kept_ids, kept_positions = [], []
    for pid, pos in zip(ids, positions):
        if pid in column_of:
            kept_ids.append(pid)
            kept_positions.append(pos)
    return kept_ids, kept_positions


def _lineup_from_starters(
    slots: list[RosterSlot], starters: list[str], projections: ProjectionSet
) -> LineupSolution:
    """Fit a declared starter list into the league's slots.

    Sleeper reports starters positionally, but a roster edit can leave the list
    out of order, so the players are re-matched to slots rather than trusted to
    line up index by index.
    """
    rows = projections.subset(starters)
    if rows.empty:
        return LineupSolution({}, slots, [], 0.0, "declared")
    ids = rows["sleeper_id"].astype(str).tolist()
    positions = rows["position"].astype(str).tolist()
    values = rows["exp_points"].to_numpy(dtype=float)
    from .lineup import solve_assignment  # local import avoids a cycle at module load

    solution = solve_assignment(slots, ids, positions, values)
    solution.objective_name = "declared"
    return solution


def _player_rows(
    projections: ProjectionSet,
    ids: list[str],
    optimal: LineupSolution,
    current: LineupSolution,
    samples: np.ndarray,
) -> list[dict[str, object]]:
    optimal_slots = {pid: optimal.slot_name(i) for i, pid in optimal.assignment.items()}
    current_slots = {pid: current.slot_name(i) for i, pid in current.assignment.items()}
    rows = projections.subset(ids).set_index("sleeper_id")
    out: list[dict[str, object]] = []
    for column, pid in enumerate(ids):
        if pid not in rows.index:
            continue
        row = rows.loc[pid]
        draws = samples[:, column]
        out.append(
            {
                "sleeper_id": pid,
                "name": row.get("name"),
                "position": row.get("position"),
                "team": row.get("team"),
                "opponent": row.get("opponent"),
                "projection": float(row.get("proj_mean", 0.0)),
                "floor": float(row.get("floor", 0.0)),
                "ceiling": float(row.get("ceiling", 0.0)),
                "p_play": float(row.get("p_play", 1.0)),
                "sim_mean": float(np.mean(draws)),
                "sim_std": float(np.std(draws)),
                "optimal_slot": optimal_slots.get(pid),
                "current_slot": current_slots.get(pid),
            }
        )
    out.sort(key=lambda r: -float(r["projection"]))
    return out
