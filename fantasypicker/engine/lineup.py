"""Lineup construction.

Two different optimal lineups exist, and which one you want depends on the
matchup:

* **Most expected points.** The right answer in a vacuum, and the right answer
  when the matchup is close. Solved exactly — filling slots from a roster is an
  assignment problem, so a greedy "best player at each position" pass can be
  beaten, and maximum-weight bipartite matching finds the true optimum.

* **Most likely to win.** The right answer when you are a heavy underdog or a
  heavy favourite. A 12-point underdog does not want the lineup that scores the
  most on average; they want the one with the fattest right tail, because only
  the top of their range wins the week. A favourite wants the opposite. This one
  has no closed form, so it is hill-climbed over simulated outcomes from the
  expected-points optimum.

The gap between the two is usually zero, occasionally one swap, and the app says
which case you are in rather than quietly picking for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..sleeper.league import RosterSlot

#: Large negative value standing in for "this player cannot fill this slot".
_INELIGIBLE = -1e6


@dataclass
class LineupSolution:
    """A filled starting lineup plus what was left on the bench."""

    #: slot index -> sleeper_id (missing key means the slot could not be filled)
    assignment: dict[int, str]
    slots: list[RosterSlot]
    bench: list[str]
    objective: float
    objective_name: str = "expected points"
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def starters(self) -> list[str]:
        return [self.assignment[i] for i in sorted(self.assignment)]

    def slot_name(self, index: int) -> str:
        return self.slots[index].name if 0 <= index < len(self.slots) else "?"

    def as_rows(self) -> list[dict[str, str]]:
        return [
            {"slot": self.slots[i].name, "sleeper_id": self.assignment[i]}
            for i in sorted(self.assignment)
        ]


def _eligibility_matrix(
    slots: list[RosterSlot], positions: list[str]
) -> np.ndarray:
    """``(n_slots, n_players)`` boolean: can this player fill this slot?"""
    return np.array(
        [[slot.accepts(position) for position in positions] for slot in slots],
        dtype=bool,
    )


def solve_assignment(
    slots: list[RosterSlot],
    player_ids: list[str],
    positions: list[str],
    values: np.ndarray,
) -> LineupSolution:
    """Fill every slot to maximise total ``values`` — exactly.

    Greedy filling fails on cases like: a roster with one elite tight end, a
    TE slot and a FLEX. Greedy puts the tight end in FLEX if he is the best flex
    option and then has nothing for TE. Matching gets it right.
    """
    if not slots or not player_ids:
        return LineupSolution({}, slots, list(player_ids), 0.0)

    eligible = _eligibility_matrix(slots, positions)
    payoff = np.where(eligible, values[None, :], _INELIGIBLE)

    n_slots, n_players = payoff.shape
    if n_players < n_slots:  # pad with unfillable dummy players
        pad = np.full((n_slots, n_slots - n_players), _INELIGIBLE)
        payoff = np.hstack([payoff, pad])

    rows, cols = linear_sum_assignment(-payoff)
    assignment: dict[int, str] = {}
    used: set[str] = set()
    total = 0.0
    for slot_index, player_index in zip(rows, cols):
        if player_index >= n_players:
            continue
        if payoff[slot_index, player_index] <= _INELIGIBLE / 2:
            continue
        player = player_ids[player_index]
        assignment[int(slot_index)] = player
        used.add(player)
        total += float(values[player_index])

    bench = [p for p in player_ids if p not in used]
    return LineupSolution(assignment, slots, bench, total)


def optimize_points(
    slots: list[RosterSlot],
    player_ids: list[str],
    positions: list[str],
    expected_points: np.ndarray,
) -> LineupSolution:
    """The maximum-expected-points lineup."""
    solution = solve_assignment(slots, player_ids, positions, expected_points)
    solution.objective_name = "expected points"
    return solution


def optimize_win_probability(
    slots: list[RosterSlot],
    player_ids: list[str],
    positions: list[str],
    samples: np.ndarray,
    opponent_totals: np.ndarray,
    *,
    start: LineupSolution | None = None,
    max_passes: int = 6,
) -> LineupSolution:
    """The lineup most likely to outscore this specific opponent.

    Hill-climbs single swaps from the expected-points optimum. Each candidate
    lineup is scored by simulating the week, so the objective already accounts
    for correlation, availability, and the shape of every player's range. Both
    directions are searched: swapping a bench player in, and moving a starter to
    a different slot to make room.

    ``samples`` is ``(n_sims, n_players)`` aligned to ``player_ids``.
    """
    if samples.size == 0 or not slots:
        return start or LineupSolution({}, slots, list(player_ids), 0.0, "win probability")

    index_of = {pid: i for i, pid in enumerate(player_ids)}
    eligible = _eligibility_matrix(slots, positions)

    if start is None:
        start = optimize_points(slots, player_ids, positions, samples.mean(axis=0))
    assignment = dict(start.assignment)

    def win_probability(current: dict[int, str]) -> float:
        cols = [index_of[p] for p in current.values() if p in index_of]
        if not cols:
            return 0.0
        totals = samples[:, cols].sum(axis=1)
        return float(np.mean(totals > opponent_totals) + 0.5 * np.mean(totals == opponent_totals))

    best = win_probability(assignment)

    for _ in range(max_passes):
        improved = False
        started = set(assignment.values())
        bench = [p for p in player_ids if p not in started]

        for slot_index in list(assignment):
            for candidate in list(bench):
                if candidate not in index_of:
                    continue
                if not eligible[slot_index, index_of[candidate]]:
                    continue
                trial = dict(assignment)
                trial[slot_index] = candidate
                score = win_probability(trial)
                if score > best + 1e-6:
                    best = score
                    assignment = trial
                    improved = True
                    started = set(assignment.values())
                    bench = [p for p in player_ids if p not in started]

        # Swapping two starters between slots can unlock an otherwise illegal
        # bench move (a TE sitting in FLEX blocking a better TE, for instance).
        slot_indices = list(assignment)
        for i, a in enumerate(slot_indices):
            for b in slot_indices[i + 1 :]:
                pa, pb = assignment[a], assignment[b]
                if not (
                    eligible[a, index_of[pb]] and eligible[b, index_of[pa]]
                ):
                    continue
                trial = dict(assignment)
                trial[a], trial[b] = pb, pa
                score = win_probability(trial)
                if score > best + 1e-6:
                    best, assignment, improved = score, trial, True

        if not improved:
            break

    used = set(assignment.values())
    solution = LineupSolution(
        assignment=assignment,
        slots=slots,
        bench=[p for p in player_ids if p not in used],
        objective=best,
        objective_name="win probability",
    )
    return solution


def swap_impacts(
    slots: list[RosterSlot],
    player_ids: list[str],
    positions: list[str],
    samples: np.ndarray,
    opponent_totals: np.ndarray,
    lineup: LineupSolution,
    *,
    limit: int = 12,
) -> list[dict[str, object]]:
    """Every legal single swap, ranked by how much it moves win probability.

    This is the honest version of a start/sit recommendation: not "player A is
    better than player B" in the abstract, but "against this opponent, this
    swap is worth 3.4 percentage points of win probability".
    """
    if samples.size == 0 or not lineup.assignment:
        return []

    index_of = {pid: i for i, pid in enumerate(player_ids)}
    eligible = _eligibility_matrix(slots, positions)

    def evaluate(current: dict[int, str]) -> tuple[float, float]:
        cols = [index_of[p] for p in current.values() if p in index_of]
        totals = samples[:, cols].sum(axis=1)
        win = float(np.mean(totals > opponent_totals) + 0.5 * np.mean(totals == opponent_totals))
        return win, float(totals.mean())

    base_win, base_points = evaluate(lineup.assignment)
    started = set(lineup.assignment.values())
    bench = [p for p in player_ids if p not in started]

    results: list[dict[str, object]] = []
    for slot_index, current_player in lineup.assignment.items():
        for candidate in bench:
            if candidate not in index_of or not eligible[slot_index, index_of[candidate]]:
                continue
            trial = dict(lineup.assignment)
            trial[slot_index] = candidate
            win, points = evaluate(trial)
            results.append(
                {
                    "slot": slots[slot_index].name,
                    "slot_index": slot_index,
                    "out": current_player,
                    "in": candidate,
                    "win_prob_delta": win - base_win,
                    "points_delta": points - base_points,
                }
            )

    results.sort(key=lambda r: -float(r["win_prob_delta"]))
    return results[:limit]
