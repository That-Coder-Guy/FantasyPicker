"""Draft engine: what to do with the pick that is on the clock.

Three ideas do the work here.

**Marginal lineup value, not raw points.** A player is worth what he adds to the
best lineup you can field. That single definition handles positional need, flex
and superflex eligibility, and the fact that your fourth tight end is worth
almost nothing — without a table of hand-tuned positional multipliers. It is
computed by solving the lineup assignment with and without the player.

**Replacement level comes from the league, not from convention.** "RB24" is only
the replacement back in a specific league shape. The cutoff here is derived from
the actual roster slots, counting flex spots fractionally across the positions
that can fill them.

**Scarcity is a probability, not a hunch.** Expert-consensus rank and its
standard deviation give each player a distribution over where he goes. From that
comes P(he lasts until my next pick), and from that comes the only question that
matters on the clock: not "who is best?" but "who will not be there later?" —
the difference between the best player available now and the best you can expect
at the same position next time round.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..model.predict import ProjectionSet
from ..sleeper.league import LeagueContext, RosterSlot
from .lineup import solve_assignment

log = logging.getLogger(__name__)

#: A bench player is worth this fraction of his starter value: insurance against
#: injuries and byes, plus trade capital, but he does not score for you.
BENCH_WEIGHT = 0.18
#: How many bench spots are worth hoarding upside in before value flattens out.
_USEFUL_BENCH = 5


def overall_pick_number(pick_type: str, teams: int, round_number: int, slot: int) -> int:
    """Overall pick number for a draft slot in a given round.

    Snake drafts reverse every even round; Sleeper's third-round reversal option
    flips the parity from round three onward, so rounds 3 and 4 both run in the
    same direction. Linear drafts never reverse.
    """
    teams = max(int(teams), 1)
    if pick_type == "linear":
        return (round_number - 1) * teams + slot
    reversed_round = round_number % 2 == 0
    if pick_type == "snake_3rr" and round_number >= 3:
        reversed_round = round_number % 2 == 1
    position = (teams - slot + 1) if reversed_round else slot
    return (round_number - 1) * teams + position


@dataclass
class DraftState:
    """Where a draft has got to, and where my next picks fall."""

    draft_id: str | None
    rounds: int
    teams: int
    pick_type: str  # "snake", "linear", or "snake_3rr"
    my_slot: int | None
    picks_made: int
    drafted: dict[str, int] = field(default_factory=dict)  # sleeper_id -> overall pick
    roster_by_slot: dict[int, list[str]] = field(default_factory=dict)
    on_the_clock_slot: int | None = None

    @property
    def current_pick(self) -> int:
        return self.picks_made + 1

    def overall_pick(self, round_number: int, slot: int) -> int:
        """The overall pick number for a slot in a given round."""
        return overall_pick_number(self.pick_type, self.teams, round_number, slot)

    def my_upcoming_picks(self, limit: int = 4) -> list[int]:
        """Overall pick numbers still ahead of me, soonest first."""
        if self.my_slot is None:
            return []
        picks = [
            self.overall_pick(r, self.my_slot) for r in range(1, self.rounds + 1)
        ]
        return [p for p in sorted(picks) if p >= self.current_pick][:limit]

    @property
    def my_next_pick(self) -> int | None:
        picks = self.my_upcoming_picks(1)
        return picks[0] if picks else None

    @property
    def my_following_pick(self) -> int | None:
        picks = self.my_upcoming_picks(2)
        return picks[1] if len(picks) > 1 else None


@dataclass
class DraftCandidate:
    sleeper_id: str
    name: str
    position: str
    team: str | None
    projected_points: float
    vor: float
    marginal_value: float
    score: float
    ecr: float | None
    adp_sd: float | None
    survival: float
    tier: int
    positional_rank: int
    bye_week: int | None
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "sleeper_id": self.sleeper_id,
            "name": self.name,
            "position": self.position,
            "team": self.team,
            "projected_points": round(self.projected_points, 1),
            "vor": round(self.vor, 1),
            "marginal_value": round(self.marginal_value, 1),
            "score": round(self.score, 2),
            "ecr": None if self.ecr is None else round(self.ecr, 1),
            "adp_sd": None if self.adp_sd is None else round(self.adp_sd, 1),
            "survival": round(self.survival, 3),
            "tier": self.tier,
            "positional_rank": self.positional_rank,
            "bye_week": self.bye_week,
            "reason": self.reason,
        }


@dataclass
class DraftAdvice:
    pick: int | None
    round_number: int | None
    recommendations: list[DraftCandidate]
    best_available: list[DraftCandidate]
    positional_runs: dict[str, float]
    roster_summary: dict[str, int]
    needs: list[str]
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# replacement level and value
# --------------------------------------------------------------------------- #


def replacement_levels(
    league: LeagueContext, board: pd.DataFrame, *, bench_factor: float = 0.5
) -> dict[str, float]:
    """Projected points of the last startable player at each position.

    The index is ``teams × starters_needed(position)``, plus a fraction of a
    bench spot per team, because managers do carry a backup running back and
    that pushes the real replacement level down.
    """
    levels: dict[str, float] = {}
    for position, group in board.groupby("position"):
        needed = league.starters_needed(str(position))
        if needed <= 0:
            # A position nobody starts (kickers in some leagues) has replacement
            # level at the top: the whole pool is interchangeable.
            levels[str(position)] = float(group["projected_points"].max() or 0.0)
            continue
        index = int(round(league.team_count * (needed + bench_factor * _bench_share(position))))
        values = group["projected_points"].sort_values(ascending=False).to_numpy()
        if len(values) == 0:
            levels[str(position)] = 0.0
        else:
            levels[str(position)] = float(values[min(index, len(values) - 1)])
    return levels


def _bench_share(position: str) -> float:
    """Roughly how many bench spots a team spends on each position."""
    return {"RB": 1.4, "WR": 1.4, "TE": 0.4, "QB": 0.4, "K": 0.0, "DST": 0.1}.get(
        str(position).upper(), 0.3
    )


def assign_tiers(values: np.ndarray, *, max_tiers: int = 12) -> np.ndarray:
    """Break a sorted value curve into tiers at its largest gaps.

    Tiers are what actually drive draft decisions — the difference between the
    last player in a tier and the first in the next is a cliff, and everything
    inside a tier is close enough to be a coin flip. Gaps are measured against
    the typical gap in the same list, so a flat position produces few tiers and
    a top-heavy one produces many.
    """
    if values.size == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(-values)
    ordered = values[order]
    gaps = -np.diff(ordered)
    tiers_ordered = np.ones(len(ordered), dtype=int)
    if gaps.size:
        threshold = max(np.mean(gaps) + np.std(gaps), 1e-6)
        breaks = np.where(gaps > threshold)[0]
        if len(breaks) > max_tiers - 1:
            breaks = breaks[np.argsort(-gaps[breaks])[: max_tiers - 1]]
            breaks.sort()
        tier = 1
        cursor = 0
        for boundary in breaks:
            tiers_ordered[cursor : boundary + 1] = tier
            cursor = boundary + 1
            tier += 1
        tiers_ordered[cursor:] = tier
    out = np.empty_like(tiers_ordered)
    out[order] = tiers_ordered
    return out


def survival_probability(ecr: float | None, sd: float | None, pick: int) -> float:
    """P(a player is still on the board at overall pick ``pick``).

    Draft position is modelled as Normal(consensus rank, consensus spread) —
    crude for a discrete ordering, but the consensus spread is a real,
    published measure of disagreement, and disagreement is exactly what makes a
    player fall.
    """
    if ecr is None or not math.isfinite(ecr):
        return 0.5
    spread = max(float(sd or 0.0), 1.0)
    z = (pick - float(ecr)) / spread
    # P(drafted before this pick) = Φ(z); survival is the complement.
    return float(min(max(0.5 * math.erfc(z / math.sqrt(2.0)), 0.0), 1.0))


def expected_best_available(
    group: pd.DataFrame, pick: int, *, value_column: str = "vor", exclude: str | None = None
) -> float:
    """E[value of the best player left at this position at ``pick``].

    Walking the position's board from the top: the best remaining player is the
    first one who survives, so his contribution is his value times the chance
    everyone above him is gone and he is not.
    """
    if group.empty:
        return 0.0
    rows = group.sort_values(value_column, ascending=False)
    if exclude is not None:
        rows = rows[rows["sleeper_id"] != exclude]
    expected = 0.0
    gone = 1.0
    for row in rows.itertuples(index=False):
        survives = survival_probability(
            getattr(row, "ecr", None), getattr(row, "adp_sd", None), pick
        )
        expected += gone * survives * float(getattr(row, value_column))
        gone *= 1 - survives
        if gone < 1e-4:
            break
    return float(expected)


# --------------------------------------------------------------------------- #
# roster value
# --------------------------------------------------------------------------- #


class RosterValuer:
    """Values a roster by the lineup it can field, plus a discount for depth.

    Every unfilled starting slot is backfilled with a replacement-level player —
    the waiver-wire body you would actually be forced to start. Without that, an
    empty roster is worth nothing and the first pick's marginal value is simply
    his raw projection, which ranks a tight end who scores 278 above a running
    back who scores 271 while ignoring that the tight end you could have had for
    free scores 109 and the running back you could have had for free scores 93.
    Backfilling makes marginal value equal value-over-replacement at the start of
    a draft, and turn into value-over-*your-own-starter* as the roster fills —
    which is exactly how the number should behave.
    """

    def __init__(self, slots: list[RosterSlot], board: pd.DataFrame) -> None:
        self.slots = slots
        self._points = dict(zip(board["sleeper_id"], board["projected_points"]))
        self._positions = dict(zip(board["sleeper_id"], board["position"]))
        self._replacement = dict(zip(board["sleeper_id"], board["replacement"]))
        self._levels: dict[str, float] = {}
        for position, group in board.groupby("position"):
            self._levels[str(position)] = float(group["replacement"].iloc[0])
        self._filler_ids, self._filler_positions, self._filler_values = self._fillers()
        self._cache: dict[frozenset[str], float] = {}

    def _fillers(self) -> tuple[list[str], list[str], list[float]]:
        """One replacement-level body per position per slot it could fill."""
        ids: list[str] = []
        positions: list[str] = []
        values: list[float] = []
        for index, slot in enumerate(self.slots):
            for position in sorted(slot.eligible):
                ids.append(f"__replacement_{position}_{index}")
                positions.append(position)
                values.append(self._levels.get(position, 0.0))
        return ids, positions, values

    def value(self, roster: list[str]) -> float:
        ids = sorted({p for p in roster if p in self._points})
        key = frozenset(ids)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        positions = [self._positions[p] for p in ids]
        values = [self._points[p] for p in ids]

        all_ids = ids + self._filler_ids
        all_positions = positions + self._filler_positions
        all_values = np.array(values + self._filler_values, dtype=float)
        solution = solve_assignment(self.slots, all_ids, all_positions, all_values)
        total = solution.objective

        # Bench: only real players who beat replacement level are worth anything,
        # and only the first few of those.
        bench_surplus = sorted(
            (
                max(0.0, self._points[p] - self._replacement.get(p, 0.0))
                for p in solution.bench
                if p in self._points
            ),
            reverse=True,
        )
        total += BENCH_WEIGHT * float(sum(bench_surplus[:_USEFUL_BENCH]))
        self._cache[key] = total
        return total

    def marginal(self, roster: list[str], candidate: str) -> float:
        if candidate not in self._points:
            return 0.0
        return self.value(list(roster) + [candidate]) - self.value(roster)

    def marginal_hypothetical(
        self, roster: list[str], position: str, points: float
    ) -> float:
        """What a *hypothetical* player at ``position`` scoring ``points`` would add.

        Used for the second ply: "if I take this running back now, how much is
        the best receiver I can expect next time round worth to the roster I
        will have by then?" That question is about a player who does not exist
        yet, so he is invented and valued like any other.
        """
        synthetic = f"__hypothetical_{position}"
        self._points[synthetic] = float(points)
        self._positions[synthetic] = str(position).upper()
        self._replacement[synthetic] = self._levels.get(str(position).upper(), 0.0)
        try:
            return self.value(list(roster) + [synthetic]) - self.value(roster)
        finally:
            for store in (self._points, self._positions, self._replacement):
                store.pop(synthetic, None)
            # Any cached value that included the synthetic player is now stale.
            self._cache = {
                key: value for key, value in self._cache.items() if synthetic not in key
            }


# --------------------------------------------------------------------------- #
# the board
# --------------------------------------------------------------------------- #


def build_board(
    league: LeagueContext,
    projections: ProjectionSet,
    ranks: pd.DataFrame,
    *,
    drafted: set[str] | None = None,
) -> pd.DataFrame:
    """Join projections with consensus ranks into one draftable board."""
    if projections.frame.empty:
        return pd.DataFrame()
    board = projections.frame.rename(columns={"exp_points": "projected_points"}).copy()
    board["sleeper_id"] = board["sleeper_id"].astype(str)

    if ranks is not None and not ranks.empty:
        market = ranks[["sleeper_id", "ecr", "sd", "bye", "overall_rank"]].copy()
        market["sleeper_id"] = market["sleeper_id"].astype(str)
        market = market.rename(columns={"sd": "adp_sd", "bye": "bye_week"})
        board = board.merge(market, on="sleeper_id", how="left")
    else:
        board["ecr"] = np.nan
        board["adp_sd"] = np.nan
        board["bye_week"] = np.nan
        board["overall_rank"] = np.nan

    # Anyone the market has not ranked is undraftable in practice; keep them on
    # the board but at the back, where a deep-league pick can still find them.
    board["ecr"] = board["ecr"].fillna(board["projected_points"].rank(ascending=False) + 250)
    board["adp_sd"] = board["adp_sd"].fillna(board["ecr"] * 0.3)

    levels = replacement_levels(league, board)
    board["replacement"] = board["position"].map(levels).fillna(0.0)
    board["vor"] = board["projected_points"] - board["replacement"]

    board["positional_rank"] = (
        board.groupby("position")["projected_points"].rank(ascending=False, method="min").astype(int)
    )
    board["tier"] = 0
    for position, group in board.groupby("position"):
        board.loc[group.index, "tier"] = assign_tiers(
            group["projected_points"].to_numpy(dtype=float)
        )

    if drafted:
        board["drafted"] = board["sleeper_id"].isin(drafted)
    else:
        board["drafted"] = False
    return board.sort_values("vor", ascending=False).reset_index(drop=True)


def _needs(league: LeagueContext, roster: list[str], positions: list[str]) -> list[str]:
    """Starting slots the current roster still cannot fill.

    Solved as an assignment rather than counted by position, because "I have
    three receivers" only answers the question once you know whether one of them
    is already needed in the flex.
    """
    if not league.slots:
        return []
    ids = list(roster)
    if ids:
        solution = solve_assignment(
            league.slots, ids, positions, np.ones(len(ids), dtype=float)
        )
        filled = set(solution.assignment)
    else:
        filled = set()
    needs: list[str] = []
    seen: set[str] = set()
    for index, slot in enumerate(league.slots):
        if index in filled:
            continue
        label = slot.name
        if label not in seen:
            seen.add(label)
            needs.append(label)
    return needs


def recommend(
    league: LeagueContext,
    board: pd.DataFrame,
    state: DraftState,
    my_roster: list[str],
    *,
    top_n: int = 8,
    pool_size: int = 60,
) -> DraftAdvice:
    """Rank the players worth taking with the pick that is on the clock."""
    notes: list[str] = []
    if board.empty:
        return DraftAdvice(None, None, [], [], {}, {}, [], ["No board available."])

    available = board[~board["drafted"]].copy()
    available = available[~available["sleeper_id"].isin(set(my_roster))]
    if available.empty:
        return DraftAdvice(None, None, [], [], {}, {}, [], ["Every ranked player is gone."])

    pick = state.my_next_pick or state.current_pick
    next_pick = state.my_following_pick
    round_number = ((pick - 1) // max(state.teams, 1)) + 1 if pick else None

    valuer = RosterValuer(league.slots, board)
    position_of = dict(zip(board["sleeper_id"].astype(str), board["position"].astype(str)))
    known_roster = [str(p) for p in my_roster if str(p) in position_of]
    base_positions = [position_of[p] for p in known_roster]

    # How far each position falls off before my next turn. This is the scarcity
    # signal: a large drop means take that position now.
    runs: dict[str, float] = {}
    expected_points: dict[str, float] = {}
    groups = {str(position): group for position, group in available.groupby("position")}
    if next_pick:
        for position, group in groups.items():
            best_now = float(group["vor"].max())
            expected_later = expected_best_available(group, next_pick)
            expected_points[position] = expected_best_available(
                group, next_pick, value_column="projected_points"
            )
            runs[position] = best_now - expected_later

    candidates_frame = available.nlargest(pool_size, "vor")
    candidates: list[DraftCandidate] = []
    for row in candidates_frame.itertuples(index=False):
        marginal = valuer.marginal(known_roster, row.sleeper_id)
        survival = survival_probability(
            float(row.ecr) if pd.notna(row.ecr) else None,
            float(row.adp_sd) if pd.notna(row.adp_sd) else None,
            next_pick or (pick + league.team_count),
        )

        # Second ply. Taking this player changes two things about my next pick:
        # he is no longer available at his own position, and my roster now has
        # one more body there — so the best receiver left is worth more to me if
        # I just took a running back than if I just took a receiver. Valuing the
        # expected best-available at each position *against the roster I would
        # then have* is what makes the recommendation account for positional
        # timing rather than just ranking by value over replacement.
        lookahead = 0.0
        if next_pick and expected_points:
            after_roster = list(known_roster) + [str(row.sleeper_id)]
            for position, points in expected_points.items():
                if position == row.position:
                    group = groups.get(position)
                    points = (
                        expected_best_available(
                            group,
                            next_pick,
                            value_column="projected_points",
                            exclude=str(row.sleeper_id),
                        )
                        if group is not None
                        else points
                    )
                lookahead = max(
                    lookahead,
                    valuer.marginal_hypothetical(after_roster, position, points),
                )

        score = marginal + lookahead
        candidates.append(
            DraftCandidate(
                sleeper_id=str(row.sleeper_id),
                name=str(row.name),
                position=str(row.position),
                team=str(row.team) if pd.notna(row.team) else None,
                projected_points=float(row.projected_points),
                vor=float(row.vor),
                marginal_value=float(marginal),
                score=float(score),
                ecr=float(row.ecr) if pd.notna(row.ecr) else None,
                adp_sd=float(row.adp_sd) if pd.notna(row.adp_sd) else None,
                survival=survival,
                tier=int(row.tier),
                positional_rank=int(row.positional_rank),
                bye_week=int(row.bye_week) if pd.notna(row.bye_week) else None,
            )
        )

    candidates.sort(key=lambda c: -c.score)
    for candidate in candidates[:top_n]:
        candidate.reason = _explain(candidate, runs, next_pick)

    best_available = sorted(candidates, key=lambda c: -c.vor)[:top_n]
    roster_summary: dict[str, int] = {}
    for position in base_positions:
        roster_summary[position] = roster_summary.get(position, 0) + 1

    if next_pick:
        notes.append(
            f"Pick {pick}; your next pick is {next_pick} "
            f"({next_pick - pick - 1} picks in between)."
        )
    if runs:
        hottest = max(runs, key=runs.get)
        if runs[hottest] > 5:
            notes.append(
                f"{hottest} falls off hardest before your next pick "
                f"({runs[hottest]:.0f} points of value)."
            )

    return DraftAdvice(
        pick=pick,
        round_number=round_number,
        recommendations=candidates[:top_n],
        best_available=best_available,
        positional_runs={k: round(v, 1) for k, v in sorted(runs.items(), key=lambda kv: -kv[1])},
        roster_summary=roster_summary,
        needs=_needs(league, known_roster, base_positions),
        notes=notes,
    )


def _explain(candidate: DraftCandidate, runs: dict[str, float], next_pick: int | None) -> str:
    bits = [
        f"{candidate.position}{candidate.positional_rank}, tier {candidate.tier}",
        f"{candidate.projected_points:.0f} projected points "
        f"({candidate.vor:+.0f} over replacement)",
    ]
    if next_pick is not None:
        if candidate.survival < 0.25:
            bits.append(f"only {candidate.survival:.0%} to last until pick {next_pick}")
        elif candidate.survival > 0.6:
            bits.append(f"{candidate.survival:.0%} likely to still be there at {next_pick}")
    drop = runs.get(candidate.position)
    if drop is not None and drop > 5:
        bits.append(f"{candidate.position} drops {drop:.0f} points before your next turn")
    return "; ".join(bits)


# --------------------------------------------------------------------------- #
# live draft state
# --------------------------------------------------------------------------- #


def parse_draft_state(
    draft: dict | None, picks: list[dict], *, my_user_id: str | None = None
) -> DraftState:
    """Build a :class:`DraftState` from Sleeper's draft object and pick feed."""
    draft = draft or {}
    settings = draft.get("settings") or {}
    teams = int(settings.get("teams") or 12)
    rounds = int(settings.get("rounds") or 15)

    kind = str(draft.get("type") or "snake").lower()
    if kind == "snake" and int(settings.get("reversal_round") or 0) == 3:
        kind = "snake_3rr"

    drafted: dict[str, int] = {}
    roster_by_slot: dict[int, list[str]] = {}
    for pick in picks:
        player_id = str(pick.get("player_id") or "")
        if not player_id:
            continue
        drafted[player_id] = int(pick.get("pick_no") or len(drafted) + 1)
        slot = int(pick.get("draft_slot") or 0)
        roster_by_slot.setdefault(slot, []).append(player_id)

    my_slot: int | None = None
    order = draft.get("draft_order") or {}
    if my_user_id and isinstance(order, dict):
        raw = order.get(str(my_user_id))
        if raw is not None:
            my_slot = int(raw)

    picks_made = len(drafted)
    current = picks_made + 1
    current_round = ((current - 1) // teams) + 1
    on_the_clock = next(
        (
            slot
            for slot in range(1, teams + 1)
            if overall_pick_number(kind, teams, current_round, slot) == current
        ),
        None,
    )

    return DraftState(
        draft_id=draft.get("draft_id"),
        rounds=rounds,
        teams=teams,
        pick_type=kind,
        my_slot=my_slot,
        picks_made=picks_made,
        drafted=drafted,
        roster_by_slot=roster_by_slot,
        on_the_clock_slot=on_the_clock,
    )
