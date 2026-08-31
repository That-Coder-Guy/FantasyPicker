"""Who on your roster is replaceable by someone still unowned.

The waiver page asks "who is worth adding" and pairs every candidate with the
same drop — your single worst player. That is the wrong pairing, and it is
wrong in a way that matters: adding a tight end when you already start one
should cost you a tight end, not your worst running back. The right drop
depends entirely on who you are adding.

So this asks the question from the other end. For every player you roster, is
there anyone in the open pool whose arrival — *with that player gone* — leaves
your best possible lineup better off? Both halves are solved exactly, with the
same assignment solver and the same roster valuation the trade engine uses, so
the two pages can never disagree about what a roster is worth.

Two kinds of answer come out, and they are different advice:

* **Upgrades** — a specific swap that gains points. Do this one.
* **Dead weight** — players who cost nothing to cut because they never reach
  your lineup, even though nobody in the pool beats them. These are who to drop
  when you need a roster spot for a bye-week fill or a waiver claim, and
  knowing them in advance is the difference between a considered move and a
  panicked one at 11pm on Saturday.

Values are rest-of-season, because a drop is permanent. A player you cut for
this week's matchup is gone for the season, so the horizon has to match the
consequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..model.predict import ProjectionSet
from ..sleeper.league import LeagueContext
from .trades import RosterEvaluator

log = logging.getLogger(__name__)

#: Below this the swap is inside the model's own error bars and not worth a
#: transaction — the projections carry several points of uncertainty, so a
#: fractional edge is noise dressed as advice.
MIN_UPGRADE = 1.0

#: What dropping a player may cost before he stops being "free to cut". Bench
#: depth is valued at a fraction of face (see BENCH_WEIGHT), so this catches
#: players who neither start nor rank as real insurance — a genuine handcuff
#: behind a starter costs meaningfully more than this and is not offered up.
DEAD_WEIGHT = 3.0

#: How many free agents to consider. The pool is sorted by projection, so this
#: reaches far past anyone who could plausibly start.
POOL_SIZE = 60


@dataclass
class DropCandidate:
    """One rostered player, and the best the open pool offers in his place."""

    drop: str
    #: What losing him alone would cost the lineup. Near zero means he is
    #: buried behind better players and never plays.
    cost: float
    add: str | None = None
    #: Rest-of-season points the swap gains, knock-on lineup changes included.
    gain: float = 0.0
    #: Rostered players at his position who are ahead of him.
    blocked_by: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def is_upgrade(self) -> bool:
        return self.add is not None and self.gain >= MIN_UPGRADE

    def as_dict(self) -> dict[str, object]:
        return {
            "drop": self.drop,
            "cost": round(self.cost, 1),
            "add": self.add,
            "gain": round(self.gain, 1),
            "blocked_by": list(self.blocked_by),
            "reason": self.reason,
            "is_upgrade": self.is_upgrade,
        }


@dataclass
class DropReport:
    upgrades: list[DropCandidate] = field(default_factory=list)
    dead_weight: list[DropCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "upgrades": [u.as_dict() for u in self.upgrades],
            "dead_weight": [d.as_dict() for d in self.dead_weight],
            "notes": list(self.notes),
        }


def _blockers(
    evaluator: RosterEvaluator, roster: frozenset, player: str
) -> list[str]:
    """Rostered players at the same position projected above this one."""
    position = evaluator.position.get(player)
    if not position:
        return []
    mine = evaluator.value.get(player, 0.0)
    ahead = [
        p
        for p in roster
        if p != player
        and evaluator.position.get(p) == position
        and evaluator.value.get(p, 0.0) > mine
    ]
    return sorted(ahead, key=lambda p: -evaluator.value.get(p, 0.0))


def _reason(
    evaluator: RosterEvaluator,
    candidate: DropCandidate,
    names: dict[str, str],
) -> str:
    position = evaluator.position.get(candidate.drop, "?")
    who = names.get(candidate.drop, candidate.drop)
    if candidate.is_upgrade:
        add_name = names.get(candidate.add or "", candidate.add or "")
        add_position = evaluator.position.get(candidate.add or "", "?")
        bits = [
            f"{add_name} ({add_position}) is unowned and projects "
            f"{candidate.gain:+.1f} rest-of-season points for your lineup "
            f"in {who}'s place."
        ]
        if candidate.cost < DEAD_WEIGHT:
            bits.append(f"{who} is not reaching your lineup as it stands.")
        return " ".join(bits)

    count = len(candidate.blocked_by)
    if count:
        ahead = ", ".join(names.get(p, p) for p in candidate.blocked_by[:3])
        return (
            f"{who} is your {position}{count + 1} behind {ahead} — cutting him "
            "costs nothing, and nobody in the pool beats him either."
        )
    return (
        f"{who} never reaches your best lineup, and no free agent at "
        f"{position} improves on him. Cut him if you need the roster spot."
    )


def find_drops(
    league: LeagueContext,
    season_projections: ProjectionSet,
    *,
    my_roster_id: int,
    pool_size: int = POOL_SIZE,
    limit: int = 10,
) -> DropReport:
    """Rank drop/add swaps, and the players who are free to cut."""
    if season_projections.frame.empty:
        return DropReport(notes=["No projections available."])
    team = league.teams.get(int(my_roster_id))
    if team is None:
        return DropReport(notes=["Pick your team first."])

    evaluator = RosterEvaluator(league, season_projections)
    # Injured reserve and taxi players do not occupy an active roster spot, so
    # they are neither droppable-for-value nor blocking anyone.
    mine = [p for p in team.active_players if evaluator.known(str(p))]
    roster = frozenset(str(p) for p in mine)
    if not roster:
        return DropReport(
            notes=["Your roster is empty — nothing to drop until you have players."]
        )

    frame = season_projections.frame
    column = "exp_points" if "exp_points" in frame.columns else "proj_mean"
    ids = frame["sleeper_id"].astype(str)
    names = dict(zip(ids, frame["name"].astype(str)))

    rostered_anywhere = league.rostered_players()
    pool = [
        str(r.sleeper_id)
        for r in frame[~ids.isin(rostered_anywhere)]
        .nlargest(pool_size, column)
        .itertuples(index=False)
    ]
    if not pool:
        return DropReport(notes=["Every projected player is already rostered."])

    base = evaluator.team_value(roster)
    candidates: list[DropCandidate] = []
    for player in roster:
        without = roster - {player}
        cost = base - evaluator.team_value(without)

        best_add, best_gain = None, 0.0
        for free_agent in pool:
            gain = evaluator.team_value(without | {free_agent}) - base
            if gain > best_gain:
                best_add, best_gain = free_agent, gain

        candidate = DropCandidate(
            drop=player,
            cost=cost,
            add=best_add,
            gain=best_gain,
            blocked_by=_blockers(evaluator, roster, player),
        )
        candidate.reason = _reason(evaluator, candidate, names)
        candidates.append(candidate)

    upgrades = sorted(
        (c for c in candidates if c.is_upgrade), key=lambda c: -c.gain
    )
    # One add can be the answer for several drops; recommending the same
    # free agent three times is one idea printed three ways.
    seen_adds: set[str] = set()
    deduped: list[DropCandidate] = []
    for candidate in upgrades:
        if candidate.add in seen_adds:
            continue
        seen_adds.add(candidate.add or "")
        deduped.append(candidate)

    dead = sorted(
        (c for c in candidates if not c.is_upgrade and c.cost < DEAD_WEIGHT),
        key=lambda c: c.cost,
    )

    notes: list[str] = []
    unknown = [p for p in team.active_players if not evaluator.known(str(p))]
    if unknown:
        notes.append(
            f"{len(unknown)} of your players have no projection and were left "
            "out of this comparison."
        )
    if not deduped:
        notes.append(
            "Nobody in the open pool improves your lineup — every free agent "
            "who would start is already rostered somewhere."
        )
    if not dead and not deduped:
        notes.append("Every player you roster is contributing to your best lineup.")

    return DropReport(
        upgrades=deduped[:limit], dead_weight=dead[:limit], notes=notes
    )
