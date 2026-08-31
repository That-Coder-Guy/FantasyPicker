"""Trade recommendations that the other manager would actually accept.

The engine rests on one fact: a player's value is not a number, it is a
function of the roster he lands on. A third running back is nearly worthless to
a team that starts two and priceless to a team starting a waiver-wire body.
Every trade both sides should want exists because of that spread, so the search
is organised around it:

* **Roster value** is what a team would actually score: the exact
  best-possible starting lineup over rest-of-season projections (the same
  assignment solver the lineup page uses), plus a discounted term for bench
  depth so hoarding usable players is worth something but never as much as
  starting them.
* **A trade is evaluated twice** — once from each side, on that side's own
  roster, including the knock-on moves an uneven package forces: the side left
  short backfills from the best free agents, the side left long drops its most
  useless player. A 2-for-1 is good *because* of those knock-ons, so they are
  in the number rather than a footnote.
* **A trade is only proposed if the other side gains too.** Not "could be
  argued into", gains — their own best lineup improves. On top of that a
  perceived-value check models the psychology the lineup math misses: nobody
  trades their best player for a pile of parts whose headline is smaller than
  what they gave, however well the pile fits. Both filters have to pass.

**Chains** exist because the best first trade can unlock a second one. Acquire
a starting receiver and your old WR2 becomes surplus — a piece some third team
wants more than you do, and that yesterday you could not spare. The chain
search applies a top trade to your roster and re-searches from the new state,
so follow-ups can spend players the first step just acquired. A chain is only
reported when its total beats the best single trade; otherwise it is noise.

Values are rest-of-season because that is the horizon a trade lives on; the
weekly page already answers "who starts on Sunday".
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np

from ..model.predict import ProjectionSet
from ..sleeper.league import LeagueContext
from .lineup import solve_assignment

log = logging.getLogger(__name__)

#: A bench player is insurance, not points: he scores only when someone ahead
#: of him gets hurt or busts. Weighting the top of the bench at a quarter of
#: face value keeps depth from being treated as free to give away, without
#: letting it rival starters.
BENCH_WEIGHT = 0.25
BENCH_DEPTH = 3

#: The other side must clear this much rest-of-season lineup gain before a
#: proposal is worth their click.
MIN_THEIR_GAIN = 0.3
#: And we must clear this much to bother proposing at all.
MIN_MY_GAIN = 1.0

#: Headline rule: a side will not accept a package whose total perceived value
#: is much below the best player they surrender. 0.9 leaves room for genuine
#: consolidation while blocking star-for-scraps offers.
HEADLINE_FACTOR = 0.9

#: Quick prune: skip exact evaluation when the two packages' perceived values
#: differ by more than this fraction of the larger side. Real accepted trades
#: live well inside this window.
BALANCE_WINDOW = 0.25

#: The consolidation discount: in a multi-player package the market values the
#: headliner at face and everything behind him at a discount, because roster
#: spots are scarce and quality beats quantity. This is why a real 2-for-1
#: sends ~130%% of face value for the stud and both sides still feel fine.
PACKAGE_DISCOUNT = 0.6


@dataclass
class TradeSide:
    roster_id: int
    label: str
    gives: list[str]
    gain: float
    #: Forced knock-on moves that the gain already includes.
    adds: list[str] = field(default_factory=list)
    drops: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "roster_id": self.roster_id,
            "label": self.label,
            "gives": list(self.gives),
            "gain": round(self.gain, 1),
            "adds": list(self.adds),
            "drops": list(self.drops),
        }


@dataclass
class Trade:
    me: TradeSide
    them: TradeSide
    appeal: str
    #: How likely the other manager is to say yes, from their gain and the
    #: perceived-value balance: "likely" / "plausible" / "a stretch".
    likelihood: str
    #: my_gain weighted by that likelihood — what proposing this is worth in
    #: expectation, and the order the list is ranked in. Chasing the single
    #: biggest heist is worse than a slightly smaller trade that gets accepted.
    expected: float
    #: Perceived-value balance from their side: received minus given, in
    #: rest-of-season points. Near zero reads as a fair deal.
    balance: float
    rationale: str

    @property
    def my_gain(self) -> float:
        return self.me.gain

    def key(self) -> tuple:
        return (
            self.them.roster_id,
            tuple(sorted(self.me.gives)),
            tuple(sorted(self.them.gives)),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "me": self.me.as_dict(),
            "them": self.them.as_dict(),
            "appeal": self.appeal,
            "likelihood": self.likelihood,
            "expected": round(self.expected, 1),
            "balance": round(self.balance, 1),
            "rationale": self.rationale,
        }


@dataclass
class TradeChain:
    steps: list[Trade]
    total_gain: float

    def as_dict(self) -> dict[str, object]:
        return {
            "steps": [s.as_dict() for s in self.steps],
            "total_gain": round(self.total_gain, 1),
        }


@dataclass
class TradeReport:
    trades: list[Trade]
    chains: list[TradeChain] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "trades": [t.as_dict() for t in self.trades],
            "chains": [c.as_dict() for c in self.chains],
            "notes": list(self.notes),
        }


class RosterEvaluator:
    """Cached roster valuation over one set of rest-of-season projections.

    Shared with the drop finder so both pages price a roster identically — two
    views that disagreed about what a team is worth would be worse than either
    alone.
    """

    def __init__(self, league: LeagueContext, projections: ProjectionSet) -> None:
        self.league = league
        self.slots = league.slots
        frame = projections.frame
        column = "exp_points" if "exp_points" in frame.columns else "proj_mean"
        ids = frame["sleeper_id"].astype(str)
        self.value: dict[str, float] = dict(
            zip(ids, frame[column].astype(float).fillna(0.0))
        )
        self.position: dict[str, str] = dict(zip(ids, frame["position"].astype(str)))
        self._cache: dict[frozenset, float] = {}

        rostered = league.rostered_players()
        pool = frame[~ids.isin(rostered)]
        # Free agents worth backfilling with after an uneven trade. Kickers and
        # defenses are always on this list, which is realistic: they are the
        # roster spots people actually churn.
        self.free_agents: list[str] = [
            str(r.sleeper_id) for r in pool.nlargest(40, column).itertuples(index=False)
        ]

    def known(self, player: str) -> bool:
        return player in self.value

    def team_value(self, players: frozenset) -> float:
        """Best-lineup points plus discounted bench depth. Exact, cached."""
        cached = self._cache.get(players)
        if cached is not None:
            return cached
        ids = [p for p in players if p in self.value]
        values = np.array([self.value[p] for p in ids], dtype=float)
        positions = [self.position[p] for p in ids]
        solution = solve_assignment(self.slots, ids, positions, values)
        starters = set(solution.starters)
        bench = sorted(
            (self.value[p] for p in ids if p not in starters), reverse=True
        )
        total = solution.objective + BENCH_WEIGHT * sum(bench[:BENCH_DEPTH])
        self._cache[players] = total
        return total

    def marginal(self, players: frozenset, player: str) -> float:
        """What this roster loses if the player vanishes."""
        if player not in players:
            return 0.0
        return self.team_value(players) - self.team_value(players - {player})

    def settle(
        self, players: frozenset, target_size: int, taken: frozenset
    ) -> tuple[frozenset, list[str], list[str]]:
        """Bring an uneven roster back to size with pickups or drops.

        This is where a 2-for-1 earns its keep: the side that consolidated
        re-opens a roster spot and fills it from waivers, and the side that
        fattened up sheds its most useless player. ``taken`` keeps a chain's
        earlier steps from handing the same free agent to two teams.
        """
        adds: list[str] = []
        drops: list[str] = []
        while len(players) < target_size:
            best, best_gain = None, -1.0
            base = self.team_value(players)
            tried = 0
            for candidate in self.free_agents:
                if candidate in players or candidate in taken:
                    continue
                gain = self.team_value(players | {candidate}) - base
                if gain > best_gain:
                    best, best_gain = candidate, gain
                tried += 1
                # The pool is sorted by raw value; past the first dozen the
                # best fit has been seen, and each try is an exact solve.
                if tried >= 12:
                    break
            if best is None:
                break
            players = players | {best}
            adds.append(best)
        while len(players) > target_size:
            # Drop whoever the lineup misses least.
            worst, worst_loss = None, float("inf")
            for candidate in sorted(players, key=lambda p: self.value.get(p, 0.0))[:5]:
                loss = self.marginal(players, candidate)
                if loss < worst_loss:
                    worst, worst_loss = candidate, loss
            if worst is None:
                break
            players = players - {worst}
            drops.append(worst)
        return players, adds, drops


def _candidates(evaluator: RosterEvaluator, players: frozenset, *, limit: int) -> list[str]:
    """Who a team might plausibly put in a deal.

    Two kinds of player move in real trades: the expendable (low marginal value
    to their current roster — depth behind a stud, the wrong position) and the
    headliners (high raw value, moved in consolidations). Everyone in between
    mostly stays put, so the search only packages these.
    """
    known = [p for p in players if evaluator.known(p)]
    if not known:
        return []
    by_marginal = sorted(known, key=lambda p: evaluator.marginal(players, p))
    by_value = sorted(known, key=lambda p: -evaluator.value[p])
    picked: list[str] = []
    for player in by_marginal[: limit - limit // 3] + by_value[: limit // 3 + 2]:
        if player not in picked:
            picked.append(player)
    return picked[:limit]


def _packages(candidates: list[str], max_package: int) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = [(p,) for p in candidates]
    if max_package >= 2:
        out.extend(itertools.combinations(candidates, 2))
    return out


def _evaluate(
    evaluator: RosterEvaluator,
    my_players: frozenset,
    their_players: frozenset,
    give: tuple[str, ...],
    get: tuple[str, ...],
    *,
    taken: frozenset = frozenset(),
) -> tuple[float, float, list[str], list[str], list[str], list[str], frozenset]:
    """Both sides' exact gains for one package, knock-on moves included."""
    my_after = (my_players - set(give)) | set(get)
    their_after = (their_players - set(get)) | set(give)

    my_after, my_adds, my_drops = evaluator.settle(
        my_after, len(my_players), taken | their_after
    )
    their_after, their_adds, their_drops = evaluator.settle(
        their_after, len(their_players), taken | my_after
    )

    my_gain = evaluator.team_value(my_after) - evaluator.team_value(my_players)
    their_gain = evaluator.team_value(their_after) - evaluator.team_value(their_players)
    return my_gain, their_gain, my_adds, my_drops, their_adds, their_drops, my_after


def _effective(values: list[float]) -> float:
    """A package's perceived value under the consolidation discount."""
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    return ordered[0] + PACKAGE_DISCOUNT * sum(ordered[1:])


def _appeal(their_gain: float, balance: float) -> str:
    if their_gain >= 2.5 and balance >= -1.0:
        return "clear win for them"
    if their_gain >= 1.0:
        return "solid for them"
    return "slight edge for them"


def _likelihood(their_gain: float, balance: float) -> tuple[str, float]:
    """Would they actually click accept?

    Their lineup gain is what a sharp manager checks; the perceived balance is
    what everyone else checks. Both feed one score, and the returned weight
    turns my gain into an expected value — a +8 trade accepted half the time
    beats a +15 heist that gets laughed out of the league chat.
    """
    score = their_gain + 0.15 * balance
    if score >= 2.5:
        return "likely", 0.75
    if score >= 0.8:
        return "plausible", 0.55
    return "a stretch", 0.3


def _positions(evaluator: RosterEvaluator, players: tuple[str, ...]) -> str:
    return "/".join(evaluator.position.get(p, "?") for p in players)


def _rationale(
    evaluator: RosterEvaluator,
    give: tuple[str, ...],
    get: tuple[str, ...],
    my_gain: float,
    their_gain: float,
    my_adds: list[str],
    their_drops: list[str],
) -> str:
    bits = [
        f"You send {_positions(evaluator, give)} for {_positions(evaluator, get)}: "
        f"your best lineup gains {my_gain:+.1f} rest-of-season points, "
        f"theirs gains {their_gain:+.1f} — it fits both rosters."
    ]
    if len(give) > len(get):
        bits.append("Consolidating two spots into one lets you add off waivers.")
    elif len(get) > len(give):
        bits.append("They consolidate; you take the extra depth.")
    if my_adds:
        bits.append("Your freed roster spot is backfilled from free agency.")
    if their_drops:
        bits.append("They shed their least useful player to make room.")
    return " ".join(bits)


def _search_pair(
    evaluator: RosterEvaluator,
    my_id: int,
    my_players: frozenset,
    their_team,
    *,
    max_package: int,
    candidate_limit: int,
    min_my_gain: float,
    min_their_gain: float,
    my_label: str,
    taken: frozenset = frozenset(),
) -> list[Trade]:
    """Every acceptable package between one pair of rosters."""
    their_players = frozenset(str(p) for p in their_team.players)
    if not their_players:
        return []
    mine = _candidates(evaluator, my_players, limit=candidate_limit)
    theirs = _candidates(evaluator, their_players, limit=candidate_limit)
    if not mine or not theirs:
        return []

    trades: list[Trade] = []
    for give in _packages(mine, max_package):
        give_eff = _effective([evaluator.value[p] for p in give])
        my_headliner = max(evaluator.value[p] for p in give)
        for get in _packages(theirs, max_package):
            get_eff = _effective([evaluator.value[p] for p in get])
            top = max(give_eff, get_eff)
            if top > 0 and abs(give_eff - get_eff) > BALANCE_WINDOW * top:
                continue  # too lopsided on perceived value to ever be accepted
            # Headline rule from their side: what they receive must roughly
            # match the best player they surrender.
            their_headliner = max(evaluator.value[p] for p in get)
            if give_eff < their_headliner * HEADLINE_FACTOR:
                continue
            # And from mine, so we never propose donating our own star.
            if get_eff < my_headliner * HEADLINE_FACTOR:
                continue

            my_gain, their_gain, my_adds, my_drops, their_adds, their_drops, _ = (
                _evaluate(evaluator, my_players, their_players, give, get, taken=taken)
            )
            if my_gain < min_my_gain or their_gain < min_their_gain:
                continue
            balance = give_eff - get_eff
            likelihood, weight = _likelihood(their_gain, balance)
            trades.append(
                Trade(
                    me=TradeSide(
                        roster_id=my_id,
                        label=my_label,
                        gives=list(give),
                        gain=my_gain,
                        adds=my_adds,
                        drops=my_drops,
                    ),
                    them=TradeSide(
                        roster_id=their_team.roster_id,
                        label=their_team.label,
                        gives=list(get),
                        gain=their_gain,
                        adds=their_adds,
                        drops=their_drops,
                    ),
                    appeal=_appeal(their_gain, balance),
                    likelihood=likelihood,
                    expected=my_gain * weight,
                    balance=balance,
                    rationale=_rationale(
                        evaluator, give, get, my_gain, their_gain, my_adds, their_drops
                    ),
                )
            )
    return trades


def _dedupe_best(trades: list[Trade]) -> list[Trade]:
    """Rank by expected value, then keep only genuinely different deals.

    Expected, not raw: the biggest one-sided heist has the smallest chance of
    being accepted, and a proposal that never lands is worth nothing.

    The same target invites many near-identical packages — their stud for my
    surplus RB plus any one of four throw-ins. Showing all four is noise, so
    once a (opponent, package) has appeared on either side of a kept deal, its
    variants are dropped and the list moves on to the next distinct idea.
    """
    ranked = sorted(trades, key=lambda t: -t.expected)
    kept: list[Trade] = []
    seen: set = set()
    for trade in ranked:
        my_key = (trade.them.roster_id, tuple(sorted(trade.me.gives)))
        their_key = (trade.them.roster_id, tuple(sorted(trade.them.gives)))
        if my_key in seen or their_key in seen:
            continue
        seen.add(my_key)
        seen.add(their_key)
        kept.append(trade)
    return kept


def find_trades(
    league: LeagueContext,
    season_projections: ProjectionSet,
    *,
    my_roster_id: int,
    max_package: int = 2,
    candidate_limit: int = 9,
    limit: int = 12,
    chains: bool = True,
) -> TradeReport:
    """Rank mutually beneficial trades, and chains of them, for one roster."""
    if season_projections.frame.empty:
        return TradeReport([], [], ["No projections available."])
    my_team = league.teams.get(int(my_roster_id))
    if my_team is None:
        return TradeReport([], [], ["Pick your team first."])
    my_players = frozenset(str(p) for p in my_team.players)
    if not my_players:
        return TradeReport(
            [], [], ["Your roster is empty — trades start mattering after the draft."]
        )

    evaluator = RosterEvaluator(league, season_projections)
    opponents = [
        t for rid, t in sorted(league.teams.items()) if rid != int(my_roster_id)
    ]

    all_trades: list[Trade] = []
    for their_team in opponents:
        all_trades.extend(
            _search_pair(
                evaluator,
                int(my_roster_id),
                my_players,
                their_team,
                max_package=max_package,
                candidate_limit=candidate_limit,
                min_my_gain=MIN_MY_GAIN,
                min_their_gain=MIN_THEIR_GAIN,
                my_label=my_team.label,
            )
        )

    ranked = _dedupe_best(all_trades)

    chain_results: list[TradeChain] = []
    if chains and ranked:
        chain_results = _find_chains(
            evaluator,
            league,
            int(my_roster_id),
            my_team.label,
            my_players,
            ranked,
            max_package=max_package,
        )

    notes: list[str] = []
    unknown = [p for p in my_players if not evaluator.known(p)]
    if unknown:
        notes.append(
            f"{len(unknown)} of your players have no projection and were left out "
            "of every package."
        )
    if not ranked:
        notes.append(
            "No trade clears the bar right now: every package that helps you "
            "either hurts the other roster or asks them to give up more face "
            "value than they get back."
        )
    return TradeReport(ranked[:limit], chain_results, notes)


def _find_chains(
    evaluator: RosterEvaluator,
    league: LeagueContext,
    my_id: int,
    my_label: str,
    my_players: frozenset,
    singles: list[Trade],
    *,
    max_package: int,
    seeds: int = 5,
    limit: int = 3,
) -> list[TradeChain]:
    """Two-step sequences that beat the best single trade.

    The second search runs on the roster the first trade produces, which is the
    whole point: a follow-up may spend a player the first step just brought in,
    or a player the first step benched into surplus.
    """
    best_single = singles[0].my_gain
    chains: list[TradeChain] = []

    for seed in singles[:seeds]:
        give = tuple(seed.me.gives)
        get = tuple(seed.them.gives)
        their_team = league.teams[seed.them.roster_id]
        their_players = frozenset(str(p) for p in their_team.players)
        _, _, _, _, _, _, after = _evaluate(
            evaluator, my_players, their_players, give, get
        )
        # Players the first step routed elsewhere are spoken for.
        taken = frozenset(give) | frozenset(seed.them.adds) | frozenset(seed.me.adds)

        for follow_team in league.teams.values():
            if follow_team.roster_id in (my_id, seed.them.roster_id):
                continue
            followups = _search_pair(
                evaluator,
                my_id,
                after,
                follow_team,
                max_package=max_package,
                candidate_limit=6,
                # The step must add real value on top of the seed, and stay
                # just as acceptable to its own counterparty.
                min_my_gain=0.8,
                min_their_gain=MIN_THEIR_GAIN,
                my_label=my_label,
                taken=taken,
            )
            followups.sort(key=lambda t: -t.expected)
            for follow in followups[:1]:
                total = (
                    evaluator.team_value(after) - evaluator.team_value(my_players)
                ) + follow.my_gain
                if total > best_single + 0.5:
                    chains.append(TradeChain(steps=[seed, follow], total_gain=total))

    chains.sort(key=lambda c: -c.total_gain)
    deduped: list[TradeChain] = []
    seen: set = set()
    for chain in chains:
        # Chains that differ only in which throw-in seeds them are one idea.
        key = (
            chain.steps[0].them.roster_id,
            tuple(sorted(chain.steps[0].them.gives)),
            chain.steps[1].key(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chain)
    return deduped[:limit]
