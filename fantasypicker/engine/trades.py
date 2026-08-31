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

Two currencies
--------------
The other manager is not looking at this app. They are looking at the
projection their platform prints, and that is the only number in front of them
when they decide. So the engine keeps two valuations and never mixes them up:

* **Model points** decide whether a roster genuinely improved. Every gain
  reported to you is in this currency, because it is the accurate one.
* **Market points** — the platform's own published projections — decide whether
  the deal *looks* fair, and whether the other side believes they are gaining.
  Every acceptance test runs here: the headline rule, the balance window, and
  their perceived gain.

That split is the difference between a trade that is good and a trade that gets
accepted. A deal our model loves but which reads as a fleece on espn.com is not
a good recommendation, it is a wasted proposal; with both currencies in hand
the engine can decline to suggest it. When no public projections are available
the market currency falls back to the model's and the behaviour is exactly what
it was before — no worse, just less informed.

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

from ..market import MarketProjections
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
    #: What this side's package is worth on the public projections — the total
    #: the other manager adds up when deciding whether the offer is insulting.
    market_value: float = 0.0
    #: The same package on this app's numbers, for the side-by-side.
    model_value: float = 0.0
    #: What this side's lineup gains as *they* would compute it, on public
    #: numbers. Differs from ``gain``, which is what they actually get.
    perceived_gain: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "roster_id": self.roster_id,
            "label": self.label,
            "gives": list(self.gives),
            "gain": round(self.gain, 1),
            "adds": list(self.adds),
            "drops": list(self.drops),
            "market_value": round(self.market_value, 1),
            "model_value": round(self.model_value, 1),
            "perceived_gain": round(self.perceived_gain, 1),
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
    #: *public* rest-of-season points. Near zero reads as a fair deal on the
    #: numbers they are actually looking at.
    balance: float
    rationale: str
    #: The same balance on this app's numbers. Where the two disagree sharply,
    #: the deal is an arbitrage: fair on their screen, a win on ours.
    model_balance: float = 0.0

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
            "model_balance": round(self.model_balance, 1),
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
    #: Where the public numbers came from, so the page can name the source
    #: rather than presenting two columns of unexplained points.
    market: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "trades": [t.as_dict() for t in self.trades],
            "chains": [c.as_dict() for c in self.chains],
            "notes": list(self.notes),
            "market": dict(self.market),
        }


class RosterEvaluator:
    """Cached roster valuation over one set of rest-of-season projections.

    Shared with the drop finder so both pages price a roster identically — two
    views that disagreed about what a team is worth would be worse than either
    alone.
    """

    def __init__(
        self,
        league: LeagueContext,
        projections: ProjectionSet,
        *,
        values: dict[str, float] | None = None,
    ) -> None:
        self.league = league
        self.slots = league.slots
        frame = projections.frame
        column = "exp_points" if "exp_points" in frame.columns else "proj_mean"
        ids = frame["sleeper_id"].astype(str)
        model_value: dict[str, float] = dict(
            zip(ids, frame[column].astype(float).fillna(0.0))
        )
        # ``values`` overrides what a player is worth without changing who
        # exists: a market-currency evaluator prices the same roster, in the
        # same slots, on the numbers the rest of the league can see. Anyone the
        # override misses keeps his model value, so a partially covered source
        # degrades player by player instead of pricing him at zero.
        self.value: dict[str, float] = (
            {pid: values.get(pid, model_value[pid]) for pid in model_value}
            if values is not None
            else model_value
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


@dataclass
class _Outcome:
    """Both rosters after one package, and what each side made on it."""

    my_gain: float
    their_gain: float
    my_adds: list[str]
    my_drops: list[str]
    their_adds: list[str]
    their_drops: list[str]
    my_after: frozenset
    their_after: frozenset


def _evaluate(
    evaluator: RosterEvaluator,
    my_players: frozenset,
    their_players: frozenset,
    give: tuple[str, ...],
    get: tuple[str, ...],
    *,
    taken: frozenset = frozenset(),
) -> _Outcome:
    """Both sides' exact gains for one package, knock-on moves included."""
    my_after = (my_players - set(give)) | set(get)
    their_after = (their_players - set(get)) | set(give)

    my_after, my_adds, my_drops = evaluator.settle(
        my_after, len(my_players), taken | their_after
    )
    their_after, their_adds, their_drops = evaluator.settle(
        their_after, len(their_players), taken | my_after
    )

    return _Outcome(
        my_gain=evaluator.team_value(my_after) - evaluator.team_value(my_players),
        their_gain=evaluator.team_value(their_after)
        - evaluator.team_value(their_players),
        my_adds=my_adds,
        my_drops=my_drops,
        their_adds=their_adds,
        their_drops=their_drops,
        my_after=my_after,
        their_after=their_after,
    )


def _effective(values: list[float]) -> float:
    """A package's perceived value under the consolidation discount."""
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    return ordered[0] + PACKAGE_DISCOUNT * sum(ordered[1:])


def _appeal(perceived_gain: float, balance: float) -> str:
    """How the offer reads to them — so, in the currency they read it in."""
    if perceived_gain >= 2.5 and balance >= -1.0:
        return "clear win for them"
    if perceived_gain >= 1.0:
        return "solid for them"
    return "slight edge for them"


def _likelihood(perceived_gain: float, balance: float) -> tuple[str, float]:
    """Would they actually click accept?

    Both inputs are in the currency they can see. The perceived lineup gain is
    what a sharp manager checks — on their platform's projections, not ours —
    and the perceived balance is what everyone else checks. Both feed one
    score, and the returned weight turns my gain into an expected value: a +8
    trade accepted half the time beats a +15 heist that gets laughed out of the
    league chat.
    """
    score = perceived_gain + 0.15 * balance
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
    *,
    market_source: str | None = None,
    give_market: float = 0.0,
    get_market: float = 0.0,
) -> str:
    bits = [
        f"You send {_positions(evaluator, give)} for {_positions(evaluator, get)}: "
        f"your best lineup gains {my_gain:+.1f} rest-of-season points, "
        f"theirs gains {their_gain:+.1f} — it fits both rosters."
    ]
    # The half of the pitch that survives contact with the other manager: what
    # the deal totals on the projections they are looking at.
    if market_source:
        bits.append(
            f"On {market_source} projections they receive {give_market:.0f} points "
            f"and send {get_market:.0f}, so it reads as fair from their side."
        )
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
    market: RosterEvaluator | None = None,
    market_source: str | None = None,
) -> list[Trade]:
    """Every acceptable package between one pair of rosters.

    ``evaluator`` prices rosters on the model — that decides whether a trade is
    *good*. ``market`` prices the same rosters on the public projections — that
    decides whether it will be *accepted*. When no public source is available
    the two are the same object and this behaves as it always did.
    """
    their_players = frozenset(str(p) for p in their_team.players)
    if not their_players:
        return []
    seen_by_them = market or evaluator
    mine = _candidates(evaluator, my_players, limit=candidate_limit)
    theirs = _candidates(evaluator, their_players, limit=candidate_limit)
    if not mine or not theirs:
        return []

    trades: list[Trade] = []
    for give in _packages(mine, max_package):
        # Every perceived-value test below is in the public currency, because
        # face value is exactly the thing this app sees differently from the
        # person being asked to say yes.
        give_eff = _effective([seen_by_them.value[p] for p in give])
        my_headliner = max(seen_by_them.value[p] for p in give)
        for get in _packages(theirs, max_package):
            get_eff = _effective([seen_by_them.value[p] for p in get])
            top = max(give_eff, get_eff)
            if top > 0 and abs(give_eff - get_eff) > BALANCE_WINDOW * top:
                continue  # too lopsided on their numbers to ever be accepted
            # Headline rule from their side: what they receive must roughly
            # match the best player they surrender.
            their_headliner = max(seen_by_them.value[p] for p in get)
            if give_eff < their_headliner * HEADLINE_FACTOR:
                continue
            # And from mine, so we never propose donating our own star.
            if get_eff < my_headliner * HEADLINE_FACTOR:
                continue

            outcome = _evaluate(
                evaluator, my_players, their_players, give, get, taken=taken
            )
            if outcome.my_gain < min_my_gain or outcome.their_gain < min_their_gain:
                continue

            # What they will compute for themselves, on the rosters this trade
            # actually produces. The knock-on moves are taken as settled above
            # rather than re-solved in the other currency: the question here is
            # how this specific deal scores on their screen, not what a
            # differently-informed manager would have done with the waiver wire.
            if market is None:
                perceived_gain = outcome.their_gain
            else:
                perceived_gain = market.team_value(
                    outcome.their_after
                ) - market.team_value(their_players)
            # An offer they read as a downgrade is not a trade, whatever our
            # numbers say about it.
            if perceived_gain < min_their_gain:
                continue

            balance = give_eff - get_eff
            model_balance = _effective(
                [evaluator.value[p] for p in give]
            ) - _effective([evaluator.value[p] for p in get])
            likelihood, weight = _likelihood(perceived_gain, balance)
            trades.append(
                Trade(
                    me=TradeSide(
                        roster_id=my_id,
                        label=my_label,
                        gives=list(give),
                        gain=outcome.my_gain,
                        adds=outcome.my_adds,
                        drops=outcome.my_drops,
                        market_value=sum(seen_by_them.value[p] for p in give),
                        model_value=sum(evaluator.value[p] for p in give),
                    ),
                    them=TradeSide(
                        roster_id=their_team.roster_id,
                        label=their_team.label,
                        gives=list(get),
                        gain=outcome.their_gain,
                        adds=outcome.their_adds,
                        drops=outcome.their_drops,
                        market_value=sum(seen_by_them.value[p] for p in get),
                        model_value=sum(evaluator.value[p] for p in get),
                        perceived_gain=perceived_gain,
                    ),
                    appeal=_appeal(perceived_gain, balance),
                    likelihood=likelihood,
                    expected=outcome.my_gain * weight,
                    balance=balance,
                    model_balance=model_balance,
                    rationale=_rationale(
                        evaluator,
                        give,
                        get,
                        outcome.my_gain,
                        outcome.their_gain,
                        outcome.my_adds,
                        outcome.their_drops,
                        market_source=market_source,
                        give_market=sum(seen_by_them.value[p] for p in give),
                        get_market=sum(seen_by_them.value[p] for p in get),
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
    market: MarketProjections | None = None,
) -> TradeReport:
    """Rank mutually beneficial trades, and chains of them, for one roster.

    ``market`` supplies the projections the rest of the league is looking at.
    Pass none and every judgement falls back to the model, which is what this
    engine did before the distinction existed.
    """
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
    market_evaluator: RosterEvaluator | None = None
    market_source: str | None = None
    if market is not None and market.available:
        market_evaluator = RosterEvaluator(
            league, season_projections, values=market.points
        )
        market_source = market.source
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
                market=market_evaluator,
                market_source=market_source,
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
            market=market_evaluator,
            market_source=market_source,
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
    if market is not None:
        notes.extend(market.notes)
    return TradeReport(
        ranked[:limit],
        chain_results,
        notes,
        market=market.as_dict() if market is not None else {},
    )


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
    market: RosterEvaluator | None = None,
    market_source: str | None = None,
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
        after = _evaluate(
            evaluator, my_players, their_players, give, get
        ).my_after
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
                market=market,
                market_source=market_source,
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
