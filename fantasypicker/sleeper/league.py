"""League shape: roster slots, teams, and weekly matchups.

The point of this module is that nothing downstream should have to know what a
``SUPER_FLEX`` string means. Slot eligibility, the identity of "my" team, and
who I play this week all get resolved here, from the league ID alone — which is
what removes the need to type an opponent's roster in by hand.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from .client import SleeperClient
from .scoring import ScoringRules

log = logging.getLogger(__name__)


class LeagueNotFound(ValueError):
    """Sleeper has no league with this ID.

    Carries enough detail for the UI to explain the two things that actually go
    wrong: the number is a draft or user ID rather than a league ID, or the
    league lives on a different platform entirely.
    """

    def __init__(self, league_id: str) -> None:
        self.league_id = league_id
        looks_numeric = str(league_id).isdigit()
        hint = (
            "Sleeper league IDs are 18-19 digit numbers — the one in "
            "sleeper.com/leagues/<this-number>/team. A draft ID or a user ID "
            "will not work here."
            if looks_numeric
            else "That does not look like a Sleeper league ID at all; they are "
            "long numbers with no letters or slashes."
        )
        super().__init__(
            f"Sleeper has no league with ID {league_id!r}. {hint} "
            "Enter your Sleeper username instead and pick the league from the list."
        )

#: Sleeper slot name -> the fantasy positions that may fill it.
SLOT_ELIGIBILITY: dict[str, frozenset[str]] = {
    "QB": frozenset({"QB"}),
    "RB": frozenset({"RB"}),
    "WR": frozenset({"WR"}),
    "TE": frozenset({"TE"}),
    "K": frozenset({"K"}),
    "DEF": frozenset({"DST"}),
    "DST": frozenset({"DST"}),
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "WRRB_FLEX": frozenset({"RB", "WR"}),
    "WRRB_WRT": frozenset({"RB", "WR", "TE"}),
    "REC_FLEX": frozenset({"WR", "TE"}),
    "SUPER_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
    "QB_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
    "IDP_FLEX": frozenset({"DL", "LB", "DB"}),
    "DL": frozenset({"DL", "DE", "DT"}),
    "LB": frozenset({"LB"}),
    "DB": frozenset({"DB", "CB", "S"}),
}

#: Slots that are not part of a starting lineup.
BENCH_SLOTS = frozenset({"BN", "IR", "TAXI"})


@dataclass(frozen=True)
class RosterSlot:
    """One starting-lineup slot, in the order Sleeper lists it."""

    index: int
    name: str

    @property
    def eligible(self) -> frozenset[str]:
        return SLOT_ELIGIBILITY.get(self.name, frozenset({self.name}))

    def accepts(self, position: str | None) -> bool:
        return bool(position) and position.upper() in self.eligible


@dataclass
class Team:
    roster_id: int
    owner_id: str | None
    display_name: str
    team_name: str
    players: list[str] = field(default_factory=list)
    starters: list[str] = field(default_factory=list)
    reserve: list[str] = field(default_factory=list)
    taxi: list[str] = field(default_factory=list)
    username: str = ""
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    avatar: str | None = None
    claimed: bool = True

    @property
    def label(self) -> str:
        """What to call this team.

        Sleeper carries three candidate names and any of them can be blank: the
        custom team name (most managers never set one), the display name, and
        the login username. Falling back through all three means a placeholder
        is reached only when Sleeper genuinely told us nothing.

        There are two ways to get there and they mean opposite things. A seat
        nobody has joined yet has no name because there is no manager — normal
        in a league that is still filling up, and worth saying outright, since
        "Team 7" otherwise reads as a name that failed to load. A roster that
        *has* players but no owner is an abandoned team, which is a real team
        and keeps the numeric name.
        """
        named = self.team_name or self.display_name or self.username
        if named:
            return named
        if not self.claimed and not self.players:
            return f"Open seat {self.roster_id}"
        return f"Team {self.roster_id}"

    @property
    def manager(self) -> str:
        """The person, as distinct from the team.

        Shown next to :attr:`label` so a custom team name still tells you who
        you are trading with.
        """
        if self.display_name or self.username:
            return self.display_name or self.username
        return "nobody has joined yet" if not self.players else "unclaimed"

    @property
    def record(self) -> str:
        base = f"{self.wins}-{self.losses}"
        return f"{base}-{self.ties}" if self.ties else base

    @property
    def active_players(self) -> list[str]:
        """Roster minus IR/taxi — the players actually available to start."""
        blocked = set(self.reserve) | set(self.taxi)
        return [p for p in self.players if p not in blocked]


@dataclass
class Matchup:
    week: int
    matchup_id: int | None
    home: Team
    away: Team | None
    home_starters: list[str] = field(default_factory=list)
    away_starters: list[str] = field(default_factory=list)
    home_points: float = 0.0
    away_points: float = 0.0


@dataclass
class LeagueContext:
    """Everything about a league that the engines need."""

    league_id: str
    raw: dict
    scoring: ScoringRules
    slots: list[RosterSlot]
    bench_size: int
    teams: dict[int, Team]
    season: int
    current_week: int
    my_roster_id: int | None = None
    #: roster_id -> matchup_id, per week, filled lazily by :meth:`matchup_for`
    _matchup_cache: dict[int, list[dict]] = field(default_factory=dict, repr=False)

    # -- shape ------------------------------------------------------------- #

    @property
    def name(self) -> str:
        return self.raw.get("name") or f"League {self.league_id}"

    @property
    def team_count(self) -> int:
        return int(self.raw.get("total_rosters") or len(self.teams) or 12)

    @property
    def roster_size(self) -> int:
        return len(self.slots) + self.bench_size

    @property
    def is_superflex(self) -> bool:
        return any(s.name in {"SUPER_FLEX", "QB_FLEX"} for s in self.slots)

    @property
    def is_idp(self) -> bool:
        return any(s.name in {"IDP_FLEX", "DL", "LB", "DB"} for s in self.slots)

    @property
    def is_dynasty(self) -> bool:
        settings = self.raw.get("settings") or {}
        return bool(settings.get("type") == 2 or (self.raw.get("previous_league_id")))

    @property
    def starting_slot_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for slot in self.slots:
            counts[slot.name] = counts.get(slot.name, 0) + 1
        return counts

    def starters_needed(self, position: str) -> float:
        """Expected number of starters of ``position`` per team.

        Dedicated slots count fully; flex slots are split across the positions
        that can fill them, which is what makes replacement level realistic in a
        3-WR-plus-flex league versus a 2-WR league.
        """
        position = position.upper()
        total = 0.0
        for slot in self.slots:
            eligible = slot.eligible
            if position in eligible:
                total += 1.0 / len(eligible) if len(eligible) > 1 else 1.0
        return total

    @property
    def my_team(self) -> Team | None:
        if self.my_roster_id is None:
            return None
        return self.teams.get(self.my_roster_id)

    def team_by_owner(self, owner_id: str) -> Team | None:
        for team in self.teams.values():
            if team.owner_id == owner_id:
                return team
        return None

    def rostered_players(self, exclude_roster_id: int | None = None) -> set[str]:
        taken: set[str] = set()
        for roster_id, team in self.teams.items():
            if roster_id == exclude_roster_id:
                continue
            taken.update(team.players)
        return taken

    # -- weekly ------------------------------------------------------------ #

    async def load_matchups(self, client: SleeperClient, week: int) -> list[dict]:
        if week not in self._matchup_cache:
            self._matchup_cache[week] = await client.matchups(self.league_id, week)
        return self._matchup_cache[week]

    async def matchup_for(
        self, client: SleeperClient, week: int, roster_id: int | None = None
    ) -> Matchup | None:
        """The week's pairing for ``roster_id`` — including the opponent's roster.

        This is the call that means an opponent's lineup never has to be entered
        manually: Sleeper hands back every roster's ``starters`` and ``players``
        for the week, keyed by a shared ``matchup_id``.
        """
        roster_id = roster_id if roster_id is not None else self.my_roster_id
        if roster_id is None:
            return None
        rows = await self.load_matchups(client, week)
        mine = next((r for r in rows if r.get("roster_id") == roster_id), None)
        if mine is None:
            return None
        matchup_id = mine.get("matchup_id")
        theirs = next(
            (
                r
                for r in rows
                if r.get("matchup_id") == matchup_id
                and r.get("roster_id") != roster_id
                and matchup_id is not None
            ),
            None,
        )
        home = self.teams.get(roster_id)
        if home is None:
            return None
        away = self.teams.get(theirs.get("roster_id")) if theirs else None
        return Matchup(
            week=week,
            matchup_id=matchup_id,
            home=home,
            away=away,
            home_starters=[p for p in (mine.get("starters") or []) if p and p != "0"],
            away_starters=[
                p for p in ((theirs or {}).get("starters") or []) if p and p != "0"
            ],
            home_points=float(mine.get("points") or 0.0),
            away_points=float((theirs or {}).get("points") or 0.0),
        )


def _parse_slots(roster_positions: list[str] | None) -> tuple[list[RosterSlot], int]:
    slots: list[RosterSlot] = []
    bench = 0
    for position in roster_positions or []:
        if position in BENCH_SLOTS:
            bench += 1
            continue
        slots.append(RosterSlot(index=len(slots), name=position))
    if not slots:  # league object was unavailable; assume a common shape
        default = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
        slots = [RosterSlot(i, name) for i, name in enumerate(default)]
        bench = 6
    return slots, bench


def build_teams(
    rosters: list[dict],
    users: list[dict],
    *,
    previous: dict[int, Team] | None = None,
) -> dict[int, Team]:
    """Turn Sleeper's roster and user lists into :class:`Team` objects.

    Identity — the team name, the owner's display name, the avatar — lives on
    the *user* object, not the roster. So a rosters response that arrives
    without a matching users response would otherwise rename every team in the
    league to "Team 4". ``previous`` carries the last known identity forward,
    because a stale name is enormously better than a wrong one.

    Sleeper also allows a roster with no ``owner_id`` at all — an orphaned team
    the commissioner is running, which most active leagues acquire eventually.
    Those fall back to the first co-owner before giving up on a name.
    """
    users_by_id = {u.get("user_id"): u for u in users if u.get("user_id")}
    known = previous or {}
    teams: dict[int, Team] = {}
    nameless: list[int] = []
    for row in rosters:
        roster_id = int(row.get("roster_id"))
        owner_id = row.get("owner_id")
        user = users_by_id.get(owner_id) or {}
        if not user:
            # An abandoned team still has co-owners we can name it after.
            for co_owner in row.get("co_owners") or []:
                if co_owner in users_by_id:
                    user = users_by_id[co_owner]
                    break
        metadata = user.get("metadata") or {}
        prior = known.get(roster_id)
        settings = row.get("settings") or {}
        teams[roster_id] = Team(
            roster_id=roster_id,
            owner_id=owner_id,
            display_name=(
                user.get("display_name")
                or (prior.display_name if prior else "")
                or user.get("username")
                or ""
            ),
            username=(
                user.get("username") or (prior.username if prior else "") or ""
            ),
            team_name=(
                metadata.get("team_name") or (prior.team_name if prior else "")
            ),
            players=[p for p in (row.get("players") or []) if p],
            starters=[p for p in (row.get("starters") or []) if p and p != "0"],
            reserve=[p for p in (row.get("reserve") or []) if p],
            taxi=[p for p in (row.get("taxi") or []) if p],
            wins=int(settings.get("wins") or 0),
            losses=int(settings.get("losses") or 0),
            ties=int(settings.get("ties") or 0),
            points_for=float(settings.get("fpts") or 0)
            + float(settings.get("fpts_decimal") or 0) / 100.0,
            avatar=user.get("avatar") or (prior.avatar if prior else None),
            claimed=bool(user),
        )
        if not user and teams[roster_id].players:
            nameless.append(roster_id)
    if nameless and users:
        # Only rosters that hold players are worth a warning. A league still
        # filling up has an unowned roster per empty seat, which is normal and
        # not something to alarm anyone about; an *abandoned* roster is the odd
        # one, because the carry-forward above cannot recover a name for it.
        log.warning(
            "%d of %d rosters have players but no matching Sleeper user "
            "(roster_ids %s); those teams fall back to a numeric name. Run "
            "`fantasypicker doctor <league_id>` to see what Sleeper returned.",
            len(nameless),
            len(rosters),
            ", ".join(str(r) for r in nameless),
        )
    return teams


async def refresh_teams(
    client: SleeperClient, league: LeagueContext, *, fresh: bool = False
) -> bool:
    """Re-pull rosters and users into an existing context.

    Rosters change between page loads — a waiver clears, a trade goes through,
    someone drops an injured back. Without this the app would keep answering
    from whatever the rosters looked like when the league was first connected,
    which is the sort of staleness that is invisible until it gives you bad
    advice. Returns True when anything actually changed.
    """
    rosters, users = await asyncio.gather(
        client.rosters(league.league_id, fresh=fresh),
        client.league_users(league.league_id, fresh=fresh),
    )
    if not rosters:
        return False
    if not users:
        # Rosters without users is a partial answer, not a league where nobody
        # has a name. Rebuilding from it would rename every team to "Team 4"
        # until the next successful refresh — and refresh runs before every
        # request, so one bad response would poison the whole UI.
        log.warning(
            "Sleeper returned rosters but no users for league %s; "
            "keeping the team names already loaded",
            league.league_id,
        )
    before = {rid: tuple(sorted(t.players)) for rid, t in league.teams.items()}
    league.teams = build_teams(rosters, users or [], previous=league.teams)
    league._matchup_cache.clear()
    after = {rid: tuple(sorted(t.players)) for rid, t in league.teams.items()}
    return before != after


async def load_league(
    client: SleeperClient,
    league_id: str,
    *,
    week: int | None = None,
    username: str | None = None,
    user_id: str | None = None,
) -> LeagueContext:
    """Build a :class:`LeagueContext` from the Sleeper API."""
    bundle = await client.league_bundle(league_id, week=None)
    league = bundle["league"] or {}
    if not league:
        raise LeagueNotFound(league_id)
    state = await client.state()

    teams = build_teams(bundle["rosters"], bundle["users"])
    slots, bench = _parse_slots(league.get("roster_positions"))
    season = int(league.get("season") or state.get("season") or 0)
    current_week = int(week or state.get("week") or 1) or 1

    ctx = LeagueContext(
        league_id=league_id,
        raw=league,
        scoring=ScoringRules.from_league(league),
        slots=slots,
        bench_size=bench,
        teams=teams,
        season=season,
        current_week=current_week,
    )

    if user_id is None and username:
        user = await client.user(username)
        user_id = (user or {}).get("user_id")
    if user_id:
        team = ctx.team_by_owner(user_id)
        if team is not None:
            ctx.my_roster_id = team.roster_id
    return ctx
