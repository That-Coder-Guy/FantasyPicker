"""ESPN's own projected points — the number everyone else in the league sees.

This app's projections are better than ESPN's. That is the point of it. But it
is also the problem when you go to make a trade, because the manager on the
other end is not looking at them. They are looking at the PROJ column on
espn.com, and a deal that is a clear win on our numbers can read as an insult on
theirs. A trade engine that only knows our numbers will keep proposing deals
nobody accepts.

So ESPN's projections are pulled as a second, separate currency. They are never
mixed into the model and never used to decide whether a roster improved — they
answer one question only: *what does this look like to them?*

Where the numbers live
----------------------
Every ESPN player object carries a ``stats`` list, and each entry is tagged:

* ``statSourceId`` — 0 for what actually happened, **1 for a projection**.
* ``statSplitTypeId`` — 0 for a season total, 1 for a single scoring period.
* ``appliedTotal`` / ``appliedAverage`` — points under *this league's* scoring
  rules, which is what makes them comparable at all. ESPN applies the league's
  own settings before serving them, so a PPR league's numbers already are PPR.

The same shape appears on roster entries and in the player pool, so both paths
read through one parser.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from ..data.crosswalk import Crosswalk
from .ids import position_of, team_of
from .league import entry_name, entry_player

log = logging.getLogger(__name__)

#: ``statSourceId`` for a projection rather than a result.
PROJECTED = 1
#: ``statSplitTypeId`` for a whole-season line rather than one week.
SEASON_SPLIT = 0

#: Games in an NFL regular season, used only to turn a season total back into a
#: per-game rate when ESPN omits its own average.
SEASON_GAMES = 17


class MarketPoints:
    """One player's published projection, in both the forms we need.

    ``total`` is the season number as ESPN prints it; ``per_game`` is what it
    works out to per week. Keeping both means a mid-season roster can be priced
    on the games that are actually left without pretending the published season
    total is a rest-of-season figure.
    """

    __slots__ = ("total", "per_game")

    def __init__(self, total: float, per_game: float) -> None:
        self.total = float(total)
        self.per_game = float(per_game)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MarketPoints(total={self.total:.1f}, per_game={self.per_game:.2f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MarketPoints):
            return NotImplemented
        return (self.total, self.per_game) == (other.total, other.per_game)


def _season_projection(player: Mapping, season: int) -> MarketPoints | None:
    """The season-long projection line from one ESPN player object.

    ESPN ships several stat lines per player — last season's actuals, this
    season to date, a projection, sometimes one per week. Only the projected
    season split is wanted, and it has to be for *this* season: a September
    payload still carries last year's totals, and reading those would price
    every roster on a year that has already been played.
    """
    best: MarketPoints | None = None
    for stat in player.get("stats") or []:
        if not isinstance(stat, Mapping):
            continue
        if _as_int(stat.get("statSourceId")) != PROJECTED:
            continue
        if _as_int(stat.get("statSplitTypeId")) != SEASON_SPLIT:
            continue
        stat_season = _as_int(stat.get("seasonId"))
        if stat_season is not None and stat_season != int(season):
            continue

        total = _as_float(stat.get("appliedTotal"))
        average = _as_float(stat.get("appliedAverage"))
        if total is None and average is None:
            continue
        if total is None:
            total = average * SEASON_GAMES  # type: ignore[operator]
        if average is None or average <= 0:
            average = total / SEASON_GAMES
        # Several matching lines is unusual but possible; the fullest one wins.
        if best is None or total > best.total:
            best = MarketPoints(total, average)
    return best


def from_players(
    players: Iterable[Mapping],
    crosswalk: Crosswalk,
    *,
    season: int,
) -> dict[str, MarketPoints]:
    """Projections keyed by Sleeper ID, from bare ESPN player objects."""
    out: dict[str, MarketPoints] = {}
    for player in players:
        if not isinstance(player, Mapping):
            continue
        points = _season_projection(player, season)
        if points is None:
            continue
        espn_id = player.get("id")
        if espn_id is None:
            continue
        sleeper_id = crosswalk.from_espn(
            espn_id,
            name=entry_name(player),
            position=position_of(player.get("defaultPositionId")) or "",
            team=team_of(player.get("proTeamId")),
        )
        if not sleeper_id:
            continue
        out[str(sleeper_id)] = points
    return out


def from_rosters(
    payload: Mapping | None, crosswalk: Crosswalk, *, season: int
) -> dict[str, MarketPoints]:
    """Projections for every rostered player, from an ``mRoster`` response.

    This is free: the rosters are fetched anyway, and the projections ride
    along on the same player objects.
    """
    if not payload:
        return {}
    players: list[Mapping] = []
    for team in payload.get("teams") or []:
        if not isinstance(team, Mapping):
            continue
        for entry in (team.get("roster") or {}).get("entries") or []:
            if isinstance(entry, Mapping):
                players.append(entry_player(entry))
    return from_players(players, crosswalk, season=season)


def from_player_pool(
    payload: Mapping | None, crosswalk: Crosswalk, *, season: int
) -> dict[str, MarketPoints]:
    """Projections from a ``kona_player_info`` response.

    The pool is how free agents get priced. ESPN nests the player one level
    down here, and older responses vary, so both shapes are accepted.
    """
    if not payload:
        return {}
    players: list[Mapping] = []
    for entry in payload.get("players") or []:
        if not isinstance(entry, Mapping):
            continue
        player = entry.get("player")
        players.append(player if isinstance(player, Mapping) else entry)
    return from_players(players, crosswalk, season=season)


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    # ESPN uses 0.0 for "no projection" as well as for "projected to score
    # nothing". Treating them the same is right here: neither is a number worth
    # showing a trade partner.
    return number if number == number else None  # NaN guard
