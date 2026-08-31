"""Async client for ESPN's fantasy football read API.

ESPN exposes one endpoint per league and selects what comes back with repeated
``view`` parameters:

    /seasons/{season}/segments/0/leagues/{id}?view=mSettings   scoring, slots
                                             ?view=mTeam       teams + members
                                             ?view=mRoster     every roster
                                             ?view=mMatchup    the full schedule
                                             ?view=mDraftDetail draft picks

A **public** league answers all of these with no credentials, which is what
makes an opponent's roster readable without anyone typing it in. A **private**
league returns 401, and needs two cookies from a browser already signed in to
ESPN: ``espn_s2`` and ``SWID``. They are session credentials for the user's own
ESPN account — they are sent to ESPN and nowhere else, kept in a file only the
user can read, and never logged.

Seasons before the current one live under a different path (``leagueHistory``)
and answer with a list rather than an object; :meth:`EspnClient.league` hides
that difference.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from ..cache import AuthError, fetch_json
from ..config import ESPN_API, get_settings

log = logging.getLogger(__name__)


class EspnLeagueNotFound(ValueError):
    """ESPN has no league with this ID for this season."""

    def __init__(self, league_id: str, season: int) -> None:
        self.league_id = league_id
        self.season = season
        super().__init__(
            f"ESPN has no league {league_id!r} in the {season} season. The ID is "
            "the number in your league's URL — fantasy.espn.com/football/league"
            "?leagueId=<this-number>. If the league exists but is set to "
            "private, it will look missing until you add your espn_s2 and SWID "
            "cookies."
        )


class EspnAuthRequired(ValueError):
    """The league is private and the cookies are missing, wrong, or expired."""

    def __init__(self, league_id: str, *, had_cookies: bool) -> None:
        self.league_id = league_id
        self.had_cookies = had_cookies
        detail = (
            "The cookies supplied were rejected — ESPN expires them "
            "periodically, so they most likely need copying again."
            if had_cookies
            else "This league is private, so it needs the espn_s2 and SWID "
            "cookies from a browser signed in to ESPN."
        )
        super().__init__(
            f"ESPN refused to show league {league_id}. {detail} "
            "In your browser: sign in to fantasy.espn.com, open developer tools "
            "(F12), go to Application (or Storage) -> Cookies -> espn.com, and "
            "copy the values of espn_s2 and SWID."
        )


class EspnClient:
    """Thin async wrapper over one league's views, disk-cached like Sleeper's."""

    def __init__(
        self,
        *,
        espn_s2: str | None = None,
        swid: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = get_settings()
        self._espn_s2 = espn_s2 or None
        self._swid = swid or None
        self._client = client
        self._owned = client is None

    @property
    def has_cookies(self) -> bool:
        return bool(self._espn_s2 and self._swid)

    def _cookies(self) -> dict[str, str]:
        if not self.has_cookies:
            return {}
        swid = self._swid or ""
        # ESPN writes SWID wrapped in braces; accept it either way.
        if not swid.startswith("{"):
            swid = "{" + swid.strip("{}") + "}"
        return {"espn_s2": self._espn_s2 or "", "SWID": swid}

    async def __aenter__(self) -> "EspnClient":
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._settings.http_timeout, cookies=self._cookies()
            )
            self._owned = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owned and self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- core ---------------------------------------------------------------- #

    def _url(self, league_id: str, season: int, views: tuple[str, ...], **params) -> str:
        """Build a view URL.

        Past seasons are served from ``leagueHistory`` with the season as a
        query parameter instead of a path segment — the same data, a different
        shape, and the source of most "my league disappeared" confusion.
        """
        current = _current_season()
        if season >= current:
            base = f"{ESPN_API}/seasons/{season}/segments/0/leagues/{league_id}"
            query: list[tuple[str, str]] = []
        else:
            base = f"{ESPN_API}/leagueHistory/{league_id}"
            query = [("seasonId", str(season))]
        query.extend(("view", v) for v in views)
        query.extend((k, str(v)) for k, v in params.items() if v is not None)
        encoded = "&".join(f"{k}={v}" for k, v in query)
        return f"{base}?{encoded}" if encoded else base

    async def _get(
        self,
        league_id: str,
        season: int,
        views: tuple[str, ...],
        ttl: float,
        *,
        fresh: bool = False,
        **params,
    ) -> Any:
        url = self._url(league_id, season, views, **params)
        client = self._client
        if client is None:
            # Used without ``async with``; make a one-shot client so the
            # cookies still travel with the request.
            async with httpx.AsyncClient(
                timeout=self._settings.http_timeout, cookies=self._cookies()
            ) as temp:
                return await self._fetch(temp, url, league_id, 0 if fresh else ttl)
        return await self._fetch(client, url, league_id, 0 if fresh else ttl)

    async def _fetch(
        self, client: httpx.AsyncClient, url: str, league_id: str, ttl: float
    ) -> Any:
        try:
            payload = await fetch_json(url, ttl, client=client, allow_404=True)
        except AuthError as exc:
            raise EspnAuthRequired(league_id, had_cookies=self.has_cookies) from exc
        # A past-season request answers with a single-element list.
        if isinstance(payload, list):
            return payload[0] if payload else None
        return payload

    # -- league views -------------------------------------------------------- #

    async def league(
        self, league_id: str, season: int, *, fresh: bool = False
    ) -> dict | None:
        """Settings, roster slots, scoring, the member list, and draft status.

        ``mDraftDetail`` rides along because whether the league has drafted
        changes how an empty roster should be read: before the draft every
        team legitimately has nobody, and that is worth saying rather than
        leaving it to look like a failure to load.
        """
        return await self._get(
            league_id,
            season,
            ("mSettings", "mTeam", "mDraftDetail"),
            self._settings.ttl_league,
            fresh=fresh,
        )

    async def rosters(
        self, league_id: str, season: int, week: int | None = None, *, fresh: bool = False
    ) -> dict | None:
        """Every team with its roster, as of ``week`` when given."""
        return await self._get(
            league_id,
            season,
            ("mTeam", "mRoster"),
            self._settings.ttl_matchups,
            fresh=fresh,
            scoringPeriodId=week,
        )

    async def schedule(
        self, league_id: str, season: int, *, fresh: bool = False
    ) -> dict | None:
        """The season's matchup schedule."""
        return await self._get(
            league_id,
            season,
            ("mMatchup",),
            self._settings.ttl_league,
            fresh=fresh,
        )

    async def matchup_scores(
        self, league_id: str, season: int, week: int, *, fresh: bool = False
    ) -> dict | None:
        """Live scores and per-week lineups for one scoring period."""
        return await self._get(
            league_id,
            season,
            ("mMatchupScore", "mRoster", "mTeam"),
            self._settings.ttl_matchups,
            fresh=fresh,
            scoringPeriodId=week,
        )

    async def draft(
        self, league_id: str, season: int, *, fresh: bool = False
    ) -> dict | None:
        """Draft settings and every pick made so far."""
        return await self._get(
            league_id,
            season,
            ("mDraftDetail", "mTeam"),
            self._settings.ttl_draft,
            fresh=fresh,
        )

    async def bundle(
        self, league_id: str, season: int, week: int | None = None
    ) -> dict[str, Any]:
        """Everything needed to build a league context, fetched concurrently."""
        league, rosters, schedule = await asyncio.gather(
            self.league(league_id, season),
            self.rosters(league_id, season, week),
            self.schedule(league_id, season),
        )
        return {"league": league, "rosters": rosters, "schedule": schedule}


def _current_season() -> int:
    """The season ESPN considers current.

    A new season's data appears on ESPN well before September, and the fantasy
    year rolls over in the spring, so anything from March onward belongs to that
    calendar year's season.
    """
    from datetime import date

    today = date.today()
    return today.year if today.month >= 3 else today.year - 1
