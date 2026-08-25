"""Async client for the public Sleeper API.

Sleeper's read API needs no authentication and no key — a league ID is enough
to read every roster in that league, which is what lets the app fill in an
opponent's starters without anyone typing them in.

Endpoints are documented at https://docs.sleeper.com. The ones used here:

    /state/nfl                                  current season + week
    /user/{name_or_id}                          user lookup
    /user/{id}/leagues/nfl/{season}             a user's leagues
    /league/{id}                                settings, scoring, roster slots
    /league/{id}/users                          display names / avatars
    /league/{id}/rosters                        every team's players
    /league/{id}/matchups/{week}                weekly pairings + starters
    /league/{id}/transactions/{week}            adds, drops, trades
    /league/{id}/drafts                         draft objects for the league
    /draft/{id}                                 draft settings + slot->roster map
    /draft/{id}/picks                           picks made so far (live)
    /players/nfl                                ~5MB player dictionary
    /players/nfl/trending/{add|drop}            waiver-wire buzz
    /projections/nfl/{type}/{season}/{week}     Sleeper's own projections
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..cache import fetch_json
from ..config import SLEEPER_API, get_settings


class SleeperClient:
    """Thin async wrapper. Every call is disk-cached with an endpoint-specific TTL."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owned = client is None
        self._settings = get_settings()

    async def __aenter__(self) -> "SleeperClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._settings.http_timeout)
            self._owned = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owned and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, ttl: float, *, allow_404: bool = True) -> Any:
        return await fetch_json(
            f"{SLEEPER_API}{path}", ttl, client=self._client, allow_404=allow_404
        )

    # -- global ------------------------------------------------------------ #

    async def state(self) -> dict:
        """``{"season": "2026", "week": 3, "season_type": "regular", ...}``"""
        return await self._get("/state/nfl", self._settings.ttl_state) or {}

    async def players(self, *, fresh: bool = False) -> dict[str, dict]:
        """The full NFL player dictionary keyed by Sleeper player_id.

        This is a ~5MB response, so it is cached hard; it is also the only place
        injury designations live, so ``fresh`` exists for the moment a user
        explicitly asks to re-check before kickoff.
        """
        ttl = 0 if fresh else self._settings.ttl_players
        return await self._get("/players/nfl", ttl) or {}

    async def trending(self, kind: str = "add", hours: int = 24, limit: int = 50) -> list[dict]:
        path = f"/players/nfl/trending/{kind}?lookback_hours={hours}&limit={limit}"
        return await self._get(path, 900) or []

    async def projections(self, season: int, week: int, season_type: str = "regular") -> Any:
        """Sleeper's own weekly projections — used as a prior, never as truth."""
        return await self._get(
            f"/projections/nfl/{season_type}/{season}/{week}", 3600
        )

    # -- users ------------------------------------------------------------- #

    async def user(self, name_or_id: str) -> dict | None:
        return await self._get(f"/user/{name_or_id}", 24 * 3600)

    async def user_leagues(self, user_id: str, season: int | str) -> list[dict]:
        return await self._get(f"/user/{user_id}/leagues/nfl/{season}", 3600) or []

    # -- league ------------------------------------------------------------ #

    async def league(self, league_id: str) -> dict | None:
        return await self._get(f"/league/{league_id}", self._settings.ttl_league)

    async def league_users(self, league_id: str, *, fresh: bool = False) -> list[dict]:
        ttl = 0 if fresh else self._settings.ttl_league
        return await self._get(f"/league/{league_id}/users", ttl) or []

    async def rosters(self, league_id: str, *, fresh: bool = False) -> list[dict]:
        """Every team's players. ``fresh`` skips the disk cache entirely, which
        is what an explicit "refresh" from the UI has to do — otherwise a click
        can be answered from a copy up to a minute old."""
        ttl = 0 if fresh else self._settings.ttl_matchups
        return await self._get(f"/league/{league_id}/rosters", ttl) or []

    async def matchups(self, league_id: str, week: int) -> list[dict]:
        return (
            await self._get(
                f"/league/{league_id}/matchups/{week}", self._settings.ttl_matchups
            )
            or []
        )

    async def transactions(self, league_id: str, week: int) -> list[dict]:
        return await self._get(f"/league/{league_id}/transactions/{week}", 300) or []

    async def traded_picks(self, league_id: str) -> list[dict]:
        return await self._get(f"/league/{league_id}/traded_picks", 3600) or []

    # -- draft ------------------------------------------------------------- #

    async def league_drafts(self, league_id: str) -> list[dict]:
        return await self._get(f"/league/{league_id}/drafts", 300) or []

    async def draft(self, draft_id: str) -> dict | None:
        return await self._get(f"/draft/{draft_id}", 60)

    async def draft_picks(self, draft_id: str) -> list[dict]:
        """Picks made so far. Short TTL — this is polled during a live draft."""
        return await self._get(f"/draft/{draft_id}/picks", self._settings.ttl_draft) or []

    # -- composites -------------------------------------------------------- #

    async def league_bundle(self, league_id: str, week: int | None = None) -> dict:
        """Fetch everything the weekly view needs in one round of concurrency."""
        league, users, rosters = await asyncio.gather(
            self.league(league_id),
            self.league_users(league_id),
            self.rosters(league_id),
        )
        matchups = await self.matchups(league_id, week) if week else []
        return {
            "league": league,
            "users": users,
            "rosters": rosters,
            "matchups": matchups,
        }
