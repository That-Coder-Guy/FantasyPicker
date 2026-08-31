"""Where a league is read from.

Two platforms host leagues this app can read, and they disagree about almost
every detail of how. Rather than branch on the platform at each of the dozen
places the service asks a question, each platform provides a small adapter
answering the same four questions in the same vocabulary:

* re-read the rosters,
* what are this week's pairings,
* what has been drafted so far,
* who picks where.

Everything else is genuinely shared. Player identity, injury designations and
waiver-wire buzz all come from Sleeper's public player endpoints, which are a
league-independent database of the NFL — they answer the same for an ESPN
league, because the question ("is this receiver questionable?") has nothing to
do with where anybody's league is hosted.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .credentials import EspnCredentials
from .espn import league as espn_league
from .espn.client import EspnClient
from .espn.ids import dst_team_from_player_id, is_unmade_pick
from .sleeper.client import SleeperClient
from .sleeper.league import LeagueContext, refresh_teams

log = logging.getLogger(__name__)


class LeagueSource(Protocol):
    """What the service needs from whichever platform hosts the league."""

    platform: str

    async def matchups(self, league_id: str, week: int) -> list[dict]:
        """This week's rows, in the shape :class:`LeagueContext` consumes."""

    async def refresh(self, league: LeagueContext, *, fresh: bool = False) -> bool:
        """Re-pull rosters; True when anything moved."""

    async def draft(self, league: LeagueContext) -> tuple[dict | None, list[dict]]:
        """The live draft object and the picks made so far, Sleeper-shaped."""

    async def draft_order(self, league: LeagueContext) -> dict[str, int]:
        """owner id -> draft slot."""

    async def draft_rosters(self, league: LeagueContext) -> dict[int, list[str]]:
        """roster_id -> the players that roster has drafted so far."""


class SleeperSource:
    """Reads a Sleeper-hosted league."""

    platform = "sleeper"

    async def matchups(self, league_id: str, week: int) -> list[dict]:
        async with SleeperClient() as client:
            return await client.matchups(league_id, week)

    async def refresh(self, league: LeagueContext, *, fresh: bool = False) -> bool:
        async with SleeperClient() as client:
            return await refresh_teams(client, league, fresh=fresh)

    async def draft(self, league: LeagueContext) -> tuple[dict | None, list[dict]]:
        async with SleeperClient() as client:
            drafts = await client.league_drafts(league.league_id)
            if not drafts:
                return None, []
            # Sleeper lists newest first; the live one is the one not complete.
            draft_obj = next(
                (d for d in drafts if d.get("status") != "complete"), drafts[0]
            )
            picks = await client.draft_picks(str(draft_obj.get("draft_id")))
            return draft_obj, picks

    async def draft_rosters(self, league: LeagueContext) -> dict[int, list[str]]:
        _, picks = await self.draft(league)
        return picks_by_roster(picks)

    async def draft_order(self, league: LeagueContext) -> dict[str, int]:
        async with SleeperClient() as client:
            drafts = await client.league_drafts(league.league_id)
        if not drafts:
            return {}
        draft = next((d for d in drafts if d.get("status") != "complete"), drafts[0])
        order = draft.get("draft_order") or {}
        if not isinstance(order, dict):
            return {}
        out: dict[str, int] = {}
        for user_id, slot in order.items():
            try:
                out[str(user_id)] = int(slot)
            except (TypeError, ValueError):
                continue
        return out


class EspnSource:
    """Reads an ESPN-hosted league.

    ESPN needs a season alongside the league ID — the same league is a different
    resource each year — and, for a private league, the user's session cookies.
    """

    platform = "espn"

    def __init__(self, season: int, credentials: EspnCredentials | None = None) -> None:
        self.season = int(season)
        self.credentials = credentials

    def client(self) -> EspnClient:
        creds = self.credentials
        return EspnClient(
            espn_s2=creds.espn_s2 if creds else None,
            swid=creds.swid if creds else None,
        )

    async def matchups(self, league_id: str, week: int) -> list[dict]:
        async with self.client() as client:
            payload = await client.schedule(league_id, self.season)
            rosters = await client.rosters(league_id, self.season, week)
        teams, _ = espn_league.build_teams(
            rosters or {}, _crosswalk(), previous=None
        )
        return espn_league.matchup_rows(payload, teams, week)

    async def refresh(self, league: LeagueContext, *, fresh: bool = False) -> bool:
        async with self.client() as client:
            # ESPN advances the scoring period on its own clock; a week frozen
            # at connect time would keep every later request asking about the
            # old one. The league call is cached for minutes, so this is cheap.
            meta = await client.league(league.league_id, self.season)
            if meta:
                league.current_week = espn_league.current_week(meta)
            payload = await client.rosters(
                league.league_id, self.season, league.current_week, fresh=fresh
            )
        if not payload or not payload.get("teams"):
            # An answer with no teams is ESPN reporting a problem, not a league
            # with nobody in it — say so rather than silently changing nothing.
            log.warning(
                "ESPN roster refresh for league %s returned no teams: %s",
                league.league_id,
                str(payload)[:300] if payload else "(empty response)",
            )
            return False
        before = {rid: tuple(sorted(t.players)) for rid, t in league.teams.items()}
        teams, unresolved = espn_league.build_teams(
            payload, _crosswalk(), previous=league.teams
        )
        if not teams:
            return False
        league.teams = teams
        league._matchup_cache.clear()
        if unresolved:
            log.info(
                "%d rostered players still unmatched after refresh", len(unresolved)
            )
        after = {rid: tuple(sorted(t.players)) for rid, t in league.teams.items()}
        return before != after

    async def draft(self, league: LeagueContext) -> tuple[dict | None, list[dict]]:
        async with self.client() as client:
            payload = await client.draft(league.league_id, self.season)
        return _espn_draft(payload, league)

    async def draft_rosters(self, league: LeagueContext) -> dict[int, list[str]]:
        async with self.client() as client:
            payload = await client.draft(league.league_id, self.season)
        _, picks = _espn_draft(payload, league)
        return picks_by_roster(picks)

    async def draft_order(self, league: LeagueContext) -> dict[str, int]:
        async with self.client() as client:
            payload = await client.draft(league.league_id, self.season)
        draft_obj, _ = _espn_draft(payload, league)
        order = (draft_obj or {}).get("draft_order") or {}
        return {str(k): int(v) for k, v in order.items()}


def picks_by_roster(picks: list[dict]) -> dict[int, list[str]]:
    """Group a Sleeper-shaped pick feed into rosters, in pick order."""
    out: dict[int, list[str]] = {}
    for pick in picks or []:
        try:
            roster_id = int(pick.get("roster_id"))
        except (TypeError, ValueError):
            continue
        player = str(pick.get("player_id") or "")
        if player:
            out.setdefault(roster_id, []).append(player)
    return out


def apply_draft_rosters(
    league: LeagueContext, by_roster: dict[int, list[str]]
) -> bool:
    """Fill empty rosters from the draft feed. True when anything changed.

    Neither platform moves drafted players onto a roster until the draft
    finishes — espn.com shows the same empty teams this app did — but both
    publish every pick the moment it happens. During a draft the pick feed is
    therefore the only live account of who owns whom, and it is what makes
    "what is left at my position" answerable while it matters.

    Only rosters that came back empty are filled, so a completed draft's real
    rosters (which also carry waiver moves and trades) always win.
    """
    changed = False
    for roster_id, drafted in by_roster.items():
        team = league.teams.get(roster_id)
        if team is None or team.players or not drafted:
            continue
        team.players = list(drafted)
        # Nobody sets a lineup mid-draft; the League page solves a best-possible
        # one from the roster, which is the useful view during a draft anyway.
        team.starters = []
        changed = True
    return changed


def _crosswalk():
    from .data.crosswalk import load_crosswalk

    return load_crosswalk()


def _slot_by_team(payload: dict | None) -> dict[int, int]:
    """ESPN ``pickOrder`` -> teamId -> 1-based draft slot."""
    detail = (payload or {}).get("draftDetail") or {}
    settings = ((payload or {}).get("settings") or {}).get("draftSettings") or {}
    order = detail.get("pickOrder") or settings.get("pickOrder") or []
    out: dict[int, int] = {}
    for index, team_id in enumerate(order, start=1):
        try:
            out[int(team_id)] = index
        except (TypeError, ValueError):
            continue
    return out


def _espn_draft(
    payload: dict | None, league: LeagueContext
) -> tuple[dict | None, list[dict]]:
    """ESPN's draft detail, translated into the Sleeper shapes the engine reads.

    The draft engine works entirely in draft slots, so the translation only has
    to get three things right: how many teams and rounds, which slot each pick
    belongs to, and which slot is the user's.
    """
    if not payload:
        return None, []
    detail = payload.get("draftDetail") or {}
    settings = payload.get("settings") or {}
    draft_settings = settings.get("draftSettings") or {}
    roster_settings = settings.get("rosterSettings") or {}

    teams = int(settings.get("size") or len(league.teams) or 12)
    slot_by_team = _slot_by_team(payload)

    rounds = 0
    for raw in (roster_settings.get("lineupSlotCounts") or {}).values():
        try:
            rounds += int(raw)
        except (TypeError, ValueError):
            continue
    rounds = rounds or league.roster_size or 15

    kind = str(draft_settings.get("type") or "SNAKE").lower()
    if kind not in {"snake", "auction"}:
        kind = "snake"

    crosswalk = _crosswalk()
    picks: list[dict] = []
    for pick in detail.get("picks") or []:
        try:
            team_id = int(pick.get("teamId"))
        except (TypeError, ValueError):
            continue
        if is_unmade_pick(pick.get("playerId")):
            continue  # a slot the draft has not reached yet
        sleeper_id = crosswalk.from_espn(pick.get("playerId")) or (
            dst_team_from_player_id(pick.get("playerId"))
        )
        if not sleeper_id:
            log.info(
                "draft pick #%s (playerId %s) could not be matched to a player",
                pick.get("overallPickNumber"),
                pick.get("playerId"),
            )
            continue
        try:
            pick_no = int(pick.get("overallPickNumber") or len(picks) + 1)
        except (TypeError, ValueError):
            pick_no = len(picks) + 1
        picks.append(
            {
                "player_id": str(sleeper_id),
                "pick_no": pick_no,
                "draft_slot": slot_by_team.get(team_id, 0),
                # Kept so a live draft can be turned back into rosters; the
                # draft engine itself works in slots and ignores this.
                "roster_id": team_id,
            }
        )

    # The engine keys "my slot" off an owner id, which on ESPN is the SWID.
    draft_order: dict[str, int] = {}
    for roster_id, team in league.teams.items():
        slot = slot_by_team.get(int(roster_id))
        if slot and team.owner_id:
            draft_order[str(team.owner_id)] = slot

    draft_obj = {
        "draft_id": str(detail.get("drafted") and league.league_id or league.league_id),
        "type": kind,
        "status": "complete" if detail.get("drafted") else "in_progress",
        "settings": {"teams": teams, "rounds": rounds, "reversal_round": 0},
        "draft_order": draft_order,
    }
    return draft_obj, picks


def source_for(
    platform: str, *, season: int | None = None, credentials: EspnCredentials | None = None
) -> LeagueSource:
    if str(platform).lower() == "espn":
        if season is None:
            raise ValueError("an ESPN league needs a season")
        return EspnSource(season, credentials)
    return SleeperSource()
