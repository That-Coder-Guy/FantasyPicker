"""Translate an ESPN league into the objects the engines already understand.

Nothing downstream of this module knows ESPN exists. The projection model, the
correlated simulator, the lineup solver, the draft engine and the League page
all consume :class:`~fantasypicker.sleeper.league.LeagueContext`, so the job
here is translation into that vocabulary:

* ESPN ``lineupSlotCounts`` become :class:`RosterSlot` objects.
* ESPN teams and members become :class:`Team` objects, with the same
  team-name/display-name/username fallbacks.
* ESPN player IDs become Sleeper IDs, which is the key every projection,
  ranking and correlation is stored under.
* ESPN's schedule becomes Sleeper-shaped matchup rows, so ``matchup_for`` works
  unchanged.

The one genuinely lossy step is player identity: a player ESPN rosters but the
crosswalk cannot place has no projection. Those are counted and reported rather
than dropped silently, because a missing star is the difference between good
advice and nonsense.
"""

from __future__ import annotations

import logging

from ..data.crosswalk import Crosswalk, load_crosswalk, normalize_team
from ..sleeper.league import BENCH_SLOTS, LeagueContext, RosterSlot, Team
from .client import EspnClient, EspnLeagueNotFound
from .ids import UNPROJECTABLE_SLOTS, position_of, slot_of, team_of
from .scoring import scoring_from_espn

log = logging.getLogger(__name__)

#: The order ESPN's own lineup card uses, so the League page reads naturally
#: rather than in numeric slot order (which would put FLEX after the kicker).
SLOT_DISPLAY_ORDER = [
    "QB", "RB", "WR", "TE", "WRRB_FLEX", "REC_FLEX", "FLEX", "SUPER_FLEX",
    "DL", "LB", "DB", "IDP_FLEX", "DEF", "K", "P", "HC",
]

#: ESPN injury wording -> the vocabulary
#: :class:`~fantasypicker.model.availability.AvailabilityModel` expects.
INJURY_STATUS = {
    "ACTIVE": "",
    "NORMAL": "",
    "QUESTIONABLE": "QUESTIONABLE",
    "DOUBTFUL": "DOUBTFUL",
    "OUT": "OUT",
    "INJURY_RESERVE": "IR",
    "SUSPENSION": "SUS",
    "PROBABLE": "",
    "DAY_TO_DAY": "QUESTIONABLE",
}


def parse_slots(roster_settings: dict | None) -> tuple[list[RosterSlot], int]:
    """ESPN ``lineupSlotCounts`` -> starting slots plus a bench size."""
    counts = (roster_settings or {}).get("lineupSlotCounts") or {}
    named: dict[str, int] = {}
    bench = 0
    unsupported: list[str] = []
    for raw_slot, raw_count in counts.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        name = slot_of(raw_slot)
        if name is None:
            unsupported.append(str(raw_slot))
            continue
        if name in BENCH_SLOTS:
            bench += count
            continue
        if name in UNPROJECTABLE_SLOTS:
            # Punters and head coaches are real ESPN slots with no box-score
            # projection behind them. Counting them as bench keeps the roster
            # size right without inventing a lineup decision we cannot make.
            bench += count
            unsupported.append(name)
            continue
        named[name] = named.get(name, 0) + count

    if unsupported:
        log.warning(
            "ESPN roster slots this app cannot project: %s", ", ".join(unsupported)
        )

    slots: list[RosterSlot] = []
    for name in SLOT_DISPLAY_ORDER:
        for _ in range(named.pop(name, 0)):
            slots.append(RosterSlot(index=len(slots), name=name))
    for name, count in sorted(named.items()):  # anything not in the display order
        for _ in range(count):
            slots.append(RosterSlot(index=len(slots), name=name))

    if not slots:
        default = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "DEF", "K"]
        slots = [RosterSlot(i, name) for i, name in enumerate(default)]
        bench = bench or 7
    return slots, bench


def _team_names(team: dict, member: dict) -> tuple[str, str, str]:
    """(team_name, display_name, username) for one ESPN team."""
    name = str(team.get("name") or "").strip()
    if not name:
        # Older leagues split the name in two.
        parts = [
            str(team.get("location") or "").strip(),
            str(team.get("nickname") or "").strip(),
        ]
        name = " ".join(p for p in parts if p).strip()
    display = str(member.get("displayName") or "").strip()
    person = " ".join(
        p
        for p in (
            str(member.get("firstName") or "").strip(),
            str(member.get("lastName") or "").strip(),
        )
        if p
    )
    return name, person or display, display


def build_teams(
    payload: dict | None,
    crosswalk: Crosswalk,
    *,
    previous: dict[int, Team] | None = None,
) -> tuple[dict[int, Team], list[dict]]:
    """ESPN teams -> :class:`Team` objects, plus any players we could not place.

    The second return value lists unresolved players so the caller can report
    them; an ESPN roster is only as useful as the fraction of it we can project.
    """
    teams: dict[int, Team] = {}
    unresolved: list[dict] = []
    known = previous or {}
    members = {
        str(m.get("id")): m for m in (payload or {}).get("members") or [] if m.get("id")
    }

    for row in (payload or {}).get("teams") or []:
        try:
            roster_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        owners = [str(o) for o in (row.get("owners") or []) if o]
        owner_id = owners[0] if owners else None
        member = members.get(str(owner_id)) if owner_id else None
        member = member or {}
        team_name, display_name, username = _team_names(row, member)
        prior = known.get(roster_id)

        players: list[str] = []
        starters: list[str] = []
        reserve: list[str] = []
        for entry in ((row.get("roster") or {}).get("entries") or []):
            resolved = _resolve_entry(entry, crosswalk)
            if resolved is None:
                info = _entry_description(entry)
                if info:
                    unresolved.append(info)
                continue
            sleeper_id, slot_name = resolved
            players.append(sleeper_id)
            if slot_name == "IR":
                reserve.append(sleeper_id)
            elif slot_name not in BENCH_SLOTS:
                starters.append(sleeper_id)

        record = ((row.get("record") or {}).get("overall")) or {}
        teams[roster_id] = Team(
            roster_id=roster_id,
            owner_id=owner_id,
            team_name=team_name or (prior.team_name if prior else ""),
            display_name=display_name or (prior.display_name if prior else ""),
            username=username or (prior.username if prior else ""),
            players=players,
            starters=starters,
            reserve=reserve,
            wins=int(record.get("wins") or 0),
            losses=int(record.get("losses") or 0),
            ties=int(record.get("ties") or 0),
            points_for=float(record.get("pointsFor") or 0.0),
            avatar=str(row.get("logo") or "") or (prior.avatar if prior else None),
            claimed=bool(member) or bool(team_name),
        )
    return teams, unresolved


def _resolve_entry(entry: dict, crosswalk: Crosswalk) -> tuple[str, str] | None:
    player = ((entry.get("playerPoolEntry") or {}).get("player")) or {}
    if not player:
        return None
    position = position_of(player.get("defaultPositionId"))
    team = team_of(player.get("proTeamId"))
    sleeper_id = crosswalk.from_espn(
        player.get("id"),
        name=str(player.get("fullName") or ""),
        position=position or "",
        team=team,
    )
    if not sleeper_id:
        return None
    slot_name = slot_of(entry.get("lineupSlotId")) or "BN"
    return str(sleeper_id), slot_name


def _entry_description(entry: dict) -> dict | None:
    player = ((entry.get("playerPoolEntry") or {}).get("player")) or {}
    name = str(player.get("fullName") or "").strip()
    if not name:
        return None
    return {
        "espn_id": player.get("id"),
        "name": name,
        "position": position_of(player.get("defaultPositionId")),
        "team": team_of(player.get("proTeamId")),
    }


def injury_map(payload: dict | None) -> dict[str, str]:
    """sleeper_id -> normalised injury designation, from rostered players.

    ESPN carries a designation on the player object itself, which spares a
    second source for the one thing that moves all week.
    """
    crosswalk = load_crosswalk()
    out: dict[str, str] = {}
    for row in (payload or {}).get("teams") or []:
        for entry in ((row.get("roster") or {}).get("entries") or []):
            player = ((entry.get("playerPoolEntry") or {}).get("player")) or {}
            position = position_of(player.get("defaultPositionId"))
            sleeper_id = crosswalk.from_espn(
                player.get("id"),
                name=str(player.get("fullName") or ""),
                position=position or "",
                team=team_of(player.get("proTeamId")),
            )
            if not sleeper_id:
                continue
            raw = str(player.get("injuryStatus") or "").upper()
            out[str(sleeper_id)] = INJURY_STATUS.get(raw, raw if raw else "")
    return out


def matchup_rows(
    schedule_payload: dict | None, teams: dict[int, Team], week: int
) -> list[dict]:
    """ESPN's schedule -> the matchup row shape ``LeagueContext`` consumes.

    Sleeper pairs teams by a shared ``matchup_id``; ESPN gives each pairing an
    id with explicit home and away sides. Emitting Sleeper's shape means
    ``matchup_for`` and the League page need no ESPN-specific branch.
    """
    rows: list[dict] = []
    for game in (schedule_payload or {}).get("schedule") or []:
        try:
            period = int(game.get("matchupPeriodId"))
        except (TypeError, ValueError):
            continue
        if period != int(week):
            continue
        matchup_id = game.get("id", period)
        for side in ("home", "away"):
            entry = game.get(side)
            if not entry:
                continue
            try:
                roster_id = int(entry.get("teamId"))
            except (TypeError, ValueError):
                continue
            team = teams.get(roster_id)
            rows.append(
                {
                    "roster_id": roster_id,
                    "matchup_id": matchup_id,
                    "starters": list(team.starters) if team else [],
                    "players": list(team.players) if team else [],
                    "points": float(entry.get("totalPoints") or 0.0),
                }
            )
    return rows


def _synthetic_raw(payload: dict, slots: list[RosterSlot], bench: int) -> dict:
    """A Sleeper-shaped settings dict, so ``LeagueContext`` needs no changes."""
    settings = (payload or {}).get("settings") or {}
    return {
        "name": settings.get("name") or payload.get("name") or "",
        "total_rosters": int(settings.get("size") or len(payload.get("teams") or []) or 12),
        "roster_positions": [s.name for s in slots] + ["BN"] * bench,
        "settings": {"type": 0},
        "platform": "espn",
    }


def current_week(payload: dict | None) -> int:
    status = (payload or {}).get("status") or {}
    for key in ("currentMatchupPeriod", "latestScoringPeriod", "finalScoringPeriod"):
        try:
            value = int(status.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 1


async def load_league(
    client: EspnClient,
    league_id: str,
    season: int,
    *,
    week: int | None = None,
) -> tuple[LeagueContext, list[dict]]:
    """Build a :class:`LeagueContext` for an ESPN league.

    Returns the context plus the list of roster entries whose identity could not
    be resolved, which the caller is expected to surface rather than swallow.
    """
    payload = await client.league(league_id, season)
    if not payload:
        raise EspnLeagueNotFound(league_id, season)

    settings = payload.get("settings") or {}
    slots, bench = parse_slots(settings.get("rosterSettings"))
    scoring = scoring_from_espn(settings.get("scoringSettings"))

    rosters = await client.rosters(league_id, season, week)
    crosswalk = load_crosswalk()
    teams, unresolved = build_teams(rosters or payload, crosswalk)

    league = LeagueContext(
        league_id=str(league_id),
        raw=_synthetic_raw(payload, slots, bench),
        scoring=scoring,
        slots=slots,
        bench_size=bench,
        teams=teams,
        season=int(season),
        current_week=current_week(payload),
    )
    if unresolved:
        log.warning(
            "%d rostered ESPN players could not be matched to a player ID: %s",
            len(unresolved),
            ", ".join(sorted({str(u["name"]) for u in unresolved})[:8]),
        )
    return league, unresolved


def find_my_roster_id(payload: dict | None, swid: str | None) -> int | None:
    """Which team belongs to the signed-in user, from the SWID cookie.

    ESPN identifies a member by the same value the SWID cookie carries, so a
    private league can name the user's own team without asking.
    """
    if not swid:
        return None
    wanted = "{" + str(swid).strip().strip("{}") + "}"
    for row in (payload or {}).get("teams") or []:
        owners = [str(o) for o in (row.get("owners") or []) if o]
        if wanted in owners:
            try:
                return int(row.get("id"))
            except (TypeError, ValueError):
                return None
    return None


__all__ = [
    "build_teams",
    "current_week",
    "find_my_roster_id",
    "injury_map",
    "load_league",
    "matchup_rows",
    "normalize_team",
    "parse_slots",
]
