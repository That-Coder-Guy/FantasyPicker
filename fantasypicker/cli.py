"""Command line entry point — mostly a way to start the web app."""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser

from . import __version__
from .config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fantasypicker",
        description="Draft and lineup assistant for Sleeper fantasy football leagues.",
    )
    parser.add_argument("--version", action="version", version=f"fantasypicker {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run the web app (default).")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--open", action="store_true", help="Open a browser window.")
    serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    serve.add_argument("-v", "--verbose", action="store_true")

    warm = sub.add_parser(
        "warm",
        help="Download data and train the model up front, so the first page load is fast.",
    )
    warm.add_argument("league_id")
    warm.add_argument("--username", default=None)
    warm.add_argument(
        "--espn", action="store_true", help="Read the league from ESPN, not Sleeper."
    )
    warm.add_argument("--season", type=int, default=None)
    warm.add_argument("--espn-s2", default=None, help="Private ESPN leagues only.")
    warm.add_argument("--swid", default=None, help="Private ESPN leagues only.")
    warm.add_argument("-v", "--verbose", action="store_true")

    leagues = sub.add_parser(
        "leagues", help="List the Sleeper leagues a username belongs to, with their IDs."
    )
    leagues.add_argument("username")
    leagues.add_argument("--season", type=int, default=None)
    leagues.add_argument("-v", "--verbose", action="store_true")

    doctor = sub.add_parser(
        "doctor",
        help="Print exactly what the platform returns for a league — use this "
        "when team names, rosters, or scoring look wrong.",
    )
    doctor.add_argument("league_id")
    doctor.add_argument(
        "--espn", action="store_true", help="Read the league from ESPN, not Sleeper."
    )
    doctor.add_argument("--season", type=int, default=None)
    doctor.add_argument("--espn-s2", default=None, help="Private ESPN leagues only.")
    doctor.add_argument("--swid", default=None, help="Private ESPN leagues only.")
    doctor.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("where", help="Print the cache and model directories.")

    args = parser.parse_args(argv)
    command = args.command or "serve"

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if command == "where":
        settings = get_settings()
        print(f"home:   {settings.home}")
        print(f"cache:  {settings.cache_dir}")
        print(f"models: {settings.model_dir}")
        return 0

    if command == "warm":
        return _warm(args.league_id, args.username, args)

    if command == "leagues":
        return _leagues(args.username, args.season)

    if command == "doctor":
        if args.espn:
            return _doctor_espn(args.league_id, args.season, args.espn_s2, args.swid)
        return _doctor(args.league_id)

    return _serve(args)


def _leagues(username: str, season: int | None) -> int:
    """Print a user's leagues and IDs — the fastest way to diagnose a bad ID."""
    import asyncio

    from .service import service

    async def run() -> int:
        try:
            data = await service.find_leagues(username, season)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        user = data["user"]
        print(f"{user['display_name'] or user['username']} (user_id {user['user_id']})")
        rows = data["leagues"] or data["previous_season_leagues"]
        if not rows:
            print(
                f"  No leagues found for {data['season']} or the season before.\n"
                "  If your league is on ESPN, Yahoo, or NFL.com, this app cannot "
                "read it — Sleeper is the only platform supported.",
                file=sys.stderr,
            )
            return 1
        if not data["leagues"]:
            print(f"  (no {data['season']} leagues; showing the previous season)")
        for league in rows:
            print(f"  {league['league_id']}  {league['name']}")
            print(f"      {league['teams']} teams · {league['scoring']} · {league['status']}")
        return 0

    return asyncio.run(run())


def _doctor(league_id: str) -> int:
    """Dump the raw shape of a league's Sleeper responses.

    Team names come from the *user* objects, joined to rosters on ``owner_id``.
    When that join misses — an orphaned team, a co-owner-only roster, a user
    with every name field blank — the app can only fall back to "Team 4", and
    from the outside every one of those causes looks identical. This prints the
    join so the cause is obvious in a single run.
    """
    import asyncio

    from .sleeper.client import SleeperClient
    from .sleeper.league import build_teams

    async def run() -> int:
        async with SleeperClient() as client:
            league, users, rosters = await asyncio.gather(
                client.league(league_id),
                client.league_users(league_id, fresh=True),
                client.rosters(league_id, fresh=True),
            )
        if not league:
            print(
                f"Sleeper has no league with ID {league_id}.\n"
                "Check the ID in your league's URL "
                "(sleeper.com/leagues/<this part>/team), and note that an ID "
                "from a previous season is a different league.",
                file=sys.stderr,
            )
            return 1

        print(f"{league.get('name')}  ({league_id})")
        print(
            f"  season {league.get('season')} · status {league.get('status')} "
            f"· {league.get('total_rosters')} rosters"
        )
        print(f"\nusers: {len(users)}")
        for user in users:
            metadata = user.get("metadata") or {}
            print(
                f"  {str(user.get('user_id') or 'no user_id'):<24} "
                f"username={str(user.get('username') or '-'):<20} "
                f"display_name={str(user.get('display_name') or '-'):<20} "
                f"team_name={metadata.get('team_name') or '-'}"
            )

        known = {u.get("user_id") for u in users}
        matched = sum(1 for r in rosters if r.get("owner_id") in known)
        print(f"\nrosters: {len(rosters)} · owner_id matches a user: {matched}")
        teams = build_teams(rosters, users)
        empty_seats = 0
        abandoned = 0
        for roster_id in sorted(teams):
            team = teams[roster_id]
            if team.claimed:
                flag = ""
            elif team.players:
                flag = "  <-- has players but no manager (abandoned)"
                abandoned += 1
            else:
                flag = "  <-- nobody has joined this seat"
                empty_seats += 1
            print(
                f"  roster {roster_id:<3} owner={team.owner_id or 'none':<24} "
                f"label={team.label!r}{flag}"
            )
            if team.avatar_url:
                print(f"      picture: {team.avatar_url}")

        if empty_seats:
            print(
                f"\n{empty_seats} of {len(rosters)} seats have no manager yet — "
                f"{len(users)} of {len(rosters)} spots in this league are filled. "
                "There are no names to show for the empty ones, and nothing to "
                "type in: each name appears on its own once someone joins.",
            )
        if abandoned:
            print(
                f"\n{abandoned} roster(s) hold players but have no owner. Those "
                "are abandoned teams; Sleeper has no name for them.",
                file=sys.stderr,
            )
        return 0

    return asyncio.run(run())


def _doctor_espn(
    league_id: str, season: int | None, espn_s2: str | None, swid: str | None
) -> int:
    """Dump what ESPN returns for a league, and how it was interpreted.

    The scoring table is the important half. Every other mistake announces
    itself — a missing team is visible, a missing player is visible — but
    scoring translated wrongly is silent: the app keeps working and quietly
    ranks players under rules that are not this league's. Printing each ESPN
    setting beside the term it became makes that checkable against the league's
    own settings page in about a minute.
    """
    import asyncio

    from .credentials import EspnCredentials, describe as describe_credentials
    from .credentials import load_credentials
    from .data.crosswalk import load_crosswalk
    from .data.nflverse import current_nfl_season
    from .espn.client import EspnAuthRequired, EspnClient, EspnLeagueNotFound
    from .espn.league import build_teams, has_drafted, parse_slots, raw_entry_counts
    from .espn.scoring import describe_items, scoring_from_espn

    year = int(season or current_nfl_season())
    if espn_s2 and swid:
        credentials = EspnCredentials(espn_s2=espn_s2, swid=swid).normalised()
    else:
        credentials = load_credentials(league_id)

    async def run() -> int:
        client = EspnClient(
            espn_s2=credentials.espn_s2 if credentials else None,
            swid=credentials.swid if credentials else None,
        )
        async with client:
            try:
                payload = await client.league(league_id, year, fresh=True)
                rosters = await client.rosters(league_id, year, fresh=True)
            except EspnAuthRequired as exc:
                print(str(exc), file=sys.stderr)
                return 1
        if not payload:
            print(str(EspnLeagueNotFound(league_id, year)), file=sys.stderr)
            return 1

        settings = payload.get("settings") or {}
        league_id_value = league_id
        stored = describe_credentials(league_id)
        print(f"{settings.get('name') or payload.get('name')}  ({league_id})")
        print(
            f"  season {year} · {settings.get('size')} teams · "
            f"week {(payload.get('status') or {}).get('currentMatchupPeriod')}"
        )
        print(
            "  cookies: "
            + (
                f"stored (espn_s2 {stored['espn_s2']}, SWID {stored['swid']})"
                if stored["stored"]
                else "none stored — fine for a public league"
            )
        )

        drafted = has_drafted(payload)
        if drafted is not None:
            print(f"  draft: {'complete' if drafted else 'not yet drafted'}")

        # The pick feed is what fills rosters while a draft runs, so when the
        # pages look empty mid-draft this is the thing to look at.
        detail = payload.get("draftDetail") or {}
        raw_picks = detail.get("picks") or []
        print(f"  draft picks in feed: {len(raw_picks)}", end="")
        if detail.get("inProgress"):
            print("  (draft in progress)")
        else:
            print()
        if raw_picks:
            from .platforms import _espn_draft, picks_by_roster

            class _Stub:
                league_id = league_id_value
                teams = {}
                roster_size = 0

            from .espn.ids import is_unmade_pick

            _, parsed = _espn_draft(payload, _Stub())
            by_roster = picks_by_roster(parsed)
            made = [p for p in raw_picks if not is_unmade_pick(p.get("playerId"))]
            print(
                f"  picks actually made: {len(made)} of {len(raw_picks)} slots "
                f"(the rest are empty slots ESPN has not reached)"
            )
            print(f"  of those, readable as players: {len(parsed)}")
            if by_roster:
                counts = ", ".join(
                    f"team {rid}: {len(players)}"
                    for rid, players in sorted(by_roster.items())
                )
                print(f"  picks per team: {counts}")
            for pick in (made or raw_picks)[:3]:
                print(
                    f"      team {pick.get('teamId')} took playerId "
                    f"{pick.get('playerId')} at #{pick.get('overallPickNumber')}"
                )
            if not made:
                print(
                    "\n  ESPN has allocated the draft board but published no "
                    "picks to this API yet. If picks have genuinely been made "
                    "in the draft room, ESPN is not exposing them here in real "
                    "time and rosters will fill in once it does.",
                    file=sys.stderr,
                )
            elif len(parsed) < len(made):
                print(
                    f"  {len(made) - len(parsed)} made picks could not be matched "
                    "to a player and will be missing from rosters.",
                    file=sys.stderr,
                )

        slots, bench = parse_slots(settings.get("rosterSettings"))
        print(f"\nstarting lineup: {', '.join(s.name for s in slots)}  (+{bench} bench)")

        scoring = scoring_from_espn(settings.get("scoringSettings"))
        print(f"\nscoring: {scoring.describe()}")
        print("  ESPN setting                          value  ->  scored as")
        for row in describe_items(settings.get("scoringSettings")):
            if not row["points"] and not row["overrides"]:
                continue
            target = row["key"] or f"(ignored: {row['note']})"
            extra = f"  {row['note']}" if row["key"] and row["note"] else ""
            print(
                f"  {str(row['label'])[:36]:<36} {row['points']:>6.3g}  ->  {target}{extra}"
            )
            for position, value in (row["overrides"] or {}).items():
                print(f"      {position}: {value}")
        if scoring.unsupported:
            print(
                "\n  These settings cannot be rebuilt from public box scores and "
                "are scored as zero:\n    " + "\n    ".join(scoring.unsupported),
                file=sys.stderr,
            )

        source = rosters or payload
        teams, unresolved = build_teams(source, load_crosswalk())
        sent = raw_entry_counts(source)
        print(f"\nteams: {len(teams)}")
        for roster_id in sorted(teams):
            team = teams[roster_id]
            # "N players" alone cannot distinguish an empty roster from one we
            # failed to read, which is the whole question when it says zero.
            entries = sent.get(roster_id, 0)
            counts = f"{len(team.players):>2} players"
            if entries != len(team.players):
                counts += f" (ESPN sent {entries} entries)"
            print(
                f"  roster {roster_id:<3} {team.label[:28]:<28} "
                f"{team.record:<7} {counts} "
                f"· {team.manager}"
            )
            picture = team.avatar_url or "(none in ESPN response)"
            if "fantasy.espn.com" in picture:
                picture += "  (authenticated; the app proxies it with your cookies)"
            print(f"      picture: {picture}")
        if not any(t.avatar_url for t in teams.values()):
            print(
                "\nESPN returned no team logos for this league. The app shows "
                "each team's initials instead; there is no picture to fetch.",
                file=sys.stderr,
            )
        total_sent = sum(sent.values())
        total_read = sum(len(t.players) for t in teams.values())
        if total_sent and not total_read:
            print(
                f"\nESPN sent {total_sent} roster entries and none could be read. "
                "That is a parsing failure in this app, not an empty league — "
                "please report the league's roster response shape.",
                file=sys.stderr,
            )
        elif not total_sent:
            drafted = has_drafted(payload)
            if drafted is False:
                print(
                    "\nThis league has not drafted yet, so every roster is empty "
                    "— espn.com shows the same thing. Nothing is wrong; rosters "
                    "appear here as soon as the draft runs."
                )
            else:
                print(
                    "\nESPN sent no roster entries at all, and reports the draft "
                    "as complete. That is worth investigating.",
                    file=sys.stderr,
                )
        if unresolved:
            print(
                f"\n{len(unresolved)} rostered players could not be matched to a "
                "player ID, so they get no projection:",
                file=sys.stderr,
            )
            for row in unresolved[:15]:
                print(
                    f"  {row['name']} ({row.get('position') or '?'}, "
                    f"{row.get('team') or 'FA'}) espn_id={row.get('espn_id')}",
                    file=sys.stderr,
                )
        return 0

    return asyncio.run(run())


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    url = f"http://{args.host}:{args.port}/"
    print(f"FantasyPicker running at {url}")
    print("The first league you connect takes a few minutes to download data and train.")
    if args.open:
        webbrowser.open(url)
    uvicorn.run(
        "fantasypicker.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


def _warm(
    league_id: str, username: str | None, args: argparse.Namespace | None = None
) -> int:
    import asyncio

    from .service import service

    espn = bool(getattr(args, "espn", False))

    async def run() -> int:
        if espn:
            summary = await service.connect_espn(
                league_id,
                season=getattr(args, "season", None),
                espn_s2=getattr(args, "espn_s2", None),
                swid=getattr(args, "swid", None),
            )
        else:
            summary = await service.connect(league_id, username)
        print(f"Connected to {summary['name']} ({summary['scoring']})")
        if summary.get("unsupported_scoring"):
            print(
                "  note: these scoring settings cannot be derived from box scores "
                "and are ignored: " + ", ".join(summary["unsupported_scoring"])
            )
        await service._warm()  # noqa: SLF001 - the CLI is an in-process caller
        if service.status.error:
            print(f"Failed: {service.status.error}", file=sys.stderr)
            return 1
        card = service.model_card()
        print(f"Model trained on seasons {card['seasons'][0]}–{card['seasons'][-1]}")
        for position, metrics in sorted(card.get("validation", {}).items()):
            print(
                f"  {position:4s} MAE {metrics['mae']:.2f} "
                f"(recent-average baseline {metrics.get('baseline_mae', float('nan')):.2f})"
            )
        return 0

    return asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
