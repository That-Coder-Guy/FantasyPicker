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
    warm.add_argument("-v", "--verbose", action="store_true")

    leagues = sub.add_parser(
        "leagues", help="List the Sleeper leagues a username belongs to, with their IDs."
    )
    leagues.add_argument("username")
    leagues.add_argument("--season", type=int, default=None)
    leagues.add_argument("-v", "--verbose", action="store_true")

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
        return _warm(args.league_id, args.username)

    if command == "leagues":
        return _leagues(args.username, args.season)

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


def _warm(league_id: str, username: str | None) -> int:
    import asyncio

    from .service import service

    async def run() -> int:
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
