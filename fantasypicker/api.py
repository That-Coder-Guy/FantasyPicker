"""HTTP API and static hosting for the dashboard.

The API is deliberately thin: every route hands off to
:class:`~fantasypicker.service.PickerService` and serialises the result. The one
piece of real logic here is the loading contract — projection-backed routes
answer HTTP 425 ("too early") with the current warm-up stage rather than
blocking for three minutes, so the UI can show progress instead of a dead tab.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .cache import FetchError
from .espn.client import EspnAuthRequired
from .service import NotReady, service

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Reopen whatever league was in use last time the app ran.

    Starting the server and landing on an empty form, session after session,
    is the difference between a tool and a chore. If the league has since been
    deleted or the network is down, this fails quietly back to the connect
    screen rather than refusing to start.
    """
    try:
        if await service.reopen_last():
            service.start_warmup()
    except Exception as exc:  # never let a bad state file block startup
        log.warning("could not reopen the last league: %s", exc)
    yield


app = FastAPI(title="FantasyPicker", version=__version__, lifespan=lifespan)


class ConnectRequest(BaseModel):
    league_id: str = Field(..., description="Sleeper league ID, or a pasted league URL")
    username: str | None = Field(None, description="Your Sleeper username, to find your team")


class EspnConnectRequest(BaseModel):
    """Connect to an ESPN league.

    ``espn_s2`` and ``swid`` are only needed for a private league. They are
    stored locally for next time and sent to nobody but ESPN; they are never
    echoed back in a response.
    """

    league_id: str = Field(..., description="ESPN league ID, or a pasted league URL")
    season: int | None = Field(None, description="Defaults to the current season")
    espn_s2: str | None = Field(None, description="espn_s2 cookie, private leagues only")
    swid: str | None = Field(None, description="SWID cookie, private leagues only")


class LeagueLookupRequest(BaseModel):
    username: str = Field(..., description="Sleeper username you log in with")
    season: int | None = Field(None, description="Defaults to the current season")


class TeamRequest(BaseModel):
    roster_id: int


def _fail(code: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=code)


@app.exception_handler(NotReady)
async def _not_ready_handler(_request, exc: NotReady) -> JSONResponse:
    # 425 Too Early: the request is valid, the server just is not there yet.
    return _fail(425, str(exc), status=service.status.as_dict())


@app.exception_handler(FetchError)
async def _fetch_error_handler(_request, exc: FetchError) -> JSONResponse:
    return _fail(
        502,
        f"Could not reach an upstream data source: {exc}",
    )


# --------------------------------------------------------------------------- #
# league
# --------------------------------------------------------------------------- #


@app.post("/api/leagues")
async def find_leagues(request: LeagueLookupRequest) -> dict[str, Any]:
    """List a Sleeper user's leagues, so nobody has to dig out a league ID."""
    try:
        return await service.find_leagues(request.username, request.season)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/connect")
async def connect(request: ConnectRequest) -> dict[str, Any]:
    """Attach to a Sleeper league and start loading projections behind it."""
    try:
        summary = await service.connect(request.league_id.strip(), request.username)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    service.start_warmup()
    return {"league": summary, "status": service.status.as_dict()}


@app.post("/api/connect/espn")
async def connect_espn(request: EspnConnectRequest) -> dict[str, Any]:
    """Attach to an ESPN league and start loading projections behind it."""
    try:
        summary = await service.connect_espn(
            request.league_id.strip(),
            season=request.season,
            espn_s2=request.espn_s2,
            swid=request.swid,
        )
    except EspnAuthRequired as exc:
        # 401 rather than 404: the league exists, we are just not allowed in.
        return _fail(401, str(exc), needs_cookies=True)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    service.start_warmup()
    return {"league": summary, "status": service.status.as_dict()}


@app.get("/api/known")
async def known_leagues() -> dict[str, Any]:
    """Leagues this machine has connected to before, newest first."""
    return {
        "leagues": service.known_leagues(),
        "active_league_id": service.state.active_league_id,
    }


@app.delete("/api/known/{league_id}")
async def forget_league(league_id: str) -> dict[str, Any]:
    """Drop a league from the remembered list (does not touch Sleeper)."""
    return {"forgotten": service.forget_league(league_id), "leagues": service.known_leagues()}


@app.post("/api/refresh")
async def refresh() -> dict[str, Any]:
    """Re-poll Sleeper for rosters and injury status right now."""
    if service.league is None:
        raise HTTPException(status_code=409, detail="Connect to a league first.")
    changed = await service.refresh_live(force=True)
    return {"changed": changed, "league": service.describe()}


@app.get("/api/status")
async def status() -> dict[str, Any]:
    # ``describe`` carries the remembered leagues in both the connected and
    # disconnected shapes, so a cold start still offers them on the front page.
    return {"status": service.status.as_dict(), "league": service.describe()}


@app.post("/api/team")
async def set_team(request: TeamRequest) -> dict[str, Any]:
    if service.league is None:
        raise HTTPException(status_code=409, detail="Connect to a league first.")
    service.set_my_team(request.roster_id)
    return {"league": service.describe()}


@app.post("/api/retrain")
async def retrain() -> dict[str, Any]:
    """Force a fresh model build (after a scoring change, or weekly in season)."""
    if service.league is None:
        raise HTTPException(status_code=409, detail="Connect to a league first.")
    service.start_warmup(force=True)
    return {"status": service.status.as_dict()}


# --------------------------------------------------------------------------- #
# projections and decisions
# --------------------------------------------------------------------------- #


@app.get("/api/matchup")
async def matchup(
    week: int | None = None,
    roster_id: int | None = None,
    sims: int = Query(20000, ge=1000, le=100000),
    opponent_mode: str = Query("auto", pattern="^(auto|declared|optimal)$"),
) -> dict[str, Any]:
    try:
        analysis = await service.matchup(
            week, roster_id=roster_id, n_sims=sims, opponent_mode=opponent_mode
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Both rosters, so an opponent's lineup renders with names rather than IDs.
    names = {
        row["sleeper_id"]: row
        for row in list(analysis.player_rows) + list(analysis.opponent_rows)
    }

    def lineup_payload(lineup) -> list[dict[str, Any]] | None:
        if lineup is None:
            return None
        return [
            {
                "slot": lineup.slot_name(index),
                "sleeper_id": player,
                "name": (names.get(player) or {}).get("name"),
                "position": (names.get(player) or {}).get("position"),
                "projection": (names.get(player) or {}).get("projection"),
            }
            for index, player in sorted(lineup.assignment.items())
        ]

    return {
        "week": analysis.week,
        "my_team": analysis.my_team,
        "opponent_team": analysis.opponent_team,
        "win_probability": analysis.win_probability,
        "my_distribution": analysis.my_distribution,
        "opponent_distribution": analysis.opponent_distribution,
        "margin_distribution": analysis.margin_distribution,
        "lineup": lineup_payload(analysis.my_lineup),
        "opponent_lineup": lineup_payload(analysis.opponent_lineup),
        "leverage_lineup": lineup_payload(analysis.leverage_lineup),
        "leverage_gain": analysis.leverage_gain,
        "strategy": analysis.strategy,
        "swaps": analysis.swaps,
        "players": analysis.player_rows,
        "opponent_players": analysis.opponent_rows,
        "notes": analysis.notes,
    }


@app.get("/api/draft")
async def draft(roster_id: int | None = None, top: int = Query(8, ge=1, le=30)) -> dict[str, Any]:
    return await service.draft_advice(roster_id=roster_id, top_n=top)


@app.get("/api/board")
async def board(
    position: str | None = None, limit: int = Query(200, ge=1, le=1000)
) -> dict[str, Any]:
    return await service.draft_board(position=position, limit=limit)


@app.get("/api/teams")
async def teams(week: int | None = None) -> dict[str, Any]:
    """Every team in the league and the lineup each of them can field."""
    return await service.league_teams(week)


@app.get("/api/waivers")
async def waivers(week: int | None = None, roster_id: int | None = None) -> dict[str, Any]:
    return await service.waivers(week, roster_id=roster_id)


@app.get("/api/team-image/{roster_id}")
async def team_image(roster_id: int) -> Response:
    """A team's picture, fetched with the user's own ESPN credentials.

    Custom ESPN logos are served from an authenticated endpoint that a browser
    cannot load directly from this app's page — cross-site cookie rules strip
    the ESPN session from an <img> request, so ESPN answers 401. The server
    holds the cookies, so it fetches on the browser's behalf.
    """
    image = await service.team_image(roster_id)
    if image is None:
        raise HTTPException(status_code=404, detail="No picture for this team.")
    content, content_type = image
    # Cacheable, unlike the app shell: a logo change is cosmetic and rare.
    return Response(
        content, media_type=content_type, headers={"Cache-Control": "max-age=3600"}
    )


@app.get("/api/trades")
async def trades(
    roster_id: int | None = None,
    chains: bool = Query(True),
    max_package: int = Query(2, ge=1, le=2),
) -> dict[str, Any]:
    """Mutually beneficial trade proposals, including multi-trade chains."""
    try:
        return await service.trades(
            roster_id=roster_id, chains=chains, max_package=max_package
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/player/{sleeper_id}")
async def player(sleeper_id: str, week: int | None = None) -> dict[str, Any]:
    return await service.player_detail(sleeper_id, week)


@app.get("/api/model")
async def model_card() -> dict[str, Any]:
    return service.model_card()


# --------------------------------------------------------------------------- #
# static
# --------------------------------------------------------------------------- #

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.middleware("http")
async def _no_stale_app_shell(request, call_next):
    """Make UI updates arrive on a plain reload.

    Without a Cache-Control header, browsers fall back to heuristic caching
    keyed on Last-Modified — a file untouched for weeks can be served from
    local cache for hours without the server ever being asked. For a local app
    that means a `git pull` visibly does nothing until a hard refresh, which
    reads exactly like the fix not working. `no-cache` still allows ETag
    revalidation, so unchanged files remain cheap 304s.
    """
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
