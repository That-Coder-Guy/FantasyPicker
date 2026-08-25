"""HTTP API and static hosting for the dashboard.

The API is deliberately thin: every route hands off to
:class:`~fantasypicker.service.PickerService` and serialises the result. The one
piece of real logic here is the loading contract — projection-backed routes
answer HTTP 425 ("too early") with the current warm-up stage rather than
blocking for three minutes, so the UI can show progress instead of a dead tab.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .cache import FetchError
from .service import NotReady, service

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="FantasyPicker", version=__version__)


class ConnectRequest(BaseModel):
    league_id: str = Field(..., description="Sleeper league ID from the league URL")
    username: str | None = Field(None, description="Your Sleeper username, to find your team")


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


@app.post("/api/connect")
async def connect(request: ConnectRequest) -> dict[str, Any]:
    """Attach to a Sleeper league and start loading projections behind it."""
    try:
        summary = await service.connect(request.league_id.strip(), request.username)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    service.start_warmup()
    return {"league": summary, "status": service.status.as_dict()}


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return {
        "status": service.status.as_dict(),
        "league": service.describe() if service.league else {"connected": False},
    }


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


@app.get("/api/waivers")
async def waivers(week: int | None = None, roster_id: int | None = None) -> dict[str, Any]:
    return await service.waivers(week, roster_id=roster_id)


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


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
