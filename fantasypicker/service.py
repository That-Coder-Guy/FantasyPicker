"""Application service: one object that holds the loaded league and answers questions.

The web layer should not know about panels, boosters, or crosswalks. It asks for
draft advice or a matchup analysis; this module owns the expensive state and the
order things have to happen in.

Loading is staged deliberately. Connecting to a league is fast and happens
immediately, so the UI can show the league, its rosters, and its scoring within
a second. Everything model-related — eleven seasons of box scores, training six
positions' worth of quantile models — runs in a background thread with progress
reported through :attr:`PickerService.status`, because the first run genuinely
takes a few minutes and a spinner with no explanation is worse than a slow load
with one.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .cache import FetchError
from .config import get_settings
from .data.crosswalk import Crosswalk, load_crosswalk, normalize_team
from .data.nflverse import current_nfl_season
from .data.rankings import load_expert_ranks
from .engine import draft as draft_engine
from .engine import waivers as waiver_engine
from .engine.correlations import CorrelationModel, estimate_correlations
from .engine.matchup import MatchupAnalysis, analyze_matchup
from .model import dataset as dataset_module
from .model.availability import AvailabilityModel, fit_availability
from .model.predict import (
    ProjectionSet,
    apply_availability,
    project_season,
    project_week,
)
from .model.train import ProjectionModel, load_model, train_model
from .sleeper.client import SleeperClient
from .sleeper.league import LeagueContext, load_league, refresh_teams
from .sleeper.scoring import ScoringRules

log = logging.getLogger(__name__)

#: Sleeper is re-polled at most this often, however many requests arrive.
REFRESH_INTERVAL = 30.0


class NotReady(RuntimeError):
    """Raised when a projection-dependent view is requested before warm-up."""


@dataclass
class LoadStatus:
    stage: str = "idle"
    detail: str = ""
    progress: float = 0.0
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        elapsed = None
        if self.started_at:
            end = self.finished_at or time.time()
            elapsed = round(end - self.started_at, 1)
        return {
            "stage": self.stage,
            "detail": self.detail,
            "progress": round(self.progress, 3),
            "error": self.error,
            "elapsed_seconds": elapsed,
            "ready": self.stage == "ready",
        }


@dataclass
class PickerService:
    """Holds one league's loaded state and serves the app's questions."""

    league_id: str | None = None
    username: str | None = None
    user_id: str | None = None

    league: LeagueContext | None = None
    players: dict[str, dict] = field(default_factory=dict)
    crosswalk: Crosswalk | None = None
    ranks: pd.DataFrame | None = None

    panel: pd.DataFrame | None = None
    model: ProjectionModel | None = None
    availability: AvailabilityModel | None = None
    correlations: CorrelationModel | None = None
    season_projections: ProjectionSet | None = None
    weekly_projections: dict[int, ProjectionSet] = field(default_factory=dict)

    status: LoadStatus = field(default_factory=LoadStatus)
    #: When Sleeper was last re-polled for rosters and injury status.
    last_refresh: float | None = None
    _warm_task: asyncio.Task | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # -- connection --------------------------------------------------------- #

    async def find_leagues(
        self, username: str, season: int | str | None = None
    ) -> dict[str, Any]:
        """Every league a Sleeper username belongs to, so nobody has to hunt for an ID.

        A league ID is an eighteen-digit number that only appears in a browser
        URL — invisible if you use the phone app, and easy to confuse with a
        draft ID. A username is something people know. This also answers the
        harder question by elimination: if the username resolves but has no
        leagues for the season, the league is not on Sleeper at all.
        """
        username = username.strip().lstrip("@")
        async with SleeperClient() as client:
            user = await client.user(username)
            if not user or not user.get("user_id"):
                raise ValueError(
                    f"Sleeper has no user called {username!r}. This is the username "
                    "you log in with, not your team name."
                )
            state = await client.state()
            season = season or state.get("season") or current_nfl_season()
            leagues = await client.user_leagues(str(user["user_id"]), season)
            # Sleeper only returns the requested season, so an off-by-one year
            # looks identical to "no leagues". Check the previous one too.
            previous: list[dict] = []
            if not leagues:
                previous = await client.user_leagues(
                    str(user["user_id"]), int(season) - 1
                )

        def summarise(rows: list[dict], year: int | str) -> list[dict[str, Any]]:
            out = []
            for row in rows or []:
                scoring = ScoringRules.from_league(row)
                positions = row.get("roster_positions") or []
                out.append(
                    {
                        "league_id": row.get("league_id"),
                        "name": row.get("name"),
                        "season": str(year),
                        "teams": row.get("total_rosters"),
                        "scoring": scoring.describe(),
                        "status": row.get("status"),
                        "superflex": any(
                            p in {"SUPER_FLEX", "QB_FLEX"} for p in positions
                        ),
                        "avatar": row.get("avatar"),
                    }
                )
            return out

        return {
            "user": {
                "user_id": user.get("user_id"),
                "username": user.get("username") or username,
                "display_name": user.get("display_name"),
            },
            "season": str(season),
            "leagues": summarise(leagues, season),
            "previous_season_leagues": summarise(previous, int(season) - 1),
        }

    async def connect(self, league_id: str, username: str | None = None) -> dict[str, Any]:
        """Load the league itself — fast, and enough to render the whole UI shell."""
        league_id = _clean_league_id(league_id)
        async with SleeperClient() as client:
            state = await client.state()
            self.players = await client.players()
            league = await load_league(client, league_id, username=username)

        self.league_id = league_id
        self.username = username
        self.league = league
        self.crosswalk = load_crosswalk(self.players)
        if username:
            user = await self._lookup_user(username)
            self.user_id = (user or {}).get("user_id")

        try:
            self.ranks = load_expert_ranks(
                self.crosswalk,
                superflex=league.is_superflex,
                dynasty=league.is_dynasty,
            )
        except FetchError as exc:
            log.warning("expert ranks unavailable: %s", exc)
            self.ranks = pd.DataFrame()

        # A new league invalidates anything trained for the previous one.
        self.model = None
        self.panel = None
        self.season_projections = None
        self.weekly_projections = {}
        self.status = LoadStatus(stage="connected", detail=league.name, progress=0.05)
        self.last_refresh = time.time()
        return self.describe(state)

    async def _lookup_user(self, username: str) -> dict | None:
        async with SleeperClient() as client:
            return await client.user(username)

    # -- staying current ---------------------------------------------------- #

    async def refresh_live(self, *, force: bool = False) -> dict[str, bool]:
        """Re-pull the things that move during a week.

        Called before every question the app answers, throttled so a burst of
        requests costs one round trip. Two things are refreshed: the rosters
        (a waiver claim, a trade, a drop) and Sleeper's player file, which is
        where injury designations live. Neither needs the model to be rebuilt —
        the projections are conditional on playing, and availability is applied
        on top of them.
        """
        if self.league is None:
            return {"rosters": False, "players": False}
        age = time.time() - (self.last_refresh or 0.0)
        if not force and age < REFRESH_INTERVAL:
            return {"rosters": False, "players": False}
        self.last_refresh = time.time()

        changed = {"rosters": False, "players": False}
        try:
            async with SleeperClient() as client:
                # An explicit refresh bypasses the disk cache; the throttled
                # background one is happy to be served from it.
                changed["rosters"] = await refresh_teams(
                    client, self.league, fresh=force
                )
                players = await client.players(fresh=force)
        except FetchError as exc:
            # Never fail a page because a refresh could not reach Sleeper; the
            # previous state is stale but still useful.
            log.warning("live refresh failed, serving cached state: %s", exc)
            return changed

        if players and players is not self.players:
            self.players = players
        if self.availability is not None:
            for projections in self.weekly_projections.values():
                if apply_availability(projections, self.availability, self.players):
                    changed["players"] = True
            if self.season_projections is not None:
                apply_availability(
                    self.season_projections, self.availability, self.players
                )
        return changed

    def describe(self, state: dict | None = None) -> dict[str, Any]:
        league = self.league
        if league is None:
            return {"connected": False}
        my_team = league.my_team
        return {
            "connected": True,
            "league_id": league.league_id,
            "name": league.name,
            "season": league.season,
            "current_week": int((state or {}).get("week") or league.current_week),
            "teams": league.team_count,
            "scoring": league.scoring.describe(),
            "unsupported_scoring": list(league.scoring.unsupported),
            "roster_slots": [s.name for s in league.slots],
            "bench_size": league.bench_size,
            "superflex": league.is_superflex,
            "dynasty": league.is_dynasty,
            "idp": league.is_idp,
            "my_roster_id": league.my_roster_id,
            "my_team": my_team.label if my_team else None,
            "teams_list": [
                {
                    "roster_id": t.roster_id,
                    "label": t.label,
                    "owner": t.display_name,
                    "record": t.record,
                    "points_for": round(t.points_for, 2),
                    "is_me": t.roster_id == league.my_roster_id,
                }
                for t in sorted(league.teams.values(), key=lambda t: t.roster_id)
            ],
        }

    def set_my_team(self, roster_id: int) -> None:
        if self.league is not None:
            self.league.my_roster_id = int(roster_id)

    # -- warm-up ------------------------------------------------------------ #

    def start_warmup(self, *, force: bool = False) -> None:
        """Kick off model loading in the background (idempotent)."""
        if self._warm_task is not None and not self._warm_task.done():
            return
        if self.model is not None and not force:
            return
        self._warm_task = asyncio.create_task(self._warm(force=force))

    async def _warm(self, *, force: bool = False) -> None:
        async with self._lock:
            if self.league is None:
                self.status = LoadStatus(stage="error", error="No league connected.")
                return
            self.status = LoadStatus(
                stage="loading", detail="Starting", progress=0.05, started_at=time.time()
            )
            try:
                await asyncio.to_thread(self._warm_blocking, force)
                self.status.stage = "ready"
                self.status.detail = "Projections ready"
                self.status.progress = 1.0
                self.status.finished_at = time.time()
            except Exception as exc:  # surfaced to the UI rather than swallowed
                log.exception("warm-up failed")
                self.status.stage = "error"
                self.status.error = str(exc)
                self.status.finished_at = time.time()

    def _set_stage(self, detail: str, progress: float) -> None:
        self.status.detail = detail
        self.status.progress = progress
        log.info("warm-up: %s (%.0f%%)", detail, progress * 100)

    def _warm_blocking(self, force: bool) -> None:
        """The expensive path. Runs in a worker thread."""
        assert self.league is not None and self.crosswalk is not None
        league = self.league
        settings = get_settings()
        season = league.season or current_nfl_season()
        seasons = tuple(s for s in settings.train_seasons if s <= season)

        self._set_stage("Loading eleven seasons of NFL box scores", 0.12)
        team_overrides = self._team_overrides()
        active = self._active_gsis_ids()
        self.panel = dataset_module.build_panel(
            league.scoring,
            seasons,
            team_overrides=team_overrides,
            active_players=active,
        )

        self._set_stage("Measuring injury-report reliability", 0.45)
        self.availability = fit_availability(self.panel, seasons)

        self._set_stage("Measuring same-game correlations", 0.5)
        self.correlations = estimate_correlations(self.panel)

        cached = None if force else load_model(league.scoring)
        if cached is not None:
            self._set_stage("Reusing a trained model for these scoring rules", 0.75)
            self.model = cached
        else:
            self._set_stage("Training quantile models for six positions", 0.6)
            self.model = train_model(league.scoring, seasons=seasons, panel=self.panel)

        self._set_stage("Projecting the rest of the season", 0.85)
        self.season_projections = project_season(
            self.model,
            self.panel,
            season=season,
            from_week=max(1, league.current_week),
            through_week=18,
            crosswalk=self.crosswalk,
            scoring=league.scoring,
            availability=self.availability,
            sleeper_players=self.players,
        )

        self._set_stage("Projecting the current week", 0.95)
        self.weekly_projections = {}
        self._project_week_blocking(league.current_week)

    def _team_overrides(self) -> dict[str, str]:
        """gsis_id -> current NFL team, straight from Sleeper's live player data."""
        assert self.crosswalk is not None
        overrides: dict[str, str] = {}
        for sleeper_id, meta in self.players.items():
            team = normalize_team(meta.get("team"))
            gsis = self.crosswalk.gsis(str(sleeper_id))
            if team and gsis:
                overrides[gsis] = team
        return overrides

    def _active_gsis_ids(self) -> set[str]:
        """Players Sleeper still has on an NFL roster, as gsis IDs."""
        assert self.crosswalk is not None
        active: set[str] = set()
        for sleeper_id, meta in self.players.items():
            if not meta.get("team"):
                continue
            gsis = self.crosswalk.gsis(str(sleeper_id))
            if gsis:
                active.add(gsis)
        active.update({normalize_team(t) or "" for t in _NFL_TEAMS})
        active.discard("")
        return active

    def _project_week_blocking(self, week: int) -> ProjectionSet:
        assert self.model is not None and self.panel is not None
        assert self.league is not None and self.crosswalk is not None
        projections = project_week(
            self.model,
            self.panel,
            season=self.league.season or current_nfl_season(),
            week=week,
            crosswalk=self.crosswalk,
            scoring=self.league.scoring,
            availability=self.availability,
            sleeper_players=self.players,
        )
        self.weekly_projections[week] = projections
        return projections

    async def projections_for_week(self, week: int) -> ProjectionSet:
        self._require_ready()
        if week not in self.weekly_projections:
            await asyncio.to_thread(self._project_week_blocking, week)
        return self.weekly_projections[week]

    def _require_ready(self) -> None:
        if self.model is None or self.panel is None:
            raise NotReady(
                "Projections are still loading. "
                f"Current stage: {self.status.detail or self.status.stage}."
            )

    # -- questions ---------------------------------------------------------- #

    async def matchup(
        self,
        week: int | None = None,
        *,
        roster_id: int | None = None,
        n_sims: int = 20_000,
        opponent_mode: str = "auto",
    ) -> MatchupAnalysis:
        self._require_ready()
        await self.refresh_live()
        assert self.league is not None
        league = self.league
        week = int(week or league.current_week)
        roster_id = roster_id if roster_id is not None else league.my_roster_id
        if roster_id is None:
            raise ValueError(
                "No team selected. Pick your team, or connect with your Sleeper username."
            )
        my_team = league.teams.get(int(roster_id))
        if my_team is None:
            raise ValueError(f"No roster {roster_id} in this league.")

        async with SleeperClient() as client:
            pairing = await league.matchup_for(client, week, roster_id)

        projections = await self.projections_for_week(week)
        return await asyncio.to_thread(
            analyze_matchup,
            league,
            projections,
            week=week,
            my_team=my_team,
            opponent=pairing.away if pairing else None,
            my_starters=pairing.home_starters if pairing else None,
            opponent_starters=pairing.away_starters if pairing else None,
            correlations=self.correlations,
            n_sims=n_sims,
            opponent_mode=opponent_mode,
        )

    async def draft_advice(
        self, *, roster_id: int | None = None, top_n: int = 8
    ) -> dict[str, Any]:
        self._require_ready()
        await self.refresh_live()
        assert self.league is not None and self.season_projections is not None
        league = self.league

        async with SleeperClient() as client:
            drafts = await client.league_drafts(league.league_id)
            draft_obj = None
            picks: list[dict] = []
            if drafts:
                # Sleeper lists newest first; the live one is the one not complete.
                draft_obj = next(
                    (d for d in drafts if d.get("status") != "complete"), drafts[0]
                )
                picks = await client.draft_picks(str(draft_obj.get("draft_id")))

        state = draft_engine.parse_draft_state(draft_obj, picks, my_user_id=self.user_id)
        drafted = set(state.drafted)
        my_slot = state.my_slot
        my_roster = list(state.roster_by_slot.get(my_slot, [])) if my_slot else []
        if not my_roster and roster_id is not None:
            team = league.teams.get(int(roster_id))
            my_roster = list(team.players) if team else []

        board = await asyncio.to_thread(
            lambda: draft_engine.build_board(
                league,
                self.season_projections,
                self.ranks if self.ranks is not None else pd.DataFrame(),
                drafted=drafted,
            )
        )
        advice = await asyncio.to_thread(
            lambda: draft_engine.recommend(league, board, state, my_roster, top_n=top_n)
        )

        return {
            "draft": {
                "draft_id": state.draft_id,
                "status": (draft_obj or {}).get("status"),
                "type": state.pick_type,
                "teams": state.teams,
                "rounds": state.rounds,
                "picks_made": state.picks_made,
                "current_pick": state.current_pick,
                "on_the_clock_slot": state.on_the_clock_slot,
                "my_slot": state.my_slot,
                "my_next_pick": state.my_next_pick,
                "my_upcoming_picks": state.my_upcoming_picks(5),
                "is_my_turn": state.on_the_clock_slot is not None
                and state.on_the_clock_slot == state.my_slot,
            },
            "pick": advice.pick,
            "round": advice.round_number,
            "recommendations": [c.as_dict() for c in advice.recommendations],
            "best_available": [c.as_dict() for c in advice.best_available],
            "positional_runs": advice.positional_runs,
            "roster": [
                self._player_summary(p, board) for p in my_roster
            ],
            "roster_summary": advice.roster_summary,
            "needs": advice.needs,
            "notes": advice.notes,
        }

    async def draft_board(self, *, position: str | None = None, limit: int = 200) -> dict[str, Any]:
        """The full ranked board, for browsing outside a live draft."""
        self._require_ready()
        assert self.league is not None and self.season_projections is not None
        board = await asyncio.to_thread(
            lambda: draft_engine.build_board(
                self.league,
                self.season_projections,
                self.ranks if self.ranks is not None else pd.DataFrame(),
            )
        )
        if position:
            board = board[board["position"] == position.upper()]
        columns = [
            "sleeper_id",
            "name",
            "position",
            "team",
            "projected_points",
            "vor",
            "tier",
            "positional_rank",
            "ecr",
            "adp_sd",
            "bye_week",
            "floor",
            "ceiling",
            "p_play",
        ]
        columns = [c for c in columns if c in board.columns]
        rows = board.head(limit)[columns].round(2)
        return {
            "players": rows.to_dict(orient="records"),
            "replacement": {
                str(k): round(v, 1)
                for k, v in draft_engine.replacement_levels(self.league, board).items()
            },
        }

    async def waivers(self, week: int | None = None, *, roster_id: int | None = None) -> dict[str, Any]:
        self._require_ready()
        await self.refresh_live()
        assert self.league is not None and self.season_projections is not None
        league = self.league
        week = int(week or league.current_week)
        roster_id = roster_id if roster_id is not None else league.my_roster_id
        team = league.teams.get(int(roster_id)) if roster_id is not None else None
        my_roster = list(team.players) if team else []

        async with SleeperClient() as client:
            trending_rows = await client.trending("add", 24, 100)
        trending = {
            str(row.get("player_id")): int(row.get("count") or 0)
            for row in trending_rows
            if isinstance(row, dict)
        }

        weekly = await self.projections_for_week(week)
        report = await asyncio.to_thread(
            lambda: waiver_engine.find_targets(
                league,
                my_roster,
                self.season_projections,
                weekly,
                trending=trending,
            )
        )
        return {
            "week": week,
            "targets": [t.as_dict() for t in report.targets],
            "droppable": report.droppable,
            "notes": report.notes,
        }

    async def player_detail(self, sleeper_id: str, week: int | None = None) -> dict[str, Any]:
        self._require_ready()
        assert self.league is not None
        week = int(week or self.league.current_week)
        weekly = await self.projections_for_week(week)
        row = weekly.by_id(sleeper_id)
        season_row = (
            self.season_projections.by_id(sleeper_id)
            if self.season_projections is not None
            else None
        )
        meta = self.players.get(str(sleeper_id), {})
        history = []
        if self.panel is not None and self.crosswalk is not None:
            gsis = self.crosswalk.gsis(str(sleeper_id)) or str(sleeper_id)
            past = self.panel[
                (self.panel["gsis_id"] == gsis) & (self.panel["played"] == 1)
            ].sort_values(["season", "week"]).tail(20)
            history = [
                {
                    "season": int(r.season),
                    "week": int(r.week),
                    "opponent": str(r.opponent_team),
                    "points": round(float(r.fantasy_points), 2),
                }
                for r in past.itertuples(index=False)
            ]
        # Sleeper's player dump is the richer source, but it can be missing a
        # player entirely (a defense, a just-signed rookie), so the projection
        # row backfills the identity fields.
        fallback = row if row is not None else season_row
        return {
            "sleeper_id": str(sleeper_id),
            "meta": {
                "name": meta.get("full_name")
                or (fallback.get("name") if fallback is not None else None),
                "team": meta.get("team")
                or (fallback.get("team") if fallback is not None else None),
                "position": meta.get("position")
                or (fallback.get("position") if fallback is not None else None),
                "injury_status": meta.get("injury_status"),
                "age": meta.get("age"),
                "years_exp": meta.get("years_exp"),
                "number": meta.get("number"),
            },
            "week": None if row is None else _row_to_dict(row),
            "season": None if season_row is None else _row_to_dict(season_row),
            "history": history,
        }

    def model_card(self) -> dict[str, Any]:
        """What the model is, how it was validated, and what it cannot see."""
        if self.model is None:
            return {"trained": False, "status": self.status.as_dict()}
        card: dict[str, Any] = {
            "trained": True,
            "trained_at": self.model.trained_at,
            "seasons": list(self.model.seasons),
            "n_features": len(self.model.features),
            "quantiles": list(self.model.quantiles),
            "validation": self.model.validation,
            "importance": {
                position: self.model.feature_importance(position, top=12)
                for position in self.model.positions
            },
        }
        if self.league is not None:
            card["scoring"] = self.league.scoring.describe()
            card["unsupported_scoring"] = list(self.league.scoring.unsupported)
        if self.availability is not None:
            card["availability"] = self.availability.rates
        if self.correlations is not None:
            card["correlations"] = self.correlations.describe(top=16)
        return card

    def _player_summary(self, sleeper_id: str, board: pd.DataFrame) -> dict[str, Any]:
        rows = board[board["sleeper_id"] == str(sleeper_id)]
        meta = self.players.get(str(sleeper_id), {})
        if rows.empty:
            return {
                "sleeper_id": str(sleeper_id),
                "name": meta.get("full_name") or str(sleeper_id),
                "position": meta.get("position"),
                "team": meta.get("team"),
                "projected_points": None,
            }
        row = rows.iloc[0]
        return {
            "sleeper_id": str(sleeper_id),
            "name": row.get("name"),
            "position": row.get("position"),
            "team": row.get("team"),
            "projected_points": round(float(row.get("projected_points", 0.0)), 1),
            "tier": int(row.get("tier", 0)),
            "bye_week": None if pd.isna(row.get("bye_week")) else int(row.get("bye_week")),
        }


def _clean_league_id(raw: str) -> str:
    """Accept a pasted Sleeper URL as readily as a bare ID.

    People copy the whole address far more often than they copy the number out
    of the middle of it, and rejecting that is a self-inflicted support burden.
    """
    text = str(raw).strip()
    match = re.search(r"(?:leagues?|draft)/(?:nfl/)?(\d{6,25})", text)
    if match:
        return match.group(1)
    digits = re.findall(r"\d{6,25}", text)
    return digits[0] if digits else text


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float) and pd.isna(value):
            out[str(key)] = None
        elif hasattr(value, "item"):
            out[str(key)] = value.item()
        else:
            out[str(key)] = value
    return out


_NFL_TEAMS = (
    "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA "
    "MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS"
).split()


#: The app is single-league by design; one process, one loaded league.
service = PickerService()
