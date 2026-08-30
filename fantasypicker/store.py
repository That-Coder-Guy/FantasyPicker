"""Remembered leagues.

The app is single-user and runs on one machine, so "remember" means a small
JSON file next to the cache. What is worth remembering is per-league, not
global: which team is yours, the username the league was found under, the
scoring fingerprint that says which trained model belongs to it. Two leagues in
the same household have different answers to all three, and the one thing that
should never happen is opening the app and being shown someone else's team.

The file is rewritten atomically and every read tolerates a corrupt or
half-written file by starting over — losing a remembered roster ID is a
two-click annoyance, and refusing to start because a JSON file is malformed is
much worse.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import get_settings

log = logging.getLogger(__name__)

#: Bump when the shape below changes incompatibly; older files are discarded.
#: 2 added ``platform``, since a remembered league is now not necessarily a
#: Sleeper one and reopening it against the wrong API would simply fail.
SCHEMA_VERSION = 2

#: Keep the recent list short enough to render as buttons without a scrollbar.
MAX_REMEMBERED = 12


@dataclass
class RememberedLeague:
    """Everything needed to reopen a league without asking any questions."""

    league_id: str
    name: str = ""
    season: str = ""
    #: "sleeper" or "espn" — which API this league is read from.
    platform: str = "sleeper"
    username: str | None = None
    user_id: str | None = None
    my_roster_id: int | None = None
    my_team: str | None = None
    scoring: str = ""
    #: Fingerprint of the scoring settings, so we can say whether the cached
    #: model and panel for this league are the right ones without loading them.
    scoring_key: str = ""
    teams: int | None = None
    superflex: bool = False
    last_used: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AppState:
    version: int = SCHEMA_VERSION
    active_league_id: str | None = None
    leagues: dict[str, RememberedLeague] = field(default_factory=dict)

    # -- queries ------------------------------------------------------------ #

    @property
    def active(self) -> RememberedLeague | None:
        if self.active_league_id is None:
            return None
        return self.leagues.get(self.active_league_id)

    def get(self, league_id: str) -> RememberedLeague | None:
        return self.leagues.get(str(league_id))

    def recent(self) -> list[RememberedLeague]:
        """Most recently opened first — the order the UI should offer them in."""
        return sorted(self.leagues.values(), key=lambda lg: -lg.last_used)

    # -- mutations ---------------------------------------------------------- #

    def remember(self, league: RememberedLeague, *, make_active: bool = True) -> RememberedLeague:
        """Record a league, merging into whatever is already known about it.

        A merge rather than a replace: reconnecting without a username must not
        wipe the roster ID that was chosen by hand last time.
        """
        existing = self.leagues.get(league.league_id)
        if existing is not None:
            merged = RememberedLeague(**{**existing.as_dict(), **_present(league)})
        else:
            merged = league
        merged.last_used = time.time()
        self.leagues[merged.league_id] = merged
        if make_active:
            self.active_league_id = merged.league_id
        self._trim()
        return merged

    def update(self, league_id: str, **fields: Any) -> RememberedLeague | None:
        league = self.leagues.get(str(league_id))
        if league is None:
            return None
        for key, value in fields.items():
            if hasattr(league, key):
                setattr(league, key, value)
        league.last_used = time.time()
        return league

    def forget(self, league_id: str) -> bool:
        removed = self.leagues.pop(str(league_id), None) is not None
        if self.active_league_id == str(league_id):
            remaining = self.recent()
            self.active_league_id = remaining[0].league_id if remaining else None
        return removed

    def _trim(self) -> None:
        if len(self.leagues) <= MAX_REMEMBERED:
            return
        for league in self.recent()[MAX_REMEMBERED:]:
            if league.league_id != self.active_league_id:
                self.leagues.pop(league.league_id, None)

    # -- serialisation ------------------------------------------------------ #

    def to_json(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "active_league_id": self.active_league_id,
            "leagues": {k: v.as_dict() for k, v in self.leagues.items()},
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "AppState":
        if int(payload.get("version") or 0) != SCHEMA_VERSION:
            log.info("discarding remembered leagues from an older schema")
            return cls()
        leagues: dict[str, RememberedLeague] = {}
        known = set(RememberedLeague.__dataclass_fields__)
        for league_id, row in (payload.get("leagues") or {}).items():
            if not isinstance(row, dict):
                continue
            fields = {k: v for k, v in row.items() if k in known}
            fields["league_id"] = str(league_id)
            try:
                leagues[str(league_id)] = RememberedLeague(**fields)
            except TypeError:
                log.warning("skipping unreadable remembered league %s", league_id)
        active = payload.get("active_league_id")
        return cls(
            active_league_id=str(active) if active in leagues else None,
            leagues=leagues,
        )


def _present(league: RememberedLeague) -> dict[str, Any]:
    """Fields worth merging — anything unset stays as it was."""
    out: dict[str, Any] = {}
    for key, value in league.as_dict().items():
        if value in (None, "", False) and key not in {"league_id"}:
            continue
        out[key] = value
    out["league_id"] = league.league_id
    return out


def load_state() -> AppState:
    """Read the remembered leagues, tolerating anything wrong with the file."""
    path = get_settings().state_file
    if not path.exists():
        return AppState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read %s (%s); starting with no remembered leagues", path, exc)
        return AppState()
    if not isinstance(payload, dict):
        return AppState()
    return AppState.from_json(payload)


def save_state(state: AppState) -> None:
    """Write atomically, so an interrupted save cannot corrupt the file."""
    settings = get_settings()
    settings.ensure_dirs()
    path = settings.state_file
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(state.to_json(), indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("could not save remembered leagues to %s: %s", path, exc)
