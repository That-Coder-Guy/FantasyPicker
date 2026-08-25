"""Runtime configuration.

Everything the app needs to find on disk or on the network is resolved here so
that tests can point the cache at a temp directory with one env var.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOME = Path(os.environ.get("FANTASYPICKER_HOME", Path.home() / ".fantasypicker"))

# nflverse publishes each dataset as a release asset; the tag is the dataset name.
NFLVERSE_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
NFLDATA_RAW = "https://raw.githubusercontent.com/nflverse/nfldata/master/data"
DYNASTYPROCESS_RAW = "https://raw.githubusercontent.com/dynastyprocess/data/master/files"
SLEEPER_API = "https://api.sleeper.app/v1"
SLEEPER_CDN = "https://sleepercdn.com"


@dataclass(frozen=True)
class Settings:
    home: Path = DEFAULT_HOME
    #: Seasons pulled from nflverse to train on. More history = slower first run.
    train_seasons: tuple[int, ...] = field(
        default_factory=lambda: tuple(range(2016, 2027))
    )
    #: Cache lifetimes, seconds.
    #: Sleeper's player file is ~5MB and they ask that it not be hammered, but it
    #: is also the only place injury designations live — and those move all week.
    #: Four hours is the compromise: six pulls a day, and a Sunday-morning
    #: downgrade is picked up before kickoff.
    ttl_players: int = 4 * 3600
    ttl_league: int = 15 * 60
    ttl_matchups: int = 60
    ttl_draft: int = 5  # live draft polling wants this near-realtime
    ttl_state: int = 10 * 60
    ttl_static: int = 12 * 3600  # nflverse/DynastyProcess files
    ttl_current_season: int = 3 * 3600  # in-season nflverse files change weekly
    http_timeout: float = 60.0
    max_retries: int = 4

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def model_dir(self) -> Path:
        return self.home / "models"

    @property
    def state_file(self) -> Path:
        return self.home / "state.json"

    def ensure_dirs(self) -> None:
        for d in (self.home, self.cache_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
        _SETTINGS.ensure_dirs()
    return _SETTINGS


def set_settings(settings: Settings) -> None:
    """Override settings (used by tests)."""
    global _SETTINGS
    settings.ensure_dirs()
    _SETTINGS = settings
