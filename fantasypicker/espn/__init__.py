"""Reading an ESPN fantasy football league.

ESPN's fantasy API is not documented or advertised, but it is the same one
espn.com's own front end talks to, and it will answer read requests for a public
league with no credentials at all. A private league needs two cookies copied out
of a browser you are already signed into — see :mod:`.client`.

The strategy throughout this package is translation, not abstraction: ESPN's
responses are converted into the shapes the rest of the app already speaks
(:class:`~fantasypicker.sleeper.league.Team`, ``RosterSlot``, ``ScoringRules``,
Sleeper-shaped matchup rows). Everything downstream — the projection model, the
simulator, the lineup solver, the draft engine — then works on an ESPN league
without knowing one exists.
"""

from __future__ import annotations

from .client import EspnAuthRequired, EspnClient, EspnLeagueNotFound
from .league import load_league
from .scoring import scoring_from_espn

__all__ = [
    "EspnAuthRequired",
    "EspnClient",
    "EspnLeagueNotFound",
    "load_league",
    "scoring_from_espn",
]
