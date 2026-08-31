"""What the rest of the league thinks your players are worth.

Every other manager is making trade decisions off one number: the projection
their platform prints next to each player. It is not as good as this app's
projection — that is the whole reason this app exists — but it is the number
the person on the other end of the trade is looking at, and a proposal that
reads as a fleece on *their* screen gets declined no matter how sound it is on
ours.

So the app carries two currencies and never confuses them:

* **Model points** decide whether a roster actually got better. Every lineup
  solve, every gain, every recommendation is in this currency.
* **Market points** decide whether the trade *looks* fair. Nothing is ever
  valued by them; they only answer "how will they read this?"

Where they come from depends on the platform, in descending order of how
closely they match what the other manager sees:

1. **The league's own platform.** For ESPN this is exact — the same PROJ column
   they are looking at, already under the league's scoring rules.
2. **FantasyPros consensus**, for platforms that publish no projections. Not
   the same numbers, but the same public consensus most managers are anchored
   to.
3. **Nothing**, in which case the model stands in for both and the UI says so.
   Advice degrades to what it was before this existed rather than breaking.

Basis
-----
Platforms publish *season* totals; this app works in *rest-of-season* points,
because that is the horizon a trade lives on. Comparing the two directly would
overstate market value for the whole back half of a season. Market numbers are
therefore prorated onto the rest-of-season basis by the games each player has
left, which the projection frame already counts. Before week 1 the two bases
coincide, so during a draft the market number is exactly the total printed on
the platform.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: How the numbers were obtained, for the UI to name honestly. Never label
#: model-derived numbers as anything else — the entire value of this page is
#: that the two currencies are distinguishable.
ESPN = "ESPN"
CONSENSUS = "FantasyPros consensus"
MODEL = "this app's model"


@dataclass(frozen=True)
class WeeklyRate:
    """A per-week consensus number, in the shape :func:`from_platform` reads.

    Sources that publish a weekly figure rather than a season total (as
    FantasyPros' consensus does) arrive here; the season total is only ever
    used as a fallback, so a nominal 17-game season is good enough for it.
    """

    per_game: float

    @property
    def total(self) -> float:
        return self.per_game * 17.0


@dataclass(frozen=True)
class MarketProjections:
    """Public rest-of-season points per player, and where they came from."""

    points: dict[str, float] = field(default_factory=dict)
    #: Display name of the source, e.g. "ESPN".
    source: str = MODEL
    #: False when this is the model standing in for a real public source, which
    #: the UI has to disclose rather than quietly presenting as consensus.
    available: bool = False
    notes: list[str] = field(default_factory=list)

    def get(self, player: str, default: float = 0.0) -> float:
        return self.points.get(str(player), default)

    def covers(self, player: str) -> bool:
        return str(player) in self.points

    def coverage(self, players: list[str]) -> float:
        """Fraction of a player list this source has a number for."""
        if not players:
            return 1.0
        return sum(1 for p in players if self.covers(p)) / len(players)

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "available": self.available,
            "covered": len(self.points),
            "notes": list(self.notes),
        }


def from_platform(
    published: Mapping[str, object],
    games_remaining: Mapping[str, float],
    *,
    source: str,
    model_points: Mapping[str, float] | None = None,
) -> MarketProjections:
    """Prorate a platform's season projections onto the rest-of-season basis.

    ``published`` maps a Sleeper ID to anything carrying ``total`` and
    ``per_game`` (see :class:`~fantasypicker.espn.projections.MarketPoints`).
    ``games_remaining`` is the per-player game count the projection frame
    already computed, so byes and a mid-season start are handled for free.

    Players the platform projects but the model does not are dropped rather
    than guessed at: without a game count there is no honest way to put them on
    this basis, and the engines cannot value them anyway.
    """
    points: dict[str, float] = {}
    for player, value in published.items():
        per_game = getattr(value, "per_game", None)
        total = getattr(value, "total", None)
        if per_game is None and total is None:
            continue
        games = games_remaining.get(str(player))
        if games is None:
            continue
        if per_game is not None:
            points[str(player)] = float(per_game) * float(games)
        else:
            points[str(player)] = float(total)

    notes: list[str] = []
    if model_points is not None:
        missing = [p for p in model_points if p not in points]
        # A source covering only half the league is worse than none: it would
        # price one side of a trade in public points and the other in zeros.
        if missing and len(missing) > len(model_points) * 0.5:
            notes.append(
                f"{source} projections covered only "
                f"{len(points)} of {len(model_points)} players, so this app's "
                "own numbers are shown instead."
            )
            return from_model(model_points, notes=notes)
        if missing:
            notes.append(
                f"{len(missing)} players have no {source} projection; their "
                "trade value is shown from this app's model only."
            )
    return MarketProjections(points=points, source=source, available=True, notes=notes)


def from_model(
    model_points: Mapping[str, float], *, notes: list[str] | None = None
) -> MarketProjections:
    """Stand in for a public source with the model's own numbers.

    Marked unavailable so nothing downstream claims these are what the other
    manager sees. Trade advice with this in place is exactly as good as it was
    before market projections existed — no better, and no worse.
    """
    return MarketProjections(
        points={str(k): float(v) for k, v in model_points.items()},
        source=MODEL,
        available=False,
        notes=list(notes or []),
    )
