"""Turn an ESPN league's scoring settings into :class:`ScoringRules`.

ESPN describes scoring as a list of ``scoringItems``, each a ``statId`` and a
point value. Mapping those onto the canonical keys in
:mod:`fantasypicker.sleeper.scoring` means one scoring engine — and therefore
one set of training labels — serves both platforms.

Two details carry real risk and are handled explicitly rather than by accident:

**Aliases.** Several ESPN stat IDs describe the same underlying event from
different angles. "Interception Return TD" and "Fumble Return TD" are disjoint
halves of the defensive touchdowns the box score reports as one number, and
"Each reception" appears under two IDs. Adding those together would double a
league's defensive touchdown value, so alias groups take the largest value set
rather than the sum.

**Position overrides.** ESPN can vary a stat's value by position — TE premium
being the common one. Where the scoring engine can express that (receptions), it
is translated into the matching per-position bonus; where it cannot, the
override is reported as unsupported instead of being quietly applied to
everybody.
"""

from __future__ import annotations

import logging

from ..sleeper.scoring import (
    DST_TERMS,
    OFFENSE_TERMS,
    PTS_ALLOW_BUCKETS,
    YDS_ALLOW_BUCKETS,
    ScoringRules,
)
from .ids import STAT_KEYS, stat_label

log = logging.getLogger(__name__)

#: Keys reachable from several stat IDs that are alternative encodings of one
#: rule, or disjoint subsets of one aggregate. The largest configured value
#: wins; summing them would multiply the league's actual scoring.
ALIAS_KEYS = frozenset(
    {
        "rec", "pass_yd", "rush_yd", "rec_yd", "pass_cmp", "pass_inc",
        "rush_att", "pass_att", "def_td", "def_st_td", "pts_allow",
        "yds_allow", "sack", "def_2pt", "fum", "idp_tkl",
    }
)

#: ESPN's ``pointsOverrides`` keys, in the slot-position space its settings
#: screen is organised by. Only the offensive skill positions matter here: a
#: D/ST override on a receiving stat is meaningless and is ignored.
OVERRIDE_POSITIONS: dict[str, str] = {
    "0": "QB",
    "2": "RB",
    "4": "WR",
    "6": "TE",
    "16": "DST",
    "17": "K",
}

#: Per-position reception bonuses the engine can express, so a TE-premium or
#: RB-only-PPR league translates exactly.
_REC_BONUS: dict[str, str] = {
    "RB": "bonus_rec_rb",
    "WR": "bonus_rec_wr",
    "TE": "bonus_rec_te",
}


def _known_keys() -> set[str]:
    return (
        set(OFFENSE_TERMS)
        | set(DST_TERMS)
        | set(PTS_ALLOW_BUCKETS)
        | set(YDS_ALLOW_BUCKETS)
    )


def scoring_from_espn(scoring_settings: dict | None) -> ScoringRules:
    """Build scoring rules from an ESPN ``settings.scoringSettings`` block."""
    items = (scoring_settings or {}).get("scoringItems") or []
    settings: dict[str, float] = {}
    unsupported: list[str] = []
    known = _known_keys()

    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            stat_id = int(item.get("statId"))
        except (TypeError, ValueError):
            continue
        try:
            points = float(item.get("points") or 0.0)
        except (TypeError, ValueError):
            points = 0.0

        mapped = STAT_KEYS.get(stat_id)
        overrides = item.get("pointsOverrides") or {}

        if mapped is None:
            # Unrecognised, or recognised but underivable from a box score.
            if points or overrides:
                unsupported.append(stat_label(stat_id))
            continue

        key, multiplier = mapped
        if key not in known:
            if points:
                unsupported.append(stat_label(stat_id))
            continue

        value = points * multiplier
        if value:
            if key in ALIAS_KEYS:
                # Largest magnitude wins — see the module docstring.
                current = settings.get(key, 0.0)
                if abs(value) > abs(current):
                    settings[key] = value
            else:
                settings[key] = settings.get(key, 0.0) + value

        _apply_overrides(
            key, value, overrides, multiplier, settings, unsupported, stat_id
        )

    if not settings:
        log.warning(
            "ESPN returned no usable scoring items; falling back to standard "
            "scoring, which will be wrong if this league is not standard."
        )
        return ScoringRules.from_league(None)

    return ScoringRules(settings=settings, unsupported=tuple(sorted(set(unsupported))))


def _apply_overrides(
    key: str,
    base_value: float,
    overrides: dict,
    multiplier: float,
    settings: dict[str, float],
    unsupported: list[str],
    stat_id: int,
) -> None:
    """Translate a stat's per-position values, or report that we cannot."""
    if not isinstance(overrides, dict):
        return
    for raw_position, raw_value in overrides.items():
        position = OVERRIDE_POSITIONS.get(str(raw_position))
        try:
            value = float(raw_value) * multiplier
        except (TypeError, ValueError):
            continue
        if position is None or value == base_value:
            continue
        if key == "rec" and position in _REC_BONUS:
            # The engine adds this on top of the league-wide reception value.
            delta = value - base_value
            if delta:
                settings[_REC_BONUS[position]] = delta
            continue
        if position == "DST" and key in set(DST_TERMS) | set(PTS_ALLOW_BUCKETS):
            # Defensive stats are already scored only for defenses.
            continue
        unsupported.append(f"{stat_label(stat_id)} (different value for {position})")


def describe_items(scoring_settings: dict | None) -> list[dict[str, object]]:
    """Every scoring item with what it was translated to — for diagnostics.

    Scoring is the one setting whose mistranslation is invisible: the app would
    keep working and quietly rank players under the wrong rules. Printing the
    ESPN label, the value, and the key it mapped to makes that checkable
    against the league's settings page in a few seconds.
    """
    rows: list[dict[str, object]] = []
    known = _known_keys()
    for item in (scoring_settings or {}).get("scoringItems") or []:
        if not isinstance(item, dict):
            continue
        try:
            stat_id = int(item.get("statId"))
        except (TypeError, ValueError):
            continue
        try:
            points = float(item.get("points") or 0.0)
        except (TypeError, ValueError):
            points = 0.0
        mapped = STAT_KEYS.get(stat_id)
        key: str | None = None
        note = ""
        if mapped is None:
            note = "not derivable from box scores"
        elif mapped[0] not in known:
            note = "not derivable from box scores"
        else:
            key = mapped[0]
            if mapped[1] != 1.0:
                note = f"x{mapped[1]:g}"
        overrides = {
            OVERRIDE_POSITIONS.get(str(k), str(k)): v
            for k, v in (item.get("pointsOverrides") or {}).items()
        }
        rows.append(
            {
                "stat_id": stat_id,
                "label": stat_label(stat_id),
                "points": points,
                "key": key,
                "note": note,
                "overrides": overrides,
            }
        )
    return rows
