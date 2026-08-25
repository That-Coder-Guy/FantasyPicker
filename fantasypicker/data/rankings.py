"""Consensus expert rankings, used as a draft-market prior.

The draft engine needs two different things from the market:

1. **Where a player is likely to go.** Expert consensus rank stands in for ADP.
2. **How much that is likely to vary.** The consensus standard deviation is what
   makes "will he last until my next pick?" a probability instead of a guess,
   which is the whole basis of the value-over-next-available recommendation.

DynastyProcess scrapes FantasyPros nightly and publishes the result, including
``ecr``, ``sd``, ``best``, and ``worst`` per player. Superflex leagues get the
``redraft-op`` board, where quarterbacks are priced correctly, instead of the
1-QB board.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..cache import fetch_file
from ..config import DYNASTYPROCESS_RAW, get_settings
from .crosswalk import Crosswalk, normalize_team

log = logging.getLogger(__name__)

#: (superflex, dynasty) -> the FantasyPros board to use.
BOARDS = {
    (False, False): "redraft-overall",
    (True, False): "redraft-op",
    (False, True): "dynasty-overall",
    (True, True): "dynasty-op",
}


def _load_ecr_file() -> pd.DataFrame:
    path = fetch_file(
        f"{DYNASTYPROCESS_RAW}/db_fpecr_latest.csv", get_settings().ttl_current_season
    )
    return pd.read_csv(path, low_memory=False)


def load_expert_ranks(
    crosswalk: Crosswalk,
    *,
    superflex: bool = False,
    dynasty: bool = False,
) -> pd.DataFrame:
    """Consensus draft board mapped onto Sleeper player IDs.

    Returns columns: ``sleeper_id, player, position, team, ecr, sd, best,
    worst, bye``. Players the crosswalk cannot resolve are dropped, with a
    count logged — silently ranking a player nobody can draft is worse than
    a shorter board.
    """
    raw = _load_ecr_file()
    board = BOARDS[(bool(superflex), bool(dynasty))]
    df = raw[raw["page_type"] == board].copy()
    if df.empty:
        log.warning("no rows for FantasyPros board %s", board)
        return pd.DataFrame(
            columns=["sleeper_id", "player", "position", "team", "ecr", "sd", "best", "worst", "bye"]
        )

    df["position"] = df["pos"].astype(str).str.upper().replace({"DEF": "DST"})
    df["team"] = df["team"].map(normalize_team)
    df["fp_id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64").astype(str)
    df["sleeper_id"] = [
        crosswalk.resolve_from_fp(fp, name, pos)
        for fp, name, pos in zip(df["fp_id"], df["player"], df["position"])
    ]

    missing = df["sleeper_id"].isna().sum()
    if missing:
        log.info("%d of %d ranked players had no Sleeper id", missing, len(df))
    df = df[df["sleeper_id"].notna()].copy()

    for col in ("ecr", "sd", "best", "worst", "bye"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # A zero standard deviation would make availability a step function; the
    # floor keeps the survival curve smooth for consensus top picks.
    df["sd"] = df["sd"].fillna(df["ecr"].clip(lower=1) * 0.25).clip(lower=0.75)
    df = df.sort_values("ecr").reset_index(drop=True)
    df["overall_rank"] = df.index + 1

    return df[
        [
            "sleeper_id",
            "player",
            "position",
            "team",
            "ecr",
            "sd",
            "best",
            "worst",
            "bye",
            "overall_rank",
        ]
    ]


def load_weekly_expert_points(crosswalk: Crosswalk) -> pd.DataFrame:
    """FantasyPros' own weekly projected points, if a fresh scrape exists.

    Used only as a blending prior and a sanity check against our model —
    never as the projection itself. The file carries its own ``scrape_date``
    so a stale copy from last season can be ignored by the caller.
    """
    path = fetch_file(
        f"{DYNASTYPROCESS_RAW}/fp_latest_weekly.csv", get_settings().ttl_current_season
    )
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return pd.DataFrame()
    df["position"] = df["pos"].astype(str).str.upper().replace({"DEF": "DST"})
    df["fp_id"] = pd.to_numeric(df["fantasypros_id"], errors="coerce").astype("Int64").astype(str)
    df["sleeper_id"] = [
        crosswalk.resolve_from_fp(fp, name, pos)
        for fp, name, pos in zip(df["fp_id"], df["player_name"], df["position"])
    ]
    df = df[df["sleeper_id"].notna()].copy()
    df["expert_points"] = pd.to_numeric(df.get("r2p_pts"), errors="coerce")
    df["scrape_date"] = pd.to_datetime(df["scrape_date"], errors="coerce")
    return df[
        ["sleeper_id", "player_name", "position", "expert_points", "ecr", "sd", "scrape_date"]
    ]
