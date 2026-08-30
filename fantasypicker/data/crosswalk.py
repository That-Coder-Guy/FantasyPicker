"""Player identity across Sleeper, nflverse, and FantasyPros.

Three ID spaces have to line up before anything else works:

* **Sleeper** ``player_id`` — what rosters, matchups, and draft picks speak.
* **nflverse** ``gsis_id`` — what every box score and snap count speaks.
* **FantasyPros** id — what the consensus draft rankings speak.

DynastyProcess publishes a maintained crosswalk covering all three, refreshed
through the current season. Sleeper's own player dump also carries a ``gsis_id``
for most players, so that is used first and the crosswalk fills the gaps. A
normalised-name-plus-position match is the last resort, which mostly matters for
rookies in the days after a draft.

Team defenses are their own case: Sleeper keys them by team abbreviation
(``"PIT"``), so the mapping there is just abbreviation normalisation.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from ..cache import fetch_file
from ..config import DYNASTYPROCESS_RAW, get_settings

log = logging.getLogger(__name__)

#: Everything -> the abbreviation nflverse uses.
TEAM_ALIASES: dict[str, str] = {
    "GBP": "GB", "GNB": "GB",
    "KCC": "KC", "KAN": "KC",
    "LVR": "LV", "OAK": "LV", "RAI": "LV",
    "JAC": "JAX",
    "NEP": "NE", "NWE": "NE",
    "NOS": "NO", "NOR": "NO",
    "SFO": "SF",
    "TBB": "TB", "TAM": "TB",
    "LAR": "LA", "RAM": "LA", "STL": "LA",
    "SDC": "LAC", "SD": "LAC",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "WSH": "WAS", "WFT": "WAS",
}

#: Full team names as FantasyPros writes them, for DST rows.
TEAM_NAME_TO_ABBR: dict[str, str] = {
    "arizona cardinals": "ARI", "atlanta falcons": "ATL", "baltimore ravens": "BAL",
    "buffalo bills": "BUF", "carolina panthers": "CAR", "chicago bears": "CHI",
    "cincinnati bengals": "CIN", "cleveland browns": "CLE", "dallas cowboys": "DAL",
    "denver broncos": "DEN", "detroit lions": "DET", "green bay packers": "GB",
    "houston texans": "HOU", "indianapolis colts": "IND", "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC", "las vegas raiders": "LV", "los angeles chargers": "LAC",
    "los angeles rams": "LA", "miami dolphins": "MIA", "minnesota vikings": "MIN",
    "new england patriots": "NE", "new orleans saints": "NO", "new york giants": "NYG",
    "new york jets": "NYJ", "philadelphia eagles": "PHI", "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF", "seattle seahawks": "SEA", "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN", "washington commanders": "WAS",
}

_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_NONALPHA = re.compile(r"[^a-z ]")


def normalize_team(code: str | None) -> str | None:
    if not code or (isinstance(code, float) and pd.isna(code)):
        return None
    code = str(code).strip().upper()
    if code in {"FA", "FA*", "NAN", ""}:
        return None
    return TEAM_ALIASES.get(code, code)


def normalize_name(name: str | None) -> str:
    """Fold a display name to a comparable key: no accents, punctuation, or suffix."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace(".", "").replace("'", "").replace("-", " ")
    text = _NONALPHA.sub(" ", text)
    text = _SUFFIXES.sub(" ", text)
    return " ".join(text.split())


def _as_id(series: pd.Series) -> pd.Series:
    """DynastyProcess writes numeric IDs as floats; make them strings again."""
    out = pd.to_numeric(series, errors="coerce")
    return out.astype("Int64").astype(str).replace("<NA>", pd.NA)


@lru_cache(maxsize=1)
def _dynastyprocess_ids() -> pd.DataFrame:
    path = fetch_file(
        f"{DYNASTYPROCESS_RAW}/db_playerids.csv", get_settings().ttl_static
    )
    df = pd.read_csv(path, low_memory=False)
    df["sleeper_id"] = _as_id(df["sleeper_id"])
    df["fantasypros_id"] = _as_id(df["fantasypros_id"])
    if "espn_id" in df.columns:
        df["espn_id"] = _as_id(df["espn_id"])
    df["gsis_id"] = df["gsis_id"].astype(str).replace({"nan": pd.NA, "NA": pd.NA})
    df["name_key"] = df["name"].map(normalize_name)
    df["position"] = df["position"].astype(str).str.upper()
    df["team"] = df["team"].map(normalize_team)
    return df


@dataclass
class Crosswalk:
    """Bidirectional ID maps plus a name-based fallback."""

    sleeper_to_gsis: dict[str, str] = field(default_factory=dict)
    gsis_to_sleeper: dict[str, str] = field(default_factory=dict)
    sleeper_to_fp: dict[str, str] = field(default_factory=dict)
    fp_to_sleeper: dict[str, str] = field(default_factory=dict)
    #: ESPN player_id -> sleeper_id, for reading ESPN-hosted leagues.
    espn_to_sleeper: dict[str, str] = field(default_factory=dict)
    #: (normalized name, position) -> sleeper_id
    name_to_sleeper: dict[tuple[str, str], str] = field(default_factory=dict)

    def gsis(self, sleeper_id: str) -> str | None:
        return self.sleeper_to_gsis.get(str(sleeper_id))

    def sleeper(self, gsis_id: str) -> str | None:
        return self.gsis_to_sleeper.get(str(gsis_id))

    def by_name(self, name: str, position: str) -> str | None:
        return self.name_to_sleeper.get((normalize_name(name), str(position).upper()))

    def from_espn(
        self, espn_id: object, name: str = "", position: str = "", team: str | None = None
    ) -> str | None:
        """Map an ESPN player onto the Sleeper id everything else is keyed by.

        Team defenses never appear in the ID crosswalk — ESPN gives them
        synthetic negative player IDs — but Sleeper keys them by team
        abbreviation, so the pro team is the identifier.

        The name fallback matters most in the days after the NFL draft, when a
        rookie is rostered on ESPN before the crosswalk has caught up.
        """
        if str(position).upper() in {"DST", "DEF", "D/ST"}:
            return normalize_team(team)
        key = str(espn_id or "").strip()
        if key.endswith(".0"):
            key = key[:-2]
        mapped = self.espn_to_sleeper.get(key)
        if mapped:
            return mapped
        if name and position:
            return self.by_name(name, position)
        return None

    def resolve_from_fp(self, fp_id: str | None, name: str, position: str) -> str | None:
        """Map a FantasyPros row to a Sleeper id, DSTs included."""
        position = str(position).upper()
        if position in {"DST", "DEF", "D/ST"}:
            return TEAM_NAME_TO_ABBR.get(normalize_name(name).replace(" ", " "), None) or (
                TEAM_NAME_TO_ABBR.get(str(name).strip().lower())
            )
        if fp_id and str(fp_id) in self.fp_to_sleeper:
            return self.fp_to_sleeper[str(fp_id)]
        return self.by_name(name, position)

    def attach_gsis(self, df: pd.DataFrame, column: str = "sleeper_id") -> pd.DataFrame:
        out = df.copy()
        out["gsis_id"] = out[column].astype(str).map(self.sleeper_to_gsis)
        return out


def load_crosswalk(sleeper_players: dict[str, dict] | None = None) -> Crosswalk:
    """Build the crosswalk, preferring Sleeper's own ``gsis_id`` when present."""
    xw = Crosswalk()

    dp = _dynastyprocess_ids()
    for row in dp.itertuples(index=False):
        sid = getattr(row, "sleeper_id", None)
        if not isinstance(sid, str) or sid in {"<NA>", "nan"}:
            continue
        gsis = getattr(row, "gsis_id", None)
        if isinstance(gsis, str) and gsis.startswith("00-"):
            xw.sleeper_to_gsis.setdefault(sid, gsis)
            xw.gsis_to_sleeper.setdefault(gsis, sid)
        fp = getattr(row, "fantasypros_id", None)
        if isinstance(fp, str) and fp not in {"<NA>", "nan"}:
            xw.sleeper_to_fp.setdefault(sid, fp)
            xw.fp_to_sleeper.setdefault(fp, sid)
        espn = getattr(row, "espn_id", None)
        if isinstance(espn, str) and espn not in {"<NA>", "nan"}:
            xw.espn_to_sleeper.setdefault(espn, sid)
        key = (getattr(row, "name_key", ""), getattr(row, "position", ""))
        if key[0]:
            xw.name_to_sleeper.setdefault(key, sid)

    for sleeper_id, meta in (sleeper_players or {}).items():
        if not isinstance(meta, dict):
            continue
        gsis = meta.get("gsis_id")
        if isinstance(gsis, str) and gsis.startswith("00-"):
            # Sleeper is authoritative for its own IDs; overwrite the crosswalk.
            xw.sleeper_to_gsis[str(sleeper_id)] = gsis
            xw.gsis_to_sleeper[gsis] = str(sleeper_id)
        name = meta.get("full_name") or meta.get("last_name") or ""
        position = str(meta.get("position") or "").upper()
        if name and position:
            xw.name_to_sleeper.setdefault((normalize_name(name), position), str(sleeper_id))

    log.info(
        "crosswalk: %d sleeper->gsis, %d sleeper->fantasypros, %d espn->sleeper",
        len(xw.sleeper_to_gsis),
        len(xw.sleeper_to_fp),
        len(xw.espn_to_sleeper),
    )
    return xw
