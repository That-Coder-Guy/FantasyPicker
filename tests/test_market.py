"""The second currency: what the rest of the league sees.

Every test here is about one distinction — the number that is *true* versus the
number the other manager is *looking at*. Confusing them is the failure mode
this whole module exists to prevent, so most of these check that a value never
leaks from one currency into the other.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fantasypicker import market as market_source
from fantasypicker.data.crosswalk import Crosswalk
from fantasypicker.espn import projections as espn_projections
from fantasypicker.espn.projections import MarketPoints

CROSSWALK = Crosswalk(espn_to_sleeper={"3139477": "4034", "4241457": "6794"})
SEASON = 2026


def stat(
    *,
    source: int = 1,
    split: int = 0,
    season: int = SEASON,
    total: float | None = 200.0,
    average: float | None = None,
) -> dict:
    row: dict = {
        "statSourceId": source,
        "statSplitTypeId": split,
        "seasonId": season,
    }
    if total is not None:
        row["appliedTotal"] = total
    if average is not None:
        row["appliedAverage"] = average
    return row


def player(espn_id: str, stats: list[dict], *, position: int = 2) -> dict:
    return {
        "id": espn_id,
        "fullName": "Test Player",
        "defaultPositionId": position,
        "proTeamId": 1,
        "stats": stats,
    }


# --------------------------------------------------------------- ESPN parsing


def test_the_season_projection_is_read_from_the_stats_block():
    payload = {
        "players": [{"player": player("3139477", [stat(total=248.6, average=14.6)])}]
    }
    out = espn_projections.from_player_pool(payload, CROSSWALK, season=SEASON)
    assert out["4034"].total == pytest.approx(248.6)
    assert out["4034"].per_game == pytest.approx(14.6)


def test_actual_results_are_not_mistaken_for_a_projection():
    """statSourceId 0 is what already happened, and pricing a trade on it would
    value a player by a season he has finished playing."""
    payload = {"players": [{"player": player("3139477", [stat(source=0, total=300.0)])}]}
    assert espn_projections.from_player_pool(payload, CROSSWALK, season=SEASON) == {}


def test_a_weekly_projection_is_not_read_as_a_season_total():
    payload = {"players": [{"player": player("3139477", [stat(split=1, total=18.0)])}]}
    assert espn_projections.from_player_pool(payload, CROSSWALK, season=SEASON) == {}


def test_last_seasons_projection_is_ignored():
    """A September payload still carries last year's lines."""
    payload = {
        "players": [{"player": player("3139477", [stat(season=SEASON - 1, total=310.0)])}]
    }
    assert espn_projections.from_player_pool(payload, CROSSWALK, season=SEASON) == {}


def test_a_per_game_average_is_derived_when_espn_omits_it():
    payload = {"players": [{"player": player("3139477", [stat(total=170.0)])}]}
    out = espn_projections.from_player_pool(payload, CROSSWALK, season=SEASON)
    assert out["4034"].per_game == pytest.approx(170.0 / 17)


def test_a_total_is_derived_when_only_an_average_is_given():
    payload = {
        "players": [{"player": player("3139477", [stat(total=None, average=12.0)])}]
    }
    out = espn_projections.from_player_pool(payload, CROSSWALK, season=SEASON)
    assert out["4034"].total == pytest.approx(12.0 * 17)


def test_a_player_the_crosswalk_cannot_place_is_skipped_not_guessed():
    payload = {"players": [{"player": player("999999", [stat(total=200.0)])}]}
    assert espn_projections.from_player_pool(payload, CROSSWALK, season=SEASON) == {}


def test_projections_are_read_from_the_roster_payload_too():
    """The rosters are fetched anyway; the projections ride along for free."""
    payload = {
        "teams": [
            {
                "roster": {
                    "entries": [
                        {
                            "playerPoolEntry": {
                                "player": player("3139477", [stat(total=210.0)])
                            }
                        }
                    ]
                }
            }
        ]
    }
    out = espn_projections.from_rosters(payload, CROSSWALK, season=SEASON)
    assert out["4034"].total == pytest.approx(210.0)


def test_a_bare_player_shape_still_parses():
    """ESPN's pool sometimes hands back the player without the wrapper."""
    payload = {"players": [player("4241457", [stat(total=155.0)])]}
    out = espn_projections.from_player_pool(payload, CROSSWALK, season=SEASON)
    assert "6794" in out


def test_junk_never_raises():
    assert espn_projections.from_player_pool(None, CROSSWALK, season=SEASON) == {}
    assert espn_projections.from_rosters({"teams": [None]}, CROSSWALK, season=SEASON) == {}
    payload = {"players": [{"player": {"id": "3139477", "stats": ["nonsense"]}}]}
    assert espn_projections.from_player_pool(payload, CROSSWALK, season=SEASON) == {}


# ------------------------------------------------------------------ prorating


def test_a_season_total_is_prorated_onto_the_rest_of_season_basis():
    """Mid-season, a published season total overstates what is left."""
    published = {"a": MarketPoints(total=170.0, per_game=10.0)}
    result = market_source.from_platform(
        published, {"a": 8.0}, source=market_source.ESPN
    )
    assert result.get("a") == pytest.approx(80.0)
    assert result.available is True
    assert result.source == market_source.ESPN


def test_before_week_one_the_prorated_number_is_the_published_one():
    """During a draft the two bases coincide, so the number on screen should be
    exactly the number on espn.com."""
    published = {"a": MarketPoints(total=17 * 10.0, per_game=10.0)}
    result = market_source.from_platform(
        published, {"a": 17.0}, source=market_source.ESPN
    )
    assert result.get("a") == pytest.approx(170.0)


def test_a_player_with_no_game_count_is_dropped_rather_than_guessed():
    published = {"a": MarketPoints(total=170.0, per_game=10.0)}
    result = market_source.from_platform(published, {}, source=market_source.ESPN)
    assert not result.covers("a")


def test_thin_coverage_falls_back_to_the_model_entirely():
    """Half a source is worse than none: it would price one side of a trade in
    public points and the other in zeros."""
    published = {"a": MarketPoints(total=170.0, per_game=10.0)}
    model = {f"p{i}": 100.0 for i in range(10)} | {"a": 90.0}
    result = market_source.from_platform(
        published, {"a": 17.0}, source=market_source.ESPN, model_points=model
    )
    assert result.available is False
    assert result.source == market_source.MODEL
    assert any("only" in note for note in result.notes)


def test_partial_coverage_is_kept_but_disclosed():
    published = {f"p{i}": MarketPoints(total=170.0, per_game=10.0) for i in range(9)}
    games = {f"p{i}": 17.0 for i in range(9)}
    model = {f"p{i}": 100.0 for i in range(10)}
    result = market_source.from_platform(
        published, games, source=market_source.ESPN, model_points=model
    )
    assert result.available is True
    assert any("no ESPN projection" in note for note in result.notes)


def test_the_model_standing_in_is_never_labelled_as_public():
    result = market_source.from_model({"a": 100.0})
    assert result.available is False
    assert result.source == market_source.MODEL
    assert result.get("a") == pytest.approx(100.0)


def test_a_weekly_rate_becomes_a_rest_of_season_total():
    result = market_source.from_platform(
        {"a": market_source.WeeklyRate(12.5)},
        {"a": 6.0},
        source=market_source.CONSENSUS,
    )
    assert result.get("a") == pytest.approx(75.0)


def test_coverage_reports_what_fraction_is_known():
    result = market_source.from_platform(
        {"a": MarketPoints(total=17.0, per_game=1.0)},
        {"a": 17.0},
        source=market_source.ESPN,
    )
    assert result.coverage(["a", "b"]) == pytest.approx(0.5)
    assert result.coverage([]) == 1.0


# -------------------------------------------------------------------- service


@pytest.mark.asyncio
async def test_a_stale_consensus_scrape_is_refused(monkeypatch):
    """Last season's weekly points would price this season's rosters on games
    that have already been played."""
    from fantasypicker.service import PickerService

    stale = pd.DataFrame(
        {
            "sleeper_id": ["a"],
            "expert_points": [15.0],
            "scrape_date": [pd.Timestamp.now() - pd.Timedelta(days=400)],
        }
    )
    monkeypatch.setattr(
        "fantasypicker.service.load_weekly_expert_points", lambda cw: stale
    )
    service = PickerService()
    service.crosswalk = CROSSWALK
    assert service._consensus_projections({"a": 17.0}) is None


@pytest.mark.asyncio
async def test_a_fresh_consensus_scrape_is_used(monkeypatch):
    from fantasypicker.service import PickerService

    fresh = pd.DataFrame(
        {
            "sleeper_id": ["a"],
            "expert_points": [15.0],
            "scrape_date": [pd.Timestamp.now() - pd.Timedelta(days=1)],
        }
    )
    monkeypatch.setattr(
        "fantasypicker.service.load_weekly_expert_points", lambda cw: fresh
    )
    service = PickerService()
    service.crosswalk = CROSSWALK
    result = service._consensus_projections({"a": 10.0})
    assert result is not None
    assert result.source == market_source.CONSENSUS
    assert result.get("a") == pytest.approx(150.0)
