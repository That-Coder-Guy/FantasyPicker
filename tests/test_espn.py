"""Reading an ESPN-hosted league.

These cover the translation layer, because that is where an ESPN bug would
hide. A wrong roster or a missing team is visible the moment the page loads;
scoring translated wrongly is not — the app keeps working and quietly ranks
players under rules that are not the league's. So the scoring cases here are
deliberately fussy about bands, aliases and unsupported settings.
"""

from __future__ import annotations

import json

import httpx
import pytest

from fantasypicker.credentials import (
    EspnCredentials,
    describe,
    forget_credentials,
    load_credentials,
    redact,
    save_credentials,
)
from fantasypicker.data.crosswalk import Crosswalk
from fantasypicker.espn.client import EspnAuthRequired, EspnClient, EspnLeagueNotFound
from fantasypicker.espn.ids import STAT_KEYS, slot_of, team_of
from fantasypicker.espn.league import (
    build_teams,
    current_week,
    find_my_roster_id,
    load_league,
    matchup_rows,
    parse_slots,
)
from fantasypicker.espn.scoring import describe_items, scoring_from_espn
from fantasypicker.sleeper.scoring import (
    DST_TERMS,
    OFFENSE_TERMS,
    PTS_ALLOW_BUCKETS,
    YDS_ALLOW_BUCKETS,
)

SCORING_ITEMS = [
    {"statId": 3, "points": 0.04},
    {"statId": 4, "points": 4},
    {"statId": 20, "points": -2},
    {"statId": 24, "points": 0.1},
    {"statId": 25, "points": 6},
    {"statId": 42, "points": 0.1},
    {"statId": 43, "points": 6},
    {"statId": 53, "points": 0.5},
    {"statId": 72, "points": -2},
]

LEAGUE = {
    "id": 999,
    "settings": {
        "name": "Test ESPN League",
        "size": 2,
        "rosterSettings": {
            "lineupSlotCounts": {
                "0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "16": 1, "17": 1,
                "20": 6, "21": 1,
            }
        },
        "scoringSettings": {"scoringItems": SCORING_ITEMS},
    },
    "status": {"currentMatchupPeriod": 3, "latestScoringPeriod": 3},
    "members": [
        {"id": "{SWID-A}", "displayName": "alice", "firstName": "Alice", "lastName": "A"},
        {"id": "{SWID-B}", "displayName": "bob"},
    ],
    "teams": [
        {
            "id": 1,
            "name": "Alice's Aces",
            "owners": ["{SWID-A}"],
            "record": {"overall": {"wins": 2, "losses": 1, "pointsFor": 250.5}},
            "roster": {
                "entries": [
                    {
                        "lineupSlotId": 0,
                        "playerPoolEntry": {
                            "player": {
                                "id": 3139477,
                                "fullName": "Patrick Mahomes",
                                "defaultPositionId": 1,
                                "proTeamId": 12,
                                "injuryStatus": "ACTIVE",
                            }
                        },
                    },
                    {
                        "lineupSlotId": 20,
                        "playerPoolEntry": {
                            "player": {
                                "id": 4241457,
                                "fullName": "Bench Guy",
                                "defaultPositionId": 3,
                                "proTeamId": 4,
                            }
                        },
                    },
                ]
            },
        },
        {
            "id": 2,
            "location": "Old",
            "nickname": "Style",
            "owners": ["{SWID-B}"],
            "record": {"overall": {"wins": 1, "losses": 2}},
            "roster": {"entries": []},
        },
    ],
}

SCHEDULE = {
    "schedule": [
        {
            "id": 7,
            "matchupPeriodId": 3,
            "home": {"teamId": 1, "totalPoints": 101.2},
            "away": {"teamId": 2, "totalPoints": 98.0},
        }
    ]
}


def crosswalk() -> Crosswalk:
    return Crosswalk(espn_to_sleeper={"3139477": "4034", "4241457": "6794"})


def stub_transport(routes: dict | None = None, *, status: int = 200):
    """Route on the ESPN view parameters, which is what distinguishes calls."""
    routes = routes or {}

    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"error": "nope"})
        views = set(request.url.params.get_list("view"))
        for key, payload in routes.items():
            if key in views:
                return httpx.Response(200, content=json.dumps(payload))
        return httpx.Response(200, content=json.dumps(LEAGUE))

    return httpx.MockTransport(handler)


def make_client(routes=None, *, status: int = 200, **kwargs) -> EspnClient:
    return EspnClient(
        client=httpx.AsyncClient(transport=stub_transport(routes, status=status)),
        **kwargs,
    )


# ------------------------------------------------------------------- scoring


def test_every_mapped_stat_points_at_a_term_the_engine_knows():
    """A typo here would silently drop a league's scoring rule."""
    known = (
        set(OFFENSE_TERMS)
        | set(DST_TERMS)
        | set(PTS_ALLOW_BUCKETS)
        | set(YDS_ALLOW_BUCKETS)
    )
    unknown = {sid: key for sid, (key, _) in STAT_KEYS.items() if key not in known}
    assert unknown == {}


def test_standard_espn_scoring_translates():
    rules = scoring_from_espn({"scoringItems": SCORING_ITEMS})
    assert rules.ppr == 0.5
    assert rules.passing_td_value == 4
    assert rules.settings["rush_td"] == 6
    assert rules.settings["fum_lost"] == -2


def test_a_position_override_becomes_a_per_position_bonus():
    """TE premium is the common ESPN override and must not apply to everyone."""
    rules = scoring_from_espn(
        {"scoringItems": [{"statId": 53, "points": 0.5, "pointsOverrides": {"6": 1.0}}]}
    )
    assert rules.ppr == 0.5
    assert rules.te_premium == 0.5


def test_disjoint_defensive_touchdowns_are_not_paid_twice():
    """INT-return and fumble-return TDs are halves of one box-score total.

    Summing them would pay 12 points for every defensive touchdown.
    """
    rules = scoring_from_espn(
        {"scoringItems": [{"statId": 103, "points": 6}, {"statId": 104, "points": 6}]}
    )
    assert rules.settings["def_td"] == 6


def test_duplicate_reception_ids_do_not_stack():
    rules = scoring_from_espn(
        {"scoringItems": [{"statId": 41, "points": 1}, {"statId": 53, "points": 1}]}
    )
    assert rules.ppr == 1.0


def test_a_fractional_counter_is_folded_into_the_value():
    """ESPN's "1/2 Sack" pays per half sack, so a point there is two per sack."""
    rules = scoring_from_espn({"scoringItems": [{"statId": 100, "points": 1}]})
    assert rules.settings["sack"] == 2.0


def test_espn_point_bands_map_to_their_own_buckets():
    """ESPN splits 14-17/18-21; Sleeper splits 14-20/21-27. They are not the same."""
    rules = scoring_from_espn(
        {"scoringItems": [{"statId": 92, "points": 1}, {"statId": 121, "points": 0}]}
    )
    assert rules.settings["pts_allow_14_17"] == 1
    assert "pts_allow_14_20" not in rules.settings


def test_settings_no_box_score_can_rebuild_are_reported_not_ignored():
    rules = scoring_from_espn(
        {"scoringItems": [{"statId": 3, "points": 0.04}, {"statId": 15, "points": 2}]}
    )
    assert any("40+ yard TD pass" in u for u in rules.unsupported)


def test_empty_scoring_falls_back_rather_than_scoring_nothing():
    rules = scoring_from_espn({"scoringItems": []})
    assert rules.settings  # standard defaults, not an empty rulebook


def test_describe_items_names_each_setting_and_its_target():
    rows = describe_items({"scoringItems": SCORING_ITEMS})
    by_id = {r["stat_id"]: r for r in rows}
    assert by_id[53]["label"] == "Each reception"
    assert by_id[53]["key"] == "rec"


# --------------------------------------------------------------------- slots


def test_lineup_slot_counts_become_roster_slots():
    slots, bench = parse_slots(LEAGUE["settings"]["rosterSettings"])
    assert [s.name for s in slots] == [
        "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "DEF", "K"
    ]
    assert bench == 7  # six bench plus one IR


def test_slots_with_no_projection_behind_them_become_bench():
    """A punter slot is real on ESPN and unprojectable here; it must not vanish."""
    slots, bench = parse_slots({"lineupSlotCounts": {"0": 1, "18": 1, "20": 2}})
    assert [s.name for s in slots] == ["QB"]
    assert bench == 3


def test_flex_slots_keep_their_eligibility():
    assert slot_of(23) == "FLEX"
    assert slot_of(7) == "SUPER_FLEX"
    assert slot_of(3) == "WRRB_FLEX"


def test_pro_team_ids_use_nflverse_abbreviations():
    assert team_of(14) == "LA"   # Rams, not LAR
    assert team_of(28) == "WAS"  # not WSH
    assert team_of(0) is None


# --------------------------------------------------------------------- teams


def test_teams_carry_names_records_and_lineups():
    teams, unresolved = build_teams(LEAGUE, crosswalk())
    assert teams[1].label == "Alice's Aces"
    assert teams[1].manager == "Alice A"
    assert teams[1].record == "2-1"
    assert teams[1].players == ["4034", "6794"]
    assert teams[1].starters == ["4034"]  # the bench player is not started
    assert unresolved == []


def test_a_team_named_the_old_way_still_gets_a_name():
    """ESPN used to split a team name into location and nickname."""
    teams, _ = build_teams(LEAGUE, crosswalk())
    assert teams[2].label == "Old Style"


def test_a_player_the_crosswalk_cannot_place_is_reported_not_dropped_silently():
    payload = json.loads(json.dumps(LEAGUE))
    payload["teams"][0]["roster"]["entries"].append(
        {
            "lineupSlotId": 2,
            "playerPoolEntry": {
                "player": {
                    "id": 99999,
                    "fullName": "Undrafted Rookie",
                    "defaultPositionId": 2,
                    "proTeamId": 9,
                }
            },
        }
    )
    teams, unresolved = build_teams(payload, crosswalk())
    assert "99999" not in teams[1].players
    assert [u["name"] for u in unresolved] == ["Undrafted Rookie"]


def test_a_team_defense_is_keyed_by_its_pro_team():
    """ESPN gives defenses synthetic IDs; every projection is keyed by team."""
    payload = json.loads(json.dumps(LEAGUE))
    payload["teams"][0]["roster"]["entries"] = [
        {
            "lineupSlotId": 16,
            "playerPoolEntry": {
                "player": {
                    "id": -16012,
                    "fullName": "Chiefs D/ST",
                    "defaultPositionId": 16,
                    "proTeamId": 12,
                }
            },
        }
    ]
    teams, unresolved = build_teams(payload, crosswalk())
    assert teams[1].players == ["KC"]
    assert unresolved == []


def test_an_injured_reserve_slot_is_not_startable():
    payload = json.loads(json.dumps(LEAGUE))
    payload["teams"][0]["roster"]["entries"][1]["lineupSlotId"] = 21
    teams, _ = build_teams(payload, crosswalk())
    assert "6794" in teams[1].reserve
    assert "6794" not in teams[1].active_players


def test_names_carry_forward_when_a_refresh_comes_back_thin():
    first, _ = build_teams(LEAGUE, crosswalk())
    thin = {"teams": [{"id": 1, "roster": {"entries": []}}], "members": []}
    second, _ = build_teams(thin, crosswalk(), previous=first)
    assert second[1].label == "Alice's Aces"


# ------------------------------------------------------------------ matchups


def test_the_schedule_becomes_sleeper_shaped_rows():
    """So LeagueContext.matchup_for needs no ESPN-specific branch."""
    teams, _ = build_teams(LEAGUE, crosswalk())
    rows = matchup_rows(SCHEDULE, teams, 3)
    assert [r["roster_id"] for r in rows] == [1, 2]
    assert len({r["matchup_id"] for r in rows}) == 1  # paired by a shared id
    assert rows[0]["starters"] == ["4034"]
    assert rows[0]["points"] == 101.2


def test_another_week_is_not_returned():
    teams, _ = build_teams(LEAGUE, crosswalk())
    assert matchup_rows(SCHEDULE, teams, 4) == []


def test_current_week_comes_from_the_league_status():
    assert current_week(LEAGUE) == 3
    assert current_week({}) == 1


def test_the_swid_cookie_identifies_the_users_own_team():
    assert find_my_roster_id(LEAGUE, "SWID-A") == 1
    assert find_my_roster_id(LEAGUE, "{SWID-B}") == 2
    assert find_my_roster_id(LEAGUE, None) is None


# -------------------------------------------------------------------- client


@pytest.mark.asyncio
async def test_load_league_builds_a_context_the_engines_understand():
    client = make_client({"mRoster": LEAGUE})
    league, unresolved = await load_league(client, "999", 2026)
    assert league.name == "Test ESPN League"
    assert league.team_count == 2
    assert [s.name for s in league.slots][:3] == ["QB", "RB", "RB"]
    assert league.scoring.ppr == 0.5
    assert league.teams[1].label == "Alice's Aces"
    assert unresolved == []


@pytest.mark.asyncio
async def test_a_missing_league_says_what_to_check():
    client = make_client({"mSettings": None, "mTeam": None})

    async def empty(*a, **k):
        return None

    client.league = empty  # type: ignore[method-assign]
    with pytest.raises(EspnLeagueNotFound) as caught:
        await load_league(client, "999", 2026)
    assert "leagueId" in str(caught.value)


@pytest.mark.asyncio
async def test_a_private_league_asks_for_cookies_rather_than_failing_obscurely():
    client = make_client(status=401)
    with pytest.raises(EspnAuthRequired) as caught:
        await client.league("999", 2026)
    message = str(caught.value)
    assert "espn_s2" in message and "SWID" in message
    assert "developer tools" in message


@pytest.mark.asyncio
async def test_rejected_cookies_say_they_are_probably_stale():
    client = make_client(status=401, espn_s2="abc", swid="{d}")
    with pytest.raises(EspnAuthRequired) as caught:
        await client.league("999", 2026)
    assert "expires them" in str(caught.value)


@pytest.mark.asyncio
async def test_a_401_is_not_retried_into_a_stale_answer():
    """An expired cookie will be rejected identically four more times."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={})

    client = EspnClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(EspnAuthRequired):
        await client.league("999", 2026)
    assert calls["n"] == 1


def test_a_past_season_uses_the_history_endpoint():
    """The same league is a different resource each year — a common confusion."""
    client = EspnClient()
    current = client._url("999", 2026, ("mTeam",))
    old = client._url("999", 2019, ("mTeam",))
    assert "/seasons/2026/segments/0/leagues/999" in current
    assert "leagueHistory/999" in old and "seasonId=2019" in old


def test_swid_is_wrapped_in_braces_however_it_was_pasted():
    assert EspnClient(espn_s2="x", swid="ABC")._cookies()["SWID"] == "{ABC}"
    assert EspnClient(espn_s2="x", swid="{ABC}")._cookies()["SWID"] == "{ABC}"


def test_no_cookies_means_no_cookie_header():
    assert EspnClient()._cookies() == {}


# --------------------------------------------------------------- credentials


def test_credentials_round_trip():
    save_credentials("999", EspnCredentials(espn_s2="secret-value", swid="ABC"))
    loaded = load_credentials("999")
    assert loaded is not None
    assert loaded.espn_s2 == "secret-value"
    assert loaded.swid == "{ABC}"  # normalised on the way in


def test_credentials_are_written_only_readable_by_this_user():
    """They are session cookies for the user's ESPN account."""
    import stat as stat_module

    from fantasypicker.config import get_settings

    save_credentials("999", EspnCredentials(espn_s2="secret", swid="ABC"))
    path = get_settings().home / "credentials.json"
    mode = stat_module.S_IMODE(path.stat().st_mode)
    assert mode & 0o077 == 0  # nothing for group or other


def test_credentials_live_outside_the_shareable_state_file():
    from fantasypicker.config import get_settings

    save_credentials("999", EspnCredentials(espn_s2="secret-value", swid="ABC"))
    state = get_settings().state_file
    if state.exists():
        assert "secret-value" not in state.read_text(encoding="utf-8")


def test_incomplete_credentials_are_not_stored():
    save_credentials("999", EspnCredentials(espn_s2="", swid="ABC"))
    assert load_credentials("999") is None


def test_forgetting_credentials_removes_them():
    save_credentials("999", EspnCredentials(espn_s2="secret", swid="ABC"))
    assert forget_credentials("999") is True
    assert load_credentials("999") is None


def test_redaction_shows_enough_to_recognise_but_not_to_use():
    assert redact("abcdefghijklmnop").startswith("abcd")
    assert "abcdefghijklmnop" not in redact("abcdefghijklmnop")
    assert redact(None) == "(none)"
    assert redact("short") == "*****"


def test_describing_credentials_never_returns_the_value():
    save_credentials("999", EspnCredentials(espn_s2="secret-value", swid="ABCDEFGH"))
    described = describe("999")
    assert described["stored"] is True
    assert "secret-value" not in json.dumps(described)


# ------------------------------------------------------------------- service


class _StubCrosswalk(Crosswalk):
    """The real crosswalk downloads a file; this one knows two players."""

    def __init__(self):
        super().__init__(espn_to_sleeper={"3139477": "4034", "4241457": "6794"})


@pytest.fixture
def espn_service(monkeypatch):
    """A PickerService wired to stub ESPN and Sleeper transports."""
    from fantasypicker.service import PickerService
    from fantasypicker.sleeper.client import SleeperClient

    from .test_sleeper import stub_transport as sleeper_stub

    def espn_client(*args, **kwargs):
        kwargs.pop("client", None)
        return EspnClient(
            client=httpx.AsyncClient(transport=stub_transport({"mRoster": LEAGUE})),
            **kwargs,
        )

    async def fake_sleeper_enter(self):
        self._client = httpx.AsyncClient(transport=sleeper_stub())
        self._owned = True
        return self

    monkeypatch.setattr("fantasypicker.platforms.EspnClient", espn_client)
    monkeypatch.setattr(SleeperClient, "__aenter__", fake_sleeper_enter)
    monkeypatch.setattr(
        "fantasypicker.service.load_crosswalk", lambda players=None: _StubCrosswalk()
    )
    monkeypatch.setattr(
        "fantasypicker.espn.league.load_crosswalk", lambda *a, **k: _StubCrosswalk()
    )
    monkeypatch.setattr("fantasypicker.platforms._crosswalk", _StubCrosswalk)
    monkeypatch.setattr("fantasypicker.service.load_expert_ranks", lambda *a, **k: None)
    return PickerService()


@pytest.mark.asyncio
async def test_connecting_to_espn_loads_the_league(espn_service):
    summary = await espn_service.connect_espn("999", season=2026)
    assert summary["platform"] == "espn"
    assert summary["name"] == "Test ESPN League"
    assert summary["scoring"].startswith("half PPR")
    assert espn_service.league.teams[1].label == "Alice's Aces"


@pytest.mark.asyncio
async def test_an_espn_league_is_remembered_as_an_espn_league(espn_service):
    """Reopening it against Sleeper's API would simply fail."""
    from fantasypicker.store import load_state

    await espn_service.connect_espn("999", season=2026)
    assert load_state().get("999").platform == "espn"


@pytest.mark.asyncio
async def test_cookies_given_at_connect_are_stored_for_next_time(espn_service):
    await espn_service.connect_espn(
        "999", season=2026, espn_s2="secret-value", swid="SWID-A"
    )
    stored = load_credentials("999")
    assert stored is not None and stored.espn_s2 == "secret-value"


@pytest.mark.asyncio
async def test_the_swid_cookie_selects_the_users_team_without_asking(espn_service):
    await espn_service.connect_espn("999", season=2026, espn_s2="x", swid="SWID-A")
    assert espn_service.league.my_roster_id == 1


@pytest.mark.asyncio
async def test_a_public_espn_league_needs_no_cookies_at_all(espn_service):
    summary = await espn_service.connect_espn("999", season=2026)
    assert summary["connected"] is True
    assert load_credentials("999") is None


@pytest.mark.asyncio
async def test_reopening_uses_espn_when_that_is_where_the_league_lives(espn_service):
    await espn_service.connect_espn("999", season=2026)
    fresh = type(espn_service)()
    summary = await fresh.reopen_last()
    assert summary is not None
    assert summary["platform"] == "espn"


@pytest.mark.asyncio
async def test_the_weekly_matchup_resolves_the_opponent_without_manual_entry(
    espn_service, monkeypatch
):
    """The whole point: an ESPN opponent's roster arrives on its own."""

    async def schedule(self, league_id, season, *, fresh=False):
        return SCHEDULE

    monkeypatch.setattr(EspnClient, "schedule", schedule)
    await espn_service.connect_espn("999", season=2026)
    pairing = await espn_service.league.matchup_for(espn_service.source, 3, 1)
    assert pairing is not None
    assert pairing.away.label == "Old Style"
    assert pairing.home_starters == ["4034"]


def test_an_espn_team_logo_is_carried_as_the_avatar():
    payload = json.loads(json.dumps(LEAGUE))
    payload["teams"][0]["logo"] = "https://g.espncdn.com/lm-static/logo.svg"
    teams, _ = build_teams(payload, crosswalk())
    assert teams[1].avatar_url == "https://g.espncdn.com/lm-static/logo.svg"
    assert teams[2].avatar_url is None


# ------------------------------------------------------------- team pictures


def _service_with_espn_league(credentials=None):
    from fantasypicker.platforms import EspnSource
    from fantasypicker.service import PickerService
    from fantasypicker.sleeper.league import Team

    service = PickerService()
    service.platform = "espn"
    service.source = EspnSource(2026, credentials)
    league_teams = {
        1: Team(
            roster_id=1,
            owner_id="u1",
            display_name="alice",
            team_name="Aces",
            avatar="https://mystique-api.fantasy.espn.com/apis/v1/domains/lm/images/abc",
        ),
        2: Team(
            roster_id=2,
            owner_id="u2",
            display_name="bob",
            team_name="Bears",
            avatar="https://g.espncdn.com/lm-static/logo-packs/core/bear.svg",
        ),
    }

    class _League:
        teams = league_teams

    service.league = _League()
    return service


def test_authenticated_espn_logos_are_routed_through_the_app():
    """A browser cannot attach ESPN cookies to an <img> on 127.0.0.1, so the
    mystique-hosted custom logos 401 when linked directly. Those go through
    the app; public logo-pack images stay direct."""
    service = _service_with_espn_league()
    assert service._avatar_link(service.league.teams[1]) == "/api/team-image/1"
    assert service._avatar_link(service.league.teams[2]) == (
        "https://g.espncdn.com/lm-static/logo-packs/core/bear.svg"
    )


def test_sleeper_avatars_are_never_proxied():
    service = _service_with_espn_league()
    service.platform = "sleeper"
    assert "sleepercdn" in service._avatar_link(service.league.teams[1]) or (
        service._avatar_link(service.league.teams[1]).startswith("https://")
    )


@pytest.mark.asyncio
async def test_the_image_proxy_sends_the_espn_cookies(monkeypatch):
    """The whole point: the server attaches the credentials the browser cannot."""
    import fantasypicker.service as service_module

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["cookie"] = request.headers.get("cookie", "")
        return httpx.Response(200, content=b"PNGDATA", headers={"content-type": "image/png"})

    real_client = httpx.AsyncClient

    def factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), cookies=kwargs.get("cookies"))

    monkeypatch.setattr(service_module.httpx, "AsyncClient", factory)
    creds = EspnCredentials(espn_s2="s2value", swid="ABC").normalised()
    service = _service_with_espn_league(creds)

    image = await service.team_image(1)
    assert image == (b"PNGDATA", "image/png")
    assert "espn_s2=s2value" in seen["cookie"]
    assert "SWID=" in seen["cookie"]
    # Second call is served from cache — no re-fetch, same answer.
    seen.clear()
    assert await service.team_image(1) == (b"PNGDATA", "image/png")
    assert seen == {}


@pytest.mark.asyncio
async def test_the_image_proxy_refuses_to_relay_non_images(monkeypatch):
    """A 200 that is secretly an HTML login page must not be served as a logo."""
    import fantasypicker.service as service_module

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>log in</html>", headers={"content-type": "text/html"})

    real_client = httpx.AsyncClient

    def factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(service_module.httpx, "AsyncClient", factory)
    service = _service_with_espn_league()
    assert await service.team_image(1) is None


@pytest.mark.asyncio
async def test_missing_teams_and_pictures_404_cleanly():
    service = _service_with_espn_league()
    assert await service.team_image(99) is None
    service.league.teams[1].avatar = None
    assert await service.team_image(1) is None


# ------------------------------------------------------------------ refreshing


@pytest.mark.asyncio
async def test_refresh_follows_espn_to_the_new_scoring_period(monkeypatch):
    """ESPN advances the week on its own clock.

    A week frozen at connect time keeps every later roster request asking
    about the old one, so the page stops changing — which reads as "the data
    is not updating".
    """
    from fantasypicker.platforms import EspnSource

    payload = json.loads(json.dumps(LEAGUE))
    payload["status"]["currentMatchupPeriod"] = 7
    asked = {}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def league(self, league_id, season, *, fresh=False):
            return payload

        async def rosters(self, league_id, season, week=None, *, fresh=False):
            asked["week"] = week
            return payload

    source = EspnSource(2026)
    monkeypatch.setattr(source, "client", lambda: _Client())
    monkeypatch.setattr(
        "fantasypicker.platforms._crosswalk", lambda: crosswalk()
    )

    league, _ = await load_league(make_client({"mRoster": LEAGUE}), "999", 2026)
    assert league.current_week == 3  # as it was at connect time

    await source.refresh(league)
    assert league.current_week == 7
    assert asked["week"] == 7


@pytest.mark.asyncio
async def test_an_expired_cookie_during_refresh_is_reported_not_swallowed(monkeypatch):
    """The failure mode behind "it stopped updating": ESPN starts refusing the
    stored cookies, the refresh fails, and the app quietly keeps serving the
    rosters it already had."""
    from fantasypicker.service import PickerService

    service = PickerService()
    service.platform = "espn"

    class _Source:
        platform = "espn"

        async def refresh(self, league, *, fresh=False):
            raise EspnAuthRequired("999", had_cookies=True)

    service.source = _Source()
    league, _ = await load_league(make_client({"mRoster": LEAGUE}), "999", 2026)
    service.league = league

    changed = await service.refresh_live(force=True)
    assert changed == {"rosters": False, "players": False}
    assert service.refresh_error is not None
    assert "espn_s2" in service.refresh_error


# ------------------------------------------------- reading entries ESPN sends


def test_a_player_carried_at_the_entry_level_is_still_read():
    """Some ESPN responses put only playerId on the entry, with no nested
    player object. Reading one shape only turned a full roster into zero
    players — the team looked empty and nothing said why."""
    payload = json.loads(json.dumps(LEAGUE))
    payload["teams"][0]["roster"]["entries"] = [
        {"lineupSlotId": 0, "playerId": 3139477},
        {"lineupSlotId": 20, "playerId": 4241457},
    ]
    teams, unresolved = build_teams(payload, crosswalk())
    assert teams[1].players == ["4034", "6794"]
    assert teams[1].starters == ["4034"]
    assert unresolved == []


def test_a_player_under_playerentry_is_read_too():
    payload = json.loads(json.dumps(LEAGUE))
    payload["teams"][0]["roster"]["entries"] = [
        {
            "lineupSlotId": 0,
            "playerEntry": {
                "player": {"id": 3139477, "fullName": "Patrick Mahomes",
                           "defaultPositionId": 1, "proTeamId": 12}
            },
        }
    ]
    teams, _ = build_teams(payload, crosswalk())
    assert teams[1].players == ["4034"]


def test_a_name_split_across_first_and_last_still_resolves():
    payload = json.loads(json.dumps(LEAGUE))
    payload["teams"][0]["roster"]["entries"] = [
        {
            "lineupSlotId": 0,
            "playerPoolEntry": {
                "player": {
                    "id": 999999,  # not in the crosswalk; must fall back to name
                    "firstName": "Patrick",
                    "lastName": "Mahomes",
                    "defaultPositionId": 1,
                    "proTeamId": 12,
                }
            },
        }
    ]
    xw = Crosswalk(name_to_sleeper={("patrick mahomes", "QB"): "4034"})
    teams, unresolved = build_teams(payload, xw)
    assert teams[1].players == ["4034"]
    assert unresolved == []


def test_an_unreadable_entry_is_reported_rather_than_vanishing():
    """The bug behind "all teams have 0 players": an entry that resolved to
    nothing and had no name disappeared from the roster AND from the
    unresolved list, leaving no evidence it ever existed."""
    payload = json.loads(json.dumps(LEAGUE))
    payload["teams"][0]["roster"]["entries"] = [{"lineupSlotId": 0, "playerId": 424242}]
    teams, unresolved = build_teams(payload, crosswalk())
    assert teams[1].players == []
    assert len(unresolved) == 1
    assert unresolved[0]["espn_id"] == 424242


def test_raw_entry_counts_separate_an_empty_roster_from_an_unread_one():
    payload = json.loads(json.dumps(LEAGUE))
    from fantasypicker.espn.league import raw_entry_counts

    counts = raw_entry_counts(payload)
    assert counts[1] == 2  # ESPN sent two
    assert counts[2] == 0  # genuinely empty


def test_a_roster_that_reads_as_empty_despite_entries_is_logged(caplog):
    payload = json.loads(json.dumps(LEAGUE))
    payload["teams"][0]["roster"]["entries"] = [{"lineupSlotId": 0, "playerId": 424242}]
    with caplog.at_level("WARNING"):
        build_teams(payload, crosswalk())
    assert "none could be read as players" in caplog.text


@pytest.mark.asyncio
async def test_load_retries_with_the_week_when_rosters_come_back_bare():
    """ESPN can answer the roster view with teams but no entries when no
    scoring period is given. Accepting that reports every team as empty."""
    bare = json.loads(json.dumps(LEAGUE))
    for row in bare["teams"]:
        row["roster"] = {"entries": []}
    asked: list = []

    class _Client:
        async def league(self, league_id, season, *, fresh=False):
            return LEAGUE

        async def rosters(self, league_id, season, week=None, *, fresh=False):
            asked.append(week)
            return bare if week is None else LEAGUE

    league, _ = await load_league(_Client(), "999", 2026)
    assert asked == [None, 3]  # retried with the league's current week
    # Two entries came back on the retry; the ids themselves come from the
    # real crosswalk, so only the count is this test's business.
    assert len(league.teams[1].players) == 2


# ------------------------------------------------------------ before the draft


def test_draft_status_is_read_when_espn_reports_it():
    from fantasypicker.espn.league import has_drafted

    assert has_drafted({"draftDetail": {"drafted": True}}) is True
    assert has_drafted({"draftDetail": {"drafted": False}}) is False
    # Unknown is its own answer: never claim a league has drafted on a guess.
    assert has_drafted({}) is None
    assert has_drafted({"draftDetail": {}}) is None


@pytest.mark.asyncio
async def test_a_league_that_has_not_drafted_is_not_retried_for_rosters():
    """Empty rosters before the draft are correct, not a failure to load.

    Retrying every load through the whole preseason would double the request
    count to prove something ESPN already told us.
    """
    predraft = json.loads(json.dumps(LEAGUE))
    predraft["draftDetail"] = {"drafted": False}
    for row in predraft["teams"]:
        row["roster"] = {"entries": []}
    asked: list = []

    class _Client:
        async def league(self, league_id, season, *, fresh=False):
            return predraft

        async def rosters(self, league_id, season, week=None, *, fresh=False):
            asked.append(week)
            return predraft

    league, unresolved = await load_league(_Client(), "999", 2026)
    assert asked == [None]  # asked once, believed the answer
    assert all(not t.players for t in league.teams.values())
    assert unresolved == []


@pytest.mark.asyncio
async def test_a_drafted_league_with_empty_rosters_is_still_retried():
    drafted = json.loads(json.dumps(LEAGUE))
    drafted["draftDetail"] = {"drafted": True}
    bare = json.loads(json.dumps(drafted))
    for row in bare["teams"]:
        row["roster"] = {"entries": []}
    asked: list = []

    class _Client:
        async def league(self, league_id, season, *, fresh=False):
            return drafted

        async def rosters(self, league_id, season, week=None, *, fresh=False):
            asked.append(week)
            return bare if week is None else drafted

    league, _ = await load_league(_Client(), "999", 2026)
    assert asked == [None, 3]
    assert len(league.teams[1].players) == 2


# --------------------------------------------------- rosters during the draft


DRAFT_PAYLOAD = {
    "settings": {
        "size": 2,
        "draftSettings": {"type": "SNAKE", "pickOrder": [1, 2]},
        "rosterSettings": {"lineupSlotCounts": {"0": 1, "2": 2, "20": 6}},
    },
    "draftDetail": {
        "drafted": False,
        "inProgress": True,
        "pickOrder": [1, 2],
        "picks": [
            {"teamId": 1, "playerId": 3139477, "overallPickNumber": 1},
            {"teamId": 2, "playerId": 4241457, "overallPickNumber": 2},
            {"teamId": 1, "playerId": -16012, "overallPickNumber": 3},
        ],
    },
}


def test_a_drafted_defense_is_identified_from_its_synthetic_id():
    """The pick feed carries only a player id, and ESPN numbers defenses
    negatively — without undoing that, every drafted D/ST is lost."""
    from fantasypicker.espn.ids import dst_team_from_player_id

    assert dst_team_from_player_id(-16012) == "KC"
    assert dst_team_from_player_id(-16001) == "ATL"
    assert dst_team_from_player_id(3139477) is None
    assert dst_team_from_player_id(None) is None


def test_picks_become_rosters_keyed_by_team(monkeypatch):
    from fantasypicker.platforms import _espn_draft, picks_by_roster

    monkeypatch.setattr("fantasypicker.platforms._crosswalk", crosswalk)

    from fantasypicker.sleeper.league import Team

    class _League:
        teams = {
            1: Team(roster_id=1, owner_id="a", display_name="A", team_name="A"),
            2: Team(roster_id=2, owner_id="b", display_name="B", team_name="B"),
        }
        league_id = "999"
        roster_size = 9

    _, picks = _espn_draft(DRAFT_PAYLOAD, _League())
    by_roster = picks_by_roster(picks)
    assert by_roster[1] == ["4034", "KC"]  # includes the drafted defense
    assert by_roster[2] == ["6794"]


def test_draft_picks_fill_empty_rosters_but_never_overwrite_real_ones():
    """During a draft the pick feed is the only live account of who owns whom;
    once the draft is done the real rosters carry waivers and trades too, so
    they must win."""
    from fantasypicker.platforms import apply_draft_rosters
    from fantasypicker.sleeper.league import Team

    class _League:
        teams = {
            1: Team(roster_id=1, owner_id="a", display_name="A", team_name="A"),
            2: Team(
                roster_id=2, owner_id="b", display_name="B", team_name="B",
                players=["already", "here"],
            ),
        }

    league = _League()
    changed = apply_draft_rosters(league, {1: ["x", "y"], 2: ["ignored"]})
    assert changed is True
    assert league.teams[1].players == ["x", "y"]
    assert league.teams[2].players == ["already", "here"]  # untouched
    # Nothing left to fill: a second pass reports no change.
    assert apply_draft_rosters(league, {1: ["x", "y"]}) is False


def test_applying_an_empty_draft_feed_changes_nothing():
    from fantasypicker.platforms import apply_draft_rosters
    from fantasypicker.sleeper.league import Team

    class _League:
        teams = {1: Team(roster_id=1, owner_id="a", display_name="A", team_name="A")}

    assert apply_draft_rosters(_League(), {}) is False
    assert apply_draft_rosters(_League(), {1: []}) is False


def test_an_unmade_pick_is_a_placeholder_not_a_failure():
    """ESPN pre-allocates the whole board and marks slots it has not reached
    with playerId -1. Counting those as unreadable made a draft in its early
    rounds look like a total parsing failure."""
    from fantasypicker.espn.ids import is_unmade_pick

    assert is_unmade_pick(-1) is True
    assert is_unmade_pick(0) is True
    assert is_unmade_pick(None) is True
    assert is_unmade_pick(3139477) is False
    assert is_unmade_pick(-16012) is False  # a defense, not a placeholder


def test_unmade_picks_are_skipped_when_rebuilding_rosters(monkeypatch):
    from fantasypicker.platforms import _espn_draft, picks_by_roster
    from fantasypicker.sleeper.league import Team

    monkeypatch.setattr("fantasypicker.platforms._crosswalk", crosswalk)
    payload = json.loads(json.dumps(DRAFT_PAYLOAD))
    payload["draftDetail"]["picks"] = [
        {"teamId": 1, "playerId": 3139477, "overallPickNumber": 1},
        {"teamId": 2, "playerId": -1, "overallPickNumber": 2},
        {"teamId": 1, "playerId": -1, "overallPickNumber": 3},
    ]

    class _League:
        teams = {
            1: Team(roster_id=1, owner_id="a", display_name="A", team_name="A"),
            2: Team(roster_id=2, owner_id="b", display_name="B", team_name="B"),
        }
        league_id = "999"
        roster_size = 9

    _, picks = _espn_draft(payload, _League())
    assert len(picks) == 1
    assert picks_by_roster(picks) == {1: ["4034"]}
