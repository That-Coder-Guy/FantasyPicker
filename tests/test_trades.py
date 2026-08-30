"""The trade engine.

The premise under test: a player's value depends on the roster he lands on, so
trades exist that help both sides, and only those should ever be proposed. The
scenarios are built so the right answer is obvious to a human reading the
rosters — RB-rich meets WR-rich — and the assertions check the engine sees the
same thing, refuses the deals a human would refuse, and finds the two-step
sequence a human would need a whiteboard for.
"""

from __future__ import annotations

import pytest

from fantasypicker.engine.trades import find_trades
from fantasypicker.model.predict import ProjectionSet
from fantasypicker.sleeper.league import Team

from .conftest import make_league, make_projection_frame

QUANTILES = (0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95)


def projections(players: list[dict]) -> ProjectionSet:
    frame = make_projection_frame(
        [{**p, "spread": 10.0} for p in players]
    )
    return ProjectionSet(frame, QUANTILES, season=2026, week=None)


def team(roster_id: int, name: str, players: list[str]) -> Team:
    return Team(
        roster_id=roster_id,
        owner_id=f"u{roster_id}",
        display_name=name,
        team_name=name,
        players=players,
        starters=[],
    )


def _staples(prefix: str, qb: float = 300.0) -> list[dict]:
    """A QB, kicker and defense so every lineup fills the same way."""
    return [
        {"id": f"{prefix}_qb", "position": "QB", "mean": qb},
        {"id": f"{prefix}_k", "position": "K", "mean": 110.0},
        {"id": f"{prefix}_dst", "position": "DST", "mean": 105.0},
    ]


def _ids(rows: list[dict]) -> list[str]:
    return [str(r["id"]) for r in rows]


# ------------------------------------------------------------- mutual benefit


def surplus_league(scoring):
    """Team 1 hoards RBs, team 2 hoards WRs. The trade writes itself."""
    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_rb3", "position": "RB", "mean": 180.0},
        {"id": "m_rb4", "position": "RB", "mean": 170.0},  # 4th RB: pure surplus
        {"id": "m_wr1", "position": "WR", "mean": 80.0},
        {"id": "m_wr2", "position": "WR", "mean": 70.0},
        {"id": "m_te", "position": "TE", "mean": 100.0},
    ]
    theirs = _staples("t") + [
        {"id": "t_wr1", "position": "WR", "mean": 200.0},
        {"id": "t_wr2", "position": "WR", "mean": 190.0},
        {"id": "t_wr3", "position": "WR", "mean": 180.0},
        {"id": "t_wr4", "position": "WR", "mean": 170.0},  # 4th WR: pure surplus
        {"id": "t_rb1", "position": "RB", "mean": 80.0},
        {"id": "t_rb2", "position": "RB", "mean": 70.0},
        {"id": "t_te", "position": "TE", "mean": 100.0},
    ]
    league = make_league(
        scoring,
        teams={
            1: team(1, "RB Hoarders", _ids(mine)),
            2: team(2, "WR Hoarders", _ids(theirs)),
        },
        team_count=2,
    )
    return league, projections(mine + theirs)


def test_surplus_for_need_is_found_and_helps_both_sides(scoring):
    league, projs = surplus_league(scoring)
    report = find_trades(league, projs, my_roster_id=1)
    assert report.trades, "the obvious RB-for-WR trade was not found"
    top = report.trades[0]
    assert top.them.roster_id == 2
    # Both lineups get meaningfully better — that is what makes it acceptable.
    assert top.me.gain > 30
    assert top.them.gain > 30
    # And the flow is the obvious one: RBs out, WRs in.
    assert any(p.startswith("m_rb") for p in top.me.gives)
    assert any(p.startswith("t_wr") for p in top.them.gives)


def test_the_other_sides_gain_is_reported_honestly(scoring):
    """The rosters are mirror images, so the advice must be too.

    Whoever asks, the top trade is the same shape and the two sides' numbers
    swap roles — if not, the engine is flattering whoever is asking.
    """
    league, projs = surplus_league(scoring)
    mine = find_trades(league, projs, my_roster_id=1).trades[0]
    theirs = find_trades(league, projs, my_roster_id=2).trades[0]
    assert mine.me.gain == pytest.approx(theirs.me.gain, abs=1.0)
    assert mine.them.gain == pytest.approx(theirs.them.gain, abs=1.0)


def test_rationale_speaks_in_positions_and_points(scoring):
    league, projs = surplus_league(scoring)
    top = find_trades(league, projs, my_roster_id=1).trades[0]
    assert "RB" in top.rationale and "WR" in top.rationale
    assert "rest-of-season" in top.rationale


# -------------------------------------------------------------- refusing bad


def test_no_trade_is_proposed_when_only_i_would_benefit(scoring):
    """My roster is bad everywhere; theirs is better everywhere.

    Every swap I want makes their lineup worse, so nothing should survive —
    an engine that proposes these gets its user laughed out of the league chat.
    """
    mine = _staples("m", qb=150.0) + [
        {"id": f"m_p{i}", "position": pos, "mean": 50.0 + i}
        for i, pos in enumerate(["RB", "RB", "WR", "WR", "TE", "RB", "WR"])
    ]
    theirs = _staples("t") + [
        {"id": f"t_p{i}", "position": pos, "mean": 200.0 + i}
        for i, pos in enumerate(["RB", "RB", "WR", "WR", "TE", "RB", "WR"])
    ]
    league = make_league(
        scoring,
        teams={1: team(1, "Bad", _ids(mine)), 2: team(2, "Good", _ids(theirs))},
        team_count=2,
    )
    report = find_trades(league, projections(mine + theirs), my_roster_id=1)
    assert report.trades == []
    assert any("hurts the other roster" in n or "clears the bar" in n for n in report.notes)


def test_identical_rosters_produce_no_trades(scoring):
    """Zero value spread means zero trades — not a list of pointless swaps."""
    rows_a = _staples("a") + [
        {"id": f"a_p{i}", "position": pos, "mean": 150.0}
        for i, pos in enumerate(["RB", "RB", "WR", "WR", "TE"])
    ]
    rows_b = _staples("b") + [
        {"id": f"b_p{i}", "position": pos, "mean": 150.0}
        for i, pos in enumerate(["RB", "RB", "WR", "WR", "TE"])
    ]
    league = make_league(
        scoring,
        teams={1: team(1, "A", _ids(rows_a)), 2: team(2, "B", _ids(rows_b))},
        team_count=2,
    )
    report = find_trades(league, projections(rows_a + rows_b), my_roster_id=1)
    assert report.trades == []


def test_no_proposal_asks_either_side_to_dump_its_headliner(scoring):
    """The headline rule, checked on real output: whatever a side surrenders,
    the package coming back must be worth roughly its best player. Lineup math
    alone would happily violate this; managers never do."""
    from fantasypicker.engine.trades import HEADLINE_FACTOR, PACKAGE_DISCOUNT

    league, projs = surplus_league(scoring)
    values = dict(
        zip(projs.frame["sleeper_id"].astype(str), projs.frame["exp_points"])
    )

    def effective(ids):
        ordered = sorted((values[p] for p in ids), reverse=True)
        return ordered[0] + PACKAGE_DISCOUNT * sum(ordered[1:])

    report = find_trades(league, projs, my_roster_id=1)
    assert report.trades
    for trade in report.trades:
        their_headliner = max(values[p] for p in trade.them.gives)
        my_headliner = max(values[p] for p in trade.me.gives)
        assert effective(trade.me.gives) >= their_headliner * HEADLINE_FACTOR - 1e-6
        assert effective(trade.them.gives) >= my_headliner * HEADLINE_FACTOR - 1e-6


# ------------------------------------------------------------- consolidation


def test_two_for_one_consolidation_backfills_from_waivers(scoring):
    """Giving two mid players for one stud only works because the freed roster
    spot refills from free agency — the engine must count that, both ways."""
    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 150.0},
        {"id": "m_rb2", "position": "RB", "mean": 140.0},
        {"id": "m_wr1", "position": "WR", "mean": 160.0},
        {"id": "m_wr2", "position": "WR", "mean": 150.0},
        {"id": "m_te", "position": "TE", "mean": 100.0},
    ]
    theirs = _staples("t") + [
        {"id": "t_rb_stud", "position": "RB", "mean": 210.0},
        {"id": "t_rb2", "position": "RB", "mean": 60.0},
        {"id": "t_wr1", "position": "WR", "mean": 160.0},
        {"id": "t_wr2", "position": "WR", "mean": 150.0},
        {"id": "t_te", "position": "TE", "mean": 100.0},
    ]
    free_agents = [
        {"id": "fa_rb", "position": "RB", "mean": 120.0},
        {"id": "fa_wr", "position": "WR", "mean": 115.0},
    ]
    league = make_league(
        scoring,
        teams={1: team(1, "Me", _ids(mine)), 2: team(2, "Them", _ids(theirs))},
        team_count=2,
    )
    report = find_trades(league, projections(mine + theirs + free_agents), my_roster_id=1)
    stud_deals = [
        t
        for t in report.trades
        if t.them.gives == ["t_rb_stud"] and len(t.me.gives) == 2
    ]
    assert stud_deals, "no 2-for-1 for the stud found"
    deal = stud_deals[0]
    assert deal.me.adds, "my freed spot should be backfilled from free agency"
    assert deal.them.drops, "their extra body should be dropped"
    assert deal.me.gain > 0 and deal.them.gain > 0


# -------------------------------------------------------------------- chains


def chain_league(scoring):
    """A follow-up trade that only makes sense after the first one.

    Step 1: my surplus RB for team 2's surplus WR (big, obvious).
    Step 2: my old starting WR — benched by step 1's arrival — goes to team 3
    for a TE upgrade. Before step 1 that WR was starting, so the same offer
    would have gutted my lineup; the chain is what makes it cheap.
    """
    mine = _staples("m") + [
        {"id": "m_rb1", "position": "RB", "mean": 200.0},
        {"id": "m_rb2", "position": "RB", "mean": 190.0},
        {"id": "m_rb3", "position": "RB", "mean": 180.0},
        {"id": "m_rb4", "position": "RB", "mean": 170.0},
        {"id": "m_wr1", "position": "WR", "mean": 90.0},
        {"id": "m_wr2", "position": "WR", "mean": 85.0},
        {"id": "m_te", "position": "TE", "mean": 60.0},
    ]
    team2 = _staples("t") + [
        {"id": "t_wr1", "position": "WR", "mean": 200.0},
        {"id": "t_wr2", "position": "WR", "mean": 190.0},
        {"id": "t_wr3", "position": "WR", "mean": 180.0},
        {"id": "t_wr4", "position": "WR", "mean": 170.0},
        {"id": "t_rb1", "position": "RB", "mean": 80.0},
        {"id": "t_rb2", "position": "RB", "mean": 70.0},
        {"id": "t_te", "position": "TE", "mean": 100.0},
    ]
    team3 = _staples("c") + [
        {"id": "c_te1", "position": "TE", "mean": 100.0},
        {"id": "c_te2", "position": "TE", "mean": 95.0},
        {"id": "c_wr1", "position": "WR", "mean": 55.0},
        {"id": "c_wr2", "position": "WR", "mean": 50.0},
        {"id": "c_rb1", "position": "RB", "mean": 150.0},
        {"id": "c_rb2", "position": "RB", "mean": 140.0},
    ]
    league = make_league(
        scoring,
        teams={
            1: team(1, "Me", _ids(mine)),
            2: team(2, "WR Bank", _ids(team2)),
            3: team(3, "TE Bank", _ids(team3)),
        },
        team_count=3,
    )
    return league, projections(mine + team2 + team3)


def test_a_chain_beats_the_best_single_trade(scoring):
    league, projs = chain_league(scoring)
    report = find_trades(league, projs, my_roster_id=1)
    assert report.trades
    assert report.chains, "the two-step sequence was not found"
    best_single = report.trades[0].my_gain
    chain = report.chains[0]
    assert chain.total_gain > best_single
    assert len(chain.steps) == 2
    # The two steps talk to two different teams.
    partners = {step.them.roster_id for step in chain.steps}
    assert partners == {2, 3}
    # Every step still clears the counterparty's bar on its own.
    assert all(step.them.gain > 0 for step in chain.steps)


def test_the_chains_second_step_was_not_a_good_single_trade(scoring):
    """The point of chains: step 2 only works on the roster step 1 creates."""
    league, projs = chain_league(scoring)
    report = find_trades(league, projs, my_roster_id=1)
    assert report.chains
    second = report.chains[0].steps[1]
    single_keys = {t.key() for t in report.trades}
    assert second.key() not in single_keys


# ---------------------------------------------------------------- edge cases


def test_an_empty_roster_gets_a_plain_answer(scoring):
    league = make_league(
        scoring, teams={1: team(1, "Me", []), 2: team(2, "Them", ["x"])}, team_count=2
    )
    report = find_trades(
        league,
        projections([{"id": "x", "position": "RB", "mean": 100.0}]),
        my_roster_id=1,
    )
    assert report.trades == []
    assert any("draft" in n for n in report.notes)


def test_unprojected_players_are_noted_and_never_packaged(scoring):
    league, projs = surplus_league(scoring)
    league.teams[1].players.append("mystery_rookie")
    report = find_trades(league, projs, my_roster_id=1)
    assert any("no projection" in n for n in report.notes)
    for trade in report.trades:
        assert "mystery_rookie" not in trade.me.gives


def test_an_unknown_roster_id_asks_for_a_team(scoring):
    league, projs = surplus_league(scoring)
    report = find_trades(league, projs, my_roster_id=99)
    assert report.trades == []
    assert any("Pick your team" in n for n in report.notes)


# ------------------------------------------------------------------- service


@pytest.mark.asyncio
async def test_the_service_decorates_ids_with_names(scoring, monkeypatch):
    """The page must never show a raw sleeper_id."""
    from fantasypicker.service import PickerService

    league, projs = surplus_league(scoring)
    league.my_roster_id = 1
    service = PickerService()
    service.league = league
    service.season_projections = projs
    service.model = object()  # readiness gate looks only at presence
    service.panel = object()

    async def no_refresh(self, *, force=False):
        return {"rosters": False, "players": False}

    monkeypatch.setattr(PickerService, "refresh_live", no_refresh)
    payload = await service.trades()
    assert payload["trades"]
    assert payload["my_team"] == "RB Hoarders"
    for trade in payload["trades"]:
        for pid in trade["me"]["gives"] + trade["them"]["gives"]:
            assert pid in payload["players"]
            assert payload["players"][pid]["name"]


@pytest.mark.asyncio
async def test_the_service_requires_a_team(scoring, monkeypatch):
    from fantasypicker.service import PickerService

    league, projs = surplus_league(scoring)
    league.my_roster_id = None
    service = PickerService()
    service.league = league
    service.season_projections = projs
    service.model = object()
    service.panel = object()

    async def no_refresh(self, *, force=False):
        return {"rosters": False, "players": False}

    monkeypatch.setattr(PickerService, "refresh_live", no_refresh)
    with pytest.raises(ValueError, match="Pick your team"):
        await service.trades()
