"""ESPN's numeric vocabulary, translated.

ESPN identifies everything by integer: a stat is ``statId`` 53, a lineup slot is
``lineupSlotId`` 23, a team is ``proTeamId`` 12. None of it is self-describing,
and a wrong entry here is invisible — the app would simply score a league
slightly wrong forever. So each table below is written against the published
values, and anything not recognised is reported as unsupported rather than
silently ignored.

The scoring table maps ESPN stat IDs onto the same canonical keys
:mod:`fantasypicker.sleeper.scoring` already evaluates, which is what lets one
scoring engine serve both platforms.
"""

from __future__ import annotations

#: ESPN ``proTeamId`` -> nflverse abbreviation. 0 means "no team" (free agent).
PRO_TEAMS: dict[int, str] = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LA",
    15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI",
    22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS",
    29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}

#: ESPN ``defaultPositionId`` -> the position the projection model uses.
POSITIONS: dict[int, str] = {
    1: "QB",
    2: "RB",
    3: "WR",
    4: "TE",
    5: "K",
    7: "P",
    9: "DT",
    10: "DE",
    11: "LB",
    12: "CB",
    13: "S",
    16: "DST",
}

#: ESPN ``lineupSlotId`` -> the slot name in
#: :data:`fantasypicker.sleeper.league.SLOT_ELIGIBILITY`, or a bench slot.
LINEUP_SLOTS: dict[int, str] = {
    0: "QB",
    1: "QB",          # team QB — vanishingly rare, behaves as a QB slot
    2: "RB",
    3: "WRRB_FLEX",   # RB/WR
    4: "WR",
    5: "REC_FLEX",    # WR/TE
    6: "TE",
    7: "SUPER_FLEX",  # "OP" — any offensive player
    8: "DL",
    9: "DL",
    10: "LB",
    11: "DL",
    12: "DB",
    13: "DB",
    14: "DB",
    15: "IDP_FLEX",
    16: "DEF",
    17: "K",
    18: "P",
    19: "HC",
    20: "BN",
    21: "IR",
    23: "FLEX",       # RB/WR/TE
    24: "BN",         # "ER" — an extra reserve slot
}

#: Slots ESPN supports that this app cannot project, because no public box
#: score carries the inputs. A league using them is still readable; the slot is
#: reported rather than silently treated as empty.
UNPROJECTABLE_SLOTS = frozenset({"P", "HC"})

#: ESPN ``statId`` -> (canonical scoring key, multiplier).
#:
#: The multiplier exists for ESPN's fractional counters: "1/2 Sack" pays per
#: half sack, so a league awarding 1 point there awards 2 points per sack.
#: Folding that into the value keeps one term per underlying stat.
STAT_KEYS: dict[int, tuple[str, float]] = {
    # passing
    0: ("pass_att", 1.0),
    1: ("pass_cmp", 1.0),
    2: ("pass_inc", 1.0),
    3: ("pass_yd", 1.0),
    4: ("pass_td", 1.0),
    5: ("pass_yd", 1 / 5),      # "every 5 passing yards"
    6: ("pass_yd", 1 / 10),
    7: ("pass_yd", 1 / 20),
    8: ("pass_yd", 1 / 25),
    9: ("pass_yd", 1 / 50),
    10: ("pass_yd", 1 / 100),
    11: ("pass_cmp", 1 / 5),
    12: ("pass_cmp", 1 / 10),
    13: ("pass_inc", 1 / 5),
    14: ("pass_inc", 1 / 10),
    17: ("bonus_pass_yd_300_399", 1.0),
    18: ("bonus_pass_yd_400", 1.0),
    19: ("pass_2pt", 1.0),
    20: ("pass_int", 1.0),
    211: ("pass_fd", 1.0),
    64: ("pass_sack", 1.0),
    # rushing
    23: ("rush_att", 1.0),
    24: ("rush_yd", 1.0),
    25: ("rush_td", 1.0),
    26: ("rush_2pt", 1.0),
    27: ("rush_yd", 1 / 5),
    28: ("rush_yd", 1 / 10),
    29: ("rush_yd", 1 / 20),
    30: ("rush_yd", 1 / 25),
    31: ("rush_yd", 1 / 50),
    32: ("rush_yd", 1 / 100),
    33: ("rush_att", 1 / 5),
    34: ("rush_att", 1 / 10),
    37: ("bonus_rush_yd_100_199", 1.0),
    38: ("bonus_rush_yd_200", 1.0),
    212: ("rush_fd", 1.0),
    # receiving
    41: ("rec", 1.0),
    42: ("rec_yd", 1.0),
    43: ("rec_td", 1.0),
    44: ("rec_2pt", 1.0),
    47: ("rec_yd", 1 / 5),
    48: ("rec_yd", 1 / 10),
    49: ("rec_yd", 1 / 20),
    50: ("rec_yd", 1 / 25),
    51: ("rec_yd", 1 / 50),
    52: ("rec_yd", 1 / 100),
    53: ("rec", 1.0),
    54: ("rec", 1 / 5),
    55: ("rec", 1 / 10),
    56: ("bonus_rec_yd_100_199", 1.0),
    57: ("bonus_rec_yd_200", 1.0),
    58: ("rec_tgt", 1.0),
    59: ("rec_yac", 1.0),
    213: ("rec_fd", 1.0),
    # combined / turnovers
    62: ("two_pt", 1.0),
    63: ("fum_rec_td", 1.0),
    68: ("fum", 1.0),
    72: ("fum_lost", 1.0),
    73: ("turnovers", 1.0),
    # kicking
    74: ("fgm_50p", 1.0),
    75: ("fga_50p", 1.0),
    76: ("fgmiss_50p", 1.0),
    77: ("fgm_40_49", 1.0),
    78: ("fga_40_49", 1.0),
    79: ("fgmiss_40_49", 1.0),
    80: ("fgm_0_39", 1.0),
    81: ("fga_0_39", 1.0),
    82: ("fgmiss_0_39", 1.0),
    83: ("fgm", 1.0),
    84: ("fga", 1.0),
    85: ("fgmiss", 1.0),
    86: ("xpm", 1.0),
    87: ("xpa", 1.0),
    88: ("xpmiss", 1.0),
    198: ("fgm_50_59", 1.0),
    199: ("fga_50_59", 1.0),
    200: ("fgmiss_50_59", 1.0),
    201: ("fgm_60p", 1.0),
    202: ("fga_60p", 1.0),
    203: ("fgmiss_60p", 1.0),
    214: ("fgm_yds", 1.0),
    215: ("fgmiss_yds", 1.0),
    # team defense
    89: ("pts_allow_0", 1.0),
    90: ("pts_allow_1_6", 1.0),
    91: ("pts_allow_7_13", 1.0),
    92: ("pts_allow_14_17", 1.0),
    93: ("def_st_td", 1.0),   # blocked kick returned for a touchdown
    94: ("def_td", 1.0),      # fumble or interception returned for a touchdown
    95: ("int", 1.0),
    96: ("fum_rec", 1.0),
    97: ("blk_kick", 1.0),
    98: ("safe", 1.0),
    99: ("sack", 1.0),
    100: ("sack", 2.0),       # scored per half sack
    101: ("def_st_td", 1.0),
    102: ("def_st_td", 1.0),
    103: ("def_td", 1.0),
    104: ("def_td", 1.0),
    105: ("def_st_td", 1.0),
    106: ("ff", 1.0),
    120: ("pts_allow", 1.0),
    121: ("pts_allow_18_21", 1.0),
    122: ("pts_allow_22_27", 1.0),
    123: ("pts_allow_28_34", 1.0),
    124: ("pts_allow_35_45", 1.0),
    125: ("pts_allow_46p", 1.0),
    127: ("yds_allow", 1.0),
    128: ("yds_allow_0_100", 1.0),
    129: ("yds_allow_100_199", 1.0),
    130: ("yds_allow_200_299", 1.0),
    131: ("yds_allow_300_349", 1.0),
    132: ("yds_allow_350_399", 1.0),
    133: ("yds_allow_400_449", 1.0),
    134: ("yds_allow_450_499", 1.0),
    135: ("yds_allow_500_549", 1.0),
    136: ("yds_allow_550p", 1.0),
    205: ("def_2pt", 1.0),
    206: ("def_2pt", 1.0),
    # ESPN duplicates the points-allowed family for D/ST in newer leagues.
    187: ("pts_allow", 1.0),
    188: ("pts_allow_0", 1.0),
    189: ("pts_allow_1_6", 1.0),
    190: ("pts_allow_7_13", 1.0),
    191: ("pts_allow_14_17", 1.0),
    192: ("pts_allow_18_21", 1.0),
    193: ("pts_allow_22_27", 1.0),
    194: ("pts_allow_28_34", 1.0),
    195: ("pts_allow_35_45", 1.0),
    196: ("pts_allow_46p", 1.0),
    # individual defensive players
    107: ("idp_tkl_ast", 1.0),
    108: ("idp_tkl_solo", 1.0),
    109: ("idp_tkl", 1.0),
    110: ("idp_tkl", 1 / 3),
    111: ("idp_tkl", 1 / 5),
    112: ("idp_tkl_loss", 1.0),
    113: ("idp_pass_def", 1.0),
}

#: Human labels, so diagnostics can name a stat instead of printing its number.
#: Only the IDs a league is plausibly scored on are listed.
STAT_LABELS: dict[int, str] = {
    0: "Each Pass Attempted", 1: "Each Pass Completed", 2: "Each Incomplete Pass",
    3: "Passing Yards", 4: "TD Pass", 5: "Every 5 passing yards",
    6: "Every 10 passing yards", 7: "Every 20 passing yards",
    8: "Every 25 passing yards", 9: "Every 50 passing yards",
    10: "Every 100 passing yards", 11: "Every 5 pass completions",
    12: "Every 10 pass completions", 13: "Every 5 pass incompletions",
    14: "Every 10 pass incompletions", 15: "40+ yard TD pass bonus",
    16: "50+ yard TD pass bonus", 17: "300-399 yard passing game",
    18: "400+ yard passing game", 19: "2pt Passing Conversion",
    20: "Interceptions Thrown", 21: "Passing Completion Pct",
    22: "Passing Yards Per Game", 23: "Rushing Attempts", 24: "Rushing Yards",
    25: "TD Rush", 26: "2pt Rushing Conversion", 27: "Every 5 rushing yards",
    28: "Every 10 rushing yards", 29: "Every 20 rushing yards",
    30: "Every 25 rushing yards", 31: "Every 50 rushing yards",
    32: "Every 100 rushing yards", 33: "Every 5 rush attempts",
    34: "Every 10 rush attempts", 35: "40+ yard TD rush bonus",
    36: "50+ yard TD rush bonus", 37: "100-199 yard rushing game",
    38: "200+ yard rushing game", 39: "Rushing Yards Per Attempt",
    40: "Rushing Yards Per Game", 41: "Receptions", 42: "Receiving Yards",
    43: "TD Reception", 44: "2pt Receiving Conversion",
    45: "40+ yard TD rec bonus", 46: "50+ yard TD rec bonus",
    47: "Every 5 receiving yards", 48: "Every 10 receiving yards",
    49: "Every 20 receiving yards", 50: "Every 25 receiving yards",
    51: "Every 50 receiving yards", 52: "Every 100 receiving yards",
    53: "Each reception", 54: "Every 5 receptions", 55: "Every 10 receptions",
    56: "100-199 yard receiving game", 57: "200+ yard receiving game",
    58: "Receiving Target", 59: "Receiving Yards After Catch",
    60: "Receiving Yards Per Catch", 61: "Receiving Yards Per Game",
    62: "Total 2pt Conversions", 63: "Fumble Recovered for TD", 64: "Sacked",
    68: "Total Fumbles", 72: "Total Fumbles Lost", 73: "Total Turnovers",
    74: "FG Made (50+ yards)", 75: "FG Attempted (50+ yards)",
    76: "FG Missed (50+ yards)", 77: "FG Made (40-49 yards)",
    78: "FG Attempted (40-49 yards)", 79: "FG Missed (40-49 yards)",
    80: "FG Made (0-39 yards)", 81: "FG Attempted (0-39 yards)",
    82: "FG Missed (0-39 yards)", 83: "Total FG Made", 84: "Total FG Attempted",
    85: "Total FG Missed", 86: "Each PAT Made", 87: "Each PAT Attempted",
    88: "Each PAT Missed", 89: "0 points allowed", 90: "1-6 points allowed",
    91: "7-13 points allowed", 92: "14-17 points allowed",
    93: "Blocked Punt or FG return for TD", 94: "Fumble or INT Return for TD",
    95: "Each Interception", 96: "Each Fumble Recovered",
    97: "Blocked Punt, PAT or FG", 98: "Each Safety", 99: "Each Sack",
    100: "1/2 Sack", 101: "Kickoff Return TD", 102: "Punt Return TD",
    103: "Interception Return TD", 104: "Fumble Return TD",
    105: "Total Return TD", 106: "Each Fumble Forced", 107: "Assisted Tackles",
    108: "Solo Tackles", 109: "Total Tackles", 110: "Every 3 Total Tackles",
    111: "Every 5 Total Tackles", 112: "Stuffs", 113: "Passes Defensed",
    114: "Kickoff Return Yards", 115: "Punt Return Yards", 120: "Points Allowed",
    121: "18-21 points allowed", 122: "22-27 points allowed",
    123: "28-34 points allowed", 124: "35-45 points allowed",
    125: "46+ points allowed", 126: "Points Allowed Per Game",
    127: "Yards Allowed", 128: "Less than 100 total yards allowed",
    129: "100-199 total yards allowed", 130: "200-299 total yards allowed",
    131: "300-349 total yards allowed", 132: "350-399 total yards allowed",
    133: "400-449 total yards allowed", 134: "450-499 total yards allowed",
    135: "500-549 total yards allowed", 136: "550+ total yards allowed",
    155: "Team Win", 156: "Team Loss", 157: "Team Tie", 158: "Points Scored",
    175: "0-9 yd TD pass bonus", 176: "10-19 yd TD pass bonus",
    177: "20-29 yd TD pass bonus", 178: "30-39 yd TD pass bonus",
    179: "0-9 yd TD rush bonus", 180: "10-19 yd TD rush bonus",
    181: "20-29 yd TD rush bonus", 182: "30-39 yd TD rush bonus",
    183: "0-9 yd TD rec bonus", 184: "10-19 yd TD rec bonus",
    185: "20-29 yd TD rec bonus", 186: "30-39 yd TD rec bonus",
    187: "D/ST Points Allowed", 188: "D/ST 0 points allowed",
    189: "D/ST 1-6 points allowed", 190: "D/ST 7-13 points allowed",
    191: "D/ST 14-17 points allowed", 192: "D/ST 18-21 points allowed",
    193: "D/ST 22-27 points allowed", 194: "D/ST 28-34 points allowed",
    195: "D/ST 35-45 points allowed", 196: "D/ST 46+ points allowed",
    198: "FG Made (50-59 yards)", 199: "FG Attempted (50-59 yards)",
    200: "FG Missed (50-59 yards)", 201: "FG Made (60+ yards)",
    202: "FG Attempted (60+ yards)", 203: "FG Missed (60+ yards)",
    204: "Offensive 2pt Return", 205: "Defensive 2pt Return", 206: "2pt Return",
    207: "Offensive 1pt Safety", 208: "Defensive 1pt Safety", 209: "1pt Safety",
    210: "Games Played", 211: "Passing First Down", 212: "Rushing First Down",
    213: "Receiving First Down", 214: "FG Made Yards", 215: "FG Missed Yards",
    216: "FG Attempt Yards",
}


#: ESPN numbers team defenses from this base, negated: ATL (proTeamId 1) is
#: -16001, HOU (34) is -16034. The draft feed carries only a player id, so a
#: drafted defense is unidentifiable without undoing that.
DST_ID_BASE = 16000


def is_unmade_pick(espn_id: object) -> bool:
    """Is this draft slot still empty?

    ESPN pre-allocates the whole draft board and fills each slot in as the
    pick is made, marking the ones still to come with ``playerId: -1``. They
    are placeholders, not players, and counting them as unreadable makes a
    draft that has not reached them look like a parsing failure.
    """
    try:
        value = int(espn_id)
    except (TypeError, ValueError):
        return True
    # Real ids are positive; defenses are large negatives (see DST_ID_BASE).
    return value == 0 or value == -1


def dst_team_from_player_id(espn_id: object) -> str | None:
    """The team abbreviation behind a synthetic D/ST player id, if it is one."""
    try:
        value = int(espn_id)
    except (TypeError, ValueError):
        return None
    if value >= 0:
        return None
    return PRO_TEAMS.get(abs(value) - DST_ID_BASE)


def stat_label(stat_id: int) -> str:
    """A readable name for a stat ID, falling back to the number itself."""
    return STAT_LABELS.get(int(stat_id), f"stat {stat_id}")


def position_of(default_position_id: object) -> str | None:
    try:
        return POSITIONS.get(int(default_position_id))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def team_of(pro_team_id: object) -> str | None:
    try:
        return PRO_TEAMS.get(int(pro_team_id))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def slot_of(lineup_slot_id: object) -> str | None:
    try:
        return LINEUP_SLOTS.get(int(lineup_slot_id))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
