# FantasyPicker

A draft board and lineup optimiser for **Sleeper** and **ESPN** fantasy football
leagues. Point it at your league and it reads your roster, your league's scoring
rules, and — crucially — **your opponent's roster**, every week, without anyone
typing a player name in by hand.

It then does two jobs:

- **Draft day.** A live board that knows what your roster still needs, what each
  player is worth *to you*, and which positions are about to be picked clean
  before your next turn.
- **Every week after.** The best lineup you can field, the odds you beat this
  specific opponent, which swap moves those odds most, and what on the waiver
  wire would actually improve your team.

---

## Quick start

**macOS / Linux**

```bash
git clone https://github.com/that-coder-guy/fantasypicker.git
cd fantasypicker
python -m venv .venv && source .venv/bin/activate
pip install -e .

fantasypicker serve --open
```

**Windows** (PowerShell)

```powershell
git clone https://github.com/that-coder-guy/fantasypicker.git
cd fantasypicker
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

fantasypicker serve --open
```

Then pick the tab for wherever your league is played.

**Sleeper.** Type your **Sleeper username** — the one you log in with, not your
team name — and pick your league from the list. That is the whole setup; the
league ID, your team, and your opponent all follow from it.

**ESPN.** Paste your league ID or the league URL. A public league needs nothing
else. A private one needs two cookies — see
[Private ESPN leagues](#private-espn-leagues) below.

If you would rather paste a Sleeper league ID, open "Enter a league ID instead".
Both the bare number and the whole URL work:

```
https://sleeper.com/leagues/1048273661924872192/team
                            ^^^^^^^^^^^^^^^^^^^
```

Note that a **draft ID is not a league ID** — `sleeper.com/draft/nfl/<id>` is a
different number, and it is the most common reason a paste is rejected. The
username route sidesteps this entirely. From a terminal:

```bash
fantasypicker leagues yourname     # prints every league and its ID
```

If that command finds your user but lists no leagues, your league is not on
Sleeper — see [Limits worth knowing](#limits-worth-knowing).

### Private ESPN leagues

A **public** ESPN league needs nothing but its ID. A **private** one — the
default for most leagues — returns "not authorised" until you supply two cookies
from a browser that is already signed in to ESPN:

1. Sign in at `fantasy.espn.com` in your normal browser.
2. Press <kbd>F12</kbd> to open developer tools.
3. Go to **Application** (Chrome/Edge) or **Storage** (Firefox) → **Cookies** →
   `https://fantasy.espn.com`.
4. Copy the values of **`espn_s2`** and **`SWID`**.
5. Paste them into "My league is private" on the ESPN tab.

These are session credentials for your own ESPN account, so they are treated as
such: stored in `~/.fantasypicker/credentials.json` with `0600` permissions
(only your user can read it), kept out of `state.json` so that file stays
harmless to share, never written to a log, never returned by the API, and sent
to nobody but `espn.com`. They are stored so you do not have to repeat this —
ESPN expires them every few weeks, and when that happens the app says so rather
than silently serving you last week's rosters.

Nobody else in your league has to do anything. The cookies identify *you*, and
reading the league then includes everyone's roster.

From a terminal, the same thing:

```bash
fantasypicker warm <league_id> --espn --espn-s2 <value> --swid <value>
```

### Checking an ESPN league was read correctly

```bash
fantasypicker doctor <league_id> --espn
```

This prints the roster slots, every team with its record and roster size, any
player whose identity could not be resolved — and, most usefully, **every ESPN
scoring setting beside the term it was translated into**. Scoring is the one
setting whose mistranslation is invisible: the app would keep working and
quietly rank players under rules that are not yours. Comparing that table
against your league's settings page takes a minute and rules it out.

### When teams show up as "Team 4"

Team names live on Sleeper's *user* objects, joined to rosters by `owner_id`,
and the app falls back through every name Sleeper offers — the custom team name,
the display name, then the login username — before resorting to a number. A
numeric name therefore means the join itself missed. To see which:

```bash
fantasypicker doctor <league_id>
```

It prints every user with all three name fields, every roster with its
`owner_id`, and the label each roster ends up with, bypassing the response
cache. A roster whose `owner_id` matches no user is an orphaned team — someone
left the league — and Sleeper genuinely has no name for it.

### Windows notes

Use `.venv\Scripts\activate.bat` instead if you are in `cmd.exe`. If PowerShell
refuses to run the activation script, `Set-ExecutionPolicy -Scope Process
RemoteSigned` for that session, or skip activation entirely and call
`.\.venv\Scripts\fantasypicker.exe serve --open` directly.

Windows needs nothing beyond a stock Python install — every dependency ships a
Windows wheel, so there is no compiler step, and `uvicorn[standard]` correctly
omits `uvloop` on Windows. Python **3.10–3.13** works; 3.12 is the safest bet
for LightGBM and pyarrow wheel coverage. Get it from python.org or
`winget install Python.Python.3.12` — the Microsoft Store build works too but
sandboxes its file writes, which makes the cache directory harder to find.

The cache and trained models live in `C:\Users\<you>\.fantasypicker`
(`fantasypicker where` prints the exact paths) — about 30 MB of NFL data plus a
~17 MB model per scoring configuration. Set `FANTASYPICKER_HOME` to move it,
which is worth doing if your user folder is synced to OneDrive:

```powershell
$env:FANTASYPICKER_HOME = "D:\fantasypicker"
```

The first league you connect takes **three to five minutes**: it downloads
eleven seasons of NFL play data and trains a model against your league's exact
scoring rules. That model is cached, keyed by your scoring settings, so every
later start is instant. To do it up front instead of in the browser:

```bash
fantasypicker warm 1048273661924872192 --username yourname
```

Everything lands in `~/.fantasypicker` (`fantasypicker where` prints the paths):
about 30 MB of cached NFL data, a ~17 MB model and a cached panel per scoring
configuration, and a small `state.json` remembering your leagues.
Nothing is uploaded anywhere; there is no account, no key, and no server but
your own.

---

## Where the data comes from

| Source | What it provides | Access |
| --- | --- | --- |
| [Sleeper API](https://docs.sleeper.com) | Leagues, rosters, matchups, live draft picks, injury status, trending adds | Public, no auth |
| [ESPN fantasy API](https://fantasy.espn.com) | ESPN leagues: rosters, scoring settings, schedule, draft picks | Undocumented; public leagues need no auth, private ones need your own cookies |
| [nflverse](https://github.com/nflverse/nflverse-data) | Weekly box scores, snap counts, depth charts, injury reports, 1999–present | Public releases |
| [nfldata](https://github.com/nflverse/nfldata) | Schedules with betting lines, rest days, weather, venue | Public |
| [DynastyProcess](https://github.com/dynastyprocess/data) | Sleeper ↔ ESPN ↔ nflverse ↔ FantasyPros ID crosswalk; consensus draft rankings with their spread | Public |

No API key is needed for any of them, and none of them is scraped — these are
all published, maintained data feeds. Sleeper's read endpoints require no
authentication at all, which is what makes automatic opponent lookup possible.
ESPN's are undocumented but are the same ones espn.com's own front end calls,
and a private league is read with your own session cookies rather than by
working around anything.

Sleeper's player endpoints double as a league-independent database of the NFL —
names, positions, injury designations, waiver-wire buzz — so they are used for
an ESPN league too. Whether a receiver is questionable has nothing to do with
where your league is hosted.

---

## How the projections work

### Scoring is yours, not a preset

Fantasy points are recomputed from raw box-score components under *your*
league's `scoring_settings` — half PPR, TE premium, first downs, 100-yard
bonuses, six-point passing touchdowns, kicker scoring by distance bucket, points
allowed tiers. That applies to the training labels as well as the output, so the
model is fitted to the game you actually play. Scoring rules that weekly box
scores cannot reconstruct (things needing play-by-play, like "50+ yard touchdown
bonus") are listed in the UI rather than silently treated as zero.

### The output is a distribution, not a number

Each position gets a **LightGBM model per quantile** — nine of them, from the
5th percentile to the 95th — plus a mean model. The quantile curve *is* the
projection: the median is the headline, the spread is the risk, and the
simulator samples from the curve directly rather than assuming a shape.

That matters because every decision the app makes is a decision under
uncertainty. Whether to start the steady player or the volatile one is not a
question about expected points at all; it depends on whether you are ahead or
behind, and only the distribution can answer it.

### Features

About 130 per player-week, all of them knowable before kickoff:

- **Usage and form** — rolling 3- and 8-game and career means of fantasy points,
  targets, target share, air-yards share, WOPR, carries, receptions, snap
  percentage, plus the volatility of each.
- **Prior seasons** — per-game rates from the last two seasons. This exists for
  a specific failure: a team that rests its starters in week 18 leaves its
  quarterback with a three-game average built from garbage time, and in week 1
  that is the freshest evidence there is. A full prior season averages over it.
- **Game context** — the betting market's implied team total, spread, game
  total, home/away, rest days, dome, temperature, wind, divisional game. The
  market is the best public estimate of how many points a team will score, and
  it is published days ahead.
- **Opponent** — how many fantasy points that defense has been allowing to this
  position lately, measured under your league's scoring and expressed relative
  to the league average that week.
- **Status** — depth chart rank, official injury designation, practice
  participation, weeks since last appearance.
- **Player** — age, experience, draft capital, size.

### Validation

Walk-forward by season: train on everything before season *S*, score season *S*.
The baseline is each player's own recent eight-game average — the simplest
honest prediction. Held out on 2025, trained on 2016–2024:

| Position | MAE | Baseline MAE | RMSE | Baseline RMSE | Spearman | Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| QB | 5.94 | 6.86 | 7.53 | 8.53 | 0.572 | 0.455 |
| RB | 3.83 | 4.20 | 5.81 | 6.07 | 0.733 | 0.695 |
| WR | 3.17 | 3.55 | 4.84 | 5.10 | 0.677 | 0.636 |
| TE | 2.79 | 3.11 | 4.48 | 4.72 | 0.684 | 0.644 |
| K | 3.69 | 3.92 | 4.62 | 4.90 | 0.180 | 0.054 |
| DST | 4.16 | 4.74 | 5.50 | 6.05 | 0.302 | 0.036 |

MAE is scored against the median model and RMSE against the mean model, since
those are the losses each one minimises — scoring a mean estimator on MAE would
penalise it for correctly reflecting the right-skew of fantasy scoring.

Weekly fantasy scoring is mostly noise; a 10–14% error reduction over a rolling
average is a real edge, not a solved problem. Anything claiming much better is
either measuring in-sample or measuring something easier.

The same held-out season measures how often outcomes actually fall below each
predicted quantile, and that measurement is folded back in as a monotone
correction. The app's own **Model** tab shows all of it, including what each
position's model leans on.

### Availability is separate from performance

The model is trained on games players appeared in, so its output is conditional
on playing. P(play) is applied on top, measured from every official injury
report since 2016 matched against whether the player took a snap:

| Designation | Play rate | Sample |
| --- | --- | --- |
| Questionable | 62% | 4,582 |
| Doubtful | 1% | 533 |
| Out | 0% | 3,173 |

Sleeper's live status wins when it says IR, PUP, or suspended — that is a fact,
not a probability. In the simulator the two stay separate, so a 60%-to-play star
reads as a genuinely bimodal outcome rather than a 60%-sized average one.

---

## How the decisions work

### Simulation with correlated outcomes

Player scores are not independent, and pretending otherwise breaks precisely the
calculation that matters. Sampling uses a **Gaussian copula**: correlated
normals supply the dependence, each player's own quantile curve supplies the
shape. Correlations are measured, not asserted — standardising each player-week
against that player's own season and correlating across every pair who shared a
game:

| Relationship | Correlation | Sample |
| --- | --- | --- |
| Defense vs opposing QB | −0.32 | 5,387 |
| Defense vs opposing kicker | −0.27 | 5,049 |
| QB with his own receivers | +0.19 | 24,607 |
| QB with his own tight end | +0.14 | 14,353 |
| Two QBs in the same game | +0.13 | 2,758 |
| Defense vs opposing RB | −0.12 | 13,980 |

A stacked lineup really is more volatile than a diversified one with the same
expected points, and win probability is a function of variance.

### Lineups

Filling slots from a roster is an **assignment problem**, so it is solved
exactly with maximum-weight bipartite matching rather than greedily. The
difference is not academic: with one tight end, a TE slot and a FLEX, greedy
puts him in the flex and strands the TE slot.

Two lineups get computed:

- **Most expected points** — right when the matchup is close.
- **Most likely to win** — hill-climbed over simulated outcomes. A heavy
  underdog does not want the highest average; they want the fattest right tail,
  because only the top of their range wins the week. A favourite wants the
  opposite.

The app says which case you are in and what the difference is worth, rather than
quietly choosing.

Start/sit advice is stated the same way: not "player A is better", but "against
this opponent, this swap is worth 3.4 percentage points of win probability".

### Draft

Three ideas:

**Marginal lineup value.** A player is worth what he adds to the best lineup you
can field, computed by solving the assignment with and without him, against a
roster whose empty slots are backfilled with replacement-level players. One
definition handles positional need, flex and superflex eligibility, and the fact
that your fourth tight end is worth almost nothing — with no hand-tuned
positional multipliers. On an empty roster it equals value over replacement; as
the roster fills it becomes value over your own starter, which is exactly how the
number should behave.

**Replacement level from the league, not from convention.** "RB24" is only the
replacement back in one specific league shape. The cutoff is derived from your
actual roster slots, counting flex spots fractionally across the positions that
can fill them, so a 3-WR-plus-flex league and a 2-WR league get different
answers. A superflex league prices quarterbacks correctly because the slot
structure says so.

**Scarcity as a probability.** Expert consensus rank and its published standard
deviation give each player a distribution over where he goes, and from that
comes P(he is still there at your next pick). The recommendation is a two-ply
lookahead — this pick's value plus the best you can expect from your next one
once this player is off the board — which is the honest form of the only
question that matters on the clock: not *who is best?* but *who will not be
there later?*

Live drafts sync from Sleeper's pick feed. Snake, linear, and third-round
reversal are all handled.

### Two currencies

Every other manager in your league is deciding off one number: the projection
their platform prints next to each player. It is not as good as this app's — if
it were, there would be no edge to have — but it is the number in front of the
person you are trying to trade with, and a proposal that reads as a fleece on
*their* screen gets declined regardless of how sound it is on yours.

So the app carries two valuations and never confuses them:

| | Model points | Market points |
| --- | --- | --- |
| Source | This app's projections | The league's own platform |
| Answers | Did this roster actually get better? | How does this look to them? |
| Used for | Every gain, every lineup, every recommendation | The headline rule, the balance window, their perceived gain |

Concretely: a trade is proposed only when it improves your lineup **on the
model**, improves theirs **on the model**, *and* reads as a fair, improving
deal **on the public numbers**. The third condition is new and it removes
proposals that used to be made and never accepted. The Trades page shows both
numbers against every player and totals each package in both, so you can see
the gap you are trading into rather than taking it on faith.

Where the public numbers come from, in descending order of how closely they
match what the other manager sees:

1. **The platform itself.** For ESPN this is exact — the same PROJ column, read
   from the `stats` block on each player object (`statSourceId: 1` for a
   projection, `statSplitTypeId: 0` for the season line), already computed
   under your league's own scoring rules.
2. **FantasyPros consensus**, for platforms that publish no projections through
   their API — Sleeper among them. Not the same numbers, but the same public
   consensus most managers are anchored to. A scrape more than ten days old is
   refused rather than used stale.
3. **Nothing**, in which case the model stands in for both, the page says so in
   as many words, and the advice is exactly what it was before this existed.

Platforms publish *season* totals while this app works in *rest-of-season*
points, so market numbers are prorated by the games each player has left —
byes included, since the projection frame already counts them. Before week 1
the two bases coincide, so during a draft the number shown is exactly the total
printed on the platform.

---

## The app

Eight tabs:

- **Draft** — ranked recommendations with the reasoning spelled out, your
  roster, unfilled slots, and a chart of which positions fall off hardest before
  your next pick. A live-refresh toggle polls the draft while it runs.
- **Matchup** — win probability, both projected totals with ranges, the
  recommended lineup, the win-probability lineup when it differs, and swaps
  ranked by what they do to your odds. The opponent's roster is fetched
  automatically; you choose whether to model them as they have their lineup set
  or at their best.
- **League** — every team in the league, ranked by the best lineup its roster
  could field, with that lineup laid out. **During a draft this fills from the
  live pick feed**: neither Sleeper nor ESPN moves drafted players onto a
  roster until the draft ends (espn.com shows empty teams too), but both
  publish every pick as it happens, so every team's haul is visible while the
  draft is still running — and the waiver and trade views work off it. Before the draft it still lists every
  team, ordered by draft slot, since that is the only thing distinguishing them
  when every roster is empty. Toggle to what each manager actually
  has set, and the difference is shown: who is leaving points on their bench,
  and how many. Per-position strength chips compare each roster to the league
  average, so it is obvious at a glance who is deep at running back and who is
  about to be short at tight end. Benches expand on request. This is the trade
  and waiver scouting view.
- **Board** — the full ranked board with VOR, tiers, ADP, consensus spread, bye
  weeks, and a floor-to-ceiling range bar. Filter by position, search by name.
- **Waivers** — free agents ranked by what they would add to your roster over
  the rest of the season and this week, with the drop candidate named. Sleeper's
  trending-adds count sits alongside as a crowd signal, clearly separate from the
  model's own view.
- **Drops** — the same question from the other end. Waivers pairs every add
  with your single worst player, which is the wrong pairing: adding a tight end
  when you already start one should cost you a tight end, not your worst
  running back. So this asks, for *every* player you roster, whether anyone in
  the open pool leaves your best lineup better off with that specific player
  gone — both halves solved exactly, with the same roster valuation the trade
  engine uses, so the two pages can never disagree about what a roster is
  worth. Two kinds of answer come out. **Upgrades** are swaps that gain points;
  do them. **Free to cut** are players nobody in the pool beats but who never
  reach your lineup either — the roster spots to spend when a bye week forces a
  move, which is worth knowing before 11pm on Saturday rather than during it.
  Values are rest-of-season, because a drop is permanent. Injured-reserve and
  taxi players are left out: they hold no active roster spot and block nobody.
- **Trades** — proposals the other manager should actually accept. A player's
  value depends on the roster he lands on, so trades exist that help both
  sides; the engine values every roster as the best lineup it could field over
  the rest of the season, searches 1-for-1 up to 2-for-2 packages against every
  team, and only proposes deals where *both* lineups improve. Uneven trades
  include their knock-on moves — the freed roster spot backfilled from free
  agency, the extra body dropped — because that is where a 2-for-1's value
  actually lives. Proposals are ranked by your gain times the odds they say
  yes (from their gain and the deal's perceived fairness), not by the biggest
  heist, and each shows both sides' numbers and a plain-language rationale.
  When a two-step **chain** beats any single deal — trade for a receiver, and
  the receiver he benches becomes the piece a third team wants — it is laid
  out step by step, each step still fair to its own counterparty.

  Every player carries **two numbers**, because the manager on the other end is
  not looking at this app. They are looking at the projection ESPN prints, and
  that is the only number in front of them when they decide. So the page shows
  the public projection alongside the model's, and the engine keeps the two
  jobs apart: model points decide whether a roster genuinely improved, and
  public points decide whether the deal will be accepted — the headline rule,
  the balance window, and the other side's perceived gain all run on their
  numbers. A trade this app loves that reads as a fleece on espn.com is not a
  good recommendation, it is a wasted proposal, and it no longer gets made. See
  [Two currencies](#two-currencies) for where the public numbers come from.
- **Model** — validation metrics, calibration, measured injury and correlation
  rates, and per-position feature importance. Everything above, checkable.

Clicking any player anywhere opens their projection, game context, and last
twenty games.

### It remembers your leagues

Connect once and the app reopens that league the next time you start it, with
your team already selected. Everything worth remembering is stored **per
league**, in `~/.fantasypicker/state.json`:

- which team is yours — including a team you picked by hand, which is not
  wiped by a later reconnect that has no username attached
- the username the league was found under
- the scoring fingerprint, which is how the app knows whether the trained model
  on disk belongs to this league

If you run more than one league, a **▾** next to the league name in the header
switches between them, and the connect screen offers them under "Pick up where
you left off". Each entry says whether reopening it is instant or needs a
training run, so a switch is never a surprise.

Two leagues that score identically share one trained model *and* one built
panel — the common case if you run several half-PPR leagues. Switching between
those is a few seconds. Switching to a league with genuinely different scoring
means a rebuild, because the labels the model is fitted to are different.

The built panel is cached too, so even a rebuild after a restart skips the
two-minute assembly step as long as the cache is under twelve hours old.
`POST /api/retrain` (or the Model tab's retrain) forces everything from scratch.

Nothing here leaves your machine, and forgetting a league is one call:
`DELETE /api/known/<league_id>`.

### Staying current

The page keeps itself up to date; there is no need to reload or to keep a second
browser open on Sleeper. A dot in the header says whether it is synced,
checking, or unable to reach Sleeper.

Two things are re-pulled: **rosters** (a waiver claim, a trade, a drop) and
Sleeper's **player file**, which is where injury designations live. Neither
requires the model to be rebuilt — the projections are conditional on a player
suiting up, and availability is applied on top, so a Sunday-morning downgrade is
arithmetic over a cached frame rather than a re-projection.

| When | What happens |
| --- | --- |
| Draft tab open | Re-checks every 20s; the **Live refresh** toggle drops that to 6s for an active draft |
| Matchup tab open | Every 90s |
| League tab open | Every 2 minutes |
| Waivers / Board | Every 3 / 5 minutes |
| Tab refocused | Immediately — a window left open overnight does not show yesterday's rosters |
| **Refresh** button | Immediately, bypassing the disk cache entirely |

The server throttles its own calls to Sleeper to one round trip per 30 seconds
however many requests arrive, and background tabs are never polled.

---

## API

The web app is a client of a plain JSON API, so it is scriptable:

```
POST /api/leagues       {username, season?} -> that user's leagues and IDs
GET  /api/known         leagues this machine has connected to before
DEL  /api/known/{id}    forget one
POST /api/connect       {league_id, username?}          Sleeper
POST /api/connect/espn  {league_id, season?, espn_s2?, swid?}
GET  /api/status        loading progress + remembered leagues
POST /api/refresh       re-poll the platform for rosters and injury status now
POST /api/team          {roster_id}
GET  /api/draft         live draft recommendations
GET  /api/board         full ranked board  ?position=RB&limit=200
GET  /api/matchup       ?week=4&opponent_mode=auto|declared|optimal&sims=20000
GET  /api/teams         ?week=4 — every team, its lineup, bench and strengths
GET  /api/waivers       ?week=4
GET  /api/drops         ?roster_id=
GET  /api/trades        ?roster_id=&chains=true&max_package=2
GET  /api/player/{id}   projection, context, and game log
GET  /api/model         the model card
POST /api/retrain       rebuild from scratch
```

Projection-backed routes answer **425 Too Early** with the current loading stage
while the model trains, rather than blocking.

`/api/trades` carries a `market` object naming where the public projections came
from (`source`, `available`, `covered`), and every row in `players` carries both
`ros_points` (the model) and `market_points` (the public number, `null` when
that source has nothing for the player — which is deliberately distinct from a
projection of zero).

---

## Limits worth knowing

- **Sleeper and ESPN only.** Yahoo and NFL.com leagues cannot be read. Yahoo
  requires registering an OAuth app and a browser consent flow; NFL.com has no
  public read path at all. Both are real projects rather than config switches.

- **A few ESPN scoring settings cannot be reproduced.** Anything needing
  play-by-play rather than a box score — "40+ yard TD bonus", "1pt safety",
  per-game rate stats — is scored as zero and named out loud at connect time
  and by `fantasypicker doctor --espn`. Everything a weekly box score supports,
  including ESPN's banded milestones and per-position reception values, is
  translated exactly.

- **Public trade values depend on the platform publishing them.** ESPN does, so
  ESPN leagues get the exact numbers the other managers see. Sleeper does not
  expose projections through its public API, so those leagues fall back to
  FantasyPros consensus — close to what most managers are anchored to, but not
  literally the figure on their screen — and to the model alone if no fresh
  consensus scrape exists. The Trades page always names which of the three it
  is using.

- **Weekly fantasy football is mostly noise.** The model beats a rolling average
  by 10–14%. It will still tell you to start someone who scores three points.
  The distributions are honest about this; the point estimates cannot be.
- **Kickers and defenses are barely predictable.** Spearman 0.18 and 0.30. The
  model knows this and its ranges are correspondingly wide. Stream them on
  matchup and do not think about it further.
- **Rookies have no history.** Projections for them lean on the market
  (Sleeper's own projections, rescored under your rules) rather than on our
  features, which have nothing to work from. The board shows model value and ADP
  side by side so you can see when the two disagree.
- **Beginning-of-season projections carry more uncertainty than the model
  shows.** Depth charts change, holdouts end, and the market moves faster than
  weekly data does.
- **IDP leagues are partially supported.** Individual defensive scoring is
  computed correctly, but no projection model is trained for defensive players —
  there is not enough signal in weekly box scores to do it honestly.
- **Two-quarterback and superflex leagues are fully supported**; so are dynasty
  leagues, though the board uses redraft rankings unless the league is flagged
  dynasty in Sleeper.

---

## Development

```bash
pip install -e ".[dev]"
pytest                          # 325 tests, no network access required
fantasypicker serve --reload
```

The same three commands work on Windows once the venv is activated. The suite is
platform-independent — no shell-outs, no POSIX paths, and the cache is
redirected to a temp directory for each test.

Tests use stubbed Sleeper responses and synthetic projections; nothing in the
suite touches the network, and the cache is redirected to a temp directory.

```
fantasypicker/
  sleeper/     API client, league shape, scoring rules
  data/        nflverse loaders, ID crosswalk, consensus rankings
  model/       panel construction, features, quantile training, prediction
  espn/        API client, league translation, published projections
  engine/      simulator, correlations, lineups, draft, waivers, matchup,
               trades, drops, league-wide team view
  web/         the dashboard (no build step)
  market.py    what the rest of the league sees, and where it came from
  service.py   the one object that holds a loaded league
  store.py     remembered leagues, per league, on disk
  api.py       HTTP routes
```

## Licence

MIT.
