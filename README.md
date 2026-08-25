# FantasyPicker

A draft board and lineup optimiser for **Sleeper** fantasy football leagues. Point
it at your league ID and it reads your roster, your league's scoring rules, and —
crucially — **your opponent's roster**, every week, without anyone typing a
player name in by hand.

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

Then paste your league ID — the long number in your Sleeper league URL:

```
https://sleeper.com/leagues/1048273661924872192/team
                            ^^^^^^^^^^^^^^^^^^^
```

Add your Sleeper username too and it works out which team is yours; otherwise
you pick it from a list.

The first league you connect takes **three to five minutes**: it downloads
eleven seasons of NFL play data and trains a model against your league's exact
scoring rules. That model is cached, keyed by your scoring settings, so every
later start is instant. To do it up front instead of in the browser:

```bash
fantasypicker warm 1048273661924872192 --username yourname
```

Everything lands in `~/.fantasypicker` (`fantasypicker where` prints the paths):
about 30 MB of cached NFL data and a ~17 MB model per scoring configuration.
Nothing is uploaded anywhere; there is no account, no key, and no server but
your own.

---

## Where the data comes from

| Source | What it provides | Access |
| --- | --- | --- |
| [Sleeper API](https://docs.sleeper.com) | Leagues, rosters, matchups, live draft picks, injury status, trending adds | Public, no auth |
| [nflverse](https://github.com/nflverse/nflverse-data) | Weekly box scores, snap counts, depth charts, injury reports, 1999–present | Public releases |
| [nfldata](https://github.com/nflverse/nfldata) | Schedules with betting lines, rest days, weather, venue | Public |
| [DynastyProcess](https://github.com/dynastyprocess/data) | Sleeper ↔ nflverse ↔ FantasyPros ID crosswalk; consensus draft rankings with their spread | Public |

No API key is needed for any of them, and none of them is scraped — these are
all published, maintained data feeds. Sleeper's read endpoints require no
authentication at all, which is what makes automatic opponent lookup possible.

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

---

## The app

Five tabs:

- **Draft** — ranked recommendations with the reasoning spelled out, your
  roster, unfilled slots, and a chart of which positions fall off hardest before
  your next pick. A live-refresh toggle polls the draft while it runs.
- **Matchup** — win probability, both projected totals with ranges, the
  recommended lineup, the win-probability lineup when it differs, and swaps
  ranked by what they do to your odds. The opponent's roster is fetched
  automatically; you choose whether to model them as they have their lineup set
  or at their best.
- **Board** — the full ranked board with VOR, tiers, ADP, consensus spread, bye
  weeks, and a floor-to-ceiling range bar. Filter by position, search by name.
- **Waivers** — free agents ranked by what they would add to your roster over
  the rest of the season and this week, with the drop candidate named. Sleeper's
  trending-adds count sits alongside as a crowd signal, clearly separate from the
  model's own view.
- **Model** — validation metrics, calibration, measured injury and correlation
  rates, and per-position feature importance. Everything above, checkable.

Clicking any player anywhere opens their projection, game context, and last
twenty games.

---

## API

The web app is a client of a plain JSON API, so it is scriptable:

```
POST /api/connect       {league_id, username?}
GET  /api/status        loading progress
POST /api/team          {roster_id}
GET  /api/draft         live draft recommendations
GET  /api/board         full ranked board  ?position=RB&limit=200
GET  /api/matchup       ?week=4&opponent_mode=auto|declared|optimal&sims=20000
GET  /api/waivers       ?week=4
GET  /api/player/{id}   projection, context, and game log
GET  /api/model         the model card
POST /api/retrain       rebuild from scratch
```

Projection-backed routes answer **425 Too Early** with the current loading stage
while the model trains, rather than blocking.

---

## Limits worth knowing

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
pytest                          # 81 tests, no network access required
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
  engine/      simulator, correlations, lineups, draft, waivers, matchup
  web/         the dashboard (no build step)
  service.py   the one object that holds a loaded league
  api.py       HTTP routes
```

## Licence

MIT.
