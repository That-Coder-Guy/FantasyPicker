/* FantasyPicker dashboard.
 *
 * Plain ES modules-free JavaScript on purpose: no build step, no node_modules,
 * no version drift. The server is the only dependency.
 */

const state = {
  league: null,
  status: null,
  view: "connect",
  boardPosition: null,
  boardRows: [],
  teams: null,
  draftTimer: null,
  refreshTimer: null,
};

/* ------------------------------------------------------------------ utils */

const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
};
const fmt = (value, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value) ? "–" : Number(value).toFixed(digits);
const pct = (value, digits = 1) =>
  value === null || value === undefined ? "–" : `${(Number(value) * 100).toFixed(digits)}%`;
/* A header must sit over its column the way the cells do: a left-aligned
 * label above right-aligned numbers reads as two different columns. Mark
 * numeric columns with num("...") and the header inherits the alignment. */
const num = (label) => ({ label, num: true });
const tableHead = (labels) => {
  const head = el("thead");
  const row = el("tr");
  labels.forEach((label) => {
    const isNum = typeof label === "object" && label.num;
    row.append(el("th", isNum ? "num-col" : null, isNum ? label.label : label));
  });
  head.append(row);
  return head;
};

const posSpan = (position) => {
  const node = el("span", `pos pos-${position}`, position);
  return node;
};

async function api(path, options) {
  const response = await fetch(path, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch (err) {
    payload = null;
  }
  if (!response.ok) {
    const error = new Error((payload && (payload.detail || payload.error)) || response.statusText);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function showError(node, error) {
  // A <p> inside a <table> gets hoisted out by the parser, so errors for
  // table-based views go into the wrapper next to the table — and the table
  // element itself survives, because the next successful load re-fills it.
  const isTable = node.tagName === "TABLE";
  const container = isTable ? node.parentElement : node;
  if (isTable) node.innerHTML = "";
  else container.innerHTML = "";
  container.querySelectorAll(".error-note").forEach((n) => n.remove());

  const loading = error.status === 425;
  const stage = (error.payload && error.payload.status) || {};
  const message = loading
    ? stage.detail || "Still loading projections…"
    : error.message || String(error);
  container.append(el("p", `error-note ${loading ? "muted" : "error"}`, message));
}

/* ------------------------------------------------------------------- tabs */

const LOADERS = {
  draft: loadDraft,
  matchup: loadMatchup,
  teams: loadTeams,
  board: loadBoard,
  waivers: loadWaivers,
  trades: loadTrades,
  model: loadModel,
};

function setView(view) {
  state.view = view;
  document.querySelectorAll(".view").forEach((section) => {
    section.classList.toggle("active", section.id === `view-${view}`);
  });
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === view);
  });
  if (LOADERS[view]) LOADERS[view]();
  scheduleAutoRefresh();
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setView(tab.dataset.view));
});

/* ----------------------------------------------------------- auto-refresh */

/* How often each view re-asks the server, in seconds. The server re-polls
 * Sleeper at most every 30s regardless, so these are about how quickly a change
 * that already reached the server reaches the screen. The draft board is the
 * one place seconds matter; the model card never changes at all. */
const REFRESH_SECONDS = { draft: 20, matchup: 90, teams: 120, board: 300, waivers: 180, trades: 0, model: 0 };

function scheduleAutoRefresh() {
  clearInterval(state.refreshTimer);
  const seconds = REFRESH_SECONDS[state.view] || 0;
  if (!seconds || !state.league || !state.status || !state.status.ready) return;
  state.refreshTimer = setInterval(() => {
    // Never poll a tab nobody is looking at — it wastes Sleeper's bandwidth and
    // the browser throttles background timers anyway.
    if (document.hidden) return;
    refreshCurrentView({ quiet: true });
  }, seconds * 1000);
}

async function refreshCurrentView({ quiet = false } = {}) {
  if (!state.league || !state.status || !state.status.ready) return;
  setLiveState("syncing");
  try {
    const result = await api("/api/refresh", { method: "POST" });
    if (result.league) applyLeague(result.league);
    if (LOADERS[state.view]) await LOADERS[state.view]();
    setLiveState("live", result.changed);
  } catch (error) {
    setLiveState("stale");
    if (!quiet) throw error;
  }
}

function setLiveState(mode, changed) {
  const wrap = $("live-state");
  if (!wrap) return;
  wrap.hidden = false;
  wrap.dataset.mode = mode;
  const labels = {
    live: "up to date",
    syncing: "checking Sleeper…",
    stale: "Sleeper unreachable — showing cached",
  };
  let label = labels[mode] || mode;
  if (mode === "live" && changed && (changed.rosters || changed.players)) {
    label = changed.rosters ? "rosters updated" : "injury status updated";
  }
  $("live-label").textContent = label;
}

// A tab left open overnight should not show yesterday's rosters the moment it
// is focused again.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refreshCurrentView({ quiet: true });
});
window.addEventListener("focus", () => refreshCurrentView({ quiet: true }));

/* ---------------------------------------------------------------- connect */

$("lookup-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.target.querySelector("button");
  const username = $("username").value.trim();
  button.disabled = true;
  $("lookup-error").hidden = true;
  try {
    const data = await api("/api/leagues", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username }),
    });
    renderLeagueChoices(data, username);
  } catch (error) {
    $("lookup-error").textContent = error.message;
    $("lookup-error").hidden = false;
    $("league-choices").hidden = true;
  } finally {
    button.disabled = false;
  }
});

function renderLeagueChoices(data, username) {
  const container = $("league-choices");
  container.innerHTML = "";
  container.hidden = false;

  const leagues = data.leagues || [];
  const previous = data.previous_season_leagues || [];
  if (!leagues.length && !previous.length) {
    container.append(
      el(
        "p",
        "muted small",
        `Sleeper knows ${data.user.display_name || username}, but has no ` +
          `${data.season} leagues for them. If your league is on ESPN, Yahoo, or ` +
          `NFL.com, this app cannot read it — Sleeper is the only platform wired up.`
      )
    );
    return;
  }

  const add = (rows, note) => {
    if (!rows.length) return;
    if (note) container.append(el("p", "muted small", note));
    rows.forEach((league) => {
      const button = el("button", "team-btn");
      button.append(el("strong", null, league.name || league.league_id));
      const bits = [
        `${league.teams} teams`,
        league.scoring,
        league.superflex ? "superflex" : null,
        league.status && league.status !== "in_season" ? league.status : null,
      ].filter(Boolean);
      button.append(el("small", null, bits.join(" · ")));
      button.addEventListener("click", () => connectTo(league.league_id, username));
      container.append(button);
    });
  };

  add(leagues, null);
  add(
    previous,
    `No ${data.season} leagues yet — these are from ${data.season - 1}. ` +
      "Projections will still be for the current season."
  );
}

async function connectTo(leagueId, username) {
  $("connect-error").hidden = true;
  // Switching leagues invalidates every loaded view, so go back to the shell
  // and let the loading card explain what is happening.
  clearInterval(state.refreshTimer);
  clearInterval(state.draftTimer);
  state.status = null;
  setView("connect");
  try {
    const data = await api("/api/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ league_id: leagueId, username: username || null }),
    });
    applyLeague(data.league);
    $("league-choices").hidden = true;
    $("loading-card").hidden = false;
    pollStatus();
  } catch (error) {
    $("connect-error").textContent = error.message;
    $("connect-error").hidden = false;
  }
}

$("connect-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.target.querySelector("button");
  button.disabled = true;
  $("connect-error").hidden = true;
  try {
    const data = await api("/api/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        league_id: $("league-id").value.trim(),
        username: $("username").value.trim() || null,
      }),
    });
    applyLeague(data.league);
    $("loading-card").hidden = false;
    pollStatus();
  } catch (error) {
    $("connect-error").textContent = error.message;
    $("connect-error").hidden = false;
  } finally {
    button.disabled = false;
  }
});

/* ------------------------------------------------------------------- platform */

document.querySelectorAll(".platform-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const wanted = tab.dataset.platform;
    document.querySelectorAll(".platform-tab").forEach((other) => {
      const on = other === tab;
      other.classList.toggle("active", on);
      other.setAttribute("aria-selected", on ? "true" : "false");
    });
    $("panel-sleeper").hidden = wanted !== "sleeper";
    $("panel-espn").hidden = wanted !== "espn";
  });
});

$("espn-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.target.querySelector("button");
  button.disabled = true;
  $("espn-error").hidden = true;
  const season = $("espn-season").value.trim();
  try {
    const data = await api("/api/connect/espn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        league_id: $("espn-league-id").value.trim(),
        season: season ? Number(season) : null,
        espn_s2: $("espn-s2").value.trim() || null,
        swid: $("espn-swid").value.trim() || null,
      }),
    });
    applyLeague(data.league);
    $("loading-card").hidden = false;
    pollStatus();
  } catch (error) {
    $("espn-error").textContent = error.message;
    $("espn-error").hidden = false;
    // A private league needs cookies; open that section rather than making
    // the user find it after reading the error.
    if (/cookie|espn_s2|SWID/i.test(error.message)) {
      $("espn-private").open = true;
    }
  } finally {
    button.disabled = false;
  }
});

/* --------------------------------------------------------- remembered leagues */

function renderRemembered(leagues) {
  const wrap = $("remembered");
  const list = $("remembered-list");
  const rows = (leagues || []).filter((lg) => !lg.is_active);
  list.innerHTML = "";
  wrap.hidden = rows.length === 0;
  rows.forEach((league) => {
    const button = el("button", "team-btn");
    button.append(el("strong", null, league.name || league.league_id));
    const bits = [
      league.my_team || (league.teams ? `${league.teams} teams` : null),
      league.scoring,
      // Reopening a league whose model is already trained is instant; one that
      // needs a rebuild is a few minutes, and saying so up front is kinder.
      league.model_ready ? "ready" : "needs training",
    ].filter(Boolean);
    button.append(el("small", null, bits.join(" · ")));
    button.addEventListener("click", () => connectTo(league.league_id, league.username));
    list.append(button);
  });
}

function renderSwitcher(league) {
  const known = (league && league.known_leagues) || [];
  const switcher = $("switcher");
  switcher.hidden = known.length < 2;
  const menu = $("switcher-menu");
  menu.innerHTML = "";
  known.forEach((entry) => {
    const row = el("button", `switcher-item${entry.is_active ? " active" : ""}`);
    row.append(el("strong", null, entry.name || entry.league_id));
    const bits = [entry.my_team, entry.scoring, entry.model_ready ? null : "needs training"]
      .filter(Boolean)
      .join(" · ");
    row.append(el("small", null, bits));
    row.addEventListener("click", () => {
      menu.hidden = true;
      if (!entry.is_active) connectTo(entry.league_id, entry.username);
    });
    menu.append(row);
  });
}

$("switcher-toggle").addEventListener("click", (event) => {
  event.stopPropagation();
  const menu = $("switcher-menu");
  menu.hidden = !menu.hidden;
});
document.addEventListener("click", () => ($("switcher-menu").hidden = true));

function applyLeague(league) {
  state.league = league;
  renderRemembered(league && league.known_leagues);
  if (!league || !league.connected) return;
  renderSwitcher(league);
  $("tabs").hidden = false;
  const bits = [
    league.name,
    `${league.teams} teams`,
    league.scoring,
    league.superflex ? "superflex" : null,
    league.dynasty ? "dynasty" : null,
  ].filter(Boolean);
  $("league-line").textContent = bits.join(" · ");
  $("matchup-week").value = league.current_week;
  if (!$("teams-week").value) $("teams-week").value = league.current_week;

  if (!league.my_roster_id) {
    $("team-card").hidden = false;
    const list = $("team-list");
    list.innerHTML = "";
    league.teams_list.forEach((team) => {
      const button = el("button", "team-btn");
      button.append(el("strong", null, team.label));
      button.append(el("small", null, `${team.owner} · ${team.record}`));
      button.addEventListener("click", async () => {
        const data = await api("/api/team", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ roster_id: team.roster_id }),
        });
        applyLeague(data.league);
        $("team-card").hidden = true;
      });
      list.append(button);
    });
  } else {
    $("team-card").hidden = true;
  }
}

async function pollStatus() {
  try {
    const data = await api("/api/status");
    state.status = data.status;
    if (data.league && data.league.connected) applyLeague(data.league);
    $("loading-stage").textContent = data.status.ready ? "Ready" : "Loading projections";
    $("progress-bar").style.width = `${Math.round(data.status.progress * 100)}%`;
    const elapsed = data.status.elapsed_seconds ? ` (${Math.round(data.status.elapsed_seconds)}s)` : "";
    $("loading-detail").textContent = (data.status.error || data.status.detail || "") + elapsed;
    if (data.status.error) return;
    if (data.status.ready) {
      $("loading-card").hidden = true;
      if (state.view === "connect") setView("draft");
      else scheduleAutoRefresh();
      setLiveState("live");
      return;
    }
  } catch (error) {
    $("loading-detail").textContent = error.message;
  }
  setTimeout(pollStatus, 2000);
}

/* ------------------------------------------------------------------ draft */

$("draft-refresh").addEventListener("click", () => refreshCurrentView());
$("draft-autorefresh").addEventListener("change", (event) => {
  clearInterval(state.draftTimer);
  // Sleeper's pick feed is cached for a few seconds server-side, so polling
  // faster than this would just re-read the same response.
  if (event.target.checked) {
    state.draftTimer = setInterval(() => {
      if (!document.hidden) refreshCurrentView({ quiet: true });
    }, 6000);
  }
});

async function loadDraft() {
  const container = $("draft-recs");
  try {
    const data = await api("/api/draft");
    renderDraftStatus(data.draft);
    renderRecommendations(container, data.recommendations);
    renderRoster(data.roster, data.needs);
    renderRuns(data.positional_runs);
  } catch (error) {
    showError(container, error);
  }
}

function renderDraftStatus(draft) {
  const row = $("draft-status");
  row.innerHTML = "";
  if (!draft || !draft.draft_id) {
    row.append(el("span", "pill", "No draft found for this league"));
    return;
  }
  const pills = [
    `${draft.type} · ${draft.teams} teams · ${draft.rounds} rounds`,
    `pick ${draft.current_pick}`,
    draft.my_slot ? `your slot ${draft.my_slot}` : "slot unknown",
    draft.my_next_pick ? `your next: ${draft.my_next_pick}` : null,
  ].filter(Boolean);
  if (draft.is_my_turn) {
    const pill = el("span", "pill on-clock", "You are on the clock");
    row.append(pill);
  }
  pills.forEach((text) => row.append(el("span", "pill", text)));
}

function renderRecommendations(container, recommendations) {
  container.innerHTML = "";
  if (!recommendations || !recommendations.length) {
    container.append(el("p", "muted", "No candidates — the board is empty."));
    return;
  }
  recommendations.forEach((player, index) => {
    const row = el("div", index === 0 ? "rec top" : "rec");
    row.append(el("div", "rank", String(index + 1)));

    const who = el("div", "who");
    const title = el("div", "name");
    title.append(posSpan(player.position));
    title.append(document.createTextNode(` ${player.name}`));
    if (player.team) title.append(el("span", "muted small", ` ${player.team}`));
    if (player.bye_week) title.append(el("span", "muted small", ` · bye ${player.bye_week}`));
    who.append(title);
    who.append(el("div", "why", player.reason));
    row.append(who);

    const num = el("div", "num");
    num.append(el("b", null, fmt(player.marginal_value, 0)));
    num.append(el("span", null, `${fmt(player.projected_points, 0)} pts · ADP ${fmt(player.ecr, 0)}`));
    row.append(num);

    row.addEventListener("click", () => openPlayer(player.sleeper_id));
    container.append(row);
  });
}

function renderRoster(roster, needs) {
  const container = $("draft-roster");
  container.innerHTML = "";
  if (!roster || !roster.length) {
    container.append(el("span", "muted small", "Nothing drafted yet."));
  } else {
    roster.forEach((player) => {
      const chip = el("span", "chip");
      chip.append(posSpan(player.position || "?"));
      chip.append(document.createTextNode(player.name || player.sleeper_id));
      container.append(chip);
    });
  }
  $("draft-needs").textContent = needs && needs.length
    ? `Still unfilled: ${needs.join(", ")}`
    : "Every starting slot is filled.";
}

function renderRuns(runs) {
  const container = $("draft-runs");
  container.innerHTML = "";
  const entries = Object.entries(runs || {});
  if (!entries.length) {
    container.append(el("span", "muted small", "Needs a known next pick to compute."));
    return;
  }
  const max = Math.max(...entries.map(([, value]) => Math.abs(value)), 1);
  entries.forEach(([position, value]) => {
    const row = el("div", "bar-row");
    row.append(posSpan(position));
    const bar = el("div", "bar");
    const fill = el("div");
    fill.style.width = `${Math.max(2, (Math.abs(value) / max) * 100)}%`;
    bar.append(fill);
    row.append(bar);
    row.append(el("span", "num-col", fmt(value, 0)));
    container.append(row);
  });
}

/* ---------------------------------------------------------------- matchup */

$("matchup-refresh").addEventListener("click", () => refreshCurrentView());
$("matchup-week").addEventListener("change", () => loadMatchup());
$("opponent-mode").addEventListener("change", () => loadMatchup());

async function loadMatchup() {
  const headline = $("matchup-headline");
  try {
    const week = $("matchup-week").value || (state.league && state.league.current_week) || 1;
    const mode = $("opponent-mode").value;
    const data = await api(`/api/matchup?week=${week}&opponent_mode=${mode}`);
    renderMatchupHeadline(headline, data);
    renderLineup($("matchup-lineup"), data.lineup);
    renderLineup($("matchup-opponent"), data.opponent_lineup);
    renderLeverage(data);
    renderSwaps(data.swaps, [...(data.players || []), ...(data.opponent_players || [])]);
    renderMatchupPlayers(data.players);
  } catch (error) {
    showError(headline, error);
  }
}

function renderMatchupHeadline(container, data) {
  container.innerHTML = "";
  const prob = el("div", "winprob");
  prob.append(el("b", null, pct(data.win_probability, 0)));
  prob.append(el("span", null, "win probability"));
  container.append(prob);

  const right = el("div");
  const score = el("div", "score-line");
  score.append(el("strong", null, `${data.my_team} ${fmt(data.my_distribution.mean)}`));
  score.append(el("span", "vs", "vs"));
  score.append(document.createTextNode(`${data.opponent_team || "bye"} ${fmt(data.opponent_distribution.mean)}`));
  right.append(score);
  right.append(
    el(
      "div",
      "muted small",
      `Your range: ${fmt(data.my_distribution.p10)} – ${fmt(data.my_distribution.p90)} ` +
        `(80% of simulated weeks). Expected margin ${fmt(data.margin_distribution.mean)}.`
    )
  );
  right.append(el("div", "strategy", data.strategy));
  (data.notes || []).forEach((note) => right.append(el("div", "muted small", note)));
  container.append(right);
}

function renderLineup(table, lineup) {
  table.innerHTML = "";
  if (!lineup || !lineup.length) {
    table.append(el("tr")).append(el("td", "muted", "No lineup available."));
    return;
  }
  const body = el("tbody");
  lineup.forEach((row) => {
    const tr = el("tr", "clickable");
    tr.append(el("td", null, row.slot));
    const name = el("td");
    name.append(posSpan(row.position || "?"));
    name.append(document.createTextNode(` ${row.name || row.sleeper_id}`));
    tr.append(name);
    tr.append(el("td", "num-col", fmt(row.projection)));
    tr.addEventListener("click", () => openPlayer(row.sleeper_id));
    body.append(tr);
  });
  table.append(body);
}

function renderLeverage(data) {
  const block = $("leverage-block");
  block.innerHTML = "";
  if (!data.leverage_lineup || !data.leverage_gain) return;
  block.append(el("h3", null, `Win-probability lineup (+${pct(data.leverage_gain, 1)})`));
  block.append(
    el(
      "p",
      "muted small",
      "Scores fewer points on average, but wins this matchup more often. " +
        "Differences from the lineup above are the swaps worth making."
    )
  );
  const table = el("table", "lineup compact");
  renderLineup(table, data.leverage_lineup);
  block.append(table);
}

function renderSwaps(swaps, players) {
  const container = $("matchup-swaps");
  container.innerHTML = "";
  const names = {};
  (players || []).forEach((player) => (names[player.sleeper_id] = player));
  const useful = (swaps || []).filter((swap) => swap.win_prob_delta > 0.0005);
  if (!useful.length) {
    container.append(el("p", "muted small", "Your lineup is already the best available."));
    return;
  }
  useful.slice(0, 6).forEach((swap) => {
    const row = el("div", "swap");
    const positive = swap.win_prob_delta >= 0;
    const delta = el("span", `delta ${positive ? "pos" : "neg"}`,
      `${positive ? "+" : ""}${(swap.win_prob_delta * 100).toFixed(1)}%`);
    row.append(delta);
    row.append(document.createTextNode(" start "));
    row.append(el("span", "starter-in", (names[swap.in] || {}).name || swap.in));
    row.append(document.createTextNode(" over "));
    row.append(el("span", "starter-out", (names[swap.out] || {}).name || swap.out));
    row.append(el("span", "muted small", ` at ${swap.slot} (${swap.points_delta >= 0 ? "+" : ""}${fmt(swap.points_delta)} pts)`));
    container.append(row);
  });
}

function renderMatchupPlayers(players) {
  const table = $("matchup-players");
  table.innerHTML = "";
  table.append(
    tableHead(
      ["Player", "Pos", "Team", "Opp", num("Proj"), num("Floor"), num("Ceiling"), num("P(play)"), "Slot"]
    )
  );

  const body = el("tbody");
  (players || []).forEach((player) => {
    const tr = el("tr", "clickable");
    tr.append(el("td", null, player.name));
    const posCell = el("td");
    posCell.append(posSpan(player.position));
    tr.append(posCell);
    tr.append(el("td", null, player.team || ""));
    tr.append(el("td", null, player.opponent || ""));
    tr.append(el("td", "num-col", fmt(player.projection)));
    tr.append(el("td", "num-col", fmt(player.floor)));
    tr.append(el("td", "num-col", fmt(player.ceiling)));
    tr.append(el("td", "num-col", pct(player.p_play, 0)));
    tr.append(el("td", null, player.optimal_slot || "bench"));
    tr.addEventListener("click", () => openPlayer(player.sleeper_id));
    body.append(tr);
  });
  table.append(body);
}

/* ------------------------------------------------------------------ league */

$("teams-refresh").addEventListener("click", () => refreshCurrentView());
$("teams-week").addEventListener("change", () => loadTeams());
$("teams-mode").addEventListener("change", () => renderTeams());
$("teams-bench").addEventListener("change", () => renderTeams());

async function loadTeams() {
  const grid = $("teams-grid");
  try {
    const week = $("teams-week").value || (state.league && state.league.current_week) || 1;
    state.teams = await api(`/api/teams?week=${week}`);
    renderTeams();
  } catch (error) {
    showError(grid, error);
  }
}

function renderTeams() {
  const data = state.teams;
  const grid = $("teams-grid");
  const summary = $("teams-summary");
  grid.innerHTML = "";
  summary.innerHTML = "";
  if (!data || !data.teams) return;

  const declared = $("teams-mode").value === "declared";
  const showBench = $("teams-bench").checked;
  // Before the draft every roster is empty and every projection is zero, so the
  // server orders by draft slot instead; re-sorting by points here would undo it.
  const undrafted = !data.teams.some((team) => team.roster_size > 0);
  const teams = undrafted
    ? data.teams.slice()
    : data.teams
        .slice()
        .sort((a, b) =>
          declared ? b.declared_points - a.declared_points : b.projected_points - a.projected_points
        );

  summary.append(el("h2", null, undrafted ? "Every team" : `Week ${data.week} — every team`));
  summary.append(
    el(
      "p",
      "muted small",
      undrafted
        ? "This league has not drafted yet, so there are no lineups to compare. " +
          "Teams are listed in draft order where Sleeper has set one. Head to the " +
          "Draft tab for the board."
        : declared
        ? "Ranked by the lineup each manager currently has set. Teams that have not set one show their best possible instead."
        : `Ranked by the best lineup each roster could field — the honest measure of team strength, ` +
          `independent of whether the manager has logged in. League average ${fmt(data.averages.projected_points)}.`
    )
  );

  teams.forEach((team, index) => {
    const points = declared ? team.declared_points : team.projected_points;
    const card = el("div", `card team-card${team.is_me ? " mine" : ""}`);

    const head = el("div", "team-head");
    const left = el("div");
    const title = el("div", "team-name");
    title.append(el("span", "team-rank", `${index + 1}`));
    title.append(document.createTextNode(team.label));
    if (team.is_me) title.append(el("span", "tag", "you"));
    left.append(title);
    const meta = [team.owner, team.record, `${fmt(team.points_for, 0)} pts for`]
      .filter(Boolean)
      .join(" · ");
    left.append(el("div", "muted small", meta));
    if (team.opponent_label) {
      left.append(el("div", "muted small", `vs ${team.opponent_label} this week`));
    }
    head.append(left);

    const score = el("div", "team-score");
    if (undrafted) {
      score.append(el("b", null, team.draft_slot ? `#${team.draft_slot}` : "–"));
      score.append(el("span", null, team.draft_slot ? "draft slot" : "no slot yet"));
    } else {
      score.append(el("b", null, fmt(points)));
      score.append(el("span", null, "projected"));
    }
    head.append(score);
    card.append(head);

    const rows = declared && team.declared.length ? team.declared : team.starters;
    if (!rows.length) {
      (team.notes || []).forEach((note) => card.append(el("p", "muted small", note)));
      grid.append(card);
      return;
    }
    const table = el("table", "lineup compact");
    const body = el("tbody");
    rows.forEach((player) => {
      const tr = el("tr", "clickable");
      tr.append(el("td", null, player.slot || ""));
      const name = el("td");
      name.append(posSpan(player.position || "?"));
      name.append(document.createTextNode(` ${player.name}`));
      if (player.p_play !== null && player.p_play < 0.9) {
        name.append(el("span", "tag warn", `${Math.round(player.p_play * 100)}%`));
      }
      if (!player.opponent) name.append(el("span", "tag", "bye"));
      tr.append(name);
      tr.append(el("td", "muted small", player.opponent || ""));
      tr.append(el("td", "num-col", fmt(player.projection)));
      tr.addEventListener("click", () => openPlayer(player.sleeper_id));
      body.append(tr);
    });
    table.append(body);
    card.append(table);

    if (team.points_left_on_bench > 0 && !declared) {
      card.append(
        el(
          "p",
          "muted small",
          `Currently set to score ${fmt(team.declared_points)} — ` +
            `${fmt(team.points_left_on_bench)} below what this roster could field.`
        )
      );
    }
    if (declared && !team.declared.length) {
      card.append(el("p", "muted small", "No readable lineup set — showing best possible."));
    }
    (team.notes || []).forEach((note) => card.append(el("p", "muted small", note)));

    const strengths = el("div", "strength-row");
    Object.entries(team.position_strength).forEach(([position, value]) => {
      const average = data.averages[`strength_${position}`];
      const chip = el("span", "chip");
      chip.append(posSpan(position));
      chip.append(document.createTextNode(fmt(value, 0)));
      if (average) {
        const delta = value - average;
        chip.append(
          el("span", `delta ${delta >= 0 ? "pos" : "neg"}`, ` ${delta >= 0 ? "+" : ""}${fmt(delta, 0)}`)
        );
        chip.title = `League average ${fmt(average, 0)}`;
      }
      strengths.append(chip);
    });
    card.append(strengths);

    if (showBench && team.bench.length) {
      const benchTable = el("table", "lineup compact bench");
      const benchBody = el("tbody");
      team.bench.slice(0, 10).forEach((player) => {
        const tr = el("tr", "clickable");
        tr.append(el("td", "muted small", "BN"));
        const name = el("td");
        name.append(posSpan(player.position || "?"));
        name.append(document.createTextNode(` ${player.name}`));
        tr.append(name);
        tr.append(el("td", "muted small", player.opponent || "bye"));
        tr.append(el("td", "num-col", fmt(player.projection)));
        tr.addEventListener("click", () => openPlayer(player.sleeper_id));
        benchBody.append(tr);
      });
      benchTable.append(benchBody);
      card.append(el("h3", null, "Bench"));
      card.append(benchTable);
    }

    grid.append(card);
  });
}

/* ------------------------------------------------------------------ board */

$("board-search").addEventListener("input", () => renderBoard());

async function loadBoard() {
  const table = $("board-table");
  try {
    const query = state.boardPosition ? `?position=${state.boardPosition}&limit=400` : "?limit=400";
    const data = await api(`/api/board${query}`);
    state.boardRows = data.players || [];
    renderBoardFilters(data.replacement);
    renderBoard();
  } catch (error) {
    showError(table, error);
  }
}

function renderBoardFilters(replacement) {
  const container = $("board-filters");
  container.innerHTML = "";
  const positions = [null, "QB", "RB", "WR", "TE", "K", "DST"];
  positions.forEach((position) => {
    const pill = el("span", `pill${state.boardPosition === position ? " active" : ""}`);
    const button = el("button", null, position || "All");
    if (position && replacement && replacement[position] !== undefined) {
      button.title = `replacement level: ${replacement[position]} points`;
    }
    button.addEventListener("click", () => {
      state.boardPosition = position;
      loadBoard();
    });
    pill.append(button);
    container.append(pill);
  });
}

function renderBoard() {
  const table = $("board-table");
  const query = $("board-search").value.trim().toLowerCase();
  const rows = state.boardRows.filter(
    (row) => !query || String(row.name).toLowerCase().includes(query)
  );
  table.innerHTML = "";
  table.append(
    tableHead(
      ["#", "Player", "Pos", "Team", num("Proj"), num("VOR"), num("Tier"), num("ADP"), num("±"), num("Bye"), "Range"]
    )
  );

  const body = el("tbody");
  // Scale the range bars across what is actually on screen. Anchoring at zero
  // would squash every bar into the right-hand third and show nothing.
  const ceilings = rows.map((r) => r.ceiling).filter((v) => v !== null && v !== undefined);
  const floors = rows.map((r) => r.floor).filter((v) => v !== null && v !== undefined);
  const scale = {
    min: floors.length ? Math.min(...floors) : 0,
    max: ceilings.length ? Math.max(...ceilings) : 1,
  };
  rows.forEach((row, index) => {
    const tr = el("tr", "clickable");
    tr.append(el("td", "muted", String(index + 1)));
    tr.append(el("td", null, row.name));
    const posCell = el("td");
    posCell.append(posSpan(row.position));
    posCell.append(document.createTextNode(String(row.positional_rank ?? "")));
    tr.append(posCell);
    tr.append(el("td", null, row.team || ""));
    tr.append(el("td", "num-col", fmt(row.projected_points, 0)));
    tr.append(el("td", "num-col", fmt(row.vor, 0)));
    tr.append(el("td", "num-col", String(row.tier ?? "")));
    tr.append(el("td", "num-col", fmt(row.ecr, 0)));
    tr.append(el("td", "num-col", fmt(row.adp_sd, 0)));
    tr.append(el("td", "num-col", row.bye_week ?? ""));
    tr.append(rangeCell(row.floor, row.projected_points, row.ceiling, scale));
    tr.addEventListener("click", () => openPlayer(row.sleeper_id));
    body.append(tr);
  });
  table.append(body);
}

function rangeCell(floor, mid, ceiling, scale) {
  const cell = el("td");
  if (floor === undefined || ceiling === undefined || floor === null) return cell;
  const span = Math.max(scale.max - scale.min, 1);
  const at = (value) => Math.min(100, Math.max(0, ((value - scale.min) / span) * 100));
  const wrap = el("div", "range");
  const track = el("div", "range-track");
  const fill = el("div", "range-fill");
  fill.style.left = `${at(floor)}%`;
  fill.style.width = `${Math.max(1, at(ceiling) - at(floor))}%`;
  track.append(fill);
  const mark = el("div", "range-mark");
  mark.style.left = `${at(mid)}%`;
  track.append(mark);
  wrap.append(track);
  wrap.title = `${fmt(floor, 0)} – ${fmt(ceiling, 0)}`;
  cell.append(wrap);
  return cell;
}

/* ---------------------------------------------------------------- waivers */

$("waivers-refresh").addEventListener("click", () => refreshCurrentView());
$("trades-refresh").addEventListener("click", () => loadTrades());

async function loadWaivers() {
  const table = $("waivers-table");
  try {
    const data = await api("/api/waivers");
    table.innerHTML = "";
    table.append(
      tableHead(
        ["Player", "Pos", "Team", num("ROS pts"), num("Roster gain"), num("This week"), num("Adds (24h)"), "Note"]
      )
    );

    const body = el("tbody");
    if (!data.targets.length) {
      const tr = el("tr");
      tr.append(el("td", "muted", "Nothing on the wire beats what you already have."));
      body.append(tr);
    }
    data.targets.forEach((target) => {
      const tr = el("tr", "clickable");
      tr.append(el("td", null, target.name));
      const posCell = el("td");
      posCell.append(posSpan(target.position));
      tr.append(posCell);
      tr.append(el("td", null, target.team || ""));
      tr.append(el("td", "num-col", fmt(target.projected_points, 0)));
      tr.append(el("td", "num-col", `+${fmt(target.roster_gain, 0)}`));
      tr.append(el("td", "num-col", `${target.weekly_gain >= 0 ? "+" : ""}${fmt(target.weekly_gain)}`));
      tr.append(el("td", "num-col", String(target.trending_adds || 0)));
      tr.append(el("td", "muted small", target.note));
      tr.addEventListener("click", () => openPlayer(target.sleeper_id));
      body.append(tr);
    });
    table.append(body);
    $("waivers-notes").textContent = (data.notes || []).join(" ");
  } catch (error) {
    showError(table, error);
  }
}


/* ----------------------------------------------------------------- trades */

function tradePlayerChip(players, id) {
  const info = players[id] || { name: id, position: "?", ros_points: 0 };
  const chip = el("span", "trade-player clickable");
  chip.append(posSpan(info.position));
  chip.append(el("span", null, ` ${info.name} `));
  chip.append(el("small", "muted", `${fmt(info.ros_points, 0)} ros`));
  chip.addEventListener("click", () => openPlayer(id));
  return chip;
}

function tradeSideBlock(players, side, heading) {
  const block = el("div", "trade-side");
  block.append(el("h4", null, heading));
  const list = el("div", "trade-chips");
  side.gives.forEach((id) => list.append(tradePlayerChip(players, id)));
  block.append(list);
  if (side.adds.length) {
    const extra = el("p", "muted small");
    extra.textContent =
      "then adds from waivers: " +
      side.adds.map((id) => (players[id] || { name: id }).name).join(", ");
    block.append(extra);
  }
  if (side.drops.length) {
    const extra = el("p", "muted small");
    extra.textContent =
      "then drops: " +
      side.drops.map((id) => (players[id] || { name: id }).name).join(", ");
    block.append(extra);
  }
  return block;
}

function tradeCard(players, trade) {
  const card = el("div", "trade-card");
  const header = el("div", "trade-header");
  header.append(el("strong", null, `Trade with ${trade.them.label}`));
  const badge = el("span", `badge trade-${trade.likelihood.replace(/\s+/g, "-")}`);
  badge.textContent = trade.likelihood;
  header.append(badge);
  card.append(header);

  const grid = el("div", "trade-grid");
  grid.append(tradeSideBlock(players, trade.me, "You send"));
  grid.append(tradeSideBlock(players, trade.them, "You receive"));
  card.append(grid);

  const gains = el("p", "trade-gains");
  gains.append(el("span", "good", `You: +${fmt(trade.me.gain, 1)} ros pts`));
  gains.append(el("span", "muted", " · "));
  gains.append(el("span", null, `${trade.them.label}: +${fmt(trade.them.gain, 1)}`));
  card.append(gains);
  card.append(el("p", "muted small", trade.rationale));
  return card;
}

async function loadTrades() {
  const list = $("trades-list");
  const chainsCard = $("trades-chains-card");
  const chainsBox = $("trades-chains");
  try {
    list.innerHTML = "";
    list.append(el("p", "muted", "Searching every roster for deals that work both ways…"));
    const data = await api("/api/trades");
    list.innerHTML = "";
    chainsBox.innerHTML = "";

    if (!data.trades.length) {
      list.append(
        el("p", "muted", "No trade clears the bar right now — nothing you want is available at a price the other side should take.")
      );
    }
    data.trades.forEach((trade) => list.append(tradeCard(data.players, trade)));

    chainsCard.hidden = !data.chains.length;
    data.chains.forEach((chain, index) => {
      const wrap = el("div", "trade-chain");
      wrap.append(
        el("h3", null, `Chain ${index + 1}: +${fmt(chain.total_gain, 1)} ros pts total`)
      );
      chain.steps.forEach((step, si) => {
        const stepWrap = el("div", "trade-chain-step");
        stepWrap.append(el("div", "chain-step-label muted", `Step ${si + 1}`));
        stepWrap.append(tradeCard(data.players, step));
        wrap.append(stepWrap);
      });
      chainsBox.append(wrap);
    });

    $("trades-notes").textContent = (data.notes || []).join(" ");
  } catch (error) {
    showError(list, error);
  }
}

/* ------------------------------------------------------------------ model */

async function loadModel() {
  const card = $("model-card");
  try {
    const data = await api("/api/model");
    card.innerHTML = "";
    if (!data.trained) {
      card.append(el("p", "muted", "The model has not finished training yet."));
      return;
    }
    card.append(el("h2", null, "How the projections are made"));
    card.append(
      el(
        "p",
        "muted small",
        `Trained on NFL seasons ${data.seasons[0]}–${data.seasons[data.seasons.length - 1]} ` +
          `using ${data.n_features} features, scored under your league's rules (${data.scoring}). ` +
          `Each position gets a separate model per quantile, so the output is a distribution, not a single number.`
      )
    );
    if (data.unsupported_scoring && data.unsupported_scoring.length) {
      card.append(
        el(
          "p",
          "small",
          `Not modelled (needs play-by-play detail weekly box scores do not carry): ${data.unsupported_scoring.join(", ")}.`
        )
      );
    }

    card.append(el("h3", null, "Validation — held-out season"));
    const wrap = el("div", "table-wrap");
    const table = el("table", "data");
    table.append(
      tableHead(
        ["Pos", "Season", num("n"), num("MAE"), num("Baseline MAE"), num("RMSE"), num("Spearman"), num("Bias")]
      )
    );
    const body = el("tbody");
    Object.entries(data.validation).forEach(([position, metrics]) => {
      const tr = el("tr");
      tr.append(el("td", null, position));
      tr.append(el("td", null, String(metrics.holdout_season)));
      tr.append(el("td", "num-col", String(metrics.n_test)));
      tr.append(el("td", "num-col", fmt(metrics.mae, 2)));
      tr.append(el("td", "num-col", fmt(metrics.baseline_mae, 2)));
      tr.append(el("td", "num-col", fmt(metrics.rmse, 2)));
      tr.append(el("td", "num-col", fmt(metrics.spearman, 3)));
      tr.append(el("td", "num-col", fmt(metrics.bias, 2)));
      body.append(tr);
    });
    table.append(body);
    wrap.append(table);
    card.append(wrap);
    card.append(
      el(
        "p",
        "muted small",
        "Baseline is each player's own recent eight-game average — the simplest honest " +
          "prediction. MAE is scored on the median model and RMSE on the mean model, " +
          "since those are the losses each one minimises."
      )
    );

    if (data.availability) {
      card.append(el("h3", null, "Injury designations, measured"));
      const rates = Object.entries(data.availability)
        .map(([status, rate]) => `${status || "no designation"}: ${pct(rate, 0)}`)
        .join(" · ");
      card.append(el("p", "small", rates));
    }

    if (data.correlations && data.correlations.length) {
      card.append(el("h3", null, "Same-game correlations, measured"));
      const list = data.correlations
        .map(([label, value, n]) => `${label} ${value >= 0 ? "+" : ""}${value.toFixed(2)} (n=${n})`)
        .join(" · ");
      card.append(el("p", "small", list));
    }

    card.append(el("h3", null, "What each position's model leans on"));
    Object.entries(data.importance).forEach(([position, features]) => {
      const line = el("p", "small");
      line.append(posSpan(position));
      line.append(
        document.createTextNode(
          " " + features.slice(0, 8).map(([name, weight]) => `${name} ${(weight * 100).toFixed(0)}%`).join(", ")
        )
      );
      card.append(line);
    });
  } catch (error) {
    showError(card, error);
  }
}

/* ------------------------------------------------------------------ modal */

$("modal-close").addEventListener("click", () => ($("player-modal").hidden = true));
$("player-modal").addEventListener("click", (event) => {
  if (event.target.id === "player-modal") $("player-modal").hidden = true;
});

async function openPlayer(sleeperId) {
  const modal = $("player-modal");
  const content = $("modal-content");
  content.innerHTML = "";
  content.append(el("p", "muted", "Loading…"));
  modal.hidden = false;
  try {
    const data = await api(`/api/player/${sleeperId}`);
    content.innerHTML = "";
    const title = el("h2");
    if (data.meta.position) title.append(posSpan(data.meta.position));
    title.append(document.createTextNode(` ${data.meta.name || sleeperId}`));
    content.append(title);
    const meta = [data.meta.team, data.meta.age ? `age ${data.meta.age}` : null,
      data.meta.years_exp !== null && data.meta.years_exp !== undefined ? `${data.meta.years_exp} yrs exp` : null,
      data.meta.injury_status].filter(Boolean).join(" · ");
    content.append(el("p", "muted small", meta));

    if (data.week) {
      content.append(el("h3", null, `Week ${data.week.week} projection`));
      content.append(
        el(
          "p",
          null,
          `${fmt(data.week.proj_mean)} points · floor ${fmt(data.week.floor)} · ceiling ${fmt(data.week.ceiling)} · ` +
            `${pct(data.week.p_play, 0)} to play`
        )
      );
      if (data.week.opponent) {
        content.append(
          el(
            "p",
            "muted small",
            `vs ${data.week.opponent} · team implied total ${fmt(data.week.implied_total)} · ` +
              `opponent allows ${fmt((data.week.opp_fp_allowed_rel || 1) * 100, 0)}% of league-average points to this position`
          )
        );
      }
    }
    if (data.season) {
      content.append(el("h3", null, "Rest of season"));
      content.append(
        el("p", null, `${fmt(data.season.exp_points, 0)} points over ${data.season.games || "?"} games` +
          (data.season.bye_weeks ? ` · bye ${data.season.bye_weeks}` : ""))
      );
    }
    if (data.history && data.history.length) {
      content.append(el("h3", null, "Last 20 games"));
      const max = Math.max(...data.history.map((game) => game.points), 1);
      const spark = el("div", "sparkline");
      data.history.forEach((game) => {
        const bar = el("div");
        bar.style.height = `${Math.max(2, (Math.max(game.points, 0) / max) * 100)}%`;
        bar.title = `${game.season} wk ${game.week} vs ${game.opponent}: ${game.points}`;
        spark.append(bar);
      });
      content.append(spark);
      const recent = data.history.slice(-6).map((game) => `${game.week}: ${fmt(game.points)}`).join("  ");
      content.append(el("p", "muted small", recent));
    }
  } catch (error) {
    showError(content, error);
  }
}

/* ------------------------------------------------------------------- init */

(async function init() {
  try {
    const data = await api("/api/status");
    renderRemembered(data.league && data.league.known_leagues);
    if (data.league && data.league.connected) {
      applyLeague(data.league);
      state.status = data.status;
      if (data.status.ready) {
        setLiveState("live");
        setView("draft");
      } else {
        $("loading-card").hidden = false;
        pollStatus();
      }
    }
  } catch (error) {
    /* first load with no league is the normal case */
  }
})();
