/* ============================================================
   main.js  — v2.3.0

   Changes vs v2.2.0
   -----------------
   RACE CONDITION FIX (squad status cards)
   - renderSquadStatusCards() is no longer called inside renderDashboard().
     Previously, dashboard data could resolve before injury data, leaving
     AppState.injury null when renderDashboard ran, resulting in a silently
     empty squad grid with only a debug log.
   - Squad cards now render in refreshAllData() only after both dashboard
     AND injury data have resolved, regardless of which arrived first.
     renderDashboard() still renders everything else (KPIs, trend chart)
     immediately when its data is available.
   ============================================================ */

const AppState = {
  teams:     [],
  seasons:   [],     // seasons available for the selected team
  player:    null,
  cohesion:  null,
  xg:        null,
  winprob:   null,
  injury:    null,
  eda:       null,   // whole-dataset EDA (team-independent)
  models:    null,   // model registry (team-independent)
};

const _pendingRender = new Set();

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  debug("DOMContentLoaded fired");

  if (window.pagesLoadedPromise) {
    debug("Waiting for pagesLoadedPromise...");
    await window.pagesLoadedPromise;
    debug("Pages loaded, DOM ready");
  }

  Navigation.init();
  Navigation.onNavigate = onPageActivated;
  setupTooltips();
  await bootstrap();
});

async function bootstrap() {
  try {
    const teamSelector = document.getElementById("teamSelector");

    let teamsPayload;
    try {
      teamsPayload = await ApiService.loadTeams();
    } catch (err) {
      console.error("[bootstrap] loadTeams failed:", err);
      showGlobalError("Could not load team list. Is the API server running?");
      return;
    }

    AppState.teams = teamsPayload.teams || [];
    renderTeams(teamSelector, AppState.teams);
    updateDataSourceBadge(teamsPayload.source || "unknown");

    if (!AppState.teams.length) {
      showGlobalError("No teams returned from API. Check /api/health for details.");
      return;
    }

    ApiService.teamId = teamSelector.value;
    debug("Initial team_id:", ApiService.teamId);

    // Populate the season selector for the initial team before the first fetch
    // so all data loads already filtered to a concrete season.
    await loadSeasonsForTeam(ApiService.teamId);

    teamSelector.addEventListener("change", async (e) => {
      ApiService.teamId = e.target.value;
      debug("Team changed to:", ApiService.teamId);
      // A new team has its own set of seasons — repopulate, then refresh.
      await loadSeasonsForTeam(ApiService.teamId);
      await refreshAllData();
    });

    const seasonSelector = document.getElementById("seasonSelector");
    if (seasonSelector) {
      seasonSelector.addEventListener("change", async (e) => {
        ApiService.season = e.target.value;
        debug("Season changed to:", ApiService.season);
        await refreshAllData();
      });
    }

    await refreshAllData();

    // EDA + model-registry data are team-independent — fetch once after the
    // first team refresh so they don't re-fire on every team change.
    loadStaticData();
  } catch (err) {
    console.error("[bootstrap] unhandled error:", err);
    showGlobalError(`Bootstrap error: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Static (team-independent) data: EDA + model registry
// ---------------------------------------------------------------------------

async function loadStaticData() {
  debug("loadStaticData() start");
  const [edaRes, modelsRes] = await Promise.allSettled([
    Promise.resolve().then(() => ApiService.loadEDA()),
    Promise.resolve().then(() => ApiService.loadModels()),
  ]);

  AppState.eda    = edaRes.status    === "fulfilled" ? edaRes.value    : null;
  AppState.models = modelsRes.status === "fulfilled" ? modelsRes.value : null;

  if (edaRes.status === "rejected")
    console.error("[loadStaticData] eda failed:", edaRes.reason);
  if (modelsRes.status === "rejected")
    console.error("[loadStaticData] models failed:", modelsRes.reason);

  _pendingRender.add("eda");
  _pendingRender.add("models");

  // Fill the per-page "About this model" panels across the analytics pages.
  renderModelInfoPanels();

  // The win-probability accuracy panel depends on the model registry, which
  // resolves after the first team refresh — fill it once models are in.
  renderWinProbFactors();

  const active = document.querySelector(".page.active")?.id?.replace("page-", "");
  if (active === "eda" || active === "models") renderPage(active);
  debug("loadStaticData() complete");
}

// ---------------------------------------------------------------------------
// Team selector rendering
// ---------------------------------------------------------------------------

function renderTeams(selector, teams) {
  if (!selector) return;
  if (!teams.length) {
    selector.innerHTML = "<option value=''>No teams available</option>";
    selector.disabled  = true;
    return;
  }
  selector.innerHTML = teams
    .map((t) => `<option value="${t.team_id}">${t.team_name}</option>`)
    .join("");
  selector.disabled = false;
  debug("Rendered", teams.length, "teams in selector");
}

// ---------------------------------------------------------------------------
// Season selector — dynamic per team. Lists the seasons the team appears in
// (most-played first) plus an "All seasons" option, and sets ApiService.season
// to the team's primary season so the first load is already filtered.
// ---------------------------------------------------------------------------

async function loadSeasonsForTeam(teamId) {
  const sel = document.getElementById("seasonSelector");
  let seasons = [];
  try {
    const payload = await ApiService.loadSeasons(teamId);
    seasons = (payload.seasons || []).map((s) => s.season);
  } catch (err) {
    console.error("[loadSeasonsForTeam] failed:", err);
  }
  AppState.seasons = seasons;

  if (!sel) {
    ApiService.season = seasons[0] || "all";
    return;
  }

  if (!seasons.length) {
    sel.innerHTML = '<option value="all">All seasons</option>';
    ApiService.season = "all";
    sel.disabled = true;
    return;
  }

  sel.disabled = false;
  sel.innerHTML =
    seasons.map((s) => `<option value="${s}">${s}</option>`).join("") +
    '<option value="all">All seasons</option>';
  // Default to the team's primary (most-played) season.
  ApiService.season = seasons[0];
  sel.value = seasons[0];
}

// ---------------------------------------------------------------------------
// Full refresh
// ---------------------------------------------------------------------------

async function refreshAllData() {
  debug("refreshAllData() start, team_id=", ApiService.teamId);

  document.querySelectorAll(".kpi-value").forEach((el) => el.classList.add("shimmer"));

  ["dashboard", "player", "cohesion", "xg", "winprob", "injury"].forEach(
    (p) => _pendingRender.add(p)
  );

  // Wrap each call in Promise.resolve().then(fn) so a *synchronous* throw
  // (e.g. a stale cached api.js missing a method) becomes an isolated
  // rejection instead of aborting the whole refresh and blanking every panel.
  const results = await Promise.allSettled([
    () => ApiService.loadPlayer(),
    () => ApiService.loadCohesion(),
    () => ApiService.loadXG(),
    () => ApiService.loadWinProb(),
    () => ApiService.loadShotMap(),
    () => ApiService.loadLeagueXG(),
    () => ApiService.loadMatches(),
    () => ApiService.loadInjury(),
  ].map((fn) => Promise.resolve().then(fn)));

  const [playerRes, cohesionRes, xgRes, winprobRes,
         shotmapRes, leagueRes, matchesRes, injuryRes] = results;

  const labels = ["player", "cohesion", "xg", "winprob",
                  "shotmap", "leaguexg", "matches", "injury"];
  results.forEach((r, i) => {
    if (r.status === "rejected") {
      console.error(`[refreshAllData] ${labels[i]} fetch failed:`, r.reason);
    } else {
      debug(`[refreshAllData] ${labels[i]} ok, source=${r.value?.source}`);
    }
  });

  AppState.player    = playerRes.status   === "fulfilled" ? playerRes.value   : null;
  AppState.cohesion  = cohesionRes.status === "fulfilled" ? cohesionRes.value : null;
  AppState.xg        = xgRes.status       === "fulfilled" ? xgRes.value       : null;
  AppState.winprob   = winprobRes.status  === "fulfilled" ? winprobRes.value  : null;
  AppState.shotmap   = shotmapRes.status  === "fulfilled" ? shotmapRes.value  : null;
  AppState.leaguexg  = leagueRes.status   === "fulfilled" ? leagueRes.value   : null;
  AppState.matches   = matchesRes.status  === "fulfilled" ? matchesRes.value  : null;
  AppState.injury    = injuryRes.status   === "fulfilled" ? injuryRes.value   : null;

  const activePage = document.querySelector(".page.active")?.id?.replace("page-", "") || "dashboard";
  debug("Active page is:", activePage);
  renderPage(activePage);

  // Finishing-leader cards on the dashboard depend on the xG data, which may
  // settle after the dashboard payload. Render them here once both are in.
  if (activePage === "dashboard") {
    renderFinishingLeaders();
  }

  const sources = results
    .filter((r) => r.status === "fulfilled")
    .map((r) => r.value?.source)
    .filter(Boolean);
  // Source labels: every endpoint reports where its data came from. A healthy
  // app combines live SQL ("database*") with trained-model artifacts (e.g. the
  // win-probability classifier reports "artifact"). That combination is the
  // normal state — only flag "demo"/"error" when data is actually missing.
  const hasError     = results.some((r) => r.status === "rejected");
  const anyFallback  = sources.some((s) => s.includes("fallback"));
  const allFallback  = sources.length > 0 && sources.every((s) => s === "fallback");
  const allDbExact   = sources.length > 0 && sources.every((s) => s === "database");

  updateDataSourceBadge(
    hasError      ? "error"             :
    allFallback   ? "fallback"          :
    anyFallback   ? "fallback+artifact" :
    allDbExact    ? "database"          :
    "database+artifact"
  );

  document.querySelectorAll(".kpi-value").forEach((el) => el.classList.remove("shimmer"));
  debug("refreshAllData() complete");
}

// ---------------------------------------------------------------------------
// Navigation hook
// ---------------------------------------------------------------------------

function onPageActivated(pageId) {
  debug("onPageActivated:", pageId);
  if (_pendingRender.has(pageId)) {
    renderPage(pageId);
    // Finishing-leader cards live on the dashboard and need the xG payload.
    // Render them here on a lazy first visit too.
    if (pageId === "dashboard") {
      renderFinishingLeaders();
    }
  }
}

function renderPage(pageId) {
  _pendingRender.delete(pageId);
  switch (pageId) {
    case "dashboard": _safeRender(renderDashboard, "dashboard"); break;
    case "player":    _safeRender(renderPlayer,    "player");    break;
    case "cohesion":  _safeRender(renderCohesion,  "cohesion");  break;
    case "xg":        _safeRender(renderXG,        "xg");        break;
    case "winprob":   _safeRender(renderWinProb,   "winprob");   break;
    case "injury":    _safeRender(renderInjury,    "injury");    break;
    case "eda":       _safeRender(renderEDA,       "eda");       break;
    case "models":    _safeRender(renderModels,    "models");    break;
    default: debug("Unknown pageId:", pageId);
  }
}

function _safeRender(fn, pageId) {
  try {
    fn();
  } catch (err) {
    console.error(`[render:${pageId}]`, err);
    showPageError(pageId, `Render error: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// renderDashboard
// ---------------------------------------------------------------------------

function renderDashboard() {
  // The dashboard summarises the selected team's season from the same real
  // xG aggregation that powers the table below (AppState.leaguexg). It does not
  // depend on /api/dashboard's composite scores any more.
  renderDashboardKPIs();

  // Finishing-leader cards are rendered in refreshAllData()/onPageActivated()
  // once the xG payload has resolved (it may arrive after the dashboard data).

  renderLeagueXG();

  animateCards();
  animateProgressBars();
}

// ---------------------------------------------------------------------------
// renderDashboardKPIs — selected-team season summary (matches, GF/xGF,
// GA/xGA, Pts/xPts). Sourced from /api/league-xg so everything is real.
// ---------------------------------------------------------------------------

function renderDashboardKPIs() {
  const teams = AppState.leaguexg?.teams || [];
  const selected = getSelectedTeamName();
  // Only show the selected team's own row — never another team's as a fallback.
  const t = teams.find((r) => r.team_name === selected);

  const season = AppState.leaguexg?.season;
  _setText("dashSeason", season ? `Season ${season}` : "This season");

  if (!t) {
    ["dashMatches", "dashGF", "dashGA", "dashPts"].forEach((id) => _setText(id, "—"));
    _setText("dashSeason",
      season ? `${selected} has no ${season} league data` : "No season data");
    ["dashXGF", "dashXGA", "dashXPts"].forEach((id) => _setText(id, ""));
    return;
  }

  _setText("dashMatches", fmtInt(t.played));
  _setText("dashGF", fmtInt(t.goals_for));
  _setText("dashGA", fmtInt(t.goals_against));
  _setText("dashPts", fmtInt(t.points));

  _setText("dashXGF", `xG ${Number(t.xg_for).toFixed(1)}`);
  _setText("dashXGA", `xGA ${Number(t.xg_against).toFixed(1)}`);

  const d = Number(t.points_diff || 0);
  _setText("dashXPts", `xPts ${Number(t.xpoints).toFixed(1)} (${d >= 0 ? "+" : ""}${d.toFixed(1)})`);
}

// ---------------------------------------------------------------------------
// renderLeagueXG  (dashboard) — season table, points vs expected points
// ---------------------------------------------------------------------------

function renderLeagueXG() {
  const data  = AppState.leaguexg;
  const tbody = document.querySelector("#leagueXgTable tbody");
  if (!tbody) return;
  const teams = data?.teams || [];

  const titleEl = document.getElementById("leagueXgTitle");
  if (titleEl && data?.season) {
    titleEl.textContent = `Season xG Performance ${data.season} — points vs expected points`;
  }

  if (!teams.length) {
    tbody.innerHTML = '<tr><td colspan="11" class="text-muted text-center">No xG data</td></tr>';
    return;
  }

  const selected = getSelectedTeamName();
  tbody.innerHTML = teams.slice(0, 20).map((t, i) => {
    const d = Number(t.points_diff || 0);
    const color = d >= 0 ? "var(--accent)" : "var(--red)";
    const isSel = t.team_name === selected;
    return `
      <tr${isSel ? ' style="background:rgba(56,189,131,0.07)"' : ""}>
        <td class="text-muted">${i + 1}</td>
        <td class="fw-700">${t.team_name}</td>
        <td>${t.played}</td>
        <td>${t.goals_for}</td>
        <td class="text-muted">${t.xg_for}</td>
        <td>${t.goals_against}</td>
        <td class="text-muted">${t.xg_against}</td>
        <td>${t.xg_diff > 0 ? "+" : ""}${t.xg_diff}</td>
        <td class="fw-700">${t.points}</td>
        <td class="text-muted">${t.xpoints}</td>
        <td style="color:${color};font-weight:600">${d >= 0 ? "+" : ""}${d.toFixed(1)}</td>
      </tr>`;
  }).join("");
}

// ---------------------------------------------------------------------------
// renderPlayer
// ---------------------------------------------------------------------------

function renderPlayer() {
  const data = AppState.player;
  if (!data) {
    showPageError("player", "Player efficiency data unavailable");
    return;
  }
  debug("renderPlayer(), source=", data.source, "players=", data.players?.length);

  const leader  = data.leader  || {};
  const players = data.players || [];

  const tbody = document.querySelector("#page-player table tbody");
  if (tbody) {
    tbody.innerHTML = players.map((p, i) => `
      <tr class="clickable" data-idx="${i}" title="Click to view ${p.player_name}">
        <td class="fw-700">${p.player_name}</td>
        <td>${p.position || "-"}</td>
        <td>${p.player_type || "-"}</td>
        <td>${p.matches || 0}</td>
        <td>${(p.xg_per_90 || 0).toFixed(2)}</td>
        <td>${(p.xa_per_90 || 0).toFixed(2)}</td>
        <td>${(p.pass_completion || 0).toFixed(1)}%</td>
        <td>${(p.key_passes || 0).toFixed(1)}</td>
      </tr>
    `).join("");

    tbody.querySelectorAll("tr.clickable").forEach((tr) => {
      tr.addEventListener("click", () => {
        const idx = Number(tr.dataset.idx);
        if (!Number.isNaN(idx) && players[idx]) _selectPlayer(players[idx], players);
      });
    });
  }

  // Default selection: the leader (top player), or first row.
  _selectPlayer(leader.player_name ? leader : players[0] || {}, players);
}

// Build a radar payload from a single player row, mirroring the server's
// _build_radar() so an arbitrary clicked player renders identically to the
// leader the API pre-computes.
function _playerRadar(p) {
  return {
    labels: ["xG", "xA", "Passing", "Key Passes", "Dribbles", "Shots"],
    values: [
      Math.min(100, +(((p.xg_per_90 || 0) * 100).toFixed(1))),
      Math.min(100, +(((p.xa_per_90 || 0) * 100).toFixed(1))),
      +((p.pass_completion || 0).toFixed(1)),
      Math.min(100, +(((p.key_passes || 0) * 25).toFixed(1))),
      Math.min(100, +(((p.dribbles || 0) * 20).toFixed(1))),
      Math.min(100, +(((p.shots || 0) * 20).toFixed(1))),
    ],
  };
}

// Render the profile card, metric grid, radar and cluster highlight for a
// single player. Reused by the default (leader) render and by table clicks.
function _selectPlayer(player, players) {
  if (!player) return;

  const nameEl   = document.querySelector("#page-player .player-name");
  const metaEl   = document.querySelector("#page-player .player-meta");
  const avatarEl = document.querySelector("#page-player .player-avatar");
  if (nameEl)   nameEl.textContent   = player.player_name || "Top Player";
  if (metaEl)   metaEl.textContent   = `${player.position || "-"} • ${getSelectedTeamName()}`;
  if (avatarEl) avatarEl.textContent = getInitials(player.player_name || "");

  const statsRow = document.querySelector("#page-player .player-stats-row");
  if (statsRow) {
    statsRow.innerHTML = `
      <div class="stat-item">Type <span>${player.player_type || "Midfielder"}</span></div>
      <div class="stat-item">Matches <span>${player.matches || 0}</span></div>
      <div class="stat-item">Minutes <span>${Math.round(player.minutes || 0).toLocaleString()}</span></div>
      <div class="stat-item">xG <span>${(player.xg_per_90 || 0).toFixed(2)}</span></div>
    `;
  }

  const metricsGrid = document.querySelector("#page-player .metrics-grid");
  if (metricsGrid) {
    const avgXG = _teamAvg(players, "xg_per_90");
    const avgXA = _teamAvg(players, "xa_per_90");
    const avgPA = _teamAvg(players, "pass_completion");
    const avgKP = _teamAvg(players, "key_passes");
    const deltaXG = _delta(player.xg_per_90,      avgXG);
    const deltaXA = _delta(player.xa_per_90,       avgXA);
    const deltaPA = _delta(player.pass_completion, avgPA);
    const deltaKP = _delta(player.key_passes,      avgKP);

    metricsGrid.innerHTML = `
      <div class="metric-card">
        <div class="metric-val text-accent">${(player.xg_per_90 || 0).toFixed(2)}</div>
        <div class="metric-label">xG per 90</div>
        <div class="metric-delta ${deltaXG >= 0 ? "pos" : "neg"}">${deltaXG >= 0 ? "+" : ""}${deltaXG.toFixed(0)}% vs avg</div>
      </div>
      <div class="metric-card">
        <div class="metric-val text-cyan">${(player.xa_per_90 || 0).toFixed(2)}</div>
        <div class="metric-label">xA per 90</div>
        <div class="metric-delta ${deltaXA >= 0 ? "pos" : "neg"}">${deltaXA >= 0 ? "+" : ""}${deltaXA.toFixed(0)}% vs avg</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">${(player.pass_completion || 0).toFixed(1)}%</div>
        <div class="metric-label">Pass Completion</div>
        <div class="metric-delta ${deltaPA >= 0 ? "pos" : "neg"}">${deltaPA >= 0 ? "+" : ""}${deltaPA.toFixed(0)}% vs avg</div>
      </div>
      <div class="metric-card">
        <div class="metric-val text-amber">${(player.key_passes || 0).toFixed(1)}</div>
        <div class="metric-label">Key Passes</div>
        <div class="metric-delta ${deltaKP >= 0 ? "pos" : "neg"}">${deltaKP >= 0 ? "+" : ""}${deltaKP.toFixed(0)}% vs avg</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">${(player.dribbles || 0).toFixed(1)}</div>
        <div class="metric-label">Dribbles</div>
      </div>
      <div class="metric-card">
        <div class="metric-val">${(player.shots || 0).toFixed(1)}</div>
        <div class="metric-label">Shots per 90</div>
      </div>
    `;
  }

  requestAnimationFrame(() => {
    Charts.initRadar(_playerRadar(player), player.player_name || "Player");
  });

  const clusterGrid = document.querySelector("#page-player .cluster-grid");
  if (clusterGrid && players && players.length) {
    const groups = {};
    for (const p of players) {
      const t = p.player_type || "Unclassified";
      if (!groups[t]) groups[t] = [];
      groups[t].push(p.player_name);
    }

    // Colour by the broad role implied by the data-driven archetype name.
    // Keyed on substrings so it survives label changes from Model 1.
    const ROLE_COLORS = [
      { match: /defender|defensive/i, c: { border: "var(--cyan)",   badge: "rgba(34,211,238,0.1)", text: "var(--cyan)"   } },
      { match: /midfield|winning/i,   c: { border: "var(--amber)",  badge: "var(--amber-dim)",     text: "var(--amber)"  } },
      { match: /winger|playmaker/i,   c: { border: "var(--accent)", badge: "var(--accent-dim)",    text: "var(--accent)" } },
      { match: /goalscorer|forward/i, c: { border: "var(--red)",    badge: "var(--red-dim)",       text: "var(--red)"    } },
    ];
    const colorFor = (type) =>
      (ROLE_COLORS.find((r) => r.match.test(type)) || ROLE_COLORS[2]).c;

    clusterGrid.innerHTML = Object.entries(groups).slice(0, 4).map(([type, names]) => {
      const c = colorFor(type);
      const isSelected = names.includes(player.player_name);
      return `
        <div class="cluster-card${isSelected ? " selected" : ""}"
             style="border-left:3px solid ${c.border}">
          <div class="cluster-name">${type}
            <span class="cluster-count" style="background:${c.badge};color:${c.text}">${names.length}</span>
          </div>
          <ul class="cluster-players">
            ${names.slice(0, 3).map((n) => `<li>${n}</li>`).join("")}
          </ul>
        </div>
      `;
    }).join("");
  }

  // Highlight the selected row in the table.
  const tbody = document.querySelector("#page-player table tbody");
  if (tbody) {
    tbody.querySelectorAll("tr").forEach((tr) => {
      const idx = Number(tr.dataset.idx);
      const isSel = players[idx] && players[idx].player_name === player.player_name;
      tr.classList.toggle("row-selected", !!isSel);
    });
  }
}

// ---------------------------------------------------------------------------
// renderCohesion
// ---------------------------------------------------------------------------

function renderCohesion() {
  const data = AppState.cohesion;
  if (!data) {
    showPageError("cohesion", "Team cohesion data unavailable");
    return;
  }
  debug("renderCohesion(), source=", data.source, "edges=", data.edges?.length);

  const kpi    = data.kpi || {};
  const values = document.querySelectorAll("#page-cohesion .kpi-value");
  const fmt = (v, dp) => (v === null || v === undefined ? "—" : Number(v).toFixed(dp));
  if (values[0]) values[0].textContent = fmt(kpi.network_density,  2);
  if (values[1]) values[1].textContent = fmt(kpi.avg_degree,       1);
  if (values[2]) values[2].textContent = fmt(kpi.clustering_coeff, 2);
  if (values[3]) values[3].textContent = fmt(kpi.mean_betweenness, 3);
  values.forEach((el) => el.classList.remove("shimmer"));

  requestAnimationFrame(() => {
    PassNetwork.init(
      data.edges && data.edges.length ? data.edges : null,
      data.nodes && data.nodes.length ? data.nodes : null,
    );
  });

  if (data.nodes && data.nodes.length) {
    _renderCohesionCards(data.nodes);
  }
}

// Honest per-player metrics taken straight from the API's node list, which is
// computed over the FULL season pass table (not the capped edge list the graph
// draws), so the numbers match reality:
//  • volume = total passes the player made over the season
//  • degree = number of distinct team-mates they exchange passes with
// No max-normalised "centrality" (which by construction always made the top
// player exactly 1.00); the displayed values are real counts.
function _renderCohesionCards(nodes) {
  const listHTML = (rows) => rows.map(([name, primary, sub]) => `
      <div class="player-list-item">
        <div>
          <div class="player-list-name">${name}</div>
          <div class="player-list-role">${sub}</div>
        </div>
        <div class="player-centrality">${primary}</div>
      </div>`).join("");

  const centralCard = document.querySelector("#page-cohesion [data-card='central']");
  if (centralCard) {
    const titleHTML = centralCard.querySelector(".card-title")?.outerHTML
      || '<div class="card-title">Most Involved Players</div>';
    const rows = [...nodes]
      .sort((a, b) => (b.volume || 0) - (a.volume || 0)).slice(0, 5)
      .map((n) => [n.name,
        `${Math.round(n.volume || 0)} <span class="text-muted fs-11">passes</span>`,
        `${n.degree || 0} team-mates linked`]);
    centralCard.innerHTML = titleHTML + listHTML(rows);
  }

  const connCard = document.querySelector("#page-cohesion [data-card='connected']");
  if (connCard) {
    const titleHTML = connCard.querySelector(".card-title")?.outerHTML
      || '<div class="card-title">Best Connected Players</div>';
    const rows = [...nodes]
      .sort((a, b) => (b.degree || 0) - (a.degree || 0)).slice(0, 5)
      .map((n) => [n.name,
        `${n.degree || 0} <span class="text-muted fs-11">links</span>`,
        `${Math.round(n.volume || 0)} passes`]);
    connCard.innerHTML = titleHTML + listHTML(rows);
  }
}

// ---------------------------------------------------------------------------
// renderXG  (Shot Quality — from-scratch Expected Goals)
// ---------------------------------------------------------------------------

function renderXG() {
  const data = AppState.xg;
  if (!data) {
    showPageError("xg", "xG data unavailable");
    return;
  }
  debug("renderXG(), source=", data.source, "players=", data.players?.length);

  const kpi     = data.kpi || {};
  const players = data.players || [];

  const values = document.querySelectorAll("#page-xg .kpi-value");
  if (values[0]) values[0].textContent = Number(kpi.team_xg || 0).toFixed(1);
  if (values[1]) values[1].textContent = String(kpi.team_goals || 0);
  if (values[2]) {
    const d = Number(kpi.xg_diff || 0);
    values[2].textContent = `${d >= 0 ? "+" : ""}${d.toFixed(1)}`;
  }
  if (values[3]) values[3].textContent = String(kpi.shots || 0);

  const allShots = AppState.shotmap?.shots || [];
  let selectedPlayer = null;   // null = show every shot

  const selEl = document.querySelector("#page-xg #shotMapSel");

  // Redraw the shot map for the current selection and update the caption +
  // selected-row highlight. Clicking the active player again clears the filter.
  const drawShotMap = () => {
    const shots = selectedPlayer
      ? allShots.filter((s) => s.player_name === selectedPlayer)
      : allShots;
    Charts.initShotMap(shots);

    if (selEl) {
      if (selectedPlayer) {
        selEl.innerHTML =
          `${selectedPlayer} · ${shots.length} shot${shots.length === 1 ? "" : "s"} ` +
          `· <a href="#" id="shotMapClear" style="color:var(--accent)">show all</a>`;
        const clr = selEl.querySelector("#shotMapClear");
        if (clr) clr.addEventListener("click", (e) => {
          e.preventDefault();
          selectedPlayer = null;
          drawShotMap();
        });
      } else {
        selEl.textContent = "all players · click a row to filter";
      }
    }

    const body = document.querySelector("#page-xg table tbody");
    if (body) body.querySelectorAll("tr").forEach((tr) => {
      tr.classList.toggle("row-selected", tr.dataset.player === selectedPlayer);
    });
  };

  requestAnimationFrame(() => {
    Charts.initXGChart(players.slice(0, 10));
    drawShotMap();
  });

  const tbody = document.querySelector("#page-xg table tbody");
  if (tbody) {
    if (!players.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="text-muted text-center">No shot data</td></tr>';
    } else {
      tbody.innerHTML = players.map((p) => {
        const d = Number(p.xg_diff || 0);
        const color = d >= 0 ? "var(--accent)" : "var(--red)";
        return `
          <tr class="clickable" data-player="${p.player_name}" title="Click to map ${p.player_name}'s shots">
            <td class="fw-700">${p.player_name}</td>
            <td>${p.position || "-"}</td>
            <td>${p.shots || 0}</td>
            <td>${p.goals || 0}</td>
            <td>${Number(p.xg || 0).toFixed(2)}</td>
            <td style="color:${color};font-weight:600">${d >= 0 ? "+" : ""}${d.toFixed(2)}</td>
          </tr>`;
      }).join("");

      tbody.querySelectorAll("tr.clickable").forEach((tr) => {
        tr.addEventListener("click", () => {
          const name = tr.dataset.player;
          selectedPlayer = selectedPlayer === name ? null : name;
          drawShotMap();
        });
      });
    }
  }
}

// ---------------------------------------------------------------------------
// renderWinProb
// ---------------------------------------------------------------------------

function renderWinProb() {
  const data = AppState.winprob;
  if (!data) {
    showPageError("winprob", "Win probability data unavailable");
    return;
  }
  debug("renderWinProb(), source=", data.source);

  const pct       = document.querySelectorAll("#page-winprob .prob-pct");
  const teams     = document.querySelectorAll("#page-winprob .prob-team");
  const matchLine = document.querySelector("#page-winprob .prob-match");
  const selected  = getSelectedTeamName();

  if (pct[0]) pct[0].textContent   = `${data.headline.win}%`;
  if (pct[1]) pct[1].textContent   = `${data.headline.draw}%`;
  if (pct[2]) pct[2].textContent   = `${data.headline.loss}%`;
  if (teams[0]) teams[0].textContent = "Win";
  if (teams[1]) teams[1].textContent = "Draw";
  if (teams[2]) teams[2].textContent = "Loss";

  // No "next match" — the data is historical. Frame the headline as the model's
  // average pre-match outcome probability for the team, with the actual record.
  if (matchLine) {
    const r = data.record;
    const scope = data.season ? data.season : "all seasons";
    const recordTxt = r && r.played
      ? ` — actual record ${r.wins}W ${r.draws}D ${r.losses}L (${r.played})`
      : "";
    matchLine.textContent =
      `${selected} • average pre-match outcome, ${scope}${recordTxt}`;
  }

  renderWinProbFactors();
  _initMatchSelector();
}

// "What the model looks at" (the real pre-match features) + the model-accuracy
// card. Both degrade gracefully when a source has not resolved yet.
function renderWinProbFactors() {
  // Model inputs — the actual features the classifier uses, averaged for the
  // selected team/season (returned by /api/win-probability).
  const host = document.getElementById("wpInputs");
  if (host) {
    const inputs = AppState.winprob?.inputs || [];
    host.innerHTML = inputs.length
      ? inputs.map((f) => `
          <div class="inline-stat">
            <div class="inline-stat-label">${f.label}</div>
            <div class="inline-stat-val">${Number(f.value).toFixed(2)}</div>
          </div>`).join("")
      : '<div class="text-muted fs-12" style="padding:16px 0;text-align:center">No model inputs for this selection</div>';
  }

  // Accuracy for both sub-models. Prefer the metrics block returned by
  // /api/win-probability (the optimized model that is actually serving
  // predictions); fall back to the registry entry if it's absent.
  const m5 = (AppState.models?.models || []).find(
    (m) => m.model_key === "model5_win_probability"
  );
  const me = AppState.winprob?.model || m5?.metrics;
  if (!me) return;

  const baselineTxt = (naive) =>
    Number.isFinite(Number(naive)) ? `baseline ${fmtPct(naive)}` : "";

  const pre = Number(me.prematch_accuracy);
  if (Number.isFinite(pre)) {
    _setText("wpAccVal", fmtPct(pre));
    _setBar("wpAccBar", pre * 100);
    _setText("wpAccBaseline", baselineTxt(me.prematch_naive));
  }

  const ig = Number(me.ingame_accuracy);
  if (Number.isFinite(ig)) {
    _setText("wpAccCv", fmtPct(ig));
    _setBar("wpAccCvBar", ig * 100);
    _setText("wpAccCvBaseline", baselineTxt(me.ingame_naive));
  }
}

function _setBar(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = `${Math.max(0, Math.min(100, Number(pct) || 0))}%`;
}

// ---------------------------------------------------------------------------
// renderInjury  (Model 3: injury risk)
// ---------------------------------------------------------------------------

function renderInjury() {
  const data = AppState.injury;
  if (!data) {
    showPageError("injury", "Injury risk data unavailable");
    return;
  }
  debug("renderInjury(), source=", data.source, "players=", data.players?.length);

  const kpi    = data.kpi || {};
  const values = document.querySelectorAll("#page-injury .kpi-value");
  if (values[0]) values[0].textContent = String(kpi.high   || 0);
  if (values[1]) values[1].textContent = String(kpi.medium || 0);
  if (values[2]) values[2].textContent = String(kpi.low    || 0);
  if (values[3]) values[3].textContent = (kpi.avg_score || 0).toFixed(2);

  renderInjuryRiskTable();
  requestAnimationFrame(() => {
    Charts.initInjuryHistory(data.players || []);
  });
}

function renderInjuryRiskTable() {
  const tbody   = document.querySelector("#page-injury .data-table tbody");
  const players = AppState.injury?.players;
  if (!tbody) {
    debug("renderInjuryRiskTable: tbody not found");
    return;
  }
  if (!players?.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-muted text-center">No data</td></tr>';
    return;
  }

  tbody.innerHTML = players.map((p) => {
    const score = Number(p.risk_score || 0);
    const pct   = Math.round(score * 100);
    const meta  = getRiskMeta(score);
    const barColor       = score >= 0.67 ? "var(--red)" : score >= 0.4 ? "var(--amber)" : "var(--accent)";
    const recommendation = score >= 0.67 ? "Rest advised" : score >= 0.4 ? "Reduce load" : "Proceed normally";
    return `
      <tr>
        <td class="fw-700">${p.player_name}</td>
        <td>${p.position || "-"}</td>
        <td>
          <div class="risk-score-bar">
            <div class="risk-bar">
              <div class="risk-bar-fill" style="width:${pct}%;background:${barColor}"></div>
            </div>
            <span class="fs-12" style="width:30px">${score.toFixed(2)}</span>
          </div>
        </td>
        <td><span class="badge ${meta.badgeClass}">${meta.label.replace(" Risk", "")}</span></td>
        <td>${Number(p.workload_30d || 0)} min</td>
        <td>${Number(p.days_since_last_injury || 0)} days ago</td>
        <td class="text-muted fs-12">${recommendation}</td>
      </tr>
    `;
  }).join("");
}

// ---------------------------------------------------------------------------
// Match xG timeline (win-probability page) — selector + cumulative-xG chart
// ---------------------------------------------------------------------------

function _initMatchSelector() {
  const sel = document.getElementById("matchSelector");
  if (!sel) return;
  const matches = AppState.matches?.matches || [];

  if (!matches.length) {
    sel.innerHTML = '<option>No matches available</option>';
    return;
  }

  sel.innerHTML = matches
    .map((m) => `<option value="${m.match_id}">${m.label}</option>`)
    .join("");

  sel.onchange = () => _loadMatchTimeline(Number(sel.value));
  _loadMatchTimeline(Number(matches[0].match_id));
}

async function _loadMatchTimeline(matchId) {
  if (!matchId) return;

  // Fetch both timelines first so they can share one x-axis range — the in-game
  // win-probability curve and the cumulative-xG race are stacked for the same
  // match, so a mismatched axis (e.g. xG to 95', win-prob to 90') reads as a bug.
  const [wpRes, tlRes] = await Promise.allSettled([
    ApiService.loadWinProbTimeline(matchId),
    ApiService.loadMatchTimeline(matchId),
  ]);
  const wp = wpRes.status === "fulfilled" ? wpRes.value : null;
  const tl = tlRes.status === "fulfilled" ? tlRes.value : null;

  // Shared x max = the latest minute either chart actually has data for, but at
  // least a full 90' (extra-time matches push this to 120'+).
  const xMax = Math.max(90, _wpMaxMinute(wp), _xgMaxMinute(tl));

  // Dynamic in-game win-probability curve (Model 5B) — the primary chart.
  if (wp) {
    requestAnimationFrame(() => Charts.initWinProbTimeline(wp, xMax));
    const wpSum = document.getElementById("winProbTlSummary");
    if (wpSum && wp.series && wp.series.length) {
      const last = wp.series[wp.series.length - 1];
      const score = (wp.home_score != null)
        ? `${wp.home_name} ${wp.home_score}–${wp.away_score} ${wp.away_name}` : "";
      const preTxt = wp.prematch
        ? `Pre-match, the static model gave <strong>${wp.team_name || getSelectedTeamName()}</strong> a ${wp.prematch.win}% win chance (dashed line). `
        : "";
      wpSum.innerHTML =
        preTxt +
        `The live model then re-reads the match every minute` +
        (score ? ` · final: ${score}` : "") +
        `, ending at <strong>${last.win}% win / ${last.draw}% draw / ${last.loss}% loss</strong>.`;
    } else if (wpSum) {
      wpSum.textContent = "No in-game snapshots available for this match.";
    }
  } else {
    console.error("[_loadMatchTimeline] win-prob", wpRes.reason);
  }

  // Supporting cumulative-xG race for the same match.
  if (tl) {
    requestAnimationFrame(() => Charts.initMatchTimeline(tl, xMax));
    const summary = document.getElementById("matchXgSummary");
    if (summary) {
      summary.innerHTML =
        `<strong>${tl.home_name} ${tl.home_score}–${tl.away_score} ${tl.away_name}</strong> · ` +
        `xG ${Number(tl.home_xg).toFixed(2)}–${Number(tl.away_xg).toFixed(2)}. ` +
        `Diamonds mark actual goals; the steeper line created chances faster.`;
    }
  } else {
    console.error("[_loadMatchTimeline] xg", tlRes.reason);
  }
}

// Latest minute present in each payload, for a shared x-axis.
function _wpMaxMinute(wp) {
  const s = wp?.series;
  return (s && s.length) ? s[s.length - 1].minute : 0;
}
function _xgMaxMinute(tl) {
  if (!tl) return 0;
  const xs = []
    .concat(tl.home_series || [], tl.away_series || [],
            tl.home_goals || [], tl.away_goals || [])
    .map((p) => Number(p.x) || 0);
  return xs.length ? Math.max(...xs) : 0;
}

// ---------------------------------------------------------------------------
// Finishing leaders (dashboard) — Goals vs xG from the xG model
// ---------------------------------------------------------------------------

function renderFinishingLeaders() {
  const grid    = document.querySelector("#page-dashboard .squad-grid");
  const players = AppState.xg?.players;
  if (!grid) {
    debug("renderFinishingLeaders: .squad-grid not found");
    return;
  }
  if (!players?.length) {
    debug("renderFinishingLeaders: no xG data yet");
    return;
  }

  // Highest over-performers (Goals - xG) first.
  const top = [...players]
    .sort((a, b) => Number(b.xg_diff) - Number(a.xg_diff))
    .slice(0, 10);

  grid.innerHTML = top.map((p) => {
    const d = Number(p.xg_diff || 0);
    const cls = d >= 0 ? "risk-ok" : "risk-high";
    const badgeClass = d >= 0 ? "badge-low" : "badge-high";
    return `
      <div class="squad-card ${cls}">
        <div class="squad-name">${p.player_name}</div>
        <div class="squad-pos">${p.goals || 0}G / ${Number(p.xg || 0).toFixed(1)} xG</div>
        <span class="badge ${badgeClass}">${d >= 0 ? "+" : ""}${d.toFixed(1)}</span>
      </div>
    `;
  }).join("");
}

// ===========================================================================
// EDA page — exploratory data analysis over the source tables
// ===========================================================================

const EDA_COLORS = { accent: "#38bd83", cyan: "#06b6d4", amber: "#f59e0b", purple: "#8b5cf6" };

function renderEDA() {
  const data = AppState.eda;
  const root = document.getElementById("eda-content");
  if (!data || data.source === "fallback") {
    if (root) {
      const note = document.getElementById("edaNote");
      if (note) note.textContent =
        "Exploratory data is unavailable — the database is unreachable. Check /api/health.";
    }
    return;
  }
  debug("renderEDA(), source=", data.source);

  const ov = data.overview || {};
  _setText("edaMatches", fmtInt(ov.matches));
  _setText("edaPlayers", fmtInt(ov.players));
  _setText("edaShots",   fmtInt(ov.shots));
  _setText("edaEdges",   fmtInt(ov.pass_network_edges));

  // The distribution visuals are the analysis notebook's own static figures
  // (served from /artifacts/eda/ and embedded directly in eda.html), so there
  // are no live charts to build here — only the headline counts and the shot
  // conversion table below come from /api/eda.

  // Shot conversion by body part.
  const convBody = document.querySelector("#edaConversionTable tbody");
  if (convBody) {
    const rows = data.conversion_by_bodypart || [];
    convBody.innerHTML = rows.length
      ? rows.map((r) => {
          const pct = (Number(r.conversion || 0) * 100).toFixed(1);
          return `<tr><td class="fw-700">${r.body_part}</td>
            <td>${fmtInt(r.shots)}</td><td>${fmtInt(r.goals)}</td>
            <td>${pct}%</td></tr>`;
        }).join("")
      : '<tr><td colspan="4" class="text-muted text-center">No data</td></tr>';
  }
}

// ===========================================================================
// Models & Methodology page — model registry cards
// ===========================================================================

function modelHeadline(m) {
  const me = m.metrics || {};
  switch (m.model_key) {
    case "model_xg":
      return { value: fmtMetric(me.roc_auc),
               label: `ROC-AUC (StatsBomb ${fmtMetric(me.statsbomb_roc_auc)})` };
    case "model3_injury_risk":
      return { value: fmtMetric(me.xgb_roc_auc ?? me.rf_roc_auc), label: "ROC-AUC" };
    case "model5_win_probability":
      return { value: fmtPct(me.ingame_accuracy ?? me.prematch_accuracy),
               label: "In-game accuracy" };
    case "model2_team_cohesion":
      return { value: fmtMetric(me.gbr_r2), label: "GBR R² (grouped CV)" };
    case "model1_player_clustering":
      return { value: fmtMetric(me.spatial?.silhouette), label: "Spatial silhouette" };
    default:
      return { value: "—", label: "" };
  }
}

function flattenMetrics(metrics, prefix = "") {
  const out = [];
  for (const [k, v] of Object.entries(metrics || {})) {
    if (k === "feature_importances") continue;
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) {
      out.push(...flattenMetrics(v, key));
    } else if (typeof v === "number" || typeof v === "string") {
      out.push({ k: key, v });
    }
  }
  return out;
}

function modelCardHTML(m) {
  const h       = modelHeadline(m);
  const metrics = flattenMetrics(m.metrics);
  const figs    = (m.figures || []).slice(0, 6);
  const feats   = m.features || [];

  const metricTiles = metrics.map((row) => `
    <div class="mm-tile">
      <div class="mm-val">${typeof row.v === "number" ? fmtMetric(row.v) : row.v}</div>
      <div class="mm-key">${row.k}</div>
    </div>`).join("");

  const figHTML = figs.length ? `
    <div class="model-figs">
      ${figs.map((src) => `
        <a href="${src}" target="_blank" rel="noopener" title="Open full size">
          <img loading="lazy" src="${src}" alt="diagnostic figure">
        </a>`).join("")}
    </div>` : "";

  return `
    <div class="card model-card">
      <div class="model-card-head">
        <div>
          <div class="model-card-title">${m.display_name || m.model_key}</div>
          <div class="text-muted fs-12">${m.model_key} · v${m.version} · ${m.task || "—"}</div>
        </div>
        <div class="model-card-headline">
          <div class="mch-val">${h.value}</div>
          <div class="mch-lbl">${h.label}</div>
        </div>
      </div>

      <div class="model-card-meta">
        <div><span class="text-muted fs-11">Algorithm</span><div>${m.algorithm || "—"}</div></div>
        <div><span class="text-muted fs-11">Target / objective</span><div>${m.target || "—"}</div></div>
        <div><span class="text-muted fs-11">Training rows</span><div>${fmtInt(m.n_train_rows)}</div></div>
        <div><span class="text-muted fs-11">Trained</span><div>${(m.trained_at || "—").slice(0, 10)}</div></div>
      </div>

      <div class="model-metrics-grid">${metricTiles || '<span class="text-muted fs-12">No metrics recorded</span>'}</div>

      <details class="model-features">
        <summary>Input features (${feats.length})</summary>
        <div class="feature-chips">${feats.map((f) => `<span class="chip">${f}</span>`).join("")}</div>
      </details>

      ${figHTML}
    </div>`;
}

function renderModels() {
  const container = document.getElementById("modelsContainer");
  if (!container) return;
  const data   = AppState.models;
  const models = data?.models || [];

  if (!models.length) {
    container.innerHTML = `
      <div class="card text-muted text-center" style="padding:32px">
        ${data?.note || "No models in the registry yet."}<br>
        <span class="fs-12">Train models with <code>python main.py --train</code> to populate the registry.</span>
      </div>`;
    return;
  }

  container.innerHTML = models.map(modelCardHTML).join("");

  const xg = models.find((m) => m.model_key === "model_xg");
  if (xg && xg.metrics) {
    requestAnimationFrame(() => Charts.initXgBenchmark("xgBenchmarkChart", xg.metrics));
  }
}

// ---------------------------------------------------------------------------
// Per-page "About this model" panels (driven by the registry)
// ---------------------------------------------------------------------------

function renderModelInfoPanels() {
  const models = AppState.models?.models || [];
  if (!models.length) return;
  const byKey = Object.fromEntries(models.map((m) => [m.model_key, m]));

  document.querySelectorAll(".model-info[data-model]").forEach((el) => {
    const m = byKey[el.dataset.model];
    if (!m) return;
    const h = modelHeadline(m);
    el.innerHTML = `
      <div class="mi-left">
        <span class="mi-icon" aria-hidden="true">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
            <rect x="4.5" y="4.5" width="7" height="7" rx="1"/><rect x="6.6" y="6.6" width="2.8" height="2.8" rx="0.5"/>
            <path d="M6.5 2v2M9.5 2v2M6.5 12v2M9.5 12v2M2 6.5h2M2 9.5h2M12 6.5h2M12 9.5h2"/>
          </svg>
        </span>
        <div>
          <div class="mi-name">${m.display_name || m.model_key}
            <span class="badge badge-blue" style="margin-left:6px">${m.task || ""}</span>
          </div>
          <div class="text-muted fs-11">${m.algorithm || ""}</div>
        </div>
      </div>
      <div class="mi-right">
        <div class="mi-metric">${h.value}</div>
        <div class="text-muted fs-11">${h.label}</div>
        <a class="mi-link" onclick="navigateTo('models')">Methodology →</a>
      </div>`;
  });
}

// ---------------------------------------------------------------------------
// Number/format helpers for EDA + Models
// ---------------------------------------------------------------------------

function _setText(id, text) {
  const el = document.getElementById(id);
  if (el) { el.textContent = text; el.classList.remove("shimmer"); }
}

function fmtInt(n) {
  const v = Number(n);
  return Number.isFinite(v) ? v.toLocaleString() : "—";
}

function fmtMetric(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3);
}

function fmtPct(n) {
  const v = Number(n);
  return Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—";
}

// ---------------------------------------------------------------------------
// Error display helpers
// ---------------------------------------------------------------------------

function showGlobalError(message) {
  console.error("[global error]", message);
  const area = document.querySelector(".content-area");
  if (!area) return;
  const existing = area.querySelector(".global-error-banner");
  if (existing) existing.remove();
  const el = document.createElement("div");
  el.className = "global-error-banner";
  el.style.cssText = [
    "background:rgba(239,68,68,0.15)",
    "border:1px solid rgba(239,68,68,0.4)",
    "border-radius:8px",
    "padding:16px 20px",
    "margin-bottom:16px",
    "color:#ef4444",
    "font-size:13px",
    "display:flex",
    "align-items:center",
    "gap:10px",
  ].join(";");
  el.innerHTML = `<span style="display:flex;align-items:center" aria-hidden="true">
      <svg viewBox="0 0 16 16" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2.5 14.5 13.5H1.5z"/><path d="M8 6.5v3.2M8 11.6v0.01"/></svg>
    </span>
    <div>
      <strong>Error:</strong> ${message}<br>
      <span style="color:var(--text-secondary);font-size:11px">
        Check <a href="/api/health" target="_blank" style="color:#ef4444">/api/health</a> for diagnostics.
      </span>
    </div>`;
  area.prepend(el);
}

function showPageError(pageId, message) {
  console.error(`[page:${pageId}]`, message);
  const container = document.querySelector(`#page-${pageId} .kpi-grid`);
  if (!container) return;
  const existing = document.querySelector(`#page-${pageId} .page-error-note`);
  if (existing) existing.remove();
  const el = document.createElement("div");
  el.className = "page-error-note";
  el.style.cssText = [
    "grid-column:1/-1",
    "background:rgba(239,68,68,0.1)",
    "border:1px solid rgba(239,68,68,0.3)",
    "border-radius:6px",
    "padding:10px 14px",
    "font-size:12px",
    "color:#ef4444",
  ].join(";");
  el.innerHTML = `<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><path d="M8 2.5 14.5 13.5H1.5z"/><path d="M8 6.5v3.2M8 11.6v0.01"/></svg>${message} — showing placeholder data.
    <a href="/api/health" target="_blank" style="color:#ef4444;margin-left:8px">Check /api/health</a>`;
  container.prepend(el);
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function getRiskMeta(score) {
  if (score >= 0.67) return { label: "High Risk",  badgeClass: "badge-high",   cardClass: "risk-high" };
  if (score >= 0.4)  return { label: "Monitor",    badgeClass: "badge-medium", cardClass: "risk-med"  };
  return               { label: "Available",  badgeClass: "badge-low",    cardClass: "risk-ok"  };
}

function getInitials(name) {
  if (!name) return "--";
  return name.split(" ").filter(Boolean).slice(0, 2)
    .map((n) => n[0].toUpperCase()).join("");
}

function getSelectedTeamName() {
  const selector = document.getElementById("teamSelector");
  if (!selector) return "Selected Team";
  return selector.options[selector.selectedIndex]?.text || "Selected Team";
}

function updateDataSourceBadge(source) {
  const badge = document.getElementById("dataSourceBadge");
  if (!badge) return;
  const map = {
    database:              { text: "Data: Live DB",      color: "#3ecf8e" },
    "database+artifact":   { text: "Data: DB + Model",   color: "#22d3ee" },
    "database+model3":     { text: "Data: DB + Model 3", color: "#22d3ee" },
    "database+model4":     { text: "Data: DB + Model 4", color: "#22d3ee" },
    "database+model5":     { text: "Data: DB + Model 5", color: "#22d3ee" },
    artifact:              { text: "Data: ML Artifact",  color: "#f59e0b" },
    fallback:              { text: "Data: Demo",         color: "#f59e0b" },
    "fallback+artifact":   { text: "Data: Demo + Model", color: "#f59e0b" },
    error:                 { text: "Data: Error",        color: "#ef4444" },
    unknown:               { text: "Data: Unknown",      color: "#94a3b8" },
  };
  const item = map[source] || map.unknown;
  badge.textContent       = item.text;
  badge.style.borderColor = `${item.color}66`;
  badge.style.color       = item.color;
}

function animateCards() {
  const cards = document.querySelectorAll(
    "#page-dashboard .kpi-card, #page-dashboard .card"
  );
  cards.forEach((el, i) => {
    el.style.opacity   = "0";
    el.style.transform = "translateY(12px)";
    setTimeout(() => {
      el.style.transition = "opacity 0.4s ease, transform 0.4s ease";
      el.style.opacity    = "1";
      el.style.transform  = "translateY(0)";
    }, i * 60);
  });
}

function animateProgressBars() {
  document.querySelectorAll(".progress-fill").forEach((bar) => {
    const target = bar.style.width || "0%";
    bar.style.width = "0%";
    setTimeout(() => { bar.style.width = target; }, 300);
  });
}

function setupTooltips() {
  const tooltip = document.getElementById("tooltip");
  if (!tooltip) return;
  document.addEventListener("mouseover", (e) => {
    const el = e.target.closest("[data-tip]");
    if (!el) { tooltip.style.display = "none"; return; }
    tooltip.textContent   = el.dataset.tip;
    tooltip.style.display = "block";
  });
  document.addEventListener("mousemove", (e) => {
    tooltip.style.left = (e.clientX + 12) + "px";
    tooltip.style.top  = (e.clientY - 8)  + "px";
  });
  document.addEventListener("mouseout", (e) => {
    if (!e.target.closest("[data-tip]")) tooltip.style.display = "none";
  });
}

function _teamAvg(players, key) {
  if (!players.length) return 0;
  return players.reduce((s, p) => s + Number(p[key] || 0), 0) / players.length;
}

function _delta(val, avg) {
  if (!avg) return 0;
  return ((Number(val || 0) - avg) / avg) * 100;
}

function _capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function formatNumber(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

function debounce(fn, delay = 200) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

function debug(...args) {
  if (localStorage.getItem("debug") === "1") {
    console.debug("[soccer-analytics]", ...args);
  }
}

window.addEventListener("resize", debounce(() => {
  if (document.getElementById("page-cohesion")?.classList.contains("active")) {
    PassNetwork.init(AppState.cohesion?.edges || null, AppState.cohesion?.nodes || null);
  }
}, 300));
