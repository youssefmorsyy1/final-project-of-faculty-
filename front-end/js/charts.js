/* =============================================================================
   charts.js
   Chart.js configuration and chart constructors.
   All charts read from the CSS design tokens where possible.
   ============================================================================= */

// Global Chart.js defaults — monospace font for axis labels and data
Chart.defaults.color         = '#8896aa';
Chart.defaults.borderColor   = 'rgba(56, 189, 131, 0.07)';
Chart.defaults.font.family   = "'IBM Plex Mono', monospace";
Chart.defaults.font.size     = 11;

const C = {
  accent: '#38bd83',
  cyan:   '#06b6d4',
  amber:  '#f59e0b',
  red:    '#ef4444',
  purple: '#8b5cf6',
};

const gridColor = 'rgba(56, 189, 131, 0.06)';

function destroyChart(canvas) {
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();
}

// Shared tooltip style applied to every chart
const tooltipDefaults = {
  backgroundColor:  '#111827',
  borderColor:      'rgba(56, 189, 131, 0.3)',
  borderWidth:      1,
  padding:          10,
  cornerRadius:     4,
  titleFont:        { family: "'IBM Plex Mono', monospace", size: 11 },
  bodyFont:         { family: "'IBM Plex Mono', monospace", size: 11 },
  titleColor:       '#8896aa',
  bodyColor:        '#e8edf5',
};

// Shared scale style
function makeScales(opts = {}) {
  return {
    x: {
      grid:   { color: gridColor },
      ticks:  { color: '#4a5568', font: { size: 10 } },
      ...opts.x,
    },
    y: {
      grid:   { color: gridColor },
      ticks:  { color: '#4a5568', font: { size: 10 } },
      ...opts.y,
    },
  };
}

const Charts = {

  // ── Player radar ──────────────────────────────────────────────────────────
  initRadar(payload, playerName = 'Player') {
    const canvas = document.getElementById('radarChart');
    if (!canvas || !payload) return;
    destroyChart(canvas);

    new Chart(canvas.getContext('2d'), {
      type: 'radar',
      data: {
        labels:   payload.labels,
        datasets: [{
          label:                playerName,
          data:                 payload.values,
          borderColor:          C.accent,
          backgroundColor:      'rgba(56,189,131,0.12)',
          pointBackgroundColor: C.accent,
          pointHoverBackgroundColor: C.cyan,
          borderWidth:          2,
          pointRadius:          3,
        }],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        scales: {
          r: {
            min:        0,
            max:        100,
            grid:       { color: 'rgba(56,189,131,0.10)' },
            angleLines: { color: 'rgba(56,189,131,0.08)' },
            ticks: {
              stepSize:       25,
              color:          '#4a5568',
              font:           { size: 9 },
              backdropColor:  'transparent',
            },
            pointLabels: {
              color: '#8896aa',
              font:  { size: 11, family: "'IBM Plex Mono', monospace" },
            },
          },
        },
        plugins: {
          legend:  { display: false },
          tooltip: tooltipDefaults,
        },
      },
    });
  },

  // ── Injury history bar ────────────────────────────────────────────────────
  initInjuryHistory(players) {
    const canvas = document.getElementById('injuryHistChart');
    if (!canvas || !players || !players.length) return;
    destroyChart(canvas);

    const data   = players.slice(0, 12).map(p => Math.round((p.risk_score || 0) * 20));
    const labels = players.slice(0, 12).map(p => p.player_name.split(' ').slice(-1)[0]);
    const colors = data.map(v =>
      v >= 13 ? C.red :
      v >= 8  ? C.amber : C.accent
    );

    new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data:                 data,
          backgroundColor:      colors,
          borderRadius:         3,
          borderSkipped:        false,
          hoverBackgroundColor: C.cyan,
        }],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        plugins: {
          legend:  { display: false },
          tooltip: {
            ...tooltipDefaults,
            callbacks: { label: ctx => ` Risk score: ${(ctx.raw / 20).toFixed(2)}` },
          },
        },
        scales: makeScales({ x: { grid: { display: false } } }),
      },
    });
  },

  // ── Temperature vs performance scatter ────────────────────────────────────
  initTempPerf(points) {
    const canvas = document.getElementById('tempPerfChart');
    if (!canvas || !points || !points.length) return;
    destroyChart(canvas);

    const sorted = [...points].sort((a, b) => a.x - b.x);

    new Chart(canvas.getContext('2d'), {
      data: {
        datasets: [
          {
            type:            'scatter',
            label:           'Matches',
            data:            sorted,
            backgroundColor: C.cyan,
            pointRadius:     5,
            pointHoverRadius:7,
          },
          {
            type:            'line',
            label:           'Trend',
            data:            sorted,
            borderColor:     'rgba(6,182,212,0.4)',
            backgroundColor: 'transparent',
            borderWidth:     1.5,
            pointRadius:     0,
            tension:         0.4,
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        plugins: {
          legend:  { display: false },
          tooltip: {
            ...tooltipDefaults,
            callbacks: {
              label: ctx => ` ${ctx.dataset.label === 'Matches' ? `Temp: ${ctx.parsed.x}°C  Score: ${ctx.parsed.y}` : ''}`,
            },
          },
        },
        scales: makeScales({
          x: {
            type:  'linear',
            title: { display: true, text: 'Temperature (°C)', color: '#4a5568', font: { size: 10 } },
          },
          y: {
            min:   60,
            max:   100,
            title: { display: true, text: 'Performance', color: '#4a5568', font: { size: 10 } },
          },
        }),
      },
    });
  },

  // ── Goals vs xG (grouped bars) ────────────────────────────────────────────
  initXGChart(players) {
    const canvas = document.getElementById('xgChart');
    if (!canvas || !players || !players.length) return;
    destroyChart(canvas);

    const labels = players.map(p => p.player_name.split(' ').slice(-1)[0]);
    new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label:           'Goals',
            data:            players.map(p => p.goals || 0),
            backgroundColor: C.accent,
            borderRadius:    3,
            borderSkipped:   false,
          },
          {
            label:           'xG',
            data:            players.map(p => Number(p.xg || 0)),
            backgroundColor: C.cyan,
            borderRadius:    3,
            borderSkipped:   false,
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true,
              color:         '#8896aa',
              font:          { size: 11, family: "'IBM Plex Mono', monospace" },
              padding:       16,
            },
          },
          tooltip: {
            ...tooltipDefaults,
            callbacks: { label: ctx => ` ${ctx.dataset.label}: ${Number(ctx.raw).toFixed(2)}` },
          },
        },
        scales: makeScales({ x: { grid: { display: false } } }),
      },
    });
  },

  // ── Shot map (scatter on an attacking-third pitch, goal at TOP) ────────────
  // StatsBomb coords: x 0-120 (goal at 120), y 0-80. We render vertically so
  // the pitch keeps its true proportions inside the card: the horizontal axis
  // is the pitch width (SB y, 0-80) and the vertical axis is the pitch length
  // (SB x, 60-120) with the goal at the top. The container enforces a fixed
  // aspect ratio (see .shot-map-wrap), so the pitch is never stretched.
  // Point radius scales with xG; colour marks goal vs miss.
  initShotMap(shots) {
    const canvas = document.getElementById('shotMapChart');
    if (!canvas) return;
    destroyChart(canvas);
    if (!shots || !shots.length) return;

    const goals  = shots.filter(s => s.is_goal);
    const misses = shots.filter(s => !s.is_goal);
    const r = s => 3 + Math.sqrt(Math.max(0, s.xg)) * 15;

    // Axis ranges (with a small margin) — kept in sync with the container's
    // aspect-ratio so the drawn pitch is proportional. width 84, length 64.
    const X_MIN = -2, X_MAX = 82;   // pitch width  (SB y)
    const Y_MIN = 58, Y_MAX = 122;  // pitch length (SB x), goal near top

    // Plugin: draw pitch markings (penalty box, 6-yard box, goal, spot).
    // X maps SB y → horizontal pixels; Y maps SB x → vertical pixels.
    const pitch = {
      id: 'pitch',
      beforeDatasetsDraw(chart) {
        const { ctx, chartArea: a, scales } = chart;
        const X = v => scales.x.getPixelForValue(v);
        const Y = v => scales.y.getPixelForValue(v);
        ctx.save();
        ctx.strokeStyle = 'rgba(56,189,131,0.28)';
        ctx.lineWidth = 1;
        // outer (attacking third shown)
        ctx.strokeRect(a.left, a.top, a.right - a.left, a.bottom - a.top);
        // penalty box: SB x 102-120, y 18-62
        ctx.strokeRect(X(18), Y(120), X(62) - X(18), Y(102) - Y(120));
        // six-yard box: SB x 114-120, y 30-50
        ctx.strokeRect(X(30), Y(120), X(50) - X(30), Y(114) - Y(120));
        // penalty arc (the "D")
        ctx.beginPath();
        ctx.arc(X(40), Y(108), Math.abs(Y(108) - Y(98)), Math.PI * 0.18, Math.PI * 0.82);
        ctx.stroke();
        // goal: SB x 120, y 36-44
        ctx.strokeStyle = 'rgba(56,189,131,0.7)';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(X(36), Y(120)); ctx.lineTo(X(44), Y(120)); ctx.stroke();
        // penalty spot
        ctx.fillStyle = 'rgba(56,189,131,0.45)';
        ctx.beginPath(); ctx.arc(X(40), Y(108), 2, 0, 2 * Math.PI); ctx.fill();
        ctx.restore();
      },
    };

    new Chart(canvas.getContext('2d'), {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Goals',
            data: goals.map(s => ({ x: s.y, y: s.x, _s: s })),
            backgroundColor: 'rgba(56,189,131,0.78)',
            borderColor: C.accent,
            borderWidth: 1,
            pointRadius: ctx => r(ctx.raw._s),
            pointHoverRadius: ctx => r(ctx.raw._s) + 2,
          },
          {
            label: 'Shots (no goal)',
            data: misses.map(s => ({ x: s.y, y: s.x, _s: s })),
            backgroundColor: 'rgba(136,150,170,0.28)',
            borderColor: 'rgba(136,150,170,0.55)',
            borderWidth: 1,
            pointRadius: ctx => r(ctx.raw._s),
            pointHoverRadius: ctx => r(ctx.raw._s) + 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: 0 },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...tooltipDefaults,
            callbacks: {
              title: () => '',
              label: ctx => {
                const s = ctx.raw._s;
                return ` ${s.player_name} — xG ${Number(s.xg).toFixed(2)}` +
                       `${s.is_goal ? ' (GOAL)' : ''} • ${s.body_part} • ${s.minute}'`;
              },
            },
          },
        },
        scales: {
          x: { min: X_MIN, max: X_MAX, display: false, grid: { display: false } },
          y: { min: Y_MIN, max: Y_MAX, display: false, grid: { display: false } },
        },
      },
      plugins: [pitch],
    });
  },

  // ── Match xG timeline (cumulative xG race) ────────────────────────────────
  initMatchTimeline(payload, xMax = 95) {
    const canvas = document.getElementById('matchTimelineChart');
    if (!canvas || !payload) return;
    destroyChart(canvas);

    // Step lines: prepend (0,0) so each team starts at zero.
    const series = pts => [{ x: 0, y: 0 }, ...pts.map(p => ({ x: p.x, y: p.y }))];
    const goalPts = (arr, color) => ({
      type: 'scatter',
      label: '_goals',
      data: arr.map(g => ({ x: g.x, y: g.y, _g: g })),
      backgroundColor: color,
      borderColor: '#0a0e1a',
      borderWidth: 2,
      pointRadius: 7,
      pointHoverRadius: 9,
      pointStyle: 'rectRot',
      showLine: false,
    });

    new Chart(canvas.getContext('2d'), {
      data: {
        datasets: [
          {
            type: 'line', label: payload.home_name,
            data: series(payload.home_series),
            borderColor: C.accent, backgroundColor: 'rgba(56,189,131,0.06)',
            borderWidth: 2, stepped: true, fill: true, pointRadius: 0, tension: 0,
          },
          {
            type: 'line', label: payload.away_name,
            data: series(payload.away_series),
            borderColor: C.cyan, backgroundColor: 'rgba(6,182,212,0.05)',
            borderWidth: 2, stepped: true, fill: true, pointRadius: 0, tension: 0,
          },
          goalPts(payload.home_goals, C.accent),
          goalPts(payload.away_goals, C.cyan),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: true },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true, color: '#8896aa',
              font: { size: 11, family: "'IBM Plex Mono', monospace" }, padding: 16,
              filter: item => item.text !== '_goals',
            },
          },
          tooltip: {
            ...tooltipDefaults,
            callbacks: {
              label: ctx => ctx.raw._g
                ? ` Goal — ${ctx.raw._g.player} (${ctx.raw._g.x}')`
                : ` ${ctx.dataset.label}: ${Number(ctx.raw.y).toFixed(2)} xG`,
            },
          },
        },
        scales: makeScales({
          x: { type: 'linear', min: 0, max: xMax,
               title: { display: true, text: 'Minute', color: '#4a5568', font: { size: 10 } } },
          y: { min: 0,
               title: { display: true, text: 'Cumulative xG', color: '#4a5568', font: { size: 10 } } },
        }),
      },
    });
  },

  // ── In-game win probability (Model 5B, dynamic) ──────────────────────────
  // Three independent lines (win / draw / loss), each its own 0–100% probability
  // re-estimated every minute. The win line is filled so the team's live chances
  // read at a glance; a dashed marker shows the static pre-match call at kickoff.
  initWinProbTimeline(payload, xMax = 90) {
    const canvas = document.getElementById('winProbTimelineChart');
    if (!canvas || !payload) return;
    destroyChart(canvas);

    const series = payload.series || [];
    if (!series.length) return;
    const pre  = payload.prematch;  // {win,draw,loss} or null

    // Linear x ({x:minute,y:value}) so this chart shares an identical 0..xMax
    // axis with the cumulative-xG chart stacked beneath it.
    const line = (key, label, color, fill) => ({
      label,
      data: series.map(p => ({ x: p.minute, y: p[key] })),
      borderColor: color,
      backgroundColor: fill || 'transparent',
      borderWidth: key === 'win' ? 2.5 : 1.6,
      fill: !!fill,
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.3,
    });

    const datasets = [
      line('win',  'Win',  C.accent, 'rgba(56,189,131,0.16)'),
      line('draw', 'Draw', '#8896aa', null),
      line('loss', 'Loss', C.red,    null),
    ];

    // Dashed pre-match reference for the win probability (flat across the match).
    if (pre) {
      datasets.push({
        label: 'Pre-match win',
        data: [{ x: 0, y: pre.win }, { x: xMax, y: pre.win }],
        borderColor: 'rgba(56,189,131,0.55)',
        borderWidth: 1.2,
        borderDash: [5, 4],
        fill: false,
        pointRadius: 0,
        pointHoverRadius: 0,
      });
    }

    new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true, color: '#8896aa',
              font: { size: 11, family: "'IBM Plex Mono', monospace" }, padding: 16,
            },
          },
          tooltip: {
            ...tooltipDefaults,
            callbacks: {
              title: items => `Minute ${Math.round(items[0].parsed.x)}`,
              label: ctx => ` ${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(0)}%`,
            },
          },
        },
        scales: makeScales({
          x: { type: 'linear', min: 0, max: xMax,
               title: { display: true, text: 'Minute', color: '#4a5568', font: { size: 10 } },
               grid: { display: false } },
          y: { min: 0, max: 100,
               ticks: { color: '#4a5568', font: { size: 10 }, callback: v => `${v}%` },
               title: { display: true, text: 'Win probability', color: '#4a5568', font: { size: 10 } } },
        }),
      },
    });
  },

  // ── Generic histogram / bar (EDA distributions) ───────────────────────────
  initSimpleBar(canvasId, labels, values, color = C.cyan, opts = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    destroyChart(canvas);
    if (!labels || !labels.length) return;

    new Chart(canvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data:                 values,
          backgroundColor:      color,
          borderRadius:         3,
          borderSkipped:        false,
          hoverBackgroundColor: C.accent,
        }],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        plugins: {
          legend:  { display: false },
          tooltip: {
            ...tooltipDefaults,
            callbacks: { label: ctx => ` ${Number(ctx.raw).toLocaleString()}` },
          },
        },
        scales: makeScales({
          x: {
            grid:  { display: false },
            title: opts.xTitle
              ? { display: true, text: opts.xTitle, color: '#4a5568', font: { size: 10 } }
              : undefined,
          },
          y: {
            beginAtZero: true,
            title: opts.yTitle
              ? { display: true, text: opts.yTitle, color: '#4a5568', font: { size: 10 } }
              : undefined,
          },
        }),
      },
    });
  },

  // ── Doughnut (categorical counts, e.g. players by position) ────────────────
  initDoughnut(canvasId, labels, values) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    destroyChart(canvas);
    if (!labels || !labels.length) return;

    const palette = [
      C.accent, C.cyan, C.amber, C.purple, C.red,
      '#3ecf8e', '#22d3ee', '#fbbf24', '#a78bfa', '#f87171',
      '#10b981', '#60a5fa', '#f59e0b', '#c084fc', '#fb7185',
    ];

    new Chart(canvas.getContext('2d'), {
      type: 'doughnut',
      data: {
        labels,
        datasets: [{
          data:            values,
          backgroundColor: labels.map((_, i) => palette[i % palette.length]),
          borderColor:     '#0a0e1a',
          borderWidth:     2,
        }],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        cutout:              '58%',
        plugins: {
          legend: {
            position: 'right',
            labels: {
              usePointStyle: true, color: '#8896aa',
              font: { size: 10, family: "'IBM Plex Mono', monospace" }, padding: 10,
              boxWidth: 8,
            },
          },
          tooltip: {
            ...tooltipDefaults,
            callbacks: { label: ctx => ` ${ctx.label}: ${Number(ctx.raw).toLocaleString()}` },
          },
        },
      },
    });
  },

  // ── xG benchmark vs StatsBomb (dual-axis: ROC-AUC + log-loss) ──────────────
  initXgBenchmark(canvasId, metrics) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !metrics) return;
    destroyChart(canvas);

    const labels = ['Our xG', 'StatsBomb', 'Naive baseline'];
    const auc    = [metrics.roc_auc, metrics.statsbomb_roc_auc, 0.5];
    const logloss = [metrics.log_loss, metrics.statsbomb_log_loss, metrics.naive_log_loss];

    new Chart(canvas.getContext('2d'), {
      data: {
        labels,
        datasets: [
          {
            type: 'bar', label: 'ROC-AUC (higher is better)',
            data: auc, yAxisID: 'y',
            backgroundColor: C.accent, borderRadius: 3, borderSkipped: false,
          },
          {
            type: 'bar', label: 'Log-loss (lower is better)',
            data: logloss, yAxisID: 'y1',
            backgroundColor: C.amber, borderRadius: 3, borderSkipped: false,
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true, color: '#8896aa',
              font: { size: 11, family: "'IBM Plex Mono', monospace" }, padding: 16,
            },
          },
          tooltip: {
            ...tooltipDefaults,
            callbacks: { label: ctx => ` ${ctx.dataset.label}: ${Number(ctx.raw).toFixed(3)}` },
          },
        },
        scales: {
          x:  { grid: { display: false }, ticks: { color: '#8896aa', font: { size: 11 } } },
          y:  {
            position: 'left', min: 0, max: 1, grid: { color: gridColor },
            ticks: { color: '#4a5568', font: { size: 10 } },
            title: { display: true, text: 'ROC-AUC', color: '#4a5568', font: { size: 10 } },
          },
          y1: {
            position: 'right', min: 0, grid: { display: false },
            ticks: { color: '#4a5568', font: { size: 10 } },
            title: { display: true, text: 'Log-loss', color: '#4a5568', font: { size: 10 } },
          },
        },
      },
    });
  },
};