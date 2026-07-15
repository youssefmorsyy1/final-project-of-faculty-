/* ============================================================
   passNetwork.js
   Draws the pass network as an SVG force-directed graph.

   Two modes:
   1. Real mode  — receives the edges array from /api/team-cohesion
                   and builds a dynamic node layout from it.
   2. Demo mode  — falls back to a fixed 4-3-3 formation when no
                   edges are provided.
   ============================================================ */

const PassNetwork = {

  /* ── Demo formation (fallback only) ── */
  _demoPlayers: [
    { id: "GK",  label: "GK",  x: 0.10, y: 0.50, size: 13, highlight: false },
    { id: "RB",  label: "RB",  x: 0.27, y: 0.12, size: 15, highlight: false },
    { id: "CB1", label: "CB",  x: 0.27, y: 0.35, size: 17, highlight: false },
    { id: "CB2", label: "CB",  x: 0.27, y: 0.65, size: 17, highlight: false },
    { id: "LB",  label: "LB",  x: 0.27, y: 0.88, size: 15, highlight: false },
    { id: "CDM", label: "MF",  x: 0.44, y: 0.50, size: 24, highlight: true  },
    { id: "CM1", label: "MF",  x: 0.57, y: 0.27, size: 22, highlight: false },
    { id: "CM2", label: "MF",  x: 0.57, y: 0.73, size: 19, highlight: false },
    { id: "RW",  label: "FW",  x: 0.75, y: 0.14, size: 16, highlight: false },
    { id: "ST",  label: "FW",  x: 0.78, y: 0.50, size: 18, highlight: false },
    { id: "LW",  label: "FW",  x: 0.75, y: 0.86, size: 16, highlight: false },
  ],

  _demoConnections: [
    ["GK",  "CB1", 8], ["GK",  "CB2", 7], ["GK",  "RB",  4],
    ["RB",  "CB1", 6], ["CB1", "CB2", 9], ["CB2", "LB",  6],
    ["CB1", "CDM", 8], ["CB2", "CDM", 8], ["RB",  "CM1", 7], ["LB",  "CM2", 6],
    ["CDM", "CM1", 9], ["CDM", "CM2", 8], ["CDM", "ST",  6],
    ["CM1", "RW",  8], ["CM1", "ST",  7], ["CM2", "LW",  8], ["CM2", "ST",  6],
    ["RW",  "ST",  7], ["LW",  "ST",  5],
  ],

  /* ──────────────────────────────────────────────────────────
     init(edges)
     Public entry point.
     edges: array of {from: string, to: string, weight: number}
            from the /api/team-cohesion response — or null/empty
            for the demo layout.
  ────────────────────────────────────────────────────────── */
  init(edges, nodes) {
    const svg = document.getElementById("passNetworkSvg");
    if (!svg) return;

    requestAnimationFrame(() => {
      const W = svg.clientWidth  || 800;
      const H = svg.clientHeight || 320;

      if (nodes && nodes.length > 0 && edges && edges.length > 0) {
        this._renderOnPitch(svg, W, H, nodes, edges);
      } else if (edges && edges.length > 0) {
        this._renderFromEdges(svg, W, H, edges);
      } else {
        this._renderDemo(svg, W, H);
      }
    });
  },

  /* ──────────────────────────────────────────────────────────
     _renderOnPitch
     Place each node at its real average pitch position (StatsBomb
     units: x 0-120 toward goal, y 0-80). The team attacks left->right.
  ────────────────────────────────────────────────────────── */
  _renderOnPitch(svg, W, H, apiNodes, edges) {
    const maxVol = Math.max(...apiNodes.map(n => n.volume || 1), 1);
    const nodes = {};
    for (const n of apiNodes) {
      nodes[n.name] = {
        x:         n.x / 120,                  // 0..1 across the pitch
        y:         n.y / 80,                    // 0..1 down the pitch
        size:      8 + (n.volume / maxVol) * 14,
        highlight: n.volume === maxVol,
        label:     _initials(n.name),
        full:      n.name,
        // True season totals from the DB. The drawn edge list is capped, so a
        // node's real volume/degree must come from here, not from its edges.
        volume:    n.volume,
        degree:    n.degree,
      };
    }
    const maxW = Math.max(...edges.map(e => e.weight || 1), 1);
    this._render(svg, W, H, nodes, edges, maxW, /* pitch */ true);
  },

  /* ──────────────────────────────────────────────────────────
     _renderFromEdges
     Build nodes dynamically from the passer/receiver names
     in the edge list, then place them using a simple radial
     layout sorted by total pass volume.
  ────────────────────────────────────────────────────────── */
  _renderFromEdges(svg, W, H, edges) {
    // 1. Collect unique player names and their total pass volume
    const volume = {};   // name -> total passes (as passer or receiver)
    for (const e of edges) {
      volume[e.from] = (volume[e.from] || 0) + (e.weight || 1);
      volume[e.to]   = (volume[e.to]   || 0) + (e.weight || 1);
    }

    // 2. Sort by volume descending; top player gets the central node
    const names = Object.keys(volume).sort((a, b) => volume[b] - volume[a]);

    // 3. Assign positions: top player centre, rest in a circle
    const CX = 0.45, CY = 0.50;
    const RADIUS = 0.36;
    const nodes = {};
    names.forEach((name, i) => {
      if (i === 0) {
        nodes[name] = { x: CX, y: CY, size: 22, highlight: true, label: _initials(name) };
      } else {
        const angle = ((i - 1) / (names.length - 1)) * 2 * Math.PI;
        nodes[name] = {
          x:         CX + RADIUS * Math.cos(angle),
          y:         CY + RADIUS * Math.sin(angle),
          size:      Math.max(10, Math.min(18, 10 + (volume[name] / Math.max(...Object.values(volume))) * 8)),
          highlight: false,
          label:     _initials(name),
        };
      }
    });

    // 4. Determine max weight for opacity/thickness scaling
    const maxW = Math.max(...edges.map(e => e.weight || 1), 1);

    this._render(svg, W, H, nodes, edges, maxW);
  },

  /* ──────────────────────────────────────────────────────────
     _renderDemo
     Fixed formation layout used when there are no real edges.
  ────────────────────────────────────────────────────────── */
  _renderDemo(svg, W, H) {
    const nodes = {};
    for (const p of this._demoPlayers) {
      nodes[p.id] = { x: p.x, y: p.y, size: p.size, highlight: p.highlight, label: p.label };
    }
    const maxW = 10;
    const edges = this._demoConnections.map(([from, to, weight]) => ({ from, to, weight }));
    this._render(svg, W, H, nodes, edges, maxW);
  },

  /* ──────────────────────────────────────────────────────────
     _render(svg, W, H, nodes, edges, maxW)
     Core SVG renderer — shared by both modes.

     nodes: {id -> {x, y, size, highlight, label}}
     edges: [{from, to, weight}]
  ────────────────────────────────────────────────────────── */
  _render(svg, W, H, nodes, edges, maxW, pitch = false) {
    // Convert relative (0-1) coords to pixels. On a pitch we inset by a margin
    // so nodes near the touchlines aren't clipped.
    const M = pitch ? 22 : 0;
    const px = (n) => ({ x: M + n.x * (W - 2 * M), y: M + n.y * (H - 2 * M) });

    // Per-player passing volume (sum of edge weights touching the node) and
    // degree (distinct team-mates linked), keyed by node id (= full name in the
    // real modes). Used by the hover tooltip.
    const vol = {}, links = {};
    for (const e of edges) {
      const w = e.weight || 1;
      vol[e.from] = (vol[e.from] || 0) + w;
      vol[e.to]   = (vol[e.to]   || 0) + w;
      (links[e.from] = links[e.from] || new Set()).add(e.to);
      (links[e.to]   = links[e.to]   || new Set()).add(e.from);
    }
    const esc = (s) => String(s).replace(/"/g, "&quot;").replace(/</g, "&lt;");

    let html = `
      <defs>
        <filter id="nodeGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="nodeGlowStrong" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="7" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
    `;

    if (pitch) {
      // Pitch markings (team attacks left -> right). Coords map StatsBomb
      // 120x80 onto the inset drawing area.
      const fx = (x) => M + (x / 120) * (W - 2 * M);
      const fy = (y) => M + (y / 80)  * (H - 2 * M);
      const ps = 'stroke="rgba(56,189,131,0.22)" stroke-width="1" fill="none"';
      html += `
        <rect x="${fx(0)}" y="${fy(0)}" width="${fx(120) - fx(0)}" height="${fy(80) - fy(0)}" ${ps}/>
        <line x1="${fx(60)}" y1="${fy(0)}" x2="${fx(60)}" y2="${fy(80)}" ${ps}/>
        <circle cx="${fx(60)}" cy="${fy(40)}" r="${(fy(50) - fy(40)).toFixed(1)}" ${ps}/>
        <rect x="${fx(102)}" y="${fy(18)}" width="${fx(120) - fx(102)}" height="${fy(62) - fy(18)}" ${ps}/>
        <rect x="${fx(0)}"   y="${fy(18)}" width="${fx(18) - fx(0)}"   height="${fy(62) - fy(18)}" ${ps}/>
      `;
    } else {
      html += `
        <line x1="${W * 0.5}" y1="15" x2="${W * 0.5}" y2="${H - 15}"
              stroke="rgba(99,225,180,0.08)" stroke-width="1" stroke-dasharray="6,4"/>
      `;
    }

    // Edges
    for (const e of edges) {
      const a = nodes[e.from], b = nodes[e.to];
      if (!a || !b) continue;
      const pa = px(a), pb = px(b);
      const strength = Math.max(1, Math.min(maxW, e.weight || 1));
      const opacity  = 0.12 + (strength / maxW) * 0.45;
      const width    = 1.2  + (strength / maxW) * 3.5;
      html += `<line x1="${pa.x.toFixed(1)}" y1="${pa.y.toFixed(1)}"
                     x2="${pb.x.toFixed(1)}" y2="${pb.y.toFixed(1)}"
                     stroke="#3ecf8e"
                     stroke-width="${width.toFixed(1)}"
                     stroke-opacity="${opacity.toFixed(2)}"
                     stroke-linecap="round"/>`;
    }

    // Nodes
    for (const [id, n] of Object.entries(nodes)) {
      const { x, y } = px(n);
      const fill   = n.highlight ? "#3ecf8e" : "#22d3ee";
      const filter = n.highlight ? "url(#nodeGlowStrong)" : "url(#nodeGlow)";
      const opacity = n.highlight ? 1 : 0.85;
      // Label font scales with node radius so initials stay inside the marker.
      const fs = Math.max(8, Math.min(12, n.size * 0.72));
      // Hover data: full name + real passing volume / degree for this node.
      // Prefer the node's true season volume (from the API) over the edge-summed
      // value, since the edge list is capped and undercounts top passers.
      const full = esc(n.full || id);
      const v    = Math.round(n.volume != null ? n.volume : (vol[id] || 0));
      const d    = n.degree != null ? n.degree : ((links[id]?.size) || 0);
      const hover = `data-name="${full}" data-vol="${v}" data-deg="${d}" `
        + `onmousemove="PassNetwork._tip(event)" onmouseleave="PassNetwork._hideTip()"`;
      html += `
        <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${n.size}"
                fill="${fill}" fill-opacity="${opacity}" filter="${filter}"
                style="cursor:pointer" ${hover}/>
        <text x="${x.toFixed(1)}" y="${(y + fs * 0.35).toFixed(1)}"
              text-anchor="middle" fill="#0a0e1a"
              font-size="${fs.toFixed(1)}" font-weight="700"
              font-family="DM Sans, sans-serif"
              pointer-events="none">${n.label || id}</text>
      `;
    }

    svg.innerHTML = html;
  },

  /* ── Hover tooltip ───────────────────────────────────────────────────────
     A lightweight HTML tooltip anchored to the network container, showing the
     player's name, total passing volume and number of distinct links. */
  _tip(ev) {
    const cont = document.querySelector(".pass-network-container");
    if (!cont) return;
    let tip = cont.querySelector(".pn-tooltip");
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "pn-tooltip";
      cont.appendChild(tip);
    }
    const c = ev.target;
    tip.innerHTML =
      `<div class="pn-tip-name">${c.getAttribute("data-name")}</div>` +
      `<div class="pn-tip-row">${c.getAttribute("data-vol")} passes</div>` +
      `<div class="pn-tip-row">${c.getAttribute("data-deg")} team-mates linked</div>`;
    tip.style.display = "block";
    const r = cont.getBoundingClientRect();
    const x = ev.clientX - r.left + 14;
    const y = ev.clientY - r.top + 14;
    tip.style.left = Math.min(x, cont.clientWidth  - tip.offsetWidth  - 6) + "px";
    tip.style.top  = Math.min(y, cont.clientHeight - tip.offsetHeight - 6) + "px";
  },

  _hideTip() {
    const tip = document.querySelector(".pass-network-container .pn-tooltip");
    if (tip) tip.style.display = "none";
  },
};

/* ── Helper: abbreviate a full name to 2-3 uppercase initials ── */
function _initials(name) {
  const parts = (name || "").split(" ").filter(Boolean);
  if (parts.length === 1) return parts[0].slice(0, 3).toUpperCase();
  return parts.slice(0, 2).map(p => p[0].toUpperCase()).join("");
}