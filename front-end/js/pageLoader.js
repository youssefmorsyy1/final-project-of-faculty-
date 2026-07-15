/* ============================================================
   pageLoader.js  — v2.3.0

   KEY FIX: browsers do not execute <script> tags injected via
   innerHTML. This is by spec (HTML5 §8.4). After writing the
   HTML we must clone each <script> into a newly created element
   and append it to the document so the browser treats it as a
   real script and executes it.
   ============================================================ */

const ASSET_VERSION = "2.14.0";

const PAGES = [
  { id: "dashboard-content", file: "/static/pages/dashboard.html" },
  { id: "player-content",    file: "/static/pages/player.html"    },
  { id: "cohesion-content",  file: "/static/pages/cohesion.html"  },
  { id: "xg-content",        file: "/static/pages/xg.html"        },
  { id: "injury-content",    file: "/static/pages/injury.html"    },
  { id: "winprob-content",   file: "/static/pages/winprob.html"   },
  { id: "eda-content",       file: "/static/pages/eda.html"       },
  { id: "models-content",    file: "/static/pages/models.html"    },
  { id: "debug-content",     file: "/static/pages/debug.html"     },
].map((p) => ({ ...p, file: `${p.file}?v=${ASSET_VERSION}` }));

function runScripts(container) {
  // Find every <script> tag inside the injected HTML and re-execute it
  // by creating a fresh <script> element (innerHTML scripts are inert).
  const scripts = container.querySelectorAll("script");
  scripts.forEach((oldScript) => {
    const newScript = document.createElement("script");
    // Copy all attributes (type, src, etc.)
    Array.from(oldScript.attributes).forEach((attr) => {
      newScript.setAttribute(attr.name, attr.value);
    });
    newScript.textContent = oldScript.textContent;
    oldScript.parentNode.replaceChild(newScript, oldScript);
  });
}

async function loadPage({ id, file }) {
  try {
    const res = await fetch(file);
    if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${file}`);
    const html = await res.text();
    const container = document.getElementById(id);
    if (!container) {
      console.warn(`[pageLoader] Container #${id} not found in DOM`);
      return;
    }
    container.innerHTML = html;
    runScripts(container);   // <-- execute the now-inert script tags
  } catch (err) {
    console.error(`[pageLoader] Failed to load ${file}:`, err.message);
    const container = document.getElementById(id);
    if (container) {
      container.innerHTML = `
        <div style="
          padding:20px; margin:20px 0;
          background:rgba(239,68,68,0.1);
          border:1px solid rgba(239,68,68,0.3);
          border-radius:8px; color:#ef4444; font-size:13px;">
          Could not load page template: <strong>${file}</strong><br>
          <span style="color:var(--text-secondary);font-size:11px">${err.message}</span>
        </div>`;
    }
  }
}

async function loadAllPages() {
  await Promise.all(PAGES.map(loadPage));
}

window.pagesLoadedPromise = loadAllPages();