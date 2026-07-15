# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A grad/final-project data pipeline + ML system on StatsBomb open data and Transfermarkt injury records.
It ingests match events, computes per-player stats, links injury history, and trains five models
(player clustering, team cohesion, injury risk, win probability, from-scratch xG). A FastAPI backend
serves everything to a vanilla-JS single-page dashboard. No build step, no test suite — verification is
manual (run the pipeline, hit the API, click through the dashboard).

## Commands

```bash
# Environment
python -m venv venv
venv\Scripts\activate            # Windows (this project's venv lives at ./venv, not ./.venv despite README)
pip install -r requirements.txt

# One-time schema setup (safe to re-run; CREATE TABLE IF NOT EXISTS — column renames need manual ALTER)
python init_db.py

# Full pipeline (ingest -> labels), then train + persist models to model_registry
python main.py
python main.py --train
python main.py --skip-ingest --train     # re-train only, data already loaded
python main.py --skip-weather            # skip Open-Meteo geocoding/fetch (slow, rate-limited)
python main.py --workers N               # default: CPU count - 1

# Individual pipeline stages
python -m pipelines.ingest_statsbomb     # events, players, teams, pass networks (parallel, ProcessPoolExecutor)
python -m pipelines.extract_shots        # shot events + freeze-frame xG features
python -m pipelines.ingest_injuries      # Transfermarkt player + injury matching
python -m pipelines.ingest_weather       # stadium geocoding + Open-Meteo historical fetch
python -m pipelines.compute_labels       # workload features + injury risk labels
python -m pipelines.ingest_pass_network  # backfill-only: pass edges (normally done during statsbomb ingest)

# Dashboard
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000 ; diagnostics at /api/health and the in-app Debug page

# EDA notebook (after schema/data changes, re-run and re-extract figures so served PNGs match)
jupyter nbconvert --to notebook --execute --inplace analysis/eda.ipynb
python analysis/extract_eda_figures.py

# Schema validation against the live DB
python utils/validate_schema.py
```

There is no lint/test/build command configured in this repo — don't invent one.

## Architecture

### Pipeline flow (`main.py`)
Ingestion stages run in order, then optional weather, then label computation, then optional training:
StatsBomb ingest → shot extraction → injury ingest → weather ingest (`--skip-weather` to skip) →
`compute_labels` → (`--train`) model training + registry persistence. Each stage is idempotent against
re-runs on the same data.

### Database is the source of truth
`schema.sql` is the full DDL, applied by `init_db.py`. ID convention: `*_id` = internal SERIAL PK,
`sb_*_id` = StatsBomb source ID, `tm_*_id` = Transfermarkt source ID. Core fact table is
`player_match_stats` (one row per player × match) — every model trains off it or a feature table
derived from it (`player_match_features`, `model5_features_pre`, etc.). `pass_network_edges` and
`shots` are the other two raw-but-structured tables (graph metrics, xG inputs respectively). `weather`
joins to matches via `stadiums.stadium_lat/lng` (curated map → geopy/Nominatim fallback) and Open-Meteo's
historical archive API.

### Hybrid model persistence (`core/registry.py`)
Trained `.pkl`/`.parquet` artifacts still live on disk under `artifacts/model{1,2,3,5,_xg}/` (written
first, so a DB hiccup never aborts training). In addition, `main.py` calls `core/registry.py` after each
model's `run()` to:
- upsert a row into `model_registry` (algorithm, target, features JSONB, metrics JSONB, training row
  count, artifact path) keyed by `(model_key, version)`
- `replace_table()` the model's derived DataFrame into its own Postgres table (`model1_player_clusters`,
  `xg_shot_predictions`, `model2_graph_features`, `model3_features`, `model5_features_pre`) — dropped and
  recreated from the DataFrame schema on every `--train` run

This is what backs `GET /api/models` and the dashboard's Models/Methodology page. When changing a model's
output shape, the prediction table regenerates automatically; only `model_registry`'s JSONB columns need
no migration.

### API fallback chain (`api_server.py`)
Every team-scoped endpoint tries, in order, and reports which it used via a `source` field in the JSON
response (and a topbar badge in the UI): **DB + ML model** (live query fed into a loaded artifact) → **DB
only** (heuristic score, no artifact loaded) → **artifact only** (parquet read directly, DB unreachable) →
**demo fallback** (hardcoded data, UI never blank). When adding/debugging an endpoint, preserve this
fallback ordering rather than letting a missing artifact 500 the request. `_query()`/`_coerce()` are the
shared DB-access helpers; artifacts are loaded once at startup into an in-memory dict.

Season filtering is a cross-cutting concern: most team-scoped endpoints accept `season=S` (or omit/`all`)
via `_season_filter()`/`_season_match_ids()` — when adding a new team-scoped endpoint, wire this in rather
than special-casing it.

### Frontend (no build step, vanilla JS)
- `pageLoader.js` fetches each page's HTML fragment and injects it, then re-executes `<script>` tags
  (browsers don't run scripts inserted via `innerHTML`).
- `navigation.js` switches visible pages and fires `onNavigate` so `main.js` can lazy-render on first
  visit only (avoids Chart.js canvas-sizing bugs when a canvas is off-screen at init).
- `main.js` awaits `pagesLoadedPromise`, then refreshes team-dependent pages via `Promise.allSettled` on
  team/season change (one page's fetch failure doesn't break the others); EDA/Models data is
  team-independent and fetched once at bootstrap.
- To add a new page: mirror the existing pattern across `index.html` (nav item + container),
  `pageLoader.js` (`PAGES`), `navigation.js` (`PAGE_TITLES`), `api.js` (fetch wrapper), `main.js` (render
  function + registration), plus a new `front-end/pages/<name>.html`. Bump the asset-version query param
  used for cache-busting when changing JS/CSS.
- CSS load order matters: `variables.css` → `reset.css` → `layout.css` → `components.css` → `pages.css`.

### Config
All configuration is environment-driven via `config/settings.py` (reads `.env`, fails loudly if a
required var is missing — no silent defaults for `DB_DSN`/data paths). `COMPETITIONS` in that file is the
single source of truth for dataset scope (currently: full 2015/16 season of the Big-5 leagues + WC
2018/22 + Euro 2020/24, ~2,025 matches). Changing scope means editing that set and re-running ingestion.

## Known gotchas (don't re-discover these)

- `main.py --train` always re-derives prediction tables from scratch (`replace_table` truncates) — if a
  model's output schema changes, no migration is needed for the prediction table, only check
  `model_registry`'s JSONB usage stays compatible.
- `compute_labels.py` must run before training Model 3 (injury risk) — if `player_match_features` is
  empty, `/api/injury-risk` silently falls back to a `minutes_played` heuristic rather than failing.
- Model 1 (`model1_player_clustering.py` v4.0) is *not* re-predicted live by the API — `/api/player-efficiency`
  reads the precomputed `final_player_archetype` column from `player_clusters.parquet` directly, because
  the spatial and style axes each need their own feature vector/scaler. Players absent from that table
  show as `"Unclassified"`.
- `pipelines/ingest_pass_network.py` is backfill-only; pass edges are normally produced during StatsBomb
  ingestion, not via this script.
- Stadium geocoding (`ingest_weather.py`) tries DB coords → a curated lat/lng map → Nominatim, in that
  order, and caches resolved coordinates back into `stadiums`; a handful of stadiums (renamed/demolished
  venues, training-ground "stadiums") are unresolvable and simply have no weather rows — this is expected,
  not a bug to chase.
- **Model artifacts are keyed to the local DB's SERIAL ids and are NOT portable.** Any prediction/feature
  parquet that carries `match_id`/`team_id`/`player_id` (e.g. `artifacts/model5/features_*_optimized.parquet`,
  `model2/graph_features.parquet`, `model_xg/shots_xg.parquet`) is keyed by ids that a *different* ingest run
  assigns differently. Never pull these from a teammate's machine/commit and serve them against your own DB —
  the ids silently won't line up (you'll serve the wrong team/match). After re-ingesting, or when an artifact
  was generated elsewhere, regenerate it locally (`python main.py --train`, or a single model e.g.
  `python -m models.model5_win_probability --optimize`). The `.pkl` estimators themselves are fine (trained on
  feature *values*); it's the id-keyed tables that drift.
