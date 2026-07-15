# Soccer Analytics ML Pipeline

A data pipeline and machine learning system built on StatsBomb open data and
Transfermarkt injury records. It ingests match events, computes per-player
statistics, links injury history, and trains five models covering player
clustering, team cohesion, injury risk, win probability, and a from-scratch
expected-goals (xG) model. A FastAPI backend serves everything through a live
single-page analytics dashboard with dedicated **EDA** and **Models &
Methodology** pages.

---

## Model persistence (hybrid registry)

Trained model **binaries** (`.pkl`) and derived **parquet** files are written to
`artifacts/model{1,2,3,5,_xg}/` on disk — the standard place for sklearn /
xgboost estimators. In addition, training records a queryable record of every
model in PostgreSQL:

- **`model_registry`** table — one row per trained model: algorithm, target,
  feature list (`JSONB`), evaluation metrics (`JSONB`, e.g. ROC-AUC, silhouette,
  R², accuracy, held-out scores), training row count, sklearn version, artifact
  path, and timestamp. Upserted on `(model_key, version)`.
- **Prediction tables** — each model's derived tabular output (player cluster
  assignments, per-shot xG predictions, feature matrices) is loaded into its own
  Postgres table (`model1_player_clusters`, `xg_shot_predictions`,
  `model2_graph_features`, `model3_features`, `model5_features_pre`). These are
  created dynamically from the DataFrame schema by `core/registry.replace_table()`.

This is what powers the **Models** page (`/api/models`) and the model metrics
shown across the dashboard. Persistence runs from `main.py` after each model's
`run()`; a DB hiccup never aborts training because the on-disk artifacts are
written first. See `core/registry.py`.

---

## Quick Start

### Requirements
- Python 3.11+
- PostgreSQL 14+
- ~4 GB disk for raw data and DB

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the example env file and fill in your values:
Edit `.env` — set `DB_DSN`, `DATA_ROOT`, `TRANSFERMARKT_CSV`, `TRANSFERMARKT_PLAYERS_CSV`

### 2. Create the database schema

Run once before anything else (creates all tables, including `model_registry`):

```bash
python init_db.py
```

### 3. Run the full pipeline

```bash
python main.py
```

This runs all ingestion and labelling stages in order. To also train the ML
models and populate the model registry + prediction tables:

```bash
python main.py --train
```

### 4. Launch the dashboard

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

### Other pipeline flags

```
--skip-ingest     Skip ingestion, run only labels and optional training
--workers N       Set number of parallel worker processes (default: CPU count − 1)
```

### Running pipeline stages individually

```bash
python -m pipelines.ingest_statsbomb   # StatsBomb events, players, teams, pass networks
python -m pipelines.extract_shots      # Shot events + freeze-frame features (xG inputs)
python -m pipelines.ingest_injuries    # Transfermarkt player and injury data
python -m pipelines.compute_labels     # Workload features and injury risk labels
```

---

## Diagnosing problems

Check the health endpoint before clicking around the dashboard:

```
http://localhost:8000/api/health
```

It reports DB connectivity, row counts for every table, which ML artifact files
are loaded, and the most recent exception per API endpoint. The dashboard also
has a built-in **Debug** page (🔧 icon in the sidebar).

| Endpoint | What it shows |
|---|---|
| `/api/health` | DB status, table counts, artifact load state, last errors |
| `/api/debug/artifacts` | Type and shape of every loaded `.pkl` / `.parquet` file |
| `/api/debug/db` | PostgreSQL version, psycopg2 version |

**Common issues:**

- `db_ok: false` — check `DB_DSN` in `.env` and confirm PostgreSQL is running.
- Tables showing `0` rows — run the ingestion pipeline first (`python main.py`).
- Artifacts showing `missing`, or `/api/models` returning `source: "empty"` —
  run training (`python main.py --train`).
- Pages loading but showing demo data — check the `source` field returned by
  each API endpoint. `"fallback"` means neither DB nor artifacts produced real data.

---

## Data Sources

| Source | What it provides | How to get it |
|---|---|---|
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Match events, lineups, player IDs, shots | Clone the repo; point `DATA_ROOT` at the `data/` folder |
| [Transfermarkt injuries (Kaggle)](https://www.kaggle.com/datasets/irrazional/transfermarkt-injuries) | Injury records with dates | Download CSV |
| [Transfermarkt players (Kaggle)](https://www.kaggle.com/datasets/davidcariboo/player-scores) | DOB, nationality, position, TM player IDs | Download `players.csv` |

### Competitions in scope

The default scope is the complete **2015/16 season of all five major European
leagues** (La Liga, Premier League, Serie A, Ligue 1, Bundesliga) plus the men's
international tournaments **World Cup 2018 & 2022** and **Euro 2020 & 2024**
(~2,025 matches across ~100 clubs and 50+ nations). To change scope, edit the
`COMPETITIONS` set in `config/settings.py`.

---

## Repository Structure

```
.
├── config/settings.py                 # DB connection, file paths, competition scope
├── core/
│   ├── caches.py                      # In-memory team and player caches (batch DB writes)
│   ├── registry.py                    # Hybrid persistence: model_registry + prediction tables
│   └── utils.py                       # norm_name() for accent-stripped name matching
├── extract/statsbomb_local.py         # Read StatsBomb JSON files from disk
├── transform/
│   ├── features.py                    # Vectorised per-player stat aggregation, xa extraction
│   └── schema.py                      # StatsBomb nested-dict field accessors
├── pipelines/
│   ├── ingest_statsbomb.py            # Parallel match ingestion (ProcessPoolExecutor)
│   ├── extract_shots.py               # Shot events + freeze-frame features for xG
│   ├── ingest_injuries.py            # Transfermarkt player and injury matching
│   ├── compute_labels.py             # Workload features and injury risk labels
│   └── ingest_pass_network.py         # Backfill tool for pass edges only
├── models/
│   ├── model1_player_clustering.py    # KMeans (spatial) + GMM (style) archetypes
│   ├── model2_team_cohesion.py        # Pass-network graph metrics + GBR/Ridge regression
│   ├── model3_injury_risk.py          # XGBoost / RF / LR binary classifier
│   ├── model5_win_probability.py      # Win/draw/loss classifier (pre-match + in-game)
│   ├── model_xg.py                    # From-scratch xG (HistGradientBoostingClassifier)
│   └── eval_utils.py                  # Leakage-safe CV (GroupKFold, held-out season)
├── front-end/
│   ├── index.html                     # App shell (sidebar, topbar, page containers)
│   ├── css/                           # variables → reset → layout → components → pages
│   ├── js/
│   │   ├── charts.js                  # Chart.js wrappers (incl. EDA + xG-benchmark charts)
│   │   ├── passNetwork.js             # SVG force-directed pass network graph
│   │   ├── navigation.js              # Page switching with onNavigate callback hook
│   │   ├── api.js                     # Fetch wrappers for all API endpoints
│   │   ├── main.js                    # Bootstrap, render functions, error display
│   │   └── pageLoader.js              # Async HTML injection with script re-execution
│   └── pages/
│       ├── dashboard.html             # KPIs, performance trend, recent results, league xG
│       ├── player.html                # Player profile, radar chart, style clusters
│       ├── xg.html                    # Shot map, goals-vs-xG, finishing leaderboard
│       ├── cohesion.html              # Pass network visualization, centrality cards
│       ├── injury.html                # Risk table, factors, injury risk chart
│       ├── winprob.html               # Probability banner, timelines, model accuracy
│       ├── eda.html                   # Exploratory data analysis (distributions, coverage)
│       ├── models.html                # Model cards: methodology, metrics, diagnostic figures
│       └── debug.html                 # System diagnostics (DB, artifacts, console log)
├── api_server.py                      # FastAPI backend — all dashboard API endpoints
├── schema.sql                         # Full DDL (run via init_db.py)
├── init_db.py                         # One-time schema creation and verification
├── main.py                            # Pipeline orchestrator (+ registry persistence)
└── requirements.txt
```

---

## Database Schema

ID naming convention used throughout:

| Prefix | Meaning |
|---|---|
| `*_id` | Internal surrogate primary key (SERIAL, generated by this DB) |
| `sb_*_id` | StatsBomb source identifier |
| `tm_*_id` | Transfermarkt source identifier |

### Core tables

**`teams`**, **`players`**, **`stadiums`**, **`matches`** — reference/dimension tables
**`injuries`** — one row per injury record from Transfermarkt
**`player_match_stats`** — one row per player × match; the central fact table used by all models
**`player_match_features`** — computed ML columns (workload, injury label) kept separate from raw stats
**`pass_network_edges`** — aggregated passer → receiver counts per match and team
**`match_minute_snapshots`** — cumulative in-game stats per team per minute (Model 5 in-game sub-model)
**`shots`** — one row per shot event with geometry, context, and freeze-frame features (xG model + shot maps)

### Model-persistence tables

**`model_registry`** — one row per trained model (metadata + metrics, see above)
**Prediction tables** — `model1_player_clusters`, `xg_shot_predictions`,
`model2_graph_features`, `model3_features`, `model5_features_pre`; created
dynamically at training time from each model's derived DataFrame.

---

## ML Models

| Key | Model | Type | Target | Headline metric |
|---|---|---|---|---|
| `model1_player_clustering` | Player Efficiency & Style Profiling | Clustering (KMeans + GMM) | Player archetype (dual-axis) | Spatial silhouette ≈ 0.29 |
| `model2_team_cohesion` | Team Cohesion (Pass Networks) | Graph metrics + GBR/Ridge | Goals scored | GBR R² ≈ 0.10 (grouped CV) |
| `model3_injury_risk` | Injury Risk Prediction | Classification (XGBoost / RF / LR) | `is_injured_next_30d` | ROC-AUC ≈ 0.77 |
| `model5_win_probability` | Win Probability | Classification (GBC, pre-match + in-game) | Win / draw / loss | In-game accuracy ≈ 0.64 |
| `model_xg` | Expected Goals (from scratch) | Classification (HistGradientBoosting) | `is_goal` per shot | ROC-AUC ≈ 0.82 (vs StatsBomb 0.81) |

All models are evaluated with leakage-safe cross-validation (GroupKFold /
StratifiedGroupKFold by match) plus a held-out 2022 season. Trained artifacts are
saved to `artifacts/model{N}/` and registered in `model_registry` (see above).

---

## Dashboard

The FastAPI server (`api_server.py`) serves a browser dashboard at
`http://localhost:8000`. Each analytics page corresponds to one ML model; the
**EDA** and **Models** pages give a whole-dataset and methodology view.

### Data source priority

Every endpoint tries sources in this order and falls back automatically:

1. **DB + ML model** — live database query fed into the trained artifact for inference
2. **DB only** — database query with a heuristic score when no artifact is loaded
3. **Artifact only** — parquet features file used directly when the DB is unreachable
4. **Demo fallback** — hardcoded plausible data so the UI is never blank

The data source badge in the topbar and the `source` field in every API response
indicate which path was taken.

### API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/options/teams` | List of teams for the selector dropdown |
| `GET /api/options/seasons?team_id=N` | Seasons a team appears in (drives the season selector) |
| `GET /api/player-efficiency?team_id=N&season=S` | Player stats, archetypes, radar chart data |
| `GET /api/team-cohesion?team_id=N&season=S` | Pass network edges and graph metrics |
| `GET /api/xg-finishing?team_id=N&season=S` | Goals vs xG per player, finishing leaderboard |
| `GET /api/shot-map?team_id=N&season=S` | Shot positions and xG for the pitch map |
| `GET /api/league-xg?season=S` | Season table: goals vs xG, points vs expected points |
| `GET /api/matches?team_id=N&season=S` | Matches for a team (optionally one season) |
| `GET /api/match-xg-timeline?match_id=N` | Cumulative xG race for both teams in a match |
| `GET /api/injury-risk?team_id=N&season=S` | Per-player injury risk scores |
| `GET /api/win-probability?team_id=N&season=S` | Model's average pre-match win/draw/loss + actual record |
| `GET /api/eda` | Exploratory data analysis aggregations over the source tables |
| `GET /api/models` | Model registry: algorithm, features, metrics, diagnostic figures |
| `GET /api/health` | DB status, table counts, artifact state, last errors |
| `GET /api/debug/artifacts` | Detailed type and shape of every loaded artifact |
| `GET /api/debug/db` | PostgreSQL and psycopg2 version |
| `GET /artifacts/...` | Static mount serving trained-model diagnostic figures (PNGs) |

The `season=S` parameter is optional on every team-scoped endpoint — omit it (or pass
`season=all`) for all matches, or pass a value from `/api/options/seasons` (e.g.
`2015/2016`, `2024`) to filter the whole dashboard to one season. The topbar season
selector is populated per team and defaults to that team's most-played season.

### Frontend architecture

A vanilla JS single-page application with no build step.

- **`pageLoader.js`** fetches each page's HTML template and injects it into the
  shell, re-executing `<script>` tags after injection (browsers do not run
  scripts added via `innerHTML`).
- **`navigation.js`** switches visible pages and fires an `onNavigate` callback
  so `main.js` can lazy-render pages on first visit (avoids Chart.js sizing bugs
  on off-screen canvases).
- **`main.js`** awaits `pagesLoadedPromise` before initialising. Team-dependent
  pages refresh on team change via `Promise.allSettled` (failures are isolated);
  the team-independent **EDA** and **Models** data is fetched once on bootstrap.
- API errors are surfaced as visible red banners rather than silently swallowed.

To add a new page, mirror the existing pattern across `index.html` (nav item +
container), `pageLoader.js` (`PAGES`), `navigation.js` (`PAGE_TITLES`), `api.js`
(fetch wrapper), `main.js` (render function), and a new `pages/<name>.html`.

### Design system

A "Precision Analytics" theme — dark industrial palette, data-forward aesthetic.

| Role | Font |
|---|---|
| Headings, page title | Syne |
| KPI values, numbers, labels | IBM Plex Mono |
| Body, UI copy | DM Sans |

CSS is split across five files loaded in dependency order: `variables.css` →
`reset.css` → `layout.css` → `components.css` → `pages.css`. All colors,
spacing, and typography are defined as CSS custom properties in `variables.css`.

---

## Notes

- Re-run `init_db.py` after changing `schema.sql`. It uses `CREATE TABLE IF NOT
  EXISTS` so it is safe to re-run; column renames need manual `ALTER TABLE`.
  Use `utils/validate_schema.py` to check the live DB against the expected DDL.
- The model **prediction tables** are dropped and recreated on every
  `main.py --train` run by `core/registry.replace_table()`; the `model_registry`
  row is upserted, keeping one row per `(model_key, version)`.
- `ingest_pass_network.py` is a backfill-only tool — pass edges are normally
  extracted during StatsBomb ingestion.
- The `player_match_features` table must be populated by `compute_labels.py`
  before training Model 3. If empty, the injury endpoint falls back to a simpler
  heuristic derived from `minutes_played`.
- Only `artifacts/model1/model1_kmeans_spatial.pkl` / `model1_scaler_spatial.pkl`
  / `model1_gmm_style.pkl` / `model1_scaler.pkl` and the parquet/figure outputs
  are produced by the current (v4.0) `model1_player_clustering.py` and read by
  `api_server.py`. `/api/player-efficiency` serves archetypes from the
  precomputed `final_player_archetype` column in `player_clusters.parquet`
  (joined by `player_id`) rather than re-predicting live — the spatial and
  style axes each need their own feature vector and scaler, so a faithful live
  re-prediction isn't worth duplicating here. Players missing from that table
  are labelled `"Unclassified"`.
