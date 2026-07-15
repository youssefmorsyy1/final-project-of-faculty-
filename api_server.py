"""
api_server.py  — v2.5.0

FastAPI backend for the soccer-analytics dashboard.

Key design points
-----------------
- _query() uses a SimpleConnectionPool (min 1, max 10) rather than opening a
  fresh psycopg2 connection per request.
- Every team-scoped endpoint accepts an optional `season` query param and
  filters to a single season (see _season_filter / _season_match_ids). The
  frontend drives this via the per-team season selector and /api/options/seasons.
- Each endpoint reports a `source` field (database / artifact / fallback) so the
  UI can show where its data came from.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import traceback
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import sklearn          # noqa: F401
import sklearn.base     # noqa: F401
import sklearn.utils    # noqa: F401
import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.pool
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from psycopg2.extras import RealDictCursor

from config.settings import DB_DSN

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api_server")

BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "front-end"
ARTIFACT_DIR = BASE_DIR / "artifacts"

_last_errors: dict[str, str] = {}

# =============================================================================
# Connection pool
# =============================================================================

_pool: psycopg2.pool.SimpleConnectionPool | None = None


def _get_pool() -> psycopg2.pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.SimpleConnectionPool(1, 10, DB_DSN)
        logger.info("Connection pool created (min=1, max=10)")
    return _pool


# =============================================================================
# Artifact store
# =============================================================================

_A: dict[str, dict[str, Any]] = {}


def _load(path) -> Any:
    p = Path(path)
    if not p.exists():
        logger.debug("Artifact not found (skipping): %s", p)
        return None
    try:
        obj = joblib.load(p)
        logger.info("Loaded artifact: %s", p.name)
        return obj
    except Exception as exc:
        logger.warning("Could not load artifact %s: %s", p, exc)
        return None


def _parquet(path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        logger.debug("Parquet not found (skipping): %s", p)
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:
        logger.warning("Could not load parquet %s: %s", p, exc)
        return None


def _load_json(path) -> Any:
    p = Path(path)
    if not p.exists():
        logger.debug("JSON not found (skipping): %s", p)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load JSON %s: %s", p, exc)
        return None


def _load_all_artifacts() -> None:
    a = ARTIFACT_DIR
    # model1_player_clustering.py (v4.0) is dual-axis: a spatial KMeans (fit on
    # avg_x_start/avg_y_start) and a style GaussianMixture (fit on per-90 rate
    # features), each with its own scaler. There is no single "kmeans.pkl" /
    # "scaler.pkl" pair any more — those names belonged to a pre-v4.0 version
    # of the script. Loaded here under their real names for debug visibility
    # only; live serving uses the precomputed final_player_archetype column in
    # the parquet below (see player_efficiency()), not a live re-prediction.
    _A["m1"] = {
        "kmeans_spatial": _load(a / "model1" / "model1_kmeans_spatial.pkl"),
        "scaler_spatial": _load(a / "model1" / "model1_scaler_spatial.pkl"),
        "gmm_style":      _load(a / "model1" / "model1_gmm_style.pkl"),
        "scaler_style":   _load(a / "model1" / "model1_scaler.pkl"),
        "df":             _parquet(a / "model1" / "player_clusters.parquet"),
    }
    _A["m2"] = {
        "gbr":     _load(a / "model2" / "gbr.pkl"),
        "scaler":  _load(a / "model2" / "scaler.pkl"),
        "feat_df": _parquet(a / "model2" / "graph_features.parquet"),
    }
    _A["m5"] = {
        # v1 (legacy, kept as a fallback): own-form-only pre-match model.
        "gbc_pre":    _load(a / "model5" / "gbc_pre.pkl"),
        "scaler_pre": _load(a / "model5" / "scaler_pre.pkl"),
        "gbc_ig":     _load(a / "model5" / "gbc_ingame.pkl"),
        "scaler_ig":  _load(a / "model5" / "scaler_ingame.pkl"),
        "df_pre":     _parquet(a / "model5" / "features_pre.parquet"),
        # v2 (optimized, now the primary path). The pre-match (static) and
        # in-game (dynamic) sub-models, their scalers, the exact feature-column
        # order each was trained on, the per-match/per-minute feature tables,
        # and the training diagnostics metadata (held-out accuracy + baselines).
        "gbc_pre_v2":    _load(a / "model5" / "gbc_pre_optimized.pkl"),
        "scaler_pre_v2": _load(a / "model5" / "scaler_pre_optimized.pkl"),
        "feats_pre_v2":  _load_json(a / "model5" / "feature_columns_pre_optimized.json"),
        "df_pre_v2":     _parquet(a / "model5" / "features_pre_optimized.parquet"),
        "gbc_ig_v2":     _load(a / "model5" / "gbc_ingame_optimized.pkl"),
        "scaler_ig_v2":  _load(a / "model5" / "scaler_ingame_optimized.pkl"),
        "feats_ig_v2":   _load_json(a / "model5" / "feature_columns_ingame_optimized.json"),
        "df_ig_v2":      _parquet(a / "model5" / "features_ingame_optimized.parquet"),
        "meta_v2":       _load_json(a / "model5" / "model5_optimized_metadata.json"),
    }
    _A["mxg"] = {
        "shots": _parquet(a / "model_xg" / "shots_xg.parquet"),
        "model": _load(a / "model_xg" / "xg_model.pkl"),
    }
    _A["m3"] = {
        "xgb":    _load(a / "model3" / "xgb.pkl"),
        "rf":     _load(a / "model3" / "rf.pkl"),
        "scaler": _load(a / "model3" / "scaler.pkl"),
        "df":     _parquet(a / "model3" / "features.parquet"),
    }
    loaded = sum(1 for m in _A.values() for v in m.values() if v is not None)
    total  = sum(len(m) for m in _A.values())
    logger.info("Artifacts: %d / %d objects loaded", loaded, total)
    if loaded == 0:
        logger.warning(
            "No artifacts loaded — all endpoints will use DB or fallback data. "
            "Run `python main.py --train` to generate artifacts."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML artifacts from: %s", ARTIFACT_DIR)
    _load_all_artifacts()
    _get_pool()
    logger.info("Server ready.")
    yield
    if _pool:
        _pool.closeall()
        logger.info("Connection pool closed.")


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app = FastAPI(title="Soccer Analytics API", version="2.5.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Serve trained-model diagnostic figures (PCA scatter, silhouette bars, radar
# profiles, feature heatmaps, calibration, ...) straight from the artifacts dir
# so the Models & Methodology page can embed them with <img src="/artifacts/...">.
if ARTIFACT_DIR.exists():
    app.mount("/artifacts", StaticFiles(directory=ARTIFACT_DIR), name="artifacts")


# =============================================================================
# Request / response logging middleware
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    ms = (time.monotonic() - start) * 1000
    if request.url.path.startswith("/api"):
        logger.info(
            "%s %s -> %d  (%.1f ms)",
            request.method, request.url.path, response.status_code, ms,
        )
    return response


# =============================================================================
# DB helpers
# =============================================================================

def _coerce(val: Any) -> Any:
    if isinstance(val, Decimal):
        return float(val)
    return val


def _query(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [
                {k: _coerce(v) for k, v in row.items()}
                for row in cur.fetchall()
            ]
    finally:
        pool.putconn(conn)


def _db_ok() -> bool:
    try:
        _query("SELECT 1")
        return True
    except Exception as exc:
        logger.warning("DB connectivity check failed: %s", exc)
        return False


def _table_count(table: str) -> int | str:
    try:
        rows = _query(f"SELECT COUNT(*) AS n FROM {table}")
        return int(rows[0]["n"]) if rows else 0
    except Exception as exc:
        return f"error: {exc}"


def _validate_team_id(team_id: int) -> None:
    """Raise HTTP 404 if team_id does not exist in the teams table."""
    if not _db_ok():
        return
    rows = _query("SELECT 1 FROM teams WHERE team_id = %s LIMIT 1", (team_id,))
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"team_id={team_id} not found. Use /api/options/teams to list valid IDs.",
        )


def _has_season(season: str | None) -> bool:
    """True when a concrete season filter should be applied (not 'all'/blank)."""
    return bool(season) and season.lower() != "all"


def _season_filter(season: str | None, alias: str = "m") -> tuple[str, list]:
    """SQL fragment + params constraining a query (already joined to a `matches`
    alias) to one season. Returns ('', []) when no season filter is requested."""
    if _has_season(season):
        return f" AND {alias}.season = %s", [season]
    return "", []


def _season_match_ids(season: str | None) -> set[int] | None:
    """Set of match_ids in a season (for filtering artifact DataFrames by
    match_id). None means 'no season filter'."""
    if not _has_season(season):
        return None
    try:
        rows = _query("SELECT match_id FROM matches WHERE season = %s", (season,))
        return {int(r["match_id"]) for r in rows}
    except Exception as exc:
        logger.warning("_season_match_ids(%s) error: %s", season, exc)
        return None


# =============================================================================
# Static routes
# =============================================================================

@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


# =============================================================================
# /api/health
# =============================================================================

@app.get("/api/health")
def health() -> dict[str, Any]:
    db_reachable = _db_ok()

    try:
        import re
        dsn_host = re.sub(r":[^:@]+@", ":***@", DB_DSN)
    except Exception:
        dsn_host = "<parse error>"

    table_counts: dict[str, Any] = {}
    if db_reachable:
        for tbl in (
            "teams", "players", "matches", "stadiums", "weather",
            "player_match_stats", "shots", "injuries", "player_match_features",
            "pass_network_edges", "match_minute_snapshots",
        ):
            table_counts[tbl] = _table_count(tbl)
    else:
        table_counts = {"error": "DB unreachable"}

    artifact_status: dict[str, dict] = {}
    for model_key, model_dict in _A.items():
        artifact_status[model_key] = {
            k: ("loaded" if v is not None else "missing")
            for k, v in model_dict.items()
        }

    return {
        "db_ok":        db_reachable,
        "db_dsn_host":  dsn_host,
        "table_counts": table_counts,
        "artifacts":    artifact_status,
        "last_errors":  _last_errors,
    }


# =============================================================================
# /api/debug/artifacts
# =============================================================================

@app.get("/api/debug/artifacts")
def debug_artifacts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_key, model_dict in _A.items():
        out[model_key] = {}
        for k, v in model_dict.items():
            if v is None:
                out[model_key][k] = {"status": "missing"}
            elif isinstance(v, pd.DataFrame):
                out[model_key][k] = {
                    "status": "loaded",
                    "type": "DataFrame",
                    "rows": len(v),
                    "cols": list(v.columns),
                }
            else:
                out[model_key][k] = {
                    "status": "loaded",
                    "type": type(v).__name__,
                }
    return out


# =============================================================================
# /api/debug/db
# =============================================================================

@app.get("/api/debug/db")
def debug_db() -> dict[str, Any]:
    try:
        rows = _query("SELECT version() AS v")
        pg_version = rows[0]["v"] if rows else "unknown"
        return {
            "ok": True,
            "pg_version": pg_version,
            "psycopg2_version": psycopg2.__version__,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# /api/models  — model registry (powers the Models & Methodology page)
# =============================================================================

def _model_figures(artifact_path: str | None) -> list[str]:
    """Return URLs for the diagnostic PNGs of a model, served via /artifacts.

    The registry stores artifact_path like "artifacts/model1"; the on-disk
    folder name (its last component) is also the /artifacts URL segment."""
    if not artifact_path:
        return []
    name = Path(artifact_path).name           # e.g. "model1", "model_xg"
    d = ARTIFACT_DIR / name
    if not d.exists():
        return []
    return sorted(f"/artifacts/{name}/{p.name}" for p in d.glob("*.png"))


@app.get("/api/models")
def models() -> dict[str, Any]:
    try:
        rows = _query(
            """
            SELECT model_key, version, display_name, task, algorithm, target,
                   features, metrics, n_train_rows, sklearn_version,
                   artifact_path, prediction_table, trained_at
            FROM model_registry
            ORDER BY model_key
            """
        )
        if rows:
            for r in rows:
                if r.get("trained_at") is not None:
                    r["trained_at"] = str(r["trained_at"])
                r["figures"] = _model_figures(r.get("artifact_path"))
            logger.info("/api/models: %d models from registry", len(rows))
            return {"models": rows, "source": "database"}
        logger.warning("/api/models: registry empty")
        return {
            "models": [],
            "source": "empty",
            "note": "model_registry is empty — run `python main.py --train` "
                    "to train models and populate the registry.",
        }
    except Exception as exc:
        logger.warning("/api/models DB error: %s", exc)
        _last_errors["models"] = str(exc)
        return {"models": [], "source": "fallback", "error": str(exc)}


# =============================================================================
# /api/eda  — exploratory data analysis over the source tables
# =============================================================================

def _safe_rows(sql: str, params: tuple = ()) -> list[dict]:
    """Run a query and return [] on any error (EDA panels degrade independently)."""
    try:
        return _query(sql, params)
    except Exception as exc:
        logger.warning("/api/eda query failed: %s", exc)
        return []


@app.get("/api/eda")
def eda() -> dict[str, Any]:
    if not _db_ok():
        return {"source": "fallback", "error": "DB unreachable", "overview": {}}

    overview = {
        tbl: _table_count(tbl)
        for tbl in (
            "teams", "players", "matches", "shots",
            "injuries", "player_match_stats", "pass_network_edges",
        )
    }

    # Matches per competition + season (dataset scope/coverage).
    coverage = _safe_rows(
        """
        SELECT competition, season, COUNT(*) AS matches
        FROM matches
        WHERE competition IS NOT NULL
        GROUP BY competition, season
        ORDER BY competition, season
        """
    )

    # Players per nominal position.
    positions = _safe_rows(
        """
        SELECT COALESCE(NULLIF(TRIM(position), ''), 'Unknown') AS position,
               COUNT(*) AS n
        FROM players
        GROUP BY 1
        ORDER BY n DESC
        LIMIT 15
        """
    )

    # Shot distance distribution (metres to goal, 16 buckets over 0–40 m).
    shot_distance = _safe_rows(
        """
        SELECT width_bucket(distance, 0, 40, 16) AS bucket, COUNT(*) AS n
        FROM shots
        WHERE distance IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
        """
    )

    # StatsBomb xG distribution (10 buckets over 0–1).
    xg_distribution = _safe_rows(
        """
        SELECT width_bucket(statsbomb_xg, 0, 1, 10) AS bucket, COUNT(*) AS n
        FROM shots
        WHERE statsbomb_xg IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
        """
    )

    # Goals per match distribution.
    goals_per_match = _safe_rows(
        """
        SELECT (home_score + away_score) AS total_goals, COUNT(*) AS n
        FROM matches
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        GROUP BY total_goals
        ORDER BY total_goals
        """
    )

    # Shot conversion by body part (data-quality + finishing insight).
    conversion_by_bodypart = _safe_rows(
        """
        SELECT COALESCE(body_part, 'Unknown') AS body_part,
               COUNT(*) AS shots,
               SUM(CASE WHEN is_goal THEN 1 ELSE 0 END) AS goals,
               ROUND(AVG(CASE WHEN is_goal THEN 1.0 ELSE 0.0 END)::numeric, 3) AS conversion
        FROM shots
        GROUP BY 1
        ORDER BY shots DESC
        """
    )

    # Injury label balance (for the injury model).
    injury_balance = _safe_rows(
        """
        SELECT is_injured_next_30d AS injured, COUNT(*) AS n
        FROM player_match_features
        GROUP BY 1
        """
    )

    return {
        "source": "database",
        "overview": overview,
        "coverage": coverage,
        "positions": positions,
        "shot_distance": shot_distance,
        "xg_distribution": xg_distribution,
        "goals_per_match": goals_per_match,
        "conversion_by_bodypart": conversion_by_bodypart,
        "injury_balance": injury_balance,
    }


# =============================================================================
# /api/options/teams
# =============================================================================

@app.get("/api/options/teams")
def teams() -> dict[str, Any]:
    try:
        rows = _query(
            "SELECT t.team_id, t.team_name FROM teams t ORDER BY t.team_name"
        )
        if rows:
            logger.info("/api/options/teams: %d teams from DB", len(rows))
            return {"teams": rows, "source": "database"}
    except Exception as exc:
        logger.warning("/api/options/teams DB error: %s", exc)
        _last_errors["teams"] = str(exc)

    df = _A.get("m1", {}).get("df")
    if df is not None and "team_id" in df.columns and "team_name" in df.columns:
        teams_from_artifact = (
            df[["team_id", "team_name"]].drop_duplicates().to_dict("records")
        )
        if teams_from_artifact:
            logger.info("/api/options/teams: %d teams from artifact", len(teams_from_artifact))
            return {"teams": teams_from_artifact, "source": "artifact"}

    logger.warning("/api/options/teams: using fallback data")
    return {
        "teams": [
            {"team_id": 1, "team_name": "Manchester City"},
            {"team_id": 2, "team_name": "Arsenal"},
            {"team_id": 3, "team_name": "Liverpool"},
            {"team_id": 4, "team_name": "Chelsea"},
        ],
        "source": "fallback",
    }


# =============================================================================
# /api/options/seasons — seasons a given team appears in (drives the season
# selector). Ordered most-played-first so the frontend can default to the
# team's primary season.
# =============================================================================

@app.get("/api/options/seasons")
def seasons(team_id: int) -> dict[str, Any]:
    _validate_team_id(team_id)
    try:
        rows = _query(
            """
            SELECT m.season, COUNT(*) AS matches
            FROM matches m
            WHERE (m.home_team_id = %s OR m.away_team_id = %s)
              AND m.season IS NOT NULL
            GROUP BY m.season
            ORDER BY COUNT(*) DESC, m.season DESC
            """,
            (team_id, team_id),
        )
        if rows:
            return {
                "seasons": [
                    {"season": r["season"], "matches": int(r["matches"])}
                    for r in rows
                ],
                "source": "database",
            }
    except Exception as exc:
        logger.warning("/api/options/seasons team=%d DB error: %s", team_id, exc)
        _last_errors["seasons"] = str(exc)

    return {"seasons": [{"season": "2015/2016", "matches": 0}], "source": "fallback"}


# =============================================================================
# /api/player-efficiency
# =============================================================================

@app.get("/api/player-efficiency")
def player_efficiency(team_id: int, season: str | None = None) -> dict[str, Any]:
    _validate_team_id(team_id)
    season_sql, season_params = _season_filter(season)

    cluster_df = _A.get("m1", {}).get("df")

    # Player archetypes come from the precomputed final_player_archetype
    # column in player_clusters.parquet (Model 1's dual-axis spatial+style
    # clustering, run offline over the full historical dataset), looked up by
    # player_id. There is no cheap, correct way to re-predict an archetype
    # live for a player missing from that table: the spatial axis needs
    # avg_x_start/avg_y_start (not part of this per-match-stats query) and the
    # style axis needs the full per-90 feature vector + GMM posterior. Players
    # missing from the parquet (e.g. below Model 1's training thresholds) are
    # simply labelled "Unclassified" rather than guessed.
    archetype_by_player: dict[int, str] = {}
    if (cluster_df is not None
            and "player_id" in cluster_df.columns
            and "final_player_archetype" in cluster_df.columns):
        archetype_by_player = {
            int(pid): str(arch)
            for pid, arch in cluster_df[["player_id", "final_player_archetype"]]
            .drop_duplicates(subset="player_id")
            .itertuples(index=False, name=None)
        }

    try:
        rows = _query(
            """
            SELECT p.player_id, p.player_name,
                   mode() WITHIN GROUP (ORDER BY pms.starting_position) AS position,
                   COUNT(*) AS matches,
                   AVG(pms.minutes_played) * COUNT(*) AS minutes,
                   AVG(pms.xg)                  AS xg_per_90,
                   AVG(pms.xa)                  AS xa_per_90,
                   AVG(pms.pass_accuracy)        AS pass_completion,
                   AVG(pms.key_passes)           AS key_passes,
                   AVG(pms.dribbles_completed)   AS dribbles,
                   AVG(pms.shots)                AS shots,
                   AVG(pms.passes_attempted)     AS passes_attempted,
                   AVG(pms.tackles)              AS tackles,
                   AVG(pms.interceptions)        AS interceptions,
                   AVG(pms.clearances)           AS clearances,
                   AVG(pms.carry_distance)       AS carry_distance,
                   AVG(pms.progressive_carries)  AS progressive_carries,
                   AVG(pms.progressive_passes)   AS progressive_passes,
                   AVG(pms.pressures)            AS pressures
            FROM player_match_stats pms
            JOIN players p ON p.player_id = pms.player_id
            JOIN matches m ON m.match_id = pms.match_id
            WHERE pms.team_id = %s""" + season_sql + """
            GROUP BY p.player_id, p.player_name
            HAVING COUNT(*) >= 3
            ORDER BY AVG(pms.xa + pms.xg) DESC
            LIMIT 12
            """,
            tuple([team_id] + season_params),
        )
        if rows:
            logger.info("/api/player-efficiency team=%d: %d players from DB", team_id, len(rows))
            rows = [dict(r) for r in rows]

            for r in rows:
                r["player_type"] = archetype_by_player.get(
                    int(r["player_id"]), "Unclassified"
                )

            leader = rows[0]
            radar  = _build_radar(leader)
            return {
                "leader":  leader,
                "radar":   radar,
                "players": rows,
                "source":  "database+artifact",
            }
    except Exception as exc:
        logger.warning("/api/player-efficiency team=%d DB error: %s", team_id, exc)
        _last_errors["player_efficiency"] = str(exc)

    if cluster_df is not None:
        team_df = (
            cluster_df[cluster_df["team_id"] == team_id]
            if "team_id" in cluster_df.columns
            else cluster_df
        ).head(12)

        if not team_df.empty:
            players_out = []
            for _, row in team_df.iterrows():
                players_out.append({
                    "player_name":     str(row["player_name"]),
                    "position":        str(row.get("spatial_cluster_name") or "-"),
                    "player_type":     str(row.get("final_player_archetype") or "Unclassified"),
                    "matches":         int(row.get("matches_played", 10)),
                    "minutes":         int(row.get("matches_played", 10)) * 85,
                    "xg_per_90":       round(float(row.get("shots") or 2) * 0.10, 2),
                    "xa_per_90":       round(float(row.get("key_passes") or 2) * 0.12, 2),
                    "pass_completion": round(float(row.get("passes_attempted") or 40), 1),
                    "key_passes":      round(float(row.get("key_passes") or 2), 1),
                    "dribbles":        round(float(row.get("dribbles_completed") or 1), 1),
                    "shots":           round(float(row.get("shots") or 2), 1),
                })
            if players_out:
                leader = players_out[0]
                return {
                    "leader":  leader,
                    "radar":   _build_radar(leader),
                    "players": players_out,
                    "source":  "artifact",
                }

    logger.warning("/api/player-efficiency team=%d: using fallback", team_id)
    return _fallback_player(team_id)


# =============================================================================
# /api/team-cohesion
# =============================================================================

@app.get("/api/team-cohesion")
def team_cohesion(team_id: int, season: str | None = None) -> dict[str, Any]:
    _validate_team_id(team_id)
    season_sql, season_params = _season_filter(season)

    feat_df = _A.get("m2", {}).get("feat_df")
    gbr     = _A.get("m2", {}).get("gbr")
    scaler  = _A.get("m2", {}).get("scaler")

    GRAPH_FEATURES = [
        "network_density", "clustering_coefficient",
        "mean_in_centrality", "mean_out_centrality",
        "mean_betweenness", "max_betweenness",
        "mean_pagerank", "max_pagerank",
        "n_nodes", "n_edges", "total_passes", "pass_per_edge",
    ]
    # Context features appended by model2 (team xG/xGA, home, opponent quality).
    # Must match models.model2_team_cohesion.MODEL_FEATURES order so the
    # persisted scaler/gbr receive the correct feature vector.
    CONTEXT_FEATURES = ["team_xg", "team_xga", "is_home", "opponent_quality"]
    MODEL_FEATURES = GRAPH_FEATURES + CONTEXT_FEATURES

    kpi_from_artifact = None
    predicted_goals   = None

    if feat_df is not None and "team_id" in feat_df.columns:
        team_feats = feat_df[feat_df["team_id"] == team_id]
        if _has_season(season) and "season" in team_feats.columns:
            team_feats = team_feats[team_feats["season"] == season]
        if not team_feats.empty:
            avg = team_feats[GRAPH_FEATURES].fillna(0).mean()
            # Real average degree of an undirected graph = 2E / V, averaged over
            # the team's matches. (The old code mistakenly reported the node
            # count here.)
            n_nodes = float(avg.get("n_nodes", 0))
            n_edges = float(avg.get("n_edges", 0))
            avg_degree = (2 * n_edges / n_nodes) if n_nodes else 0.0
            kpi_from_artifact = {
                "network_density":  round(float(avg.get("network_density", 0)), 2),
                "avg_degree":       round(avg_degree, 1),
                "clustering_coeff": round(float(avg.get("clustering_coefficient", 0)), 2),
                "mean_betweenness": round(float(avg.get("mean_betweenness", 0)), 3),
            }
            # Prediction uses the full feature vector the model was trained on.
            have_ctx = all(c in team_feats.columns for c in CONTEXT_FEATURES)
            if gbr is not None and scaler is not None and have_ctx:
                full_avg = team_feats[MODEL_FEATURES].fillna(0).mean()
                X = scaler.transform(full_avg.values.reshape(1, -1))
                predicted_goals = round(float(gbr.predict(X)[0]), 2)

    # Average pitch position per player (weighted by passes made from there),
    # for the on-pitch network layout. StatsBomb units: x 0-120, y 0-80.
    nodes = []
    try:
        node_rows = _query(
            """
            SELECT p.player_name AS name,
                   SUM(pne.avg_x_start * pne.pass_count) / NULLIF(SUM(pne.pass_count),0) AS x,
                   SUM(pne.avg_y_start * pne.pass_count) / NULLIF(SUM(pne.pass_count),0) AS y,
                   SUM(pne.pass_count) AS volume
            FROM pass_network_edges pne
            JOIN players p ON p.player_id = pne.passer_id
            JOIN matches m ON m.match_id = pne.match_id
            WHERE pne.team_id = %s""" + season_sql + """
            GROUP BY p.player_name
            HAVING SUM(pne.pass_count) > 0
            ORDER BY SUM(pne.pass_count) DESC
            LIMIT 16
            """,
            tuple([team_id] + season_params),
        )
        nodes = [{
            "name":   r["name"],
            "x":      round(float(r["x"]), 1),
            "y":      round(float(r["y"]), 1),
            "volume": int(r["volume"]),
        } for r in node_rows if r["x"] is not None and r["y"] is not None]

        # True passing degree per player = number of distinct team-mates they
        # exchange passes with (as passer OR receiver), over the whole season.
        # Derived directly from the full edge table — NOT the capped edge list
        # the frontend draws — so the "Best Connected" table is accurate.
        deg_rows = _query(
            """
            SELECT p.player_name AS name, COUNT(DISTINCT t.partner) AS degree
            FROM (
                SELECT pne.passer_id AS player, pne.receiver_id AS partner
                FROM pass_network_edges pne
                JOIN matches m ON m.match_id = pne.match_id
                WHERE pne.team_id = %s""" + season_sql + """ AND pne.pass_count > 0
                UNION
                SELECT pne.receiver_id AS player, pne.passer_id AS partner
                FROM pass_network_edges pne
                JOIN matches m ON m.match_id = pne.match_id
                WHERE pne.team_id = %s""" + season_sql + """ AND pne.pass_count > 0
            ) t
            JOIN players p ON p.player_id = t.player
            GROUP BY p.player_name
            """,
            tuple([team_id] + season_params + [team_id] + season_params),
        )
        deg_by_name = {r["name"]: int(r["degree"]) for r in deg_rows}
        for n in nodes:
            n["degree"] = deg_by_name.get(n["name"], 0)
    except Exception as exc:
        logger.warning("/api/team-cohesion team=%d node error: %s", team_id, exc)

    try:
        edges = _query(
            """
            SELECT p1.player_name AS passer,
                   p2.player_name AS receiver,
                   AVG(pne.pass_count) AS weight
            FROM pass_network_edges pne
            JOIN players p1 ON p1.player_id = pne.passer_id
            JOIN players p2 ON p2.player_id = pne.receiver_id
            JOIN matches m ON m.match_id = pne.match_id
            WHERE pne.team_id = %s""" + season_sql + """
            GROUP BY p1.player_name, p2.player_name
            ORDER BY AVG(pne.pass_count) DESC
            LIMIT 40
            """,
            tuple([team_id] + season_params),
        )
        if edges:
            logger.info("/api/team-cohesion team=%d: %d edges from DB", team_id, len(edges))
            # DB-only fallback (no model2 artifact loaded): derive degree from the
            # distinct passer/receiver links actually returned. These are
            # approximate (the edge list is capped), and betweenness isn't
            # available without the full per-match graph, so it's left null.
            n_nodes_db = len({e["passer"] for e in edges} | {e["receiver"] for e in edges})
            avg_degree_db = (2 * len(edges) / n_nodes_db) if n_nodes_db else 0.0
            possible = n_nodes_db * (n_nodes_db - 1)
            kpi_db = {
                "network_density":  round(min(1.0, len(edges) / possible), 2) if possible else 0.0,
                "avg_degree":       round(avg_degree_db, 1),
                "clustering_coeff": None,
                "mean_betweenness": None,
            }
            final_kpi = kpi_from_artifact if kpi_from_artifact else kpi_db
            if predicted_goals is not None:
                final_kpi["predicted_goals_per_match"] = predicted_goals

            return {
                "kpi":   final_kpi,
                "nodes": nodes,
                "edges": [
                    {"from": e["passer"], "to": e["receiver"],
                     "weight": round(float(e["weight"] or 0), 1)}
                    for e in edges
                ],
                "source": "database+artifact" if kpi_from_artifact else "database",
            }
    except Exception as exc:
        logger.warning("/api/team-cohesion team=%d DB error: %s", team_id, exc)
        _last_errors["team_cohesion"] = str(exc)

    if kpi_from_artifact:
        return {"kpi": kpi_from_artifact, "edges": [], "source": "artifact"}

    logger.warning("/api/team-cohesion team=%d: using fallback", team_id)
    return {
        "kpi": {"network_density": 0.74, "avg_degree": 15.4,
                "clustering_coeff": 0.81, "mean_betweenness": 0.077},
        "edges": [
            {"from": "Player A", "to": "Player B", "weight": 8.0},
            {"from": "Player B", "to": "Player C", "weight": 7.0},
            {"from": "Player C", "to": "Player D", "weight": 6.0},
        ],
        "source": "fallback",
    }


# =============================================================================
# /api/xg-finishing  (Model: from-scratch xG)
# =============================================================================

@app.get("/api/xg-finishing")
def xg_finishing(team_id: int, season: str | None = None) -> dict[str, Any]:
    _validate_team_id(team_id)

    shots = _A.get("mxg", {}).get("shots")
    if shots is None or "team_id" not in shots.columns:
        logger.warning("/api/xg-finishing: xG artifact unavailable")
        return {"players": [], "kpi": {}, "source": "fallback"}

    ts = shots[shots["team_id"] == team_id]
    mids = _season_match_ids(season)
    if mids is not None and "match_id" in ts.columns:
        ts = ts[ts["match_id"].isin(mids)]
    if ts.empty:
        return {"players": [], "kpi": {}, "source": "artifact"}

    team_xg    = float(ts["xg_pred"].sum())
    team_goals = int(ts["is_goal"].sum())
    n_shots    = int(len(ts))

    grp = ts.groupby("player_id").agg(
        shots=("is_goal", "size"),
        goals=("is_goal", "sum"),
        xg=("xg_pred", "sum"),
    ).reset_index()

    name_map: dict[int, tuple] = {}
    source = "artifact"
    try:
        ids = [int(x) for x in grp["player_id"].dropna().tolist()]
        if ids:
            rows = _query(
                """
                SELECT p.player_id, p.player_name,
                       COALESCE(
                           mode() WITHIN GROUP (ORDER BY pms.starting_position),
                           '-'
                       ) AS position
                FROM players p
                LEFT JOIN player_match_stats pms ON pms.player_id = p.player_id
                WHERE p.player_id = ANY(%s)
                GROUP BY p.player_id, p.player_name
                """,
                (ids,),
            )
            name_map = {r["player_id"]: (r["player_name"], r["position"]) for r in rows}
            source = "database+artifact"
    except Exception as exc:
        logger.warning("/api/xg-finishing name lookup error: %s", exc)
        _last_errors["xg_finishing"] = str(exc)

    players = []
    for _, r in grp.iterrows():
        if pd.isna(r["player_id"]):
            continue
        pid = int(r["player_id"])
        nm, pos = name_map.get(pid, (f"Player {pid}", "-"))
        xg    = round(float(r["xg"]), 2)
        goals = int(r["goals"])
        players.append({
            "player_name": nm,
            "position":    pos,
            "shots":       int(r["shots"]),
            "goals":       goals,
            "xg":          xg,
            "xg_diff":     round(goals - xg, 2),
        })
    players.sort(key=lambda p: p["xg_diff"], reverse=True)

    return {
        "players": players[:15],
        "kpi": {
            "team_xg":    round(team_xg, 1),
            "team_goals": team_goals,
            "xg_diff":    round(team_goals - team_xg, 1),
            "shots":      n_shots,
        },
        "source": source,
    }


# =============================================================================
# /api/shot-map   — every shot for a team on the pitch, sized/coloured by xG
# =============================================================================

def _xg_pred_lookup() -> dict[int, float]:
    """shot_id -> our model's xg_pred, from the xG artifact (empty if missing)."""
    shots = _A.get("mxg", {}).get("shots")
    if shots is None or "shot_id" not in shots.columns or "xg_pred" not in shots.columns:
        return {}
    return dict(zip(shots["shot_id"].astype(int), shots["xg_pred"].astype(float)))


@app.get("/api/shot-map")
def shot_map(team_id: int, season: str | None = None) -> dict[str, Any]:
    """All shots taken by a team, with pitch coordinates and xG.

    Coordinates are StatsBomb pitch units (x 0-120 toward goal, y 0-80).
    xG is our from-scratch model's prediction (falls back to statsbomb_xg).
    """
    _validate_team_id(team_id)
    season_sql, season_params = _season_filter(season)
    xgp = _xg_pred_lookup()
    try:
        rows = _query(
            """
            SELECT s.shot_id, s.x, s.y, s.minute, s.body_part,
                   s.statsbomb_xg, s.is_goal, p.player_name
            FROM shots s
            JOIN players p ON p.player_id = s.player_id
            JOIN matches m ON m.match_id = s.match_id
            WHERE s.team_id = %s AND s.x IS NOT NULL AND s.y IS NOT NULL""" + season_sql + """
            ORDER BY s.minute
            """,
            tuple([team_id] + season_params),
        )
    except Exception as exc:
        logger.warning("/api/shot-map team=%d DB error: %s", team_id, exc)
        _last_errors["shot_map"] = str(exc)
        return {"shots": [], "kpi": {}, "source": "fallback"}

    shots_out, tot_xg, goals = [], 0.0, 0
    for r in rows:
        xg = xgp.get(int(r["shot_id"]))
        if xg is None:
            xg = float(r["statsbomb_xg"] or 0.0)
        is_goal = bool(r["is_goal"])
        tot_xg += xg
        goals  += int(is_goal)
        shots_out.append({
            "x":           round(float(r["x"]), 1),
            "y":           round(float(r["y"]), 1),
            "xg":          round(xg, 3),
            "is_goal":     is_goal,
            "minute":      int(r["minute"]) if r["minute"] is not None else None,
            "body_part":   r["body_part"] or "-",
            "player_name": r["player_name"],
        })
    return {
        "shots": shots_out,
        "kpi": {
            "shots":    len(shots_out),
            "goals":    goals,
            "team_xg":  round(tot_xg, 1),
            "xg_per_shot": round(tot_xg / len(shots_out), 3) if shots_out else 0,
        },
        "source": "database+artifact" if xgp else "database",
    }


# =============================================================================
# /api/league-xg   — season table ranked by xG over/under-performance + xPoints
# =============================================================================

def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _xpoints(xg_for: float, xg_against: float, max_goals: int = 8) -> float:
    """Expected league points from a match, modelling each side's goals as
    independent Poisson(xG). xPoints = 3*P(win) + 1*P(draw)."""
    pf = [_poisson_pmf(i, xg_for)     for i in range(max_goals + 1)]
    pa = [_poisson_pmf(j, xg_against) for j in range(max_goals + 1)]
    p_win = p_draw = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pf[i] * pa[j]
            if i > j:
                p_win += p
            elif i == j:
                p_draw += p
    return 3 * p_win + p_draw


@app.get("/api/league-xg")
def league_xg(season: str = "2015/2016") -> dict[str, Any]:
    """Team table for a season: goals vs xG (for and against), points vs
    expected points. Aggregates our shot-level xG up to the match, then to the
    season. Defaults to 2015/16 — the only full league season in the data."""
    shots = _A.get("mxg", {}).get("shots")
    if shots is None or "match_id" not in shots.columns:
        return {"teams": [], "season": season, "source": "fallback"}

    xg_col = "xg_pred" if "xg_pred" in shots.columns else "statsbomb_xg"
    # match_id, team_id -> summed xG
    mt = (shots.groupby(["match_id", "team_id"])[xg_col].sum()
                .reset_index().rename(columns={xg_col: "xg"}))
    xg_by = {(int(r.match_id), int(r.team_id)): float(r.xg) for r in mt.itertuples()}

    try:
        matches = _query(
            """
            SELECT m.match_id, m.home_team_id, m.away_team_id,
                   m.home_score, m.away_score,
                   th.team_name AS home_name, ta.team_name AS away_name
            FROM matches m
            JOIN teams th ON th.team_id = m.home_team_id
            JOIN teams ta ON ta.team_id = m.away_team_id
            WHERE m.season = %s
            """,
            (season,),
        )
    except Exception as exc:
        logger.warning("/api/league-xg DB error: %s", exc)
        _last_errors["league_xg"] = str(exc)
        return {"teams": [], "season": season, "source": "fallback"}

    agg: dict[int, dict[str, Any]] = {}

    def _row(tid: int, name: str) -> dict[str, Any]:
        return agg.setdefault(tid, {
            "team_id": tid, "team_name": name, "played": 0,
            "goals_for": 0, "goals_against": 0, "xg_for": 0.0, "xg_against": 0.0,
            "points": 0, "xpoints": 0.0,
        })

    for m in matches:
        if m["home_score"] is None or m["away_score"] is None:
            continue
        hid, aid = int(m["home_team_id"]), int(m["away_team_id"])
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        hxg = xg_by.get((int(m["match_id"]), hid), 0.0)
        axg = xg_by.get((int(m["match_id"]), aid), 0.0)
        h, a = _row(hid, m["home_name"]), _row(aid, m["away_name"])
        h["played"] += 1; a["played"] += 1
        h["goals_for"] += hs; h["goals_against"] += as_
        a["goals_for"] += as_; a["goals_against"] += hs
        h["xg_for"] += hxg; h["xg_against"] += axg
        a["xg_for"] += axg; a["xg_against"] += hxg
        h["points"] += 3 if hs > as_ else (1 if hs == as_ else 0)
        a["points"] += 3 if as_ > hs else (1 if hs == as_ else 0)
        h["xpoints"] += _xpoints(hxg, axg)
        a["xpoints"] += _xpoints(axg, hxg)

    teams = []
    for t in agg.values():
        if t["played"] == 0:
            continue
        t["xg_for"] = round(t["xg_for"], 1)
        t["xg_against"] = round(t["xg_against"], 1)
        t["xg_diff"] = round(t["xg_for"] - t["xg_against"], 1)
        t["xpoints"] = round(t["xpoints"], 1)
        t["points_diff"] = round(t["points"] - t["xpoints"], 1)
        teams.append(t)
    teams.sort(key=lambda x: x["xpoints"], reverse=True)
    return {"teams": teams, "season": season,
            "source": "database+artifact" if teams else "fallback"}


# =============================================================================
# /api/matches   — match list for the in-game / timeline selector
# =============================================================================

@app.get("/api/matches")
def matches(team_id: int, season: str | None = None) -> dict[str, Any]:
    _validate_team_id(team_id)
    season_sql, season_params = _season_filter(season)
    try:
        rows = _query(
            """
            SELECT m.match_id, m.match_date, m.home_score, m.away_score,
                   m.home_team_id, m.away_team_id,
                   th.team_name AS home_name, ta.team_name AS away_name
            FROM matches m
            JOIN teams th ON th.team_id = m.home_team_id
            JOIN teams ta ON ta.team_id = m.away_team_id
            WHERE (m.home_team_id = %s OR m.away_team_id = %s)""" + season_sql + """
            ORDER BY m.match_date DESC NULLS LAST
            LIMIT 40
            """,
            tuple([team_id, team_id] + season_params),
        )
    except Exception as exc:
        logger.warning("/api/matches team=%d DB error: %s", team_id, exc)
        return {"matches": [], "source": "fallback"}
    out = [{
        "match_id":  int(r["match_id"]),
        "date":      str(r["match_date"]) if r["match_date"] else "",
        "home_name": r["home_name"], "away_name": r["away_name"],
        "home_score": r["home_score"], "away_score": r["away_score"],
        "label": f'{r["home_name"]} {r["home_score"]}-{r["away_score"]} {r["away_name"]}',
    } for r in rows]
    return {"matches": out, "source": "database"}


# =============================================================================
# /api/match-xg-timeline   — cumulative xG race for both teams in one match
# =============================================================================

@app.get("/api/match-xg-timeline")
def match_xg_timeline(match_id: int) -> dict[str, Any]:
    xgp = _xg_pred_lookup()
    try:
        meta = _query(
            """
            SELECT m.home_team_id, m.away_team_id, m.home_score, m.away_score,
                   th.team_name AS home_name, ta.team_name AS away_name
            FROM matches m
            JOIN teams th ON th.team_id = m.home_team_id
            JOIN teams ta ON ta.team_id = m.away_team_id
            WHERE m.match_id = %s
            """,
            (match_id,),
        )
        if not meta:
            raise HTTPException(status_code=404, detail="match not found")
        m = meta[0]
        shots = _query(
            """
            SELECT s.shot_id, s.team_id, s.minute, s.statsbomb_xg, s.is_goal,
                   p.player_name
            FROM shots s
            JOIN players p ON p.player_id = s.player_id
            WHERE s.match_id = %s
            ORDER BY s.minute
            """,
            (match_id,),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("/api/match-xg-timeline match=%d DB error: %s", match_id, exc)
        return {"source": "fallback"}

    hid, aid = int(m["home_team_id"]), int(m["away_team_id"])
    home_pts, away_pts, home_goals, away_goals = [], [], [], []
    h_cum = a_cum = 0.0
    for s in shots:
        xg = xgp.get(int(s["shot_id"]))
        if xg is None:
            xg = float(s["statsbomb_xg"] or 0.0)
        minute = int(s["minute"]) if s["minute"] is not None else 0
        if int(s["team_id"]) == hid:
            h_cum += xg
            home_pts.append({"x": minute, "y": round(h_cum, 2)})
            if s["is_goal"]:
                home_goals.append({"x": minute, "y": round(h_cum, 2), "player": s["player_name"]})
        else:
            a_cum += xg
            away_pts.append({"x": minute, "y": round(a_cum, 2)})
            if s["is_goal"]:
                away_goals.append({"x": minute, "y": round(a_cum, 2), "player": s["player_name"]})

    return {
        "home_name": m["home_name"], "away_name": m["away_name"],
        "home_score": m["home_score"], "away_score": m["away_score"],
        "home_xg": round(h_cum, 2), "away_xg": round(a_cum, 2),
        "home_series": home_pts, "away_series": away_pts,
        "home_goals": home_goals, "away_goals": away_goals,
        "source": "database+artifact" if xgp else "database",
    }


# =============================================================================
# /api/injury-risk   (Model 3: injury risk, XGBoost on workload/recovery)
# =============================================================================

def _injury_response(players: list, source: str) -> dict[str, Any]:
    high = sum(1 for p in players if p["risk_score"] >= 0.67)
    med  = sum(1 for p in players if 0.4 <= p["risk_score"] < 0.67)
    low  = len(players) - high - med
    return {
        "kpi": {
            "high": high, "medium": med, "low": low,
            "avg_score": round(sum(p["risk_score"] for p in players) / len(players), 2),
        },
        "players": players,
        "source":  source,
    }


@app.get("/api/injury-risk")
def injury_risk(team_id: int, season: str | None = None) -> dict[str, Any]:
    _validate_team_id(team_id)
    season_sql, season_params = _season_filter(season)

    m3     = _A.get("m3", {})
    model  = m3.get("xgb") or m3.get("rf")
    scaler = m3.get("scaler")

    FEATURES = [
        "minutes_played", "matches_last_30_days", "minutes_last_30_days",
        "days_since_last_injury", "age_at_match", "sub_minute_flag",
        "xg", "xa", "pressures", "tackles", "carry_distance",
        "interceptions", "clearances",
    ]

    try:
        rows = _query(
            """
            SELECT p.player_name,
                   COALESCE(
                       mode() WITHIN GROUP (ORDER BY pms.starting_position), '-'
                   ) AS position,
                   AVG(pms.minutes_played)                       AS minutes_played,
                   AVG(COALESCE(pmf.matches_last_30_days, 0))    AS matches_last_30_days,
                   AVG(COALESCE(pmf.minutes_last_30_days, 0))    AS minutes_last_30_days,
                   AVG(COALESCE(pmf.days_since_last_injury, -1)) AS days_since_last_injury,
                   MAX(CASE WHEN pms.sub_minute IS NOT NULL THEN 1 ELSE 0 END) AS sub_minute_flag,
                   AVG(pms.xg) AS xg, AVG(pms.xa) AS xa,
                   AVG(pms.pressures) AS pressures, AVG(pms.tackles) AS tackles,
                   AVG(pms.carry_distance) AS carry_distance,
                   AVG(pms.interceptions) AS interceptions,
                   AVG(pms.clearances) AS clearances,
                   AVG(COALESCE(
                       EXTRACT(YEAR FROM AGE(m.match_date, p.date_of_birth))::INT, 25
                   )) AS age_at_match
            FROM player_match_stats pms
            JOIN player_match_features pmf ON pmf.stat_id  = pms.stat_id
            JOIN players p                 ON p.player_id  = pms.player_id
            JOIN matches  m                ON m.match_id   = pms.match_id
            WHERE pms.team_id = %s""" + season_sql + """
            GROUP BY p.player_id, p.player_name
            HAVING COUNT(*) >= 3
            ORDER BY AVG(pms.minutes_played) DESC
            LIMIT 15
            """,
            tuple([team_id] + season_params),
        )
    except Exception as exc:
        logger.warning("/api/injury-risk team=%d query error: %s", team_id, exc)
        _last_errors["injury_risk"] = str(exc)
        rows = []

    if rows and model is not None and scaler is not None:
        X = np.array([[float(r.get(f) or 0) for f in FEATURES] for r in rows])
        probs = model.predict_proba(scaler.transform(X))[:, 1]
        players_out = [
            {
                "player_name":            r["player_name"],
                "position":               r.get("position") or "-",
                "workload_30d":           int(r.get("minutes_last_30_days") or 0),
                "days_since_last_injury": int(r.get("days_since_last_injury") or 0),
                "risk_score":             round(float(prob), 2),
            }
            for r, prob in zip(rows, probs)
        ]
        players_out.sort(key=lambda p: p["risk_score"], reverse=True)
        return _injury_response(players_out, "database+model3")

    logger.warning("/api/injury-risk team=%d: artifact/data unavailable", team_id)
    return {"kpi": {}, "players": [], "source": "fallback"}


# =============================================================================
# /api/win-probability  (static, pre-match) + /api/win-probability-timeline
# (dynamic, in-game) — Model 5's two sub-models.
# =============================================================================

def _m5_model_block() -> dict[str, Any] | None:
    """Held-out accuracy + majority-class baseline for both Model 5 sub-models,
    read from the optimized training metadata so the UI reflects the model that
    is actually serving predictions (independent of the registry being re-run)."""
    meta = _A.get("m5", {}).get("meta_v2")
    if not meta:
        return None
    pre = meta.get("pre_match", {})
    ig  = meta.get("in_game", {})
    return {
        "prematch_accuracy":  pre.get("holdout_2022_accuracy"),
        "prematch_naive":     pre.get("naive_majority_accuracy"),
        "prematch_algorithm": pre.get("winner_algorithm"),
        "prematch_features":  len(pre.get("features", []) or []),
        "ingame_accuracy":    ig.get("holdout_2022_accuracy"),
        "ingame_naive":       ig.get("naive_majority_accuracy"),
        "ingame_algorithm":   ig.get("winner_algorithm"),
        "ingame_features":    len(ig.get("features", []) or []),
    }


@app.get("/api/win-probability-timeline")
def win_probability_timeline(match_id: int, team_id: int) -> dict[str, Any]:
    """The DYNAMIC (in-game) win-probability curve for one team in one match.

    Runs Model 5B (the optimized in-game classifier) on every minute snapshot
    of the match, producing a genuine win/draw/loss probability curve that
    evolves with the live score and chances — not a hand-rolled formula. The
    minute-0 point is seeded from the STATIC pre-match model for the same match,
    so a single chart shows the pre-match call handing off to the live model."""
    _validate_team_id(team_id)
    m5 = _A.get("m5", {})
    ig, scaler, feats, df = (
        m5.get("gbc_ig_v2"), m5.get("scaler_ig_v2"),
        m5.get("feats_ig_v2"), m5.get("df_ig_v2"),
    )
    if any(x is None for x in (ig, scaler, feats, df)):
        return {"series": [], "source": "fallback",
                "note": "in-game model artifacts not loaded"}

    # Match metadata (names, score, which side this team is) for the chart.
    meta = None
    if _db_ok():
        try:
            rows = _query(
                """
                SELECT m.home_team_id, m.away_team_id, m.home_score, m.away_score,
                       th.team_name AS home_name, ta.team_name AS away_name
                FROM matches m
                JOIN teams th ON th.team_id = m.home_team_id
                JOIN teams ta ON ta.team_id = m.away_team_id
                WHERE m.match_id = %s
                """,
                (match_id,),
            )
            meta = rows[0] if rows else None
        except Exception as exc:
            logger.warning("/api/win-probability-timeline meta error: %s", exc)

    try:
        rows = df[(df["match_id"] == match_id) & (df["team_id"] == team_id)]
        rows = rows.sort_values("minute")
        if rows.empty:
            return {"series": [], "source": "fallback",
                    "note": "no in-game snapshots for this match/team"}

        X = scaler.transform(rows[feats].fillna(0).values)
        proba = ig.predict_proba(X)
        cls = list(ig.classes_)
        idx = {c: i for i, c in enumerate(cls)}
        series = []
        for (_, r), p in zip(rows.iterrows(), proba):
            series.append({
                "minute":    int(r["minute"]),
                "win":       round(float(p[idx.get(2, 0)]) * 100, 1),
                "draw":      round(float(p[idx.get(1, 0)]) * 100, 1),
                "loss":      round(float(p[idx.get(0, 0)]) * 100, 1),
                "goal_diff": int(r.get("goal_diff_so_far", 0)),
            })

        # The static pre-match call for this exact match — returned separately
        # (not spliced into the in-game series) so the chart can show it as a
        # reference without an artificial "kickoff dip" between two models.
        pre_point = _m5_prematch_point(match_id, team_id)
    except Exception as exc:
        logger.warning("/api/win-probability-timeline match=%d team=%d error: %s",
                       match_id, team_id, exc)
        _last_errors["win_probability_timeline"] = str(exc)
        return {"series": [], "source": "fallback"}

    out: dict[str, Any] = {"series": series, "match_id": match_id,
                           "team_id": team_id, "source": "artifact",
                           "prematch": pre_point,
                           "model": _m5_model_block()}
    if meta:
        is_home = int(meta["home_team_id"]) == team_id
        out.update({
            "home_name":  meta["home_name"],
            "away_name":  meta["away_name"],
            "home_score": meta["home_score"],
            "away_score": meta["away_score"],
            "team_name":  meta["home_name"] if is_home else meta["away_name"],
            "opp_name":   meta["away_name"] if is_home else meta["home_name"],
            "is_home":    is_home,
        })
    return out


def _m5_prematch_point(match_id: int, team_id: int) -> dict[str, float] | None:
    """The static (v2) pre-match win/draw/loss for one specific (match, team),
    shown as the in-game chart's pre-kickoff reference."""
    m5 = _A.get("m5", {})
    gbc, scaler, feats, df = (
        m5.get("gbc_pre_v2"), m5.get("scaler_pre_v2"),
        m5.get("feats_pre_v2"), m5.get("df_pre_v2"),
    )
    if any(x is None for x in (gbc, scaler, feats, df)):
        return None
    try:
        r = df[(df["match_id"] == match_id) & (df["team_id"] == team_id)]
        if r.empty:
            return None
        p = gbc.predict_proba(scaler.transform(r[feats].fillna(0).values))[0]
        cls = list(gbc.classes_)
        idx = {c: i for i, c in enumerate(cls)}
        return {
            "win":  round(float(p[idx.get(2, 0)]) * 100, 1),
            "draw": round(float(p[idx.get(1, 0)]) * 100, 1),
            "loss": round(float(p[idx.get(0, 0)]) * 100, 1),
        }
    except Exception:
        return None


@app.get("/api/win-probability")
def win_probability(team_id: int, season: str | None = None) -> dict[str, Any]:
    _validate_team_id(team_id)
    season_sql, season_params = _season_filter(season)

    m5 = _A.get("m5", {})

    # Curated subset of the model's features surfaced to the UI as "what the
    # model looks at". The optimized (v2) static model adds opponent-relative
    # form and an Elo rating edge — the headline upgrade over the own-form-only
    # v1 — so those are highlighted here.
    INPUT_LABELS_V2 = {
        "elo_diff":                   "Elo rating edge",
        "season_points_per_game":     "Season points/game",
        "opp_season_points_per_game": "Opponent points/game",
        "avg_xg_last5":               "Avg xG (last 5)",
        "avg_xg_last5_opp":           "Opponent xG (last 5)",
        "goal_diff_last5":            "Goal diff (last 5)",
    }
    INPUT_LABELS_V1 = {
        "avg_xg_last5":       "Avg xG (last 5)",
        "avg_shots_last5":    "Avg shots (last 5)",
        "avg_pass_acc_last5": "Pass accuracy (last 5)",
        "avg_pressures_last5": "Avg pressures (last 5)",
    }
    FEATURES_PRE_V1 = [
        "avg_xg_last5", "avg_shots_last5", "avg_passes_last5",
        "avg_pass_acc_last5", "avg_tackles_last5", "avg_pressures_last5",
        "red_cards_match", "subs_made", "is_home",
    ]

    win = draw = loss = None
    inputs: dict[str, float] = {}
    input_labels = INPUT_LABELS_V2
    source = "fallback"
    model_version = None

    def _predict_avg(df, model, scaler, feats):
        """Average predict_proba over the team's (season-filtered) matches.

        Predicting each match then averaging probabilities — not averaging the
        inputs and predicting once — is the statistically correct way to get
        'the model's typical pre-match call for this team'."""
        rows = df[df["team_id"] == team_id] if "team_id" in df.columns else df
        if _has_season(season) and "season" in rows.columns:
            rows = rows[rows["season"] == season]
        if rows.empty:
            return None, None
        X = rows[feats].fillna(0).values
        proba = model.predict_proba(scaler.transform(X)).mean(axis=0)
        cls = list(model.classes_)
        p_map = {c: proba[i] for i, c in enumerate(cls)}
        return rows, p_map

    # Primary path: the OPTIMIZED v2 static model. Its feature table already
    # carries each historical match's opponent-relative form and pre-match Elo,
    # so we get the stronger model's call without needing an opponent parameter.
    gbc2, sc2, feats2, df2 = (
        m5.get("gbc_pre_v2"), m5.get("scaler_pre_v2"),
        m5.get("feats_pre_v2"), m5.get("df_pre_v2"),
    )
    if all(x is not None for x in (gbc2, sc2, feats2, df2)):
        try:
            rows, p_map = _predict_avg(df2, gbc2, sc2, feats2)
            if rows is not None:
                win  = round(float(p_map.get(2, 0)) * 100, 1)
                draw = round(float(p_map.get(1, 0)) * 100, 1)
                loss = round(float(p_map.get(0, 0)) * 100, 1)
                source = "artifact"
                model_version = "v2-optimized"
                means = rows[feats2].fillna(0).mean()
                inputs = {k: round(float(means.get(k, 0)), 2) for k in INPUT_LABELS_V2}
                input_labels = INPUT_LABELS_V2
                logger.info(
                    "/api/win-probability team=%d season=%s [v2]: win=%.1f draw=%.1f "
                    "loss=%.1f over %d matches", team_id, season, win, draw, loss, len(rows),
                )
        except Exception as exc:
            logger.warning("/api/win-probability team=%d v2 error: %s", team_id, exc)
            _last_errors["win_probability"] = str(exc)

    # Fallback: the legacy v1 own-form-only model.
    if win is None:
        gbc, scaler, df_pre = m5.get("gbc_pre"), m5.get("scaler_pre"), m5.get("df_pre")
        if all(x is not None for x in (gbc, scaler, df_pre)):
            try:
                rows, p_map = _predict_avg(df_pre, gbc, scaler, FEATURES_PRE_V1)
                if rows is not None:
                    win  = round(float(p_map.get(2, 0)) * 100, 1)
                    draw = round(float(p_map.get(1, 0)) * 100, 1)
                    loss = round(float(p_map.get(0, 0)) * 100, 1)
                    source = "artifact"
                    model_version = "v1"
                    means = rows[FEATURES_PRE_V1].fillna(0).mean()
                    inputs = {k: round(float(means.get(k, 0)), 2) for k in INPUT_LABELS_V1}
                    input_labels = INPUT_LABELS_V1
            except Exception as exc:
                logger.warning("/api/win-probability team=%d v1 error: %s", team_id, exc)
                _last_errors["win_probability"] = str(exc)

    # Fallback: no model/feature artifact — use the team's actual outcome rates.
    if win is None and _db_ok():
        try:
            raw = _query(
                """
                SELECT AVG(CASE WHEN pms.result='win'  THEN 1.0 ELSE 0.0 END) AS win_rate,
                       AVG(CASE WHEN pms.result='draw' THEN 1.0 ELSE 0.0 END) AS draw_rate,
                       AVG(CASE WHEN pms.result='loss' THEN 1.0 ELSE 0.0 END) AS loss_rate
                FROM player_match_stats pms
                JOIN matches m ON m.match_id = pms.match_id
                WHERE pms.team_id = %s AND pms.result IS NOT NULL""" + season_sql + """
                """,
                tuple([team_id] + season_params),
            )
            rates = raw[0] if raw else {}
            if rates and rates.get("win_rate") is not None:
                win  = round(float(rates.get("win_rate")  or 0) * 100, 1)
                draw = round(float(rates.get("draw_rate") or 0) * 100, 1)
                loss = round(float(rates.get("loss_rate") or 0) * 100, 1)
                source = "database"
        except Exception as exc:
            logger.warning("/api/win-probability team=%d DB error: %s", team_id, exc)
            _last_errors["win_probability"] = str(exc)

    if win is None:
        offset = team_id % 4
        win    = 60.0 + offset * 2
        draw   = 24.0 - (team_id % 3)
        loss   = round(100 - win - draw, 1)
        source = "fallback"
        logger.info("/api/win-probability team=%d: using fallback", team_id)

    # Actual outcome record for the team over the (optionally season-filtered)
    # set of matches — concrete context alongside the model's averaged
    # probabilities. The dataset is historical, so there is no "next match".
    record = _team_record(team_id, season)

    return {
        "headline": {"win": win, "draw": draw, "loss": loss},
        "record": record,
        "inputs": [
            {"key": k, "label": input_labels[k], "value": inputs[k]}
            for k in input_labels if k in inputs
        ],
        "model": _m5_model_block(),
        "model_version": model_version,
        "season": season if _has_season(season) else None,
        "source": source,
    }


# =============================================================================
# Internal helpers
# =============================================================================

def _build_radar(leader: dict) -> dict:
    return {
        "labels": ["xG", "xA", "Passing", "Key Passes", "Dribbles", "Shots"],
        "values": [
            min(100, round(float(leader.get("xg_per_90") or 0) * 100, 1)),
            min(100, round(float(leader.get("xa_per_90") or 0) * 100, 1)),
            round(float(leader.get("pass_completion") or 0), 1),
            min(100, round(float(leader.get("key_passes") or 0) * 25, 1)),
            min(100, round(float(leader.get("dribbles") or 0) * 20, 1)),
            min(100, round(float(leader.get("shots") or 0) * 20, 1)),
        ],
    }


def _team_record(team_id: int, season: str | None) -> dict[str, int]:
    """Actual W/D/L record for a team over the (optionally season-filtered)
    matches with a final score. Returns zeros on any error."""
    season_sql, season_params = _season_filter(season)
    try:
        rows = _query(
            """
            SELECT
              SUM(CASE WHEN (m.home_team_id = %s AND m.home_score > m.away_score)
                         OR (m.away_team_id = %s AND m.away_score > m.home_score)
                       THEN 1 ELSE 0 END) AS wins,
              SUM(CASE WHEN m.home_score = m.away_score THEN 1 ELSE 0 END) AS draws,
              SUM(CASE WHEN (m.home_team_id = %s AND m.home_score < m.away_score)
                         OR (m.away_team_id = %s AND m.away_score < m.home_score)
                       THEN 1 ELSE 0 END) AS losses,
              COUNT(*) AS played
            FROM matches m
            WHERE (m.home_team_id = %s OR m.away_team_id = %s)
              AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL""" + season_sql + """
            """,
            tuple([team_id] * 6 + season_params),
        )
        r = rows[0] if rows else {}
        return {
            "wins":   int(r.get("wins")   or 0),
            "draws":  int(r.get("draws")  or 0),
            "losses": int(r.get("losses") or 0),
            "played": int(r.get("played") or 0),
        }
    except Exception as exc:
        logger.debug("_team_record team=%d error: %s", team_id, exc)
        return {"wins": 0, "draws": 0, "losses": 0, "played": 0}


# =============================================================================
# Fallbacks
# =============================================================================

def _fallback_player(team_id: int) -> dict[str, Any]:
    roster = {
        1: ["Kevin De Bruyne", "Erling Haaland", "Rodri", "Phil Foden", "Bernardo Silva"],
        2: ["Martin Odegaard", "Bukayo Saka", "Declan Rice", "Kai Havertz", "William Saliba"],
        3: ["Mohamed Salah", "Virgil van Dijk", "Alexis Mac Allister", "Trent Alexander-Arnold", "Darwin Nunez"],
        4: ["Cole Palmer", "Enzo Fernandez", "Reece James", "Nicolas Jackson", "Levi Colwill"],
    }.get(team_id, ["Player A", "Player B", "Player C", "Player D", "Player E"])

    players = [
        {
            "player_name":     name,
            "position":        "Midfielder" if i < 3 else "Forward",
            "player_type":     "Creator" if i == 0 else "Box-to-Box",
            "matches":         30 - i,
            "minutes":         2400 - (i * 120),
            "xg_per_90":       round(0.25 + i * 0.03, 2),
            "xa_per_90":       round(0.30 + i * 0.04, 2),
            "pass_completion": round(84 + i, 1),
            "key_passes":      round(2.1 + i * 0.3, 1),
            "dribbles":        round(1.8 + i * 0.2, 1),
            "shots":           round(2.3 + i * 0.25, 1),
        }
        for i, name in enumerate(roster)
    ]
    leader = players[0]
    return {
        "leader":  leader,
        "radar":   _build_radar(leader),
        "players": players,
        "source":  "fallback",
    }

