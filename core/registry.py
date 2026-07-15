"""
core/registry.py — hybrid model persistence helpers.

The project keeps trained sklearn/xgboost binaries as .pkl files on disk under
artifacts/<model_key>/ (the industry-standard place for model binaries). What
lives only in files today — *which* models exist, *how* they were trained and
*how well* they scored, plus the derived tabular prediction outputs — is what
this module persists into PostgreSQL so it becomes queryable by the API/EDA.

Two helpers, both psycopg2-only (the project does not use SQLAlchemy):

  register_model(conn, ...)   upsert one row into model_registry.
  replace_table(conn, t, df)  (re)create a typed table from a DataFrame and
                              bulk-load it (TRUNCATE-and-replace semantics).

Both commit on success and are safe to call from main.py after each model's
run(). Callers should wrap in try/except so a persistence hiccup never aborts
training (the .pkl/.parquet artifacts are still written regardless).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from psycopg2.extras import Json, execute_values

logger = logging.getLogger("registry")


# ---------------------------------------------------------------------------
# model_registry
# ---------------------------------------------------------------------------

def register_model(
    conn,
    *,
    model_key: str,
    version: str,
    metrics: dict[str, Any] | None = None,
    features: Sequence[str] | None = None,
    display_name: str | None = None,
    task: str | None = None,
    algorithm: str | None = None,
    target: str | None = None,
    n_train_rows: int | None = None,
    sklearn_version: str | None = None,
    artifact_path: str | None = None,
    prediction_table: str | None = None,
) -> None:
    """Upsert a single model into model_registry (keyed by model_key+version)."""
    if sklearn_version is None:
        try:
            import sklearn
            sklearn_version = sklearn.__version__
        except Exception:
            sklearn_version = None

    sql = """
        INSERT INTO model_registry (
            model_key, version, display_name, task, algorithm, target,
            features, metrics, n_train_rows, sklearn_version, artifact_path,
            prediction_table, trained_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now()
        )
        ON CONFLICT (model_key, version) DO UPDATE SET
            display_name     = EXCLUDED.display_name,
            task             = EXCLUDED.task,
            algorithm        = EXCLUDED.algorithm,
            target           = EXCLUDED.target,
            features         = EXCLUDED.features,
            metrics          = EXCLUDED.metrics,
            n_train_rows     = EXCLUDED.n_train_rows,
            sklearn_version  = EXCLUDED.sklearn_version,
            artifact_path    = EXCLUDED.artifact_path,
            prediction_table = EXCLUDED.prediction_table,
            trained_at       = now();
    """
    params = (
        model_key,
        version,
        display_name,
        task,
        algorithm,
        target,
        Json(list(features)) if features is not None else None,
        Json(_jsonable(metrics)) if metrics is not None else None,
        int(n_train_rows) if n_train_rows is not None else None,
        sklearn_version,
        artifact_path,
        prediction_table,
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()
    logger.info("model_registry <- %s v%s (%d metrics)",
                model_key, version, len(metrics or {}))


# ---------------------------------------------------------------------------
# Derived prediction tables
# ---------------------------------------------------------------------------

# pandas dtype kind -> PostgreSQL column type
_PG_TYPES = {
    "b": "BOOLEAN",
    "i": "BIGINT",
    "u": "BIGINT",
    "f": "DOUBLE PRECISION",
    "M": "TIMESTAMPTZ",
}


def _pg_type(series: pd.Series) -> str:
    return _PG_TYPES.get(series.dtype.kind, "TEXT")


def _ident(name: str) -> str:
    """Sanitise a DataFrame column into a safe, lowercase SQL identifier."""
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(name))
    safe = safe.strip("_").lower() or "col"
    if safe[0].isdigit():
        safe = f"c_{safe}"
    return safe


def replace_table(conn, table: str, df: pd.DataFrame,
                  columns: Iterable[str] | None = None) -> int:
    """
    Create (or replace) `table` to match `df`'s schema and bulk-load every row.

    Columns are created from the DataFrame dtypes (BIGINT / DOUBLE PRECISION /
    BOOLEAN / TIMESTAMPTZ / TEXT). Non-scalar cells (lists/dicts) are stored as
    JSON text. Returns the number of rows written.
    """
    if df is None or df.empty:
        logger.warning("replace_table: %s skipped (empty DataFrame)", table)
        return 0

    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]

    table = _ident(table)
    col_names = [_ident(c) for c in df.columns]
    col_defs = ", ".join(f"{cn} {_pg_type(df[orig])}"
                         for cn, orig in zip(col_names, df.columns))

    rows = [tuple(_cell(v) for v in rec) for rec in df.itertuples(index=False, name=None)]

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"CREATE TABLE {table} ({col_defs})")
        execute_values(
            cur,
            f"INSERT INTO {table} ({', '.join(col_names)}) VALUES %s",
            rows,
            page_size=1000,
        )
    conn.commit()
    logger.info("replace_table: %s <- %d rows x %d cols",
                table, len(rows), len(col_names))
    return len(rows)


# ---------------------------------------------------------------------------
# Coercion helpers (numpy/pandas -> JSON-safe Python natives)
# ---------------------------------------------------------------------------

def _cell(v: Any) -> Any:
    """Coerce one DataFrame cell into a psycopg2-friendly Python value."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, dict, tuple, np.ndarray)):
        return json.dumps(v.tolist() if isinstance(v, np.ndarray) else v, default=str)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays inside metrics dicts to natives."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
