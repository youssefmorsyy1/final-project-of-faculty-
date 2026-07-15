"""
Pipeline orchestrator. Run init_db.py once before first use.

Data sources: StatsBomb event data (primary), Transfermarkt CSVs (injuries),
and Open-Meteo historical weather (one row per match, by stadium coordinates).

Flags:
  --train         Train ML models after ingestion
  --skip-ingest   Skip ingestion, run only training
  --skip-weather  Skip the (network-bound) weather ingestion step
  --workers N     Worker processes for StatsBomb ingestion (default: CPU count - 1)

Each worker process imports numpy/pandas, which load BLAS and otherwise try
to use one thread per core *inside every worker*. With N worker processes
that is N x cores BLAS threads fighting over RAM at once -- the proximate
cause of an "OpenBLAS error: Memory allocation still failed" crash on
memory-constrained machines. Pin BLAS libraries to 1 thread per worker
before numpy is imported anywhere (env vars must be set pre-import).
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import logging
import psycopg2

from config.settings import DB_DSN, DATA_ROOT
from core.caches import TeamCache, PlayerCache
from extract import statsbomb_local as sb
from pipelines.ingest_statsbomb import run as run_statsbomb
from pipelines.extract_shots   import run as run_shots
from pipelines.ingest_injuries import run as run_injuries
from pipelines.ingest_weather  import run as run_weather
from pipelines.compute_labels  import run as run_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_SEP = "=" * 60


def parse_args():
    p = argparse.ArgumentParser(description="Soccer Analytics ML pipeline")
    p.add_argument("--train",        action="store_true")
    p.add_argument("--skip-ingest",  action="store_true")
    p.add_argument("--skip-weather", action="store_true")
    p.add_argument("--workers",      type=int, default=None)
    return p.parse_args()


def _persist_to_db(conn, registry, result, label):
    """Hybrid persistence: write each model's registry row + derived prediction
    tables into PostgreSQL. The .pkl/.parquet artifacts on disk are unaffected.
    Wrapped so a DB hiccup never aborts training (artifacts are already saved)."""
    if not isinstance(result, dict) or "_registry" not in result:
        return
    try:
        registry.register_model(conn, **result["_registry"])
        for table, df in (result.get("_predictions") or {}).items():
            registry.replace_table(conn, table, df)
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        logger.warning("Registry persistence failed for %s: %s", label, exc)


def main():
    args = parse_args()
    sb.set_root(DATA_ROOT)
    conn = psycopg2.connect(DB_DSN)

    if not args.skip_ingest:
        logger.info("%s\nStep 1: StatsBomb ingestion", _SEP)
        kwargs = {"workers": args.workers} if args.workers is not None else {}
        run_statsbomb(conn, TeamCache(conn), PlayerCache(conn), **kwargs)

        logger.info("%s\nStep 2: Shot extraction (xG inputs)", _SEP)
        run_shots(conn)

        logger.info("%s\nStep 3: Injuries ingestion (Transfermarkt)", _SEP)
        run_injuries(conn)

        if args.skip_weather:
            logger.info("%s\nStep 4: Weather ingestion -- skipped (--skip-weather)", _SEP)
        else:
            logger.info("%s\nStep 4: Weather ingestion (Open-Meteo)", _SEP)
            run_weather(conn)

        logger.info("%s\nStep 5: Computing labels (workload + injury)", _SEP)
        run_labels(conn)

    if args.train:
        logger.info("%s\nStep 6: Training ML models", _SEP)

        from models.model1_player_clustering import run as train1
        from models.model2_team_cohesion     import run as train2
        from models.model3_injury_risk       import run as train3
        from models.model5_win_probability   import run as train5
        from models.model_xg                 import run as train_xg
        from core import registry

        for label, fn in [
            ("Model 1: Player Efficiency & Style Profiling", train1),
            ("Model 2: Team Cohesion / Pass Networks",        train2),
            ("Model 3: Injury Risk",                          train3),
            ("Model 5: Win Probability",                      train5),
            ("Model xG: Expected Goals",                      train_xg),
        ]:
            logger.info(label)
            result = fn(conn)
            _persist_to_db(conn, registry, result, label)

    conn.close()
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
