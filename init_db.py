"""
One-time database initialisation. Run before any pipeline.

Usage:
    python init_db.py
"""

import logging
import psycopg2
from config.settings import DB_DSN

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def init_db(conn):
    with open("schema.sql") as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info("Schema created / verified.")


def verify(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [r[0] for r in cur.fetchall()]
    logger.info("Tables: %s", tables)
    missing = {
        "matches", "pass_network_edges", "match_minute_snapshots",
        "player_match_stats", "player_match_features", "injuries",
        "players", "teams", "stadiums", "shots", "model_registry",
        "weather",
    } - set(tables)
    if missing:
        logger.error("Missing tables: %s", missing)
    else:
        logger.info("All expected tables present.")


if __name__ == "__main__":
    logger.info("Connecting to %s ...", DB_DSN.split("@")[-1])
    conn = psycopg2.connect(DB_DSN)
    init_db(conn)
    verify(conn)
    conn.close()
    logger.info("Done.")