"""
pipelines/backfill_starting_positions.py

One-time backfill: populate player_match_stats.starting_position
for all rows where it is currently NULL.

Reads lineup JSON files from DATA_ROOT/lineups/{sb_match_id}.json,
extracts each player's dominant starting position, and writes it back.

Run once after applying the schema change:
    python -m pipelines.backfill_starting_positions
"""

import logging
from psycopg2.extras import execute_batch

from config.settings import DB_DSN, DATA_ROOT
from extract import statsbomb_local as sb
from transform.features import extract_starting_positions
from load.postgres import connect

logger = logging.getLogger(__name__)


def run(conn=None):
    if conn is None:
        conn = connect(DB_DSN)

    sb.set_root(DATA_ROOT)

    # Fetch all (pg_match_id, sb_match_id) pairs that have at least one
    # NULL starting_position row — so we skip already-complete matches.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT m.match_id, m.sb_match_id
            FROM   matches m
            JOIN   player_match_stats pms ON pms.match_id = m.match_id
            WHERE  pms.starting_position IS NULL
            ORDER  BY m.match_id
        """)
        pending = cur.fetchall()

    logger.info("Backfill starting_position: %d matches to process", len(pending))
    if not pending:
        logger.info("Nothing to do — all rows already have starting_position set.")
        return

    total_updated = 0
    failed        = 0

    for pg_match_id, sb_match_id in pending:
        try:
            lineups_df = sb.lineups(sb_match_id)
        except Exception as exc:
            logger.error("Could not load lineups for sb_match_id=%d: %s", sb_match_id, exc)
            failed += 1
            continue

        pos_map = extract_starting_positions(lineups_df)
        if not pos_map:
            logger.warning("Empty position map for sb_match_id=%d", sb_match_id)
            continue

        # pos_map is {sb_player_id -> position_name}
        # We need to translate sb_player_id -> player_id (pg internal id)
        # for the players that appear in this match.
        sb_ids = list(pos_map.keys())

        with conn.cursor() as cur:
            cur.execute("""
                SELECT sb_player_id, player_id
                FROM   players
                WHERE  sb_player_id = ANY(%s)
            """, (sb_ids,))
            sb_to_pg = {row[0]: row[1] for row in cur.fetchall()}

        # Build update rows: (position_name, player_id, match_id)
        update_rows = []
        for sb_pid, pos_name in pos_map.items():
            pg_pid = sb_to_pg.get(sb_pid)
            if pg_pid is None:
                continue
            update_rows.append((pos_name, pg_pid, pg_match_id))

        if not update_rows:
            continue

        with conn.cursor() as cur:
            execute_batch(cur, """
                UPDATE player_match_stats
                SET    starting_position = %s
                WHERE  player_id = %s
                  AND  match_id  = %s
                  AND  starting_position IS NULL
            """, update_rows, page_size=500)
            total_updated += cur.rowcount

        conn.commit()

        if pg_match_id % 100 == 0:
            logger.info("  Progress: processed match_id=%d | updated so far: %d",
                        pg_match_id, total_updated)

    logger.info(
        "Backfill complete: %d rows updated | %d matches failed to load lineups",
        total_updated, failed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()