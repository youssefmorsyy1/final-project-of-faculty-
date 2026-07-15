"""
pipelines/ingest_pass_network.py

Pass network edges are extracted inside ingest_statsbomb.py during the
same event-loading pass, so running the full pipeline via main.py no
longer needs this file.

This stub exists as a recovery / backfill tool: if you need to (re)populate
pass_network_edges for matches ingested before that fix, run this script
directly.  It reads each event file exactly once and writes edges for any
match that currently has no edge rows.
"""

import logging
from collections import defaultdict

from psycopg2.extras import execute_values

from config.settings import DB_DSN
from extract import statsbomb_local as sb
from core.caches import TeamCache, PlayerCache
from load.postgres import connect

logger = logging.getLogger(__name__)


def _extract_edges(events, pg_match_id: int,
                   player_cache, team_cache) -> list:
    """Extract completed-pass edges from a match events DataFrame."""
    from transform.features import extract_type_col, extract_player_id_col, extract_team_id_col

    type_col      = extract_type_col(events)
    player_id_col = extract_player_id_col(events)
    team_id_col   = extract_team_id_col(events)

    is_pass = type_col == "Pass"
    if not is_pass.any():
        return []

    acc = defaultdict(lambda: {"n": 0, "xs": 0.0, "ys": 0.0, "xe": 0.0, "ye": 0.0})

    pass_rows = events.loc[is_pass]
    for (_, row), pid, tid in zip(pass_rows.iterrows(),
                                   player_id_col[is_pass],
                                   team_id_col[is_pass]):
        pass_data = row.get("pass") or {}
        if pass_data.get("outcome") is not None:
            continue
        recip = pass_data.get("recipient")
        if not isinstance(recip, dict):
            continue
        rid = recip.get("id")
        if not (pid and rid and tid):
            continue

        try:
            pg_pid = player_cache.resolve(int(pid))
            pg_rid = player_cache.resolve(int(rid))
            pg_tid = team_cache.resolve(int(tid))
        except KeyError:
            continue

        key = (pg_tid, pg_pid, pg_rid)
        loc_s = row.get("location") or []
        loc_e = pass_data.get("end_location") or []
        a = acc[key]
        a["n"] += 1
        if len(loc_s) >= 2:
            a["xs"] += loc_s[0]; a["ys"] += loc_s[1]
        if len(loc_e) >= 2:
            a["xe"] += loc_e[0]; a["ye"] += loc_e[1]

    rows = []
    for (pg_tid, pg_pid, pg_rid), a in acc.items():
        n = a["n"]
        rows.append((
            pg_match_id, pg_tid, pg_pid, pg_rid, n,
            a["xs"] / n, a["ys"] / n,
            a["xe"] / n, a["ye"] / n,
        ))
    return rows


def run(conn=None):
    """Backfill pass_network_edges for matches that have none."""
    if conn is None:
        conn = connect(DB_DSN)

    team_cache   = TeamCache(conn)
    player_cache = PlayerCache(conn)

    # FIX: was m.statsbomb_match_id — column is named sb_match_id in the schema.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.match_id, m.sb_match_id
            FROM   matches m
            WHERE  NOT EXISTS (
                SELECT 1 FROM pass_network_edges e
                WHERE  e.match_id = m.match_id
            )
        """)
        pending = cur.fetchall()

    logger.info("Pass network backfill: %d matches without edges", len(pending))
    if not pending:
        logger.info("Nothing to do.")
        return

    inserted_total = 0
    for pg_match_id, sb_match_id in pending:
        try:
            events = sb.events(sb_match_id)
        except Exception as exc:
            logger.error("Could not load events for match %d: %s", sb_match_id, exc)
            continue

        edge_rows = _extract_edges(events, pg_match_id, player_cache, team_cache)
        if not edge_rows:
            continue

        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO pass_network_edges (
                    match_id, team_id, passer_id, receiver_id,
                    pass_count,
                    avg_x_start, avg_y_start, avg_x_end, avg_y_end
                ) VALUES %s
                ON CONFLICT DO NOTHING
            """, edge_rows, page_size=500)
        conn.commit()
        inserted_total += len(edge_rows)

    logger.info("Backfill complete: %d edges inserted", inserted_total)


if __name__ == "__main__":
    from extract import statsbomb_local as sb_mod
    from config.settings import DATA_ROOT
    logging.basicConfig(level=logging.INFO)
    sb_mod.set_root(DATA_ROOT)
    run()