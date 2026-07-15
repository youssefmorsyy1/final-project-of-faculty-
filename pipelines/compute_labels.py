"""
pipelines/compute_labels.py

Post-ingestion SQL passes that compute derived ML columns.

Schema change
-------------
Computed columns (matches_last_30_days, minutes_last_30_days,
days_since_last_injury, is_injured_next_30d) have moved from
player_match_stats to the separate player_match_features table.

This pipeline now:
  1. Ensures a player_match_features row exists for every player_match_stats
     row (INSERT … ON CONFLICT DO NOTHING).
  2. UPDATEs player_match_features for the three computed label passes.

Workload fix
------------
The join now also filters m_inner.match_id != m_outer.match_id to prevent
same-match rows for other players on the same team from being counted inside
the 30-day window. The original stat_id != stat_id guard was insufficient
because two players on the same team in the same match share a match_date,
so both rows could fall inside each other's windows.
"""

import logging
from load.postgres import connect
from config.settings import DB_DSN

logger = logging.getLogger(__name__)


def _ensure_feature_rows(conn):
    """
    Insert a player_match_features row for every player_match_stats row
    that doesn't have one yet.  Safe to re-run (ON CONFLICT DO NOTHING).
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO player_match_features (stat_id, player_id, match_id)
            SELECT stat_id, player_id, match_id
            FROM   player_match_stats
            ON CONFLICT (player_id, match_id) DO NOTHING
        """)
        inserted = cur.rowcount
    conn.commit()
    if inserted:
        logger.info("player_match_features: %d new rows scaffolded", inserted)


def compute_injury_label(conn):
    """Set is_injured_next_30d = TRUE where an injury follows within 30 days."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE player_match_features pmf
            SET    is_injured_next_30d = TRUE
            FROM   player_match_stats pms
            JOIN   matches m ON m.match_id = pms.match_id
            WHERE  pmf.stat_id = pms.stat_id
              AND  EXISTS (
                SELECT 1
                FROM injuries i
                WHERE i.player_id    = pms.player_id
                  AND i.injury_date >= m.match_date
                  AND i.injury_date <= m.match_date + INTERVAL '30 days'
              );
        """)
        updated = cur.rowcount
    conn.commit()
    logger.info("is_injured_next_30d: %d rows set to TRUE", updated)


def compute_days_since_last_injury(conn):
    """Days between each match and the player's most recent prior return_date."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE player_match_features pmf
            SET    days_since_last_injury = sub.days_since
            FROM (
                SELECT
                    pms2.stat_id,
                    (m2.match_date - MAX(i.return_date))::INT AS days_since
                FROM player_match_stats pms2
                JOIN matches  m2 ON m2.match_id  = pms2.match_id
                JOIN injuries i  ON i.player_id  = pms2.player_id
                WHERE i.return_date < m2.match_date
                GROUP BY pms2.stat_id, m2.match_date
            ) sub
            WHERE pmf.stat_id = sub.stat_id
        """)
        updated = cur.rowcount
    conn.commit()
    logger.info("days_since_last_injury: %d rows updated", updated)


def compute_workload(conn):
    """
    matches_last_30_days and minutes_last_30_days for each player-match row.

    The join filters on both stat_id and match_id to ensure only prior matches
    for the same player fall inside the 30-day window. Filtering by stat_id
    alone was insufficient: two players on the same team share a match_date,
    so a team-mate's row could satisfy the date range check for the outer row.
    Filtering by match_id != match_id directly excludes same-match rows.
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE player_match_features pmf
            SET
                matches_last_30_days = sub.match_count,
                minutes_last_30_days = sub.minute_sum
            FROM (
                SELECT
                    pms_outer.stat_id,
                    COUNT(pms_inner.stat_id)                        AS match_count,
                    COALESCE(SUM(pms_inner.minutes_played), 0)::INT AS minute_sum
                FROM player_match_stats pms_outer
                JOIN matches m_outer ON m_outer.match_id = pms_outer.match_id
                LEFT JOIN (
                    player_match_stats pms_inner
                    JOIN matches m_inner ON m_inner.match_id = pms_inner.match_id
                ) ON  pms_inner.player_id  = pms_outer.player_id
                  AND pms_inner.match_id  != pms_outer.match_id
                  AND m_inner.match_date  >= m_outer.match_date - INTERVAL '30 days'
                  AND m_inner.match_date   < m_outer.match_date
                GROUP BY pms_outer.stat_id
            ) sub
            WHERE pmf.stat_id = sub.stat_id
        """)
        updated = cur.rowcount
    conn.commit()
    logger.info("Workload features: %d rows updated", updated)


def run(conn=None):
    if conn is None:
        conn = connect(DB_DSN)

    _ensure_feature_rows(conn)
    compute_workload(conn)
    compute_days_since_last_injury(conn)
    compute_injury_label(conn)

    logger.info("compute_labels complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
