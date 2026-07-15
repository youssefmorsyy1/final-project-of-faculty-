"""
utils/validate_schema.py

Validate that the live PostgreSQL schema matches the expected DDL.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

_TYPE_ALIASES = {
    "varchar":           "text",
    "character varying": "text",
    "int4":              "integer",
    "int8":              "bigint",
    "float4":            "real",
    "float8":            "double precision",
    "numeric":           "double precision",
    "bool":              "boolean",
}


def _norm(t: str) -> str:
    return _TYPE_ALIASES.get(t.lower(), t.lower())


def validate_schema(conn) -> Tuple[bool, List[str], List[str]]:
    warnings: List[str] = []
    errors:   List[str] = []

    # Every table and column that must exist, with expected PG type.
    expected = {
        "teams": {
            "team_id":    "integer",
            "team_name":  "text",
            "sb_team_id": "integer",
        },
        "players": {
            "player_id":     "integer",
            "player_name":   "text",
            "norm_name":     "text",
            "sb_player_id":  "integer",
            "tm_player_id":  "integer",
            "nationality":   "text",
            "position":      "text",
            "date_of_birth": "date",
        },
        "stadiums": {
            "stadium_id":   "integer",
            "stadium_name": "text",
        },
        "injuries": {
            "injury_id":     "integer",
            "player_id":     "integer",
            "injury_type":   "text",
            "injury_date":   "date",
            "return_date":   "date",
            "matches_missed":"integer",
            "season":        "text",
        },
        "matches": {
            "match_id":     "integer",
            "sb_match_id":  "integer",
            "match_date":   "date",
            "home_team_id": "integer",
            "away_team_id": "integer",
            "home_score":   "integer",
            "away_score":   "integer",
            "competition":  "text",
            "season":       "text",
            "stadium_id":   "integer",
        },
        "player_match_stats": {
            "stat_id":           "integer",
            "player_id":         "integer",
            "match_id":          "integer",
            "team_id":           "integer",
            "result":            "text",
            "goals":             "integer",
            "assists":           "integer",
            "shots":             "integer",
            "xg":                "double precision",
            "xa":                "double precision",
            "key_passes":        "integer",
            "passes_attempted":  "integer",
            "passes_completed":  "integer",
            "pass_accuracy":     "double precision",
            "progressive_passes":"integer",
            "carry_distance":    "double precision",
            "progressive_carries":"integer",
            "dribbles_completed":"integer",
            "tackles":           "integer",
            "interceptions":     "integer",
            "clearances":        "integer",
            "pressures":         "integer",
            "yellow_cards":      "integer",
            "red_cards":         "integer",
            "minutes_played":    "integer",
            "starting_position":  "text",
            "sub_minute":        "integer",
        },
        "player_match_features": {
            "feature_id":            "integer",
            "stat_id":               "integer",
            "player_id":             "integer",
            "match_id":              "integer",
            "matches_last_30_days":  "integer",
            "minutes_last_30_days":  "integer",
            "days_since_last_injury":"integer",
            "is_injured_next_30d":   "boolean",
        },
        "shots": {
            "shot_id":             "integer",
            "sb_event_id":         "text",
            "match_id":            "integer",
            "player_id":           "integer",
            "team_id":             "integer",
            "minute":              "integer",
            "x":                   "double precision",
            "y":                   "double precision",
            "distance":            "double precision",
            "angle":               "double precision",
            "body_part":           "text",
            "shot_type":           "text",
            "technique":           "text",
            "play_pattern":        "text",
            "under_pressure":      "boolean",
            "first_time":          "boolean",
            "defenders_in_cone":   "integer",
            "dist_to_nearest_def": "double precision",
            "gk_dist_to_goal":     "double precision",
            "gk_dist_to_shot":     "double precision",
            "statsbomb_xg":        "double precision",
            "is_goal":             "boolean",
        },
        "match_minute_snapshots": {
            "snapshot_id":       "integer",
            "match_id":          "integer",
            "team_id":           "integer",
            "minute":            "integer",
            "goals_so_far":      "integer",
            "xg_so_far":         "double precision",
            "shots_so_far":      "integer",
            "passes_so_far":     "integer",
            "pass_acc_so_far":   "double precision",
            "pressures_so_far":  "integer",
            "red_cards_so_far":  "integer",
        },
        "pass_network_edges": {
            "edge_id":     "integer",
            "match_id":    "integer",
            "team_id":     "integer",
            "passer_id":   "integer",
            "receiver_id": "integer",
            "pass_count":  "integer",
            "avg_x_start": "double precision",
            "avg_y_start": "double precision",
            "avg_x_end":   "double precision",
            "avg_y_end":   "double precision",
        },
    }

    # Columns / tables that were removed and should no longer exist anywhere.
    stale = {
        "teams":              {"country"},
        "matches":            {"stadium_name", "stadium_lat", "stadium_lng"},
        "stadiums":           {"stadium_lat", "stadium_lng"},
        "player_match_stats": {"weather_id"},
    }

    # Tables that were removed entirely (weather / Open-Meteo).
    stale_tables = {"weather"}

    # UNIQUE constraints: table -> minimum expected count.
    unique_constraints = {
        "player_match_stats":    1,   # (player_id, match_id)
        "player_match_features": 2,   # (stat_id) + (player_id, match_id)
        "injuries":              1,   # (player_id, injury_date, injury_type)
        "pass_network_edges":    1,   # (match_id, team_id, passer_id, receiver_id)
        "match_minute_snapshots": 1,
        "shots":                 1,   # (sb_event_id)
    }

    try:
        with conn.cursor() as cur:

            # ------------------------------------------------------------------
            # 1. Column presence and type
            # ------------------------------------------------------------------
            for table, cols in expected.items():
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                """, (table,))
                existing = {r[0]: _norm(r[1]) for r in cur.fetchall()}

                if not existing:
                    errors.append(f"Table '{table}' does not exist")
                    continue

                for col, exp_type in cols.items():
                    if col not in existing:
                        errors.append(f"{table}.{col} missing")
                    elif existing[col] != _norm(exp_type):
                        warnings.append(
                            f"{table}.{col}: expected {exp_type}, got {existing[col]}"
                        )

            # ------------------------------------------------------------------
            # 2. Stale columns that should have been dropped
            # ------------------------------------------------------------------
            for table, cols in stale.items():
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                """, (table,))
                present = {r[0] for r in cur.fetchall()}
                for col in cols & present:
                    warnings.append(f"{table}.{col} should have been dropped")

            # Removed tables that should no longer exist.
            for table in stale_tables:
                cur.execute("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                """, (table,))
                if cur.fetchone()[0]:
                    warnings.append(f"table '{table}' should have been dropped")

            # ------------------------------------------------------------------
            # 3. UNIQUE constraints
            # ------------------------------------------------------------------
            for table, min_count in unique_constraints.items():
                cur.execute("""
                    SELECT COUNT(*) FROM information_schema.table_constraints
                    WHERE table_schema = 'public'
                      AND table_name = %s
                      AND constraint_type = 'UNIQUE'
                """, (table,))
                count = cur.fetchone()[0]
                if count < min_count:
                    errors.append(
                        f"{table}: expected >= {min_count} UNIQUE constraint(s), found {count}"
                    )

            # ------------------------------------------------------------------
            # 4. Referential integrity (orphan row counts)
            # ------------------------------------------------------------------
            orphan_checks = [
                ("player_match_stats",    "player_id", "players",            "player_id"),
                ("player_match_stats",    "match_id",  "matches",            "match_id"),
                ("player_match_stats",    "team_id",   "teams",              "team_id"),
                ("player_match_features", "stat_id",   "player_match_stats", "stat_id"),
                ("player_match_features", "player_id", "players",            "player_id"),
                ("player_match_features", "match_id",  "matches",            "match_id"),
                ("injuries",              "player_id", "players",            "player_id"),
                ("matches",               "stadium_id","stadiums",           "stadium_id"),
                ("pass_network_edges",    "match_id",  "matches",            "match_id"),
                ("shots",                 "match_id",  "matches",            "match_id"),
                ("shots",                 "player_id", "players",            "player_id"),
            ]
            for fk_table, fk_col, ref_table, ref_col in orphan_checks:
                cur.execute(f"""
                    SELECT COUNT(*) FROM {fk_table} f
                    LEFT JOIN {ref_table} r ON r.{ref_col} = f.{fk_col}
                    WHERE f.{fk_col} IS NOT NULL AND r.{ref_col} IS NULL
                """)
                orphans = cur.fetchone()[0]
                if orphans:
                    warnings.append(
                        f"{fk_table}.{fk_col}: {orphans} orphaned row(s)"
                    )

        success = len(errors) == 0
        if success:
            logger.info("Schema validation passed")
        for e in errors:
            logger.error("Schema error: %s", e)
        for w in warnings:
            logger.warning("Schema warning: %s", w)

        return success, warnings, errors

    except Exception as exc:
        logger.error("Schema validation exception: %s", exc)
        return False, [], [str(exc)]