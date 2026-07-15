"""
pipelines/ingest_injuries.py

Load Transfermarkt data from two CSVs into the database.

Fix: date_of_birth null check simplified. The original nested
pd.notna(row.get(...) if "date_of_birth" in row.index else None)
pattern was fragile — it called .get() twice and would silently
return None for empty-string nulls. Replaced with a direct notna
check on the extracted value.
"""

import logging
import re
import unicodedata

import pandas as pd
from psycopg2.extras import execute_values

try:
    from rapidfuzz import fuzz as _fuzz
    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False

from core.utils import norm_name
from load.postgres import connect
from config.settings import DB_DSN, TRANSFERMARKT_CSV, TRANSFERMARKT_PLAYERS_CSV

logger = logging.getLogger(__name__)

LAST_THRESHOLD = 80
FULL_THRESHOLD = 85

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tm_id(raw) -> int | None:
    if not pd.notna(raw):
        return None
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return None


def _detect_id_col(df: pd.DataFrame, label: str = "") -> str | None:
    for candidate in ("transfermarkt_player_id", "player_id", "tm_player_id", "id"):
        if candidate in df.columns:
            return candidate
    logger.error("%s: cannot find player id column. Columns: %s", label, list(df.columns))
    return None


def _parse_dates(series: pd.Series) -> pd.Series:
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.8:
            return parsed.dt.date
    parsed = pd.to_datetime(series, errors="coerce")
    null_rate = parsed.isna().mean()
    if null_rate > 0.1:
        sample = series[parsed.isna() & series.notna()].head(3).tolist()
        logger.warning("Date parse: %.0f%% null after all formats. Sample: %s",
                       null_rate * 100, sample)
    return parsed.dt.date


# ---------------------------------------------------------------------------
# Player index
# ---------------------------------------------------------------------------

class PlayerIndex:
    def __init__(self, conn):
        self.tm_id_map: dict[int, int]         = {}
        self.norm_map:  dict[str, int]          = {}
        self.token_map: dict[frozenset, int]    = {}
        self.blocks:    dict[str, list[tuple]]  = {}

        with conn.cursor() as cur:
            cur.execute(
                "SELECT player_id, tm_player_id, norm_name FROM players"
            )
            for pid, tmid, nn in cur.fetchall():
                if tmid is not None:
                    self.tm_id_map[int(tmid)] = pid
                if not nn:
                    continue
                self.norm_map[nn] = pid
                tokens = nn.split()
                if not tokens:
                    continue
                self.token_map[frozenset(tokens)] = pid
                last = tokens[-1]
                if last:
                    self.blocks.setdefault(last[0], []).append((nn, last, pid))

        logger.info("Player index: %d TM-linked | %d norm | %d blocks",
                    len(self.tm_id_map), len(self.norm_map), len(self.blocks))

    def register(self, tm_id: int | None, pg_pid: int, nn: str):
        if tm_id is not None:
            self.tm_id_map[tm_id] = pg_pid
        if nn:
            self.norm_map[nn] = pg_pid


def _candidate_norms(row: pd.Series) -> list[str]:
    first = str(row.get("first_name") or "").strip()
    last  = str(row.get("last_name")  or "").strip()
    full  = str(row.get("full_name")  or "").strip()

    seen, seen_set = [], set()
    for s in [
        f"{first} {last}".strip() if first and last else "",
        full,
        f"{first[0]} {last}".strip() if first and last else "",
    ]:
        nn = norm_name(s)
        if nn and nn not in seen_set:
            seen.append(nn)
            seen_set.add(nn)
    return seen


def _resolve(row, index, last_threshold, full_threshold):
    tm_id = _to_tm_id(row.get("transfermarkt_player_id"))
    if tm_id is not None and tm_id in index.tm_id_map:
        return index.tm_id_map[tm_id], "tm_id"

    candidates = _candidate_norms(row)

    for nn in candidates:
        if nn in index.norm_map:
            return index.norm_map[nn], "exact"

    for nn in candidates:
        ct = frozenset(nn.split())
        if not ct:
            continue
        hits = [pid for db_t, pid in index.token_map.items() if ct.issubset(db_t)]
        if len(hits) == 1:
            return hits[0], "token_subset"

    if _RAPIDFUZZ:
        last_raw = str(row.get("last_name") or "").strip()
        if last_raw:
            last_nn   = norm_name(last_raw)
            block_key = last_nn[0] if last_nn else ""
            block     = index.blocks.get(block_key, [])
            full_nn   = candidates[0] if candidates else ""
            best_score, best_pid = -1, None
            for db_nn, _db_last, pid in block:
                ls = max(_fuzz.ratio(last_nn, tok) for tok in db_nn.split())
                if ls < last_threshold:
                    continue
                fs = _fuzz.token_set_ratio(full_nn, db_nn)
                if fs < full_threshold:
                    continue
                combined = (ls + fs) / 2
                if combined > best_score:
                    best_score, best_pid = combined, pid
            if best_pid is not None:
                return best_pid, "fuzzy"

    return None, "no_match"


# ---------------------------------------------------------------------------
# Pass 1: players CSV -> players table
# ---------------------------------------------------------------------------

def ingest_players(conn, players_csv, last_threshold=LAST_THRESHOLD,
                   full_threshold=FULL_THRESHOLD):
    logger.info("Pass 1: %s", players_csv)
    df = pd.read_csv(players_csv, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    id_col = _detect_id_col(df, "players CSV")
    if id_col is None:
        return {}
    if id_col != "transfermarkt_player_id":
        df = df.rename(columns={id_col: "transfermarkt_player_id"})

    if "date_of_birth" in df.columns:
        df["date_of_birth"] = _parse_dates(df["date_of_birth"])

    if not _RAPIDFUZZ:
        logger.warning("rapidfuzz not installed — fuzzy matching disabled (pip install rapidfuzz)")

    index   = PlayerIndex(conn)
    updates = []
    linked  = no_match = 0
    by_method: dict[str, int] = {}

    for _, row in df.iterrows():
        pid, method = _resolve(row, index, last_threshold, full_threshold)
        if pid is None:
            no_match += 1
            continue

        by_method[method] = by_method.get(method, 0) + 1
        linked += 1

        tm_id = _to_tm_id(row.get("transfermarkt_player_id"))

        # Simplified null check: extract once, test once.
        dob_raw = row.get("date_of_birth")
        dob = dob_raw if pd.notna(dob_raw) else None

        nat   = (str(row.get("country_of_citizenship") or "").strip()
                 or str(row.get("country_of_birth") or "").strip() or None) or None
        pos   = (str(row.get("sub_position") or "").strip()
                 or str(row.get("position") or "").strip() or None) or None

        updates.append((tm_id, dob, nat, pos, pid))
        index.register(tm_id, pid, norm_name(
            f"{row.get('first_name') or ''} {row.get('last_name') or ''}".strip()
        ))

    logger.info("Pass 1: %d linked | %d unmatched | methods: %s",
                linked, no_match, by_method)

    if not updates:
        logger.warning("0 players matched — verify CSV paths and column names")
        return index.tm_id_map

    with conn.cursor() as cur:
        for tm_id, dob, nat, pos, pid in updates:
            cur.execute("""
                UPDATE players
                SET
                    tm_player_id  = %s,
                    date_of_birth = COALESCE(date_of_birth, %s),
                    nationality   = COALESCE(nationality,   %s),
                    position      = COALESCE(position,      %s)
                WHERE player_id = %s
            """, (tm_id, dob, nat, pos, pid))
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM players WHERE tm_player_id IS NOT NULL")
        logger.info("Players with TM id in DB: %d", cur.fetchone()[0])

    return index.tm_id_map


# ---------------------------------------------------------------------------
# Pass 2: injuries CSV -> injuries table
# ---------------------------------------------------------------------------

def ingest_injuries(conn, injuries_csv, tm_id_map: dict[int, int]):
    logger.info("Pass 2: %s", injuries_csv)
    df = pd.read_csv(injuries_csv, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    id_col = _detect_id_col(df, "injuries CSV")
    if id_col is None:
        return
    if id_col != "player_id":
        df = df.rename(columns={id_col: "player_id"})

    if not tm_id_map:
        logger.error("tm_id_map is empty — Pass 1 matched 0 players")
        return

    for col in ("from", "until"):
        if col in df.columns:
            df[col] = _parse_dates(df[col])

    df["games_missed"] = pd.to_numeric(
        df.get("games_missed", pd.Series(dtype=float)), errors="coerce"
    )

    rows      = []
    matched   = 0
    unmatched = 0

    for _, row in df.iterrows():
        tm_id = _to_tm_id(row.get("player_id"))
        if tm_id is None or tm_id not in tm_id_map:
            unmatched += 1
            continue

        matched += 1
        date_from  = row.get("from")
        date_until = row.get("until")

        # injury_date (from) is NOT NULL in the schema. Skip rows where it is
        # missing rather than attempting an insert that will always fail.
        if not date_from or not pd.notna(date_from):
            logger.warning(
                "Skipping injury row with null injury_date for player tm_id=%s: %s",
                tm_id, row.get("injury"),
            )
            matched -= 1
            unmatched += 1
            continue

        if not date_until or not pd.notna(date_until):
            logger.warning(
                "Injury row missing return_date for player tm_id=%s, inserting with NULL return_date",
                tm_id,
            )

        rows.append((
            tm_id_map[tm_id],
            str(row.get("injury") or "").strip() or None,
            date_from  if pd.notna(date_from)  else None,
            date_until if pd.notna(date_until) else None,
            int(row["games_missed"]) if pd.notna(row.get("games_missed")) else None,
            str(row.get("season") or "").strip() or None,
        ))

    logger.info("Pass 2: %d matched | %d unmatched", matched, unmatched)

    if rows:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO injuries (
                    player_id, injury_type, injury_date,
                    return_date, matches_missed, season
                ) VALUES %s
                ON CONFLICT (player_id, injury_date, injury_type) DO NOTHING
            """, rows)
        conn.commit()
        logger.info("Inserted %d injury rows", len(rows))
    else:
        logger.warning("No injury rows inserted")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(conn=None, players_csv=None, injuries_csv=None,
        last_threshold=LAST_THRESHOLD, full_threshold=FULL_THRESHOLD):
    if conn is None:
        conn = connect(DB_DSN)
    if players_csv is None:
        players_csv = TRANSFERMARKT_PLAYERS_CSV
    if injuries_csv is None:
        injuries_csv = TRANSFERMARKT_CSV

    tm_id_map = ingest_players(conn, players_csv, last_threshold, full_threshold)
    ingest_injuries(conn, injuries_csv, tm_id_map)
    logger.info("Injuries ingestion complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
