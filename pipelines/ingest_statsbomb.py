"""
pipelines/ingest_statsbomb.py

High-performance StatsBomb ingestion pipeline.

Fixes in this version
---------------------
- stadium_cache is now pre-populated from the DB at the start of run() so
  re-runs do not re-upsert every stadium, and partial-run restarts cannot
  overwrite real coordinates with NULL.
- pg_match_id is now captured directly from upsert_match() return value
  instead of being re-queried per match inside _write_results(), removing
  one redundant SELECT per match in every batch.
- team_name empty string bug: player-team registrations called
  team_cache.get_or_create(sb_tid, "") which could overwrite real names
  with blank placeholders. TeamCache now guards against this.
- Schema rename: statsbomb_match_id -> sb_match_id, statsbomb_team_id ->
  sb_team_id, statsbomb_player_id -> sb_player_id throughout.
- Stadium refactor: matches no longer carries stadium_name/lat/lng directly.
  A stadiums table is upserted first; upsert_match receives stadium_id.
"""

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config.settings import COMPETITIONS, DATA_ROOT, DB_DSN
from extract import statsbomb_local as sb
from load.postgres import connect, insert_stats, upsert_match, upsert_pass_edges, upsert_stadium
from transform.features import (
    agg_match_by_player,
    extract_player_id_col,
    extract_team_id_col,
    extract_type_col,
)

logger = logging.getLogger(__name__)

# cpu_count - 1 assumes RAM scales with cores, which is false on this box
# (16 logical cores, ~10GB RAM) -- that many workers each parsing full match
# JSON concurrently caused an OpenBLAS allocation failure and later a
# Postgres-side OOM. Cap at 4 by default; override with --workers if the
# host has more headroom.
_WORKERS      = min(4, max(1, (os.cpu_count() or 2) - 1))
_COMMIT_EVERY = 50


@dataclass
class MatchResult:
    sb_match_id:  int
    match_date:   object
    home_sb_id:   int
    away_sb_id:   int
    home_name:    str
    away_name:    str
    home_score:   int
    away_score:   int
    comp_name:    str
    season:       str
    stadium_name: Optional[str]
    player_team:  dict = field(default_factory=dict)   # sb_pid -> (name, sb_tid)
    player_stats: dict = field(default_factory=dict)   # sb_pid -> stats_dict
    pass_edges_sb: list = field(default_factory=list)
    starting_positions: dict = field(default_factory=dict)  # sb_pid -> position_name
    minute_snapshots: dict   = field(default_factory=dict)  # (sb_tid, minute) -> stats

    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Worker (no DB access)
# ---------------------------------------------------------------------------

def _process_match(
    sb_match_id, comp_name, season, match_date,
    home_team, away_team, home_score, away_score,
    stadium_name, data_root,
) -> MatchResult:
    result = MatchResult(
        sb_match_id  = sb_match_id,
        match_date   = match_date,
        home_sb_id   = home_team["home_team_id"],
        away_sb_id   = away_team["away_team_id"],
        home_name    = home_team["home_team_name"],
        away_name    = away_team["away_team_name"],
        home_score   = home_score,
        away_score   = away_score,
        comp_name    = comp_name,
        season       = season,
        stadium_name = stadium_name,
    )
    try:
        sb.set_root(data_root)
        events = sb.events(sb_match_id)
        if events.empty:
            return result

        type_col      = extract_type_col(events)
        player_id_col = extract_player_id_col(events)
        team_id_col   = extract_team_id_col(events)

        for _, row in events.iterrows():
            p = row["player"]
            t = row["team"]
            if isinstance(p, dict) and isinstance(t, dict):
                pid = p.get("id")
                tid = t.get("id")
                if isinstance(pid, int) and isinstance(tid, int):
                    result.player_team[pid] = (p.get("name", ""), tid)

        result.player_stats = agg_match_by_player(events, type_col, player_id_col)
        result.pass_edges_sb = _extract_pass_edges_sb(
            events, type_col, player_id_col, team_id_col, sb_match_id
        )
        from transform.features import extract_starting_positions, build_minute_snapshots
        try:
            lineups_df = sb.lineups(sb_match_id)
            result.starting_positions = extract_starting_positions(lineups_df)
        except Exception as exc:
            logger.debug("Lineup load failed for match %d: %s", sb_match_id, exc)

        result.minute_snapshots = build_minute_snapshots(
            events,
            home_team["home_team_id"],
            away_team["away_team_id"],
        )
    except Exception as exc:
        result.error = str(exc)
    return result


def _extract_pass_edges_sb(events, type_col, player_id_col, team_id_col, sb_match_id):
    is_pass = type_col == "Pass"
    if not is_pass.any():
        return []

    pass_df = events.loc[is_pass].copy(deep=False)
    pass_df["_pid"] = player_id_col[is_pass].values
    pass_df["_tid"] = team_id_col[is_pass].values

    from collections import defaultdict
    acc = defaultdict(lambda: {"n": 0, "xs": 0.0, "ys": 0.0, "xe": 0.0, "ye": 0.0})

    for _, row in pass_df.iterrows():
        pass_data = row.get("pass") or {}
        if pass_data.get("outcome") is not None:
            continue
        recip = pass_data.get("recipient")
        if not isinstance(recip, dict):
            continue
        recip_id  = recip.get("id")
        passer_id = row["_pid"]
        team_id   = row["_tid"]
        if not (passer_id and recip_id and team_id):
            continue

        key = (int(team_id), int(passer_id), int(recip_id))
        loc_s = row.get("location") or []
        loc_e = pass_data.get("end_location") or []
        a = acc[key]
        a["n"] += 1
        if len(loc_s) >= 2:
            a["xs"] += loc_s[0]; a["ys"] += loc_s[1]
        if len(loc_e) >= 2:
            a["xe"] += loc_e[0]; a["ye"] += loc_e[1]

    return [
        (sb_match_id, tid, pid, rid, a["n"],
         a["xs"]/a["n"], a["ys"]/a["n"], a["xe"]/a["n"], a["ye"]/a["n"])
        for (tid, pid, rid), a in acc.items()
    ]


# ---------------------------------------------------------------------------
# DB writer (main process)
# ---------------------------------------------------------------------------

def _upsert_snapshots(conn, rows: list, page_size: int = 1000):
    from psycopg2.extras import execute_values
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO match_minute_snapshots (
                match_id, team_id, minute,
                goals_so_far, xg_so_far, shots_so_far,
                passes_so_far, pass_acc_so_far,
                pressures_so_far, red_cards_so_far
            ) VALUES %s
            ON CONFLICT (match_id, team_id, minute) DO NOTHING
        """, rows, page_size=page_size)


def _write_results(conn, team_cache, player_cache, results,
                   stadium_cache: dict):
    """
    Persist a batch of MatchResult objects.

    stadium_cache: mutable dict {stadium_name -> stadium_id} maintained across
    batches. Pre-populated from DB at the start of run() so re-runs and
    partial-run restarts skip redundant stadium upserts.
    """
    all_stat_rows     = []
    all_edge_rows     = []
    all_snapshot_rows = []

    for res in results:
        if res.error:
            logger.error("Match %d worker error: %s", res.sb_match_id, res.error)
            continue

        team_cache.get_or_create(res.home_sb_id, res.home_name)
        team_cache.get_or_create(res.away_sb_id, res.away_name)

        for sb_pid, (p_name, sb_tid) in res.player_team.items():
            player_cache.get_or_create(sb_pid, p_name)
            team_cache.get_or_create(sb_tid, "")

        team_cache.flush()
        player_cache.flush()

        stadium_id: Optional[int] = None
        if res.stadium_name:
            if res.stadium_name not in stadium_cache:
                stadium_cache[res.stadium_name] = upsert_stadium(
                    conn, res.stadium_name
                )
            stadium_id = stadium_cache[res.stadium_name]

        hs, aw = res.home_score, res.away_score
        if hs > aw:
            result_map = {res.home_sb_id: "win",  res.away_sb_id: "loss"}
        elif hs < aw:
            result_map = {res.home_sb_id: "loss", res.away_sb_id: "win"}
        else:
            result_map = {res.home_sb_id: "draw", res.away_sb_id: "draw"}

        # Capture pg_match_id directly from upsert_match — no re-query needed.
        pg_match_id = upsert_match(conn, {
            "sb_match_id":  res.sb_match_id,
            "match_date":   res.match_date,
            "home_team_id": team_cache.resolve(res.home_sb_id),
            "away_team_id": team_cache.resolve(res.away_sb_id),
            "home_score":   res.home_score,
            "away_score":   res.away_score,
            "competition":  res.comp_name,
            "season":       res.season,
            "stadium_id":   stadium_id,
        })

        for sb_pid, stats in res.player_stats.items():
            info = res.player_team.get(sb_pid)
            if info is None:
                continue
            _, sb_tid = info
            try:
                pg_pid = player_cache.resolve(sb_pid)
                pg_tid = team_cache.resolve(sb_tid)
            except KeyError:
                continue

            all_stat_rows.append((
                pg_pid, pg_match_id, pg_tid,
                result_map.get(sb_tid),
                stats["goals"], stats["assists"], stats["shots"],
                stats["xg"], stats["xa"], stats["key_passes"],
                stats["passes_attempted"], stats["passes_completed"],
                stats["pass_accuracy"], stats["progressive_passes"],
                stats["carry_distance"], stats["progressive_carries"],
                stats["dribbles_completed"],
                stats["tackles"], stats["interceptions"],
                stats["clearances"], stats["pressures"],
                stats["yellow_cards"], stats["red_cards"],
                stats["minutes_played"], stats["sub_minute"],
                res.starting_positions.get(sb_pid),
            ))

        for (_, sb_tid, sb_pid, sb_rid, n, axs, ays, axe, aye) in res.pass_edges_sb:
            try:
                pg_passer   = player_cache.resolve(sb_pid)
                pg_receiver = player_cache.resolve(sb_rid)
                pg_team     = team_cache.resolve(sb_tid)
            except KeyError:
                continue
            all_edge_rows.append((
                pg_match_id, pg_team, pg_passer, pg_receiver,
                n, axs, ays, axe, aye,
            ))

        # Build snapshot rows using the already-resolved pg_match_id.
        for (sb_tid, minute), snap in res.minute_snapshots.items():
            try:
                pg_tid = team_cache.resolve(sb_tid)
            except KeyError:
                continue
            all_snapshot_rows.append((
                pg_match_id, pg_tid, minute,
                snap["goals_so_far"], snap["xg_so_far"],
                snap["shots_so_far"], snap["passes_so_far"],
                snap["pass_acc_so_far"], snap["pressures_so_far"],
                snap["red_cards_so_far"],
            ))

    insert_stats(conn, all_stat_rows)
    if all_snapshot_rows:
        _upsert_snapshots(conn, all_snapshot_rows)
    upsert_pass_edges(conn, all_edge_rows)
    conn.commit()
    return len(all_stat_rows), len(all_edge_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(conn, team_cache, player_cache,
        workers=_WORKERS, commit_every=_COMMIT_EVERY):
    # Pre-populate stadium cache from DB so re-runs skip redundant upserts.
    with conn.cursor() as cur:
        cur.execute("SELECT stadium_name, stadium_id FROM stadiums")
        stadium_cache: dict[str, int] = dict(cur.fetchall())
    logger.info("Stadium cache pre-loaded: %d entries", len(stadium_cache))

    comps = sb.competitions()
    if COMPETITIONS:
        comps = comps[comps.apply(
            lambda r: (int(r["competition_id"]), int(r["season_id"])) in COMPETITIONS,
            axis=1,
        )]

    logger.info("%d competition-seasons in scope | workers=%d", len(comps), workers)

    total_stats = total_edges = 0

    for _, comp in comps.iterrows():
        t0        = time.time()
        comp_name = comp.get("competition_name", f"Comp {comp['competition_id']}")
        season    = comp.get("season_name", str(comp["season_id"]))

        try:
            matches = sb.matches(comp["competition_id"], comp["season_id"])
            if matches.empty:
                logger.warning("No matches found for %s %s", comp_name, season)
                continue

            logger.info("%-30s %s  (%d matches)", comp_name, season, len(matches))

            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for _, match in matches.iterrows():
                    sb_mid       = int(match["match_id"])
                    home_team    = match["home_team"]
                    away_team    = match["away_team"]
                    stadium      = match.get("stadium")
                    stadium_name = stadium.get("name") if isinstance(stadium, dict) else None

                    futures[pool.submit(
                        _process_match,
                        sb_mid, comp_name, season,
                        match.get("match_date"),
                        home_team if isinstance(home_team, dict) else {"home_team_id": 0, "home_team_name": ""},
                        away_team if isinstance(away_team, dict) else {"away_team_id": 0, "away_team_name": ""},
                        int(match.get("home_score") or 0),
                        int(match.get("away_score") or 0),
                        stadium_name,
                        str(DATA_ROOT),
                    )] = sb_mid

                pending = []
                done    = 0
                for fut in as_completed(futures):
                    sb_mid = futures[fut]
                    try:
                        pending.append(fut.result())
                    except Exception as exc:
                        logger.error("Worker exception match %d: %s", sb_mid, exc)
                        continue
                    done += 1
                    if len(pending) >= commit_every:
                        s, e = _write_results(conn, team_cache, player_cache,
                                              pending, stadium_cache)
                        total_stats += s
                        total_edges += e
                        logger.info("  %d/%d matches flushed", done, len(matches))
                        pending.clear()

                if pending:
                    s, e = _write_results(conn, team_cache, player_cache,
                                          pending, stadium_cache)
                    total_stats += s
                    total_edges += e

            logger.info("  Done in %.1fs | stat rows: %d | edge rows: %d",
                        time.time() - t0, total_stats, total_edges)

        except Exception as exc:
            logger.error("Failed %s %s: %s", comp_name, season, exc)
            raise

    logger.info("StatsBomb ingestion complete | %d stat rows | %d edge rows",
                total_stats, total_edges)
