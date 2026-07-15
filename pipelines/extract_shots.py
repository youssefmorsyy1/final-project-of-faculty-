"""
pipelines/extract_shots.py

Build the `shots` table: one row per Shot event with geometry, context and
freeze-frame-derived features, for the from-scratch xG model.

StatsBomb shot events embed a `shot.freeze_frame` (positions of all visible
players at the moment of the shot) in ~98% of shots, so defender/goalkeeper
context is available without the separate 360 files.

Run:  python -m pipelines.extract_shots
"""

import json
import logging
import math
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import execute_batch

from config.settings import DB_DSN, DATA_ROOT, COMPETITIONS

logger = logging.getLogger(__name__)

# StatsBomb pitch is 120 x 80. Goal mouth is the 8-yard span on the goal line.
GOAL_X = 120.0
GOAL_C = (120.0, 40.0)
POST_A = (120.0, 36.0)
POST_B = (120.0, 44.0)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _shot_angle(p) -> float:
    """Angle (radians) subtended by the two goalposts from point p. Larger = better."""
    v1 = (POST_A[0] - p[0], POST_A[1] - p[1])
    v2 = (POST_B[0] - p[0], POST_B[1] - p[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.acos(cos)


def _sign(p1, p2, p3) -> float:
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def _in_triangle(pt, a, b, c) -> bool:
    d1, d2, d3 = _sign(pt, a, b), _sign(pt, b, c), _sign(pt, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _freeze_features(p, freeze_frame) -> Dict[str, Any]:
    """Defender/goalkeeper context from the shot freeze-frame."""
    out = {
        "defenders_in_cone": None,
        "dist_to_nearest_def": None,
        "gk_dist_to_goal": None,
        "gk_dist_to_shot": None,
    }
    if not freeze_frame:
        return out
    opponents = [f for f in freeze_frame if not f.get("teammate", False)]
    if not opponents:
        return out

    in_cone = 0
    nearest = math.inf
    gk_loc = None
    for f in opponents:
        loc = f.get("location")
        if not loc:
            continue
        if f.get("position", {}).get("name") == "Goalkeeper":
            gk_loc = loc
            continue  # keeper handled separately, not counted as a cone defender
        nearest = min(nearest, _dist(p, loc))
        if _in_triangle(loc, p, POST_A, POST_B):
            in_cone += 1

    out["defenders_in_cone"] = in_cone
    out["dist_to_nearest_def"] = None if nearest is math.inf else round(nearest, 3)
    if gk_loc:
        out["gk_dist_to_goal"] = round(_dist(gk_loc, GOAL_C), 3)
        out["gk_dist_to_shot"] = round(_dist(gk_loc, p), 3)
    return out


def _norm_body_part(name: str) -> str:
    if name in ("Right Foot", "Left Foot", "Head"):
        return name
    return "Other"


def _load_id_maps(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT sb_match_id, match_id FROM matches")
        match_map = {int(s): m for s, m in cur.fetchall()}
        cur.execute("SELECT sb_player_id, player_id FROM players WHERE sb_player_id IS NOT NULL")
        player_map = {int(s): p for s, p in cur.fetchall()}
        cur.execute("SELECT sb_team_id, team_id FROM teams")
        team_map = {int(s): t for s, t in cur.fetchall()}
    return match_map, player_map, team_map


def _shot_rows(sb_match_id: int, match_id: int, player_map, team_map) -> List[tuple]:
    path = DATA_ROOT / "events" / f"{sb_match_id}.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        events = json.load(fh)

    rows = []
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        loc = e.get("location")
        if not loc or len(loc) < 2:
            continue
        shot = e.get("shot", {})
        p = (loc[0], loc[1])

        ff = _freeze_features(p, shot.get("freeze_frame"))
        rows.append((
            e["id"],
            match_id,
            player_map.get(e.get("player", {}).get("id")),
            team_map.get(e.get("team", {}).get("id")),
            e.get("minute"),
            round(p[0], 2), round(p[1], 2),
            round(_dist(p, GOAL_C), 3),
            round(_shot_angle(p), 4),
            _norm_body_part(shot.get("body_part", {}).get("name", "Other")),
            shot.get("type", {}).get("name"),
            shot.get("technique", {}).get("name"),
            e.get("play_pattern", {}).get("name"),
            bool(e.get("under_pressure", False)),
            bool(shot.get("first_time", False)),
            ff["defenders_in_cone"],
            ff["dist_to_nearest_def"],
            ff["gk_dist_to_goal"],
            ff["gk_dist_to_shot"],
            shot.get("statsbomb_xg"),
            shot.get("outcome", {}).get("name") == "Goal",
        ))
    return rows


_INSERT = """
    INSERT INTO shots (
        sb_event_id, match_id, player_id, team_id, minute,
        x, y, distance, angle,
        body_part, shot_type, technique, play_pattern, under_pressure, first_time,
        defenders_in_cone, dist_to_nearest_def, gk_dist_to_goal, gk_dist_to_shot,
        statsbomb_xg, is_goal
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (sb_event_id) DO NOTHING
"""


def run(conn) -> int:
    match_map, player_map, team_map = _load_id_maps(conn)
    total = 0
    for (cid, sid) in sorted(COMPETITIONS):
        mfile = DATA_ROOT / "matches" / str(cid) / f"{sid}.json"
        if not mfile.exists():
            continue
        with open(mfile, encoding="utf-8") as fh:
            sb_match_ids = [m["match_id"] for m in json.load(fh)]
        comp_rows = 0
        with conn.cursor() as cur:
            for sb_mid in sb_match_ids:
                match_id = match_map.get(int(sb_mid))
                if match_id is None:
                    continue
                rows = _shot_rows(sb_mid, match_id, player_map, team_map)
                if rows:
                    execute_batch(cur, _INSERT, rows, page_size=500)
                    comp_rows += len(rows)
        conn.commit()
        total += comp_rows
        logger.info("competition %s/%s: %d shots", cid, sid, comp_rows)
    logger.info("Shot extraction complete | %d shots", total)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()
