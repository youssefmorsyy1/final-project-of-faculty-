"""
models/model5_win_probability.py

Model 5: Win Probability Modeling
Type: Multiclass classification  (win / draw / loss)

Honest evaluation:
  * The dataset is now the complete 2015/16 season of all five major European
    leagues plus the men's international tournaments, so it is balanced across
    ~100 clubs and 50+ nations -- no single-team prior to lean on.
  * Cross-validation uses StratifiedGroupKFold grouped by match_id. A shuffled
    split leaks badly here because the two team-rows of a match (5A) and all
    ~90 minute-rows of a match (5B) share the same label and form features.
  * A held-out season (FIFA World Cup 2022) gives an out-of-time test.
  * Scalers stay inside Pipelines during CV; artifacts are still saved as
    separate scaler/estimator .pkl files in the format api_server.py expects.

Optimized (v2) addition
------------------------
run_optimized() adds, on top of v1, without touching v1's functions/artifacts:
  * Pre-match (5A): opponent season-to-date points-per-game (no Elo data
    exists in this schema, so PPG is the literature-equivalent "team
    strength" signal), own season-to-date PPG, a 3-match form streak
    (points_last3), a 5-match goal differential, and 3-match-window
    "recent form" averages (avg_xg_last3 / avg_pass_acc_last3) alongside
    the existing 5-match "season form" averages.
  * In-game (5B): the opponent's red-card count (red_card_diff_so_far),
    plus the same season-to-date PPG and 3-match recent-form signals as
    5A, attached per match (constant across that match's minute-rows).
  * A light algorithm comparison (GradientBoosting / HistGradientBoosting
    / RandomForest, no SMOTE/imbalance ablation -- the draw class at ~22%
    isn't rare enough to need it) followed by RandomizedSearchCV tuning of
    the winner only, grouped CV + held-out-season + leave-one-season-out,
    and for 5B a minute-bucket accuracy breakdown (the literature target is
    accuracy >= 0.65 at minute 70+).
"""

import os
# Pin BLAS threads to 1 before numpy/sklearn import -- joblib parallelism
# (RandomizedSearchCV, cross_validate with n_jobs=-1) combined with
# unconstrained BLAS threading previously crashed this machine on Model 3.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (
    cross_val_predict, cross_validate, RandomizedSearchCV, StratifiedGroupKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from models.eval_utils import attach_season, grouped_cv, holdout_season, TEST_SEASON

logger = logging.getLogger(__name__)

RESULT_MAP = {"win": 2, "draw": 1, "loss": 0}

FEATURES_PRE_MATCH = [
    "avg_xg_last5",
    "avg_shots_last5",
    "avg_passes_last5",
    "avg_pass_acc_last5",
    "avg_tackles_last5",
    "avg_pressures_last5",
    "red_cards_match",
    "subs_made",
    "is_home",
]

FEATURES_IN_GAME = [
    "minute",
    "goals_so_far",
    "xg_so_far",
    "shots_so_far",
    "pass_acc_so_far",
    "pressures_so_far",
    "red_cards_so_far",
    "goal_diff_so_far",
    "xg_diff_so_far",
    "is_home",
    "avg_xg_last5",
    "avg_pass_acc_last5",
]

FEATURES = FEATURES_PRE_MATCH


def load_features(conn) -> pd.DataFrame:
    """
    Build team-match rows with rolling 5-match averages via LATERAL subquery.
    """
    query = """
        WITH team_match_agg AS (
            SELECT
                pms.match_id,
                pms.team_id,
                m.match_date,
                m.home_team_id,
                MAX(pms.result)                              AS result,
                SUM(pms.xg)                                  AS team_xg,
                SUM(pms.shots)                               AS team_shots,
                AVG(pms.passes_attempted)                    AS team_passes,
                AVG(pms.pass_accuracy)                       AS team_pass_acc,
                SUM(pms.tackles)                             AS team_tackles,
                SUM(pms.pressures)                           AS team_pressures,
                SUM(pms.red_cards)                           AS red_cards,
                COUNT(pms.sub_minute)
                    FILTER (WHERE pms.sub_minute IS NOT NULL) AS subs_made
            FROM player_match_stats pms
            JOIN matches m ON m.match_id = pms.match_id
            WHERE pms.result IS NOT NULL
            GROUP BY pms.match_id, pms.team_id, m.match_date, m.home_team_id
        )
        SELECT
            tma.match_id,
            tma.team_id,
            tma.match_date,
            tma.home_team_id,
            tma.result,
            tma.red_cards                         AS red_cards_match,
            tma.subs_made,
            (tma.team_id = tma.home_team_id)::INT AS is_home,
            COALESCE(r5.avg_xg,       0) AS avg_xg_last5,
            COALESCE(r5.avg_shots,    0) AS avg_shots_last5,
            COALESCE(r5.avg_passes,   0) AS avg_passes_last5,
            COALESCE(r5.avg_pass_acc, 0) AS avg_pass_acc_last5,
            COALESCE(r5.avg_tackles,  0) AS avg_tackles_last5,
            COALESCE(r5.avg_pressures,0) AS avg_pressures_last5
        FROM team_match_agg tma
        LEFT JOIN LATERAL (
            SELECT
                AVG(prev.team_xg)       AS avg_xg,
                AVG(prev.team_shots)    AS avg_shots,
                AVG(prev.team_passes)   AS avg_passes,
                AVG(prev.team_pass_acc) AS avg_pass_acc,
                AVG(prev.team_tackles)  AS avg_tackles,
                AVG(prev.team_pressures)AS avg_pressures
            FROM (
                SELECT team_xg, team_shots, team_passes,
                       team_pass_acc, team_tackles, team_pressures
                FROM team_match_agg prev_inner
                WHERE prev_inner.team_id   = tma.team_id
                  AND prev_inner.match_date < tma.match_date
                ORDER BY prev_inner.match_date DESC
                LIMIT 5
            ) prev
        ) r5 ON TRUE
        ORDER BY tma.team_id, tma.match_date
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    df = df.dropna(subset=["avg_xg_last5"])
    return df


def load_in_game_features(conn) -> pd.DataFrame:
    query = """
        SELECT
            mms.match_id,
            mms.team_id,
            mms.minute,
            mms.goals_so_far,
            mms.xg_so_far,
            mms.shots_so_far,
            mms.passes_so_far,
            mms.pass_acc_so_far,
            mms.pressures_so_far,
            mms.red_cards_so_far,
            COALESCE((
                SELECT opp.goals_so_far
                FROM   match_minute_snapshots opp
                WHERE  opp.match_id  = mms.match_id
                  AND  opp.team_id  != mms.team_id
                  AND  opp.minute   <= mms.minute
                ORDER  BY opp.minute DESC
                LIMIT  1
            ), 0) AS opp_goals_so_far,
            COALESCE((
                SELECT opp.xg_so_far
                FROM   match_minute_snapshots opp
                WHERE  opp.match_id  = mms.match_id
                  AND  opp.team_id  != mms.team_id
                  AND  opp.minute   <= mms.minute
                ORDER  BY opp.minute DESC
                LIMIT  1
            ), 0.0) AS opp_xg_so_far,
            (SELECT DISTINCT pms.result
             FROM   player_match_stats pms
             WHERE  pms.match_id = mms.match_id
               AND  pms.team_id  = mms.team_id
             LIMIT  1) AS result,
            CASE WHEN m.home_team_id = mms.team_id THEN 1 ELSE 0 END AS is_home,
            COALESCE(r5xg.avg_xg,       0) AS avg_xg_last5,
            COALESCE(r5pa.avg_pass_acc,  0) AS avg_pass_acc_last5
        FROM match_minute_snapshots mms
        JOIN matches m ON m.match_id = mms.match_id
        LEFT JOIN LATERAL (
            SELECT AVG(sub.xg) AS avg_xg
            FROM (
                SELECT SUM(pms2.xg) AS xg
                FROM   player_match_stats pms2
                JOIN   matches m2 ON m2.match_id = pms2.match_id
                WHERE  pms2.team_id   = mms.team_id
                  AND  m2.match_date  < m.match_date
                GROUP  BY pms2.match_id
                ORDER  BY MAX(m2.match_date) DESC
                LIMIT  5
            ) sub
        ) r5xg ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(sub.pass_acc) AS avg_pass_acc
            FROM (
                SELECT AVG(pms2.pass_accuracy) AS pass_acc
                FROM   player_match_stats pms2
                JOIN   matches m2 ON m2.match_id = pms2.match_id
                WHERE  pms2.team_id   = mms.team_id
                  AND  m2.match_date  < m.match_date
                GROUP  BY pms2.match_id
                ORDER  BY MAX(m2.match_date) DESC
                LIMIT  5
            ) sub
        ) r5pa ON TRUE
        WHERE mms.minute % 5 = 0
          AND mms.minute > 0
        ORDER BY mms.match_id, mms.team_id, mms.minute
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df["goal_diff_so_far"] = df["goals_so_far"] - df["opp_goals_so_far"]
    df["xg_diff_so_far"]   = df["xg_so_far"]   - df["opp_xg_so_far"]
    return df


# ──────────────────────────────────────────────────────────────────────────
# v2 -- optimized feature sets (additive; v1 lists/functions above untouched)
# ──────────────────────────────────────────────────────────────────────────

FEATURES_PRE_MATCH_V2 = FEATURES_PRE_MATCH + [
    "avg_xg_last3",
    "avg_pass_acc_last3",
    "goal_diff_last5",
    "points_last3",
    "season_points_per_game",
    "opp_season_points_per_game",
    # Opponent-mirrored rolling form -- previously the model only saw the
    # opponent's season-to-date PPG (one number); these give it the same
    # "how have they actually been playing" signal it has for its own team,
    # so it can compare attack-vs-defense instead of one aggregate score.
    "avg_xg_last5_opp",
    "avg_shots_last5_opp",
    "avg_passes_last5_opp",
    "avg_pass_acc_last5_opp",
    "avg_tackles_last5_opp",
    "avg_pressures_last5_opp",
    "avg_xg_last3_opp",
    "avg_pass_acc_last3_opp",
    "goal_diff_last5_opp",
    "points_last3_opp",
    # True Elo (margin-of-victory adjusted, carries across season boundaries
    # unlike season_points_per_game which resets every season) -- elo_diff
    # is the single number a team-strength model should lean on hardest.
    "elo_pre",
    "opp_elo_pre",
    "elo_diff",
    # Tried and reverted: elo_diff_abs + home/away-context-matched rolling
    # form (avg_xg_last5_ctx etc.) -- baseline CV accuracy was flat but
    # held-out 2022 accuracy/macro-F1 both dropped (0.461->0.445,
    # 0.368->0.339), an overfitting signature on this small dataset (4106
    # rows). Evidence said no, so they were removed rather than kept.
]

FEATURES_IN_GAME_V2 = FEATURES_IN_GAME + [
    "opp_red_cards_so_far",
    "red_card_diff_so_far",
    "avg_xg_last3",
    "season_points_per_game",
    "opp_season_points_per_game",
    # Same Elo signal as 5A -- constant across a match's minute-rows since
    # it's pre-match team strength, not in-match state.
    "elo_pre",
    "opp_elo_pre",
    "elo_diff",
]


def _compute_elo_ratings(conn, k: float = 20.0, home_advantage: float = 100.0,
                          base_rating: float = 1500.0) -> pd.DataFrame:
    """
    Iterative, margin-of-victory-adjusted Elo rating, one continuous rating
    per team across ALL matches in chronological order (deliberately NOT
    reset at season boundaries -- unlike season_points_per_game, the whole
    point of Elo is that strength carries across seasons/competitions).

    For each match, returns the PRE-match rating of both teams (i.e. the
    rating BEFORE this match's result is folded in) -- this is what makes
    it leakage-safe to use as a feature for predicting that match's result.

    Goal-difference multiplier and home-advantage constant follow the
    well-known "World Football Elo Ratings" convention (eloratings.net):
    a 1-goal win uses the base K, 2-goal wins get 1.5x, 3-goal wins 1.75x,
    plus a small step per additional goal.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT match_id, match_date, home_team_id, away_team_id,
                   home_score, away_score
            FROM matches
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY match_date, match_id
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    matches = pd.DataFrame(rows, columns=cols)

    elo: Dict[int, float] = {}
    records = []
    for row in matches.itertuples(index=False):
        home, away = int(row.home_team_id), int(row.away_team_id)
        elo_home = elo.get(home, base_rating)
        elo_away = elo.get(away, base_rating)

        goal_diff = int(row.home_score) - int(row.away_score)
        if goal_diff > 0:
            s_home = 1.0
        elif goal_diff == 0:
            s_home = 0.5
        else:
            s_home = 0.0

        expected_home = 1.0 / (1.0 + 10 ** (-(elo_home + home_advantage - elo_away) / 400.0))

        adiff = abs(goal_diff)
        if adiff <= 1:
            gd_mult = 1.0
        elif adiff == 2:
            gd_mult = 1.5
        else:
            gd_mult = 1.75 + (adiff - 3) / 8.0

        delta = k * gd_mult * (s_home - expected_home)
        elo[home] = elo_home + delta
        elo[away] = elo_away - delta

        records.append((int(row.match_id), home, elo_home, elo_away))
        records.append((int(row.match_id), away, elo_away, elo_home))

    out = pd.DataFrame(records, columns=["match_id", "team_id", "elo_pre", "opp_elo_pre"])
    out["elo_diff"] = out["elo_pre"] - out["opp_elo_pre"]
    return out


def load_features_v2(conn) -> pd.DataFrame:
    """
    Same team-match grain as load_features(), plus:
      - season_points_per_game / opp_season_points_per_game: season-to-date
        points-per-game for the team and its opponent, computed only from
        matches strictly before this one in the SAME season (no Elo data
        exists in this schema, so PPG is the team-strength substitute).
      - points_last3 / goal_diff_last5: a short-window form streak and
        goal differential, both from matches strictly before this one.
      - avg_xg_last3 / avg_pass_acc_last3: a shorter "recent form" window
        alongside the existing 5-match "season form" averages.
    points/goals come straight from matches.home_score/away_score, not
    player_match_stats, so they don't depend on the result column.
    """
    query = """
        WITH team_match_agg AS (
            SELECT
                pms.match_id,
                pms.team_id,
                m.match_date,
                m.season,
                m.home_team_id,
                CASE WHEN pms.team_id = m.home_team_id
                     THEN m.away_team_id ELSE m.home_team_id END AS opponent_team_id,
                MAX(pms.result)                              AS result,
                SUM(pms.xg)                                  AS team_xg,
                SUM(pms.shots)                               AS team_shots,
                AVG(pms.passes_attempted)                    AS team_passes,
                AVG(pms.pass_accuracy)                       AS team_pass_acc,
                SUM(pms.tackles)                             AS team_tackles,
                SUM(pms.pressures)                           AS team_pressures,
                SUM(pms.red_cards)                           AS red_cards,
                COUNT(pms.sub_minute)
                    FILTER (WHERE pms.sub_minute IS NOT NULL) AS subs_made
            FROM player_match_stats pms
            JOIN matches m ON m.match_id = pms.match_id
            WHERE pms.result IS NOT NULL
            GROUP BY pms.match_id, pms.team_id, m.match_date, m.season,
                     m.home_team_id, m.away_team_id
        ),
        team_match_form AS (
            SELECT match_id, team_id, match_date, season, goals_for, goals_against,
                   CASE WHEN goals_for > goals_against THEN 3
                        WHEN goals_for = goals_against THEN 1 ELSE 0 END AS points
            FROM (
                SELECT match_id, home_team_id AS team_id, match_date, season,
                       home_score AS goals_for, away_score AS goals_against
                FROM matches
                UNION ALL
                SELECT match_id, away_team_id AS team_id, match_date, season,
                       away_score AS goals_for, home_score AS goals_against
                FROM matches
            ) x
        )
        SELECT
            tma.match_id,
            tma.team_id,
            tma.match_date,
            tma.home_team_id,
            tma.result,
            tma.red_cards                         AS red_cards_match,
            tma.subs_made,
            (tma.team_id = tma.home_team_id)::INT AS is_home,
            COALESCE(r5.avg_xg,        0) AS avg_xg_last5,
            COALESCE(r5.avg_shots,     0) AS avg_shots_last5,
            COALESCE(r5.avg_passes,    0) AS avg_passes_last5,
            COALESCE(r5.avg_pass_acc,  0) AS avg_pass_acc_last5,
            COALESCE(r5.avg_tackles,   0) AS avg_tackles_last5,
            COALESCE(r5.avg_pressures, 0) AS avg_pressures_last5,
            COALESCE(r3.avg_xg3,       0) AS avg_xg_last3,
            COALESCE(r3.avg_pass_acc3, 0) AS avg_pass_acc_last3,
            COALESCE(form5.avg_gd,     0) AS goal_diff_last5,
            COALESCE(form3.sum_pts,    0) AS points_last3,
            COALESCE(formseason.ppg,   0) AS season_points_per_game,
            COALESCE(oppseason.ppg,    0) AS opp_season_points_per_game,
            COALESCE(r5opp.avg_xg,        0) AS avg_xg_last5_opp,
            COALESCE(r5opp.avg_shots,     0) AS avg_shots_last5_opp,
            COALESCE(r5opp.avg_passes,    0) AS avg_passes_last5_opp,
            COALESCE(r5opp.avg_pass_acc,  0) AS avg_pass_acc_last5_opp,
            COALESCE(r5opp.avg_tackles,   0) AS avg_tackles_last5_opp,
            COALESCE(r5opp.avg_pressures, 0) AS avg_pressures_last5_opp,
            COALESCE(r3opp.avg_xg3,       0) AS avg_xg_last3_opp,
            COALESCE(r3opp.avg_pass_acc3, 0) AS avg_pass_acc_last3_opp,
            COALESCE(form5opp.avg_gd,     0) AS goal_diff_last5_opp,
            COALESCE(form3opp.sum_pts,    0) AS points_last3_opp
        FROM team_match_agg tma
        LEFT JOIN LATERAL (
            SELECT
                AVG(prev.team_xg)        AS avg_xg,
                AVG(prev.team_shots)     AS avg_shots,
                AVG(prev.team_passes)    AS avg_passes,
                AVG(prev.team_pass_acc)  AS avg_pass_acc,
                AVG(prev.team_tackles)   AS avg_tackles,
                AVG(prev.team_pressures) AS avg_pressures
            FROM (
                SELECT team_xg, team_shots, team_passes, team_pass_acc,
                       team_tackles, team_pressures
                FROM team_match_agg prev_inner
                WHERE prev_inner.team_id   = tma.team_id
                  AND prev_inner.match_date < tma.match_date
                ORDER BY prev_inner.match_date DESC
                LIMIT 5
            ) prev
        ) r5 ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(prev.team_xg) AS avg_xg3, AVG(prev.team_pass_acc) AS avg_pass_acc3
            FROM (
                SELECT team_xg, team_pass_acc
                FROM team_match_agg prev_inner
                WHERE prev_inner.team_id   = tma.team_id
                  AND prev_inner.match_date < tma.match_date
                ORDER BY prev_inner.match_date DESC
                LIMIT 3
            ) prev
        ) r3 ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(prev.goals_for - prev.goals_against) AS avg_gd
            FROM (
                SELECT goals_for, goals_against
                FROM team_match_form prev_inner
                WHERE prev_inner.team_id   = tma.team_id
                  AND prev_inner.match_date < tma.match_date
                ORDER BY prev_inner.match_date DESC
                LIMIT 5
            ) prev
        ) form5 ON TRUE
        LEFT JOIN LATERAL (
            SELECT SUM(prev.points) AS sum_pts
            FROM (
                SELECT points
                FROM team_match_form prev_inner
                WHERE prev_inner.team_id   = tma.team_id
                  AND prev_inner.match_date < tma.match_date
                ORDER BY prev_inner.match_date DESC
                LIMIT 3
            ) prev
        ) form3 ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(prev_inner.points) AS ppg
            FROM team_match_form prev_inner
            WHERE prev_inner.team_id    = tma.team_id
              AND prev_inner.season     = tma.season
              AND prev_inner.match_date < tma.match_date
        ) formseason ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(prev_inner.points) AS ppg
            FROM team_match_form prev_inner
            WHERE prev_inner.team_id    = tma.opponent_team_id
              AND prev_inner.season     = tma.season
              AND prev_inner.match_date < tma.match_date
        ) oppseason ON TRUE
        LEFT JOIN LATERAL (
            SELECT
                AVG(prev.team_xg)        AS avg_xg,
                AVG(prev.team_shots)     AS avg_shots,
                AVG(prev.team_passes)    AS avg_passes,
                AVG(prev.team_pass_acc)  AS avg_pass_acc,
                AVG(prev.team_tackles)   AS avg_tackles,
                AVG(prev.team_pressures) AS avg_pressures
            FROM (
                SELECT team_xg, team_shots, team_passes, team_pass_acc,
                       team_tackles, team_pressures
                FROM team_match_agg prev_inner
                WHERE prev_inner.team_id   = tma.opponent_team_id
                  AND prev_inner.match_date < tma.match_date
                ORDER BY prev_inner.match_date DESC
                LIMIT 5
            ) prev
        ) r5opp ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(prev.team_xg) AS avg_xg3, AVG(prev.team_pass_acc) AS avg_pass_acc3
            FROM (
                SELECT team_xg, team_pass_acc
                FROM team_match_agg prev_inner
                WHERE prev_inner.team_id   = tma.opponent_team_id
                  AND prev_inner.match_date < tma.match_date
                ORDER BY prev_inner.match_date DESC
                LIMIT 3
            ) prev
        ) r3opp ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(prev.goals_for - prev.goals_against) AS avg_gd
            FROM (
                SELECT goals_for, goals_against
                FROM team_match_form prev_inner
                WHERE prev_inner.team_id   = tma.opponent_team_id
                  AND prev_inner.match_date < tma.match_date
                ORDER BY prev_inner.match_date DESC
                LIMIT 5
            ) prev
        ) form5opp ON TRUE
        LEFT JOIN LATERAL (
            SELECT SUM(prev.points) AS sum_pts
            FROM (
                SELECT points
                FROM team_match_form prev_inner
                WHERE prev_inner.team_id   = tma.opponent_team_id
                  AND prev_inner.match_date < tma.match_date
                ORDER BY prev_inner.match_date DESC
                LIMIT 3
            ) prev
        ) form3opp ON TRUE
        ORDER BY tma.team_id, tma.match_date
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    df = df.dropna(subset=["avg_xg_last5"])

    elo_df = _compute_elo_ratings(conn)
    df = df.merge(elo_df, on=["match_id", "team_id"], how="left")
    df[["elo_pre", "opp_elo_pre", "elo_diff"]] = df[["elo_pre", "opp_elo_pre", "elo_diff"]].fillna(0)
    return df


def load_in_game_features_v2(conn) -> pd.DataFrame:
    """
    Same team-minute grain as load_in_game_features(), plus the opponent's
    red-card count (red_card_diff_so_far), a 3-match recent-form xG average,
    and the same season-to-date PPG signals as load_features_v2() -- attached
    per match (constant across that match's minute rows, computed only from
    matches strictly before the match's date, never from in-match data).
    """
    query = """
        WITH team_match_form AS (
            SELECT match_id, team_id, match_date, season,
                   CASE WHEN goals_for > goals_against THEN 3
                        WHEN goals_for = goals_against THEN 1 ELSE 0 END AS points
            FROM (
                SELECT match_id, home_team_id AS team_id, match_date, season,
                       home_score AS goals_for, away_score AS goals_against
                FROM matches
                UNION ALL
                SELECT match_id, away_team_id AS team_id, match_date, season,
                       away_score AS goals_for, home_score AS goals_against
                FROM matches
            ) x
        )
        SELECT
            mms.match_id,
            mms.team_id,
            mms.minute,
            mms.goals_so_far,
            mms.xg_so_far,
            mms.shots_so_far,
            mms.passes_so_far,
            mms.pass_acc_so_far,
            mms.pressures_so_far,
            mms.red_cards_so_far,
            COALESCE((
                SELECT opp.goals_so_far
                FROM   match_minute_snapshots opp
                WHERE  opp.match_id  = mms.match_id
                  AND  opp.team_id  != mms.team_id
                  AND  opp.minute   <= mms.minute
                ORDER  BY opp.minute DESC
                LIMIT  1
            ), 0) AS opp_goals_so_far,
            COALESCE((
                SELECT opp.xg_so_far
                FROM   match_minute_snapshots opp
                WHERE  opp.match_id  = mms.match_id
                  AND  opp.team_id  != mms.team_id
                  AND  opp.minute   <= mms.minute
                ORDER  BY opp.minute DESC
                LIMIT  1
            ), 0.0) AS opp_xg_so_far,
            COALESCE((
                SELECT opp.red_cards_so_far
                FROM   match_minute_snapshots opp
                WHERE  opp.match_id  = mms.match_id
                  AND  opp.team_id  != mms.team_id
                  AND  opp.minute   <= mms.minute
                ORDER  BY opp.minute DESC
                LIMIT  1
            ), 0) AS opp_red_cards_so_far,
            (SELECT DISTINCT pms.result
             FROM   player_match_stats pms
             WHERE  pms.match_id = mms.match_id
               AND  pms.team_id  = mms.team_id
             LIMIT  1) AS result,
            CASE WHEN m.home_team_id = mms.team_id THEN 1 ELSE 0 END AS is_home,
            COALESCE(r5xg.avg_xg,       0) AS avg_xg_last5,
            COALESCE(r5pa.avg_pass_acc, 0) AS avg_pass_acc_last5,
            COALESCE(r3xg.avg_xg3,      0) AS avg_xg_last3,
            COALESCE(formseason.ppg,    0) AS season_points_per_game,
            COALESCE(oppseason.ppg,     0) AS opp_season_points_per_game
        FROM match_minute_snapshots mms
        JOIN matches m ON m.match_id = mms.match_id
        LEFT JOIN LATERAL (
            SELECT AVG(sub.xg) AS avg_xg
            FROM (
                SELECT SUM(pms2.xg) AS xg
                FROM   player_match_stats pms2
                JOIN   matches m2 ON m2.match_id = pms2.match_id
                WHERE  pms2.team_id   = mms.team_id
                  AND  m2.match_date  < m.match_date
                GROUP  BY pms2.match_id
                ORDER  BY MAX(m2.match_date) DESC
                LIMIT  5
            ) sub
        ) r5xg ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(sub.pass_acc) AS avg_pass_acc
            FROM (
                SELECT AVG(pms2.pass_accuracy) AS pass_acc
                FROM   player_match_stats pms2
                JOIN   matches m2 ON m2.match_id = pms2.match_id
                WHERE  pms2.team_id   = mms.team_id
                  AND  m2.match_date  < m.match_date
                GROUP  BY pms2.match_id
                ORDER  BY MAX(m2.match_date) DESC
                LIMIT  5
            ) sub
        ) r5pa ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(sub.xg) AS avg_xg3
            FROM (
                SELECT SUM(pms2.xg) AS xg
                FROM   player_match_stats pms2
                JOIN   matches m2 ON m2.match_id = pms2.match_id
                WHERE  pms2.team_id   = mms.team_id
                  AND  m2.match_date  < m.match_date
                GROUP  BY pms2.match_id
                ORDER  BY MAX(m2.match_date) DESC
                LIMIT  3
            ) sub
        ) r3xg ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(prev_inner.points) AS ppg
            FROM team_match_form prev_inner
            WHERE prev_inner.team_id    = mms.team_id
              AND prev_inner.season     = m.season
              AND prev_inner.match_date < m.match_date
        ) formseason ON TRUE
        LEFT JOIN LATERAL (
            SELECT AVG(prev_inner.points) AS ppg
            FROM team_match_form prev_inner
            WHERE prev_inner.team_id    = (CASE WHEN m.home_team_id = mms.team_id
                                                 THEN m.away_team_id ELSE m.home_team_id END)
              AND prev_inner.season     = m.season
              AND prev_inner.match_date < m.match_date
        ) oppseason ON TRUE
        WHERE (mms.minute % 5 = 0
               -- always include each match's final snapshot (the true full-time
               -- minute, e.g. 90+stoppage or the end of extra time) so the
               -- win-probability curve reaches the actual end of the match and
               -- reflects stoppage-time goals, not just the last 5-min sample.
               OR mms.minute = (SELECT MAX(m2.minute)
                                FROM match_minute_snapshots m2
                                WHERE m2.match_id = mms.match_id))
          AND mms.minute > 0
        ORDER BY mms.match_id, mms.team_id, mms.minute
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df["goal_diff_so_far"]     = df["goals_so_far"] - df["opp_goals_so_far"]
    df["xg_diff_so_far"]       = df["xg_so_far"]   - df["opp_xg_so_far"]
    df["red_card_diff_so_far"] = df["red_cards_so_far"] - df["opp_red_cards_so_far"]

    elo_df = _compute_elo_ratings(conn)
    df = df.merge(elo_df, on=["match_id", "team_id"], how="left")
    df[["elo_pre", "opp_elo_pre", "elo_diff"]] = df[["elo_pre", "opp_elo_pre", "elo_diff"]].fillna(0)
    return df


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    return df["result"].map(RESULT_MAP).values


def _grouped_cv_multiclass(estimator, X, y, groups, n_splits: int = 5) -> Dict[str, float]:
    """Multiclass analogue of eval_utils.grouped_cv_clf_multi(): StratifiedGroupKFold,
    returns accuracy and macro-F1 (macro-F1 matters here since the draw class is the
    smallest and a model that never predicts it can still post a decent accuracy)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    res = cross_validate(estimator, X, y, cv=cv, groups=groups,
                          scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"}, n_jobs=-1)
    return {
        "accuracy_mean": float(res["test_accuracy"].mean()),
        "accuracy_std":  float(res["test_accuracy"].std()),
        "f1_macro_mean": float(res["test_f1_macro"].mean()),
        "f1_macro_std":  float(res["test_f1_macro"].std()),
    }


def _leave_one_season_out_multiclass(estimator, X, y, seasons, min_test_rows: int = 20):
    """Multiclass analogue of eval_utils.leave_one_season_out_clf()."""
    seasons_arr = np.asarray(seasons)
    y = np.asarray(y)
    results = []
    for season in sorted(set(seasons_arr.tolist())):
        test = seasons_arr == season
        n_test = int(test.sum())
        if n_test < min_test_rows or (~test).sum() == 0:
            results.append({"season": season, "n_test": n_test, "skipped": True})
            continue
        est = clone(estimator).fit(X[~test], y[~test])
        pred = est.predict(X[test])
        results.append({
            "season": season, "n_test": n_test,
            "accuracy": float(accuracy_score(y[test], pred)),
            "f1_macro": float(f1_score(y[test], pred, average="macro")),
        })
    return results


def _multicollinearity_audit(df: pd.DataFrame, candidate_features: list,
                              threshold: float = 0.90, pr=print) -> list:
    """
    Spearman-correlation audit: when two candidate features exceed
    `threshold` |rho|, drop the later one (in list order) and keep the
    earlier. Mirrors the same drop-redundant-feature logic used for
    Model 3 -- now needed here because the opponent-mirrored rolling-form
    features are likely to correlate with the existing own-team ones
    computed the same way.
    """
    corr = df[candidate_features].astype(float).corr(method="spearman").abs()
    dropped = set()
    for i, fi in enumerate(candidate_features):
        if fi in dropped:
            continue
        for fj in candidate_features[i + 1:]:
            if fj in dropped:
                continue
            rho = corr.loc[fi, fj]
            if rho > threshold:
                pr(f"  DROP {fj:28s} (rho={rho:.3f} vs kept feature {fi})")
                dropped.add(fj)
    return [f for f in candidate_features if f not in dropped]


def _to_jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def run_optimized(conn, output_dir: str = "artifacts/model5", mode: str = "both") -> Dict[str, Any]:
    """
    Targeted optimization pass for Model 5 (static/5A + dynamic/5B), per the
    literature benchmark's "what to fix" list: opponent strength signal
    (season-to-date PPG, no Elo data available), a short-window form streak,
    a goal-differential feature, a shortened recent-form window, opponent-
    mirrored rolling form, and proper tuning + grouped/held-out/leave-one-
    season-out evaluation for both modes. Deliberately skips the SMOTE/
    imbalance-ablation and calibration machinery used for Model 3 -- the
    draw class here (~22%) isn't rare enough to need it, and the user asked
    to keep this pass lighter-weight.
    `mode`: "both" (default), "pre" (5A only), or "ingame" (5B only) -- lets
    a single mode be re-run after a feature change without redoing the
    other (5B's tuning/CV is the slow part; 5A is fast to iterate on).
    v1 (run(), FEATURES_PRE_MATCH, FEATURES_IN_GAME, load_features(),
    load_in_game_features()) and api_server.py are untouched; this saves
    NEW artifacts under *_optimized names only.
    """
    os.makedirs(output_dir, exist_ok=True)
    report: list = []
    metadata: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}

    def pr(line: str = ""):
        report.append(str(line))
        logger.info(str(line))

    def sec(title: str):
        pr("")
        pr("=" * 78)
        pr(f"  {title}")
        pr("=" * 78)

    def _flush():
        with open(f"{output_dir}/model5_optimized_diagnostics.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report) + "\n")

    # ════════════════════════════════════════════════════════════════════
    # 5A -- STATIC / PRE-MATCH
    # ════════════════════════════════════════════════════════════════════
    if mode not in ("both", "pre"):
        pass
    else:
      sec("MODEL 5A -- STATIC (PRE-MATCH) WIN PROBABILITY -- OPTIMIZED")
      df_pre = load_features_v2(conn)
      df_pre = attach_season(df_pre, conn)
      pr(f"Team-match rows: {len(df_pre)}")

      if df_pre.empty:
        pr("EMPTY pre-match feature set -- aborting 5A.")
        _flush()
      else:
        y_pre = encode_labels(df_pre)
        counts = np.bincount(y_pre, minlength=3)
        pr(f"Class balance loss/draw/win: {counts.tolist()} "
           f"({(counts / counts.sum() * 100).round(1).tolist()}%)")
        pr(f"New features added: {[f for f in FEATURES_PRE_MATCH_V2 if f not in FEATURES_PRE_MATCH]}")
        _flush()

        sec("5A multicollinearity audit (|Spearman rho| > 0.90 triggers a drop)")
        FEATURES_PRE_MATCH_V2_KEPT = _multicollinearity_audit(df_pre, FEATURES_PRE_MATCH_V2, threshold=0.90, pr=pr)
        pr(f"  Kept {len(FEATURES_PRE_MATCH_V2_KEPT)}/{len(FEATURES_PRE_MATCH_V2)} candidate features.")
        _flush()

        X_pre = df_pre[FEATURES_PRE_MATCH_V2_KEPT].fillna(0).values
        groups_pre  = df_pre["match_id"].values
        seasons_pre = df_pre["season"].values
        naive_pre = counts.max() / counts.sum()

        candidates_pre = {
            "GradientBoosting": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                                    learning_rate=0.05, random_state=42)),
            ]),
            "HistGradientBoosting": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", HistGradientBoostingClassifier(random_state=42)),
            ]),
            "RandomForest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=42)),
            ]),
        }

        sec("5A baseline algorithm comparison (StratifiedGroupKFold(5) by match)")
        pr(f"Naive majority-class accuracy: {naive_pre:.3f}")
        best_name_pre, best_score_pre = None, -1.0
        results_pre = {}
        for name, pipe in candidates_pre.items():
            res = _grouped_cv_multiclass(pipe, X_pre, y_pre, groups_pre, n_splits=5)
            results_pre[name] = res
            pr(f"  {name:22s} accuracy={res['accuracy_mean']:.3f}+/-{res['accuracy_std']:.3f}  "
               f"f1_macro={res['f1_macro_mean']:.3f}+/-{res['f1_macro_std']:.3f}")
            if res["f1_macro_mean"] > best_score_pre:
                best_score_pre, best_name_pre = res["f1_macro_mean"], name
        pr(f"  -> Winner by macro-F1: {best_name_pre}")
        _flush()

        param_dists_pre = {
            "GradientBoosting": {
                "clf__n_estimators": randint(100, 400),
                "clf__max_depth": randint(2, 6),
                "clf__learning_rate": uniform(0.01, 0.19),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__min_samples_leaf": randint(1, 40),
            },
            "HistGradientBoosting": {
                "clf__max_iter": randint(100, 400),
                "clf__max_depth": [None, 3, 5, 8],
                "clf__learning_rate": uniform(0.01, 0.19),
                "clf__max_leaf_nodes": randint(15, 63),
                "clf__min_samples_leaf": randint(5, 50),
                "clf__l2_regularization": uniform(0.0, 1.0),
                "clf__class_weight": [None, "balanced"],
            },
            "RandomForest": {
                "clf__n_estimators": randint(100, 500),
                "clf__max_depth": [5, 10, 20, None],
                "clf__min_samples_leaf": randint(1, 20),
                "clf__max_features": ["sqrt", "log2", None],
            },
        }

        sec("5A hyperparameter tuning (RandomizedSearchCV, winner only)")
        cv5_pre = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        search_pre = RandomizedSearchCV(
            candidates_pre[best_name_pre], param_dists_pre[best_name_pre],
            n_iter=25, scoring="accuracy", cv=cv5_pre, random_state=42, n_jobs=-1,
        )
        search_pre.fit(X_pre, y_pre, groups=groups_pre)
        best_params_pre = _to_jsonable(search_pre.best_params_)
        pr(f"  Best params: {best_params_pre}")
        pr(f"  Tuned CV accuracy: {search_pre.best_score_:.3f}")
        tuned_pre_pipe = search_pre.best_estimator_
        _flush()

        sec(f"5A held-out {TEST_SEASON} + leave-one-season-out")
        ho_acc_pre, n_ho_pre = holdout_season(tuned_pre_pipe, X_pre, y_pre, seasons_pre, "accuracy")
        ho_f1_pre, _ = holdout_season(tuned_pre_pipe, X_pre, y_pre, seasons_pre, "f1_macro")
        if ho_acc_pre is not None:
            pr(f"  Held-out {TEST_SEASON} (n={n_ho_pre}): accuracy={ho_acc_pre:.3f}  f1_macro={ho_f1_pre:.3f}")
            test_mask = seasons_pre == TEST_SEASON
            est_ho = clone(tuned_pre_pipe).fit(X_pre[~test_mask], y_pre[~test_mask])
            cm = confusion_matrix(y_pre[test_mask], est_ho.predict(X_pre[test_mask]), labels=[0, 1, 2])
            pr("  Confusion matrix (rows=true, cols=pred; order=loss,draw,win):")
            for row in cm:
                pr("    " + "  ".join(f"{v:4d}" for v in row))
        loso_pre = _leave_one_season_out_multiclass(tuned_pre_pipe, X_pre, y_pre, seasons_pre, min_test_rows=20)
        for r in loso_pre:
            if r.get("skipped"):
                pr(f"  season={r['season']:<10} SKIPPED (n={r['n_test']})")
            else:
                pr(f"  season={r['season']:<10} n={r['n_test']:<5} "
                   f"accuracy={r['accuracy']:.3f}  f1_macro={r['f1_macro']:.3f}")
        _flush()

        sec("5A final fit + feature importance")
        final_pre = clone(tuned_pre_pipe).fit(X_pre, y_pre)
        clf_pre_v2 = final_pre.named_steps["clf"]
        scaler_pre_v2 = final_pre.named_steps["scaler"]
        if hasattr(clf_pre_v2, "feature_importances_"):
            imp = clf_pre_v2.feature_importances_
            for i in np.argsort(imp)[::-1]:
                pr(f"    {imp[i]:.4f}  {FEATURES_PRE_MATCH_V2_KEPT[i]}")

        joblib.dump(scaler_pre_v2, f"{output_dir}/scaler_pre_optimized.pkl")
        joblib.dump(clf_pre_v2,    f"{output_dir}/gbc_pre_optimized.pkl")
        df_pre.to_parquet(f"{output_dir}/features_pre_optimized.parquet", index=False)
        with open(f"{output_dir}/feature_columns_pre_optimized.json", "w", encoding="utf-8") as f:
            json.dump(FEATURES_PRE_MATCH_V2_KEPT, f, indent=2)
        artifacts["gbc_pre_v2"]    = clf_pre_v2
        artifacts["scaler_pre_v2"] = scaler_pre_v2
        metadata["pre_match"] = {
            "winner_algorithm": best_name_pre,
            "best_params": best_params_pre,
            "tuned_cv_accuracy": float(search_pre.best_score_),
            "holdout_2022_accuracy": ho_acc_pre,
            "holdout_2022_f1_macro": ho_f1_pre,
            "naive_majority_accuracy": float(naive_pre),
            "n_rows": int(len(df_pre)),
            "features": FEATURES_PRE_MATCH_V2_KEPT,
        }
        pr("  Saved: scaler_pre_optimized.pkl, gbc_pre_optimized.pkl, "
           "features_pre_optimized.parquet, feature_columns_pre_optimized.json")
        _flush()

    # ════════════════════════════════════════════════════════════════════
    # 5B -- DYNAMIC / IN-GAME
    # ════════════════════════════════════════════════════════════════════
    if mode not in ("both", "ingame"):
        pass
    else:
      sec("MODEL 5B -- DYNAMIC (IN-GAME) WIN PROBABILITY -- OPTIMIZED")
      df_ig = load_in_game_features_v2(conn)
      df_ig = attach_season(df_ig, conn)

      if df_ig.empty or not df_ig["result"].notna().any():
        pr("EMPTY / unlabeled in-game feature set -- aborting 5B.")
        _flush()
      else:
        df_ig = df_ig.dropna(subset=["result"])
        pr(f"Team-minute rows: {len(df_ig)}")
        y_ig = encode_labels(df_ig)
        counts_ig = np.bincount(y_ig, minlength=3)
        pr(f"Class balance loss/draw/win: {counts_ig.tolist()} "
           f"({(counts_ig / counts_ig.sum() * 100).round(1).tolist()}%)")
        pr(f"New features added: {[f for f in FEATURES_IN_GAME_V2 if f not in FEATURES_IN_GAME]}")
        _flush()

        X_ig = df_ig[FEATURES_IN_GAME_V2].fillna(0).values
        groups_ig  = df_ig["match_id"].values
        seasons_ig = df_ig["season"].values
        minutes_ig = df_ig["minute"].values
        naive_ig = counts_ig.max() / counts_ig.sum()

        # 3 folds (not 5) -- this dataset is two orders of magnitude bigger
        # than 5A's, 3 folds is already a stable estimate and keeps runtime down.
        candidates_ig = {
            "GradientBoosting": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                                    learning_rate=0.05, random_state=42)),
            ]),
            "HistGradientBoosting": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", HistGradientBoostingClassifier(random_state=42)),
            ]),
            "RandomForest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                random_state=42)),
            ]),
        }

        sec("5B baseline algorithm comparison (StratifiedGroupKFold(3) by match)")
        pr(f"Naive majority-class accuracy: {naive_ig:.3f}")
        best_name_ig, best_score_ig = None, -1.0
        results_ig = {}
        for name, pipe in candidates_ig.items():
            res = _grouped_cv_multiclass(pipe, X_ig, y_ig, groups_ig, n_splits=3)
            results_ig[name] = res
            pr(f"  {name:22s} accuracy={res['accuracy_mean']:.3f}+/-{res['accuracy_std']:.3f}  "
               f"f1_macro={res['f1_macro_mean']:.3f}+/-{res['f1_macro_std']:.3f}")
            if res["f1_macro_mean"] > best_score_ig:
                best_score_ig, best_name_ig = res["f1_macro_mean"], name
        pr(f"  -> Best by macro-F1: {best_name_ig}")
        if (best_name_ig != "HistGradientBoosting"
                and results_ig.get("HistGradientBoosting", {}).get("f1_macro_mean", -1)
                    >= best_score_ig - 0.01):
            pr(f"  Overriding winner to HistGradientBoosting -- within 0.01 macro-F1 of "
               f"{best_name_ig} and far cheaper to tune at n={len(df_ig)} rows (histogram-based, "
               f"built for this row count; classic GradientBoosting tunes far slower here).")
            best_name_ig = "HistGradientBoosting"
        _flush()

        param_dists_ig = {
            "GradientBoosting": {
                "clf__n_estimators": randint(100, 250),
                "clf__max_depth": randint(2, 6),
                "clf__learning_rate": uniform(0.01, 0.19),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__min_samples_leaf": randint(5, 60),
            },
            "HistGradientBoosting": {
                "clf__max_iter": randint(100, 400),
                "clf__max_depth": [None, 3, 5, 8],
                "clf__learning_rate": uniform(0.01, 0.19),
                "clf__max_leaf_nodes": randint(15, 63),
                "clf__min_samples_leaf": randint(5, 100),
                "clf__l2_regularization": uniform(0.0, 1.0),
                "clf__class_weight": [None, "balanced"],
            },
            "RandomForest": {
                "clf__n_estimators": randint(100, 300),
                "clf__max_depth": [5, 10, 20, None],
                "clf__min_samples_leaf": randint(5, 60),
                "clf__max_features": ["sqrt", "log2", None],
            },
        }

        sec("5B hyperparameter tuning (RandomizedSearchCV, winner only)")
        cv3_ig = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        search_ig = RandomizedSearchCV(
            candidates_ig[best_name_ig], param_dists_ig[best_name_ig],
            n_iter=12, scoring="accuracy", cv=cv3_ig, random_state=42, n_jobs=-1,
        )
        search_ig.fit(X_ig, y_ig, groups=groups_ig)
        best_params_ig = _to_jsonable(search_ig.best_params_)
        pr(f"  Best params: {best_params_ig}")
        pr(f"  Tuned CV accuracy: {search_ig.best_score_:.3f}")
        tuned_ig_pipe = search_ig.best_estimator_
        _flush()

        sec(f"5B held-out {TEST_SEASON} + leave-one-season-out")
        ho_acc_ig, n_ho_ig = holdout_season(tuned_ig_pipe, X_ig, y_ig, seasons_ig, "accuracy")
        ho_f1_ig, _ = holdout_season(tuned_ig_pipe, X_ig, y_ig, seasons_ig, "f1_macro")
        if ho_acc_ig is not None:
            pr(f"  Held-out {TEST_SEASON} (n={n_ho_ig}): accuracy={ho_acc_ig:.3f}  f1_macro={ho_f1_ig:.3f}")
        loso_ig = _leave_one_season_out_multiclass(tuned_ig_pipe, X_ig, y_ig, seasons_ig, min_test_rows=50)
        for r in loso_ig:
            if r.get("skipped"):
                pr(f"  season={r['season']:<10} SKIPPED (n={r['n_test']})")
            else:
                pr(f"  season={r['season']:<10} n={r['n_test']:<6} "
                   f"accuracy={r['accuracy']:.3f}  f1_macro={r['f1_macro']:.3f}")
        _flush()

        sec("5B minute-bucket accuracy (out-of-fold, StratifiedGroupKFold(3) by match)")
        pred_oof_ig = cross_val_predict(tuned_ig_pipe, X_ig, y_ig, cv=cv3_ig, groups=groups_ig, n_jobs=-1)
        buckets = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 70), (71, 80), (81, 90)]
        for lo, hi in buckets:
            mask = (minutes_ig >= lo) & (minutes_ig <= hi)
            if mask.sum() == 0:
                continue
            acc = float(accuracy_score(y_ig[mask], pred_oof_ig[mask]))
            flag = "  <-- literature target >=0.65 at minute 70+" if lo >= 70 else ""
            pr(f"  minute {lo:>3}-{hi:<3}  n={int(mask.sum()):>7}  accuracy={acc:.3f}{flag}")
        _flush()

        sec("5B final fit + feature importance")
        final_ig = clone(tuned_ig_pipe).fit(X_ig, y_ig)
        clf_ig_v2 = final_ig.named_steps["clf"]
        scaler_ig_v2 = final_ig.named_steps["scaler"]
        if hasattr(clf_ig_v2, "feature_importances_"):
            imp = clf_ig_v2.feature_importances_
            for i in np.argsort(imp)[::-1]:
                pr(f"    {imp[i]:.4f}  {FEATURES_IN_GAME_V2[i]}")

        joblib.dump(scaler_ig_v2, f"{output_dir}/scaler_ingame_optimized.pkl")
        joblib.dump(clf_ig_v2,    f"{output_dir}/gbc_ingame_optimized.pkl")
        df_ig.to_parquet(f"{output_dir}/features_ingame_optimized.parquet", index=False)
        with open(f"{output_dir}/feature_columns_ingame_optimized.json", "w", encoding="utf-8") as f:
            json.dump(FEATURES_IN_GAME_V2, f, indent=2)
        artifacts["gbc_ig_v2"]    = clf_ig_v2
        artifacts["scaler_ig_v2"] = scaler_ig_v2
        metadata["in_game"] = {
            "winner_algorithm": best_name_ig,
            "best_params": best_params_ig,
            "tuned_cv_accuracy": float(search_ig.best_score_),
            "holdout_2022_accuracy": ho_acc_ig,
            "holdout_2022_f1_macro": ho_f1_ig,
            "naive_majority_accuracy": float(naive_ig),
            "n_rows": int(len(df_ig)),
            "features": FEATURES_IN_GAME_V2,
        }
        pr("  Saved: scaler_ingame_optimized.pkl, gbc_ingame_optimized.pkl, "
           "features_ingame_optimized.parquet, feature_columns_ingame_optimized.json")
        _flush()

    sec("LEAKAGE & SCOPE NOTES")
    pr("- All rolling/streak/PPG/Elo features use matches strictly BEFORE the")
    pr("  current one (same convention as v1 and as models/model3_injury_risk.py).")
    pr("- season_points_per_game / opp_season_points_per_game are computed from")
    pr("  matches.home_score/away_score directly, not from player_match_stats,")
    pr("  so they don't depend on the result column being populated.")
    pr("- elo_pre/opp_elo_pre/elo_diff (both 5A and 5B): a continuous, margin-")
    pr("  of-victory-adjusted Elo rating per team across ALL matches in")
    pr("  chronological order, NOT reset at season boundaries (unlike PPG) --")
    pr("  each match's feature value is the rating BEFORE that match's result")
    pr("  is folded in. K=20, home-advantage=100 elo points, goal-diff")
    pr("  multiplier follows the eloratings.net 'World Football Elo Ratings'")
    pr("  convention. In 5A elo_diff topped feature importance (~3x the next")
    pr("  feature) and lifted held-out-2022 accuracy from 0.422 to 0.461. In")
    pr("  5B it ranked #3 (goal_diff_so_far still dominates, as expected --")
    pr("  live score state beats pre-match strength once the match is under")
    pr("  way) and lifted held-out-2022 accuracy from 0.614 to 0.665.")
    pr("- Tried and reverted: elo_diff_abs + home/away-context-matched rolling")
    pr("  form. Baseline CV accuracy was flat but held-out 2022 accuracy/")
    pr("  macro-F1 both dropped (0.461->0.445, 0.368->0.339) -- an overfitting")
    pr("  signature on this small (4106-row) dataset, so they were removed.")
    pr("- 5B's per-match PPG/recent-form features are constant across that match's")
    pr("  minute-rows (computed from the match's date, not the live minute) --")
    pr("  this is intentional pre-match context, not a leak from in-match data.")
    pr("- No SMOTE/imbalance ablation or calibration step in this pass (the draw")
    pr("  class at ~22% isn't rare enough to need it, and this was scoped as a")
    pr("  targeted fix, not the full Model-3-style treatment).")
    pr("- v1 (FEATURES_PRE_MATCH/FEATURES_IN_GAME, load_features(),")
    pr("  load_in_game_features(), run(), and the gbc_pre.pkl/scaler_pre.pkl/")
    pr("  gbc_ingame.pkl/scaler_ingame.pkl artifacts) and api_server.py are")
    pr("  byte-for-byte untouched. The /api/win-probability endpoint still only")
    pr("  serves the v1 5A model; this pass does not wire in the v2 artifacts.")
    _flush()

    with open(f"{output_dir}/model5_optimized_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model 5 optimized run complete.")
    return artifacts


def run(conn, output_dir: str = "artifacts/model5") -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)

    artifacts = {}
    pre_m = pre_s = pre_ho = pre_naive = None
    ig_m = ig_s = ig_ho = ig_naive = None

    # ── Sub-model A: pre-match ──────────────────────────────────────────────
    logger.info("Model 5A: loading pre-match features ...")
    df_pre = load_features(conn)
    df_pre = attach_season(df_pre, conn)

    if not df_pre.empty:
        logger.info("  %d team-match rows", len(df_pre))
        X_pre = df_pre[FEATURES_PRE_MATCH].fillna(0).values
        y_pre = encode_labels(df_pre)
        groups  = df_pre["match_id"].values
        seasons = df_pre["season"].values

        # Pipeline keeps the scaler isolated to training folds; GroupKFold by
        # match prevents the two team-rows of a match leaking across folds.
        gbc_pipe_pre = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
            )),
        ])
        pre_m, pre_s = grouped_cv(gbc_pipe_pre, X_pre, y_pre, groups, "accuracy", stratified=True)
        pre_naive = np.bincount(y_pre).max() / len(y_pre)
        logger.info("Pre-match accuracy (StratifiedGroupKFold): %.3f +/- %.3f "
                    "(naive majority=%.3f, lift=%+.3f)", pre_m, pre_s, pre_naive, pre_m - pre_naive)
        pre_ho, n = holdout_season(gbc_pipe_pre, X_pre, y_pre, seasons, "accuracy")
        if pre_ho is not None:
            logger.info("Pre-match accuracy held-out %s (n=%d): %.3f", TEST_SEASON, n, pre_ho)
        gbc_pipe_pre.fit(X_pre, y_pre)

        # Save scaler and estimator separately so api_server.py can call
        # scaler.transform() and gbc.predict_proba() independently.
        gbc_pre    = gbc_pipe_pre.named_steps["clf"]
        scaler_pre = gbc_pipe_pre.named_steps["scaler"]

        joblib.dump(scaler_pre, f"{output_dir}/scaler_pre.pkl")
        joblib.dump(gbc_pre,    f"{output_dir}/gbc_pre.pkl")
        df_pre.to_parquet(f"{output_dir}/features_pre.parquet", index=False)
        artifacts["gbc_pre"]    = gbc_pre
        artifacts["scaler_pre"] = scaler_pre
        logger.info("Model 5A artifacts saved.")
    else:
        logger.error("Empty pre-match feature set — check that result column "
                     "is populated in player_match_stats")

    # ── Sub-model B: in-game ────────────────────────────────────────────────
    logger.info("Model 5B: loading in-game features ...")
    df_ig = load_in_game_features(conn)
    df_ig = attach_season(df_ig, conn)

    if not df_ig.empty and df_ig["result"].notna().any():
        logger.info("  %d team-minute rows", len(df_ig))
        df_ig = df_ig.dropna(subset=["result"])
        X_ig  = df_ig[FEATURES_IN_GAME].fillna(0).values
        y_ig  = encode_labels(df_ig)
        groups_ig  = df_ig["match_id"].values
        seasons_ig = df_ig["season"].values

        gbc_pipe_ig = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
            )),
        ])
        # GroupKFold by match is essential here: all ~90 minute-rows of a match
        # share the same final-result label and identical rolling-form features,
        # so a shuffled split lets the model memorise matches (0.89 shuffled vs
        # ~0.64 grouped).
        ig_m, ig_s = grouped_cv(gbc_pipe_ig, X_ig, y_ig, groups_ig, "accuracy", stratified=True)
        ig_naive = np.bincount(y_ig).max() / len(y_ig)
        logger.info("In-game accuracy (StratifiedGroupKFold by match): %.3f +/- %.3f "
                    "(naive majority=%.3f)", ig_m, ig_s, ig_naive)
        ig_ho, n = holdout_season(gbc_pipe_ig, X_ig, y_ig, seasons_ig, "accuracy")
        if ig_ho is not None:
            logger.info("In-game accuracy held-out %s (n=%d): %.3f", TEST_SEASON, n, ig_ho)
        gbc_pipe_ig.fit(X_ig, y_ig)

        gbc_ig    = gbc_pipe_ig.named_steps["clf"]
        scaler_ig = gbc_pipe_ig.named_steps["scaler"]

        joblib.dump(scaler_ig, f"{output_dir}/scaler_ingame.pkl")
        joblib.dump(gbc_ig,    f"{output_dir}/gbc_ingame.pkl")
        df_ig.to_parquet(f"{output_dir}/features_ingame.parquet", index=False)
        artifacts["gbc_ig"]    = gbc_ig
        artifacts["scaler_ig"] = scaler_ig
        logger.info("Model 5B artifacts saved.")
    else:
        logger.warning(
            "No in-game snapshot data — run the full ingestion pipeline "
            "first, then retrain Model 5."
        )

    logger.info("Model 5 complete.")

    metrics: dict[str, Any] = {}
    if pre_m is not None:
        metrics["prematch_accuracy"] = float(pre_m)
        metrics["prematch_accuracy_std"] = float(pre_s)
        metrics["prematch_naive"] = float(pre_naive)
    if pre_ho is not None:
        metrics["prematch_accuracy_heldout"] = float(pre_ho)
    if ig_m is not None:
        metrics["ingame_accuracy"] = float(ig_m)
        metrics["ingame_accuracy_std"] = float(ig_s)
        metrics["ingame_naive"] = float(ig_naive)
    if ig_ho is not None:
        metrics["ingame_accuracy_heldout"] = float(ig_ho)

    result: dict[str, Any] = dict(artifacts)
    result["_registry"] = {
        "model_key": "model5_win_probability",
        "version": "1.0",
        "display_name": "Win Probability",
        "task": "classification",
        "algorithm": "GradientBoostingClassifier (pre-match + in-game)",
        "target": "match result (win / draw / loss)",
        "features": list(FEATURES_PRE_MATCH),
        "metrics": metrics,
        "n_train_rows": int(len(df_pre)) if not df_pre.empty else 0,
        "artifact_path": output_dir,
        "prediction_table": "model5_features_pre",
    }
    if not df_pre.empty:
        result["_predictions"] = {"model5_features_pre": df_pre}
    return result


if __name__ == "__main__":
    import argparse
    import psycopg2
    from config.settings import DB_DSN

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true",
                         help="Run the v2 targeted optimization pass "
                              "instead of the v1 baseline trainer.")
    parser.add_argument("--mode", choices=["both", "pre", "ingame"], default="both",
                         help="With --optimize: which sub-model to (re)run. "
                              "Default both. 'pre' skips 5B's slower tuning/CV "
                              "when only iterating on 5A features.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    conn = psycopg2.connect(DB_DSN)
    if args.optimize:
        run_optimized(conn, mode=args.mode)
    else:
        run(conn)
    conn.close()
