"""
models/model3_injury_risk.py

Model 3: Injury Risk Prediction
Type: Binary classification
Target: is_injured_next_30d
Algorithms: XGBoost (primary), Random Forest, Logistic Regression (baseline)

Fix: data leakage in cross-validation.
Previously StandardScaler was fit on the full dataset before cross_val_score,
meaning the scaler had seen test-fold values during fitting. This inflated
reported CV AUC scores. The scaler is now wrapped inside a Pipeline so it is
fit only on the training fold during each CV split. The final scaler fit on
the full dataset for inference is unchanged.

-------------------------------------------------------------------------------
v2 addition: run_optimized()
-------------------------------------------------------------------------------
Everything above (FEATURES, load_features(), preprocess(), run()) is UNCHANGED
and still produces the exact artifacts api_server.py expects
(artifacts/model3/{xgb,rf,lr,scaler}.pkl, features.parquet). run_optimized()
is a separate, additive pipeline that:

  1. Engineers 7 new workload/recovery features (matches/minutes in the last
     7 and 14 days, alongside the existing 30-day windows; a playing-streak
     feature; days of rest before this match; a capped days_since_last_injury)
     and runs a data-driven multicollinearity audit over all 20 candidates.
  2. Fixes a methodological gap shared with the original: the original
     StratifiedKFold splits at the row level, so the same player's other
     match-rows can sit in both train and test folds, leaking player-identity
     signal. run_optimized() uses StratifiedGroupKFold grouped by player_id
     instead, so no player straddles the train/test boundary.
  3. Tests whether the literature's prescribed SMOTE-on-train-folds-only
     fix actually beats the existing class_weight="balanced" approach on
     this data (an "imbalance ablation"), instead of assuming SMOTE helps.
  4. Compares 5-6 classifiers (LogisticRegression, RandomForest, ExtraTrees,
     GradientBoosting, HistGradientBoosting, XGBoost if installed) with
     StratifiedGroupKFold(5) by player_id, tunes the top 2 with
     RandomizedSearchCV scored on AUC-PR (the literature's key metric at a
     ~10% positive rate), and reports held-out-2022 plus
     leave-one-season-out generalisation.
  5. Saves new artifacts under new names (e.g. *_optimized.pkl,
     scaler_optimized.pkl, feature_columns_optimized.json,
     features_optimized.parquet) -- it never overwrites the original
     xgb.pkl/rf.pkl/lr.pkl/scaler.pkl/features.parquet, so nothing
     currently served by api_server.py can break.

-------------------------------------------------------------------------------
v3 addition: run_optimized_v2() -- genuine prematch forecast + richer evaluation
-------------------------------------------------------------------------------
run_optimized() (v2 above) predicts "is_injured_next_30d" from a mix of
(a) information only knowable once the player has actually played this match
(minutes_played, xg, xa, pressures, tackles, carry_distance, interceptions,
clearances, sub_minute) and (b) information knowable before kickoff (age,
prior workload, rest, injury history, the announced lineup position) -- so
it's a post-appearance risk read, not a real forecast. run_optimized_v2() is
additive again (nothing above is touched) and builds a model using ONLY (b)
-- SAMEMATCH_ONLY_FEATURES is dropped entirely -- plus several new
prior-appearance features: starts/full-90s/avg/max minutes over the last 5
appearances, a playing streak restricted to STARTS, a substitute-pattern
rate, prior injury count, an injury-recency bucket, age-squared, fixture
congestion, and the announced lineup position (treated as pre-match-known
since starting XIs are published before kickoff in practice, unlike
sub_minute, which is an in-game event and stays excluded).

(An earlier version of this pipeline also built a second "samematch" mode
that kept the (a)+(b) mix, to compare directly against this prematch model.
By request it was removed -- this file now ships the prematch model only;
see git history if the samematch comparison needs revisiting.)

The pipeline runs its own multicollinearity audit, an imbalance-strategy
ablation (raw / class_weight / SMOTE / SMOTEENN / RandomUnderSampler --
selected by AUC-PR, not AUC-ROC, since the question that matters at a ~10%
positive rate is precision/recall tradeoff, not ranking ability), model
comparison + RandomizedSearchCV tuning (including BalancedRandomForestClassifier
when available), calibration (Brier score + CalibratedClassifierCV applied to
the winner only -- wrapping every candidate in calibration would multiply an
already expensive sweep), a threshold table for four practical risk tiers,
and held-out-2022 + leave-one-season-out checks. Position/categorical
features go through a ColumnTransformer (OneHotEncoder) INSIDE the pipeline,
so SMOTE/SMOTEENN/RandomUnderSampler and the scaler still only ever see the
training fold.

Saves model3_prematch_best.pkl / scaler_prematch.pkl /
feature_columns_prematch.json / thresholds_prematch.json, and
model3_optimized_diagnostics_v2.txt. Does not touch v1's or v2's artifacts.

Run: python -m models.model3_injury_risk --optimize        (v2, samematch)
     python -m models.model3_injury_risk --optimize-v2     (v3, prematch)
"""

import os
# Must run before numpy/sklearn/xgboost import -- on this machine,
# unconstrained per-process BLAS threading combined with joblib's own
# parallelism (RandomizedSearchCV, SMOTE's nearest-neighbour search) caused
# OpenBLAS memory-allocation crashes on the larger ingestion workloads.
# main.py pins these for the API/ingestion entrypoint; this file is run
# standalone via `python -m models.model3_injury_risk`, so it needs its own.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import inspect
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
)
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, StratifiedGroupKFold, RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import joblib

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

from models.eval_utils import (
    grouped_cv_clf_multi, holdout_season, leave_one_season_out_clf, TEST_SEASON,
    threshold_table, recall_at_precision, precision_at_recall, calibration_summary,
)
from models.model2_team_cohesion import audit_multicollinearity, _night_style, _apply_night

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    _HAS_BRF = True
except ImportError:
    _HAS_BRF = False

logger = logging.getLogger(__name__)

FEATURES = [
    "minutes_played",
    "matches_last_30_days",
    "minutes_last_30_days",
    "days_since_last_injury",
    "age_at_match",
    "sub_minute_flag",
    "xg",
    "xa",
    "pressures",
    "tackles",
    "carry_distance",
    "interceptions",
    "clearances",
]


def load_features(conn) -> pd.DataFrame:
    query = """
        SELECT
            pms.stat_id,
            pms.player_id,
            pms.match_id,
            pms.minutes_played,
            pmf.matches_last_30_days,
            pmf.minutes_last_30_days,
            pmf.days_since_last_injury,
            pms.sub_minute,
            pms.xg,
            pms.xa,
            pms.pressures,
            pms.tackles,
            pms.carry_distance,
            pms.interceptions,
            pms.clearances,
            pmf.is_injured_next_30d        AS label,
            COALESCE(
                EXTRACT(YEAR FROM AGE(m.match_date, p.date_of_birth))::INT,
                25
            ) AS age_at_match
        FROM player_match_stats pms
        JOIN player_match_features pmf ON pmf.stat_id  = pms.stat_id
        JOIN matches m                 ON m.match_id   = pms.match_id
        JOIN players p                 ON p.player_id  = pms.player_id
        WHERE pms.minutes_played >= 1
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)

    df["sub_minute_flag"] = df["sub_minute"].notna().astype(int)
    df["days_since_last_injury"] = df["days_since_last_injury"].fillna(-1)

    return df


def preprocess(df: pd.DataFrame):
    X = df[FEATURES].fillna(0).values
    y = df["label"].astype(int).values
    return X, y


def run(conn, output_dir: str = "artifacts/model3") -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Model 3: loading features ...")
    df = load_features(conn)
    logger.info("  %d rows  |  positive rate: %.2f%%",
                len(df), df["label"].mean() * 100)

    X, y = preprocess(df)

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = dict(zip(classes, weights))
    logger.info("Class weights: %s", class_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Wrap each estimator in a Pipeline with its own scaler so that
    # cross_val_score fits the scaler only on training folds. This prevents
    # test-fold data from leaking into the scaler's mean/variance, which
    # previously inflated reported CV AUC scores.
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight=class_weight, max_iter=1000, random_state=42
        )),
    ])
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight=class_weight,
            random_state=42, n_jobs=-1
        )),
    ])

    lr_auc = cross_val_score(lr_pipe, X, y, cv=cv, scoring="roc_auc")
    logger.info("LR AUC-ROC (5-fold): %.3f +/- %.3f", lr_auc.mean(), lr_auc.std())

    rf_auc = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc")
    logger.info("RF  AUC-ROC (5-fold): %.3f +/- %.3f", rf_auc.mean(), rf_auc.std())

    xgb_auc = None
    try:
        from xgboost import XGBClassifier
        scale_pos = (y == 0).sum() / max(1, (y == 1).sum())
        xgb_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                scale_pos_weight=scale_pos, eval_metric="aucpr",
                random_state=42, n_jobs=-1, verbosity=0,
            )),
        ])
        xgb_auc = cross_val_score(xgb_pipe, X, y, cv=cv, scoring="roc_auc")
        logger.info("XGB AUC-ROC (5-fold): %.3f +/- %.3f",
                    xgb_auc.mean(), xgb_auc.std())
        xgb_pipe.fit(X, y)
        # Save only the fitted estimator and a separate scaler for inference
        # so api_server.py can call scaler.transform() + model.predict_proba()
        # without depending on sklearn Pipeline internals.
        joblib.dump(xgb_pipe.named_steps["clf"],    f"{output_dir}/xgb.pkl")
        joblib.dump(xgb_pipe.named_steps["scaler"], f"{output_dir}/scaler.pkl")
        xgb = xgb_pipe.named_steps["clf"]
    except ImportError:
        logger.warning("xgboost not installed -- skipping XGBClassifier")
        xgb = None

    # Fit final LR and RF on full data for completeness; primary artifact
    # is XGB when available, RF otherwise.
    scaler_final = StandardScaler()
    X_sc = scaler_final.fit_transform(X)

    lr = LogisticRegression(
        class_weight=class_weight, max_iter=1000, random_state=42
    )
    lr.fit(X_sc, y)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight=class_weight,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_sc, y)

    importances = pd.Series(rf.feature_importances_, index=FEATURES)
    logger.info("RF feature importances:\n%s",
                importances.sort_values(ascending=False))

    # Only write scaler.pkl from the full-data fit if XGB was not available
    # (XGB pipeline already wrote it above with a properly CV-isolated scaler).
    if xgb is None:
        joblib.dump(scaler_final, f"{output_dir}/scaler.pkl")

    joblib.dump(lr, f"{output_dir}/lr.pkl")
    joblib.dump(rf, f"{output_dir}/rf.pkl")
    df.to_parquet(f"{output_dir}/features.parquet", index=False)

    logger.info("Model 3 artifacts saved to %s", output_dir)

    metrics = {
        "lr_roc_auc": float(lr_auc.mean()),
        "lr_roc_auc_std": float(lr_auc.std()),
        "rf_roc_auc": float(rf_auc.mean()),
        "rf_roc_auc_std": float(rf_auc.std()),
        "positive_rate": float(df["label"].mean()),
        "primary": "xgb" if xgb is not None else "rf",
    }
    if xgb_auc is not None:
        metrics["xgb_roc_auc"] = float(xgb_auc.mean())
        metrics["xgb_roc_auc_std"] = float(xgb_auc.std())
    metrics["feature_importances"] = {
        f: float(v) for f, v in zip(FEATURES, rf.feature_importances_)
    }

    return {
        "lr": lr, "rf": rf, "xgb": xgb, "scaler": scaler_final,
        "_registry": {
            "model_key": "model3_injury_risk",
            "version": "1.0",
            "display_name": "Injury Risk",
            "task": "classification",
            "algorithm": "XGBoost / RandomForest / LogisticRegression (balanced)",
            "target": "is_injured_next_30d",
            "features": list(FEATURES),
            "metrics": metrics,
            "n_train_rows": int(len(df)),
            "artifact_path": output_dir,
            "prediction_table": "model3_features",
        },
        "_predictions": {"model3_features": df},
    }


# ══════════════════════════════════════════════════════════════════════════
# v2: ENGINEERED WORKLOAD / RECOVERY FEATURES
# ══════════════════════════════════════════════════════════════════════════

GAP_RESET_DAYS = 45    # a gap this large in a TEAM's own calendar resets a
                       # playing streak -- our `matches` table is StatsBomb's
                       # open-data subset, not a team's true continuous
                       # fixture list, so e.g. a national team's 2018 and 2022
                       # World Cup matches are adjacent rows here despite being
                       # years apart in real life.
REST_CAP_DAYS = 60     # cap days_rest_before_match -- beyond ~2 months a
                       # player is "fully rested" for fatigue purposes; the
                       # raw value can run into the thousands (years between
                       # a club season and a later international tournament).
INJURY_CAP_DAYS = 180  # PDF fix #5: cap days_since_last_injury's right-skew.

NEW_FEATURES = [
    "matches_last_7_days", "minutes_last_7_days",
    "matches_last_14_days", "minutes_last_14_days",
    "consecutive_matches_played", "days_rest_before_match",
    "days_since_last_injury_capped",
]
EXPANDED_FEATURES = FEATURES + NEW_FEATURES

# Tie-break order for the multicollinearity audit: shorter, more proximate
# workload windows are favoured over longer ones (acute load is the more
# sensitive fatigue signal in the sports-science literature), and the
# de-skewed injury-recency feature is favoured over the raw one.
FEATURE_KEEP_PRIORITY = [
    "days_since_last_injury_capped", "days_since_last_injury",
    "matches_last_7_days", "minutes_last_7_days",
    "matches_last_14_days", "minutes_last_14_days",
    "matches_last_30_days", "minutes_last_30_days",
    "consecutive_matches_played", "days_rest_before_match",
    "minutes_played", "age_at_match", "sub_minute_flag",
    "xg", "xa", "pressures", "tackles", "carry_distance",
    "interceptions", "clearances",
]


def _team_match_calendar(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Every (team_id, match_id, match_date) a team played, home or away."""
    home = matches_df[["match_id", "match_date", "home_team_id"]].rename(
        columns={"home_team_id": "team_id"})
    away = matches_df[["match_id", "match_date", "away_team_id"]].rename(
        columns={"away_team_id": "team_id"})
    cal = pd.concat([home, away], ignore_index=True)
    return cal.sort_values(["team_id", "match_date", "match_id"]).reset_index(drop=True)


def _add_streak_and_rest(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Adds consecutive_matches_played and days_rest_before_match, both computed
    from matches STRICTLY BEFORE the current one -- the same leakage-safe
    convention pipelines/compute_labels.py already uses for matches_last_30_days.

    consecutive_matches_played resets to 0 when the player misses one of
    their team's fixtures, OR when the team's own calendar has a gap larger
    than GAP_RESET_DAYS (see constant above).
    """
    df = df.sort_values(["player_id", "team_id", "match_date"]).reset_index(drop=True)
    n = len(df)
    streak_vals = np.zeros(n, dtype=int)
    rest_vals = np.full(n, -1.0)

    cal_by_team = {tid: g.sort_values("match_date") for tid, g in calendar.groupby("team_id")}
    for (player_id, team_id), idxs in df.groupby(["player_id", "team_id"]).indices.items():
        sched = cal_by_team[team_id]
        sched_mids = sched["match_id"].values
        sched_dates = sched["match_date"].values
        player_mids = set(df.loc[idxs, "match_id"])
        appeared = np.isin(sched_mids, list(player_mids)).astype(int)

        gap_days = np.r_[0, (sched_dates[1:] - sched_dates[:-1]).astype("timedelta64[D]").astype(float)]
        big_gap = gap_days > GAP_RESET_DAYS
        reset_groups = ((appeared == 0) | big_gap).cumsum()
        streak_incl = pd.Series(appeared).groupby(reset_groups).cumsum().values
        streak_incl = np.where(big_gap, 0, streak_incl)
        streak_before = np.r_[0, streak_incl[:-1]]

        pos_of_mid = {mid: i for i, mid in enumerate(sched_mids)}
        for ridx in idxs:
            pos = pos_of_mid[df.at[ridx, "match_id"]]
            streak_vals[ridx] = streak_before[pos]
            prior = np.where(appeared[:pos] == 1)[0]
            if len(prior):
                rest = (sched_dates[pos] - sched_dates[prior[-1]]).astype("timedelta64[D]").astype(float)
                rest_vals[ridx] = min(float(rest), REST_CAP_DAYS)

    df["consecutive_matches_played"] = streak_vals
    df["days_rest_before_match"] = rest_vals
    return df


def _add_rolling_workload(df: pd.DataFrame, windows=(7, 14)) -> pd.DataFrame:
    """matches_last_N_days / minutes_last_N_days, strictly prior to this match."""
    df = df.sort_values(["player_id", "match_date"]).reset_index(drop=True)
    for w in windows:
        m_vals = np.zeros(len(df))
        min_vals = np.zeros(len(df))
        for player_id, idxs in df.groupby("player_id").indices.items():
            sub_dates = df.loc[idxs, "match_date"].values
            sub_mins = df.loc[idxs, "minutes_played"].values
            for i, ridx in enumerate(idxs):
                mask = (sub_dates < sub_dates[i]) & (sub_dates >= sub_dates[i] - np.timedelta64(w, "D"))
                m_vals[ridx] = mask.sum()
                min_vals[ridx] = sub_mins[mask].sum()
        df[f"matches_last_{w}_days"] = m_vals
        df[f"minutes_last_{w}_days"] = min_vals
    return df


def load_features_optimized(conn) -> pd.DataFrame:
    """
    Same base rows as load_features(), plus team_id/match_date/season and
    the engineered workload/recovery features in NEW_FEATURES.
    """
    df = load_features(conn)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT pms.stat_id, pms.team_id, m.match_date, m.season
            FROM player_match_stats pms
            JOIN matches m ON m.match_id = pms.match_id
            WHERE pms.minutes_played >= 1
        """)
        cols = [d[0] for d in cur.description]
        extra = pd.DataFrame(cur.fetchall(), columns=cols)

        cur.execute("SELECT match_id, match_date, home_team_id, away_team_id FROM matches")
        cols2 = [d[0] for d in cur.description]
        matches_df = pd.DataFrame(cur.fetchall(), columns=cols2)

    df = df.merge(extra, on="stat_id", how="left")
    df["match_date"] = pd.to_datetime(df["match_date"])
    matches_df["match_date"] = pd.to_datetime(matches_df["match_date"])

    calendar = _team_match_calendar(matches_df)
    df = _add_streak_and_rest(df, calendar)
    df = _add_rolling_workload(df, windows=(7, 14))

    df["days_since_last_injury_capped"] = np.where(
        df["days_since_last_injury"] == -1, -1,
        np.minimum(df["days_since_last_injury"], INJURY_CAP_DAYS),
    ).astype(float)

    return df


def preprocess_optimized(df: pd.DataFrame, feature_list: List[str]):
    X = df[feature_list].fillna(0).values
    y = df["label"].astype(int).values
    groups = df["player_id"].values
    seasons = df["season"].values
    return X, y, groups, seasons


# ══════════════════════════════════════════════════════════════════════════
# v2: MODEL REGISTRY (baseline kwargs + RandomizedSearchCV param grids)
# Estimators are kept single-threaded (no internal n_jobs); the OUTER
# RandomizedSearchCV/cross_validate calls are the only level of parallelism
# allowed at once, to avoid the joblib x BLAS process-explosion that crashed
# this machine before (see the env-var pinning at the top of this file).
# ══════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, tuple] = {
    "LogisticRegression": (LogisticRegression, {"max_iter": 2000, "random_state": 42}, {
        "est__C": loguniform(1e-3, 1e2),
    }),
    "RandomForest": (RandomForestClassifier, {"random_state": 42}, {
        "est__n_estimators": randint(100, 300),
        "est__max_depth": randint(3, 16),
        "est__min_samples_leaf": randint(1, 30),
    }),
    "ExtraTrees": (ExtraTreesClassifier, {"random_state": 42}, {
        "est__n_estimators": randint(100, 300),
        "est__max_depth": randint(3, 16),
        "est__min_samples_leaf": randint(1, 30),
    }),
    "GradientBoosting": (GradientBoostingClassifier, {"random_state": 42}, {
        "est__n_estimators": randint(100, 250),
        "est__max_depth": randint(2, 5),
        "est__learning_rate": loguniform(1e-2, 3e-1),
        "est__min_samples_leaf": randint(1, 30),
        "est__subsample": uniform(0.6, 0.4),
    }),
    "HistGradientBoosting": (HistGradientBoostingClassifier, {"random_state": 42}, {
        "est__max_iter": randint(100, 300),
        "est__max_leaf_nodes": randint(8, 64),
        "est__learning_rate": loguniform(1e-2, 3e-1),
        "est__min_samples_leaf": randint(5, 40),
        "est__l2_regularization": loguniform(1e-3, 1e1),
    }),
}
if _HAS_XGB:
    MODEL_REGISTRY["XGB"] = (
        XGBClassifier,
        {"random_state": 42, "n_jobs": 1, "eval_metric": "aucpr", "verbosity": 0},
        {
            "est__n_estimators": randint(100, 300),
            "est__max_depth": randint(2, 8),
            "est__learning_rate": loguniform(1e-2, 3e-1),
            "est__subsample": uniform(0.6, 0.4),
            "est__colsample_bytree": uniform(0.5, 0.5),
            "est__min_child_weight": randint(1, 10),
        },
    )


def _build_pipeline(estimator_cls, kwargs):
    """SMOTE only ever sees the training fold: imblearn's Pipeline (unlike
    sklearn's) skips the sampler step at predict/score time automatically."""
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("est", estimator_cls(**kwargs)),
    ])


def _imbalance_ablation(X, y, groups) -> Dict[str, dict]:
    """
    Tests raw / class_weight="balanced" / SMOTE on a fixed algorithm
    (RandomForest, since it supports all three) under StratifiedGroupKFold
    by player -- so the model-comparison section's choice of strategy is
    measured, not assumed.
    """
    def rf(**kw):
        return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, **kw)

    results = {}
    raw_pipe = Pipeline([("scaler", StandardScaler()), ("est", rf())])
    results["raw"] = grouped_cv_clf_multi(raw_pipe, X, y, groups)

    cw_pipe = Pipeline([("scaler", StandardScaler()), ("est", rf(class_weight="balanced"))])
    results["class_weight"] = grouped_cv_clf_multi(cw_pipe, X, y, groups)

    smote_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("est", rf()),
    ])
    results["smote"] = grouped_cv_clf_multi(smote_pipe, X, y, groups)
    return results


# ══════════════════════════════════════════════════════════════════════════
# v2: FIGURES (same dark "night" style as models/model1_player_clustering.py
# and models/model2_team_cohesion.py, reusing its helpers directly)
# ══════════════════════════════════════════════════════════════════════════

def make_optimized_figures(
    df: pd.DataFrame,
    candidate_features: List[str],
    dropped: List[dict],
    kept_features: List[str],
    ablation: Dict[str, dict],
    comparison_df: pd.DataFrame,
    best_meta: Dict[str, Any],
    final_est, scaler, X: np.ndarray, y: np.ndarray,
    importances: pd.Series,
    perm_importances: pd.Series,
    output_dir: str,
) -> None:
    """Six diagnostic figures for the optimized model, written to output_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed -- skipping figures.")
        return

    N = _night_style()
    plt.rcParams.update({
        "figure.facecolor": N["bg"], "axes.facecolor": N["surface"],
        "axes.edgecolor": N["grid"], "axes.labelcolor": N["muted"],
        "xtick.color": N["muted"], "ytick.color": N["muted"],
        "text.color": N["text"], "grid.color": N["grid"],
        "grid.linestyle": "--", "grid.linewidth": 0.4,
        "font.family": "DejaVu Sans",
    })
    out = Path(output_dir)

    # ── Fig 1: correlation heatmap of all 20 candidate features ─────────────
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    corr = df[candidate_features].fillna(0).corr(method="spearman")
    im = ax1.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    dropped_set = {d["dropped"] for d in dropped}
    ax1.set_xticks(range(len(candidate_features)))
    ax1.set_yticks(range(len(candidate_features)))
    ax1.set_xticklabels(candidate_features, rotation=60, ha="right", fontsize=8)
    ax1.set_yticklabels(candidate_features, fontsize=8)
    for tick, f in zip(ax1.get_xticklabels(), candidate_features):
        tick.set_color(N["coral"] if f in dropped_set else N["text"])
    for tick, f in zip(ax1.get_yticklabels(), candidate_features):
        tick.set_color(N["coral"] if f in dropped_set else N["text"])
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label("Spearman rho", color=N["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=N["muted"])
    ax1.set_title("Feature Correlation Matrix\n(20 candidates -- red labels = dropped by the multicollinearity audit)",
                  color=N["text"], fontweight="bold", fontsize=11)
    _apply_night(fig1)
    plt.tight_layout()
    fig1.savefig(out / "fig1_feature_correlation_heatmap.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig1)

    # ── Fig 2: imbalance-handling ablation ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    strategies = list(ablation.keys())
    x = np.arange(len(strategies))
    width = 0.32
    roc_vals = [ablation[s]["roc_auc_mean"] for s in strategies]
    roc_err  = [ablation[s]["roc_auc_std"] for s in strategies]
    pr_vals  = [ablation[s]["pr_auc_mean"] for s in strategies]
    pr_err   = [ablation[s]["pr_auc_std"] for s in strategies]
    ax2.bar(x - width / 2, roc_vals, width, yerr=roc_err, label="AUC-ROC",
            color=N["teal"], edgecolor=N["bg"], capsize=3)
    ax2.bar(x + width / 2, pr_vals, width, yerr=pr_err, label="AUC-PR",
            color=N["coral"], edgecolor=N["bg"], capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies)
    ax2.set_ylabel("StratifiedGroupKFold(5) score, by player_id")
    ax2.set_title("Imbalance-Handling Ablation (RandomForest)", fontweight="bold")
    leg = ax2.legend(fontsize=9, framealpha=0.2, labelcolor=N["text"])
    leg.get_frame().set_facecolor(N["surface"])
    _apply_night(fig2)
    plt.tight_layout()
    fig2.savefig(out / "fig2_imbalance_ablation.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig2)

    # ── Fig 3: model comparison, untuned baseline ────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    models_order = comparison_df["model"].tolist()
    x = np.arange(len(models_order))
    width = 0.32
    ax3.bar(x - width / 2, comparison_df["roc_auc_mean"], width,
            yerr=comparison_df["roc_auc_std"], label="AUC-ROC",
            color=N["teal"], edgecolor=N["bg"], capsize=3)
    ax3.bar(x + width / 2, comparison_df["pr_auc_mean"], width,
            yerr=comparison_df["pr_auc_std"], label="AUC-PR",
            color=N["coral"], edgecolor=N["bg"], capsize=3)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models_order, rotation=20, ha="right")
    ax3.set_title("Model Comparison -- Untuned Baseline (SMOTE)", fontweight="bold")
    leg = ax3.legend(fontsize=9, framealpha=0.2, labelcolor=N["text"])
    leg.get_frame().set_facecolor(N["surface"])
    _apply_night(fig3)
    plt.tight_layout()
    fig3.savefig(out / "fig3_model_comparison.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig3)

    # ── Fig 4: native + permutation feature importance, side by side ────────
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 7))
    imp_sorted = importances.sort_values()
    ax4a.barh(range(len(imp_sorted)), imp_sorted.values, color=N["teal"], edgecolor=N["bg"])
    ax4a.set_yticks(range(len(imp_sorted)))
    ax4a.set_yticklabels(imp_sorted.index, fontsize=8)
    ax4a.set_xlabel("Native importance")
    ax4a.set_title(f"Feature Importance -- {best_meta['model']}", fontweight="bold")

    perm_sorted = perm_importances.reindex(imp_sorted.index)
    ax4b.barh(range(len(perm_sorted)), perm_sorted.values, color=N["coral"], edgecolor=N["bg"])
    ax4b.set_yticks(range(len(perm_sorted)))
    ax4b.set_yticklabels(perm_sorted.index, fontsize=8)
    ax4b.set_xlabel("Permutation importance (AUC-PR drop, in-sample)")
    ax4b.set_title("Permutation Importance", fontweight="bold")
    _apply_night(fig4, [ax4a, ax4b])
    plt.tight_layout()
    fig4.savefig(out / "fig4_feature_importance.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig4)

    # ── Fig 5: ROC + Precision-Recall curves for the final model (in-sample) ─
    proba = final_est.predict_proba(scaler.transform(X))[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)
    pos_rate = float(y.mean())

    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(13, 6))
    ax5a.plot(fpr, tpr, color=N["teal"], linewidth=2)
    ax5a.plot([0, 1], [0, 1], color=N["grid"], linestyle="--", linewidth=1)
    ax5a.set_xlabel("False positive rate")
    ax5a.set_ylabel("True positive rate")
    ax5a.set_title("ROC Curve (in-sample)", fontweight="bold")

    ax5b.plot(rec, prec, color=N["coral"], linewidth=2)
    ax5b.axhline(pos_rate, color=N["grid"], linestyle="--", linewidth=1,
                 label=f"random baseline ({pos_rate:.3f})")
    ax5b.set_xlabel("Recall")
    ax5b.set_ylabel("Precision")
    ax5b.set_title("Precision-Recall Curve (in-sample)", fontweight="bold")
    leg = ax5b.legend(fontsize=8, framealpha=0.2, labelcolor=N["text"])
    leg.get_frame().set_facecolor(N["surface"])
    _apply_night(fig5, [ax5a, ax5b])
    plt.tight_layout()
    fig5.savefig(out / "fig5_roc_pr_curves.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig5)

    # ── Fig 6: leave-one-season-out + held-out generalisation (AUC-PR) ──────
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    loso = [r for r in best_meta.get("loso", []) if not r.get("skipped")]
    season_labels = [str(r["season"]) for r in loso]
    pr_vals = [r["pr_auc"] for r in loso]
    bar_colors = [N["amber"] if s == TEST_SEASON else N["teal"] for s in season_labels]
    ax6.bar(season_labels, pr_vals, color=bar_colors, edgecolor=N["bg"])
    ax6.axhline(best_meta["pr_auc_cv"], color=N["coral"], linestyle="--", linewidth=1.5,
                label=f"Grouped CV AUC-PR={best_meta['pr_auc_cv']:.3f}")
    ax6.set_ylabel("AUC-PR (trained on all other seasons)")
    ax6.set_title("Leave-One-Season-Out Generalisation\n(amber = held-out 2022 World Cup)",
                  fontweight="bold", fontsize=11)
    leg = ax6.legend(fontsize=9, framealpha=0.2, labelcolor=N["text"])
    leg.get_frame().set_facecolor(N["surface"])
    _apply_night(fig6)
    plt.tight_layout()
    fig6.savefig(out / "fig6_holdout_loso_performance.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig6)

    logger.info("Figures saved to %s", out)


# ══════════════════════════════════════════════════════════════════════════
# v2: OPTIMIZED PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_optimized(conn, output_dir: str = "artifacts/model3") -> Dict[str, Any]:
    """
    Workload/recovery-engineered optimization of Model 3 (new features,
    multicollinearity audit, player-grouped CV, imbalance-handling ablation,
    model comparison, hyperparameter tuning, out-of-time generalisation).
    Does NOT touch the original xgb.pkl/rf.pkl/lr.pkl/scaler.pkl/
    features.parquet -- see module docstring.
    """
    warnings.filterwarnings("ignore")
    os.makedirs(output_dir, exist_ok=True)

    report: List[str] = []
    report_path = f"{output_dir}/model3_optimized_diagnostics.txt"

    def _flush():
        # Rewrite the report-so-far after every line. This run takes ~1 hour;
        # if a late section crashes (as one already has -- a missing
        # feature_importances_ attribute after a ~50 min tuning step), we
        # still have everything up to the crash on disk instead of losing it.
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

    def sec(title: str):
        report.append("\n" + "=" * 78)
        report.append(f"  {title}")
        report.append("=" * 78)
        _flush()
        logger.info(title)

    def pr(line: str = ""):
        report.append(str(line))
        _flush()

    sec("MODEL 3 OPTIMIZED -- INJURY RISK (workload/recovery features)")

    logger.info("Loading + engineering features ...")
    df = load_features_optimized(conn)
    pr(f"Player-match rows: {len(df):,}  |  positive rate: {df['label'].mean()*100:.2f}%")
    pr(f"Distinct players: {df['player_id'].nunique():,}")
    pr(f"Seasons: {sorted(df['season'].dropna().unique().tolist())}")

    # ── 1. Multicollinearity audit ────────────────────────────────────────
    sec(f"1. MULTICOLLINEARITY AUDIT ({len(EXPANDED_FEATURES)} candidate features)")
    kept, dropped, audit_lines = audit_multicollinearity(
        df, EXPANDED_FEATURES, FEATURE_KEEP_PRIORITY, threshold=0.90,
    )
    report.extend(audit_lines)
    _flush()
    pr(f"\nFinal feature set ({len(kept)}):")
    for f in kept:
        pr(f"  {f}")

    df.to_parquet(f"{output_dir}/features_optimized.parquet", index=False)
    X, y, groups, seasons = preprocess_optimized(df, kept)

    # ── 2. Imbalance-handling ablation ──────────────────────────────────────
    sec("2. IMBALANCE-HANDLING ABLATION (RandomForest, StratifiedGroupKFold by player)")
    ablation = _imbalance_ablation(X, y, groups)
    for strat, res in ablation.items():
        pr(f"  {strat:<14} AUC-ROC={res['roc_auc_mean']:.4f}+/-{res['roc_auc_std']:.4f}  "
           f"AUC-PR={res['pr_auc_mean']:.4f}+/-{res['pr_auc_std']:.4f}")
    best_strategy = max(ablation, key=lambda s: ablation[s]["pr_auc_mean"])
    pr(f"\n  Best strategy by AUC-PR: {best_strategy}")
    pr("  SMOTE is applied uniformly below regardless (matches the spec's explicit")
    pr("  request and works identically across every algorithm in the registry,")
    pr("  unlike class_weight which only some support) -- compare against the")
    pr("  ablation above to see whether it earns its keep on this data.")

    # ── 3. Model comparison (baseline, untuned), SMOTE + StratifiedGroupKFold
    sec("3. MODEL COMPARISON -- BASELINE (UNTUNED), StratifiedGroupKFold(5) BY PLAYER")
    comparison_rows = []
    for name, (cls, kwargs, _params) in MODEL_REGISTRY.items():
        pipe = _build_pipeline(cls, kwargs)
        res = grouped_cv_clf_multi(pipe, X, y, groups)
        comparison_rows.append({"model": name, **res})
        pr(f"  {name:<22} AUC-ROC={res['roc_auc_mean']:.4f}+/-{res['roc_auc_std']:.4f}  "
           f"AUC-PR={res['pr_auc_mean']:.4f}+/-{res['pr_auc_std']:.4f}")
    comparison_df = pd.DataFrame(comparison_rows)

    # ── 4. Hyperparameter tuning -- top 2 by AUC-PR ─────────────────────────
    sec("4. HYPERPARAMETER TUNING -- RandomizedSearchCV(n_iter=20), StratifiedGroupKFold(5)")
    top2 = comparison_df.sort_values("pr_auc_mean", ascending=False).head(2)["model"].tolist()
    pr(f"\nTuning candidates (top-2 baseline by AUC-PR): {top2}")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -np.inf
    best_meta: Dict[str, Any] = {}
    best_estimator = None
    for name in top2:
        cls, kwargs, param_dist = MODEL_REGISTRY[name]
        pipe = _build_pipeline(cls, kwargs)
        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=20,
            cv=cv, scoring="average_precision",
            random_state=42, n_jobs=4,
        )
        search.fit(X, y, groups=groups)
        tuned_params = {k.replace("est__", ""): v for k, v in search.best_params_.items()}
        pr(f"  {name:<22} tuned AUC-PR={search.best_score_:.4f}  params={tuned_params}")
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_estimator = search.best_estimator_
            best_meta = {
                "model": name,
                "pr_auc_cv": round(float(search.best_score_), 4),
                "params": tuned_params,
            }
    pr(f"  -> BEST: {best_meta['model']}  AUC-PR={best_meta['pr_auc_cv']}")

    # ── 5. Out-of-time generalisation ───────────────────────────────────────
    sec("5. OUT-OF-TIME GENERALISATION -- held-out 2022 + leave-one-season-out")
    ho_roc, n_ho = holdout_season(best_estimator, X, y, seasons, "roc_auc")
    ho_pr, _ = holdout_season(best_estimator, X, y, seasons, "average_precision")
    pr(f"\nHeld-out 2022 (n={n_ho}): AUC-ROC={ho_roc}  AUC-PR={ho_pr}")
    best_meta["holdout_2022_roc_auc"] = ho_roc
    best_meta["holdout_2022_pr_auc"] = ho_pr

    loso = leave_one_season_out_clf(best_estimator, X, y, seasons)
    best_meta["loso"] = loso
    for rec in loso:
        if rec.get("skipped"):
            pr(f"  season={rec['season']:<12} SKIPPED (n_test={rec['n_test']}, n_pos={rec.get('n_pos','?')})")
        else:
            pr(f"  season={rec['season']:<12} n={rec['n_test']:<5} n_pos={rec['n_pos']:<4} "
               f"AUC-ROC={rec['roc_auc']:.4f}  AUC-PR={rec['pr_auc']:.4f}")

    # ── 6. Final artifacts, feature importance, permutation importance ─────
    sec("6. FINAL ARTIFACTS, FEATURE IMPORTANCE & PERMUTATION IMPORTANCE")
    final_scaler = StandardScaler().fit(X)
    X_sc = final_scaler.transform(X)
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_sc, y)

    final_cls, final_kwargs, _ = MODEL_REGISTRY[best_meta["model"]]
    final_est = final_cls(**{**final_kwargs, **best_meta["params"]})
    final_est.fit(X_res, y_res)

    artifact_stem = best_meta["model"].lower()
    joblib.dump(final_est, f"{output_dir}/{artifact_stem}_optimized.pkl")
    joblib.dump(final_scaler, f"{output_dir}/scaler_optimized.pkl")
    with open(f"{output_dir}/feature_columns_optimized.json", "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2)

    pr(f"\nFinal model: {best_meta['model']}  (saved {artifact_stem}_optimized.pkl, "
       f"scaler_optimized.pkl, feature_columns_optimized.json)")
    pr("Trained on SMOTE-resampled data; scaler.transform() + model.predict_proba()")
    pr("at inference time never sees synthetic rows -- SMOTE only ever touches training.")

    perm = permutation_importance(
        final_est, X_res, y_res, scoring="average_precision",
        n_repeats=10, random_state=42, n_jobs=4,
    )
    perm_s = pd.Series(perm.importances_mean, index=kept).sort_values(ascending=False)

    if hasattr(final_est, "feature_importances_"):
        imp = pd.Series(final_est.feature_importances_, index=kept).sort_values(ascending=False)
        pr("  Native feature importances:")
        for f, v in imp.items():
            pr(f"     {v*100:5.1f}%  {f}")
    elif hasattr(final_est, "coef_"):
        imp = pd.Series(np.abs(final_est.coef_[0]), index=kept).sort_values(ascending=False)
        pr("  |Coefficient| ranking (LogisticRegression):")
        for f, v in imp.items():
            pr(f"     {v:7.4f}  {f}")
    else:
        # HistGradientBoostingClassifier exposes neither feature_importances_
        # nor coef_ -- permutation importance is the only ranking available.
        imp = perm_s
        pr(f"  No native importance API on {best_meta['model']} -- using permutation")
        pr("  importance (below) as the feature ranking too.")

    pr("  Permutation importance (in-sample AUC-PR drop -- diagnostic, not held-out):")
    for f, v in perm_s.items():
        pr(f"    {v:+.4f}  {f}")
    best_meta["permutation_importance"] = {f: round(float(v), 4) for f, v in perm_s.items()}

    with open(f"{output_dir}/model3_optimized_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "dropped_features": dropped,
            "kept_features": kept,
            "imbalance_ablation": ablation,
            "best_imbalance_strategy": best_strategy,
            "model_comparison": comparison_rows,
            "best_model": best_meta,
        }, f, indent=2, default=str)

    try:
        make_optimized_figures(
            df, EXPANDED_FEATURES, dropped, kept, ablation, comparison_df, best_meta,
            final_est, final_scaler, X, y, imp, perm_s, output_dir,
        )
    except Exception as exc:
        logger.warning("Figures skipped due to an error: %s", exc)
        pr(f"\n(Figures skipped due to an error: {exc})")

    # ── 7. Limitations / leakage notes ──────────────────────────────────────
    sec("7. LIMITATIONS / LEAKAGE NOTES")
    pr("- consecutive_matches_played and the rolling 7/14-day windows are computed")
    pr("  from matches strictly BEFORE the current one, matching the existing")
    pr("  matches_last_30_days convention -- no same-match leakage.")
    pr("- This is still a same-match model: minutes_played, xg, xa, pressures etc.")
    pr("  for THIS match are used as features (same framing as the original model),")
    pr("  it is not a pure pre-match forecast.")
    pr("- consecutive_matches_played is bounded by what's IN this StatsBomb subset,")
    pr("  not a player's true career fixture list -- the GAP_RESET_DAYS=45 cutoff")
    pr("  limits but doesn't eliminate this; treat it as directional.")
    pr("- 2015/16 league season dominates the row count; leave-one-season-out")
    pr("  results on the small seasons are directional, not precise.")
    pr("- Permutation importance above is computed on the same (SMOTE-resampled)")
    pr("  data the final model was fit on (in-sample) -- it shows what the model")
    pr("  leans on, not an unbiased held-out estimate.")
    pr("- Existing artifacts (xgb.pkl, rf.pkl, lr.pkl, scaler.pkl, features.parquet)")
    pr("  and api_server.py are byte-for-byte untouched by this pipeline.")

    logger.info("Optimized diagnostics report saved to %s", report_path)

    return {
        "best_estimator": best_estimator,
        "best_meta": best_meta,
        "df": df,
        "comparison_df": comparison_df,
        "dropped_features": dropped,
        "kept_features": kept,
    }


# ══════════════════════════════════════════════════════════════════════════
# v3: PREMATCH PIPELINE -- genuine before-kickoff forecast
# ══════════════════════════════════════════════════════════════════════════

# Features only knowable once the player has actually appeared in THIS
# match -- excluded entirely from the prematch model. sub_minute_flag is the
# literal in-game substitution event and is excluded too, beyond the user's
# explicit list, for the same reason.
SAMEMATCH_ONLY_FEATURES = {
    "minutes_played", "xg", "xa", "pressures", "tackles",
    "carry_distance", "interceptions", "clearances", "sub_minute_flag",
}

CATEGORICAL_FEATURES_V2 = ["position_group", "injury_recency_bucket"]

COUNT_WINDOW_FEATURES = [
    "starts_last_5_matches", "full_90s_last_5_matches",
    "avg_minutes_last_5", "max_minutes_last_5",
    "consecutive_starts", "recent_substitute_pattern",
]
INJURY_HISTORY_FEATURES = ["previous_injury_count"]
DERIVED_FEATURES_V2 = ["age_squared", "fixture_congestion_score"]

ALL_NUMERIC_CANDIDATES_V2 = (
    EXPANDED_FEATURES + COUNT_WINDOW_FEATURES + INJURY_HISTORY_FEATURES + DERIVED_FEATURES_V2
)

# Tie-break priority for the v2 multicollinearity audit. Extends
# FEATURE_KEEP_PRIORITY (v1) with the new candidates; entries missing from
# this list default to "drop first" in audit_multicollinearity(), so every
# new feature is listed explicitly rather than relying on that default.
FEATURE_KEEP_PRIORITY_V2 = FEATURE_KEEP_PRIORITY + [
    "previous_injury_count", "age_squared", "fixture_congestion_score",
    "consecutive_starts", "recent_substitute_pattern",
    "starts_last_5_matches", "full_90s_last_5_matches",
    "avg_minutes_last_5", "max_minutes_last_5",
]

RISK_TIER_QUANTILES = {"Low": 0.0, "Medium": 0.50, "High": 0.80, "Very High": 0.95}


def _map_position_group(starting_position) -> str:
    """
    Collapses StatsBomb's ~25 granular starting_position labels into
    GK/DEF/MID/FWD. Order matters: "Wing Back" must hit the Back-> DEF rule
    before the Wing-> FWD fallback, so DEF is checked before the fallback.
    """
    if not starting_position or pd.isna(starting_position):
        return "unknown"
    s = str(starting_position)
    if "Goalkeeper" in s:
        return "GK"
    if "Back" in s:
        return "DEF"
    if "Midfield" in s:
        return "MID"
    return "FWD"  # Wing, Forward, Striker, and any unmapped remainder


def _add_count_window_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Adds 5-prior-appearance window features (strictly before the current
    match, same convention as the rest of this file): starts_last_5_matches,
    full_90s_last_5_matches, avg_minutes_last_5, max_minutes_last_5,
    recent_substitute_pattern (fraction of those 5 entered as a sub), and
    consecutive_starts (a playing streak like consecutive_matches_played,
    but only counting STARTS, reset by either a missed team fixture, an
    appearance that was a sub appearance, or a big calendar gap).
    """
    df = df.sort_values(["player_id", "match_date"]).reset_index(drop=True)
    is_start = 1 - df["sub_minute_flag"].values
    n = len(df)
    starts5 = np.zeros(n)
    full90s5 = np.zeros(n)
    avg_min5 = np.zeros(n)
    max_min5 = np.zeros(n)
    sub_pattern5 = np.zeros(n)
    consec_starts = np.zeros(n, dtype=int)

    for player_id, idxs in df.groupby("player_id").indices.items():
        mins = df.loc[idxs, "minutes_played"].values
        starts = is_start[idxs]
        sub_flags = df.loc[idxs, "sub_minute_flag"].values
        run_start_streak = 0
        for i in range(len(idxs)):
            lo = max(0, i - window)
            prior_mins = mins[lo:i]
            prior_starts = starts[lo:i]
            prior_subs = sub_flags[lo:i]
            if len(prior_mins) > 0:
                starts5[idxs[i]] = prior_starts.sum()
                full90s5[idxs[i]] = (prior_mins >= 90).sum()
                avg_min5[idxs[i]] = prior_mins.mean()
                max_min5[idxs[i]] = prior_mins.max()
                sub_pattern5[idxs[i]] = prior_subs.mean()
            consec_starts[idxs[i]] = run_start_streak
            run_start_streak = run_start_streak + 1 if starts[i] == 1 else 0

    df["starts_last_5_matches"] = starts5
    df["full_90s_last_5_matches"] = full90s5
    df["avg_minutes_last_5"] = avg_min5
    df["max_minutes_last_5"] = max_min5
    df["recent_substitute_pattern"] = sub_pattern5
    df["consecutive_starts"] = consec_starts
    return df


def _add_injury_history_features(df: pd.DataFrame, conn) -> pd.DataFrame:
    """
    previous_injury_count: cumulative count of injuries STRICTLY BEFORE this
    match's date (a proneness proxy, distinct from days_since_last_injury's
    recency signal). injury_recency_bucket: a categorical coarsening of
    days_since_last_injury_capped, kept alongside the continuous version per
    spec for interpretability even though it's informationally redundant.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT player_id, injury_date FROM injuries")
        cols = [d[0] for d in cur.description]
        inj = pd.DataFrame(cur.fetchall(), columns=cols)
    inj["injury_date"] = pd.to_datetime(inj["injury_date"])

    counts = np.zeros(len(df), dtype=int)
    inj_by_player = {pid: np.sort(g["injury_date"].values) for pid, g in inj.groupby("player_id")}
    for player_id, idxs in df.groupby("player_id").indices.items():
        dates = inj_by_player.get(player_id)
        if dates is None:
            continue
        match_dates = df.loc[idxs, "match_date"].values
        counts[idxs] = np.searchsorted(dates, match_dates, side="left")
    df["previous_injury_count"] = counts

    capped = df["days_since_last_injury_capped"].values
    bucket = np.select(
        [capped == -1, capped <= 30, capped <= 90, capped <= 180],
        ["no_history", "0-30d", "31-90d", "91-180d"],
        default="91-180d",
    )
    df["injury_recency_bucket"] = bucket
    return df


def load_features_v2(conn) -> pd.DataFrame:
    """load_features_optimized() plus the v3 features: position_group,
    count-window features, injury history, age_squared, fixture_congestion."""
    df = load_features_optimized(conn)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT pms.stat_id, pms.starting_position
            FROM player_match_stats pms
        """)
        cols = [d[0] for d in cur.description]
        pos = pd.DataFrame(cur.fetchall(), columns=cols)
    df = df.merge(pos, on="stat_id", how="left")
    df["position_group"] = df["starting_position"].apply(_map_position_group)

    df = _add_count_window_features(df, window=5)
    df = _add_injury_history_features(df, conn)

    df["age_squared"] = df["age_at_match"].astype(float) ** 2
    # Simple, documented density heuristic (matches-per-day over the last 7
    # and 14 days combined) -- not a validated clinical workload index, just
    # a compact proxy for short-term fixture congestion.
    df["fixture_congestion_score"] = (
        df["matches_last_7_days"] / 7.0 + df["matches_last_14_days"] / 14.0
    )
    return df


def preprocess_v2(df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]):
    X = df[numeric_features + categorical_features].copy()
    X[numeric_features] = X[numeric_features].fillna(0)
    X[categorical_features] = X[categorical_features].fillna("unknown")
    y = df["label"].astype(int).values
    groups = df["player_id"].values
    seasons = df["season"].values
    return X, y, groups, seasons


def _build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Scaling + one-hot encoding INSIDE the pipeline (as a ColumnTransformer
    step), so any sampler (SMOTE etc.) downstream still only ever sees
    training-fold data, same as the plain StandardScaler in v1/v2."""
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])


# ── Model registry (extends MODEL_REGISTRY with max_features / class_weight
#    tuning for the tree ensembles, and BalancedRandomForestClassifier) ────
MODEL_REGISTRY_V2: Dict[str, tuple] = {
    "LogisticRegression": (LogisticRegression, {"max_iter": 2000, "random_state": 42}, {
        "est__C": loguniform(1e-3, 1e2),
        "est__class_weight": [None, "balanced"],
    }),
    "RandomForest": (RandomForestClassifier, {"random_state": 42}, {
        **MODEL_REGISTRY["RandomForest"][2],
        "est__max_features": uniform(0.3, 0.7),
        "est__class_weight": [None, "balanced"],
    }),
    "ExtraTrees": (ExtraTreesClassifier, {"random_state": 42}, {
        **MODEL_REGISTRY["ExtraTrees"][2],
        "est__max_features": uniform(0.3, 0.7),
        "est__class_weight": [None, "balanced"],
    }),
    "GradientBoosting": MODEL_REGISTRY["GradientBoosting"],
    "HistGradientBoosting": MODEL_REGISTRY["HistGradientBoosting"],
}
if _HAS_XGB:
    MODEL_REGISTRY_V2["XGB"] = MODEL_REGISTRY["XGB"]
if _HAS_BRF:
    # Does its OWN internal balancing (bootstrap resampling per tree) -- never
    # paired with an EXTERNAL sampler in the comparison/tuning loops below.
    MODEL_REGISTRY_V2["BalancedRandomForest"] = (
        BalancedRandomForestClassifier, {"random_state": 42}, {
            "est__n_estimators": randint(100, 300),
            "est__max_depth": randint(3, 16),
            "est__min_samples_leaf": randint(1, 30),
            "est__max_features": uniform(0.3, 0.7),
        },
    )


def _imbalance_ablation_v2(X, y, groups, numeric_features: List[str],
                            categorical_features: List[str]) -> Dict[str, dict]:
    """
    Tests raw / class_weight="balanced" / SMOTE / SMOTEENN / RandomUnderSampler
    on a fixed algorithm (RandomForest) under StratifiedGroupKFold by player.
    Ranked by AUC-PR, not AUC-ROC: at a ~10% positive rate, AUC-ROC barely
    moves between strategies (it's dominated by how well the negatives rank
    against each other, which there are a lot of) while precision/recall --
    what actually changes when you resample -- shows up in AUC-PR.
    """
    def rf(**kw):
        return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, **kw)

    pre = _build_preprocessor(numeric_features, categorical_features)
    results = {}

    raw_pipe = ImbPipeline([("prep", pre), ("est", rf())])
    results["raw"] = grouped_cv_clf_multi(raw_pipe, X, y, groups)

    cw_pipe = ImbPipeline([("prep", pre), ("est", rf(class_weight="balanced"))])
    results["class_weight"] = grouped_cv_clf_multi(cw_pipe, X, y, groups)

    smote_pipe = ImbPipeline([("prep", pre), ("sampler", SMOTE(random_state=42)), ("est", rf())])
    results["smote"] = grouped_cv_clf_multi(smote_pipe, X, y, groups)

    smoteenn_pipe = ImbPipeline([("prep", pre), ("sampler", SMOTEENN(random_state=42)), ("est", rf())])
    results["smoteenn"] = grouped_cv_clf_multi(smoteenn_pipe, X, y, groups)

    rus_pipe = ImbPipeline([("prep", pre), ("sampler", RandomUnderSampler(random_state=42)), ("est", rf())])
    results["random_undersample"] = grouped_cv_clf_multi(rus_pipe, X, y, groups)

    return results


def _supports_class_weight(cls) -> bool:
    return "class_weight" in inspect.signature(cls.__init__).parameters


def _evaluate_mode(
    mode_name: str, mode_label: str, df: pd.DataFrame,
    candidate_numeric: List[str], categorical_features: List[str],
    keep_priority: List[str], pr, sec, output_dir: str,
) -> Dict[str, Any]:
    """
    Runs the full audit -> imbalance ablation -> model comparison -> tuning
    -> calibration -> out-of-time generalisation -> threshold table pipeline
    for ONE mode (samematch or prematch), writing its section of the report
    via `pr`/`sec` and saving that mode's artifacts under mode_name. Returns
    a result dict used for the final cross-mode summary in run_optimized_v2.
    """
    sec(f"MODE: {mode_label}")
    pr(f"Candidate numeric features: {len(candidate_numeric)}  |  "
       f"categorical features: {categorical_features}")

    # ── 1. Multicollinearity audit (numeric only -- categorical features
    #      aren't continuous, so Spearman correlation doesn't apply; they're
    #      always kept) ──────────────────────────────────────────────────
    kept_numeric, dropped, audit_lines = audit_multicollinearity(
        df, candidate_numeric, keep_priority, threshold=0.90,
    )
    pr("")
    for line in audit_lines:
        pr(line)
    kept_features = kept_numeric + categorical_features
    pr(f"\nFinal feature set ({len(kept_features)}): {kept_features}")

    X, y, groups, seasons = preprocess_v2(df, kept_numeric, categorical_features)
    pos_rate = float(y.mean())

    # ── 2. Imbalance-handling ablation ───────────────────────────────────
    pr("\nImbalance-handling ablation (RandomForest, StratifiedGroupKFold by player):")
    ablation = _imbalance_ablation_v2(X, y, groups, kept_numeric, categorical_features)
    for strat, res in ablation.items():
        pr(f"  {strat:<20} AUC-ROC={res['roc_auc_mean']:.4f}+/-{res['roc_auc_std']:.4f}  "
           f"AUC-PR={res['pr_auc_mean']:.4f}+/-{res['pr_auc_std']:.4f}")
    best_strategy = max(ablation, key=lambda s: ablation[s]["pr_auc_mean"])
    pr(f"  -> Best strategy by AUC-PR: {best_strategy}")
    pr("  (Ranked on AUC-PR, not AUC-ROC: at a ~10% positive rate AUC-ROC barely")
    pr("   moves between strategies since it's dominated by how the large negative")
    pr("   class ranks against itself; AUC-PR is what actually reflects whether")
    pr("   resampling changed the precision/recall tradeoff that matters here.)")

    sampler_map = {
        "raw": None, "class_weight": None,
        "smote": SMOTE(random_state=42),
        "smoteenn": SMOTEENN(random_state=42),
        "random_undersample": RandomUnderSampler(random_state=42),
    }
    chosen_sampler = sampler_map[best_strategy]
    use_class_weight = best_strategy == "class_weight"

    # ── 3. Baseline + model comparison (untuned), chosen strategy ───────
    pre = _build_preprocessor(kept_numeric, categorical_features)
    dummy_pipe = ImbPipeline([("prep", pre), ("est", DummyClassifier(strategy="prior"))])
    dummy_res = grouped_cv_clf_multi(dummy_pipe, X, y, groups)
    pr(f"\nBaseline (always predict the majority class): AUC-ROC={dummy_res['roc_auc_mean']:.4f}  "
       f"AUC-PR={dummy_res['pr_auc_mean']:.4f}  (~= positive rate {pos_rate:.4f})")

    pr(f"\nModel comparison -- baseline (untuned), StratifiedGroupKFold(5) by player, "
       f"strategy={best_strategy}:")
    comparison_rows = [{"model": "Baseline(prior)", **dummy_res}]
    for name, (cls, kwargs, _params) in MODEL_REGISTRY_V2.items():
        is_brf = name == "BalancedRandomForest"
        kw = dict(kwargs)
        sampler = None if is_brf else chosen_sampler
        if use_class_weight and not is_brf and _supports_class_weight(cls):
            kw["class_weight"] = "balanced"
        steps = [("prep", pre)]
        if sampler is not None:
            steps.append(("sampler", sampler))
        steps.append(("est", cls(**kw)))
        pipe = ImbPipeline(steps)
        res = grouped_cv_clf_multi(pipe, X, y, groups)
        comparison_rows.append({"model": name, **res})
        pr(f"  {name:<22} AUC-ROC={res['roc_auc_mean']:.4f}+/-{res['roc_auc_std']:.4f}  "
           f"AUC-PR={res['pr_auc_mean']:.4f}+/-{res['pr_auc_std']:.4f}")
    comparison_df = pd.DataFrame(comparison_rows)

    # ── 4. Hyperparameter tuning -- top 2 by AUC-PR ─────────────────────
    top2 = (comparison_df[comparison_df["model"] != "Baseline(prior)"]
            .sort_values("pr_auc_mean", ascending=False).head(2)["model"].tolist())
    pr(f"\nTuning candidates (top-2 baseline by AUC-PR): {top2}")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -np.inf
    best_meta: Dict[str, Any] = {}
    best_pipe = None
    for name in top2:
        cls, kwargs, param_dist = MODEL_REGISTRY_V2[name]
        is_brf = name == "BalancedRandomForest"
        sampler = None if is_brf else chosen_sampler
        steps = [("prep", pre)]
        if sampler is not None:
            steps.append(("sampler", sampler))
        steps.append(("est", cls(**kwargs)))
        pipe = ImbPipeline(steps)
        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=20,
            cv=cv, scoring="average_precision", random_state=42, n_jobs=4,
        )
        search.fit(X, y, groups=groups)
        tuned_params = {k.replace("est__", ""): v for k, v in search.best_params_.items()}
        pr(f"  {name:<22} tuned AUC-PR={search.best_score_:.4f}  params={tuned_params}")
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_pipe = search.best_estimator_
            best_meta = {
                "model": name, "pr_auc_cv": round(float(search.best_score_), 4),
                "params": tuned_params, "imbalance_strategy": best_strategy,
            }
    pr(f"  -> BEST: {best_meta['model']}  AUC-PR={best_meta['pr_auc_cv']}")

    # ── 5. Out-of-time generalisation ────────────────────────────────────
    ho_roc, n_ho = holdout_season(best_pipe, X, y, seasons, "roc_auc")
    ho_pr, _ = holdout_season(best_pipe, X, y, seasons, "average_precision")
    pr(f"\nHeld-out 2022 (n={n_ho}): AUC-ROC={ho_roc}  AUC-PR={ho_pr}")
    best_meta["holdout_2022_roc_auc"] = ho_roc
    best_meta["holdout_2022_pr_auc"] = ho_pr

    loso = leave_one_season_out_clf(best_pipe, X, y, seasons)
    best_meta["loso"] = loso
    for rec in loso:
        if rec.get("skipped"):
            pr(f"  season={rec['season']:<12} SKIPPED (n_test={rec['n_test']}, n_pos={rec.get('n_pos','?')})")
        else:
            pr(f"  season={rec['season']:<12} n={rec['n_test']:<5} n_pos={rec['n_pos']:<4} "
               f"AUC-ROC={rec['roc_auc']:.4f}  AUC-PR={rec['pr_auc']:.4f}")

    # ── 6. Calibration check (out-of-fold, grouped) ──────────────────────
    pr("\nCalibration check (out-of-fold predictions, StratifiedGroupKFold by player):")
    oof_uncal = cross_val_predict(best_pipe, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    calib_uncal = calibration_summary(y, oof_uncal)

    calibrated_template = CalibratedClassifierCV(clone(best_pipe), method="isotonic", cv=3)
    oof_cal = cross_val_predict(calibrated_template, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    calib_cal = calibration_summary(y, oof_cal)

    use_calibration = calib_cal["brier_score"] < calib_uncal["brier_score"]
    pr(f"  Uncalibrated Brier score:          {calib_uncal['brier_score']:.4f}")
    pr(f"  Isotonic-calibrated Brier score:   {calib_cal['brier_score']:.4f}")
    pr(f"  -> {'Using the calibrated model' if use_calibration else 'Keeping the uncalibrated model'} "
       f"for the saved artifact (lower Brier wins).")
    pr("  Calibration bins (predicted mean vs. observed rate), chosen model:")
    for b in (calib_cal["bins"] if use_calibration else calib_uncal["bins"]):
        pr(f"    predicted~{b['predicted_mean']:.3f}  observed={b['observed_rate']:.3f}")
    best_meta["brier_uncalibrated"] = calib_uncal["brier_score"]
    best_meta["brier_calibrated"] = calib_cal["brier_score"]
    best_meta["used_calibration"] = use_calibration

    oof_final = oof_cal if use_calibration else oof_uncal

    # ── 7. Extra evaluation metrics at default threshold + tradeoff points ─
    pr("\nThreshold table (out-of-fold probabilities, every 0.05):")
    pr(f"  {'thr':>5}  {'prec':>6}  {'rec':>6}  {'f1':>6}  {'bal_acc':>8}  {'flag%':>7}")
    thresh_tbl = threshold_table(y, oof_final, list(np.arange(0.05, 0.96, 0.05)))
    for row in thresh_tbl:
        pr(f"  {row['threshold']:.2f}  {row['precision']:6.3f}  {row['recall']:6.3f}  "
           f"{row['f1']:6.3f}  {row['balanced_accuracy']:8.3f}  {row['flagged_rate']*100:6.1f}%")

    r_p50, p_at_r_p50, t_r_p50 = recall_at_precision(y, oof_final, 0.50)
    p_r50, r_at_p_r50, t_p_r50 = precision_at_recall(y, oof_final, 0.50)
    pr(f"\nRecall at precision>=0.50: "
       f"{'n/a (not achievable)' if r_p50 is None else f'{r_p50:.3f} (threshold={t_r_p50:.3f})'}")
    pr(f"Precision at recall>=0.50: "
       f"{'n/a (not achievable)' if p_r50 is None else f'{p_r50:.3f} (threshold={t_p_r50:.3f})'}")
    best_meta["recall_at_precision_50"] = r_p50
    best_meta["precision_at_recall_50"] = p_r50

    # ── 8. Four practical risk tiers from OOF score quantiles ───────────
    pr("\nRisk tiers (cut points from out-of-fold score quantiles, with the")
    pr("actual observed injury rate inside each tier -- the practically useful")
    pr("number for deciding what 'High risk' should mean):")
    tier_names = list(RISK_TIER_QUANTILES.keys())
    cut_points = [float(np.quantile(oof_final, q)) for q in RISK_TIER_QUANTILES.values()]
    thresholds_json: Dict[str, Any] = {}
    for i, name in enumerate(tier_names):
        lo = cut_points[i]
        hi = cut_points[i + 1] if i + 1 < len(tier_names) else float(oof_final.max())
        mask = (oof_final >= lo) & ((oof_final < hi) if i + 1 < len(tier_names) else (oof_final <= hi))
        n_in_tier = int(mask.sum())
        rate_in_tier = float(y[mask].mean()) if n_in_tier else float("nan")
        pr(f"  {name:<10} score in [{lo:.3f}, {hi:.3f}{']' if i+1==len(tier_names) else ')'}  "
           f"n={n_in_tier:<6}  observed injury rate={rate_in_tier:.3f}")
        thresholds_json[name] = {
            "score_low": lo, "score_high": hi, "n": n_in_tier,
            "observed_injury_rate": rate_in_tier,
        }
    best_meta["risk_tiers"] = thresholds_json

    # ── 9. Final fit on full data + artifacts ────────────────────────────
    final_prep = clone(pre).fit(X, y)
    X_transformed = final_prep.transform(X)
    is_brf_final = best_meta["model"] == "BalancedRandomForest"
    final_sampler = None if is_brf_final else chosen_sampler
    if final_sampler is not None:
        X_res, y_res = clone(final_sampler).fit_resample(X_transformed, y)
    else:
        X_res, y_res = X_transformed, y

    cls, kwargs, _ = MODEL_REGISTRY_V2[best_meta["model"]]
    plain_est = cls(**{**kwargs, **best_meta["params"]})
    if use_calibration:
        final_est = CalibratedClassifierCV(clone(plain_est), method="isotonic", cv=3)
    else:
        final_est = plain_est
    final_est.fit(X_res, y_res)

    # Feature importance is read from an UNCALIBRATED refit for interpretability
    # (CalibratedClassifierCV holds several inner-fold base estimators, not one
    # coherent set of importances) -- this is independent of which one is served.
    importance_est = clone(plain_est).fit(X_res, y_res)
    feature_names_out = list(final_prep.get_feature_names_out())
    if hasattr(importance_est, "feature_importances_"):
        imp = pd.Series(importance_est.feature_importances_, index=feature_names_out).sort_values(ascending=False)
        imp_label = "Native feature importances"
    elif hasattr(importance_est, "coef_"):
        imp = pd.Series(np.abs(importance_est.coef_[0]), index=feature_names_out).sort_values(ascending=False)
        imp_label = "|Coefficient| ranking"
    else:
        imp = None
        imp_label = None

    perm = permutation_importance(
        importance_est, X_res, y_res, scoring="average_precision",
        n_repeats=10, random_state=42, n_jobs=4,
    )
    perm_s = pd.Series(perm.importances_mean, index=feature_names_out).sort_values(ascending=False)
    if imp is None:
        imp, imp_label = perm_s, "Permutation importance (no native importance API on this model)"

    pr(f"\nFinal model: {best_meta['model']}"
       f"{' + isotonic calibration' if use_calibration else ''}"
       f"  (saved model3_{mode_name}_best.pkl, scaler_{mode_name}.pkl, "
       f"feature_columns_{mode_name}.json, thresholds_{mode_name}.json)")
    pr(f"  {imp_label}:")
    for f, v in imp.items():
        pr(f"     {v:.4f}  {f}")
    pr("  Permutation importance (in-sample on the resampled training data --")
    pr("  diagnostic, not a held-out estimate):")
    for f, v in perm_s.items():
        pr(f"    {v:+.4f}  {f}")

    joblib.dump(final_prep, f"{output_dir}/scaler_{mode_name}.pkl")
    joblib.dump(final_est, f"{output_dir}/model3_{mode_name}_best.pkl")
    with open(f"{output_dir}/feature_columns_{mode_name}.json", "w", encoding="utf-8") as f:
        json.dump(kept_features, f, indent=2)
    with open(f"{output_dir}/thresholds_{mode_name}.json", "w", encoding="utf-8") as f:
        json.dump(thresholds_json, f, indent=2, default=str)

    return {
        "mode_name": mode_name, "kept_features": kept_features, "dropped": dropped,
        "ablation": ablation, "best_strategy": best_strategy,
        "comparison_rows": comparison_rows, "best_meta": best_meta,
        "positive_rate": pos_rate, "n_rows": len(df),
    }


def run_optimized_v2(conn, output_dir: str = "artifacts/model3") -> Dict[str, Any]:
    """
    Single-mode (prematch) optimization of Model 3: a genuine before-kickoff
    forecast that excludes every same-match performance feature
    (SAMEMATCH_ONLY_FEATURES). An earlier version of this pipeline also built
    a "samematch" mode that mixed in this match's own xg/xa/pressures/etc.;
    by request it was removed so this file only ships the prematch model --
    see git history / a prior model3_optimized_diagnostics_v2.txt if the
    samematch comparison ever needs to be revisited. Does NOT touch v1's or
    v2's artifacts.
    """
    warnings.filterwarnings("ignore")
    os.makedirs(output_dir, exist_ok=True)

    report: List[str] = []
    report_path = f"{output_dir}/model3_optimized_diagnostics_v2.txt"

    def _flush():
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

    def sec(title: str):
        report.append("\n" + "=" * 78)
        report.append(f"  {title}")
        report.append("=" * 78)
        _flush()
        logger.info(title)

    def pr(line: str = ""):
        report.append(str(line))
        _flush()

    sec("MODEL 3 OPTIMIZED v2 -- PREMATCH INJURY RISK (genuine before-kickoff forecast)")

    logger.info("Loading + engineering v3 features ...")
    df = load_features_v2(conn)
    pr(f"Player-match rows: {len(df):,}  |  positive rate: {df['label'].mean()*100:.2f}%")
    pr(f"Distinct players: {df['player_id'].nunique():,}")
    pr(f"Seasons: {sorted(df['season'].dropna().unique().tolist())}")
    season_pos = df.groupby("season")["label"].agg(["count", "sum"])
    pr("\nPositives by season (seasons with 0 positives are skipped in leave-one-season-out):")
    for s, row in season_pos.iterrows():
        pr(f"  {s:<12} n={int(row['count']):<6} positives={int(row['sum'])}")

    candidates = [f for f in ALL_NUMERIC_CANDIDATES_V2 if f not in SAMEMATCH_ONLY_FEATURES]

    result = _evaluate_mode(
        "prematch", f"PREMATCH FEATURE SET (excludes {sorted(SAMEMATCH_ONLY_FEATURES)})", df,
        candidates, CATEGORICAL_FEATURES_V2, FEATURE_KEEP_PRIORITY_V2,
        pr, sec, output_dir,
    )

    sec("LEAKAGE & SCOPE NOTES")
    pr("- This model excludes minutes_played/xg/xa/pressures/tackles/carry_distance/")
    pr("  interceptions/clearances/sub_minute_flag for THIS match -- only information")
    pr("  knowable before kickoff is used. position_group is treated as pre-match-known")
    pr("  (starting XIs are published before kickoff in practice) -- the one judgment")
    pr("  call beyond the user's explicit exclusion list, documented here rather")
    pr("  than silently assumed.")
    pr("- All rolling/count-window/streak features use matches strictly BEFORE")
    pr("  the current one (same convention as pipelines/compute_labels.py's")
    pr("  matches_last_30_days), including the new starts_last_5_matches,")
    pr("  consecutive_starts, recent_substitute_pattern and previous_injury_count.")
    pr("- consecutive_starts/consecutive_matches_played are bounded by what's IN")
    pr("  this StatsBomb subset, not a player's true career fixture list.")
    pr("- fixture_congestion_score is a documented heuristic (matches-per-day")
    pr("  over the last 7+14 days), not a validated clinical workload index.")
    pr("- Calibration's inner CalibratedClassifierCV split (cv=3) is NOT grouped")
    pr("  by player -- a deliberate, documented scope simplification for this")
    pr("  secondary diagnostic; the OUTER evaluation that decides whether")
    pr("  calibration helps (cross_val_predict with StratifiedGroupKFold) is")
    pr("  fully grouped and leakage-safe.")
    pr("- Risk tiers and the threshold table are built from out-of-fold")
    pr("  (StratifiedGroupKFold) probabilities -- not in-sample -- so the")
    pr("  observed injury rate per tier is an honest number, not an inflated one.")
    pr("- Permutation importance is computed on the resampled TRAINING data the")
    pr("  final model was fit on (in-sample) -- it shows what the model leans")
    pr("  on, not an unbiased held-out estimate.")
    pr("- v1 (xgb.pkl/rf.pkl/lr.pkl/scaler.pkl/features.parquet), v2")
    pr("  (*_optimized.pkl) and api_server.py are byte-for-byte untouched.")

    sec("FINAL INTERPRETATION")
    pos_rate = df["label"].mean()
    bm = result["best_meta"]
    pr(f"At a {pos_rate*100:.1f}% positive rate, a baseline that always predicts")
    pr(f"'no injury' gets ~{1-pos_rate:.3f} accuracy while being useless --")
    pr("AUC-PR against that baseline is the honest yardstick used throughout.")
    pr(f"Tuned grouped-CV AUC-PR: {bm['pr_auc_cv']:.4f}; held-out 2022 AUC-PR: {bm['holdout_2022_pr_auc']:.4f}.")
    pr("Neither AUC-PR is high in absolute terms -- this remains a modest-signal")
    pr("problem (injuries have large random components no feature set here")
    pr("captures: contact incidents, opponent actions, training-ground load not")
    pr("in this data). The honest gains over the original same-match model are: a")
    pr("methodologically correct prematch option that can actually be used")
    pr("before kickoff, calibrated probabilities checked against a real")
    pr("alternative (not assumed), an imbalance strategy chosen by evidence")
    pr("instead of habit, and risk tiers grounded in observed outcome rates")
    pr("instead of an arbitrary 0.5 cutoff. That is the improvement being")
    pr("claimed -- not a leap in raw predictive power.")

    with open(f"{output_dir}/model3_optimized_v2_metadata.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info("Optimized v2 diagnostics report saved to %s", report_path)
    return result


if __name__ == "__main__":
    import argparse
    import psycopg2
    from config.settings import DB_DSN

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true",
                         help="Run the v2 optimized pipeline instead of the original run().")
    parser.add_argument("--optimize-v2", action="store_true",
                         help="Run the v3 prematch (before-kickoff forecast) optimized pipeline.")
    args = parser.parse_args()

    conn = psycopg2.connect(DB_DSN)
    if args.optimize_v2:
        run_optimized_v2(conn)
    elif args.optimize:
        run_optimized(conn)
    else:
        run(conn)
    conn.close()
