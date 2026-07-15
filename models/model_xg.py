"""
models/model_xg.py

From-scratch Expected Goals (xG) model.

Unlike the rest of the platform (which just *summed* StatsBomb's precomputed
statsbomb_xg), this trains a real probabilistic model: P(goal | shot context).
It is the canonical, well-validated football ML model and it generalises,
because shot outcome depends on shot characteristics that are stable across
leagues.

Features (from the `shots` table)
  geometry   : distance, angle
  context    : body_part, shot_type, technique, play_pattern,
               under_pressure, first_time
  freeze-frame: defenders_in_cone, dist_to_nearest_def,
               gk_dist_to_goal, gk_dist_to_shot

Honest evaluation
  * GroupKFold by match (out-of-fold probabilities).
  * Metrics are proper scoring rules: log-loss and Brier (not accuracy), plus
    ROC-AUC and a calibration check (predicted goals vs actual).
  * Benchmarked against StatsBomb's own xG and a naive constant-rate baseline.
  * Held-out season (FIFA World Cup 2022) for an out-of-distribution test.

Run:  python -m models.model_xg
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import joblib

logger = logging.getLogger(__name__)

NUMERIC = ["distance", "angle", "dist_to_nearest_def", "defenders_in_cone",
           "gk_dist_to_goal", "gk_dist_to_shot"]
CATEGORICAL = ["body_part", "shot_type", "technique", "play_pattern"]
BOOLEAN = ["under_pressure", "first_time"]
FEATURES = NUMERIC + CATEGORICAL + BOOLEAN


def load_shots(conn) -> pd.DataFrame:
    query = """
        SELECT s.shot_id, s.match_id, s.player_id, s.team_id,
               s.distance, s.angle, s.dist_to_nearest_def, s.defenders_in_cone,
               s.gk_dist_to_goal, s.gk_dist_to_shot,
               s.body_part, s.shot_type, s.technique, s.play_pattern,
               s.under_pressure, s.first_time,
               s.statsbomb_xg, s.is_goal,
               m.season
        FROM shots s
        JOIN matches m ON m.match_id = s.match_id
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    for b in BOOLEAN:
        df[b] = df[b].fillna(False).astype(int)
    for c in CATEGORICAL:
        df[c] = df[c].fillna("Unknown")
    return df


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("bool", "passthrough", BOOLEAN),
    ])
    clf = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=400,
        l2_regularization=1.0, random_state=42,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


def _report(name: str, y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    m = {
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
    }
    logger.info("  %-22s log-loss=%.4f  brier=%.4f  ROC-AUC=%.4f  "
                "pred_goals=%.0f (actual=%.0f)",
                name, m["log_loss"], m["brier"], m["auc"], p.sum(), y.sum())
    return m


def run(conn, output_dir: str = "artifacts/model_xg") -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)

    df = load_shots(conn)
    y = df["is_goal"].astype(int).values
    groups = df["match_id"].values
    logger.info("Model xG: %d shots | goal rate %.3f", len(df), y.mean())

    X = df[FEATURES]
    cv = GroupKFold(n_splits=5)

    # Out-of-fold probabilities (grouped by match) -- our model.
    oof = cross_val_predict(build_pipeline(), X, y, cv=cv, groups=groups,
                            method="predict_proba")[:, 1]

    logger.info("Out-of-fold evaluation (GroupKFold by match):")
    m_ours = _report("xG model (ours)", y, oof)

    # Benchmark: StatsBomb's professional xG on the same shots.
    sb = df["statsbomb_xg"].astype(float)
    mask = sb.notna().values
    m_sb = _report("StatsBomb xG (bench)", y[mask], sb.values[mask])

    # Naive baseline: constant league goal rate.
    m_naive = _report("naive (mean rate)", y, np.full_like(y, y.mean(), dtype=float))

    corr = np.corrcoef(oof[mask], sb.values[mask])[0, 1]
    logger.info("Correlation(our xG, StatsBomb xG) = %.3f", corr)

    # Held-out season (out-of-distribution generalisation).
    m_ho = None
    test = (df["season"] == "2022").values
    if test.sum() and (~test).sum():
        pipe = build_pipeline().fit(X[~test], y[~test])
        p_test = pipe.predict_proba(X[test])[:, 1]
        logger.info("Held-out WC-2022 (n=%d):", test.sum())
        m_ho = _report("xG model held-out", y[test], p_test)

    # Calibration by decile of predicted xG.
    logger.info("Calibration (decile of predicted xG -> actual goal rate):")
    dec = pd.qcut(oof, 10, duplicates="drop")
    cal = pd.DataFrame({"p": oof, "y": y}).groupby(dec, observed=True)
    for interval, g in cal:
        logger.info("  pred~%.3f  actual=%.3f  (n=%d)", g["p"].mean(), g["y"].mean(), len(g))

    # Fit final model on all shots; persist artifact + per-shot xG.
    final = build_pipeline().fit(X, y)
    joblib.dump(final, f"{output_dir}/xg_model.pkl")
    df["xg_pred"] = final.predict_proba(X)[:, 1]
    pred_cols = ["shot_id", "match_id", "player_id", "team_id", "distance",
                 "angle", "statsbomb_xg", "xg_pred", "is_goal"]
    pred_df = df[pred_cols].copy()
    pred_df.to_parquet(f"{output_dir}/shots_xg.parquet", index=False)
    logger.info("xG artifacts saved to %s", output_dir)

    metrics = {
        "roc_auc": m_ours["auc"],
        "log_loss": m_ours["log_loss"],
        "brier": m_ours["brier"],
        "statsbomb_roc_auc": m_sb["auc"],
        "statsbomb_log_loss": m_sb["log_loss"],
        "naive_log_loss": m_naive["log_loss"],
        "corr_with_statsbomb": float(corr),
        "goal_rate": float(y.mean()),
    }
    if m_ho is not None:
        metrics["heldout_2022_roc_auc"] = m_ho["auc"]
        metrics["heldout_2022_log_loss"] = m_ho["log_loss"]

    return {
        "model": final,
        "df": df,
        "_registry": {
            "model_key": "model_xg",
            "version": "1.0",
            "display_name": "Expected Goals (xG)",
            "task": "classification",
            "algorithm": "HistGradientBoostingClassifier (calibrated pipeline)",
            "target": "is_goal (P(goal | shot context))",
            "features": list(FEATURES),
            "metrics": metrics,
            "n_train_rows": int(len(df)),
            "artifact_path": output_dir,
            "prediction_table": "xg_shot_predictions",
        },
        "_predictions": {"xg_shot_predictions": pred_df},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import psycopg2
    from config.settings import DB_DSN
    conn = psycopg2.connect(DB_DSN)
    run(conn)
    conn.close()
