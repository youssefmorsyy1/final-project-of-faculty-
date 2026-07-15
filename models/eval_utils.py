"""
models/eval_utils.py

Shared honest-evaluation helpers for the supervised models.

Why this exists
---------------
The original models used plain shuffled k-fold cross-validation. Because the
rows are not independent -- the two team-rows of a match, every minute-snapshot
of a match, and every player-season of the same player share information -- a
shuffled split puts near-duplicates of a row in both the train and test folds.
That leaks and inflates the reported metrics (most starkly Model 5B in-game:
0.89 shuffled vs 0.64 grouped-by-match).

These helpers provide:
  * grouped_cv()             -- (Stratified)GroupKFold cross-validation grouped
                                by match_id, so no match straddles the
                                train/test boundary.
  * grouped_cv_multi()       -- same grouping, but returns R2/MAE/RMSE in one
                                pass (cross_validate instead of cross_val_score)
                                for model-comparison tables.
  * holdout_season()         -- a single out-of-time test on a held-out season
                                (FIFA World Cup 2022 by default), the
                                strictest generalisation check available here.
  * leave_one_season_out()   -- repeats holdout_season() for every season in
                                turn, so the 2022 result isn't read as the only
                                out-of-time evidence when one season (2015/16)
                                dominates the row count.
  * grouped_cv_clf_multi()   -- classification analogue of grouped_cv_multi():
                                StratifiedGroupKFold, returns AUC-ROC/AUC-PR.
  * leave_one_season_out_clf() -- classification analogue of
                                leave_one_season_out(): AUC-ROC/AUC-PR per
                                held-out season via predict_proba.
  * threshold_table()        -- precision/recall/F1/balanced-accuracy/confusion
                                matrix at a list of probability thresholds, the
                                basis for picking real-use risk-tier cutoffs
                                instead of defaulting to 0.5.
  * recall_at_precision() / precision_at_recall() -- find the operating point
                                that hits a target on the OTHER metric, since
                                with a ~10% positive rate AUC-PR-style tradeoffs
                                are what a real deployment threshold needs.
  * calibration_summary()    -- Brier score + binned calibration curve, so a
                                model isn't just ranking correctly (AUC) but
                                also producing probabilities that mean what
                                they say (P(injury)=0.3 actually injuring ~30%
                                of the time).
  * attach_season()          -- map match_id -> season onto a feature frame.

The StatsBomb free data is one full La Liga season (2015/16) + five
Barcelona-only La Liga seasons + two World Cups, so BALANCED_SEASONS marks the
slice with genuine team diversity (used to de-bias Model 5).
"""

import numpy as np
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, get_scorer, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    cross_val_score, cross_validate, GroupKFold, StratifiedGroupKFold,
)

# Held-out season for the out-of-time generalisation test.
TEST_SEASON = "2022"  # FIFA World Cup 2022 -- recent, balanced, out-of-distribution

# Seasons with league-wide team diversity (no Barcelona over-representation).
BALANCED_SEASONS = {"2015/2016", "2018", "2022"}


def season_of_matches(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT match_id, season FROM matches")
        return {int(m): s for m, s in cur.fetchall()}


def attach_season(df, conn, match_col: str = "match_id"):
    """Return a copy of df with a 'season' column mapped from match_id."""
    df = df.copy()
    df["season"] = df[match_col].map(season_of_matches(conn))
    return df


def grouped_cv(estimator, X, y, groups, scoring, n_splits: int = 5,
               stratified: bool = False):
    """
    Cross-validate with GroupKFold (or StratifiedGroupKFold for classifiers)
    grouped by `groups` (match_id). Returns (mean, std).
    """
    cv = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        if stratified else GroupKFold(n_splits=n_splits)
    )
    scores = cross_val_score(estimator, X, y, cv=cv, groups=groups, scoring=scoring)
    return scores.mean(), scores.std()


def holdout_season(estimator, X, y, seasons, scoring, test_season: str = TEST_SEASON):
    """
    Train on every season except `test_season`, evaluate once on it.
    Returns (score, n_test). Returns (None, n_test) if the split is degenerate.
    X may be a DataFrame (boolean row-masking works for both df and ndarray).
    """
    seasons = np.asarray(seasons)
    y = np.asarray(y)
    test = seasons == test_season
    if test.sum() == 0 or (~test).sum() == 0:
        return None, int(test.sum())
    est = clone(estimator).fit(X[~test], y[~test])
    score = get_scorer(scoring)(est, X[test], y[test])
    return float(score), int(test.sum())


def grouped_cv_multi(estimator, X, y, groups, n_splits: int = 5):
    """
    GroupKFold cross-validation returning R2, MAE and RMSE in a single pass
    (one fit per fold instead of one fit per metric). Returns a dict:
        {"r2_mean", "r2_std", "mae_mean", "mae_std", "rmse_mean", "rmse_std"}
    """
    cv = GroupKFold(n_splits=n_splits)
    scoring = {
        "r2":   "r2",
        "mae":  "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    res = cross_validate(estimator, X, y, cv=cv, groups=groups, scoring=scoring)
    return {
        "r2_mean":   float(res["test_r2"].mean()),
        "r2_std":    float(res["test_r2"].std()),
        "mae_mean":  float(-res["test_mae"].mean()),
        "mae_std":   float(res["test_mae"].std()),
        "rmse_mean": float(-res["test_rmse"].mean()),
        "rmse_std":  float(res["test_rmse"].std()),
    }


def leave_one_season_out(estimator, X, y, seasons, min_test_rows: int = 30):
    """
    Repeats holdout_season() for every distinct season present in `seasons`.

    Train on all-other-seasons, test once on the held-out season -- run in
    turn for each season. Seasons with fewer than `min_test_rows` rows are
    skipped (too few points for a stable R2 estimate) and reported as
    skipped rather than silently dropped.

    Returns a list of dicts: {"season", "n_test", "r2", "mae", "rmse"} or
    {"season", "n_test", "skipped": True} for thin seasons.
    """
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
            "season": season,
            "n_test": n_test,
            "r2":   float(r2_score(y[test], pred)),
            "mae":  float(mean_absolute_error(y[test], pred)),
            "rmse": float(np.sqrt(mean_squared_error(y[test], pred))),
        })
    return results


def grouped_cv_clf_multi(estimator, X, y, groups, n_splits: int = 5):
    """
    Classification analogue of grouped_cv_multi(): StratifiedGroupKFold
    (preserves class balance per fold AND keeps a group, e.g. player_id,
    from straddling train/test), returns AUC-ROC and AUC-PR in one pass.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
    res = cross_validate(estimator, X, y, cv=cv, groups=groups, scoring=scoring)
    return {
        "roc_auc_mean": float(res["test_roc_auc"].mean()),
        "roc_auc_std":  float(res["test_roc_auc"].std()),
        "pr_auc_mean":  float(res["test_pr_auc"].mean()),
        "pr_auc_std":   float(res["test_pr_auc"].std()),
    }


def leave_one_season_out_clf(estimator, X, y, seasons, min_test_rows: int = 30,
                              min_test_positives: int = 5):
    """
    Classification analogue of leave_one_season_out(): trains on all other
    seasons, scores AUC-ROC/AUC-PR on the held-out one via predict_proba.
    Seasons with too few rows OR too few positive examples (AUC is undefined
    with zero positives in the test fold) are skipped.
    """
    seasons_arr = np.asarray(seasons)
    y = np.asarray(y)
    results = []
    for season in sorted(set(seasons_arr.tolist())):
        test = seasons_arr == season
        n_test = int(test.sum())
        n_pos = int(y[test].sum())
        if n_test < min_test_rows or n_pos < min_test_positives or (~test).sum() == 0:
            results.append({"season": season, "n_test": n_test, "n_pos": n_pos, "skipped": True})
            continue
        est = clone(estimator).fit(X[~test], y[~test])
        proba = est.predict_proba(X[test])[:, 1]
        results.append({
            "season": season,
            "n_test": n_test,
            "n_pos": n_pos,
            "roc_auc": float(roc_auc_score(y[test], proba)),
            "pr_auc":  float(average_precision_score(y[test], proba)),
        })
    return results


def threshold_table(y_true, y_prob, thresholds):
    """
    Precision/recall/F1/balanced-accuracy/confusion-matrix at each of
    `thresholds`, so a deployment threshold can be picked deliberately
    instead of defaulting to 0.5 (which is a bad choice at a ~10% positive
    rate -- the model can have decent ranking ability and still predict
    "no injury" for almost everyone at 0.5).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rows = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, pred))
        rows.append({
            "threshold": float(t), "precision": precision, "recall": recall,
            "f1": f1, "balanced_accuracy": bal_acc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "flagged_rate": float(pred.mean()),
        })
    return rows


def recall_at_precision(y_true, y_prob, target_precision: float, n_steps: int = 500):
    """
    Highest recall achievable while keeping precision >= target_precision.
    Returns (recall, precision, threshold) or (None, None, None) if no
    threshold reaches the target (common when the model's top scores still
    aren't precise enough -- an honest "not achievable" result, not an error).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    best = (None, None, None)
    for t in np.linspace(0.0, 1.0, n_steps):
        pred = (y_prob >= t).astype(int)
        if pred.sum() == 0:
            continue
        p = precision_score(y_true, pred, zero_division=0)
        if p >= target_precision:
            r = recall_score(y_true, pred, zero_division=0)
            if best[0] is None or r > best[0]:
                best = (float(r), float(p), float(t))
    return best


def precision_at_recall(y_true, y_prob, target_recall: float, n_steps: int = 500):
    """
    Highest precision achievable while keeping recall >= target_recall.
    Returns (precision, recall, threshold) or (None, None, None).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    best = (None, None, None)
    for t in np.linspace(1.0, 0.0, n_steps):
        pred = (y_prob >= t).astype(int)
        r = recall_score(y_true, pred, zero_division=0)
        if r >= target_recall:
            p = precision_score(y_true, pred, zero_division=0)
            if best[0] is None or p > best[0]:
                best = (float(p), float(r), float(t))
    return best


def calibration_summary(y_true, y_prob, n_bins: int = 10):
    """
    Brier score (lower is better, 0=perfect) plus a binned calibration curve
    (mean predicted probability vs. observed positive rate per bin) -- shows
    whether the model's probabilities are usable as probabilities, which AUC
    alone never tells you (AUC only cares about rank order).
    """
    brier = float(brier_score_loss(y_true, y_prob))
    obs, pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    bins = [{"predicted_mean": float(p), "observed_rate": float(o)} for p, o in zip(pred, obs)]
    return {"brier_score": brier, "bins": bins}
