"""
Model 2 Deep Diagnostics — Team Cohesion (Graph + Regression)
Run: python -m models.model2_deep_diagnostics

Produces the same class of evidence as model1_deep_diagnostics.py but for
Model 2: feature distributions, multicollinearity, variance, graph-only vs
contextual vs full feature-set ablation, per-season breakdown, residuals,
and feature importance — written to artifacts/model2/model2_report.txt.

Uses the cached artifacts/model2/graph_features.parquet (built from a prior
live run against the populated DB) rather than re-querying Postgres, since
the local DB instance may be empty/reset between sessions while the cached
feature matrix already reflects the full StatsBomb dataset.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.model2_team_cohesion import GRAPH_FEATURES, CONTEXT_FEATURES, MODEL_FEATURES
from models.eval_utils import grouped_cv, holdout_season, TEST_SEASON

OUT = []
def sep(title=""):
    line = "\n" + "=" * 78
    OUT.append(line)
    if title:
        OUT.append(f"  {title}")
        OUT.append("=" * 78)
    print(line)
    if title:
        print(f"  {title}")
        print("=" * 78)

def pr(s=""):
    OUT.append(str(s))
    print(s)


FEAT_PATH = "artifacts/model2/graph_features.parquet"

# ── 1. RAW DATA ───────────────────────────────────────────────────────────
sep("1. RAW DATASET — cached team-match graph feature matrix")
feat_df = pd.read_parquet(FEAT_PATH)
pr(f"Source file                    : {FEAT_PATH}")
pr(f"Team-match graph rows           : {len(feat_df):,}")
pr(f"Distinct matches                : {feat_df['match_id'].nunique():,}")
pr(f"Mean nodes per graph (players)  : {feat_df['n_nodes'].mean():.2f}")
pr(f"Mean edges per graph            : {feat_df['n_edges'].mean():.2f}")
pr(f"Mean total passes per graph     : {feat_df['total_passes'].mean():.1f}")
pr(f"Goals distribution              : mean={feat_df['goals'].mean():.3f}  "
   f"std={feat_df['goals'].std():.3f}  max={feat_df['goals'].max():.0f}")
for c in CONTEXT_FEATURES:
    feat_df[c] = feat_df[c].fillna(0.0)
pr(f"\nSeasons covered                : {sorted(feat_df['season'].dropna().unique())}")

# ── 2. FEATURE DISTRIBUTIONS ─────────────────────────────────────────────
sep("2. FEATURE DISTRIBUTIONS — graph metrics + context")
desc = feat_df[MODEL_FEATURES + ["goals"]].describe().T[["mean", "std", "min", "50%", "max"]]
pr(desc.round(3).to_string())

zero_pct = (feat_df[MODEL_FEATURES] == 0).mean() * 100
high_zero = zero_pct[zero_pct > 20]
if len(high_zero):
    pr("\nFeatures with >20% zero values:")
    for f, p in high_zero.items():
        pr(f"  {f:<24} {p:.1f}% zeros")
else:
    pr("\nNo feature exceeds 20% zero-inflation.")

# ── 3. MULTICOLLINEARITY ─────────────────────────────────────────────────
sep("3. MULTICOLLINEARITY — |Spearman rho| > 0.75")
corr = feat_df[MODEL_FEATURES].corr(method="spearman").abs()
corr_vals = np.array(corr.values, copy=True)
np.fill_diagonal(corr_vals, 0)
corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)
pairs = (corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
              .stack().sort_values(ascending=False))
high = pairs[pairs > 0.75]
if len(high):
    for (a, b), v in high.items():
        pr(f"  rho={v:.3f}  {a}  <->  {b}")
else:
    pr("  None above 0.75 — no severe redundancy among the 16 features.")

# ── 4. VARIANCE AFTER SCALING ────────────────────────────────────────────
sep("4. VARIANCE AFTER STANDARD SCALING — weak discriminators")
Xall = feat_df[MODEL_FEATURES].fillna(0).values
Xsc = StandardScaler().fit_transform(Xall)
var = pd.Series(Xsc.var(axis=0), index=MODEL_FEATURES).sort_values()
pr(var.round(3).to_string())

# ── 5. UNIVARIATE CORRELATION WITH TARGET ────────────────────────────────
sep("5. UNIVARIATE SIGNAL — Spearman rho(feature, goals)")
target_corr = {}
for f in MODEL_FEATURES:
    rho, p = spearmanr(feat_df[f], feat_df["goals"])
    target_corr[f] = (rho, p)
tc = pd.DataFrame(target_corr, index=["rho", "p"]).T.sort_values("rho", key=abs, ascending=False)
for f, row in tc.iterrows():
    sig = "*" if row["p"] < 0.05 else " "
    pr(f"  {sig} {f:<24} rho={row['rho']:+.3f}  p={row['p']:.2e}")

# ── 6. MODEL ABLATION: graph-only vs context-only vs full ───────────────
sep("6. FEATURE-SET ABLATION — GroupKFold(5) by match_id, scoring=R2")
groups = feat_df["match_id"].values
y = feat_df["goals"].values.astype(float)
seasons = feat_df["season"].values

feature_sets = {
    "Baseline (mean predictor)": [],
    "Graph-only (12 topology features)": GRAPH_FEATURES,
    "Context-only (xg/xga/home/opp_quality)": CONTEXT_FEATURES,
    "Full (graph + context, 16 features)": MODEL_FEATURES,
}

ablation_rows = []
for name, feats in feature_sets.items():
    if name.startswith("Baseline"):
        dummy = DummyRegressor(strategy="mean")
        m, s = grouped_cv(dummy, np.zeros((len(y), 1)), y, groups, "r2")
        ridge_m, ridge_s, gbr_m, gbr_s = m, s, m, s
    else:
        X = feat_df[feats].fillna(0).values
        ridge_pipe = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=1.0))])
        gbr_pipe = Pipeline([("sc", StandardScaler()),
                              ("est", GradientBoostingRegressor(
                                  n_estimators=200, max_depth=3,
                                  learning_rate=0.05, random_state=42))])
        ridge_m, ridge_s = grouped_cv(ridge_pipe, X, y, groups, "r2")
        gbr_m, gbr_s = grouped_cv(gbr_pipe, X, y, groups, "r2")
    ablation_rows.append({"feature_set": name, "n_feat": len(feats),
                           "ridge_r2": ridge_m, "ridge_std": ridge_s,
                           "gbr_r2": gbr_m, "gbr_std": gbr_s})
    pr(f"  {name:<42} n_feat={len(feats):<3}  "
       f"Ridge R2={ridge_m:+.4f}+/-{ridge_s:.4f}   GBR R2={gbr_m:+.4f}+/-{gbr_s:.4f}")

ablation_df = pd.DataFrame(ablation_rows)

# ── 7. HELD-OUT SEASON GENERALISATION ────────────────────────────────────
sep(f"7. OUT-OF-TIME GENERALISATION — held-out season {TEST_SEASON}")
n_test_season = int((seasons == TEST_SEASON).sum())
pr(f"Held-out rows (season={TEST_SEASON}): {n_test_season:,}  "
   f"of {len(feat_df):,} total ({n_test_season/len(feat_df)*100:.1f}%)")

X_full = feat_df[MODEL_FEATURES].fillna(0).values
gbr_pipe = Pipeline([("sc", StandardScaler()),
                      ("est", GradientBoostingRegressor(
                          n_estimators=200, max_depth=3,
                          learning_rate=0.05, random_state=42))])
ridge_pipe = Pipeline([("sc", StandardScaler()), ("est", Ridge(alpha=1.0))])

ho_gbr, n_ho = holdout_season(gbr_pipe, X_full, y, seasons, "r2")
ho_ridge, _ = holdout_season(ridge_pipe, X_full, y, seasons, "r2")
pr(f"  GBR   R2 on held-out {TEST_SEASON}: {ho_gbr if ho_gbr is not None else 'N/A (no rows)'}")
pr(f"  Ridge R2 on held-out {TEST_SEASON}: {ho_ridge if ho_ridge is not None else 'N/A (no rows)'}")

# ── 8. ERROR METRICS (MAE / RMSE) — full feature set, grouped CV ────────
sep("8. ERROR METRICS — MAE / RMSE (full feature set, GroupKFold by match)")
cv = GroupKFold(n_splits=5)
mae_scores, rmse_scores = [], []
for tr, te in cv.split(X_full, y, groups):
    est = Pipeline([("sc", StandardScaler()),
                     ("est", GradientBoostingRegressor(
                         n_estimators=200, max_depth=3,
                         learning_rate=0.05, random_state=42))])
    est.fit(X_full[tr], y[tr])
    pred = est.predict(X_full[te])
    mae_scores.append(mean_absolute_error(y[te], pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y[te], pred)))
pr(f"  GBR MAE  : {np.mean(mae_scores):.3f} +/- {np.std(mae_scores):.3f} goals")
pr(f"  GBR RMSE : {np.mean(rmse_scores):.3f} +/- {np.std(rmse_scores):.3f} goals")
pr(f"  (Reference: league-wide goals/team/match mean = {y.mean():.3f}, std = {y.std():.3f})")

# ── 9. FEATURE IMPORTANCE — full model fit on all data ───────────────────
sep("9. FEATURE IMPORTANCE — GBR fit on full dataset")
scaler = StandardScaler()
Xsc_full = scaler.fit_transform(X_full)
gbr_final = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, random_state=42)
gbr_final.fit(Xsc_full, y)
importances = pd.Series(gbr_final.feature_importances_, index=MODEL_FEATURES).sort_values(ascending=False)
for f, v in importances.items():
    group = "context" if f in CONTEXT_FEATURES else "graph"
    pr(f"  {v*100:5.1f}%  {f:<24} ({group})")

ridge_final = Ridge(alpha=1.0)
ridge_final.fit(Xsc_full, y)
ridge_coef = pd.Series(ridge_final.coef_, index=MODEL_FEATURES).sort_values(key=abs, ascending=False)
pr("\nRidge standardised coefficients (sign + magnitude):")
for f, v in ridge_coef.items():
    group = "context" if f in CONTEXT_FEATURES else "graph"
    pr(f"  {v:+.4f}  {f:<24} ({group})")

graph_importance_sum = importances[GRAPH_FEATURES].sum() * 100
context_importance_sum = importances[CONTEXT_FEATURES].sum() * 100
pr(f"\nAggregate GBR importance — graph features: {graph_importance_sum:.1f}%  |  "
   f"context features: {context_importance_sum:.1f}%")

# ── 10. PER-SEASON BREAKDOWN ─────────────────────────────────────────────
sep("10. PER-SEASON ROW COUNTS & MEAN GOALS")
season_tbl = feat_df.groupby("season").agg(
    n=("goals", "size"), mean_goals=("goals", "mean"),
    mean_density=("network_density", "mean"),
    mean_passes=("total_passes", "mean"),
).round(3)
pr(season_tbl.to_string())

# ── 11. SUMMARY ───────────────────────────────────────────────────────────
sep("11. SUMMARY")
best_row = ablation_df.loc[ablation_df["gbr_r2"].idxmax()]
pr(f"Best feature set (GBR, grouped CV)  : {best_row['feature_set']}  "
   f"(R2={best_row['gbr_r2']:.4f})")
graph_row = ablation_df[ablation_df["feature_set"].str.startswith("Graph-only")].iloc[0]
full_row = ablation_df[ablation_df["feature_set"].str.startswith("Full")].iloc[0]
pr(f"Graph-only ceiling                  : R2={graph_row['gbr_r2']:.4f} (GBR)")
pr(f"Full feature set                    : R2={full_row['gbr_r2']:.4f} (GBR)")
pr(f"Lift from adding context features   : +{full_row['gbr_r2']-graph_row['gbr_r2']:.4f} R2")
pr(f"Held-out-season ({TEST_SEASON}) R2 (GBR)     : "
   f"{ho_gbr if ho_gbr is not None else 'N/A'}")

# ── SAVE ───────────────────────────────────────────────────────────────
os.makedirs("artifacts/model2", exist_ok=True)
report_path = "artifacts/model2/model2_deep_diagnostics.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print(f"\nReport saved to {report_path}")
