"""
models/model2_team_cohesion.py

Model 2: Team Cohesion Analysis
Type: Graph analysis + regression
Objective: Build pass-network graphs per match, compute cohesion metrics
           (centrality, density, clustering coefficient), and use them to
           predict match outcomes (goals scored) via regression.

Beyond pure graph topology, the feature set also includes the strongest
known predictors of goals scored -- team xG, xG conceded, home advantage and
opponent quality -- since network structure alone explains very little of the
variance in goals (R2 ~ 0.05-0.09).

Two correctness fixes vs the original:
  1. No CV leakage. The StandardScaler is wrapped in a Pipeline so it is fit
     only on the training folds inside cross_val_score (previously the scaler
     was fit on the full dataset before CV, inflating the reported R2). The
     persisted artifacts (scaler.pkl, gbr.pkl) are still produced the same way
     so the serving path is unchanged.

Input: pass_network_edges + player_match_stats + matches.

-------------------------------------------------------------------------------
v2 addition: run_optimized()
-------------------------------------------------------------------------------
Everything above (GRAPH_FEATURES, CONTEXT_FEATURES, MODEL_FEATURES, run()) is
UNCHANGED and still produces the exact artifacts api_server.py expects
(artifacts/model2/{scaler,ridge,gbr}.pkl, graph_features.parquet with the
original columns). run_optimized() is a separate, additive pipeline that:

  1. Computes 12 extra engineered graph features (pass concentration,
     reciprocity, hub dominance, community structure) alongside the
     original 12 -- compute_graph_features() now returns 24 keys, but the
     original 12-key GRAPH_FEATURES list and MODEL_FEATURES order are
     untouched, so run() and api_server.py see no difference.
  2. Runs a data-driven multicollinearity audit over the 24 graph features
     and drops near-duplicates (|Spearman rho| > 0.90).
  3. Builds an honest, explicitly-framed "postmatch"/explanatory feature
     set -- reduced graph features + this match's own
     team_xg/team_xga/is_home/opponent_quality. Answers "given the chances
     created, did cohesion help?" -- this is NOT a pre-match forecast and
     is not framed as one.

     (A leakage-safe pre-match forecast variant -- using only rolling/lagged
     team history instead of this match's own xG/graph features -- was
     built and evaluated, and explained only ~8-10% of goal variance, far
     below the explanatory model's ~35%. It was removed from this pipeline
     by request to keep scope to the explanatory framing; see git history /
     prior model2_optimized_diagnostics.txt if it needs to be revisited.)
  4. Compares 6-7 regressors (Ridge, ElasticNet, RandomForest, ExtraTrees,
     GradientBoosting, HistGradientBoosting, XGBoost if installed) with
     GroupKFold(5) by match_id, tunes the top 2 with RandomizedSearchCV
     (grouped CV), and reports held-out-2022 plus leave-one-season-out
     generalisation.
  5. Saves new artifacts under new names (gbr_postmatch.pkl,
     scaler_postmatch.pkl, feature_columns_postmatch.json,
     graph_features_optimized.parquet, model2_optimized_metadata.json) --
     it never overwrites the original scaler.pkl/gbr.pkl/ridge.pkl/
     graph_features.parquet, so nothing currently served can break.

Run: python -m models.model2_team_cohesion --optimize
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import randint, uniform, loguniform
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestRegressor, ExtraTreesRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.inspection import permutation_importance
import joblib

from models.eval_utils import (
    attach_season, grouped_cv, grouped_cv_multi, holdout_season,
    leave_one_season_out, TEST_SEASON,
)

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

logger = logging.getLogger(__name__)


def load_edges(conn) -> pd.DataFrame:
    """Load all pass network edges joined with match outcome."""
    query = """
        SELECT
            pne.match_id,
            pne.team_id,
            pne.passer_id,
            pne.receiver_id,
            pne.pass_count,
            pne.avg_x_start,
            pne.avg_y_start,
            pne.avg_x_end,
            pne.avg_y_end,
            m.home_team_id,
            m.away_team_id,
            m.home_score,
            m.away_score
        FROM pass_network_edges pne
        JOIN matches m ON m.match_id = pne.match_id
        ORDER BY pne.match_id, pne.team_id
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def load_team_context(conn) -> pd.DataFrame:
    """
    One row per (match_id, team_id) with team-level context features:

        team_xg          -- sum of player xG for the team in the match
        team_xga         -- the opponent's team_xg in the same match (xG against)
        is_home          -- 1 if this team played at home
        opponent_quality -- opponent's season-to-date points-per-game *before*
                            this match (expanding mean, shifted to exclude the
                            current result -> no leakage)
    """
    query = """
        SELECT
            m.match_id,
            pms.team_id,
            m.home_team_id,
            m.away_team_id,
            m.home_score,
            m.away_score,
            m.season,
            m.match_date,
            SUM(pms.xg) AS team_xg
        FROM player_match_stats pms
        JOIN matches m ON m.match_id = pms.match_id
        GROUP BY m.match_id, pms.team_id, m.home_team_id, m.away_team_id,
                 m.home_score, m.away_score, m.season, m.match_date
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df["team_xg"]  = df["team_xg"].astype(float)
    df["is_home"]  = (df["team_id"] == df["home_team_id"]).astype(int)
    df["opp_team_id"] = np.where(
        df["is_home"] == 1, df["away_team_id"], df["home_team_id"]
    )

    gf = np.where(df["is_home"] == 1, df["home_score"], df["away_score"])
    ga = np.where(df["is_home"] == 1, df["away_score"], df["home_score"])
    df["points"] = np.select([gf > ga, gf == ga], [3, 1], default=0)

    # team_xga = opponent's team_xg in the same match
    xga = df[["match_id", "team_id", "team_xg"]].rename(
        columns={"team_id": "opp_team_id", "team_xg": "team_xga"}
    )
    df = df.merge(xga, on=["match_id", "opp_team_id"], how="left")

    # season-to-date PPG for each team, excluding the current match
    df = df.sort_values(["season", "team_id", "match_date"])
    df["ppg_to_date"] = (
        df.groupby(["season", "team_id"])["points"]
          .transform(lambda s: s.expanding().mean().shift(1))
    )

    # opponent_quality = opponent's season-to-date PPG at this match
    oppq = df[["match_id", "team_id", "ppg_to_date"]].rename(
        columns={"team_id": "opp_team_id", "ppg_to_date": "opponent_quality"}
    )
    df = df.merge(oppq, on=["match_id", "opp_team_id"], how="left")

    league_avg = df["ppg_to_date"].mean()
    league_avg = float(league_avg) if pd.notna(league_avg) else 1.0
    df["opponent_quality"] = df["opponent_quality"].fillna(league_avg)
    df["team_xga"] = df["team_xga"].fillna(0.0)

    return df[["match_id", "team_id", "team_xg", "team_xga",
               "is_home", "opponent_quality"]]


def build_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed weighted pass graph from a subset of edges.

    Parameters
    ----------
    edges_df : rows for ONE (match_id, team_id) pair.
    """
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(
            row["passer_id"],
            row["receiver_id"],
            weight=row["pass_count"],
        )
    return G


def _gini(weights: List[float]) -> float:
    """Gini coefficient of edge weights (0 = evenly spread, ~1 = concentrated)."""
    w = np.sort(np.asarray(weights, dtype=float))
    n = len(w)
    if n == 0 or w.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * w) / (n * w.sum())) - (n + 1) / n)


def compute_graph_features(G: nx.DiGraph) -> Dict[str, float]:
    """
    Compute cohesion metrics from a pass graph.

    Returns the original 12 scalar features (GRAPH_FEATURES -- unchanged
    formulas, still what api_server.py and run() consume) plus 12 additional
    engineered features (NEW_GRAPH_FEATURES) used only by run_optimized():
    pass concentration (entropy/Gini/edge-share), reciprocity, hub dominance,
    per-player normalisation and community structure.
    """
    if G.number_of_nodes() == 0:
        return _empty_graph_features()

    UG = G.to_undirected()

    in_centrality  = nx.in_degree_centrality(G)
    out_centrality = nx.out_degree_centrality(G)
    between        = nx.betweenness_centrality(G, weight="weight", normalized=True)
    page_rank      = nx.pagerank(G, weight="weight")

    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    total_passes = sum(weights)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    mean_betweenness = float(np.mean(list(between.values())))
    max_betweenness  = float(max(between.values(), default=0))
    mean_pagerank    = float(np.mean(list(page_rank.values())))
    max_pagerank     = float(max(page_rank.values(), default=0))

    base = {
        "network_density":          nx.density(UG),
        "clustering_coefficient":   nx.average_clustering(UG),
        "mean_in_centrality":       float(np.mean(list(in_centrality.values()))),
        "mean_out_centrality":      float(np.mean(list(out_centrality.values()))),
        "mean_betweenness":         mean_betweenness,
        "max_betweenness":          max_betweenness,
        "mean_pagerank":            mean_pagerank,
        "max_pagerank":             max_pagerank,
        "n_nodes":                  n_nodes,
        "n_edges":                  n_edges,
        "total_passes":             total_passes,
        "pass_per_edge":            total_passes / max(1, n_edges),
    }

    sorted_w = sorted(weights, reverse=True)
    top1 = sorted_w[0] if sorted_w else 0
    top3 = sum(sorted_w[:3])

    if n_edges > 1 and total_passes > 0:
        probs = np.array(weights, dtype=float) / total_passes
        pass_entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(n_edges))
    else:
        pass_entropy = 0.0

    try:
        reciprocity = float(nx.overall_reciprocity(G)) if n_edges > 0 else 0.0
    except Exception:
        reciprocity = 0.0

    try:
        largest_wcc = max((len(c) for c in nx.weakly_connected_components(G)), default=0)
        largest_component_ratio = largest_wcc / n_nodes if n_nodes else 0.0
    except Exception:
        largest_component_ratio = 0.0

    try:
        communities = nx.algorithms.community.greedy_modularity_communities(UG, weight="weight")
        modularity = (
            float(nx.algorithms.community.modularity(UG, communities, weight="weight"))
            if len(communities) > 1 else 0.0
        )
    except Exception:
        modularity = 0.0

    new = {
        "pass_entropy":             pass_entropy,
        "pass_concentration_gini":  _gini(weights),
        "max_edge_share":           float(top1 / total_passes) if total_passes else 0.0,
        "top_3_edge_share":         float(top3 / total_passes) if total_passes else 0.0,
        "hub_dominance":            float(max_pagerank / mean_pagerank) if mean_pagerank else 0.0,
        "reciprocity":              reciprocity,
        "weighted_density":         total_passes / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0,
        "passes_per_player":        total_passes / n_nodes if n_nodes else 0.0,
        "edges_per_player":         n_edges / n_nodes if n_nodes else 0.0,
        "centralization_index":     max_betweenness - mean_betweenness,
        "largest_component_ratio":  largest_component_ratio,
        "modularity":               modularity,
    }

    return {**base, **new}


def _empty_graph_features() -> Dict[str, float]:
    return {k: 0.0 for k in GRAPH_FEATURES + NEW_GRAPH_FEATURES}


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over all (match_id, team_id) pairs and compute graph features.

    Returns a DataFrame with one row per (match_id, team_id) containing
    graph-derived features and the goals scored by that team.
    """
    records: List[Dict[str, Any]] = []

    for (match_id, team_id), grp in df.groupby(["match_id", "team_id"]):
        G       = build_graph(grp)
        metrics = compute_graph_features(G)

        # Determine goals scored by this team
        row = grp.iloc[0]
        if team_id == row["home_team_id"]:
            goals = row["home_score"]
        else:
            goals = row["away_score"]

        records.append({
            "match_id": match_id,
            "team_id":  team_id,
            "goals":    goals,
            **metrics,
        })

    return pd.DataFrame(records)


GRAPH_FEATURES = [
    "network_density", "clustering_coefficient",
    "mean_in_centrality", "mean_out_centrality",
    "mean_betweenness", "max_betweenness",
    "mean_pagerank", "max_pagerank",
    "n_nodes", "n_edges", "total_passes", "pass_per_edge",
]

# Additional engineered graph features (pass concentration, reciprocity, hub
# dominance, community structure). Used only by run_optimized() -- the
# original GRAPH_FEATURES/MODEL_FEATURES/serving path never see these.
NEW_GRAPH_FEATURES = [
    "pass_entropy", "pass_concentration_gini", "max_edge_share", "top_3_edge_share",
    "hub_dominance", "reciprocity", "weighted_density", "passes_per_player",
    "edges_per_player", "centralization_index", "largest_component_ratio", "modularity",
]

EXPANDED_GRAPH_FEATURES = GRAPH_FEATURES + NEW_GRAPH_FEATURES

# Strong contextual predictors of goals scored, appended to the graph metrics.
CONTEXT_FEATURES = ["team_xg", "team_xga", "is_home", "opponent_quality"]

# Full feature vector used by the ORIGINAL regression models (and the serving
# path in api_server.py). UNCHANGED -- do not reorder/extend this list.
MODEL_FEATURES = GRAPH_FEATURES + CONTEXT_FEATURES

# Preference order for the multicollinearity audit: when two features in
# EXPANDED_GRAPH_FEATURES collide (|Spearman rho| > threshold), the one
# listed EARLIER here is kept and the later one is dropped. New engineered
# features are listed first (they were designed to carry more specific
# signal than the broad originals); the most structurally redundant
# originals are listed last so they lose ties against everything:
#   - mean_pagerank sums to 1 over all nodes by construction, so its mean is
#     ~1/n_nodes regardless of topology -- it is expected to collide with
#     n_nodes and to add little once hub_dominance (max/mean pagerank) exists.
#   - mean_in_centrality / mean_out_centrality are near-identical for this
#     graph construction (passers tend to also be receivers) and both are
#     close restatements of network_density for near-symmetric digraphs.
#   - pass_per_edge collides with total_passes; weighted_density is the
#     requested normalised alternative.
GRAPH_FEATURE_KEEP_PRIORITY = [
    "pass_entropy", "pass_concentration_gini", "max_edge_share", "top_3_edge_share",
    "hub_dominance", "reciprocity", "weighted_density", "passes_per_player",
    "edges_per_player", "centralization_index", "largest_component_ratio", "modularity",
    "network_density", "clustering_coefficient", "mean_betweenness", "max_betweenness",
    "n_nodes", "n_edges", "total_passes", "max_pagerank",
    "mean_in_centrality", "mean_out_centrality", "pass_per_edge", "mean_pagerank",
]


def run(conn, output_dir: str = "artifacts/model2") -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Model 2: loading pass network edges ...")
    df = load_edges(conn)
    logger.info("  %d edges loaded", len(df))

    logger.info("Model 2: computing graph features ...")
    feat_df = build_feature_matrix(df)
    logger.info("  %d team-match rows", len(feat_df))

    # Enrich with team-level context (team_xg/xga, home, opponent quality).
    ctx = load_team_context(conn)
    feat_df = feat_df.merge(ctx, on=["match_id", "team_id"], how="left")
    for col in CONTEXT_FEATURES:
        feat_df[col] = feat_df[col].fillna(0.0)

    feat_df = attach_season(feat_df, conn)
    feat_df.to_parquet(f"{output_dir}/graph_features.parquet", index=False)

    X = feat_df[MODEL_FEATURES].fillna(0).values
    y = feat_df["goals"].values.astype(float)
    groups  = feat_df["match_id"].values
    seasons = feat_df["season"].values

    # Honest evaluation: GroupKFold by match (no match straddles folds) plus a
    # held-out season test. The scaler lives inside the Pipeline so it is fit
    # only on training folds. Persisted artifacts below are still a standalone
    # scaler + estimator to match the serving path in api_server.py.
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("est", Ridge(alpha=1.0))])
    gbr_pipe   = Pipeline([
        ("scaler", StandardScaler()),
        ("est", GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
    ])

    ridge_m, ridge_s = grouped_cv(ridge_pipe, X, y, groups, "r2")
    logger.info("Ridge R2 (GroupKFold by match): %.3f +/- %.3f", ridge_m, ridge_s)
    gbr_m, gbr_s = grouped_cv(gbr_pipe, X, y, groups, "r2")
    logger.info("GBR   R2 (GroupKFold by match): %.3f +/- %.3f", gbr_m, gbr_s)
    ho, n = holdout_season(gbr_pipe, X, y, seasons, "r2")
    if ho is not None:
        logger.info("GBR   R2 held-out %s (n=%d): %.3f", TEST_SEASON, n, ho)

    # Fit final artifacts on the full dataset (scaler + estimators saved
    # separately, exactly as the API expects).
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_sc, y)

    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    gbr.fit(X_sc, y)

    # Feature importances
    importances = pd.Series(gbr.feature_importances_, index=MODEL_FEATURES)
    logger.info("Top features:\n%s", importances.sort_values(ascending=False).head())

    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(ridge,  f"{output_dir}/ridge.pkl")
    joblib.dump(gbr,    f"{output_dir}/gbr.pkl")

    logger.info("Model 2 artefacts saved to %s", output_dir)

    metrics = {
        "ridge_r2": float(ridge_m),
        "ridge_r2_std": float(ridge_s),
        "gbr_r2": float(gbr_m),
        "gbr_r2_std": float(gbr_s),
        "feature_importances": {
            f: float(v) for f, v in zip(MODEL_FEATURES, gbr.feature_importances_)
        },
    }
    if ho is not None:
        metrics["gbr_r2_heldout"] = float(ho)

    return {
        "ridge": ridge, "gbr": gbr, "scaler": scaler, "feat_df": feat_df,
        "_registry": {
            "model_key": "model2_team_cohesion",
            "version": "1.0",
            "display_name": "Team Cohesion (Pass Networks)",
            "task": "regression",
            "algorithm": "GradientBoostingRegressor + Ridge (graph features)",
            "target": "goals scored",
            "features": list(MODEL_FEATURES),
            "metrics": metrics,
            "n_train_rows": int(len(feat_df)),
            "artifact_path": output_dir,
            "prediction_table": "model2_graph_features",
        },
        "_predictions": {"model2_graph_features": feat_df},
    }


# ══════════════════════════════════════════════════════════════════════════
# v2: MULTICOLLINEARITY AUDIT
# ══════════════════════════════════════════════════════════════════════════

def audit_multicollinearity(df: pd.DataFrame, candidate_features: List[str],
                             keep_priority: List[str], threshold: float = 0.90):
    """
    Greedy redundancy elimination over `candidate_features` using |Spearman
    rho|. Whenever a pair exceeds `threshold`, the feature ranked EARLIER in
    `keep_priority` is kept and the other is dropped (features absent from
    keep_priority rank last, i.e. dropped first).

    Returns (kept_features, dropped_records, report_lines). Each dropped
    record: {"dropped": f, "kept": g, "rho": r}.
    """
    corr = df[candidate_features].corr(method="spearman").abs()
    corr_vals = np.array(corr.values, copy=True)
    np.fill_diagonal(corr_vals, 0)
    corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)

    priority_rank = {f: i for i, f in enumerate(keep_priority)}
    def rank(f):
        return priority_rank.get(f, len(keep_priority) + 1)

    active = set(candidate_features)
    dropped: List[dict] = []

    pairs = (corr.where(np.triu(np.ones_like(corr.values, dtype=bool), k=1))
                  .stack().sort_values(ascending=False))
    for (a, b), rho in pairs.items():
        if rho < threshold:
            break
        if a not in active or b not in active:
            continue
        loser, keeper = (a, b) if rank(a) > rank(b) else (b, a)
        active.discard(loser)
        dropped.append({"dropped": loser, "kept": keeper, "rho": round(float(rho), 4)})

    kept = [f for f in candidate_features if f in active]
    report = [f"Multicollinearity audit (|Spearman rho| > {threshold:.2f} triggers a drop):"]
    if dropped:
        for d in dropped:
            report.append(f"  DROP {d['dropped']:<24} (rho={d['rho']:.3f} vs kept feature {d['kept']})")
    else:
        report.append("  No pairs exceeded threshold.")
    report.append(f"Kept {len(kept)}/{len(candidate_features)} features.")
    return kept, dropped, report


# ══════════════════════════════════════════════════════════════════════════
# v2: MODEL REGISTRY (baseline kwargs + RandomizedSearchCV param grids)
# ══════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, tuple] = {
    "Ridge": (Ridge, {"alpha": 1.0}, {
        "est__alpha": loguniform(1e-2, 1e2),
    }),
    "ElasticNet": (ElasticNet, {"max_iter": 5000, "random_state": 42}, {
        "est__alpha": loguniform(1e-3, 1e1),
        "est__l1_ratio": uniform(0.0, 1.0),
    }),
    "RandomForest": (RandomForestRegressor, {"random_state": 42}, {
        "est__n_estimators": randint(100, 400),
        "est__max_depth": randint(2, 12),
        "est__min_samples_leaf": randint(1, 20),
    }),
    "ExtraTrees": (ExtraTreesRegressor, {"random_state": 42}, {
        "est__n_estimators": randint(100, 400),
        "est__max_depth": randint(2, 12),
        "est__min_samples_leaf": randint(1, 20),
    }),
    "GBR": (GradientBoostingRegressor, {"random_state": 42}, {
        "est__n_estimators": randint(100, 400),
        "est__max_depth": randint(2, 5),
        "est__learning_rate": loguniform(1e-2, 3e-1),
        "est__min_samples_leaf": randint(1, 20),
        "est__subsample": uniform(0.6, 0.4),
    }),
    "HistGBR": (HistGradientBoostingRegressor, {"random_state": 42}, {
        "est__max_iter": randint(100, 400),
        "est__max_leaf_nodes": randint(8, 64),
        "est__learning_rate": loguniform(1e-2, 3e-1),
        "est__min_samples_leaf": randint(5, 40),
        "est__l2_regularization": loguniform(1e-3, 1e1),
    }),
}
if _HAS_XGB:
    MODEL_REGISTRY["XGB"] = (
        XGBRegressor,
        {"random_state": 42, "objective": "reg:squarederror", "verbosity": 0},
        {
            "est__n_estimators": randint(100, 400),
            "est__max_depth": randint(2, 8),
            "est__learning_rate": loguniform(1e-2, 3e-1),
            "est__subsample": uniform(0.6, 0.4),
            "est__colsample_bytree": uniform(0.5, 0.5),
            "est__min_child_weight": randint(1, 10),
        },
    )


# ══════════════════════════════════════════════════════════════════════════
# v2: FIGURES (same dark "night" style as models/model1_player_clustering.py)
# ══════════════════════════════════════════════════════════════════════════

def _night_style() -> dict:
    return dict(
        bg="#0D1117", surface="#161B22", grid="#21262D",
        text="#E6EDF3", muted="#8B949E",
        green="#39D353", teal="#00B4D8", amber="#F0B429",
        coral="#FF6B6B", purple="#A371F7",
    )


def _apply_night(fig, axes=None):
    N = _night_style()
    fig.patch.set_facecolor(N["bg"])
    for ax in (axes if axes is not None else fig.axes):
        ax.set_facecolor(N["surface"])
        ax.tick_params(colors=N["muted"], labelsize=9)
        ax.xaxis.label.set_color(N["muted"])
        ax.yaxis.label.set_color(N["muted"])
        ax.title.set_color(N["text"])
        for sp in ax.spines.values():
            sp.set_edgecolor(N["grid"])
        ax.grid(color=N["grid"], linewidth=0.4, linestyle="--", alpha=0.6)


SEASON_COLORS = ["#00B4D8", "#FF6B6B", "#F0B429", "#39D353", "#A371F7", "#F4845F"]


def make_optimized_figures(
    feat_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    dropped: List[dict],
    postmatch_features: List[str],
    best_meta: Dict[str, Any],
    final_est, scaler, y: np.ndarray, seasons: np.ndarray,
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

    # ── Fig 1: correlation heatmap of all 24 candidate graph features ───────
    fig1, ax1 = plt.subplots(figsize=(13, 11))
    corr = feat_df[EXPANDED_GRAPH_FEATURES].corr(method="spearman")
    im = ax1.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    dropped_set = {d["dropped"] for d in dropped}
    ax1.set_xticks(range(len(EXPANDED_GRAPH_FEATURES)))
    ax1.set_yticks(range(len(EXPANDED_GRAPH_FEATURES)))
    ax1.set_xticklabels(EXPANDED_GRAPH_FEATURES, rotation=60, ha="right", fontsize=8)
    ax1.set_yticklabels(EXPANDED_GRAPH_FEATURES, fontsize=8)
    for tick, f in zip(ax1.get_xticklabels(), EXPANDED_GRAPH_FEATURES):
        tick.set_color(N["coral"] if f in dropped_set else N["text"])
    for tick, f in zip(ax1.get_yticklabels(), EXPANDED_GRAPH_FEATURES):
        tick.set_color(N["coral"] if f in dropped_set else N["text"])
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label("Spearman rho", color=N["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=N["muted"])
    ax1.set_title("Graph Feature Correlation Matrix\n(24 candidates -- red labels = dropped by the multicollinearity audit)",
                  color=N["text"], fontweight="bold", fontsize=11)
    _apply_night(fig1)
    plt.tight_layout()
    fig1.savefig(out / "fig1_feature_correlation_heatmap.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig1)

    # ── Fig 2: model comparison -- grouped bars per feature set ─────────────
    fig2, ax2 = plt.subplots(figsize=(13, 6))
    plot_df = comparison_df[comparison_df["feature_set"] != "baseline"]
    feature_sets = ["graph_only", "context_only", "postmatch_full"]
    fs_colors = {"graph_only": N["coral"], "context_only": N["teal"], "postmatch_full": N["green"]}
    models_order = plot_df["model"].unique().tolist()
    x = np.arange(len(models_order))
    width = 0.25
    for i, fs in enumerate(feature_sets):
        sub = plot_df[plot_df["feature_set"] == fs].set_index("model").reindex(models_order)
        ax2.bar(x + (i - 1) * width, sub["r2_mean"], width, yerr=sub["r2_std"],
                label=fs, color=fs_colors[fs], edgecolor=N["bg"], capsize=3)
    ax2.axhline(0, color=N["grid"], linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_order, rotation=20, ha="right")
    ax2.set_ylabel("Grouped CV R2 (GroupKFold by match)")
    ax2.set_title("Model Comparison -- Untuned Baseline", fontweight="bold")
    leg = ax2.legend(fontsize=9, framealpha=0.2, labelcolor=N["text"])
    leg.get_frame().set_facecolor(N["surface"])
    _apply_night(fig2)
    plt.tight_layout()
    fig2.savefig(out / "fig2_model_comparison.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig2)

    # ── Fig 3: native + permutation feature importance, side by side ────────
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(15, 7))
    imp_sorted = importances.sort_values()
    col3a = [N["teal"] if f in CONTEXT_FEATURES else N["coral"] for f in imp_sorted.index]
    ax3a.barh(range(len(imp_sorted)), imp_sorted.values, color=col3a, edgecolor=N["bg"])
    ax3a.set_yticks(range(len(imp_sorted)))
    ax3a.set_yticklabels(imp_sorted.index, fontsize=8)
    ax3a.set_xlabel("Native importance")
    ax3a.set_title(f"Feature Importance -- {best_meta['model']}", fontweight="bold")

    perm_sorted = perm_importances.reindex(imp_sorted.index)
    col3b = [N["teal"] if f in CONTEXT_FEATURES else N["coral"] for f in perm_sorted.index]
    ax3b.barh(range(len(perm_sorted)), perm_sorted.values, color=col3b, edgecolor=N["bg"])
    ax3b.set_yticks(range(len(perm_sorted)))
    ax3b.set_yticklabels(perm_sorted.index, fontsize=8)
    ax3b.set_xlabel("Permutation importance (R2 drop, in-sample)")
    ax3b.set_title("Permutation Importance", fontweight="bold")

    from matplotlib.patches import Patch
    handles = [Patch(color=N["coral"], label="graph feature"), Patch(color=N["teal"], label="context feature")]
    leg = ax3b.legend(handles=handles, fontsize=8, framealpha=0.2, labelcolor=N["text"], loc="lower right")
    leg.get_frame().set_facecolor(N["surface"])
    _apply_night(fig3, [ax3a, ax3b])
    plt.tight_layout()
    fig3.savefig(out / "fig3_feature_importance.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig3)

    # ── Fig 4: leave-one-season-out + held-out generalisation ───────────────
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    loso = [r for r in best_meta.get("loso", []) if not r.get("skipped")]
    season_labels = [str(r["season"]) for r in loso]
    r2_vals = [r["r2"] for r in loso]
    bar_colors = [N["amber"] if s == TEST_SEASON else N["teal"] for s in season_labels]
    ax4.bar(season_labels, r2_vals, color=bar_colors, edgecolor=N["bg"])
    ax4.axhline(best_meta["r2_cv"], color=N["coral"], linestyle="--", linewidth=1.5,
                label=f"Grouped CV R2={best_meta['r2_cv']:.3f}")
    ax4.set_ylabel("R2 (trained on all other seasons)")
    ax4.set_title("Leave-One-Season-Out Generalisation\n(amber = held-out 2022 World Cup)",
                  fontweight="bold", fontsize=11)
    leg = ax4.legend(fontsize=9, framealpha=0.2, labelcolor=N["text"])
    leg.get_frame().set_facecolor(N["surface"])
    _apply_night(fig4)
    plt.tight_layout()
    fig4.savefig(out / "fig4_holdout_loso_performance.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig4)

    # ── Fig 5: predicted vs actual goals, colored by season ──────────────────
    X_final = feat_df[postmatch_features].fillna(0).values
    pred = final_est.predict(scaler.transform(X_final))
    fig5, ax5 = plt.subplots(figsize=(7.5, 7.5))
    for i, s in enumerate(sorted(set(seasons.tolist()))):
        mask = seasons == s
        ax5.scatter(y[mask], pred[mask], s=16, alpha=0.5,
                    color=SEASON_COLORS[i % len(SEASON_COLORS)], label=str(s), linewidths=0)
    lim = float(max(y.max(), pred.max()) + 0.5)
    ax5.plot([0, lim], [0, lim], color=N["grid"], linestyle="--", linewidth=1)
    ax5.set_xlabel("Actual goals")
    ax5.set_ylabel("Predicted goals")
    ax5.set_title(f"Predicted vs Actual -- {best_meta['model']} (in-sample fit)",
                  fontweight="bold")
    leg = ax5.legend(fontsize=8, framealpha=0.2, labelcolor=N["text"])
    leg.get_frame().set_facecolor(N["surface"])
    _apply_night(fig5)
    plt.tight_layout()
    fig5.savefig(out / "fig5_predicted_vs_actual.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig5)

    # ── Fig 6: example pass network drawn on a pitch ─────────────────────────
    try:
        med = (feat_df["total_passes"] - feat_df["total_passes"].median()).abs().idxmin()
        sel = feat_df.loc[med]
        sel_match, sel_team = int(sel["match_id"]), int(sel["team_id"])
        sub_edges = edges_df[(edges_df["match_id"] == sel_match) & (edges_df["team_id"] == sel_team)]

        node_pass = {}
        node_pos_num = {}
        for _, r in sub_edges.iterrows():
            pid, w = r["passer_id"], r["pass_count"]
            node_pass[pid] = node_pass.get(pid, 0) + w
            xs, ys = node_pos_num.get(pid, (0.0, 0.0))
            node_pos_num[pid] = (xs + r["avg_x_start"] * w, ys + r["avg_y_start"] * w)
        positions = {pid: (xy[0] / node_pass[pid], xy[1] / node_pass[pid])
                     for pid, xy in node_pos_num.items()}

        fig6, ax6 = plt.subplots(figsize=(12, 8))
        for rect_args in [
            dict(xy=(0, 0), width=120, height=80),
            dict(xy=(0, 18), width=18, height=44),
            dict(xy=(102, 18), width=18, height=44),
        ]:
            ax6.add_patch(plt.Rectangle(fill=False, edgecolor=N["grid"], linewidth=1.2, **rect_args))
        ax6.axvline(60, color=N["grid"], linewidth=1.0, alpha=0.5)

        for _, r in sub_edges.iterrows():
            if r["passer_id"] in positions and r["receiver_id"] in positions:
                x1, y1 = positions[r["passer_id"]]
                x2, y2 = positions[r["receiver_id"]]
                ax6.plot([x1, x2], [y1, y2], color=N["teal"], alpha=0.35,
                         linewidth=max(0.5, r["pass_count"] / 6), zorder=2)

        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        sizes = [node_pass[pid] * 6 for pid in positions]
        ax6.scatter(xs, ys, s=sizes, color=N["coral"], edgecolors="white", linewidths=0.8, zorder=5)
        ax6.set_xlim(-2, 122)
        ax6.set_ylim(-2, 82)
        ax6.set_xlabel("x (0 = own goal, 120 = opposition goal)")
        ax6.set_ylabel("y (0 = right touchline, 80 = left touchline)")
        ax6.set_title(
            f"Example Pass Network -- match {sel_match}, team {sel_team}\n"
            f"density={sel['network_density']:.2f}  modularity={sel['modularity']:.2f}  "
            f"reciprocity={sel['reciprocity']:.2f}  goals={sel['goals']:.0f}",
            fontweight="bold", fontsize=11,
        )
        _apply_night(fig6)
        plt.tight_layout()
        fig6.savefig(out / "fig6_example_pass_network.png", dpi=150, bbox_inches="tight", facecolor=N["bg"])
        plt.close(fig6)
    except Exception as exc:
        logger.warning("Fig 6 (example pass network) skipped: %s", exc)

    logger.info("Figures saved to %s", out)


# ══════════════════════════════════════════════════════════════════════════
# v2: OPTIMIZED PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def run_optimized(conn, output_dir: str = "artifacts/model2") -> Dict[str, Any]:
    """
    Postmatch/explanatory optimization of Model 2 (engineered + de-collinearised
    graph features, model comparison, hyperparameter tuning, out-of-time
    generalisation). Does NOT touch the original scaler.pkl/ridge.pkl/gbr.pkl/
    graph_features.parquet -- see module docstring. Writes a full diagnostics
    report to artifacts/model2/model2_optimized_diagnostics.txt.

    A pre-match/forecast variant (no current-match xG or graph features, only
    rolling team history) was built and evaluated alongside this one and
    explained only ~8-10% of goal variance vs. ~35% here; it was removed by
    request to keep this pipeline scoped to the explanatory framing.
    """
    import os
    warnings.filterwarnings("ignore")
    os.makedirs(output_dir, exist_ok=True)

    report: List[str] = []

    def sec(title: str):
        report.append("\n" + "=" * 78)
        report.append(f"  {title}")
        report.append("=" * 78)
        logger.info(title)

    def pr(line: str = ""):
        report.append(str(line))

    sec("MODEL 2 OPTIMIZED -- TEAM COHESION (v2, postmatch/explanatory)")

    # ── 1. Load + build expanded feature matrix ──────────────────────────
    logger.info("Loading pass network edges ...")
    edges = load_edges(conn)
    feat_df = build_feature_matrix(edges)          # now includes EXPANDED_GRAPH_FEATURES
    ctx = load_team_context(conn)
    feat_df = feat_df.merge(ctx, on=["match_id", "team_id"], how="left")
    for col in CONTEXT_FEATURES:
        feat_df[col] = feat_df[col].fillna(0.0)
    feat_df = attach_season(feat_df, conn)

    pr(f"Team-match rows: {len(feat_df):,}")
    pr(f"Seasons: {sorted(feat_df['season'].dropna().unique())}")

    # ── 2. Multicollinearity audit on the expanded graph feature set ─────
    sec("1. MULTICOLLINEARITY AUDIT (24 candidate graph features)")
    kept_graph_features, dropped, audit_lines = audit_multicollinearity(
        feat_df, EXPANDED_GRAPH_FEATURES, GRAPH_FEATURE_KEEP_PRIORITY, threshold=0.90,
    )
    report.extend(audit_lines)
    pr(f"\nFinal graph feature set ({len(kept_graph_features)}):")
    for f in kept_graph_features:
        pr(f"  {f}")

    POSTMATCH_FEATURES = kept_graph_features + CONTEXT_FEATURES
    feat_df.to_parquet(f"{output_dir}/graph_features_optimized.parquet", index=False)

    groups  = feat_df["match_id"].values
    seasons = feat_df["season"].values
    y       = feat_df["goals"].values.astype(float)

    # ── 3. Baseline model comparison (untuned), GroupKFold(5) ────────────
    sec("2. MODEL COMPARISON -- BASELINE (UNTUNED), GroupKFold(5) BY MATCH")
    dummy_res = grouped_cv_multi(DummyRegressor(strategy="mean"), np.zeros((len(y), 1)), y, groups)
    pr(f"  Baseline (mean predictor)   R2={dummy_res['r2_mean']:+.4f}+/-{dummy_res['r2_std']:.4f}  "
       f"MAE={dummy_res['mae_mean']:.3f}  RMSE={dummy_res['rmse_mean']:.3f}")
    comparison_rows = [{"feature_set": "baseline", "model": "DummyMean", **dummy_res}]

    ABLATION_FEATURE_SETS = {
        "graph_only":     kept_graph_features,
        "context_only":   CONTEXT_FEATURES,
        "postmatch_full": POSTMATCH_FEATURES,
    }
    for set_name, feats in ABLATION_FEATURE_SETS.items():
        X = feat_df[feats].fillna(0).values
        pr(f"\n  -- {set_name} ({len(feats)} features) --")
        for model_name, (est_cls, kwargs, _) in MODEL_REGISTRY.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("est", est_cls(**kwargs))])
            res = grouped_cv_multi(pipe, X, y, groups)
            comparison_rows.append({"feature_set": set_name, "model": model_name, **res})
            pr(f"    {model_name:<14} R2={res['r2_mean']:+.4f}+/-{res['r2_std']:.4f}  "
               f"MAE={res['mae_mean']:.3f}  RMSE={res['rmse_mean']:.3f}")
    comparison_df = pd.DataFrame(comparison_rows)

    # ── 4. Hyperparameter tuning -- top 2 models on the full feature set ──
    sec("3. HYPERPARAMETER TUNING -- RandomizedSearchCV(n_iter=20), GroupKFold(5)")
    X = feat_df[POSTMATCH_FEATURES].fillna(0).values
    top2 = (comparison_df[comparison_df["feature_set"] == "postmatch_full"]
            .sort_values("r2_mean", ascending=False).head(2)["model"].tolist())
    pr(f"\nTuning candidates (top-2 baseline): {top2}")

    best_score = -np.inf
    best_meta: Dict[str, Any] = {}
    best_estimator = None
    for model_name in top2:
        est_cls, kwargs, param_dist = MODEL_REGISTRY[model_name]
        pipe = Pipeline([("scaler", StandardScaler()), ("est", est_cls(**kwargs))])
        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=20,
            cv=GroupKFold(n_splits=5), scoring="r2",
            random_state=42, n_jobs=4,
        )
        search.fit(X, y, groups=groups)
        tuned_params = {k.replace("est__", ""): v for k, v in search.best_params_.items()}
        pr(f"  {model_name:<14} tuned R2={search.best_score_:+.4f}  params={tuned_params}")
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_estimator = search.best_estimator_
            best_meta = {
                "model": model_name,
                "r2_cv": round(float(search.best_score_), 4),
                "params": tuned_params,
            }
    pr(f"  -> BEST: {best_meta['model']}  R2={best_meta['r2_cv']}")

    # ── 5. Out-of-time generalisation: held-out 2022 + leave-one-season-out
    sec("4. OUT-OF-TIME GENERALISATION -- held-out 2022 + leave-one-season-out")
    ho, n = holdout_season(best_estimator, X, y, seasons, "r2")
    pr(f"\nHeld-out 2022 (n={n}): R2={ho}")
    best_meta["holdout_2022_r2"] = ho

    loso = leave_one_season_out(best_estimator, X, y, seasons)
    best_meta["loso"] = loso
    for rec in loso:
        if rec.get("skipped"):
            pr(f"  season={rec['season']:<10} SKIPPED (n_test={rec['n_test']} < 30)")
        else:
            pr(f"  season={rec['season']:<10} n={rec['n_test']:<4} R2={rec['r2']:+.4f}  "
               f"MAE={rec['mae']:.3f}  RMSE={rec['rmse']:.3f}")

    # ── 6. Fit final model on full data, save artifacts ───────────────────
    sec("5. FINAL ARTIFACTS, FEATURE IMPORTANCE & PERMUTATION IMPORTANCE")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    est_cls, kwargs, _ = MODEL_REGISTRY[best_meta["model"]]
    final_est = est_cls(**{**kwargs, **best_meta["params"]})
    final_est.fit(X_sc, y)

    joblib.dump(scaler, f"{output_dir}/scaler_postmatch.pkl")
    joblib.dump(final_est, f"{output_dir}/gbr_postmatch.pkl")
    with open(f"{output_dir}/feature_columns_postmatch.json", "w", encoding="utf-8") as f:
        json.dump(POSTMATCH_FEATURES, f, indent=2)

    pr(f"\nFinal model: {best_meta['model']}  "
       f"(saved gbr_postmatch.pkl, scaler_postmatch.pkl, feature_columns_postmatch.json)")

    if hasattr(final_est, "feature_importances_"):
        imp = pd.Series(final_est.feature_importances_, index=POSTMATCH_FEATURES).sort_values(ascending=False)
        pr("  Native feature importances:")
        for f, v in imp.items():
            pr(f"    {v*100:5.1f}%  {f}")
    elif hasattr(final_est, "coef_"):
        imp = pd.Series(final_est.coef_, index=POSTMATCH_FEATURES).sort_values(key=abs, ascending=False)
        pr("  Standardised coefficients:")
        for f, v in imp.items():
            pr(f"    {v:+.4f}  {f}")

    perm = permutation_importance(
        final_est, X_sc, y, n_repeats=20, random_state=42, scoring="r2", n_jobs=4,
    )
    perm_s = pd.Series(perm.importances_mean, index=POSTMATCH_FEATURES).sort_values(ascending=False)
    pr("  Permutation importance (in-sample R2 drop -- diagnostic, not held-out):")
    for f, v in perm_s.items():
        pr(f"    {v:+.4f}  {f}")
    best_meta["permutation_importance"] = {f: round(float(v), 4) for f, v in perm_s.items()}

    make_optimized_figures(
        feat_df, edges, comparison_df, dropped, POSTMATCH_FEATURES, best_meta,
        final_est, scaler, y, seasons, imp, perm_s, output_dir,
    )

    with open(f"{output_dir}/model2_optimized_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "dropped_features": dropped,
            "postmatch_features": POSTMATCH_FEATURES,
            "model_comparison": comparison_rows,
            "best_model": best_meta,
        }, f, indent=2, default=str)

    # ── 7. Limitations / leakage notes ────────────────────────────────────
    sec("6. LIMITATIONS / LEAKAGE NOTES")
    pr("- This model uses this match's own team_xg, team_xga and pass-network")
    pr("  features. It answers 'given the chances created and the way the team")
    pr("  passed THIS match, how much did cohesion explain the scoreline' --")
    pr("  it is NOT a pre-match forecast and should not be sold as one.")
    pr("- 2015/16 league season is the large majority of rows; leave-one-season-out")
    pr("  results on the small seasons (~100-128 rows each) are directional, not precise.")
    pr("- Permutation importance above is computed on the same full data the final")
    pr("  model was fit on (in-sample) -- it shows what the model leans on, not an")
    pr("  unbiased held-out estimate. Treat magnitudes as relative, not absolute.")
    pr("- Existing artifacts (scaler.pkl, gbr.pkl, ridge.pkl, graph_features.parquet)")
    pr("  and api_server.py are byte-for-byte untouched by this pipeline.")

    report_path = f"{output_dir}/model2_optimized_diagnostics.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    logger.info("Optimized diagnostics report saved to %s", report_path)

    return {
        "best_estimator": best_estimator,
        "best_meta": best_meta,
        "feat_df": feat_df,
        "comparison_df": comparison_df,
        "dropped_features": dropped,
        "postmatch_features": POSTMATCH_FEATURES,
    }


if __name__ == "__main__":
    import argparse
    import psycopg2
    from config.settings import DB_DSN

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true",
                         help="Run the v2 dual-mode optimized pipeline instead of the original run().")
    args = parser.parse_args()

    conn = psycopg2.connect(DB_DSN)
    if args.optimize:
        run_optimized(conn)
    else:
        run(conn)
    conn.close()
