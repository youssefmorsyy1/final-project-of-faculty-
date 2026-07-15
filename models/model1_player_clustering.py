"""
models/model1_player_clustering.py
===================================
Dual-Axis Player Profiling — v4.0

Two independent clustering axes:

  SPATIAL  (KMeans, K=9, ALL players including GKs)
    Source  : pass_network_edges
    Features: avg_x_start, avg_y_start  (Set A — diagnostic confirmed)
    Init    : k-means++
    Labels  : Goalkeeper · LCB · RCB · Left Wide Defender · Right Wide Defender
              Defensive Midfielder · Advanced Midfielder · Left Attacker · Right Attacker

  STYLE    (GMM, K selected by BIC 4→10, outfield only)
    Source  : player_match_stats
    Features: 12 per-90 style features
    Output  : soft probability vector; primary = argmax; secondary if ≥ 0.20

Final archetype: "PrimaryStyle · SecondaryStyle · SpatialRole"

Diagnostic script: models/model1_spatial_diagnostic.py
  → K=9 confirmed  |  Set A confirmed  |  k-means++ confirmed

Run:
    python -m models.model1_player_clustering
    python model1_player_clustering.py

Outputs (./artifacts/model1/):
    model1_scaler.pkl, model1_kmeans_<role>.pkl — inference artefacts
    player_clusters.parquet                     — labelled dataset
    cluster_labels.json                         — id → archetype
    model1_report.txt                           — full metric printout
    fig1_*.png … fig6_*.png                     — publication figures
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path("artifacts/model1")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ── spatial constants ──────────────────────────────────────────────────────────
SPATIAL_K        = 9
SPATIAL_FEATURES = ["avg_x_start", "avg_y_start"]
MIN_PASSES       = 10   # minimum total passes for spatial eligibility

# Zone label seeds (x, y) in StatsBomb pitch space (120×80).
# Each cluster centroid is matched to its nearest zone seed.
ZONE_SEEDS: dict[str, tuple[float, float]] = {
    "Goalkeeper":           ( 10.0, 40.0),
    "Left Center Back":     ( 25.0, 25.0),
    "Right Center Back":    ( 25.0, 55.0),
    "Left Wide Defender":   ( 30.0, 10.0),
    "Right Wide Defender":  ( 30.0, 70.0),
    "Defensive Midfielder": ( 55.0, 40.0),
    "Advanced Midfielder":  ( 80.0, 40.0),
    "Left Attacker":        ( 95.0, 15.0),
    "Right Attacker":       ( 95.0, 65.0),
}

# ── style constants ────────────────────────────────────────────────────────────
STYLE_K_MIN = 4
STYLE_K_MAX = 10
MIN_CLUSTER_FRACTION = 0.03   # reject GMM solutions where any cluster < 3% of data
MIN_SECONDARY_PROB   = 0.20   # threshold for secondary style assignment

# 12 per-90 style features from player_match_stats
STYLE_FEATURES = [
    "xg_90",
    "xa_90",
    "shots_90",
    "goals_90",
    "passes_90",
    "pass_accuracy",
    "carries_90",           # progressive_carries / 90 (best proxy in DB)
    "touches_90",           # (passes_attempted + dribbles_completed) / 90
    "progressive_actions_90",  # (progressive_passes + progressive_carries) / 90
    "tackles_90",
    "interceptions_90",
    "pressures_90",
]

# log1p applied before scaling (positively-skewed per-90 rates)
STYLE_LOG1P_COLS = [
    "xg_90", "xa_90", "shots_90", "goals_90",
    "carries_90", "progressive_actions_90",
    "tackles_90", "pressures_90",
]

# Signature vectors for data-driven style cluster labeling (cosine similarity)
STYLE_SIGNATURES: dict[str, dict[str, float]] = {
    "Pure Striker":       {"xg_90": 2.0, "goals_90": 1.5, "shots_90": 1.5, "passes_90": -0.5},
    "Clinical Finisher":  {"goals_90": 2.0, "xg_90": 1.5, "carries_90": 0.5},
    "Chance Creator":     {"xa_90": 2.0, "progressive_actions_90": 1.0, "passes_90": 0.8},
    "Deep Playmaker":     {"passes_90": 2.0, "pass_accuracy": 1.5, "progressive_actions_90": 1.0},
    "Ball Winner":        {"tackles_90": 2.0, "interceptions_90": 1.5, "pressures_90": 0.8},
    "High Press":         {"pressures_90": 2.0, "tackles_90": 1.0, "carries_90": 0.5},
    "Carrier":            {"carries_90": 2.0, "touches_90": 1.0, "progressive_actions_90": 0.8},
    "Box-to-Box":         {"progressive_actions_90": 1.2, "pressures_90": 1.0, "passes_90": 0.5},
    "Defensive Mid":      {"passes_90": 1.0, "pass_accuracy": 1.0, "tackles_90": 1.2, "interceptions_90": 1.0},
    "All-Rounder":        {"progressive_actions_90": 0.5, "passes_90": 0.5, "pressures_90": 0.3},
}

# ── figure colors ──────────────────────────────────────────────────────────────
ARCH_COLORS = [
    "#00B4D8", "#FF6B6B", "#F0B429", "#39D353",
    "#A371F7", "#F4845F", "#56CFE1", "#80FFDB",
    "#FFB347", "#87CEEB", "#DDA0DD",
]


def _night_style() -> dict:
    return dict(
        bg="#0D1117", surface="#161B22", grid="#21262D",
        text="#E6EDF3", muted="#8B949E",
        green="#39D353", teal="#00B4D8", amber="#F0B429",
        coral="#FF6B6B", purple="#A371F7",
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_style_data(conn) -> pd.DataFrame:
    """
    Player-season style aggregates from player_match_stats.
    Returns ALL positions (GKs filtered downstream for GMM only).
    """
    query = """
        SELECT
            p.player_id,
            p.player_name,
            pms.team_id,
            mode() WITHIN GROUP (ORDER BY pms.starting_position) AS modal_position,
            m.season,
            COUNT(pms.stat_id)                  AS matches_played,
            SUM(pms.minutes_played)             AS total_minutes,
            SUM(pms.xg)                         AS xg_sum,
            SUM(pms.xa)                         AS xa_sum,
            SUM(pms.shots)                      AS shots_sum,
            SUM(pms.goals)                      AS goals_sum,
            SUM(pms.passes_attempted)           AS pa_sum,
            SUM(pms.dribbles_completed)         AS dr_sum,
            SUM(pms.progressive_passes)         AS pp_sum,
            SUM(pms.progressive_carries)        AS prc_sum,
            SUM(pms.tackles)                    AS tk_sum,
            SUM(pms.interceptions)              AS int_sum,
            SUM(pms.pressures)                  AS pr_sum,
            AVG(pms.pass_accuracy)              AS pass_accuracy
        FROM player_match_stats pms
        JOIN players p  ON p.player_id  = pms.player_id
        JOIN matches  m ON m.match_id   = pms.match_id
        WHERE pms.minutes_played >= 45
        GROUP BY p.player_id, p.player_name, pms.team_id, m.season
        HAVING COUNT(pms.stat_id) >= 3
           AND SUM(pms.minutes_played) >= 200
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        raise RuntimeError("No style data returned — check DB connection and thresholds.")

    m90 = df["total_minutes"] / 90.0

    df["xg_90"]                  = df["xg_sum"]  / m90
    df["xa_90"]                  = df["xa_sum"]  / m90
    df["shots_90"]               = df["shots_sum"] / m90
    df["goals_90"]               = df["goals_sum"] / m90
    df["passes_90"]              = df["pa_sum"]  / m90
    df["carries_90"]             = df["prc_sum"] / m90
    df["touches_90"]             = (df["pa_sum"] + df["dr_sum"]) / m90
    df["progressive_actions_90"] = (df["pp_sum"] + df["prc_sum"]) / m90
    df["tackles_90"]             = df["tk_sum"]  / m90
    df["interceptions_90"]       = df["int_sum"] / m90
    df["pressures_90"]           = df["pr_sum"]  / m90
    # pass_accuracy is already an AVG ratio from the query

    df["is_gk"] = df["modal_position"] == "Goalkeeper"

    logger.info(
        "Style data: %d player-season rows | %d GKs | %d outfield",
        len(df), df["is_gk"].sum(), (~df["is_gk"]).sum(),
    )
    return df


def load_spatial_data(conn) -> pd.DataFrame:
    """
    Player-season spatial aggregates from pass_network_edges.
    Weighted by pass_count.  Includes all positions (GKs not excluded).
    """
    query = """
        SELECT
            pne.passer_id                         AS player_id,
            m.season,
            SUM(pne.pass_count)                   AS total_passes,
            SUM(pne.avg_x_start * pne.pass_count)
                / NULLIF(SUM(pne.pass_count), 0)  AS avg_x_start,
            SUM(pne.avg_y_start * pne.pass_count)
                / NULLIF(SUM(pne.pass_count), 0)  AS avg_y_start
        FROM pass_network_edges pne
        JOIN matches m ON m.match_id = pne.match_id
        GROUP BY pne.passer_id, m.season
        HAVING SUM(pne.pass_count) >= %(min_passes)s
    """
    with conn.cursor() as cur:
        cur.execute(query, {"min_passes": MIN_PASSES})
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    df = df.dropna(subset=SPATIAL_FEATURES).reset_index(drop=True)

    logger.info("Spatial data: %d player-season rows", len(df))
    return df


def load_features(conn) -> pd.DataFrame:
    """
    Merge style and spatial data on (player_id, season).
    Returns inner-joined dataset; both axes required per player-season.
    """
    style_df   = load_style_data(conn)
    spatial_df = load_spatial_data(conn)

    df = style_df.merge(
        spatial_df[["player_id", "season", "avg_x_start", "avg_y_start", "total_passes"]],
        on=["player_id", "season"],
        how="inner",
    )

    logger.info(
        "Merged dataset: %d rows | %d unique players",
        len(df), df["player_id"].nunique(),
    )
    return df


# ── ground-truth position → broad group (for external validation) ──────────────
_GT_BROAD: dict[str, str] = {
    "Goalkeeper":               "GK",
    "Center Back":              "CB", "Left Center Back":  "CB", "Right Center Back":  "CB",
    "Left Back":                "FB", "Right Back":        "FB",
    "Left Wing Back":           "FB", "Right Wing Back":   "FB",
    "Center Defensive Midfield":"DM", "Left Defensive Midfield": "DM", "Right Defensive Midfield": "DM",
    "Center Midfield":          "CM", "Left Center Midfield": "CM", "Right Center Midfield": "CM",
    "Center Attacking Midfield":"AM", "Left Attacking Midfield": "AM", "Right Attacking Midfield": "AM",
    "Left Midfield":            "WM", "Right Midfield":    "WM",
    "Left Wing":                "W",  "Right Wing":        "W",
    "Center Forward":           "ST", "Left Center Forward": "ST", "Right Center Forward": "ST",
    "Secondary Striker":        "ST",
}


def _load_gt_positions(conn) -> pd.DataFrame:
    """Modal StatsBomb starting position per player-season (min 45 min/match)."""
    query = """
        SELECT pms.player_id, m.season,
               mode() WITHIN GROUP (ORDER BY pms.starting_position) AS modal_position
        FROM player_match_stats pms
        JOIN matches m ON m.match_id = pms.match_id
        WHERE pms.minutes_played >= 45
        GROUP BY pms.player_id, m.season
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def _spatial_ext_report(df: pd.DataFrame, gt: pd.DataFrame) -> tuple[list[str], dict]:
    """
    Compute ARI, NMI, L/R side-accuracy, and per-zone purity against StatsBomb
    ground-truth positions.  Returns (report_lines, metrics_dict).
    """
    # rename to avoid collision with modal_position already in df (from load_style_data)
    gt = gt.rename(columns={"modal_position": "gt_modal_position"})
    merged = df.merge(gt, on=["player_id", "season"], how="left")
    merged["gt_broad"] = merged["gt_modal_position"].map(_GT_BROAD)
    ev = merged[merged["gt_broad"].notna()].copy()

    ari = adjusted_rand_score(ev["gt_broad"], ev["spatial_cluster_id"])
    nmi = normalized_mutual_info_score(ev["gt_broad"], ev["spatial_cluster_id"])

    def _side(s: str) -> str | None:
        if str(s).startswith("Left"):  return "L"
        if str(s).startswith("Right"): return "R"
        return None

    ev["lbl_side"]  = ev["spatial_cluster_name"].map(_side)
    ev["true_side"] = ev["gt_modal_position"].map(_side)
    sided = ev[ev["lbl_side"].notna() & ev["true_side"].notna()]
    side_acc = float((sided["lbl_side"] == sided["true_side"]).mean()) if len(sided) else float("nan")

    lines = [
        "\n── Spatial external validation (vs StatsBomb modal_position) ───────",
        f"  n evaluated              : {len(ev):,}",
        f"  Adjusted Rand Index      : {ari:.4f}",
        f"  Normalized Mutual Info   : {nmi:.4f}",
        f"  L/R side-accuracy        : {side_acc:.4f}  (n={len(sided):,})",
        "  Per-zone purity:",
    ]
    for zone in sorted(ev["spatial_cluster_name"].unique()):
        sub = ev[ev["spatial_cluster_name"] == zone]
        vc  = sub["gt_modal_position"].value_counts()
        purity = vc.iloc[0] / len(sub) if len(sub) else 0
        lines.append(
            f"    {zone:<22}  n={len(sub):>4}  top='{vc.index[0]}' ({purity*100:.0f}%)"
        )

    metrics = {"ari": round(ari, 4), "nmi": round(nmi, 4), "side_accuracy": round(side_acc, 4)}
    return lines, metrics


def _style_posterior_report(proba: np.ndarray, chosen_k: int) -> tuple[list[str], dict]:
    """Posterior confidence summary for the GMM style model."""
    mx   = proba.max(axis=1)
    ent  = -(proba * np.log(proba + 1e-12)).sum(axis=1)
    ll   = float(np.mean(np.log(proba.max(axis=1) + 1e-12)))   # approx per-sample LL
    lines = [
        "\n  Posterior confidence (outfield):",
        f"    Mean max-posterior      : {mx.mean():.3f}",
        f"    Median max-posterior    : {float(np.median(mx)):.3f}",
        f"    Frac confident (>=0.80) : {(mx >= 0.80).mean():.3f}",
        f"    Mean entropy            : {ent.mean():.3f} nats  (max={np.log(chosen_k):.3f})",
    ]
    metrics = {
        "mean_max_posterior":    round(float(mx.mean()), 4),
        "median_max_posterior":  round(float(np.median(mx)), 4),
        "frac_confident_80":     round(float((mx >= 0.80).mean()), 4),
        "mean_entropy":          round(float(ent.mean()), 4),
        "max_entropy":           round(float(np.log(chosen_k)), 4),
    }
    return lines, metrics


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def apply_style_log1p(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in STYLE_LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def fit_style_scaler(df_log: pd.DataFrame) -> tuple[RobustScaler, np.ndarray]:
    X = df_log[STYLE_FEATURES].fillna(0).values
    scaler = RobustScaler()
    return scaler, scaler.fit_transform(X)


def fit_spatial_scaler(df: pd.DataFrame) -> tuple[StandardScaler, np.ndarray]:
    X = df[SPATIAL_FEATURES].fillna(0).values
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(X)


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL MODEL (KMeans, K=9, all players)
# ══════════════════════════════════════════════════════════════════════════════

def assign_spatial_labels(centers_raw: np.ndarray) -> dict[int, str]:
    """
    Greedy nearest-seed assignment.
    centers_raw : (K, 2) in original pitch coordinates (x: 0-120, y: 0-80)
    Returns dict mapping cluster_id → zone name.
    """
    seeds = list(ZONE_SEEDS.items())
    k = len(centers_raw)

    # Build distance matrix: shape (k_clusters, n_zone_seeds)
    seed_coords = np.array([c for _, c in seeds])
    dist = np.linalg.norm(
        centers_raw[:, None, :] - seed_coords[None, :, :], axis=2
    )  # (k, n_seeds)

    labels: dict[int, str] = {}
    assigned_seeds: set[int] = set()

    # Greedy: pick minimum distance pairs first
    flat_order = np.argsort(dist.ravel())
    for idx in flat_order:
        ci, si = divmod(int(idx), len(seeds))
        if ci in labels or si in assigned_seeds:
            continue
        labels[ci] = seeds[si][0]
        assigned_seeds.add(si)
        if len(labels) == k:
            break

    # Fallback for extra clusters beyond 9 zone seeds (shouldn't happen at K=9)
    for ci in range(k):
        labels.setdefault(ci, f"Zone {ci}")

    return labels


def fit_spatial_model(
    X_scaled: np.ndarray,
    spatial_scaler: StandardScaler,
) -> tuple[KMeans, np.ndarray, dict[int, str], float, float, float]:
    """
    Fit KMeans (K=9, k-means++).
    Returns (kmeans, labels, zone_label_map, silhouette, davies_bouldin, calinski_harabasz).
    """
    km = KMeans(
        n_clusters=SPATIAL_K,
        init="k-means++",
        n_init=30,
        random_state=42,
    )
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)

    centers_raw = spatial_scaler.inverse_transform(km.cluster_centers_)
    zone_map    = assign_spatial_labels(centers_raw)

    return km, labels, zone_map, sil, db, ch


# ══════════════════════════════════════════════════════════════════════════════
# STYLE MODEL (GMM, BIC selection K=4-10, outfield only)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_gmm(X: np.ndarray, k: int, seed: int = 42) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=5,
        random_state=seed,
        max_iter=300,
    )
    gmm.fit(X)
    return gmm


def select_gmm_k(X: np.ndarray) -> tuple[int, GaussianMixture, dict]:
    """
    Select K by lowest BIC, subject to:
      - all clusters have ≥ MIN_CLUSTER_FRACTION of the data
    Returns (chosen_k, gmm, bic_table_dict).
    """
    n = len(X)
    min_size = max(1, int(n * MIN_CLUSTER_FRACTION))

    records: list[dict] = []
    best_bic  = np.inf
    best_gmm  = None
    chosen_k  = STYLE_K_MIN

    for k in range(STYLE_K_MIN, STYLE_K_MAX + 1):
        gmm = _fit_gmm(X, k)
        labels = gmm.predict(X)
        counts = np.bincount(labels, minlength=k)

        if counts.min() < min_size:
            records.append({"k": k, "bic": np.nan, "aic": np.nan,
                            "log_likelihood": np.nan, "valid": False})
            continue

        bic = gmm.bic(X)
        aic = gmm.aic(X)
        ll  = gmm.score(X) * n   # total log-likelihood

        records.append({"k": k, "bic": round(bic, 1), "aic": round(aic, 1),
                        "log_likelihood": round(ll, 1), "valid": True})

        if bic < best_bic:
            best_bic, best_gmm, chosen_k = bic, gmm, k

    if best_gmm is None:
        # Fallback: accept any solution even if cluster-size constraint fails
        logger.warning("GMM: no valid solution found; relaxing cluster-size constraint.")
        for k in range(STYLE_K_MIN, STYLE_K_MAX + 1):
            gmm = _fit_gmm(X, k)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic, best_gmm, chosen_k = bic, gmm, k

    return chosen_k, best_gmm, {"records": records, "chosen_k": chosen_k, "best_bic": best_bic}


def label_style_clusters(
    gmm: GaussianMixture,
    style_scaler: RobustScaler,
) -> dict[int, str]:
    """
    Assign style archetype names to GMM components via cosine similarity
    between the component mean (in original feature space) and STYLE_SIGNATURES.
    """
    means_scaled = gmm.means_                           # (k, n_features)
    means_raw    = style_scaler.inverse_transform(means_scaled)
    means_df     = pd.DataFrame(means_raw, columns=STYLE_FEATURES)

    # z-score across components for fair comparison
    std_ = means_df.std(ddof=0).replace(0, 1)
    means_z = (means_df - means_df.mean()) / std_

    scored: list[tuple[float, int, str]] = []
    for cid in means_z.index:
        z = means_z.loc[cid]
        for sig_name, weights in STYLE_SIGNATURES.items():
            common = [f for f in weights if f in z.index]
            if not common:
                continue
            sig_vec  = np.array([weights[f] for f in common])
            cent_vec = np.array([float(z[f]) for f in common])
            ns, nc = np.linalg.norm(sig_vec), np.linalg.norm(cent_vec)
            sim = float(np.dot(sig_vec, cent_vec) / (ns * nc)) if ns > 0 and nc > 0 else 0.0
            scored.append((sim, int(cid), sig_name))

    scored.sort(reverse=True)
    labels: dict[int, str] = {}
    used: set[str] = set()
    for sim, cid, name in scored:
        if cid in labels or name in used:
            continue
        labels[cid] = name
        used.add(name)

    for cid in range(len(gmm.means_)):
        labels.setdefault(cid, "Hybrid")

    return labels


def assign_style(gmm: GaussianMixture, X_outfield: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (primary_ids, secondary_ids, proba_matrix).
    secondary_ids == -1 where second-highest prob < MIN_SECONDARY_PROB.
    """
    proba = gmm.predict_proba(X_outfield)
    primary = proba.argmax(axis=1)
    sorted_idx = np.argsort(proba, axis=1)[:, ::-1]
    secondary_ids  = sorted_idx[:, 1]
    secondary_prob = proba[np.arange(len(proba)), secondary_ids]
    secondary_ids  = np.where(secondary_prob >= MIN_SECONDARY_PROB, secondary_ids, -1)
    return primary, secondary_ids, proba


# ══════════════════════════════════════════════════════════════════════════════
# FINAL LABEL CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_final_archetype(row: pd.Series) -> str:
    """
    Combine spatial role + style into a display archetype string.
    GKs get "Goalkeeper" only.
    """
    spatial = str(row.get("spatial_cluster_name", "")) or "Unknown Zone"

    if row.get("is_gk", False):
        return "Goalkeeper"

    primary   = str(row.get("primary_style", "")) or ""
    secondary = str(row.get("secondary_style", "")) or ""

    if primary and secondary:
        return f"{primary} · {secondary} · {spatial}"
    if primary:
        return f"{primary} · {spatial}"
    return spatial


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

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


def make_figures(df: pd.DataFrame, spatial_metrics: dict, style_metrics: dict, chosen_k: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping figures.")
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

    # ── Fig 1: Spatial pitch scatter ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    zones = df["spatial_cluster_name"].dropna().unique()
    for i, zone in enumerate(sorted(zones)):
        sub = df[df["spatial_cluster_name"] == zone]
        col = ARCH_COLORS[i % len(ARCH_COLORS)]
        ax1.scatter(
            sub["avg_x_start"], sub["avg_y_start"],
            c=col, s=12, alpha=0.45, linewidths=0, label=zone,
        )
        cx, cy = sub["avg_x_start"].mean(), sub["avg_y_start"].mean()
        ax1.scatter(cx, cy, c=col, s=250, marker="*",
                    edgecolors="white", linewidths=0.8, zorder=5)
        ax1.text(cx, cy + 1.5, zone.split()[-1], fontsize=7,
                 color=col, ha="center", va="bottom", fontweight="bold")

    # Pitch outline
    for rect_args in [
        dict(xy=(0, 0), width=120, height=80),      # pitch
        dict(xy=(0, 18), width=18, height=44),       # left penalty box
        dict(xy=(102, 18), width=18, height=44),     # right penalty box
    ]:
        ax1.add_patch(plt.Rectangle(
            fill=False, edgecolor=N["grid"], linewidth=1.2, **rect_args
        ))
    ax1.axvline(60, color=N["grid"], linewidth=1.0, linestyle="-", alpha=0.5)
    ax1.set_xlim(-2, 122)
    ax1.set_ylim(-2, 82)
    ax1.set_xlabel("x (0 = own goal, 120 = opposition goal)")
    ax1.set_ylabel("y (0 = right touchline, 80 = left touchline)")
    sil = spatial_metrics.get("silhouette", 0)
    ax1.set_title(f"Spatial Clustering — K={SPATIAL_K} Zones  (sil={sil:.4f})",
                  fontweight="bold")
    leg = ax1.legend(fontsize=8, loc="upper left", framealpha=0.2, labelcolor=N["text"])
    leg.get_frame().set_facecolor(N["surface"])
    _apply_night(fig1)
    plt.tight_layout()
    fig1.savefig(ARTIFACT_DIR / "fig1_pca_scatter.png",
                 dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig1)

    # ── Fig 2: Silhouette + BIC summary bars ─────────────────────────────────
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

    # left: spatial silhouette per zone (bar of how many players per zone)
    zone_counts = df["spatial_cluster_name"].value_counts().sort_values(ascending=False)
    bar_cols = [ARCH_COLORS[i % len(ARCH_COLORS)] for i in range(len(zone_counts))]
    ax2a.bar(range(len(zone_counts)), zone_counts.values,
             color=bar_cols, edgecolor=N["bg"], linewidth=0.5, zorder=3)
    ax2a.set_xticks(range(len(zone_counts)))
    ax2a.set_xticklabels(zone_counts.index, rotation=35, ha="right", fontsize=8)
    ax2a.set_title(f"Spatial Zone — Player Counts  (sil={sil:.4f})", fontweight="bold")
    ax2a.set_ylabel("Player-Season Rows")

    # right: GMM BIC curve
    bic_records = style_metrics.get("bic_records", [])
    if bic_records:
        valid = [r for r in bic_records if r.get("valid")]
        if valid:
            ks   = [r["k"] for r in valid]
            bics = [r["bic"] for r in valid]
            ax2b.plot(ks, bics, marker="o", color=N["teal"], linewidth=2)
            ax2b.axvline(chosen_k, color=N["amber"], linestyle="--",
                         linewidth=1.5, label=f"Chosen K={chosen_k}")
            ax2b.set_xlabel("K (number of style clusters)")
            ax2b.set_ylabel("BIC")
            ax2b.set_title("Style GMM — BIC Curve", fontweight="bold")
            ax2b.legend(fontsize=9, framealpha=0.2, labelcolor=N["text"])

    _apply_night(fig2)
    plt.tight_layout()
    fig2.savefig(ARTIFACT_DIR / "fig2_silhouette_bars.png",
                 dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig2)

    # ── Fig 3: Style radar profiles (outfield GMM clusters) ──────────────────
    RADAR_FEATS = [
        "xg_90", "xa_90", "shots_90", "goals_90",
        "passes_90", "pass_accuracy",
        "carries_90", "touches_90", "progressive_actions_90",
        "tackles_90", "interceptions_90", "pressures_90",
    ]
    RADAR_LABELS = [
        "xG/90", "xA/90", "Shots/90", "Goals/90",
        "Passes/90", "Pass Acc%",
        "Carries/90", "Touches/90", "Prog Act/90",
        "Tackles/90", "Intercept/90", "Pressures/90",
    ]
    n_feat  = len(RADAR_FEATS)
    angles  = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    outfield = df[~df["is_gk"]].copy()
    styles_present = sorted(outfield["primary_style"].dropna().unique())
    arch_means = outfield.groupby("primary_style")[RADAR_FEATS].mean()
    arch_z = (arch_means - arch_means.mean()) / (arch_means.std(ddof=0).replace(0, 1))

    n_s = len(styles_present)
    cols_r = min(4, n_s)
    rows_r = (n_s + cols_r - 1) // cols_r if n_s else 1
    fig3, axes3 = plt.subplots(
        rows_r, cols_r,
        figsize=(cols_r * 4.5, rows_r * 4.5),
        subplot_kw=dict(polar=True),
    )
    fig3.suptitle("Style Archetype Radar Profiles (GMM)",
                  color=N["text"], fontsize=13, fontweight="bold")
    ax3_flat = np.array(axes3).flatten() if n_s > 1 else [axes3]

    style_counts = outfield["primary_style"].value_counts()
    for idx, style in enumerate(styles_present):
        if idx >= len(ax3_flat):
            break
        ax = ax3_flat[idx]
        ax.set_facecolor(N["surface"])
        col  = ARCH_COLORS[idx % len(ARCH_COLORS)]
        if style not in arch_z.index:
            ax.set_visible(False)
            continue
        vals = arch_z.loc[style, RADAR_FEATS].values.clip(-2, 2) + 2
        vals_plot = list(vals) + [vals[0]]
        ax.plot(angles, vals_plot, color=col, linewidth=2)
        ax.fill(angles, vals_plot, color=col, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(RADAR_LABELS, size=7, color=N["muted"])
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["-2", "-1", "0", "+1", "+2"], size=6, color=N["muted"])
        ax.set_ylim(0, 4)
        ax.grid(color=N["grid"], linewidth=0.4)
        ax.spines["polar"].set_color(N["grid"])
        n_p = style_counts.get(style, 0)
        ax.set_title(f"{style}\n(n={n_p:,})", color=N["text"],
                     fontsize=8, fontweight="bold", pad=10)

    for idx in range(n_s, len(ax3_flat)):
        ax3_flat[idx].set_visible(False)

    fig3.patch.set_facecolor(N["bg"])
    plt.tight_layout()
    fig3.savefig(ARTIFACT_DIR / "fig3_radar_profiles.png",
                 dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig3)

    # ── Fig 4: Style feature heatmap ──────────────────────────────────────────
    if not arch_z.empty:
        fig4, ax4 = plt.subplots(figsize=(18, max(5, len(arch_z) * 0.7 + 2)))
        im = ax4.imshow(arch_z[RADAR_FEATS].values,
                        cmap=plt.cm.RdYlGn, vmin=-2, vmax=2, aspect="auto")
        ax4.set_xticks(range(n_feat))
        ax4.set_xticklabels(RADAR_LABELS, rotation=35, ha="right",
                            color=N["text"], fontsize=9)
        ax4.set_yticks(range(len(arch_z)))
        ax4.set_yticklabels(arch_z.index, color=N["text"], fontsize=9)
        for i in range(len(arch_z)):
            for j in range(n_feat):
                val  = arch_z.values[i, j]
                tcol = N["bg"] if abs(val) > 1.0 else N["text"]
                ax4.text(j, i, f"{val:+.1f}", ha="center", va="center",
                         fontsize=7.5, color=tcol, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax4, shrink=0.7, pad=0.02)
        cbar.set_label("Z-score vs mean", color=N["text"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=N["muted"])
        ax4.set_title("Style Feature Heatmap — Z-scored Cluster Means",
                      color=N["text"], fontweight="bold", fontsize=12, pad=12)
        ax4.set_facecolor(N["surface"])
        fig4.patch.set_facecolor(N["bg"])
        for sp in ax4.spines.values():
            sp.set_edgecolor(N["grid"])
        plt.tight_layout()
        fig4.savefig(ARTIFACT_DIR / "fig4_feature_heatmap.png",
                     dpi=150, bbox_inches="tight", facecolor=N["bg"])
        plt.close(fig4)

    # ── Fig 5: Archetype distribution bar ────────────────────────────────────
    arch_counts = df["final_player_archetype"].value_counts().head(20)
    fig5, ax5 = plt.subplots(figsize=(16, 6))
    bar_cols5 = [ARCH_COLORS[i % len(ARCH_COLORS)] for i in range(len(arch_counts))]
    ax5.barh(range(len(arch_counts)), arch_counts.values,
             color=bar_cols5, edgecolor=N["bg"], linewidth=0.5)
    ax5.set_yticks(range(len(arch_counts)))
    ax5.set_yticklabels(arch_counts.index, fontsize=8)
    ax5.invert_yaxis()
    ax5.set_xlabel("Player-Season Rows")
    ax5.set_title("Final Archetype Distribution (top 20)", fontweight="bold")
    _apply_night(fig5)
    plt.tight_layout()
    fig5.savefig(ARTIFACT_DIR / "fig5_archetype_distribution.png",
                 dpi=150, bbox_inches="tight", facecolor=N["bg"])
    plt.close(fig5)

    # ── Fig 6: Top players per primary style ─────────────────────────────────
    if "primary_style" in df.columns and "player_name" in df.columns:
        fig6, ax6 = plt.subplots(figsize=(14, 6))
        top_rows = []
        for style in sorted(outfield["primary_style"].dropna().unique()):
            sub = outfield[outfield["primary_style"] == style]
            if sub.empty:
                continue
            # proxy for prominence: xg_90 + xa_90
            sub = sub.copy()
            sub["prominence"] = sub.get("xg_90", 0) + sub.get("xa_90", 0)
            top = sub.nlargest(1, "prominence")[["player_name", "primary_style"]].copy()
            top["label"] = f"{top['primary_style'].iloc[0]}: {top['player_name'].iloc[0]}"
            top_rows.append(top)
        if top_rows:
            top_df = pd.concat(top_rows, ignore_index=True)
            ax6.barh(
                range(len(top_df)),
                [1] * len(top_df),
                color=[ARCH_COLORS[i % len(ARCH_COLORS)] for i in range(len(top_df))],
            )
            ax6.set_yticks(range(len(top_df)))
            ax6.set_yticklabels(top_df["label"].tolist(), fontsize=9)
            ax6.invert_yaxis()
            ax6.set_xlabel("")
            ax6.set_xticks([])
            ax6.set_title("Top Player Representative per Style Cluster", fontweight="bold")
        _apply_night(fig6)
        plt.tight_layout()
        fig6.savefig(ARTIFACT_DIR / "fig6_top_players.png",
                     dpi=150, bbox_inches="tight", facecolor=N["bg"])
        plt.close(fig6)

    logger.info("Figures saved to %s", ARTIFACT_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN run()
# ══════════════════════════════════════════════════════════════════════════════

def run(conn, output_dir: str = "artifacts/model1") -> dict[str, Any]:
    """
    Dual-axis player clustering pipeline (v4.0).

    Parameters
    ----------
    conn       : psycopg2 connection (already open)
    output_dir : directory for all artefacts

    Returns
    -------
    dict with keys: spatial_kmeans, style_gmm, style_scaler, spatial_scaler,
                    df, labels, spatial_silhouette, style_bic, chosen_gmm_k
    """
    global ARTIFACT_DIR
    ARTIFACT_DIR = Path(output_dir)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    report: list[str] = []
    report.append("=" * 65)
    report.append("MODEL 1 — DUAL-AXIS PLAYER CLUSTERING  v4.0")
    report.append("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Model 1: loading features ...")
    df = load_features(conn)
    report.append(f"\nRows loaded : {len(df):,}  |  Players: {df['player_id'].nunique():,}")
    report.append(f"GKs         : {df['is_gk'].sum():,}  |  Outfield: {(~df['is_gk']).sum():,}")

    # ── 2. Spatial model — ALL players ────────────────────────────────────────
    logger.info("Fitting spatial KMeans (K=%d) ...", SPATIAL_K)
    spatial_scaler, X_spatial = fit_spatial_scaler(df)
    km, spatial_labels, zone_map, sil_sp, db_sp, ch_sp = fit_spatial_model(
        X_spatial, spatial_scaler
    )

    df["spatial_cluster_id"]   = spatial_labels
    df["spatial_cluster_name"] = pd.Series(spatial_labels).map(zone_map).values

    spatial_metrics = {
        "k": SPATIAL_K,
        "silhouette": round(sil_sp, 4),
        "davies_bouldin": round(db_sp, 4),
        "calinski_harabasz": round(ch_sp, 1),
        "n": len(df),
    }

    report.append("\n── Spatial clustering ──────────────────────────────────────")
    report.append(f"  K = {SPATIAL_K}  |  sil={sil_sp:.4f}  db={db_sp:.4f}  ch={ch_sp:.1f}")
    report.append("  Zone assignments:")
    for cid, name in sorted(zone_map.items()):
        n_zone = int((spatial_labels == cid).sum())
        centers_raw = spatial_scaler.inverse_transform(km.cluster_centers_)
        cx, cy = centers_raw[cid]
        report.append(f"    {cid}: {name:<28} n={n_zone:>5}  centroid=({cx:.1f},{cy:.1f})")

    joblib.dump(km,             ARTIFACT_DIR / "model1_kmeans_spatial.pkl")
    joblib.dump(spatial_scaler, ARTIFACT_DIR / "model1_scaler_spatial.pkl")

    # ── 2b. Spatial external validation ──────────────────────────────────────
    try:
        gt = _load_gt_positions(conn)
        ext_lines, ext_metrics = _spatial_ext_report(df, gt)
        report.extend(ext_lines)
        spatial_metrics.update(ext_metrics)
    except Exception as exc:
        logger.warning("Spatial external validation skipped: %s", exc)
        gt = pd.DataFrame()

    # ── 3. Style model — OUTFIELD only ───────────────────────────────────────
    logger.info("Fitting style GMM (k-sweep %d-%d) ...", STYLE_K_MIN, STYLE_K_MAX)
    df_log    = apply_style_log1p(df)
    style_scaler, X_style_all = fit_style_scaler(df_log)

    outfield_mask = (~df["is_gk"]).values
    X_outfield    = X_style_all[outfield_mask]

    chosen_k, gmm, bic_info = select_gmm_k(X_outfield)

    style_labels_out, secondary_ids_out, proba_out = assign_style(gmm, X_outfield)
    style_label_map = label_style_clusters(gmm, style_scaler)

    # Populate style columns for all rows
    df["style_cluster_id"] = -1
    df["primary_style"]    = ""
    df["secondary_style"]  = ""
    df["chosen_gmm_k"]     = chosen_k

    out_idx = np.where(outfield_mask)[0]
    for i, row_i in enumerate(out_idx):
        pid     = int(style_labels_out[i])
        sec_id  = int(secondary_ids_out[i])
        df.at[df.index[row_i], "style_cluster_id"] = pid
        df.at[df.index[row_i], "primary_style"]    = style_label_map.get(pid, "Hybrid")
        df.at[df.index[row_i], "secondary_style"]  = (
            style_label_map.get(sec_id, "") if sec_id >= 0 else ""
        )

    # Store probability vectors as JSON strings
    proba_strings = [""] * len(df)
    for i, row_i in enumerate(out_idx):
        proba_strings[row_i] = json.dumps(
            {style_label_map.get(j, str(j)): round(float(p), 4)
             for j, p in enumerate(proba_out[i])}
        )
    df["probability_vector"] = proba_strings

    # GKs: override style with "Goalkeeper"
    df.loc[df["is_gk"], "primary_style"]    = "Goalkeeper"
    df.loc[df["is_gk"], "secondary_style"]  = ""
    df.loc[df["is_gk"], "style_cluster_id"] = -1

    n_with_secondary = (df["secondary_style"] != "").sum()

    style_metrics = {
        "chosen_k":       chosen_k,
        "bic":            round(bic_info["best_bic"], 1) if not np.isnan(bic_info["best_bic"]) else None,
        "bic_records":    bic_info["records"],
        "n_outfield":     int(outfield_mask.sum()),
        "n_with_secondary": int(n_with_secondary),
    }

    report.append("\n── Style clustering (GMM) ──────────────────────────────────")
    report.append(f"  Chosen K = {chosen_k}  |  BIC = {style_metrics['bic']}")
    report.append(f"  Players with secondary style (>={MIN_SECONDARY_PROB:.0%}): {n_with_secondary:,}")
    report.append("  BIC sweep:")
    for rec in bic_info["records"]:
        valid_str = "OK" if rec.get("valid") else "SKIP (small cluster)"
        mark = " <-- chosen" if rec["k"] == chosen_k else ""
        report.append(
            f"    k={rec['k']}  BIC={rec['bic']}  AIC={rec['aic']}  {valid_str}{mark}"
        )
    report.append("\n  Style cluster breakdown:")
    primary_counts = df[~df["is_gk"]]["primary_style"].value_counts()
    n_outfield = float(outfield_mask.sum())
    for style, cnt in primary_counts.items():
        report.append(f"    {style:<30} {cnt:>5,}  ({cnt/n_outfield*100:.1f}%)")

    post_lines, post_metrics = _style_posterior_report(proba_out, chosen_k)
    report.extend(post_lines)
    style_metrics.update(post_metrics)

    joblib.dump(gmm,          ARTIFACT_DIR / "model1_gmm_style.pkl")
    joblib.dump(style_scaler, ARTIFACT_DIR / "model1_scaler.pkl")

    # ── 4. Final archetype label ───────────────────────────────────────────────
    df["final_player_archetype"] = df.apply(build_final_archetype, axis=1)

    report.append("\n── Final archetype distribution (top 20) ────────────────────")
    arch_counts = df["final_player_archetype"].value_counts().head(20)
    for arch, cnt in arch_counts.items():
        report.append(f"  {arch:<50} {cnt:>5,}")

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    output_cols = [
        "player_id", "player_name", "season", "team_id",
        "matches_played", "total_minutes", "is_gk",
        "avg_x_start", "avg_y_start",
        "spatial_cluster_id", "spatial_cluster_name",
        "style_cluster_id", "chosen_gmm_k",
        "probability_vector", "primary_style", "secondary_style",
        "final_player_archetype",
    ] + STYLE_FEATURES

    df_out = df[[c for c in output_cols if c in df.columns]]
    df_out.to_parquet(ARTIFACT_DIR / "player_clusters.parquet", index=False)
    logger.info("player_clusters.parquet saved (%d rows).", len(df_out))

    # cluster_labels.json  — spatial and style maps for API/debug
    label_map = {
        "spatial": {str(k): v for k, v in zone_map.items()},
        "style":   {str(k): v for k, v in style_label_map.items()},
    }
    with open(ARTIFACT_DIR / "cluster_labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    report_path = ARTIFACT_DIR / "model1_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    logger.info("Report saved to %s", report_path)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
    print("\n" + "=" * 65)
    print("MODEL 1 COMPLETE - v4.0 DUAL-AXIS SUMMARY")
    print("=" * 65)
    for line in report:
        print(line)

    # ── 6. Figures ────────────────────────────────────────────────────────────
    make_figures(df, spatial_metrics, style_metrics, chosen_k)

    return {
        "spatial_kmeans":     km,
        "spatial_scaler":     spatial_scaler,
        "style_gmm":          gmm,
        "style_scaler":       style_scaler,
        "df":                 df,
        "labels":             label_map,
        "spatial_silhouette": sil_sp,
        "style_bic":          style_metrics["bic"],
        "chosen_gmm_k":       chosen_k,
        "zone_map":           zone_map,
        "style_label_map":    style_label_map,
        "spatial_metrics":    spatial_metrics,
        "style_metrics":      style_metrics,
        "_registry": {
            "model_key": "model1_player_clustering",
            "version": "4.0",
            "display_name": "Player Efficiency & Style Profiling",
            "task": "clustering",
            "algorithm": "KMeans (spatial axis) + GaussianMixture (style axis)",
            "target": "player archetype (unsupervised, dual-axis)",
            "features": list(STYLE_FEATURES),
            "metrics": {"spatial": spatial_metrics, "style": style_metrics},
            "n_train_rows": int(len(df_out)),
            "artifact_path": str(ARTIFACT_DIR),
            "prediction_table": "model1_player_clusters",
        },
        "_predictions": {"model1_player_clusters": df_out},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Backward-compatible helpers used by api_server.py
# ══════════════════════════════════════════════════════════════════════════════

def position_group(pos) -> str:
    """Legacy stub — spatial clustering no longer uses position groups."""
    if not pos:
        return "Unknown"
    if "Goalkeeper" in str(pos):
        return "Goalkeeper"
    return "Outfield"


# ══════════════════════════════════════════════════════════════════════════════
# __main__ entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    try:
        from config.settings import DB_DSN
    except ImportError:
        DB_DSN = os.environ.get(
            "DB_DSN", "postgresql://postgres:111222@localhost:5432/soccer_db"
        )

    try:
        conn = psycopg2.connect(DB_DSN)
        logger.info("Connected to DB.")
    except Exception as exc:
        logger.error("Cannot connect to DB: %s", exc)
        sys.exit(1)

    try:
        result = run(conn)
        print(f"\nSpatial silhouette : {result['spatial_silhouette']:.4f}")
        print(f"Style BIC          : {result['style_bic']}")
        print(f"Style GMM K chosen : {result['chosen_gmm_k']}")
    finally:
        conn.close()
