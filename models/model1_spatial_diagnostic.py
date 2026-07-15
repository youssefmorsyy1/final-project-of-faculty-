"""
models/model1_spatial_diagnostic.py
=====================================
Phase 0 Diagnostic — Spatial Clustering Validation

Validates assumptions for the dual-axis player clustering refactor:
  - optimal K for spatial KMeans
  - whether (x,y) alone is sufficient vs adding (x_end, y_end)
  - whether manual centroid seeding beats k-means++
  - centroid drift from manual seeds

Run:
    python -m models.model1_spatial_diagnostic
    python models/model1_spatial_diagnostic.py

Output:
    MODEL1_SPATIAL_DIAGNOSTIC.txt  (repo root)
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
K_SWEEP_MIN = 5
K_SWEEP_MAX = 13
K_TARGET    = 9
N_INIT_RUNS = 10
MIN_PASSES  = 10   # minimum total passes for a player-season to qualify

OUTPUT_FILE = Path("MODEL1_SPATIAL_DIAGNOSTIC.txt")

# Manual centroid seeds: (x, y) in StatsBomb pitch space (120×80)
MANUAL_SEEDS_9 = np.array([
    [10,  40],   # Goalkeeper
    [25,  55],   # Left Center Back
    [25,  25],   # Right Center Back
    [30,  70],   # Left Wide Back / WB
    [30,  10],   # Right Wide Back / WB
    [55,  40],   # Defensive Midfielder
    [80,  40],   # Advanced Midfielder
    [95,  65],   # Left Attacker / LW
    [95,  15],   # Right Attacker / RW
], dtype=float)

SEED_NAMES_9 = [
    "Goalkeeper",
    "Left Center Back",
    "Right Center Back",
    "Left Wide Defender",
    "Right Wide Defender",
    "Defensive Midfielder",
    "Advanced Midfielder",
    "Left Attacker",
    "Right Attacker",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_spatial_data(conn) -> pd.DataFrame:
    """
    Aggregate pass_network_edges to player-season level, weighted by pass_count.
    Excludes goalkeepers (diagnostic phase only) via mode starting_position.
    """
    query = """
        WITH player_season_positions AS (
            SELECT
                pms.player_id,
                m.season,
                mode() WITHIN GROUP (ORDER BY pms.starting_position) AS modal_position
            FROM player_match_stats pms
            JOIN matches m ON m.match_id = pms.match_id
            WHERE pms.minutes_played >= 45
            GROUP BY pms.player_id, m.season
        ),
        spatial_agg AS (
            SELECT
                pne.passer_id                         AS player_id,
                m.season,
                SUM(pne.pass_count)                   AS total_passes,
                SUM(pne.avg_x_start * pne.pass_count)
                    / NULLIF(SUM(pne.pass_count), 0)  AS avg_x_start,
                SUM(pne.avg_y_start * pne.pass_count)
                    / NULLIF(SUM(pne.pass_count), 0)  AS avg_y_start,
                SUM(pne.avg_x_end   * pne.pass_count)
                    / NULLIF(SUM(pne.pass_count), 0)  AS avg_x_end,
                SUM(pne.avg_y_end   * pne.pass_count)
                    / NULLIF(SUM(pne.pass_count), 0)  AS avg_y_end
            FROM pass_network_edges pne
            JOIN matches m ON m.match_id = pne.match_id
            GROUP BY pne.passer_id, m.season
        )
        SELECT
            sa.player_id,
            sa.season,
            sa.total_passes,
            sa.avg_x_start,
            sa.avg_y_start,
            sa.avg_x_end,
            sa.avg_y_end,
            psp.modal_position
        FROM spatial_agg sa
        LEFT JOIN player_season_positions psp
            ON psp.player_id = sa.player_id AND psp.season = sa.season
        WHERE sa.total_passes >= %(min_passes)s
    """
    with conn.cursor() as cur:
        cur.execute(query, {"min_passes": MIN_PASSES})
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    logger.info("Raw rows: %d", len(df))

    # Drop rows with null spatial features (shouldn't happen but defensive)
    df = df.dropna(subset=["avg_x_start", "avg_y_start", "avg_x_end", "avg_y_end"])

    # Exclude goalkeepers for diagnostic phase
    gk_mask = df["modal_position"] == "Goalkeeper"
    n_gk = gk_mask.sum()
    df = df[~gk_mask].reset_index(drop=True)

    logger.info(
        "After GK exclusion: %d rows | %d unique players | %d GKs excluded",
        len(df),
        df["player_id"].nunique(),
        n_gk,
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# K SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_k_sweep(X: np.ndarray, feature_set_label: str) -> pd.DataFrame:
    """Run KMeans K sweep from K_SWEEP_MIN to K_SWEEP_MAX. Returns metrics DataFrame."""
    records = []
    for k in range(K_SWEEP_MIN, K_SWEEP_MAX + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=N_INIT_RUNS, random_state=42)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        db  = davies_bouldin_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)
        records.append({
            "feature_set": feature_set_label,
            "k": k,
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
            "calinski_harabasz": round(ch, 1),
        })
        logger.info("  %s k=%2d  sil=%.4f  db=%.4f  ch=%.1f", feature_set_label, k, sil, db, ch)

    return pd.DataFrame(records)


def best_k_from_sweep(sweep_df: pd.DataFrame) -> int:
    """
    Composite rank: sil (desc) + db (asc) + ch (desc).
    Returns k with lowest composite rank (ties broken by silhouette).
    """
    df = sweep_df.copy()
    df["r_sil"] = df["silhouette"].rank(ascending=False)
    df["r_db"]  = df["davies_bouldin"].rank(ascending=True)
    df["r_ch"]  = df["calinski_harabasz"].rank(ascending=False)
    df["composite"] = df["r_sil"] + df["r_db"] + df["r_ch"]
    best = df.sort_values(["composite", "silhouette"], ascending=[True, False]).iloc[0]
    return int(best["k"])


# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION TEST (K=9)
# ══════════════════════════════════════════════════════════════════════════════

def test_kmeans_plus_plus(X: np.ndarray) -> dict:
    """10 runs with k-means++; return best silhouette and mean/std across runs."""
    sils = []
    best_sil, best_labels, best_centers = -1.0, None, None
    for seed in range(N_INIT_RUNS):
        km = KMeans(n_clusters=K_TARGET, init="k-means++", n_init=1, random_state=seed * 7)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sils.append(sil)
        if sil > best_sil:
            best_sil, best_labels, best_centers = sil, labels, km.cluster_centers_
    return {
        "method": "k-means++",
        "best_sil": round(best_sil, 4),
        "mean_sil": round(float(np.mean(sils)), 4),
        "std_sil":  round(float(np.std(sils)), 4),
        "best_labels": best_labels,
        "best_centers": best_centers,
    }


def test_random_init(X: np.ndarray) -> dict:
    """10 runs with random init; return best silhouette and mean/std across runs."""
    sils = []
    best_sil, best_labels, best_centers = -1.0, None, None
    for seed in range(N_INIT_RUNS):
        km = KMeans(n_clusters=K_TARGET, init="random", n_init=1, random_state=seed * 13)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sils.append(sil)
        if sil > best_sil:
            best_sil, best_labels, best_centers = sil, labels, km.cluster_centers_
    return {
        "method": "random",
        "best_sil": round(best_sil, 4),
        "mean_sil": round(float(np.mean(sils)), 4),
        "std_sil":  round(float(np.std(sils)), 4),
        "best_labels": best_labels,
        "best_centers": best_centers,
    }


def test_manual_seeds(X: np.ndarray, scaler: StandardScaler) -> dict:
    """
    Single run with manually defined centroid seeds (scaled to match X's space).
    Uses only (x, y) for seeding even if X contains 4 features — seeds are
    zero-padded for the end coordinates (neutral prior).
    """
    n_features = X.shape[1]
    seeds_raw = MANUAL_SEEDS_9.copy().astype(float)   # shape (9, 2) — (x, y) only

    # Build full seed matrix: (x_start, y_start [, x_end=x_start, y_end=y_start])
    if n_features == 4:
        # Extend each seed: end coords = start coords (neutral prior)
        seeds_full = np.hstack([seeds_raw, seeds_raw])
    else:
        seeds_full = seeds_raw

    seeds_scaled = scaler.transform(seeds_full)

    km = KMeans(n_clusters=K_TARGET, init=seeds_scaled, n_init=1, random_state=42)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)

    return {
        "method": "manual_seeds",
        "best_sil": round(sil, 4),
        "mean_sil": round(sil, 4),   # single run
        "std_sil":  0.0,
        "best_labels": labels,
        "best_centers": km.cluster_centers_,
        "final_centers_raw": scaler.inverse_transform(km.cluster_centers_),
    }


def compute_drift(scaler: StandardScaler, final_centers: np.ndarray, n_features: int) -> pd.DataFrame:
    """
    Compute Euclidean distance between each manual seed and its final centroid
    after fitting, in the original (unscaled) coordinate space.
    """
    seeds_raw = MANUAL_SEEDS_9.copy().astype(float)

    if n_features == 4:
        seeds_full = np.hstack([seeds_raw, seeds_raw])
    else:
        seeds_full = seeds_raw

    rows = []
    for i, name in enumerate(SEED_NAMES_9):
        seed = seeds_full[i]
        centroid = final_centers[i]
        drift = float(np.linalg.norm(centroid - seed))
        rows.append({
            "cluster": name,
            "seed_x": seed[0],
            "seed_y": seed[1],
            "final_x": round(centroid[0], 2),
            "final_y": round(centroid[1], 2),
            "drift": round(drift, 3),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# DECISIONS
# ══════════════════════════════════════════════════════════════════════════════

def decide_k(best_k_a: int, best_k_b: int, sweep_a: pd.DataFrame, sweep_b: pd.DataFrame) -> tuple[int, str]:
    """
    Lock K=9 if it is within 5% of the best silhouette for the chosen feature set.
    Otherwise use best K.
    """
    # Use feature set A metrics as the primary reference for K decision
    row_k9 = sweep_a[sweep_a["k"] == K_TARGET]
    sil_k9 = float(row_k9["silhouette"].iloc[0]) if not row_k9.empty else 0.0
    best_sil_a = float(sweep_a["silhouette"].max())
    threshold = best_sil_a * 0.95

    if sil_k9 >= threshold:
        return K_TARGET, f"K=9 sil={sil_k9:.4f} is within 5% of best ({best_sil_a:.4f}) → LOCK K=9"
    else:
        chosen = best_k_a
        return chosen, (
            f"K=9 sil={sil_k9:.4f} is below 5% threshold ({threshold:.4f}) "
            f"→ USE BEST K={chosen}"
        )


def decide_features(sweep_a: pd.DataFrame, sweep_b: pd.DataFrame, k: int) -> tuple[str, str]:
    """Use Set B if it improves silhouette at the chosen K by ≥1%."""
    sil_a = float(sweep_a[sweep_a["k"] == k]["silhouette"].iloc[0])
    sil_b = float(sweep_b[sweep_b["k"] == k]["silhouette"].iloc[0])
    if sil_b >= sil_a * 1.01:
        return "B", f"Set B sil={sil_b:.4f} improves over Set A sil={sil_a:.4f} at K={k} → USE SET B"
    return "A", f"Set A sil={sil_a:.4f} sufficient (Set B={sil_b:.4f}) → USE SET A"


def decide_init(kpp: dict, rand: dict, manual: dict) -> tuple[str, str]:
    """Prefer manual seeds if they beat k-means++ best silhouette; else use k-means++."""
    if manual["best_sil"] >= kpp["best_sil"]:
        return "manual", (
            f"Manual seeds sil={manual['best_sil']:.4f} >= k-means++ sil={kpp['best_sil']:.4f} "
            "→ USE MANUAL SEEDS"
        )
    return "k-means++", (
        f"k-means++ sil={kpp['best_sil']:.4f} > manual sil={manual['best_sil']:.4f} "
        "→ USE K-MEANS++"
    )


# ══════════════════════════════════════════════════════════════════════════════
# REPORT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_report(
    df_raw: pd.DataFrame,
    sweep_a: pd.DataFrame,
    sweep_b: pd.DataFrame,
    best_k_a: int,
    best_k_b: int,
    kpp: dict,
    rand: dict,
    manual: dict,
    drift_df: pd.DataFrame,
    chosen_k: int,
    k_reason: str,
    chosen_feat: str,
    feat_reason: str,
    chosen_init: str,
    init_reason: str,
) -> str:
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("MODEL 1 — SPATIAL CLUSTERING DIAGNOSTIC REPORT")
    lines.append("=" * 70)
    lines.append(f"\nDataset: {len(df_raw):,} player-season rows | "
                 f"{df_raw['player_id'].nunique():,} unique players")
    lines.append("Goalkeepers: EXCLUDED (diagnostic phase)")
    lines.append(f"Min total passes filter: {MIN_PASSES}")

    # ── K Sweep ────────────────────────────────────────────────────────────
    lines.append("\n" + "─" * 70)
    lines.append("K SWEEP  (K = 5 → 13,  k-means++, 10 runs each)")
    lines.append("─" * 70)

    for label, sweep in [("Set A: (x_start, y_start)", sweep_a),
                          ("Set B: (x_start, y_start, x_end, y_end)", sweep_b)]:
        lines.append(f"\n{label}")
        lines.append(f"  {'K':>3}  {'Silhouette':>12}  {'Davies-Bouldin':>16}  {'Calinski-Harabasz':>20}")
        lines.append(f"  {'-'*3}  {'-'*12}  {'-'*16}  {'-'*20}")
        for _, row in sweep.iterrows():
            marker = " ◄ best" if int(row["k"]) == (best_k_a if label.startswith("Set A") else best_k_b) else ""
            k9mark = " [K=9]" if int(row["k"]) == K_TARGET else ""
            lines.append(
                f"  {int(row['k']):>3}  {row['silhouette']:>12.4f}  "
                f"{row['davies_bouldin']:>16.4f}  {row['calinski_harabasz']:>20.1f}"
                f"{k9mark}{marker}"
            )
        lines.append(f"  → Best K for this set: {best_k_a if label.startswith('Set A') else best_k_b}")

    # ── Initialization Test (K=9) ──────────────────────────────────────────
    lines.append("\n" + "─" * 70)
    lines.append(f"INITIALIZATION TEST  (K={K_TARGET}, {N_INIT_RUNS} runs each)")
    lines.append("─" * 70)
    for result in [kpp, rand, manual]:
        n_runs = 1 if result["method"] == "manual_seeds" else N_INIT_RUNS
        lines.append(
            f"  {result['method']:<15}  best_sil={result['best_sil']:.4f}  "
            f"mean={result['mean_sil']:.4f}  std={result['std_sil']:.4f}  "
            f"({n_runs} run{'s' if n_runs > 1 else ''})"
        )

    # ── Centroid Drift ─────────────────────────────────────────────────────
    lines.append("\n" + "─" * 70)
    lines.append("CENTROID DRIFT  (manual seeds → final centroids, raw pitch coords)")
    lines.append("─" * 70)
    lines.append(
        f"  {'Cluster':<24}  {'Seed(x,y)':>14}  {'Final(x,y)':>14}  {'Drift':>8}"
    )
    lines.append(f"  {'-'*24}  {'-'*14}  {'-'*14}  {'-'*8}")
    for _, row in drift_df.iterrows():
        lines.append(
            f"  {row['cluster']:<24}  "
            f"({row['seed_x']:>5.1f},{row['seed_y']:>5.1f})  "
            f"({row['final_x']:>5.1f},{row['final_y']:>5.1f})  "
            f"{row['drift']:>8.3f}"
        )
    lines.append(f"\n  Mean drift : {drift_df['drift'].mean():.3f}")
    lines.append(f"  Max drift  : {drift_df['drift'].max():.3f}")
    lines.append(f"  (Cluster with max drift: {drift_df.loc[drift_df['drift'].idxmax(), 'cluster']})")

    # ── Decisions ─────────────────────────────────────────────────────────
    lines.append("\n" + "─" * 70)
    lines.append("DECISIONS")
    lines.append("─" * 70)
    lines.append(f"\n  Spatial K         : {chosen_k}")
    lines.append(f"  Reason            : {k_reason}")
    lines.append(f"\n  Feature Set       : {chosen_feat}")
    lines.append(f"  Reason            : {feat_reason}")
    lines.append(f"\n  Init Method       : {chosen_init}")
    lines.append(f"  Reason            : {init_reason}")

    # ── Recommendations ────────────────────────────────────────────────────
    lines.append("\n" + "─" * 70)
    lines.append("RECOMMENDATIONS FOR PRODUCTION")
    lines.append("─" * 70)
    lines.append(f"\n  FIX K=9            : {'YES — use K=9' if chosen_k == K_TARGET else f'NO — use K={chosen_k}'}")
    lines.append(f"  USE avg_x_end/y_end: {'YES — include Set B features' if chosen_feat == 'B' else 'NO — Set A (x,y) only is sufficient'}")
    lines.append(f"  INIT METHOD        : {chosen_init}")

    lines.append("\n" + "=" * 70)
    lines.append("END OF DIAGNOSTIC REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostic(conn) -> None:
    logger.info("Phase 0: loading spatial data ...")
    df = load_spatial_data(conn)

    FEAT_A = ["avg_x_start", "avg_y_start"]
    FEAT_B = ["avg_x_start", "avg_y_start", "avg_x_end", "avg_y_end"]

    X_raw_a = df[FEAT_A].values.astype(float)
    X_raw_b = df[FEAT_B].values.astype(float)

    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    X_a = scaler_a.fit_transform(X_raw_a)
    X_b = scaler_b.fit_transform(X_raw_b)

    # ── K sweep ──────────────────────────────────────────────────────────
    logger.info("Running K sweep for Set A ...")
    sweep_a = run_k_sweep(X_a, "A")
    logger.info("Running K sweep for Set B ...")
    sweep_b = run_k_sweep(X_b, "B")

    best_k_a = best_k_from_sweep(sweep_a)
    best_k_b = best_k_from_sweep(sweep_b)
    logger.info("Best K (Set A) = %d | Best K (Set B) = %d", best_k_a, best_k_b)

    # ── Initialization test at K=9 (use Set A for init comparison) ───────
    logger.info("Testing initialization methods at K=%d ...", K_TARGET)
    kpp    = test_kmeans_plus_plus(X_a)
    rand   = test_random_init(X_a)
    manual = test_manual_seeds(X_a, scaler_a)

    logger.info(
        "Init results: k-means++ best=%.4f | random best=%.4f | manual best=%.4f",
        kpp["best_sil"], rand["best_sil"], manual["best_sil"],
    )

    # ── Centroid drift ────────────────────────────────────────────────────
    final_centers_raw = manual["final_centers_raw"][:, :2]   # (x, y) only
    drift_df = compute_drift(scaler_a, final_centers_raw, n_features=2)

    # ── Decisions ─────────────────────────────────────────────────────────
    chosen_k, k_reason     = decide_k(best_k_a, best_k_b, sweep_a, sweep_b)
    chosen_feat, feat_reason = decide_features(sweep_a, sweep_b, chosen_k)
    chosen_init, init_reason = decide_init(kpp, rand, manual)

    # ── Build and save report ──────────────────────────────────────────────
    report = build_report(
        df, sweep_a, sweep_b,
        best_k_a, best_k_b,
        kpp, rand, manual,
        drift_df,
        chosen_k, k_reason,
        chosen_feat, feat_reason,
        chosen_init, init_reason,
    )

    OUTPUT_FILE.write_text(report, encoding="utf-8")
    logger.info("Diagnostic report saved to %s", OUTPUT_FILE)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
    print(report)


if __name__ == "__main__":
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
        run_diagnostic(conn)
    finally:
        conn.close()
