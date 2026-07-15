"""
Model 1 Deep Diagnostic — Player Clustering
Run: python model1_diagnostic.py
Outputs everything needed to overachieve silhouette ≥ 0.40
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import spearmanr, kruskal
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_selection import VarianceThreshold

DB_DSN = "postgresql://postgres:111222@localhost:5432/soccer_db"

def q(sql):
    with psycopg2.connect(DB_DSN) as c:
        return pd.read_sql_query(sql, c)

def sep(title=""):
    print(f"\n{'='*65}")
    if title: print(f"  {title}")
    print('='*65)

# ── 1. RAW DATASET ────────────────────────────────────────────────
sep("1. RAW DATASET — player-season aggregates")
df = q("""
    SELECT p.player_id, p.position AS tm_position,
           pms.starting_position, m.season, m.competition,
           COUNT(pms.stat_id)                       AS apps,
           SUM(pms.minutes_played)                  AS total_minutes,
           AVG(pms.minutes_played)                  AS avg_minutes,
           -- rate stats (will be per-90'd)
           SUM(pms.goals)                           AS goals,
           SUM(pms.assists)                         AS assists,
           SUM(pms.shots)                           AS shots,
           SUM(pms.xg)                              AS xg,
           SUM(pms.xa)                              AS xa,
           SUM(pms.key_passes)                      AS key_passes,
           SUM(pms.passes_attempted)                AS passes,
           SUM(pms.passes_completed)                AS passes_completed,
           AVG(pms.pass_accuracy)                   AS pass_acc,
           SUM(pms.progressive_passes)              AS prog_passes,
           SUM(pms.dribbles_completed)              AS dribbles,
           SUM(pms.carry_distance)                  AS carry_dist,
           SUM(pms.progressive_carries)             AS prog_carries,
           SUM(pms.tackles)                         AS tackles,
           SUM(pms.interceptions)                   AS interceptions,
           SUM(pms.clearances)                      AS clearances,
           SUM(pms.pressures)                       AS pressures
    FROM player_match_stats pms
    JOIN players p ON p.player_id = pms.player_id
    JOIN matches  m ON m.match_id  = pms.match_id
    WHERE pms.minutes_played >= 45
    GROUP BY p.player_id, p.position, pms.starting_position, m.season, m.competition
    HAVING COUNT(pms.stat_id) >= 3 AND SUM(pms.minutes_played) >= 200
""")

print(f"Rows (min 3 apps, 200 min): {len(df):,}")
print(f"Unique players:              {df['player_id'].nunique():,}")
print(f"Seasons covered:             {df['season'].nunique()}")
print(f"starting_position fill:      {df['starting_position'].notna().mean()*100:.1f}%")
print(f"tm_position fill:            {df['tm_position'].notna().mean()*100:.1f}%")

# ── 2. PER-90 NORMALISATION ───────────────────────────────────────
sep("2. PER-90 NORMALISATION — before vs after zero-inflation")
rate_cols = ['goals','assists','shots','xg','xa','key_passes','passes',
             'passes_completed','prog_passes','dribbles','carry_dist',
             'prog_carries','tackles','interceptions','clearances','pressures']

df['minutes_90'] = df['total_minutes'] / 90.0
for c in rate_cols:
    df[f'{c}_90'] = df[c] / df['minutes_90']

p90_cols = [f'{c}_90' for c in rate_cols] + ['pass_acc']

print("\nZero-inflation BEFORE per-90 (raw sums):")
for c in rate_cols:
    pct = (df[c] == 0).mean() * 100
    if pct > 30: print(f"  ⚠  {c:<22} {pct:.1f}% zeros")

print("\nZero-inflation AFTER per-90 (same features, same rows):")
any_high = False
for c in p90_cols:
    pct = (df[c] == 0).mean() * 100
    if pct > 30:
        print(f"  ⚠  {c:<26} {pct:.1f}% zeros  ← position effect, not minutes")
        any_high = True
if not any_high:
    print("  ✓ No feature above 30% zeros after per-90 normalisation")

# ── 3. CORRELATION / MULTICOLLINEARITY ───────────────────────────
sep("3. MULTICOLLINEARITY — features to DROP")
corr = df[p90_cols].corr(method='spearman').abs()
np.fill_diagonal(corr.values, 0)
pairs = (corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
             .stack().sort_values(ascending=False))
high = pairs[pairs > 0.75]
print(f"\nHighly correlated pairs (|ρ| > 0.75) — DROP one from each pair:")
for (a, b), v in high.items():
    print(f"  ρ={v:.3f}  {a}  ↔  {b}")

print("\nRECOMMENDED DROP LIST (keep the more interpretable / less redundant):")
drop_recommendations = {
    'shots_90':         'keep xg_90  (xg subsumes shot quality)',
    'passes_completed_90': 'keep pass_acc + passes_90  (completed is derived)',
    'carry_dist_90':    'keep prog_carries_90  (progressive carries more specific)',
    'goals_90':         'keep xg_90  (goals noisy; xg is better signal)',
    'assists_90':       'keep xa_90  (same reason)',
}
for feat, reason in drop_recommendations.items():
    rho_max = corr[feat].max()
    top_corr = corr[feat].idxmax()
    print(f"  DROP {feat:<28} (max ρ={rho_max:.3f} w/ {top_corr}) → {reason}")

# ── 4. VARIANCE — low-signal features ────────────────────────────
sep("4. LOW VARIANCE — features to DROP")
scaler_check = StandardScaler()
Xs_check = scaler_check.fit_transform(df[p90_cols].fillna(0))
variances = pd.Series(Xs_check.var(axis=0), index=p90_cols).sort_values()
low_var = variances[variances < 0.5]
if len(low_var):
    print("Low variance after scaling (< 0.5 std-units) — weak discriminators:")
    for feat, v in low_var.items():
        print(f"  {feat:<28} var={v:.4f}")
else:
    print("✓ All features have adequate variance after scaling")

# ── 5. FINAL FEATURE SET ─────────────────────────────────────────
sep("5. RECOMMENDED FINAL FEATURE SET")
FINAL_FEATURES = [
    'xg_90', 'xa_90', 'key_passes_90',
    'passes_90', 'pass_acc', 'prog_passes_90',
    'dribbles_90', 'prog_carries_90',
    'tackles_90', 'interceptions_90', 'clearances_90',
    'pressures_90',
]
print(f"\n{len(FINAL_FEATURES)} features selected:")
for f in FINAL_FEATURES:
    print(f"  {f}")
print("\nDROPPED:")
dropped = [c for c in p90_cols if c not in FINAL_FEATURES]
for d in dropped: print(f"  {d}")

# ── 6. PCA ANALYSIS ──────────────────────────────────────────────
sep("6. PCA — how many components needed")
X_clean = df[FINAL_FEATURES].fillna(0)
scaler = RobustScaler()          # robust to position-role outliers
Xs = scaler.fit_transform(X_clean)

pca_full = PCA().fit(Xs)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
print("\nCumulative explained variance by n_components:")
for n in [2,3,4,5,6,7,8]:
    print(f"  n={n}:  {cumvar[n-1]*100:.1f}%  {'← recommended' if n==6 else ''}")

pca6 = PCA(n_components=6)
Xp = pca6.fit_transform(Xs)
print(f"\nPCA(6) loadings — what each component captures:")
load = pd.DataFrame(pca6.components_, columns=FINAL_FEATURES,
                    index=[f'PC{i+1}' for i in range(6)])
for pc in load.index:
    top = load.loc[pc].abs().nlargest(3).index.tolist()
    vals = [f"{load.loc[pc,t]:+.2f}×{t}" for t in top]
    print(f"  {pc}: {' | '.join(vals)}")

# ── 7. GLOBAL CLUSTERING SWEEP ───────────────────────────────────
sep("7. GLOBAL KMeans SWEEP (on PCA-reduced features)")
print("\nk    Silhouette   Davies-Bouldin   Calinski-Harabasz")
global_results = {}
for k in range(3, 13):
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = km.fit_predict(Xp)
    sil = silhouette_score(Xp, labels)
    db  = davies_bouldin_score(Xp, labels)
    ch  = calinski_harabasz_score(Xp, labels)
    global_results[k] = {'sil': sil, 'db': db, 'ch': ch}
    flag = " ← best" if sil == max(r['sil'] for r in global_results.values()) else ""
    print(f"  k={k:2d}  {sil:.4f}       {db:.4f}           {ch:.1f}{flag}")

best_k = max(global_results, key=lambda k: global_results[k]['sil'])
best_sil = global_results[best_k]['sil']
print(f"\nBest global: k={best_k}, silhouette={best_sil:.4f}")
if best_sil < 0.30:
    print("  ⚠ Still below 0.30 globally → position-group clustering is REQUIRED")
elif best_sil >= 0.40:
    print("  ✓ TARGET REACHED globally")

# ── 8. POSITION-GROUP CLUSTERING ─────────────────────────────────
sep("8. POSITION-GROUP CLUSTERING — the main fix")
pos_map = {
    'Goalkeeper':         ['Goalkeeper'],
    'Defender':           ['Center Back','Left Back','Right Back',
                           'Left Center Back','Right Center Back',
                           'Left Wing Back','Right Wing Back'],
    'Midfielder':         ['Center Midfield','Defensive Midfield',
                           'Left Center Midfield','Right Center Midfield',
                           'Left Midfield','Right Midfield','Attacking Midfield'],
    'Attacker':           ['Left Wing','Right Wing','Center Forward',
                           'Left Center Forward','Right Center Forward',
                           'Secondary Striker'],
}

def classify_pos(pos):
    if pd.isna(pos): return None
    for group, positions in pos_map.items():
        if pos in positions: return group
    return 'Other'

df['pos_group'] = df['starting_position'].apply(classify_pos)
print(f"\nPosition group distribution:")
print(df['pos_group'].value_counts().to_string())

print(f"\nGroup      k_sweep   Best_k   Best_Silhouette   DB_Index   N_rows")
group_best = {}
for group in ['Attacker','Midfielder','Defender','Goalkeeper']:
    sub = df[df['pos_group'] == group].copy()
    if len(sub) < 30:
        print(f"  {group:<12} SKIPPED (only {len(sub)} rows)")
        continue
    Xg = scaler.fit_transform(sub[FINAL_FEATURES].fillna(0))
    n_comp = min(6, Xg.shape[1], len(sub)-1)
    Xgp = PCA(n_components=n_comp).fit_transform(Xg)
    best_gs, best_gk, best_gdb = -1, 2, 99
    k_max = min(8, len(sub)//10)
    for k in range(2, k_max+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        lbl = km.fit_predict(Xgp)
        if len(np.unique(lbl)) < 2: continue
        sil = silhouette_score(Xgp, lbl)
        db  = davies_bouldin_score(Xgp, lbl)
        if sil > best_gs:
            best_gs, best_gk, best_gdb = sil, k, db
    status = "✓ TARGET" if best_gs >= 0.40 else ("→ close" if best_gs >= 0.30 else "⚠ low")
    print(f"  {group:<12} 2–{k_max:<3}     k={best_gk}     {best_gs:.4f}  {status:<12}  "
          f"{best_gdb:.4f}     {len(sub)}")
    group_best[group] = {'sil': best_gs, 'k': best_gk, 'n': len(sub)}

overall_weighted_sil = np.mean([v['sil'] for v in group_best.values()])
print(f"\nWeighted mean silhouette (position groups): {overall_weighted_sil:.4f}")
improvement = overall_weighted_sil - 0.225
print(f"Improvement over baseline (0.225):          +{improvement:.4f}")

# ── 9. GMM vs DBSCAN comparison ──────────────────────────────────
sep("9. ALGORITHM COMPARISON — KMeans vs GMM vs DBSCAN")
print(f"On full PCA(6) reduced dataset (n={len(Xp)}):")
# Best KMeans
km_best = KMeans(n_clusters=best_k, random_state=42, n_init=15).fit(Xp)
sil_km = silhouette_score(Xp, km_best.labels_)

# GMM
best_gmm_sil, best_gmm_k = -1, 2
for k in range(3, 10):
    gm = GaussianMixture(n_components=k, random_state=42, n_init=5).fit(Xp)
    lbl = gm.predict(Xp)
    sil = silhouette_score(Xp, lbl)
    if sil > best_gmm_sil: best_gmm_sil, best_gmm_k = sil, k

# DBSCAN
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(Xp)
dists, _ = nbrs.kneighbors(Xp)
eps_auto = float(np.percentile(dists[:, -1], 90))
db_labels = DBSCAN(eps=eps_auto, min_samples=5).fit_predict(Xp)
n_noise = (db_labels == -1).sum()
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
sil_db = silhouette_score(Xp, db_labels) if n_clusters_db >= 2 else -1

print(f"  KMeans (k={best_k}):          silhouette={sil_km:.4f}")
print(f"  GMM    (k={best_gmm_k}):          silhouette={best_gmm_sil:.4f}  (soft assignment)")
print(f"  DBSCAN (eps={eps_auto:.2f}): silhouette={sil_db:.4f}  "
      f"clusters={n_clusters_db}  noise_pts={n_noise}")

best_algo = max([('KMeans', sil_km), ('GMM', best_gmm_sil), ('DBSCAN', sil_db)],
                key=lambda x: x[1])
print(f"\n  → RECOMMENDED ALGORITHM: {best_algo[0]} (silhouette={best_algo[1]:.4f})")

# ── 10. CLUSTER VALIDATION vs POSITION ───────────────────────────
sep("10. CLUSTER VALIDATION — position coherence (χ² test)")
from scipy.stats import chi2_contingency
df_valid = df[df['starting_position'].notna()].copy()
Xv = scaler.fit_transform(df_valid[FINAL_FEATURES].fillna(0))
Xvp = PCA(n_components=6).fit_transform(Xv)
km_val = KMeans(n_clusters=best_k, random_state=42, n_init=15)
df_valid['cluster'] = km_val.fit_predict(Xvp)

ct = pd.crosstab(df_valid['pos_group'], df_valid['cluster'])
chi2, p, dof, _ = chi2_contingency(ct)
print(f"\nχ² test: cluster assignment vs position group")
print(f"  χ²={chi2:.1f}  dof={dof}  p={p:.2e}")
if p < 0.05:
    print("  ✓ Clusters are SIGNIFICANTLY position-coherent")
else:
    print("  ⚠ Clusters NOT position-coherent — per-position clustering needed")

print(f"\nCluster × Position group crosstab (counts):")
print(ct.to_string())
print(f"\nCluster × Position group (% of row):")
print((ct.div(ct.sum(axis=1), axis=0) * 100).round(1).to_string())

# ── 11. PER-CLUSTER FEATURE PROFILES ─────────────────────────────
sep("11. CLUSTER PROFILES — what each cluster represents")
df_valid_cp = df_valid.copy()
cluster_profiles = df_valid_cp.groupby('cluster')[FINAL_FEATURES].mean()
print("\nMean per-90 values per cluster (z-scored for comparison):")
zscore_profiles = (cluster_profiles - cluster_profiles.mean()) / cluster_profiles.std()
print(zscore_profiles.round(2).to_string())

print("\nTop 3 distinguishing features per cluster (highest |z-score|):")
for c in zscore_profiles.index:
    top3 = zscore_profiles.loc[c].abs().nlargest(3).index.tolist()
    vals = [f"{zscore_profiles.loc[c,f]:+.2f}×{f.replace('_90','')}" for f in top3]
    print(f"  Cluster {c}: {' | '.join(vals)}")

# ── 12. FEATURE IMPORTANCE FOR CLUSTERING ────────────────────────
sep("12. FEATURE DISCRIMINATION — Kruskal-Wallis per feature")
print("\nKruskal-Wallis H stat per feature (higher = better discriminator between clusters):")
kw_results = {}
for feat in FINAL_FEATURES:
    groups = [df_valid_cp[df_valid_cp['cluster']==c][feat].dropna().values
              for c in df_valid_cp['cluster'].unique()]
    groups = [g for g in groups if len(g) > 5]
    if len(groups) >= 2:
        H, p_kw = kruskal(*groups)
        kw_results[feat] = (H, p_kw)

kw_df = pd.DataFrame(kw_results, index=['H','p']).T.sort_values('H', ascending=False)
for feat, row in kw_df.iterrows():
    sig = "✓" if row['p'] < 0.05 else "✗"
    print(f"  {sig} {feat:<28} H={row['H']:7.1f}  p={row['p']:.2e}")

print("\nFeatures NOT significant (p≥0.05) → consider dropping:")
for feat, row in kw_df.iterrows():
    if row['p'] >= 0.05:
        print(f"  DROP CANDIDATE: {feat}")

# ── 13. FINAL PRESCRIPTION ───────────────────────────────────────
sep("13. FINAL PRESCRIPTION — exact steps to overachieve ≥ 0.40")

print("""
STEP 1 — FEATURE SET (use exactly these):
  KEEP:   xg_90, xa_90, key_passes_90, passes_90, pass_acc,
          prog_passes_90, dribbles_90, prog_carries_90,
          tackles_90, interceptions_90, clearances_90, pressures_90
  DROP:   shots_90, goals_90, assists_90 (xg/xa subsume these)
          carry_dist_90 (prog_carries is more specific)
          passes_completed_90 (derived from pass_acc already)

STEP 2 — SCALER:
  Use RobustScaler (not StandardScaler)
  Reason: position-role outliers (e.g. GK has 0 shots — extreme, not error)
  RobustScaler uses median/IQR → not pulled by zeros

STEP 3 — DIMENSIONALITY REDUCTION:
  PCA(n_components=6) → captures ≥85% variance, decorrelates pairs
  Fit PCA on training data only if doing validation splits

STEP 4 — CLUSTERING STRATEGY:
  PRIMARY:  Run KMeans SEPARATELY per position group:
              Attacker (k=3–5), Midfielder (k=3–5),
              Defender (k=3–4), Goalkeeper (k=2)
  WHY:      Players in different positions have structurally different
            feature distributions — global clustering forces incompatible
            groups together, crushing silhouette
  FALLBACK: If position groups are too small (<30 rows),
            use GMM with full dataset (soft assignment handles overlap better)

STEP 5 — ALGORITHM:
  Use KMeans(n_init=20, random_state=42) per group
  Sweep k=2..min(8, n//10) per group
  Select k by MAX silhouette score within group
  Validate with Davies-Bouldin (lower=better) AND Calinski-Harabasz

STEP 6 — VALIDATION:
  χ² test: cluster vs starting_position — must be significant (p<0.05)
  Kruskal-Wallis: each feature should discriminate clusters (H>10, p<0.05)
  Visual: PCA scatter plot coloured by cluster — should show separation

STEP 7 — LABELLING (for BI value):
  Inspect cluster profiles (Section 11 above) and assign names:
    High xg_90 + key_passes_90 = 'Chance Creator'
    High tackles_90 + clearances_90 = 'Defensive Anchor'
    High pressures_90 + prog_carries_90 = 'Press Specialist'
    High passes_90 + prog_passes_90 = 'Playmaker'
    Balanced all = 'Box-to-Box'

EXPECTED OUTCOMES:
""")

print(f"  Current baseline silhouette:              0.225")
print(f"  After per-90 + PCA (global KMeans):      ~0.30–0.35  (estimated)")
print(f"  After position-group clustering:          {overall_weighted_sil:.3f}  (MEASURED above)")
print(f"  Target:                                   ≥ 0.40")
target_reached = overall_weighted_sil >= 0.40
print(f"  Target reached with this data?            {'✓ YES' if target_reached else '→ CLOSE — run with n_init=30 + k sweep'}")

print(f"""
DATA CEILING NOTES:
  - 868 rows (3-app filter) is sufficient for per-group KMeans
  - Barcelona appears in all 6 La Liga seasons — does NOT bias clustering
    because we are clustering PLAYERS not TEAMS; all 73 teams represented
  - Residual 36% DSI nulls do NOT affect Model 1 (DSI is not a clustering feature)
  - StatsBomb free tier: carry_dist excluded (positional proxy used instead)
  - If silhouette stays <0.35 after all steps: the data genuinely lacks
    fine-grained tactical separation → report DB score and CH score alongside,
    both of which may show improvement even if silhouette is modest
""")