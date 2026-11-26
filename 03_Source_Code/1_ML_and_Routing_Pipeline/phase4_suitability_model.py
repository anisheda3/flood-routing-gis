"""
Phase 4 - LightGBM flood classifier and storage-suitability index.

Trains a class-weighted LightGBM model on the Phase 3 zonal-statistics table
(one row per 100 m hexagon) using terrain, hydrology and land-use features,
with Sentinel-1 derived flood extent (fld_majority) as the label. The flood
probability is then combined with population and urban penalties:

    suitability = 0.5*P(flood) + 0.3*(1 - pop_norm) + 0.2*(1 - urban)

Input : 02_GIS_Outputs/ml_inputs/vijayawada_dataset.csv.gz
Output: 02_GIS_Outputs/ml_intermediate/vijayawada_suitability_nodes.csv.gz
        06_Presentation_Figures/model_diagnostics/ultimate_feature_importance.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, roc_auc_score)
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from paths import DATASET_CSV, SUITABILITY_NODES_CSV, FIGURES

print("=" * 60)
print("  PHASE 4: SPATIALLY-AWARE LIGHTGBM FLOOD CLASSIFIER")
print("  Vijayawada Flood Storage Suitability Model")
print("=" * 60)

# ============================================================
# CONFIGURATION — Verify these against your actual dataset
# ============================================================
CONFIG = {
    'file_name':        DATASET_CSV,
    'target':           'fld_majority',
    'urban_lulc_code':  50.0,           # ← VERIFY: Check your LULC legend
    'cell_size_m':      12,          # DEM resolution in meters
    'test_size':        0.30,
    'random_state':     42,
    
    # Suitability index weights (must sum to 1.0)
    'w_flood':          0.50,
    'w_population':     0.30,
    'w_urban':          0.20,
    
    # Elevation valid range (catches -9999 NoData artifacts)
    'elev_min_valid':   -100,
    'elev_max_valid':   3000,
    
    # QGIS grid artifact columns to exclude from ML features
    'grid_artifacts': [
        'row_index', 'col_index', 'id', 'fid', 'objectid',
        'top', 'bottom', 'left', 'right',
        'shape_area', 'shape_length', 'layer', 'path'
    ],
    
    # Output files
    'output_csv':   SUITABILITY_NODES_CSV,
    'output_chart': FIGURES / 'ultimate_feature_importance.png',
}

# ============================================================
# STEP 1: LOAD AND VALIDATE DATA
# ============================================================
print(f"\n[1/9] Loading '{CONFIG['file_name']}'...")
try:
    df = pd.read_csv(CONFIG['file_name'])
except FileNotFoundError:
    raise FileNotFoundError(
        f"Cannot find '{CONFIG['file_name']}'. "
        "Ensure Phase 3 completed and the file is in the same folder."
    )

target = CONFIG['target']
print(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}")

# Validate required columns exist
required_cols = ['X', 'Y', 'elev_mean', 'slp_mean', 'dist_mean',
                 'flow_mean', 'pop_sum', target]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}\n"
        f"Available: {df.columns.tolist()}"
    )

# Fill target and population NaN
df[target]     = df[target].fillna(0).astype(int)
df['pop_sum']  = df['pop_sum'].fillna(0)

# ============================================================
# STEP 2: DATA SANITIZATION (Scientifically Correct)
# ============================================================
print(f"\n[2/9] Sanitizing Data...")

# Elevation: CLIP, never abs() — preserves depressions
# Report suspicious values first
nodata_mask = (
    (df['elev_mean'] < CONFIG['elev_min_valid']) |
    (df['elev_mean'] > CONFIG['elev_max_valid'])
)
if nodata_mask.sum() > 0:
    print(f"  ⚠ {nodata_mask.sum()} rows with elevation outside valid range "
          f"[{CONFIG['elev_min_valid']}, {CONFIG['elev_max_valid']}]m")
    print(f"    These may be NoData artifacts. Clipping to valid range.")
df['elev_mean'] = df['elev_mean'].clip(
    lower=CONFIG['elev_min_valid'],
    upper=CONFIG['elev_max_valid']
)

# Slope, Distance, Flow: physically non-negative, abs() is appropriate
df['slp_mean']  = df['slp_mean'].abs()
df['dist_mean'] = df['dist_mean'].abs()
df['flow_mean'] = df['flow_mean'].abs()

# Drop rainfall if present (too spatially uniform at city scale)
if 'rain_mean' in df.columns:
    df = df.drop(columns=['rain_mean'])
    print("  Dropped 'rain_mean' (spatially uniform at city scale)")

print(f"  Elevation range: {df['elev_mean'].min():.1f}m - "
      f"{df['elev_mean'].max():.1f}m")
print(f"  Slope range:     {df['slp_mean'].min():.2f} - "
      f"{df['slp_mean'].max():.2f}")

# ============================================================
# STEP 3: PHYSICS-BASED FEATURE ENGINEERING
# ============================================================
print(f"\n[3/9] Engineering Physics-Based Features...")

# -- Slope Unit Detection and Conversion --
slope_max = df['slp_mean'].max()
if slope_max > 90:
    print(f"  ⚠ Slope max={slope_max:.1f} → detected as PERCENT slope. Converting.")
    slope_degrees = np.degrees(np.arctan(df['slp_mean'] / 100.0))
else:
    print(f"  ✓ Slope max={slope_max:.1f} → detected as DEGREES slope.")
    slope_degrees = df['slp_mean'].copy()

# -- Topographic Wetness Index (Corrected Formula) --
# TWI = ln(specific_catchment_area / tan(slope))
# specific_catchment_area = flow_accumulation × cell_area (m²)
cell_area_m2 = CONFIG['cell_size_m'] ** 2
specific_catchment_area = (df['flow_mean'] + 1) * cell_area_m2

slope_rad = np.radians(slope_degrees)
tan_slope  = np.tan(slope_rad).clip(lower=0.001)  # Prevent division by zero

df['pseudo_twi'] = np.log(specific_catchment_area / tan_slope)
twi_median = df['pseudo_twi'].replace([np.inf, -np.inf], np.nan).median()
df['pseudo_twi'] = df['pseudo_twi'].replace([np.inf, -np.inf], np.nan).fillna(twi_median)

print(f"  TWI range: {df['pseudo_twi'].min():.2f} - {df['pseudo_twi'].max():.2f}")
if not (2 <= df['pseudo_twi'].mean() <= 30):
    print(f"  ⚠ TWI mean={df['pseudo_twi'].mean():.2f} is outside typical range [2-25]")
    print(f"    Check slope units and flow accumulation scale")

# -- Vulnerability Index --
df['vuln_index'] = 1000.0 / ((df['dist_mean'] + 1) * (df['elev_mean'].clip(lower=0) + 1))

# -- Distance Squared (Non-linear distance effect) --
df['dist_sq'] = df['dist_mean'] ** 2

# -- Elevation × Slope Interaction --
df['elev_slope_interaction'] = df['elev_mean'] * slope_degrees

# Clean up any remaining infinities
df.replace([np.inf, -np.inf], np.nan, inplace=True)
rows_before = len(df)
df = df.dropna().copy()
rows_dropped = rows_before - len(df)
if rows_dropped > 0:
    print(f"  Dropped {rows_dropped} rows with NaN after feature engineering")

print(f"  ✓ Engineered features: pseudo_twi, vuln_index, dist_sq, "
      f"elev_slope_interaction")

# ============================================================
# STEP 4: CATEGORICAL ENCODING
# ============================================================
print(f"\n[4/9] One-Hot Encoding LULC...")
if 'lulc_majority' in df.columns:
    lulc_unique = sorted(df['lulc_majority'].unique())
    print(f"  LULC classes found: {lulc_unique}")
    df = pd.get_dummies(df, columns=['lulc_majority'], prefix='LULC')
    lulc_cols = [c for c in df.columns if c.startswith('LULC_')]
    print(f"  Generated columns: {lulc_cols}")
else:
    print("  ⚠ 'lulc_majority' column not found — skipping encoding")

# ============================================================
# STEP 5: FEATURE SELECTION (Remove Grid Artifacts)
# ============================================================
print(f"\n[5/9] Selecting Features (Removing Grid Artifacts)...")

NON_FEATURES = [target, 'pop_sum'] + CONFIG['grid_artifacts']
features = [col for col in df.columns if col not in NON_FEATURES]

# Report what was excluded
excluded_present = [c for c in CONFIG['grid_artifacts'] if c in df.columns]
if excluded_present:
    print(f"  ✓ Grid artifacts excluded: {excluded_present}")
else:
    print(f"  ℹ No known grid artifacts found in dataset")

print(f"\n  Final rows:          {len(df):,}")
print(f"  Features for ML:     {len(features)}")
print(f"  Feature list:        {features}")

# ============================================================
# STEP 6: CLASS BALANCE CHECK AND MODEL TRAINING
# ============================================================
print(f"\n[6/9] Analyzing Class Balance and Training Model...")

X = df[features]
y = df[target]

# Class balance diagnostics
class_counts = y.value_counts().sort_index()
print(f"\n  Class distribution:")
for cls, cnt in class_counts.items():
    label = "Flood" if cls == 1 else "No Flood"
    print(f"    Class {cls} ({label}): {cnt:,} ({cnt/len(y)*100:.1f}%)")

flood_count    = class_counts.get(1, 0)
no_flood_count = class_counts.get(0, 0)
flood_ratio    = flood_count / len(y)

if flood_ratio < 0.05:
    print(f"\n  🚨 Severe imbalance ({flood_ratio*100:.1f}% flood)")
    print(f"  Applying strong class weight correction")
    scale_pos_weight = no_flood_count / max(flood_count, 1)
elif flood_ratio < 0.20:
    print(f"\n  ⚠ Moderate imbalance ({flood_ratio*100:.1f}% flood)")
    print(f"  Applying moderate class weight correction")
    scale_pos_weight = no_flood_count / max(flood_count, 1)
else:
    print(f"\n  ✓ Reasonably balanced ({flood_ratio*100:.1f}% flood)")
    scale_pos_weight = 1.0

print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state'],
    stratify=y  # Ensures both splits have same flood ratio
)

print(f"\n  Training set: {len(X_train):,} rows")
print(f"  Test set:     {len(X_test):,} rows")
print(f"\n  Training LightGBM (this may take 2-5 minutes)...")

lgbm_model = lgb.LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=15,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,       # Prevents learning from tiny leaf groups
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=0.1,             # L2 regularization
    scale_pos_weight=scale_pos_weight,
    random_state=CONFIG['random_state'],
    n_jobs=-1,
    verbose=-1                  # Suppress per-iteration output
)

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
               lgb.log_evaluation(period=100)]
)

best_iter = lgbm_model.best_iteration_
print(f"  Best iteration: {best_iter} / 1500 "
      f"(early stopping saved {1500 - best_iter} wasted rounds)")

# ============================================================
# STEP 7: EVALUATION
# ============================================================
print(f"\n[7/9] Evaluating Model Performance...")

y_pred      = lgbm_model.predict(X_test)
y_pred_prob = lgbm_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n  Overall Accuracy: {accuracy:.2f}%")

# ROC-AUC (more meaningful than accuracy for imbalanced data)
if len(class_counts) == 2:
    classes = lgbm_model.classes_
    flood_idx = list(classes).index(1) if 1 in classes else None
    if flood_idx is not None:
        roc_auc = roc_auc_score(y_test, y_pred_prob[:, flood_idx])
        print(f"  ROC-AUC Score:    {roc_auc:.4f}")
        if roc_auc > 0.85:
            print(f"  ✓ Excellent discrimination (AUC > 0.85)")
        elif roc_auc > 0.70:
            print(f"  ✓ Good discrimination (AUC > 0.70)")
        else:
            print(f"  ⚠ Poor discrimination (AUC < 0.70) — check features")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['No Flood', 'Flood']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"  Confusion Matrix:")
print(f"    True Neg  (correctly predicted no-flood): {cm[0,0]:,}")
print(f"    False Pos (predicted flood, was no-flood): {cm[0,1]:,}")
print(f"    False Neg (missed actual floods):          {cm[1,0]:,}")
print(f"    True Pos  (correctly predicted flood):     {cm[1,1]:,}")

false_neg_rate = cm[1,0] / max(cm[1,0] + cm[1,1], 1) * 100
if false_neg_rate > 30:
    print(f"\n  ⚠ High Miss Rate: {false_neg_rate:.1f}% of actual floods were missed!")
    print(f"  For safety applications, missed floods are dangerous.")
    print(f"  Consider increasing scale_pos_weight or lowering decision threshold.")

# ============================================================
# STEP 8: SUITABILITY INDEX CALCULATION
# ============================================================
print(f"\n[8/9] Calculating Flood Storage Suitability Index...")

# Safe probability extraction
proba_all = lgbm_model.predict_proba(X)
classes   = lgbm_model.classes_

if 1 not in classes:
    raise ValueError(
        f"Class '1' (flood) not in model classes {classes}. "
        "Check your target column values."
    )

flood_class_idx = list(classes).index(1)
df['hydro_flood_prob'] = proba_all[:, flood_class_idx]

# Population normalization
max_pop = df['pop_sum'].max()
df['pop_normalized'] = (df['pop_sum'] / max_pop).clip(0, 1) if max_pop > 0 else 0.0

# Urban penalty detection
urban_col_name = f"LULC_{CONFIG['urban_lulc_code']}"
if urban_col_name in df.columns:
    urban_penalty = df[urban_col_name].astype(float)
    print(f"  ✓ Urban penalty: '{urban_col_name}' "
          f"({urban_penalty.sum():.0f} urban hexagons, "
          f"{urban_penalty.mean()*100:.1f}%)")
else:
    urban_penalty = pd.Series(0.0, index=df.index)
    print(f"  ⚠ Urban column '{urban_col_name}' not found. "
          f"Urban penalty = 0. Verify CONFIG['urban_lulc_code'].")

# Weighted additive suitability index
W = CONFIG['w_flood'], CONFIG['w_population'], CONFIG['w_urban']
df['storage_suitability'] = (
    df['hydro_flood_prob']      * W[0] +
    (1.0 - df['pop_normalized']) * W[1] +
    (1.0 - urban_penalty)        * W[2]
)

# Validate output
suit_stats = df['storage_suitability'].describe()
print(f"\n  Suitability Index Statistics:")
print(f"    Min:    {suit_stats['min']:.4f}")
print(f"    Max:    {suit_stats['max']:.4f}")
print(f"    Mean:   {suit_stats['mean']:.4f}")
print(f"    Std:    {suit_stats['std']:.4f}")

if suit_stats['max'] < 0.01:
    raise ValueError(
        "🚨 storage_suitability is near-zero everywhere! "
        "Urban penalty or population may be masking all signal."
    )

# Top storage zones preview
top_zones = df.nlargest(10, 'storage_suitability')[
    ['X', 'Y', 'elev_mean', 'hydro_flood_prob', 'storage_suitability', 'pop_sum']
]
print(f"\n  Top 10 Storage Candidate Zones:")
print(top_zones.to_string(index=False))

# ============================================================
# STEP 9: EXPORT AND VISUALIZATION
# ============================================================
print(f"\n[9/9] Exporting Results and Generating Charts...")

# Export suitability nodes
export_cols = ['X', 'Y', 'elev_mean', 'hydro_flood_prob',
               'storage_suitability', 'pop_sum']
df[export_cols].to_csv(CONFIG['output_csv'], index=False)
print(f"  ✓ Saved '{CONFIG['output_csv']}' ({len(df):,} rows)")

# Feature importance chart
importances = lgbm_model.feature_importances_
indices     = np.argsort(importances)[::-1]
feat_names  = [features[i] for i in indices]
feat_imp    = importances[indices]

fig, ax = plt.subplots(figsize=(14, 8))
colors = ['#d62728' if 'X' == f or 'Y' == f else '#2ca02c' for f in feat_names]
bars = ax.bar(range(len(feat_names)), feat_imp, align='center', color=colors)

ax.set_title("LightGBM Feature Importance\n"
             "(Red = Spatial Coords | Green = Physical Features)",
             fontsize=13, fontweight='bold')
ax.set_xticks(range(len(feat_names)))
ax.set_xticklabels(feat_names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Split Importance Score")
ax.set_xlim([-1, len(feat_names)])

# Annotate top 3 features
for rank in range(min(3, len(feat_names))):
    ax.annotate(f'#{rank+1}',
                xy=(rank, feat_imp[rank]),
                ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='navy')

plt.tight_layout()
plt.savefig(CONFIG['output_chart'], dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved '{CONFIG['output_chart']}'")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  ✅ PHASE 4 COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Nodes processed:      {len(df):,}")
print(f"  Features trained on:  {len(features)}")
print(f"  Model accuracy:       {accuracy:.2f}%")
if 'roc_auc' in dir():
    print(f"  ROC-AUC:              {roc_auc:.4f}")
print(f"  Best iteration:       {best_iter} / 1500")
print(f"  Scale pos weight:     {scale_pos_weight:.2f}")
print()
print(f"  📂 Output Files:")
print(f"    → {CONFIG['output_csv']}")
print(f"    → {CONFIG['output_chart']}")
print()
print(f"  📌 Next Step: Run phase4b_elevation_correction.py, then phase5_flood_routing.py")
print("=" * 60)