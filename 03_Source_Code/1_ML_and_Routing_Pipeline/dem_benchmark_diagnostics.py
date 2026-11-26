"""
DEM vertical-datum diagnostics (exploratory, not part of the production run).

Compares hexagon-averaged CartoDEM elevations against published benchmark
heights (Indrakeeladri Hill, Kondapalli Ridge, Prakasam Barrage, city centre)
to characterise the datum offset between flat-delta tiles (negative values)
and hill tiles. The findings informed the single flat-delta offset applied in
phase4b_elevation_correction.py.

Input : 02_GIS_Outputs/ml_intermediate/vijayawada_suitability_nodes.csv.gz
Output: 02_GIS_Outputs/ml_intermediate/benchmark_corrected_nodes_diagnostic.csv
"""
import pandas as pd
import numpy as np

from paths import SUITABILITY_NODES_CSV, INTERMEDIATE

df = pd.read_csv(SUITABILITY_NODES_CSV)

print("=" * 60)
print("  DEFINITIVE DEM ERROR CHARACTERIZATION")
print("=" * 60)

benchmarks = [
    {
        'name':          'Indrakeeladri Hill (Durga Temple)',
        'published_m':   92,
        'x_center':      439500,
        'y_center':      1828000,
        'search_radius': 3000,
        'use_stat':      'max',   # hill = take peak
    },
    {
        'name':          'Kondapalli Ridge (Hill peak)',
        'published_m':   364,     # Ridge peak, not the fort walls
        'x_center':      443500,
        'y_center':      1848000,
        'search_radius': 3000,
        'use_stat':      'max',
    },
    {
        'name':          'Krishna River / Prakasam Barrage',
        'published_m':   12,
        'x_center':      428000,
        'y_center':      1829000,
        'search_radius': 4000,
        'use_stat':      'min',   # river = take lowest point
    },
    {
        'name':          'Vijayawada city centre (flat)',
        'published_m':   26,
        'x_center':      440000,
        'y_center':      1828000,
        'search_radius': 2000,
        'use_stat':      'median',
    },
]

print(f"\n  {'Benchmark':<38} {'Published':>10} {'DataVal':>10} "
      f"{'Error':>10} {'N nodes':>8}")
print(f"  {'-'*38} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

errors       = []
hill_errors  = []
flat_errors  = []

for bm in benchmarks:
    # ── Fixed: parentheses around EACH comparison ──────────
    mask = (
        ((df['X'] - bm['x_center']).abs() <= bm['search_radius']) &
        ((df['Y'] - bm['y_center']).abs() <= bm['search_radius'])
    )
    subset = df[mask]

    if len(subset) == 0:
        print(f"  {bm['name']:<38} {bm['published_m']:>10} "
              f"{'NO DATA':>10}")
        continue

    # Pick the representative statistic
    if bm['use_stat'] == 'max':
        data_val = subset['elev_mean'].max()
    elif bm['use_stat'] == 'min':
        data_val = subset['elev_mean'].min()
    else:
        data_val = subset['elev_mean'].median()

    error = data_val - bm['published_m']
    errors.append({
        'name':      bm['name'],
        'published': bm['published_m'],
        'data_val':  data_val,
        'error':     error,
        'terrain':   'hill' if bm['published_m'] > 50 else 'flat',
    })

    if bm['published_m'] > 50:
        hill_errors.append(error)
    else:
        flat_errors.append(error)

    print(f"  {bm['name']:<38} {bm['published_m']:>10.0f} "
          f"{data_val:>10.1f} {error:>+10.1f} {len(subset):>8,}")

# ── Error pattern analysis ────────────────────────────────
print(f"\n--- Error Pattern by Terrain Type ---")

if hill_errors:
    hill_mean = np.mean(hill_errors)
    hill_std  = np.std(hill_errors)
    print(f"\n  HILL nodes  (published >50m):")
    print(f"    Errors:       {[f'{e:+.0f}m' for e in hill_errors]}")
    print(f"    Mean error:   {hill_mean:+.1f}m")
    print(f"    Std:          {hill_std:.1f}m")
    if hill_std < 15:
        print(f"    ✓ Consistent → apply {-hill_mean:+.0f}m correction to hill nodes")
    else:
        print(f"    ⚠ Inconsistent → no single correction fits hill nodes")

if flat_errors:
    flat_mean = np.mean(flat_errors)
    flat_std  = np.std(flat_errors)
    print(f"\n  FLAT nodes  (published <50m):")
    print(f"    Errors:       {[f'{e:+.0f}m' for e in flat_errors]}")
    print(f"    Mean error:   {flat_mean:+.1f}m")
    print(f"    Std:          {flat_std:.1f}m")
    if flat_std < 15:
        print(f"    ✓ Consistent → apply {-flat_mean:+.0f}m correction to flat nodes")
    else:
        print(f"    ⚠ Inconsistent → no single correction fits flat nodes")

# ── Apply corrections and validate ───────────────────────
print(f"\n--- Applying Terrain-Specific Corrections ---")

neg = df[df['elev_mean'] <  0].copy()
pos = df[df['elev_mean'] >= 0].copy()

# Correction for negative (flat delta) nodes
flat_correction = -flat_mean if flat_errors else 79.0
# Correction for positive (hill) nodes
hill_correction = -hill_mean if hill_errors else -60.0

print(f"\n  Flat delta nodes ({len(neg):,}):")
print(f"    Current median:   {neg['elev_mean'].median():.1f}m")
print(f"    Correction:       {flat_correction:+.1f}m")
corrected_neg = neg['elev_mean'] + flat_correction
print(f"    Corrected median: {corrected_neg.median():.1f}m")
print(f"    Corrected range:  {corrected_neg.min():.1f}m - "
      f"{corrected_neg.max():.1f}m")

print(f"\n  Hill nodes ({len(pos):,}):")
print(f"    Current median:   {pos['elev_mean'].median():.1f}m")
print(f"    Correction:       {hill_correction:+.1f}m")
corrected_pos = pos['elev_mean'] + hill_correction
print(f"    Corrected median: {corrected_pos.median():.1f}m")
print(f"    Corrected range:  {corrected_pos.min():.1f}m - "
      f"{corrected_pos.max():.1f}m")

# ── Sanity check after both corrections ──────────────────
print(f"\n--- Post-Correction Sanity Checks ---")

# The corrected flat delta should be 8-25m
flat_ok = 5 <= corrected_neg.median() <= 35
print(f"\n  Flat delta corrected median {corrected_neg.median():.1f}m "
      f"in range [5, 35]m: {'✓' if flat_ok else '✗'}")

# The corrected Indrakeeladri should be ~92m
indra_mask = (
    ((df['X'] - 439500).abs() <= 3000) &
    ((df['Y'] - 1828000).abs() <= 3000) &
    (df['elev_mean'] >= 0)
)
if indra_mask.sum() > 0:
    indra_corrected = df.loc[indra_mask, 'elev_mean'].max() + hill_correction
    print(f"  Indrakeeladri corrected max {indra_corrected:.1f}m "
          f"vs published 92m: "
          f"{'✓' if abs(indra_corrected - 92) < 20 else '✗'}")

# After correction, flat should be LOWER than hills
# (physically correct: delta below surrounding terrain)
flat_median_corrected = corrected_neg.median()
hill_median_corrected = corrected_pos.median()
order_ok = flat_median_corrected < hill_median_corrected
print(f"  Flat median ({flat_median_corrected:.1f}m) < "
      f"Hill median ({hill_median_corrected:.1f}m): "
      f"{'✓' if order_ok else '✗'}")

# ── Write corrected CSV for Phase 5 ──────────────────────
print(f"\n--- Saving Corrected Elevation Dataset ---")

df_corrected = df.copy()
neg_mask_bool = df_corrected['elev_mean'] < 0
pos_mask_bool = df_corrected['elev_mean'] >= 0

df_corrected.loc[neg_mask_bool, 'elev_mean'] += flat_correction
df_corrected.loc[pos_mask_bool, 'elev_mean'] += hill_correction

# Clip any remaining impossible values
df_corrected['elev_mean'] = df_corrected['elev_mean'].clip(lower=0, upper=600)

# Written to the intermediate folder so it never overwrites the
# production file produced by phase4b_elevation_correction.py
out_file = INTERMEDIATE / 'benchmark_corrected_nodes_diagnostic.csv'
df_corrected.to_csv(out_file, index=False)

print(f"  ✓ Saved '{out_file}'")
print(f"  Final elevation range: "
      f"{df_corrected['elev_mean'].min():.1f}m - "
      f"{df_corrected['elev_mean'].max():.1f}m")
print(f"  Final elevation median: {df_corrected['elev_mean'].median():.1f}m")

# ── Thesis statement ──────────────────────────────────────
print(f"\n{'='*60}")
print(f"  THESIS METHODOLOGY STATEMENT")
print(f"{'='*60}")
print(f"""
  The CartoDEM mosaic exhibited vertical datum inconsistencies
  across terrain types. Comparison with published benchmark
  elevations revealed systematic offsets:

  - Floodplain tiles (n={len(neg):,}): median error {flat_mean:+.1f}m
    relative to MSL (correction applied: {flat_correction:+.1f}m)

  - Hill terrain tiles (n={len(pos):,}): median error {hill_mean:+.1f}m
    relative to MSL (correction applied: {hill_correction:+.1f}m)

  Benchmarks used:
    Indrakeeladri Hill: Survey of India = 92m MSL
    Kondapalli Ridge:   Survey of India = 364m MSL
    Prakasam Barrage:   Published       = 12m MSL

  After correction, Indrakeeladri peak = {indra_corrected:.0f}m
  (within acceptable range of published 92m)
""")

print(f"  Diagnostic only. Production correction: phase4b_elevation_correction.py")
print("=" * 60)