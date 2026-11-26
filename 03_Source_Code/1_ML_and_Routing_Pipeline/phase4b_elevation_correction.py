"""
Phase 4b - CartoDEM vertical-datum correction (production step).

The CartoDEM mosaic reports the Krishna delta floodplain at negative heights
(hexagon median about -55 m) while hill tiles are on a different vertical
reference. Routing on these values would make the river appear far higher
than the delta. This step keeps only the flat-delta population (elev < 0),
shifts it by +69.7 m, and drops any node still negative afterwards.

Result: 342,573 nodes, median elevation about 15.6 m MSL.
See dem_benchmark_diagnostics.py for the benchmark analysis behind it.

Input : 02_GIS_Outputs/ml_intermediate/vijayawada_suitability_nodes.csv.gz
Output: 04_Final_System_Outputs/vijayawada_suitability_nodes_corrected.csv
"""
import pandas as pd
import numpy as np

from paths import SUITABILITY_NODES_CSV, CORRECTED_NODES_CSV

print("=" * 60)
print(" TWO-POPULATION ELEVATION CORRECTION")
print("=" * 60)

# 1. Load the data
df = pd.read_csv(SUITABILITY_NODES_CSV)

# 2. Split flat delta (negative) from hills (positive)
neg_mask = df['elev_mean'] < 0
pos_mask = df['elev_mean'] >= 0

print(f"Population A (flat delta): {neg_mask.sum():,} nodes")
print(f"Population B (hills): {pos_mask.sum():,} nodes")

# 3. Apply the correction to the flat delta ONLY
FLAT_CORRECTION = +69.7
df_option1 = df[neg_mask].copy()
df_option1['elev_mean'] = df_option1['elev_mean'] + FLAT_CORRECTION

# 4. Clip out any remaining microscopic negative artifacts
before_clip = len(df_option1)
df_option1 = df_option1[df_option1['elev_mean'] >= 0].copy()
clipped = before_clip - len(df_option1)

print(f"\nNodes after +{FLAT_CORRECTION}m correction: {len(df_option1):,}")
print(f"Still-negative after correction (clipped out): {clipped:,}")
print(f"Final elevation range: {df_option1['elev_mean'].min():.1f}m - {df_option1['elev_mean'].max():.1f}m")
print(f"Final elevation median: {df_option1['elev_mean'].median():.1f}m (Krishna delta floodplain is ~10-25 m MSL)")

# 5. Save the final corrected dataset
out_file = CORRECTED_NODES_CSV
df_option1.to_csv(out_file, index=False)
print(f"\n✅ Saved '{out_file}' ({len(df_option1):,} nodes)")