"""
Quick sanity check of the Phase 5 outputs: prints per-route statistics,
the geographic extent of the routes, and the top storage sinks.
"""
import json
import numpy as np
import pandas as pd

from paths import ROUTES_GEOJSON, SINKS_CSV

with open(ROUTES_GEOJSON) as f:
    data = json.load(f)

features = data['features']
print(f"Routes in file: {len(features)}")
print(f"\nRoute property summary:")
print(f"{'ID':>4} {'Hexagons':>9} {'Cost':>8} "
      f"{'ElevDrop':>9} {'SinkElev':>9} {'SinkSuit':>9}")
print("-" * 55)

for feat in features[:10]:
    p = feat['properties']
    print(f"{p['path_id']:>4} "
          f"{p['n_hexagons']:>9} "
          f"{p['routing_cost']:>8.3f} "
          f"{p['elev_drop_m']:>9.2f} "
          f"{p['sink_elevation_m']:>9.2f} "
          f"{p['sink_suitability']:>9.3f}")

# Check if routes are geographically reasonable
print(f"\nGeographic extent of routes:")
all_coords = []
for feat in features:
    coords = feat['geometry']['coordinates']
    all_coords.extend(coords)

xs = [c[0] for c in all_coords]
ys = [c[1] for c in all_coords]
print(f"  X range: {min(xs):.0f} - {max(xs):.0f} "
      f"(span: {max(xs)-min(xs):.0f}m)")
print(f"  Y range: {min(ys):.0f} - {max(ys):.0f} "
      f"(span: {max(ys)-min(ys):.0f}m)")

df_sinks = pd.read_csv(SINKS_CSV)
print(f"\nSink locations:")
print(f"  X range: {df_sinks['X'].min():.0f} - {df_sinks['X'].max():.0f}")
print(f"  Y range: {df_sinks['Y'].min():.0f} - {df_sinks['Y'].max():.0f}")
print(f"  Elevation: {df_sinks['elev_mean'].min():.2f}m - "
      f"{df_sinks['elev_mean'].max():.2f}m")
print(f"  Suitability: {df_sinks['storage_suitability'].min():.3f} - "
      f"{df_sinks['storage_suitability'].max():.3f}")
print(f"\nTop 5 sinks by suitability:")
print(df_sinks.nlargest(5, 'storage_suitability')[
    ['X', 'Y', 'elev_mean', 'storage_suitability', 'pop_sum']
].to_string(index=False))