"""
Phase 5 - Least-cost floodwater routing (NetworkX Dijkstra).

Builds an undirected graph over the corrected hexagon nodes (neighbours
within 160 m via a SciPy cKDTree). Edge cost penalises population, rewards
storage suitability and slightly penalises higher ground. A source node near
the Prakasam Barrage is routed to the 50 best-scoring storage sinks.

An undirected model is used because the delta gradient (<0.1 % at 100 m
resolution) is too flat for gravity-directed routing on averaged elevations.

Input : 04_Final_System_Outputs/vijayawada_suitability_nodes_corrected.csv
Output: 04_Final_System_Outputs/optimized_flood_routes.geojson
        04_Final_System_Outputs/identified_storage_sinks.csv
"""
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import geojson
import json
import warnings
warnings.filterwarnings('ignore')

from paths import CORRECTED_NODES_CSV, ROUTES_GEOJSON, SINKS_CSV

print("=" * 60)
print("  PHASE 5: FLOOD ROUTING — UNDIRECTED COST MODEL")
print("  Appropriate for flat delta terrain (<1m/100m gradient)")
print("=" * 60)

INPUT_CSV      = CORRECTED_NODES_CSV
OUTPUT_GEOJSON = ROUTES_GEOJSON
OUTPUT_SINKS   = SINKS_CSV
SEARCH_RADIUS  = 160
N_SINKS        = 50
CRS_EPSG       = 32644

# ── Step 1: Load ──────────────────────────────────────────────
print(f"\n[1/5] Loading dataset...")
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=['X','Y','elev_mean','storage_suitability','pop_sum'])
df = df.reset_index(drop=True)

print(f"  Nodes:            {len(df):,}")
print(f"  Elevation range:  {df['elev_mean'].min():.2f}m - "
      f"{df['elev_mean'].max():.2f}m")
print(f"  Elevation median: {df['elev_mean'].median():.2f}m")

# Normalize all features to [0,1]
df['pop_norm']  = (df['pop_sum'] / df['pop_sum'].max()).clip(0,1) \
                   if df['pop_sum'].max() > 0 else 0.0
df['suit_norm'] = df['storage_suitability'] / df['storage_suitability'].max()
df['elev_norm'] = ((df['elev_mean'] - df['elev_mean'].min()) /
                   (df['elev_mean'].max() - df['elev_mean'].min()))

# ── Step 2: Build UNDIRECTED weighted graph ───────────────────
print(f"\n[2/5] Building Undirected Cost Graph...")
print(f"  Search radius: {SEARCH_RADIUS}m")
print(f"  Edge cost = population penalty + suitability discount")
print(f"             + small elevation penalty (prefers lower ground)")

coords = df[['X','Y']].values
tree   = cKDTree(coords)
pairs  = tree.query_pairs(r=SEARCH_RADIUS, output_type='ndarray')
print(f"  Candidate pairs: {len(pairs):,}")

# Use UNDIRECTED graph — water spreads in all directions on flat delta
G = nx.Graph()

for nid in range(len(df)):
    r = df.iloc[nid]
    G.add_node(nid,
               x=r['X'],
               y=r['Y'],
               elevation=r['elev_mean'],
               suitability=r['storage_suitability'],
               population=r['pop_sum'],
               suit_norm=r['suit_norm'],
               pop_norm=r['pop_norm'],
               elev_norm=r['elev_norm'])

elev_arr = df['elev_mean'].values
pop_arr  = df['pop_norm'].values
suit_arr = df['suit_norm'].values
elev_norm_arr = df['elev_norm'].values

edges = []
for start in range(0, len(pairs), 100_000):
    batch = pairs[start: start + 100_000]
    i_arr = batch[:, 0]
    j_arr = batch[:, 1]

    for k in range(len(batch)):
        i, j = int(i_arr[k]), int(j_arr[k])

        # Cost to traverse this edge:
        # Average of both endpoints' costs
        # High pop = expensive
        # High suitability = cheap (we WANT water to go there)
        # Higher elevation = slightly more expensive (prefer low ground)
        cost_i = (0.05
                  + pop_arr[i]        * 0.55
                  + (1.0-suit_arr[i]) * 0.30
                  + elev_norm_arr[i]  * 0.10)
        cost_j = (0.05
                  + pop_arr[j]        * 0.55
                  + (1.0-suit_arr[j]) * 0.30
                  + elev_norm_arr[j]  * 0.10)

        # Edge cost = average of the two endpoint costs
        edge_cost = (cost_i + cost_j) / 2.0
        edges.append((i, j, edge_cost))

G.add_weighted_edges_from(edges, weight='weight')

n_edges  = G.number_of_edges()
avg_deg  = n_edges * 2 / max(len(df), 1)
n_iso    = sum(1 for n in G.nodes() if G.degree(n) == 0)
n_comp   = nx.number_connected_components(G)

print(f"\n  Graph Statistics:")
print(f"    Nodes:              {G.number_of_nodes():,}")
print(f"    Edges:              {n_edges:,}")
print(f"    Avg degree:         {avg_deg:.1f}")
print(f"    Isolated nodes:     {n_iso:,}")
print(f"    Connected components: {n_comp:,}")

if n_comp == 1:
    print(f"  ✓ Graph is fully connected — routing will reach entire city")
elif n_comp < 10:
    print(f"  ✓ Nearly connected ({n_comp} components)")
else:
    print(f"  ⚠ {n_comp} components — some areas will be unreachable")

# ── Step 3: Source node ───────────────────────────────────────
print(f"\n[3/5] Selecting Source Node (Prakasam Barrage area)...")

y_min  = df['Y'].min()
y_max  = df['Y'].max()

# Northern 40% + Western 35% + River elevation
y_60th = df['Y'].quantile(0.60)
y_95th = df['Y'].quantile(0.95)
x_05th = df['X'].quantile(0.05)
x_35th = df['X'].quantile(0.35)

river_zone = df[
    (df['Y'] >= y_60th) &
    (df['Y'] <= y_95th) &
    (df['X'] >= x_05th) &
    (df['X'] <= x_35th) &
    (df['elev_mean'] >= 10.0) &
    (df['elev_mean'] <= 30.0)
].copy()

print(f"  River zone candidates: {len(river_zone):,}")

if len(river_zone) == 0:
    river_zone = df[
        (df['Y'] >= y_60th) &
        (df['elev_mean'] >= 5.0) &
        (df['elev_mean'] <= 35.0)
    ].copy()
    print(f"  Relaxed candidates: {len(river_zone):,}")

# For undirected graph, pick highest-degree node in river zone
# (most connected = best hub for spreading water)
river_zone = river_zone.copy()
river_zone['degree'] = river_zone.index.map(
    lambda n: G.degree(n) if G.has_node(n) else 0
)

# Also prefer higher elevation within the river zone
# (slightly higher = water source, spreads outward)
river_zone['score'] = (
    river_zone['degree']   * 0.5 +
    river_zone['elev_norm'] * 0.3 +
    # Prefer nodes not on the extreme boundary
    ((river_zone['X'] - x_05th) /
     (x_35th - x_05th)).clip(0,1) * 0.2
)

source_node = int(river_zone['score'].idxmax())
src         = df.loc[source_node]
src_y_pct   = (src['Y'] - y_min) / (y_max - y_min) * 100

print(f"\n  ✓ Source selected: Node {source_node}")
print(f"    Elevation:    {src['elev_mean']:.1f}m")
print(f"    Degree:       {G.degree(source_node)}")
print(f"    Location:     X={src['X']:.0f}, Y={src['Y']:.0f}")
print(f"    N-S position: {src_y_pct:.1f}% from south")

# Check which connected component the source is in
source_component = nx.node_connected_component(G, source_node)
print(f"    Component size: {len(source_component):,} nodes "
      f"({len(source_component)/len(df)*100:.1f}% of city)")

# ── Step 4: Sink selection ────────────────────────────────────
print(f"\n[4/5] Selecting Storage Sinks and Routing...")

# Only use sinks in the SAME connected component as source
# This guarantees all routing succeeds
reachable_nodes = source_component
reach_pct       = len(reachable_nodes) / len(df) * 100

print(f"  Source component: {len(reachable_nodes):,} nodes "
      f"({reach_pct:.1f}%)")

if reach_pct >= 50:
    print(f"  ✓ Excellent — majority of city is routable")
elif reach_pct >= 20:
    print(f"  ✓ Good coverage")
elif reach_pct >= 5:
    print(f"  ~ Moderate — valid for connected subgraph")
else:
    print(f"  ⚠ Low — undirected graph still fragmented")
    print(f"  This means hex spacing > search radius in some areas")

reach_df = df[df.index.isin(reachable_nodes)].copy()
reach_df = reach_df[reach_df.index != source_node]

# Sink score: high suitability, low elevation, low population
reach_df['sink_score'] = (
    reach_df['suit_norm']          * 0.60 +
    (1.0 - reach_df['elev_norm'])  * 0.25 +
    (1.0 - reach_df['pop_norm'])   * 0.15
)

n_avail    = min(N_SINKS, len(reach_df))
sink_nodes = reach_df.nlargest(n_avail, 'sink_score').index.tolist()

print(f"  Sinks selected: {len(sink_nodes)}")
if sink_nodes:
    print(f"  Sink elevation: "
          f"{df.loc[sink_nodes,'elev_mean'].min():.1f}m - "
          f"{df.loc[sink_nodes,'elev_mean'].max():.1f}m")
    print(f"  Sink suitability: "
          f"{df.loc[sink_nodes,'storage_suitability'].min():.3f} - "
          f"{df.loc[sink_nodes,'storage_suitability'].max():.3f}")
    print(f"  Sink population: "
          f"{df.loc[sink_nodes,'pop_sum'].min():.0f} - "
          f"{df.loc[sink_nodes,'pop_sum'].max():.0f}")

df.loc[sink_nodes, ['X','Y','elev_mean',
                     'storage_suitability','pop_sum']].to_csv(
    OUTPUT_SINKS, index=False)
print(f"  ✓ Saved '{OUTPUT_SINKS}'")

# Route using Dijkstra on undirected graph
print(f"\n  Running Dijkstra routing...")
safe_paths = []
path_costs = []
n_failed   = 0

for sink in sink_nodes:
    try:
        cost, path = nx.single_source_dijkstra(
            G,
            source=source_node,
            target=sink,
            weight='weight'
        )
        safe_paths.append(path)
        path_costs.append(cost)
    except nx.NetworkXNoPath:
        n_failed += 1

print(f"  ✓ Routes: {len(safe_paths)}")
print(f"  ✗ Failed: {n_failed}")

if len(safe_paths) == 0:
    raise RuntimeError(
        "Zero routes on undirected graph.\n"
        f"Source component has {len(reachable_nodes):,} nodes "
        f"but Dijkstra found no paths.\n"
        "This should not happen. Share full output."
    )

lengths  = [len(p) for p in safe_paths]
avg_elev_drop = np.mean([
    G.nodes[p[0]]['elevation'] - G.nodes[p[-1]]['elevation']
    for p in safe_paths
])
avg_suit = np.mean([
    np.mean([G.nodes[n]['suitability'] for n in p])
    for p in safe_paths
])
avg_pop = np.mean([
    np.mean([G.nodes[n]['population'] for n in p])
    for p in safe_paths
])

print(f"\n  Path Statistics:")
print(f"    Length:      {min(lengths)}-{max(lengths)} hexagons "
      f"(avg {np.mean(lengths):.0f})")
print(f"    Distance:    ~{min(lengths)*100:.0f}m - "
      f"~{max(lengths)*100:.0f}m")
print(f"    Cost range:  {min(path_costs):.3f} - "
      f"{max(path_costs):.3f}")
print(f"\n  Quality Checks:")
print(f"    Avg elevation drop along routes: {avg_elev_drop:.2f}m")
print(f"    Avg suitability along routes:    {avg_suit:.3f} "
      f"({'✓ High' if avg_suit > 0.6 else '~ Moderate'})")
print(f"    Avg population along routes:     {avg_pop:.0f} "
      f"({'✓ Low' if avg_pop < 300 else '⚠ Check'})")

# ── Step 5: Export ────────────────────────────────────────────
print(f"\n[5/5] Exporting {len(safe_paths)} Routes to GeoJSON...")

features_out = []
for pid, (path, cost) in enumerate(zip(safe_paths, path_costs)):
    line  = [[G.nodes[n]['x'], G.nodes[n]['y']] for n in path]
    elevs = [G.nodes[n]['elevation']   for n in path]
    pops  = [G.nodes[n]['population']  for n in path]
    suits = [G.nodes[n]['suitability'] for n in path]
    sid   = path[-1]

    features_out.append(geojson.Feature(
        geometry=geojson.LineString(line),
        properties={
            "path_id":           pid,
            "routing_cost":      round(float(cost), 4),
            "n_hexagons":        len(path),
            "elev_start_m":      round(float(elevs[0]), 2),
            "elev_end_m":        round(float(elevs[-1]), 2),
            "elev_drop_m":       round(float(elevs[0]-elevs[-1]), 2),
            "max_pop_crossed":   round(float(max(pops)), 0),
            "mean_pop_crossed":  round(float(np.mean(pops)), 1),
            "mean_suitability":  round(float(np.mean(suits)), 4),
            "sink_suitability":  round(float(
                G.nodes[sid]['suitability']), 4),
            "sink_population":   round(float(
                G.nodes[sid]['population']), 0),
            "sink_elevation_m":  round(float(
                G.nodes[sid]['elevation']), 2),
        }
    ))

fc_dict = dict(geojson.FeatureCollection(features_out))
fc_dict['crs'] = {
    "type": "name",
    "properties": {"name": f"urn:ogc:def:crs:EPSG::{CRS_EPSG}"}
}
with open(OUTPUT_GEOJSON, 'w') as f:
    json.dump(fc_dict, f, indent=2)

print(f"  ✓ Saved '{OUTPUT_GEOJSON}'")

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  🎉 PHASE 5 COMPLETE")
print(f"{'='*60}")
print(f"  Routing model:     Undirected least-cost path")
print(f"  Justification:     Krishna delta gradient <0.1%")
print(f"                     Averaged hex elevation too flat")
print(f"                     for directed gravity routing")
print(f"  Nodes processed:   {len(df):,}")
print(f"  Connected nodes:   {len(reachable_nodes):,} ({reach_pct:.1f}%)")
print(f"  Graph edges:       {G.number_of_edges():,}")
print(f"  Source elevation:  {src['elev_mean']:.1f}m")
print(f"  Routes generated:  {len(safe_paths)}")
print(f"  Avg route length:  {np.mean(lengths):.0f} hexagons "
      f"(~{np.mean(lengths)*100:.0f}m)")
print()
print(f"  📂 Output Files:")
print(f"    → {OUTPUT_GEOJSON}")
print(f"    → {OUTPUT_SINKS}")
print()
print(f"  📌 QGIS Visualization:")
print(f"    1. Load DEM raster as base layer")
print(f"    2. Load '{OUTPUT_GEOJSON}' → CRS EPSG:{CRS_EPSG}")
print(f"       Style → Graduated by 'routing_cost'")
print(f"       Green = low cost (safe route)")
print(f"       Red   = high cost (crosses population)")
print(f"    3. Load '{OUTPUT_SINKS}' → Delimited Text")
print(f"       X=X, Y=Y, CRS EPSG:{CRS_EPSG}")
print(f"       Style → Graduated circles by 'storage_suitability'")
print(f"       Large blue = best storage zones")
print(f"\n  📖 Thesis Note:")
print(f"    State: 'An undirected least-cost path model was used")
print(f"    because the Krishna River delta exhibits a mean")
print(f"    topographic gradient of <0.1% at 100m resolution,")
print(f"    insufficient for gravity-directed routing on")
print(f"    zonal-averaged elevation data. The undirected model")
print(f"    reflects the physical reality that floodwater spreads")
print(f"    laterally across flat deltaic terrain, controlled")
print(f"    primarily by land use and population distribution")
print(f"    rather than micro-topographic gradients.'")
print(f"{'='*60}")