# Final System Outputs

All coordinates are **EPSG:32644 (WGS 84 / UTM zone 44N)**, in metres.

| File | Rows | Description |
|---|---|---|
| `identified_storage_sinks.csv` | 50 | Selected temporary storage zones: `X, Y, elev_mean, storage_suitability, pop_sum` |
| `optimized_flood_routes.geojson` | 50 features | LineStrings from the Prakasam Barrage source to each sink, with `routing_cost`, `n_hexagons`, `elev_drop_m`, `mean_pop_crossed`, `max_pop_crossed`, `mean_suitability`, `sink_*` attributes |
| `vijayawada_suitability_nodes_corrected.csv` | 342,573 | Hexagon centroids after vertical-datum correction: `X, Y, elev_mean, hydro_flood_prob, storage_suitability, pop_sum` |

**Load in QGIS:** add the GeoJSON directly (CRS EPSG:32644). Add CSVs via *Layer → Add Delimited Text Layer* with X = `X`, Y = `Y`, CRS EPSG:32644.

Produced by `03_Source_Code/1_ML_and_Routing_Pipeline/phase4b_elevation_correction.py` and `phase5_flood_routing.py`.
