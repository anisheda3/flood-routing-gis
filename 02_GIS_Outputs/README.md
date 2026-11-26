# GIS Outputs

All layers are in **EPSG:32644 (UTM 44N)**, aligned to the 10 m CartoDEM grid.

## Included

| Path | Contents |
|---|---|
| `ml_inputs/vijayawada_dataset.csv.gz` | Phase 3 output: 393,794 hexagons × zonal statistics (`elev_mean, slp_mean, dist_mean, flow_mean, rain_mean, lulc_majority, pop_sum, fld_majority`). Input to Phase 4. |
| `ml_intermediate/vijayawada_suitability_nodes.csv.gz` | Phase 4 output before DEM correction: flood probability and suitability per hexagon. |
| `vector/vijayawada_boundary.*` | Study-area boundary |
| `vector/rivers_final.*`, `vector/rivers_utm.*` | Rivers (HydroRIVERS) |
| `vector/waterways_utm.*`, `vector/waterbodies_utm.*` | OSM waterways and waterbodies |
| `vector/pois_utm.*` | OSM points of interest |

The CSVs are gzipped to stay under GitHub size limits. pandas reads them directly (`pd.read_csv("...csv.gz")`).

## Not included (regenerable, ~14 GB)

| Layer | Size | How it was produced |
|---|---|---|
| CartoDEM / SRTM / ALOS PALSAR clips | 10–57 MB each | Clip to boundary, reproject to UTM 44N |
| Slope, flow accumulation, distance to water | 20–180 MB | QGIS Slope, GRASS `r.watershed`, QGIS Proximity |
| LULC, flood 2020, population (aligned) | 45–178 MB | Warp/resample to the 10 m CartoDEM grid |
| Population raster in UTM (`pop_utm.tif`) | 5.1 GB | WorldPop 2020 reprojected |
| OSM buildings and roads | 10–68 MB | Clipped OSM extracts |
| `city_grid.shp` / `raw_grid.shp` (hex grid) | 60–90 MB | QGIS *Create Grid* (hexagon, 100 m) |
| GPM IMERG rainfall | 110 MB | NetCDF band 250 (Sept 2020) extraction |
