# Raster and Vector Data Analyses: Visual Results

What each step of the spatial pipeline produced, in the order it was run.

## 01_Raster_Analyses: hydrological parameters

| | |
|---|---|
| ![](01_Raster_Analyses/01_Elevation_CartoDEM.png) | **CartoDEM elevation (raw).** Note the delta floodplain at negative heights. This is the vertical-datum problem fixed in Phase 4b. |
| ![](01_Raster_Analyses/02_Terrain_Slope.png) | **Slope** derived from CartoDEM. The delta is almost uniformly flat (< 1°). |
| ![](01_Raster_Analyses/03_Flow_Accumulation.png) | **Flow accumulation** (GRASS `r.watershed`). Used for the Topographic Wetness Index. |
| ![](01_Raster_Analyses/04_Distance_to_Water.png) | **Distance to water** (QGIS Proximity on rivers/waterways). |

## 02_Vector_Analyses: spatial discretisation

| | |
|---|---|
| ![](02_Vector_Analyses/06_Infrastructure_Clip.png) | OSM roads, buildings and POIs clipped to the study boundary. |
| ![](02_Vector_Analyses/07_Hexagonal_Grid_Generation.png) | The 393,794-cell, 100 m hexagonal grid (`city_grid.shp`). |
| ![](02_Vector_Analyses/08_Zonal_Statistics_Attribute_Table.png) | Zonal-statistics attribute table, the bridge from GIS to the ML dataset. |

## 03_Combined_Analytical_Results: synthesis

| | |
|---|---|
| ![](03_Combined_Analytical_Results/09_ML_Suitability_Heatmap.png) | LightGBM-based storage-suitability heatmap. |
| ![](03_Combined_Analytical_Results/10_Top_50_Storage_Sinks.png) | Top 50 uninhabited storage sinks. |
| ![](03_Combined_Analytical_Results/11_Optimized_Flood_Routes.png) | 50 least-cost routes from the Prakasam Barrage (Dijkstra). |

> Raster panels 01 and 02 are rendered previews of the original GeoTIFFs (2–98th percentile stretch). The source rasters are not committed because of their size.
