"""Shared, repo-relative file locations for the ML + routing pipeline.

Every script imports from here, so the pipeline runs from any working
directory without editing hard-coded file names.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ML_INPUTS    = ROOT / "02_GIS_Outputs" / "ml_inputs"        # Phase 3 output (zonal statistics table)
INTERMEDIATE = ROOT / "02_GIS_Outputs" / "ml_intermediate"  # Phase 4 output (raw suitability nodes)
FINAL        = ROOT / "04_Final_System_Outputs"             # Phase 4b / 5 outputs consumed by the dashboard
FIGURES      = ROOT / "06_Presentation_Figures" / "model_diagnostics"

# Phase 3 -> Phase 4
DATASET_CSV = ML_INPUTS / "vijayawada_dataset.csv.gz"

# Phase 4 -> Phase 4b
SUITABILITY_NODES_CSV = INTERMEDIATE / "vijayawada_suitability_nodes.csv.gz"

# Phase 4b -> Phase 5 (and dashboard)
CORRECTED_NODES_CSV = FINAL / "vijayawada_suitability_nodes_corrected.csv"

# Phase 5 outputs
ROUTES_GEOJSON = FINAL / "optimized_flood_routes.geojson"
SINKS_CSV      = FINAL / "identified_storage_sinks.csv"

for _d in (INTERMEDIATE, FINAL, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
