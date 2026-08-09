"""
Signal Lah - Phase 4.6
Export the final reproducible GeoAI-ready dataset.

Input:
    data/interim/grid_with_ai_profiles.geojson

Output:
    data/processed/signal_lah_cells_v2.geojson

The existing cousin-built processed dataset is deliberately
left untouched for comparison and rollback.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np


INPUT_PATH = Path(
    "data/interim/grid_with_ai_profiles.geojson"
)

OUTPUT_PATH = Path(
    "data/processed/signal_lah_cells_v2.geojson"
)


# --------------------------------------------------
# 1. Load
# --------------------------------------------------

g = gpd.read_file(INPUT_PATH)

print("=== SIGNAL LAH V2 EXPORT ===")
print()
print("Input cells:", len(g))


# --------------------------------------------------
# 2. Version metadata
# --------------------------------------------------

g["dataset_version"] = "2.0.0"

g["geoai_model_version"] = (
    "kmeans-k3-v1"
)

g["study_area"] = (
    "Kuala Selangor"
)


# --------------------------------------------------
# 3. Clean AI cluster IDs
#
# GeoJSON can preserve NaN awkwardly depending on
# reader. We keep numeric cluster IDs where available,
# while profile/status fields provide interpretation.
# --------------------------------------------------

g["ai_cluster_id"] = (
    g["ai_cluster_id"]
    .astype("Int64")
)


# --------------------------------------------------
# 4. Select final public-facing fields
# --------------------------------------------------

FIELDS = [

    # ----------------------------------------------
    # Identity / spatial
    # ----------------------------------------------
    "cell_id",
    "cell_area_km2",
    "center_lon",
    "center_lat",
    "study_area",
    "dataset_version",
    "geoai_model_version",

    # ----------------------------------------------
    # Network observations
    # ----------------------------------------------
    "network_tiles",
    "network_tests",
    "avg_download_mbps",
    "avg_upload_mbps",
    "avg_latency_ms",
    "network_data_present",

    # ----------------------------------------------
    # Population
    # ----------------------------------------------
    "population",
    "population_pixels",
    "population_data_present",

    # ----------------------------------------------
    # Essential services
    # ----------------------------------------------
    "school_count",
    "hospital_count",
    "clinic_count",
    "doctor_count",
    "healthcare_count",

    # ----------------------------------------------
    # Transparent connectivity model
    # ----------------------------------------------
    "download_gap",
    "upload_gap",
    "latency_gap",
    "connectivity_gap_score",

    # ----------------------------------------------
    # Exposure
    # ----------------------------------------------
    "population_exposure",
    "school_exposure",
    "healthcare_exposure",
    "general_exposure",
    "education_exposure",
    "healthcare_exposure_mode",

    # ----------------------------------------------
    # Impact scores
    # ----------------------------------------------
    "score_general",
    "score_education",
    "score_healthcare",

    # ----------------------------------------------
    # Priority
    # ----------------------------------------------
    "priority_general",
    "priority_education",
    "priority_healthcare",

    # ----------------------------------------------
    # Decision support
    # ----------------------------------------------
    "decision_general",
    "decision_education",
    "decision_healthcare",

    # ----------------------------------------------
    # Evidence / uncertainty
    # ----------------------------------------------
    "analysis_status",
    "evidence_label",
    "evidence_strength_score",
    "evidence_gap_exposure",
    "evidence_gap_rank",
    "validation_needed",

    # ----------------------------------------------
    # Dashboard shortlist
    # ----------------------------------------------
    "top5_general",
    "top5_education",
    "top5_healthcare",
    "top5_evidence_gap",

    # ----------------------------------------------
    # GeoAI
    # ----------------------------------------------
    "ai_cluster_id",
    "ai_profile",
    "ai_profile_status",
    "ai_profile_description",
    "ai_profile_interpretation",
    "cluster_silhouette",

    # ----------------------------------------------
    # Geometry
    # ----------------------------------------------
    "geometry",
]


missing_fields = [
    field
    for field in FIELDS
    if field not in g.columns
]

if missing_fields:
    raise ValueError(
        "Missing export fields: "
        + ", ".join(missing_fields)
    )


output = g[FIELDS].copy()


# --------------------------------------------------
# 5. Final validation
# --------------------------------------------------

assert len(output) == 331

assert (
    output["cell_id"]
    .duplicated()
    .sum()
    == 0
)

assert output.geometry.is_valid.all()

assert int(
    output["network_tests"].sum()
) == 8441

assert int(
    output["school_count"].sum()
) == 105

assert int(
    output["healthcare_count"].sum()
) == 64

assert int(
    output["network_data_present"].sum()
) == 182

assert int(
    (~output["network_data_present"]).sum()
) == 149


# Evidence gaps must remain unclassified.
evidence_gap = (
    ~output["network_data_present"]
)

assert (
    output.loc[
        evidence_gap,
        "ai_cluster_id",
    ]
    .isna()
    .all()
)

assert (
    output.loc[
        evidence_gap,
        "ai_profile",
    ]
    == "Unclassified - Evidence Gap"
).all()


# Observed cells must have a profile.
assert (
    output.loc[
        ~evidence_gap,
        "ai_cluster_id",
    ]
    .notna()
    .all()
)


# --------------------------------------------------
# 6. Print audit
# --------------------------------------------------

print()
print("--- FINAL V2 AUDIT ---")

print(
    "Cells:",
    len(output),
)

print(
    "Network tests:",
    int(
        output[
            "network_tests"
        ].sum()
    ),
)

print(
    "Observed cells:",
    int(
        output[
            "network_data_present"
        ].sum()
    ),
)

print(
    "Evidence gaps:",
    int(
        evidence_gap.sum()
    ),
)

print(
    "Population:",
    round(
        output[
            "population"
        ].sum(),
        2,
    ),
)

print(
    "Schools:",
    int(
        output[
            "school_count"
        ].sum()
    ),
)

print(
    "Healthcare:",
    int(
        output[
            "healthcare_count"
        ].sum()
    ),
)


print()
print("--- GEOAI PROFILES ---")

print(
    output[
        "ai_profile"
    ]
    .value_counts()
    .to_string()
)


print()
print("--- GENERAL PRIORITY ---")

print(
    output[
        "priority_general"
    ]
    .value_counts()
    .to_string()
)


# --------------------------------------------------
# 7. Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.to_file(
    OUTPUT_PATH,
    driver="GeoJSON",
)

print()
print("Saved:")
print(OUTPUT_PATH)

print()
print(
    "Original processed dataset was NOT overwritten."
)

print()
print("=== PHASE 4.6 COMPLETE ===")