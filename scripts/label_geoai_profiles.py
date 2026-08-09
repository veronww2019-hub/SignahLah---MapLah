"""
Signal Lah - Phase 4.3 / 4.4
Attach interpretable names and descriptions to trained GeoAI clusters.

Input:
    data/interim/grid_with_ai_clusters.geojson
    models/geoai_kmeans_k3.joblib

Outputs:
    data/interim/grid_with_ai_profiles.geojson

Also updates:
    models/geoai_kmeans_k3.joblib

IMPORTANT:
Profile names describe patterns discovered by K-Means.
They do NOT determine intervention priority.
"""

from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np


INPUT_PATH = Path(
    "data/interim/grid_with_ai_clusters.geojson"
)

OUTPUT_PATH = Path(
    "data/interim/grid_with_ai_profiles.geojson"
)

MODEL_PATH = Path(
    "models/geoai_kmeans_k3.joblib"
)


# --------------------------------------------------
# 1. Human-readable profile definitions
# --------------------------------------------------

PROFILE_NAMES = {
    0: "Strong-Connectivity Areas",
    1: "High-Exposure Essential-Service Hubs",
    2: "Connectivity-Constrained Areas",
}


PROFILE_DESCRIPTIONS = {
    0: (
        "Areas characterised by comparatively strong mobile "
        "download and upload performance, lower latency, "
        "moderate population exposure, and relatively few "
        "mapped essential-service facilities."
    ),

    1: (
    "Areas with comparatively high population exposure and "
    "concentrations of mapped schools and healthcare facilities. "
    "Connectivity is generally moderate to strong, but disruption "
    "or degradation may affect more people and essential services."
    ),

    2: (
        "Areas characterised by comparatively lower download "
        "and upload performance, higher connectivity gaps, "
        "lower population density, and relatively few mapped "
        "essential-service facilities."
    ),
}


# --------------------------------------------------
# 2. Load clustered grid
# --------------------------------------------------

g = gpd.read_file(INPUT_PATH)

observed = (
    g["ai_profile_status"]
    == "GeoAI classified"
)

print("=== SIGNAL LAH GEOAI PROFILE LABELLING ===")
print()

print(
    "Total cells:",
    len(g),
)

print(
    "GeoAI-classified cells:",
    int(observed.sum()),
)

print(
    "Unclassified evidence-gap cells:",
    int((~observed).sum()),
)


# --------------------------------------------------
# 3. Create profile labels
# --------------------------------------------------

g["ai_profile"] = (
    "Unclassified - Evidence Gap"
)

g["ai_profile_description"] = (
    "Insufficient observed connectivity data for GeoAI profiling."
)


for cluster_id, profile_name in PROFILE_NAMES.items():

    mask = (
        g["ai_cluster_id"]
        == cluster_id
    )

    g.loc[
        mask,
        "ai_profile",
    ] = profile_name

    g.loc[
        mask,
        "ai_profile_description",
    ] = PROFILE_DESCRIPTIONS[
        cluster_id
    ]


# --------------------------------------------------
# 4. Add cluster interpretation flag
#
# Cluster 1 has lower mean silhouette than the others.
# We therefore explicitly preserve that information.
# --------------------------------------------------

g["ai_profile_interpretation"] = (
    "Not classified"
)

g.loc[
    g["ai_cluster_id"] == 0,
    "ai_profile_interpretation",
] = "Distinct recurring profile"

g.loc[
    g["ai_cluster_id"] == 1,
    "ai_profile_interpretation",
] = "Mixed / transitional recurring profile"

g.loc[
    g["ai_cluster_id"] == 2,
    "ai_profile_interpretation",
] = "Distinct recurring profile"


# --------------------------------------------------
# 5. Audit profile counts
# --------------------------------------------------

print()
print("--- GEOAI PROFILE COUNTS ---")

print(
    g["ai_profile"]
    .value_counts()
    .to_string()
)


print()
print("--- PROFILE VS GENERAL PRIORITY ---")

table = (
    g[
        g["ai_profile_status"]
        == "GeoAI classified"
    ]
    .groupby(
        [
            "ai_profile",
            "priority_general",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(
    table.to_string()
)


# --------------------------------------------------
# 6. Print profile summary
# --------------------------------------------------

print()
print("--- PROFILE CHARACTERISTICS ---")

for cluster_id in sorted(
    PROFILE_NAMES
):

    cluster = g[
        g["ai_cluster_id"]
        == cluster_id
    ]

    print()
    print(
        f"Cluster {cluster_id}: "
        f"{PROFILE_NAMES[cluster_id]}"
    )

    print(
        "Cells:",
        len(cluster),
    )

    print(
        "Median download:",
        round(
            cluster[
                "avg_download_mbps"
            ].median(),
            2,
        ),
        "Mbps",
    )

    print(
        "Median upload:",
        round(
            cluster[
                "avg_upload_mbps"
            ].median(),
            2,
        ),
        "Mbps",
    )

    print(
        "Median latency:",
        round(
            cluster[
                "avg_latency_ms"
            ].median(),
            2,
        ),
        "ms",
    )

    print(
        "Median population:",
        round(
            cluster[
                "population"
            ].median(),
            2,
        ),
    )

    print(
        "Mean schools:",
        round(
            cluster[
                "school_count"
            ].mean(),
            2,
        ),
    )

    print(
        "Mean healthcare:",
        round(
            cluster[
                "healthcare_count"
            ].mean(),
            2,
        ),
    )

    print(
        "Mean connectivity gap:",
        round(
            cluster[
                "connectivity_gap_score"
            ].mean(),
            2,
        ),
    )

    print(
        "Mean cluster silhouette:",
        round(
            cluster[
                "cluster_silhouette"
            ].mean(),
            3,
        ),
    )


# --------------------------------------------------
# 7. Hard validation
# --------------------------------------------------

classified = g[
    "ai_profile_status"
] == "GeoAI classified"

if (
    g.loc[
        classified,
        "ai_profile",
    ]
    == "Unclassified - Evidence Gap"
).any():
    raise ValueError(
        "A classified cell has no GeoAI profile."
    )


if (
    g.loc[
        ~classified,
        "ai_cluster_id",
    ]
    .notna()
    .any()
):
    raise ValueError(
        "Evidence-gap cells received AI clusters."
    )


if int(
    g["network_tests"].sum()
) != 8441:
    raise ValueError(
        "Network tests changed."
    )


# --------------------------------------------------
# 8. Update saved model metadata
# --------------------------------------------------

bundle = joblib.load(
    MODEL_PATH
)

bundle[
    "profile_names"
] = PROFILE_NAMES

bundle[
    "profile_descriptions"
] = PROFILE_DESCRIPTIONS

bundle[
    "cluster_interpretation"
] = {
    0: "Distinct recurring profile",
    1: "Mixed / transitional recurring profile",
    2: "Distinct recurring profile",
}

joblib.dump(
    bundle,
    MODEL_PATH,
)


# --------------------------------------------------
# 9. Save labelled grid
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

g.to_file(
    OUTPUT_PATH,
    driver="GeoJSON",
)


print()
print("Saved:")
print(OUTPUT_PATH)

print()
print(
    "Updated model metadata:"
)
print(MODEL_PATH)

print()
print(
    "=== PHASE 4.4 COMPLETE ==="
)