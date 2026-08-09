"""
Signal Lah - Phase 5.4

Export all assets required for the browser-based
What-If Connectivity Impact Simulator.

Inputs:
    data/processed/signal_lah_cells_v2.geojson
    data/processed/scenario_presets.json
    models/geoai_kmeans_k3.joblib

Outputs:
    web/signal_lah_cells_v2.geojson
    web/scenario_presets.json
    web/simulator_config.json

The existing web/signal_lah_cells.geojson is NOT overwritten.
"""

from pathlib import Path
import json
import shutil

import geopandas as gpd
import joblib

from simulate_intervention import (
    build_percentile_reference,
    get_priority_thresholds,
)


DATA_PATH = Path(
    "data/processed/signal_lah_cells_v2.geojson"
)

PRESETS_PATH = Path(
    "data/processed/scenario_presets.json"
)

MODEL_PATH = Path(
    "models/geoai_kmeans_k3.joblib"
)

WEB_DIR = Path("web")

WEB_DATA_PATH = (
    WEB_DIR
    / "signal_lah_cells_v2.geojson"
)

WEB_PRESETS_PATH = (
    WEB_DIR
    / "scenario_presets.json"
)

WEB_CONFIG_PATH = (
    WEB_DIR
    / "simulator_config.json"
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def reference_to_json(reference):

    return {
        "values": [
            float(value)
            for value in reference[
                "value"
            ].tolist()
        ],

        "ranks": [
            float(rank)
            for rank in reference[
                "rank"
            ].tolist()
        ],
    }


# --------------------------------------------------
# 1. Load data and model
# --------------------------------------------------

g = gpd.read_file(
    DATA_PATH
)

bundle = joblib.load(
    MODEL_PATH
)

observed = g[
    g["network_data_present"]
    == True
].copy()


print(
    "=== SIGNAL LAH WEB SIMULATOR EXPORT ==="
)

print()

print(
    "Total cells:",
    len(g),
)

print(
    "Observed cells:",
    len(observed),
)


# --------------------------------------------------
# 2. Empirical percentile references
# --------------------------------------------------

download_reference = (
    build_percentile_reference(
        observed[
            "avg_download_mbps"
        ]
    )
)

upload_reference = (
    build_percentile_reference(
        observed[
            "avg_upload_mbps"
        ]
    )
)

latency_reference = (
    build_percentile_reference(
        observed[
            "avg_latency_ms"
        ]
    )
)


# --------------------------------------------------
# 3. Priority thresholds
# --------------------------------------------------

priority_thresholds = {}

for mode in [
    "general",
    "education",
    "healthcare",
]:

    threshold = (
        get_priority_thresholds(
            g,
            f"score_{mode}",
        )
    )

    priority_thresholds[
        mode
    ] = {
        "q25": float(
            threshold["q25"]
        ),

        "q75": float(
            threshold["q75"]
        ),
    }


# --------------------------------------------------
# 4. GeoAI model export
#
# StandardScaler:
#
#     z = (x - mean) / scale
#
# Then K-Means uses Euclidean distance to
# cluster_centers_scaled.
# --------------------------------------------------

scaler = bundle[
    "scaler"
]

model = bundle[
    "model"
]


profile_names = {
    str(key): value
    for key, value
    in bundle[
        "profile_names"
    ].items()
}


profile_descriptions = {
    str(key): value
    for key, value
    in bundle[
        "profile_descriptions"
    ].items()
}


cluster_interpretation = {
    str(key): value
    for key, value
    in bundle[
        "cluster_interpretation"
    ].items()
}


geoai = {

    "model_type":
        "KMeans",

    "model_version":
        "kmeans-k3-v1",

    "n_clusters":
        int(
            bundle[
                "n_clusters"
            ]
        ),

    "features":
        list(
            bundle[
                "features"
            ]
        ),

    "scaler_mean": [
        float(x)
        for x in scaler.mean_
    ],

    "scaler_scale": [
        float(x)
        for x in scaler.scale_
    ],

    "cluster_centers_scaled": [
        [
            float(value)
            for value in row
        ]
        for row
        in model.cluster_centers_
    ],

    "profile_names":
        profile_names,

    "profile_descriptions":
        profile_descriptions,

    "cluster_interpretation":
        cluster_interpretation,

    "training_cells":
        int(
            bundle[
                "training_cells"
            ]
        ),

    "silhouette_score":
        float(
            bundle[
                "silhouette_score"
            ]
        ),

    "calinski_harabasz_score":
        float(
            bundle[
                "calinski_harabasz_score"
            ]
        ),

    "davies_bouldin_score":
        float(
            bundle[
                "davies_bouldin_score"
            ]
        ),
}


# --------------------------------------------------
# 5. Browser simulator config
# --------------------------------------------------

config = {

    "version":
        "1.0.0",

    "dataset_version":
        "2.0.0",

    "study_area":
        "Kuala Selangor",

    "method": (
        "Browser configuration reproducing the "
        "transparent Signal Lah connectivity-impact "
        "simulator and trained K-Means GeoAI model."
    ),

    "important_notes": [

        (
            "Priority is a relative decision-support "
            "measure within the study area, not a "
            "regulatory service classification."
        ),

        (
            "Scenario presets are empirical what-if "
            "targets, not guaranteed infrastructure "
            "outcomes."
        ),

        (
            "GeoAI cluster assignment is descriptive "
            "and is not a probability."
        ),

        (
            "Cells without observed Ookla measurements "
            "remain Evidence Gaps."
        ),
    ],


    "percentile_references": {

        "download":
            reference_to_json(
                download_reference
            ),

        "upload":
            reference_to_json(
                upload_reference
            ),

        "latency":
            reference_to_json(
                latency_reference
            ),
    },


    "priority_thresholds":
        priority_thresholds,


    "geoai":
        geoai,
}


# --------------------------------------------------
# 6. Validation
# --------------------------------------------------

assert len(g) == 331

assert int(
    g["network_tests"].sum()
) == 8441

assert int(
    g["school_count"].sum()
) == 105

assert int(
    g["healthcare_count"].sum()
) == 64

assert len(
    config[
        "percentile_references"
    ][
        "download"
    ][
        "values"
    ]
) > 0

assert len(
    geoai[
        "scaler_mean"
    ]
) == 6

assert len(
    geoai[
        "scaler_scale"
    ]
) == 6

assert len(
    geoai[
        "cluster_centers_scaled"
    ]
) == 3


print()
print(
    "--- PRIORITY THRESHOLDS ---"
)

for mode, values in (
    priority_thresholds.items()
):

    print(
        mode.title(),
        "P25 =",
        round(
            values["q25"],
            2,
        ),
        "| P75 =",
        round(
            values["q75"],
            2,
        ),
    )


print()
print(
    "--- GEOAI EXPORT ---"
)

print(
    "Features:",
    geoai[
        "features"
    ],
)

print(
    "Clusters:",
    geoai[
        "n_clusters"
    ],
)

print(
    "Training cells:",
    geoai[
        "training_cells"
    ],
)


# --------------------------------------------------
# 7. Export
# --------------------------------------------------

WEB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Copy final GeoJSON
shutil.copy2(
    DATA_PATH,
    WEB_DATA_PATH,
)


# Copy scenario presets
shutil.copy2(
    PRESETS_PATH,
    WEB_PRESETS_PATH,
)


# Write browser configuration
WEB_CONFIG_PATH.write_text(
    json.dumps(
        config,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print(
    "Saved V2 map data:"
)

print(
    WEB_DATA_PATH
)

print()
print(
    "Saved presets:"
)

print(
    WEB_PRESETS_PATH
)

print()
print(
    "Saved simulator config:"
)

print(
    WEB_CONFIG_PATH
)

print()
print(
    "Existing web/signal_lah_cells.geojson "
    "was NOT overwritten."
)

print()
print(
    "=== PHASE 5.4 COMPLETE ==="
)