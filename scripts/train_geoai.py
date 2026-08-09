"""
Signal Lah - Phase 4.2
Train the final GeoAI K-Means model.

Role of GeoAI:
Discover recurring connectivity/community profiles among
cells with observed network measurements.

The GeoAI model does NOT determine intervention priority.

Input:
    data/interim/grid_baseline_final.geojson

Outputs:
    models/geoai_kmeans_k3.joblib
    data/interim/grid_with_ai_clusters.geojson
    data/interim/cluster_profiles.csv
"""

from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = Path(
    "data/interim/grid_baseline_final.geojson"
)

MODEL_PATH = Path(
    "models/geoai_kmeans_k3.joblib"
)

GRID_OUTPUT = Path(
    "data/interim/grid_with_ai_clusters.geojson"
)

PROFILE_OUTPUT = Path(
    "data/interim/cluster_profiles.csv"
)

N_CLUSTERS = 3
RANDOM_STATE = 37
N_INIT = 100


FEATURES = [
    "avg_download_mbps",
    "avg_upload_mbps",
    "avg_latency_ms",
    "log_population",
    "school_count",
    "healthcare_count",
]


# --------------------------------------------------
# 1. Load complete baseline
# --------------------------------------------------

g = gpd.read_file(
    INPUT_PATH
)

observed_mask = (
    g["network_data_present"]
    == True
)

observed = g.loc[
    observed_mask
].copy()

print("=== SIGNAL LAH FINAL GEOAI TRAINING ===")
print()

print(
    "Total analysis cells:",
    len(g),
)

print(
    "Observed cells used for AI:",
    len(observed),
)

print(
    "Evidence-gap cells excluded:",
    len(g) - len(observed),
)


# --------------------------------------------------
# 2. Create AI features
# --------------------------------------------------

observed["log_population"] = (
    np.log1p(
        observed["population"]
    )
)

missing = (
    observed[FEATURES]
    .isna()
    .sum()
)

print()
print("--- FEATURE MISSING VALUES ---")
print(
    missing.to_string()
)

if int(missing.sum()) != 0:
    raise ValueError(
        "Missing AI feature values detected."
    )


X_raw = (
    observed[FEATURES]
    .astype(float)
)


# --------------------------------------------------
# 3. Standardize features
# --------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(
    X_raw
)


# --------------------------------------------------
# 4. Train final K-Means
# --------------------------------------------------

model = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=N_INIT,
)

labels = model.fit_predict(
    X
)


# --------------------------------------------------
# 5. Validate clustering
# --------------------------------------------------

silhouette = silhouette_score(
    X,
    labels,
)

calinski = calinski_harabasz_score(
    X,
    labels,
)

davies = davies_bouldin_score(
    X,
    labels,
)

sample_silhouette = silhouette_samples(
    X,
    labels,
)


print()
print("--- FINAL MODEL METRICS ---")

print(
    "k:",
    N_CLUSTERS,
)

print(
    "Silhouette:",
    round(
        silhouette,
        4,
    ),
)

print(
    "Calinski-Harabasz:",
    round(
        calinski,
        2,
    ),
)

print(
    "Davies-Bouldin:",
    round(
        davies,
        4,
    ),
)


# --------------------------------------------------
# 6. Attach clusters to observed cells
# --------------------------------------------------

observed["ai_cluster_id"] = labels

observed["cluster_silhouette"] = (
    sample_silhouette
)


# --------------------------------------------------
# 7. Attach to complete 331-cell grid
#
# Evidence-gap cells remain UNCLASSIFIED.
# --------------------------------------------------

g["ai_cluster_id"] = np.nan

g["cluster_silhouette"] = np.nan

g.loc[
    observed.index,
    "ai_cluster_id",
] = observed[
    "ai_cluster_id"
].astype(float)

g.loc[
    observed.index,
    "cluster_silhouette",
] = observed[
    "cluster_silhouette"
].astype(float)


g["ai_profile_status"] = (
    "Unclassified - insufficient connectivity observations"
)

g.loc[
    observed.index,
    "ai_profile_status",
] = "GeoAI classified"


# --------------------------------------------------
# 8. Cluster-size audit
# --------------------------------------------------

cluster_counts = (
    observed[
        "ai_cluster_id"
    ]
    .value_counts()
    .sort_index()
)

print()
print("--- CLUSTER SIZES ---")

print(
    cluster_counts.to_string()
)

if int(
    cluster_counts.sum()
) != len(observed):
    raise ValueError(
        "Not all observed cells received a cluster."
    )


# --------------------------------------------------
# 9. Convert K-Means centroids back to original units
# --------------------------------------------------

centres_scaled = (
    model.cluster_centers_
)

centres_raw = (
    scaler.inverse_transform(
        centres_scaled
    )
)

centroid_df = pd.DataFrame(
    centres_raw,
    columns=FEATURES,
)

# Convert log population back to approximate
# population units.
centroid_df[
    "centroid_population"
] = np.expm1(
    centroid_df[
        "log_population"
    ]
)

centroid_df = centroid_df.drop(
    columns=[
        "log_population",
    ]
)

centroid_df = centroid_df.rename(
    columns={
        "avg_download_mbps":
            "centroid_download_mbps",

        "avg_upload_mbps":
            "centroid_upload_mbps",

        "avg_latency_ms":
            "centroid_latency_ms",

        "school_count":
            "centroid_school_count",

        "healthcare_count":
            "centroid_healthcare_count",
    }
)

centroid_df[
    "ai_cluster_id"
] = np.arange(
    N_CLUSTERS
)


# --------------------------------------------------
# 10. Build interpretable cluster summaries
# --------------------------------------------------

profile_records = []

for cluster_id in range(
    N_CLUSTERS
):

    cluster = observed[
        observed["ai_cluster_id"]
        == cluster_id
    ].copy()

    centroid = centroid_df[
        centroid_df["ai_cluster_id"]
        == cluster_id
    ].iloc[0]

    record = {
        "ai_cluster_id": cluster_id,

        "cells": len(cluster),

        "mean_cluster_silhouette":
            cluster[
                "cluster_silhouette"
            ].mean(),

        # ------------------------------------------
        # Actual-cell medians
        # ------------------------------------------

        "median_download_mbps":
            cluster[
                "avg_download_mbps"
            ].median(),

        "median_upload_mbps":
            cluster[
                "avg_upload_mbps"
            ].median(),

        "median_latency_ms":
            cluster[
                "avg_latency_ms"
            ].median(),

        "median_population":
            cluster[
                "population"
            ].median(),

        "mean_school_count":
            cluster[
                "school_count"
            ].mean(),

        "mean_healthcare_count":
            cluster[
                "healthcare_count"
            ].mean(),

        "median_network_tests":
            cluster[
                "network_tests"
            ].median(),

        # ------------------------------------------
        # Baseline scores used ONLY for interpretation,
        # never as model-training features.
        # ------------------------------------------

        "mean_connectivity_gap":
            cluster[
                "connectivity_gap_score"
            ].mean(),

        "mean_general_score":
            cluster[
                "score_general"
            ].mean(),

        "mean_education_score":
            cluster[
                "score_education"
            ].mean(),

        "mean_healthcare_score":
            cluster[
                "score_healthcare"
            ].mean(),

        "general_high_pct":
            (
                cluster[
                    "priority_general"
                ]
                == "High"
            ).mean() * 100,

        "education_high_pct":
            (
                cluster[
                    "priority_education"
                ]
                == "High"
            ).mean() * 100,

        "healthcare_high_pct":
            (
                cluster[
                    "priority_healthcare"
                ]
                == "High"
            ).mean() * 100,

        # ------------------------------------------
        # Model centroid values
        # ------------------------------------------

        "centroid_download_mbps":
            centroid[
                "centroid_download_mbps"
            ],

        "centroid_upload_mbps":
            centroid[
                "centroid_upload_mbps"
            ],

        "centroid_latency_ms":
            centroid[
                "centroid_latency_ms"
            ],

        "centroid_population":
            centroid[
                "centroid_population"
            ],

        "centroid_school_count":
            centroid[
                "centroid_school_count"
            ],

        "centroid_healthcare_count":
            centroid[
                "centroid_healthcare_count"
            ],
    }

    profile_records.append(
        record
    )


profiles = pd.DataFrame(
    profile_records
)


# --------------------------------------------------
# 11. Print profiles
# --------------------------------------------------

print()
print("=== CLUSTER PROFILE SUMMARY ===")

display_columns = [
    "ai_cluster_id",
    "cells",
    "mean_cluster_silhouette",
    "median_download_mbps",
    "median_upload_mbps",
    "median_latency_ms",
    "median_population",
    "mean_school_count",
    "mean_healthcare_count",
    "mean_connectivity_gap",
    "mean_general_score",
    "general_high_pct",
]

print(
    profiles[
        display_columns
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 12. Print examples from each cluster
# --------------------------------------------------

for cluster_id in range(
    N_CLUSTERS
):

    cluster = observed[
        observed["ai_cluster_id"]
        == cluster_id
    ].copy()

    # Show cells nearest the cluster centroid by
    # highest silhouette value.
    examples = (
        cluster
        .sort_values(
            "cluster_silhouette",
            ascending=False,
        )
        .head(5)
    )

    print()
    print(
        f"--- REPRESENTATIVE CELLS: CLUSTER {cluster_id} ---"
    )

    print(
        examples[
            [
                "cell_id",
                "avg_download_mbps",
                "avg_upload_mbps",
                "avg_latency_ms",
                "population",
                "school_count",
                "healthcare_count",
                "connectivity_gap_score",
                "score_general",
                "cluster_silhouette",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )


# --------------------------------------------------
# 13. Hard safety checks
# --------------------------------------------------

if (
    g.loc[
        ~observed_mask,
        "ai_cluster_id",
    ]
    .notna()
    .any()
):
    raise ValueError(
        "Evidence-gap cells incorrectly received "
        "AI clusters."
    )


if (
    g.loc[
        observed_mask,
        "ai_cluster_id",
    ]
    .isna()
    .any()
):
    raise ValueError(
        "Observed cells missing AI clusters."
    )


if int(
    g["network_tests"].sum()
) != 8441:
    raise ValueError(
        "Network tests changed."
    )


if int(
    g["school_count"].sum()
) != 105:
    raise ValueError(
        "School count changed."
    )


if int(
    g["healthcare_count"].sum()
) != 64:
    raise ValueError(
        "Healthcare count changed."
    )


# --------------------------------------------------
# 14. Save trained model bundle
# --------------------------------------------------

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

model_bundle = {
    "model_type": "KMeans",
    "purpose": (
        "Unsupervised discovery of recurring "
        "connectivity-community profiles"
    ),
    "n_clusters": N_CLUSTERS,
    "random_state": RANDOM_STATE,
    "n_init": N_INIT,
    "features": FEATURES,
    "scaler": scaler,
    "model": model,
    "training_cells": len(observed),
    "silhouette_score": silhouette,
    "calinski_harabasz_score": calinski,
    "davies_bouldin_score": davies,
}

joblib.dump(
    model_bundle,
    MODEL_PATH,
)


# --------------------------------------------------
# 15. Save outputs
# --------------------------------------------------

GRID_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

g.to_file(
    GRID_OUTPUT,
    driver="GeoJSON",
)

profiles.to_csv(
    PROFILE_OUTPUT,
    index=False,
)


print()
print("Saved trained model:")
print(MODEL_PATH)

print()
print("Saved clustered grid:")
print(GRID_OUTPUT)

print()
print("Saved cluster profiles:")
print(PROFILE_OUTPUT)

print()
print("=== PHASE 4.2 COMPLETE ===")