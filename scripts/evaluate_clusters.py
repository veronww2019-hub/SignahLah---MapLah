"""
Signal Lah - Phase 4.1
Evaluate candidate K-Means cluster counts for GeoAI profiles.

The GeoAI model discovers location profiles.
It does NOT determine intervention priority.

Input:
    data/interim/grid_baseline_final.geojson

Outputs:
    data/interim/cluster_evaluation.csv
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path(
    "data/interim/grid_baseline_final.geojson"
)

OUTPUT_PATH = Path(
    "data/interim/cluster_evaluation.csv"
)

K_VALUES = range(2, 7)

SEEDS = [
    11,
    23,
    37,
    51,
    79,
]


# --------------------------------------------------
# 1. Load observed cells only
# --------------------------------------------------

g = gpd.read_file(INPUT_PATH)

observed = g[
    g["network_data_present"] == True
].copy()

print("=== SIGNAL LAH GEOAI CLUSTER EVALUATION ===")
print()
print("Total analysis cells:", len(g))
print("Observed cells used for AI:", len(observed))
print(
    "Evidence-gap cells excluded from AI training:",
    len(g) - len(observed),
)


# --------------------------------------------------
# 2. Build transparent AI feature set
# --------------------------------------------------

observed["log_population"] = np.log1p(
    observed["population"]
)

FEATURES = [
    "avg_download_mbps",
    "avg_upload_mbps",
    "avg_latency_ms",
    "log_population",
    "school_count",
    "healthcare_count",
]


print()
print("--- AI FEATURES ---")

for feature in FEATURES:
    print(feature)


# --------------------------------------------------
# 3. Validate feature completeness
# --------------------------------------------------

missing = (
    observed[FEATURES]
    .isna()
    .sum()
)

print()
print("--- MISSING VALUES ---")
print(missing.to_string())

if missing.sum() != 0:
    raise ValueError(
        "AI training features contain missing values."
    )


X_raw = observed[
    FEATURES
].astype(float)


# --------------------------------------------------
# 4. Standardize
# --------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(
    X_raw
)

print()
print("Standardized feature matrix:", X.shape)


# --------------------------------------------------
# 5. Evaluate K values
# --------------------------------------------------

results = []

reference_labels = {}

print()
print("--- CLUSTER EVALUATION ---")

for k in K_VALUES:

    seed_results = []
    seed_labels = {}

    for seed in SEEDS:

        model = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=50,
        )

        labels = model.fit_predict(
            X
        )

        seed_labels[seed] = labels

        counts = np.bincount(
            labels,
            minlength=k,
        )

        seed_results.append(
            {
                "silhouette": silhouette_score(
                    X,
                    labels,
                ),
                "calinski_harabasz": calinski_harabasz_score(
                    X,
                    labels,
                ),
                "davies_bouldin": davies_bouldin_score(
                    X,
                    labels,
                ),
                "smallest_cluster": int(
                    counts.min()
                ),
                "largest_cluster": int(
                    counts.max()
                ),
            }
        )

    metrics = pd.DataFrame(
        seed_results
    )

    # ----------------------------------------------
    # Stability:
    # compare cluster assignments between every
    # pair of random seeds using Adjusted Rand Index.
    #
    # ARI = 1 means identical grouping.
    # ----------------------------------------------

    ari_values = []

    for i in range(len(SEEDS)):

        for j in range(
            i + 1,
            len(SEEDS),
        ):

            labels_a = seed_labels[
                SEEDS[i]
            ]

            labels_b = seed_labels[
                SEEDS[j]
            ]

            ari = adjusted_rand_score(
                labels_a,
                labels_b,
            )

            ari_values.append(
                ari
            )

    result = {
        "k": k,

        "silhouette_mean": (
            metrics[
                "silhouette"
            ].mean()
        ),

        "silhouette_std": (
            metrics[
                "silhouette"
            ].std()
        ),

        "calinski_harabasz_mean": (
            metrics[
                "calinski_harabasz"
            ].mean()
        ),

        "davies_bouldin_mean": (
            metrics[
                "davies_bouldin"
            ].mean()
        ),

        "stability_ari_mean": (
            np.mean(
                ari_values
            )
        ),

        "stability_ari_min": (
            np.min(
                ari_values
            )
        ),

        "smallest_cluster_mean": (
            metrics[
                "smallest_cluster"
            ].mean()
        ),

        "largest_cluster_mean": (
            metrics[
                "largest_cluster"
            ].mean()
        ),
    }

    results.append(
        result
    )

    print()
    print("k =", k)

    print(
        "Silhouette:",
        round(
            result[
                "silhouette_mean"
            ],
            4,
        ),
    )

    print(
        "Calinski-Harabasz:",
        round(
            result[
                "calinski_harabasz_mean"
            ],
            2,
        ),
    )

    print(
        "Davies-Bouldin:",
        round(
            result[
                "davies_bouldin_mean"
            ],
            4,
        ),
    )

    print(
        "Stability ARI mean:",
        round(
            result[
                "stability_ari_mean"
            ],
            4,
        ),
    )

    print(
        "Stability ARI minimum:",
        round(
            result[
                "stability_ari_min"
            ],
            4,
        ),
    )

    print(
        "Average smallest cluster:",
        round(
            result[
                "smallest_cluster_mean"
            ],
            1,
        ),
    )

    print(
        "Average largest cluster:",
        round(
            result[
                "largest_cluster_mean"
            ],
            1,
        ),
    )


# --------------------------------------------------
# 6. Results table
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

print()
print("=== COMPARISON TABLE ===")

display = results_df.copy()

for column in [
    "silhouette_mean",
    "silhouette_std",
    "davies_bouldin_mean",
    "stability_ari_mean",
    "stability_ari_min",
]:

    display[column] = (
        display[column]
        .round(4)
    )

display[
    "calinski_harabasz_mean"
] = (
    display[
        "calinski_harabasz_mean"
    ].round(2)
)

display[
    "smallest_cluster_mean"
] = (
    display[
        "smallest_cluster_mean"
    ].round(1)
)

display[
    "largest_cluster_mean"
] = (
    display[
        "largest_cluster_mean"
    ].round(1)
)

print(
    display.to_string(
        index=False
    )
)


# --------------------------------------------------
# 7. Diagnostic recommendations
# --------------------------------------------------

best_silhouette_k = int(
    results_df.loc[
        results_df[
            "silhouette_mean"
        ].idxmax(),
        "k",
    ]
)

best_db_k = int(
    results_df.loc[
        results_df[
            "davies_bouldin_mean"
        ].idxmin(),
        "k",
    ]
)

best_ch_k = int(
    results_df.loc[
        results_df[
            "calinski_harabasz_mean"
        ].idxmax(),
        "k",
    ]
)

best_stability_k = int(
    results_df.loc[
        results_df[
            "stability_ari_mean"
        ].idxmax(),
        "k",
    ]
)


print()
print("--- METRIC WINNERS ---")

print(
    "Best silhouette k:",
    best_silhouette_k,
)

print(
    "Best Davies-Bouldin k:",
    best_db_k,
)

print(
    "Best Calinski-Harabasz k:",
    best_ch_k,
)

print(
    "Best stability k:",
    best_stability_k,
)


# --------------------------------------------------
# 8. Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print()
print("Saved:")
print(OUTPUT_PATH)

print()
print(
    "IMPORTANT: Do not choose k automatically yet."
)

print(
    "We will inspect metrics, stability and cluster sizes "
    "before selecting the final model."
)

print()
print("=== PHASE 4.1 COMPLETE ===")