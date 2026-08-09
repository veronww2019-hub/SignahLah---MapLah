"""
Signal Lah - Phase 4.5
Final validation of the trained GeoAI clustering model.

This script:
1. verifies data/model integrity,
2. evaluates cluster coherence,
3. checks k=3 stability,
4. measures ambiguous cluster assignments,
5. compares GeoAI profiles with transparent priority outputs,
6. writes a reproducible model report.

Inputs:
    data/interim/grid_with_ai_profiles.geojson
    data/interim/cluster_evaluation.csv
    models/geoai_kmeans_k3.joblib

Outputs:
    data/interim/geoai_validation.csv
    docs/geoai_model_report.md
"""

from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd


GRID_PATH = Path(
    "data/interim/grid_with_ai_profiles.geojson"
)

EVALUATION_PATH = Path(
    "data/interim/cluster_evaluation.csv"
)

MODEL_PATH = Path(
    "models/geoai_kmeans_k3.joblib"
)

VALIDATION_OUTPUT = Path(
    "data/interim/geoai_validation.csv"
)

REPORT_OUTPUT = Path(
    "docs/geoai_model_report.md"
)


# --------------------------------------------------
# 1. Load
# --------------------------------------------------

g = gpd.read_file(
    GRID_PATH
)

evaluation = pd.read_csv(
    EVALUATION_PATH
)

bundle = joblib.load(
    MODEL_PATH
)


classified = (
    g["ai_cluster_id"]
    .notna()
)

unclassified = ~classified


print("=== SIGNAL LAH GEOAI VALIDATION ===")
print()

print(
    "Total cells:",
    len(g),
)

print(
    "Classified cells:",
    int(classified.sum()),
)

print(
    "Unclassified evidence-gap cells:",
    int(unclassified.sum()),
)


# --------------------------------------------------
# 2. Hard integrity checks
# --------------------------------------------------

assert len(g) == 331

assert int(
    classified.sum()
) == 182

assert int(
    unclassified.sum()
) == 149

assert int(
    g["network_tests"].sum()
) == 8441

assert int(
    g["school_count"].sum()
) == 105

assert int(
    g["healthcare_count"].sum()
) == 64

assert (
    g.loc[
        unclassified,
        "ai_cluster_id",
    ]
    .isna()
    .all()
)

assert (
    g.loc[
        classified,
        "ai_cluster_id",
    ]
    .notna()
    .all()
)

assert set(
    g.loc[
        classified,
        "ai_cluster_id",
    ]
    .astype(int)
    .unique()
) == {0, 1, 2}

assert bundle[
    "n_clusters"
] == 3

assert bundle[
    "training_cells"
] == 182


print(
    "Model/data integrity: PASS"
)


# --------------------------------------------------
# 3. Confirm overall silhouette
# --------------------------------------------------

sample_mean = (
    g.loc[
        classified,
        "cluster_silhouette",
    ]
    .mean()
)

saved_silhouette = float(
    bundle[
        "silhouette_score"
    ]
)

if not np.isclose(
    sample_mean,
    saved_silhouette,
    atol=1e-6,
):
    raise ValueError(
        "Saved overall silhouette does not "
        "match sample silhouette values."
    )


print(
    "Silhouette consistency: PASS"
)


# --------------------------------------------------
# 4. Retrieve k=3 stability
# --------------------------------------------------

k3 = evaluation[
    evaluation["k"] == 3
]

if len(k3) != 1:
    raise ValueError(
        "Expected exactly one k=3 evaluation row."
    )

k3 = k3.iloc[0]

stability_mean = float(
    k3["stability_ari_mean"]
)

stability_min = float(
    k3["stability_ari_min"]
)


print()
print("--- FINAL MODEL ---")

print(
    "k:",
    bundle["n_clusters"],
)

print(
    "Overall silhouette:",
    round(
        saved_silhouette,
        4,
    ),
)

print(
    "Davies-Bouldin:",
    round(
        bundle[
            "davies_bouldin_score"
        ],
        4,
    ),
)

print(
    "Calinski-Harabasz:",
    round(
        bundle[
            "calinski_harabasz_score"
        ],
        2,
    ),
)

print(
    "Stability ARI mean:",
    round(
        stability_mean,
        4,
    ),
)

print(
    "Stability ARI minimum:",
    round(
        stability_min,
        4,
    ),
)


# --------------------------------------------------
# 5. Per-cluster validation
# --------------------------------------------------

profile_names = bundle[
    "profile_names"
]

records = []

for cluster_id in [0, 1, 2]:

    cluster = g[
        g["ai_cluster_id"]
        == cluster_id
    ].copy()

    silhouette_values = cluster[
        "cluster_silhouette"
    ]

    negative_count = int(
        (
            silhouette_values < 0
        ).sum()
    )

    negative_pct = (
        negative_count
        / len(cluster)
        * 100
    )

    low_fit_count = int(
        (
            silhouette_values < 0.10
        ).sum()
    )

    low_fit_pct = (
        low_fit_count
        / len(cluster)
        * 100
    )

    mean_silhouette = float(
        silhouette_values.mean()
    )

    if (
        mean_silhouette < 0.15
        or negative_pct >= 20
    ):
        interpretation = (
            "Mixed / transitional profile - "
            "interpret individual assignments cautiously"
        )
    else:
        interpretation = (
            "Reasonably coherent recurring profile"
        )

    record = {
        "ai_cluster_id":
            cluster_id,

        "ai_profile":
            profile_names[
                cluster_id
            ],

        "cells":
            len(cluster),

        "mean_silhouette":
            mean_silhouette,

        "median_silhouette":
            silhouette_values.median(),

        "minimum_silhouette":
            silhouette_values.min(),

        "negative_silhouette_cells":
            negative_count,

        "negative_silhouette_pct":
            negative_pct,

        "silhouette_below_0_10_cells":
            low_fit_count,

        "silhouette_below_0_10_pct":
            low_fit_pct,

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

        "mean_connectivity_gap":
            cluster[
                "connectivity_gap_score"
            ].mean(),

        "general_high_pct":
            (
                (
                    cluster[
                        "priority_general"
                    ]
                    == "High"
                )
                .mean()
                * 100
            ),

        "education_high_pct":
            (
                (
                    cluster[
                        "priority_education"
                    ]
                    == "High"
                )
                .mean()
                * 100
            ),

        "healthcare_high_pct":
            (
                (
                    cluster[
                        "priority_healthcare"
                    ]
                    == "High"
                )
                .mean()
                * 100
            ),

        "interpretation":
            interpretation,
    }

    records.append(
        record
    )


validation = pd.DataFrame(
    records
)


print()
print("=== CLUSTER VALIDATION ===")

display_columns = [
    "ai_cluster_id",
    "ai_profile",
    "cells",
    "mean_silhouette",
    "median_silhouette",
    "negative_silhouette_cells",
    "negative_silhouette_pct",
    "silhouette_below_0_10_pct",
    "interpretation",
]

print(
    validation[
        display_columns
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 6. Overall assignment ambiguity
# --------------------------------------------------

classified_data = g[
    classified
].copy()

negative_total = int(
    (
        classified_data[
            "cluster_silhouette"
        ]
        < 0
    ).sum()
)

low_fit_total = int(
    (
        classified_data[
            "cluster_silhouette"
        ]
        < 0.10
    ).sum()
)


print()
print("--- ASSIGNMENT AMBIGUITY ---")

print(
    "Negative silhouette cells:",
    negative_total,
)

print(
    "Negative silhouette percentage:",
    round(
        negative_total
        / len(classified_data)
        * 100,
        2,
    ),
)

print(
    "Cells with silhouette < 0.10:",
    low_fit_total,
)

print(
    "Percentage with silhouette < 0.10:",
    round(
        low_fit_total
        / len(classified_data)
        * 100,
        2,
    ),
)


# --------------------------------------------------
# 7. Profile vs priority
#
# We do NOT expect perfect agreement.
# AI profile and priority have different purposes.
# --------------------------------------------------

print()
print("--- PROFILE / GENERAL PRIORITY ---")

profile_priority = (
    classified_data
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
    profile_priority
    .to_string()
)


# --------------------------------------------------
# 8. Profile distributions
# --------------------------------------------------

print()
print("--- PROFILE CHARACTERISTICS ---")

character_columns = [
    "ai_cluster_id",
    "ai_profile",
    "median_download_mbps",
    "median_upload_mbps",
    "median_latency_ms",
    "median_population",
    "mean_school_count",
    "mean_healthcare_count",
    "mean_connectivity_gap",
]

print(
    validation[
        character_columns
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 9. Save validation table
# --------------------------------------------------

VALIDATION_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

validation.to_csv(
    VALIDATION_OUTPUT,
    index=False,
)


# --------------------------------------------------
# 10. Create model report
# --------------------------------------------------

REPORT_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


report_lines = []

report_lines.append(
    "# Signal Lah GeoAI Model Report"
)

report_lines.append("")
report_lines.append(
    "## Purpose"
)

report_lines.append("")
report_lines.append(
    "The GeoAI component uses unsupervised K-Means clustering "
    "to discover recurring connectivity-community profiles "
    "among analysis cells with observed mobile-network data. "
    "The model does not determine intervention priority; "
    "priority is produced separately by the transparent "
    "connectivity-impact model."
)

report_lines.append("")
report_lines.append(
    "## Training Data"
)

report_lines.append("")
report_lines.append(
    "- Total analysis cells: 331"
)

report_lines.append(
    "- Cells used for GeoAI training: 182"
)

report_lines.append(
    "- Evidence-gap cells excluded from training: 149"
)

report_lines.append(
    "- No network values were imputed for evidence-gap cells."
)


report_lines.append("")
report_lines.append(
    "## Features"
)

report_lines.append("")

for feature in bundle["features"]:

    report_lines.append(
        f"- `{feature}`"
    )


report_lines.append("")
report_lines.append(
    "All model features were standardized before K-Means training."
)


report_lines.append("")
report_lines.append(
    "## Model Selection"
)

report_lines.append("")

report_lines.append(
    "Candidate cluster counts k=2 through k=6 were compared "
    "using silhouette score, Calinski-Harabasz score, "
    "Davies-Bouldin score, cluster size and repeated-seed "
    "Adjusted Rand Index stability."
)

report_lines.append("")

report_lines.append(
    f"The selected model uses k=3. "
    f"Its mean stability ARI was {stability_mean:.4f} "
    f"and minimum stability ARI was {stability_min:.4f}. "
    "The k=4 to k=6 alternatives produced very small clusters, "
    "while k=3 retained meaningful cluster sizes and extremely "
    "stable assignments."
)


report_lines.append("")
report_lines.append(
    "## Final Internal Validation"
)

report_lines.append("")

report_lines.append(
    f"- Silhouette score: {saved_silhouette:.4f}"
)

report_lines.append(
    "- Calinski-Harabasz score: "
    f"{bundle['calinski_harabasz_score']:.2f}"
)

report_lines.append(
    "- Davies-Bouldin score: "
    f"{bundle['davies_bouldin_score']:.4f}"
)


report_lines.append("")
report_lines.append(
    "## GeoAI Profiles"
)

report_lines.append("")

report_lines.append(
    "| Cluster | Profile | Cells | Mean silhouette | "
    "Median download (Mbps) | Median upload (Mbps) | "
    "Median latency (ms) | Median population |"
)

report_lines.append(
    "|---|---|---:|---:|---:|---:|---:|---:|"
)


for _, row in validation.iterrows():

    report_lines.append(
        f"| {int(row['ai_cluster_id'])} "
        f"| {row['ai_profile']} "
        f"| {int(row['cells'])} "
        f"| {row['mean_silhouette']:.3f} "
        f"| {row['median_download_mbps']:.2f} "
        f"| {row['median_upload_mbps']:.2f} "
        f"| {row['median_latency_ms']:.2f} "
        f"| {row['median_population']:.2f} |"
    )


report_lines.append("")
report_lines.append(
    "## Interpretation"
)

report_lines.append("")

report_lines.append(
    "- **Strong-Connectivity Areas:** comparatively strong "
    "download/upload performance, lower latency and lower "
    "connectivity-gap scores."
)

report_lines.append(
    "- **Dense Essential-Service Hubs:** high population and "
    "concentrations of mapped schools and healthcare facilities. "
    "This cluster is more mixed and should be interpreted as a "
    "transitional recurring profile rather than a sharply "
    "separated class."
)

report_lines.append(
    "- **Connectivity-Constrained Areas:** comparatively lower "
    "download/upload performance and higher connectivity-gap "
    "scores, typically with lower population and fewer mapped "
    "essential-service facilities."
)


report_lines.append("")
report_lines.append(
    "## Important Limitations"
)

report_lines.append("")

report_lines.append(
    "- K-Means clusters are descriptive profiles, not ground-truth labels."
)

report_lines.append(
    "- Cluster IDs have no inherent ranking or severity meaning."
)

report_lines.append(
    "- The Dense Essential-Service Hubs cluster has weaker internal "
    "separation than the other two profiles."
)

report_lines.append(
    "- The model is trained only on cells with observed Ookla data."
)

report_lines.append(
    "- Absence of Ookla measurements is treated as an evidence gap, "
    "not proof of poor or good connectivity."
)

report_lines.append(
    "- OSM facility counts represent mapped facilities and may not "
    "equal official administrative totals."
)

report_lines.append(
    "- GeoAI profiles supplement rather than replace the transparent "
    "General, Education and Healthcare priority scores."
)


REPORT_OUTPUT.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


print()
print("Saved validation:")
print(VALIDATION_OUTPUT)

print()
print("Saved model report:")
print(REPORT_OUTPUT)

print()
print("=== PHASE 4.5 COMPLETE ===")