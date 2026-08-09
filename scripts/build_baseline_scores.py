"""
Signal Lah - Phase 3.2
Transparent non-AI connectivity impact baseline.

This model deliberately separates:

1. observed connectivity gap,
2. population / essential-service exposure,
3. evidence strength.

Cells without Ookla measurements are NOT given an inferred
connectivity score. They are marked as Evidence Gap.

Input:
    data/interim/grid_complete_raw.geojson

Output:
    data/interim/grid_baseline_scored.geojson
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/interim/grid_complete_raw.geojson"
)

OUTPUT_PATH = Path(
    "data/interim/grid_baseline_scored.geojson"
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def percentile_rank_positive(series):
    """
    Return percentile ranks for positive counts.
    Zero remains zero.
    """
    result = pd.Series(
        0.0,
        index=series.index,
        dtype=float,
    )

    positive = series > 0

    if positive.any():
        result.loc[positive] = (
            series.loc[positive]
            .rank(
                method="average",
                pct=True,
            )
        )

    return result


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

g = gpd.read_file(INPUT_PATH)

print("=== SIGNAL LAH BASELINE IMPACT MODEL ===")
print()
print("Analysis cells:", len(g))


# --------------------------------------------------
# 2. Data-status masks
# --------------------------------------------------

measured = (
    g["network_data_present"]
    == True
)

population_available = (
    g["population_data_present"]
    == True
)


# --------------------------------------------------
# 3. Connectivity gap
#
# Empirical percentile approach:
#
# lower download = worse
# lower upload   = worse
# higher latency = worse
#
# Each metric contributes equally.
# --------------------------------------------------

g["download_gap"] = np.nan
g["upload_gap"] = np.nan
g["latency_gap"] = np.nan


download_rank = (
    g.loc[
        measured,
        "avg_download_mbps",
    ]
    .rank(
        method="average",
        pct=True,
    )
)

upload_rank = (
    g.loc[
        measured,
        "avg_upload_mbps",
    ]
    .rank(
        method="average",
        pct=True,
    )
)

latency_rank = (
    g.loc[
        measured,
        "avg_latency_ms",
    ]
    .rank(
        method="average",
        pct=True,
    )
)


# High rank in speed = GOOD,
# therefore invert it to obtain a gap.
g.loc[
    measured,
    "download_gap",
] = 1 - download_rank

g.loc[
    measured,
    "upload_gap",
] = 1 - upload_rank


# High latency = BAD,
# therefore rank already represents severity.
g.loc[
    measured,
    "latency_gap",
] = latency_rank


g["connectivity_gap"] = np.nan

g.loc[
    measured,
    "connectivity_gap",
] = (
    g.loc[
        measured,
        [
            "download_gap",
            "upload_gap",
            "latency_gap",
        ],
    ]
    .mean(axis=1)
)


g["connectivity_gap_score"] = (
    g["connectivity_gap"] * 100
)


# --------------------------------------------------
# 4. Population exposure
# --------------------------------------------------

g["population_exposure"] = np.nan

population_rank = (
    g.loc[
        population_available,
        "population",
    ]
    .rank(
        method="average",
        pct=True,
    )
)

g.loc[
    population_available,
    "population_exposure",
] = population_rank


# --------------------------------------------------
# 5. School exposure
# --------------------------------------------------

g["school_exposure"] = (
    percentile_rank_positive(
        g["school_count"]
    )
)


# --------------------------------------------------
# 6. Healthcare exposure
# --------------------------------------------------

g["healthcare_exposure"] = (
    percentile_rank_positive(
        g["healthcare_count"]
    )
)


# --------------------------------------------------
# 7. Mode-specific exposure
#
# General:
#     population only
#
# Education:
#     population + school exposure
#
# Healthcare:
#     population + healthcare exposure
#
# For education and healthcare we use:
#
#   1 - (1-A)(1-B)
#
# This increases exposure when either component is high,
# without inventing arbitrary 60/40 or 70/30 weights.
# --------------------------------------------------

g["general_exposure"] = (
    g["population_exposure"]
)


g["education_exposure"] = (
    1
    - (
        1 - g["population_exposure"]
    )
    * (
        1 - g["school_exposure"]
    )
)


g["healthcare_exposure_mode"] = (
    1
    - (
        1 - g["population_exposure"]
    )
    * (
        1 - g["healthcare_exposure"]
    )
)


# --------------------------------------------------
# 8. Impact scores
#
# Connectivity gap × exposure
# --------------------------------------------------

g["score_general"] = (
    g["connectivity_gap"]
    * g["general_exposure"]
    * 100
)

g["score_education"] = (
    g["connectivity_gap"]
    * g["education_exposure"]
    * 100
)

g["score_healthcare"] = (
    g["connectivity_gap"]
    * g["healthcare_exposure_mode"]
    * 100
)


# --------------------------------------------------
# 9. Evidence strength
#
# Relative measurement-density rank.
# This is separate from the impact score.
# --------------------------------------------------

g["evidence_strength"] = np.nan

evidence_rank = (
    g.loc[
        measured,
        "network_tests",
    ]
    .rank(
        method="average",
        pct=True,
    )
)

g.loc[
    measured,
    "evidence_strength",
] = evidence_rank

g["evidence_strength_score"] = (
    g["evidence_strength"] * 100
)


# --------------------------------------------------
# 10. Data status
# --------------------------------------------------

g["analysis_status"] = "Evidence Gap"

g.loc[
    measured,
    "analysis_status",
] = "Observed"

g.loc[
    measured & (~population_available),
    "analysis_status",
] = "Observed Network / Population Unknown"


# --------------------------------------------------
# 11. Rank scored cells
# --------------------------------------------------

for column in [
    "score_general",
    "score_education",
    "score_healthcare",
]:

    rank_column = (
        column.replace(
            "score_",
            "rank_",
        )
    )

    g[rank_column] = np.nan

    valid = g[column].notna()

    g.loc[
        valid,
        rank_column,
    ] = (
        g.loc[
            valid,
            column,
        ]
        .rank(
            method="min",
            ascending=False,
        )
    )


# --------------------------------------------------
# 12. Validation
# --------------------------------------------------

print()
print("--- CONNECTIVITY GAP ---")

observed_gap = g.loc[
    measured,
    "connectivity_gap_score",
]

print(
    "Minimum:",
    round(observed_gap.min(), 2),
)

print(
    "Median:",
    round(observed_gap.median(), 2),
)

print(
    "Maximum:",
    round(observed_gap.max(), 2),
)


print()
print("--- SCORING COVERAGE ---")

print(
    "Observed cells:",
    int(measured.sum()),
)

print(
    "Evidence-gap cells:",
    int((~measured).sum()),
)

print(
    "General scores available:",
    int(
        g["score_general"]
        .notna()
        .sum()
    ),
)

print(
    "Education scores available:",
    int(
        g["score_education"]
        .notna()
        .sum()
    ),
)

print(
    "Healthcare scores available:",
    int(
        g["score_healthcare"]
        .notna()
        .sum()
    ),
)


print()
print("--- IMPACT SCORE DISTRIBUTIONS ---")

for column in [
    "score_general",
    "score_education",
    "score_healthcare",
]:

    valid = g[
        column
    ].dropna()

    q = valid.quantile(
        [
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )

    print()
    print(column)

    print(
        "Min:",
        round(valid.min(), 2),
    )

    print(
        "P10:",
        round(q.loc[0.10], 2),
    )

    print(
        "P25:",
        round(q.loc[0.25], 2),
    )

    print(
        "Median:",
        round(q.loc[0.50], 2),
    )

    print(
        "P75:",
        round(q.loc[0.75], 2),
    )

    print(
        "P90:",
        round(q.loc[0.90], 2),
    )

    print(
        "Max:",
        round(valid.max(), 2),
    )


# --------------------------------------------------
# 13. Top cells
# --------------------------------------------------

display_columns = [
    "cell_id",
    "population",
    "school_count",
    "hospital_count",
    "clinic_count",
    "network_tests",
    "avg_download_mbps",
    "avg_upload_mbps",
    "avg_latency_ms",
    "connectivity_gap_score",
]


for mode in [
    "general",
    "education",
    "healthcare",
]:

    score = f"score_{mode}"

    print()
    print(
        f"--- TOP 10 {mode.upper()} IMPACT CELLS ---"
    )

    top = (
        g[g[score].notna()]
        .sort_values(
            score,
            ascending=False,
        )
        .head(10)
    )

    print(
        top[
            display_columns
            + [score]
        ].to_string(
            index=False
        )
    )


# --------------------------------------------------
# 14. Evidence-gap shortlist
# --------------------------------------------------

g["evidence_gap_exposure"] = (
    g["population_exposure"]
)

# Increase evidence-gap concern when essential
# services are present.
g.loc[
    g["school_count"] > 0,
    "evidence_gap_exposure",
] = (
    1
    - (
        1
        - g.loc[
            g["school_count"] > 0,
            "evidence_gap_exposure",
        ]
    )
    * (
        1
        - g.loc[
            g["school_count"] > 0,
            "school_exposure",
        ]
    )
)

g.loc[
    g["healthcare_count"] > 0,
    "evidence_gap_exposure",
] = (
    1
    - (
        1
        - g.loc[
            g["healthcare_count"] > 0,
            "evidence_gap_exposure",
        ]
    )
    * (
        1
        - g.loc[
            g["healthcare_count"] > 0,
            "healthcare_exposure",
        ]
    )
)


evidence_gap = g[
    ~measured
].copy()

evidence_gap = evidence_gap.sort_values(
    "evidence_gap_exposure",
    ascending=False,
)


print()
print("--- TOP 10 EVIDENCE GAPS ---")

print(
    evidence_gap[
        [
            "cell_id",
            "population",
            "school_count",
            "healthcare_count",
            "evidence_gap_exposure",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# --------------------------------------------------
# 15. Hard checks
# --------------------------------------------------

if int(g["network_tests"].sum()) != 8441:
    raise ValueError(
        "Network tests were altered."
    )

if int(g["school_count"].sum()) != 105:
    raise ValueError(
        "School count was altered."
    )

if int(g["healthcare_count"].sum()) != 64:
    raise ValueError(
        "Healthcare count was altered."
    )

if g.loc[
    ~measured,
    "connectivity_gap",
].notna().any():
    raise ValueError(
        "Unmeasured cells received connectivity scores."
    )


# --------------------------------------------------
# 16. Save
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
print("=== COMPLETE ===")