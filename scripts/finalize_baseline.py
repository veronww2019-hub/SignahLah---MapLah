"""
Signal Lah - Phase 3.4

Final validation and dashboard shortlist generation for the
transparent non-AI baseline model.

Input:
    data/interim/grid_priority_labelled.geojson

Outputs:
    data/interim/grid_baseline_final.geojson
    data/interim/baseline_top5.csv
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/interim/grid_priority_labelled.geojson"
)

OUTPUT_PATH = Path(
    "data/interim/grid_baseline_final.geojson"
)

TOP5_PATH = Path(
    "data/interim/baseline_top5.csv"
)

MODES = [
    "general",
    "education",
    "healthcare",
]


# --------------------------------------------------
# 1. Load
# --------------------------------------------------

g = gpd.read_file(INPUT_PATH)

print("=== SIGNAL LAH BASELINE FINALIZATION ===")
print()
print("Cells:", len(g))


# --------------------------------------------------
# 2. Core preservation checks
# --------------------------------------------------

assert len(g) == 331
assert int(g["network_tests"].sum()) == 8441
assert int(g["school_count"].sum()) == 105
assert int(g["healthcare_count"].sum()) == 64
assert g["cell_id"].duplicated().sum() == 0
assert g.geometry.is_valid.all()

print("Core data preservation: PASS")


# --------------------------------------------------
# 3. Evidence-gap safety checks
# --------------------------------------------------

unmeasured = ~g["network_data_present"]
measured = g["network_data_present"]

for mode in MODES:

    score = f"score_{mode}"
    priority = f"priority_{mode}"

    # No measurements means no inferred impact score.
    assert g.loc[
        unmeasured,
        score,
    ].isna().all()

    # No measurements means Evidence Gap.
    assert (
        g.loc[
            unmeasured,
            priority,
        ]
        == "Evidence Gap"
    ).all()

    # Observed cells must have scores.
    assert g.loc[
        measured,
        score,
    ].notna().all()


print("Evidence-gap handling: PASS")


# --------------------------------------------------
# 4. Score range checks
# --------------------------------------------------

for mode in MODES:

    score = f"score_{mode}"

    valid = g[
        score
    ].dropna()

    assert (
        (valid >= 0)
        & (valid <= 100)
    ).all()


print("Score range validation: PASS")


# --------------------------------------------------
# 5. Exposure logic checks
#
# Education and healthcare exposure can increase the
# general impact score, but should never reduce it.
# --------------------------------------------------

valid_general = g[
    "score_general"
].notna()

assert (
    g.loc[
        valid_general,
        "score_education",
    ]
    + 1e-9
    >=
    g.loc[
        valid_general,
        "score_general",
    ]
).all()

assert (
    g.loc[
        valid_general,
        "score_healthcare",
    ]
    + 1e-9
    >=
    g.loc[
        valid_general,
        "score_general",
    ]
).all()


# If there are no schools, education score should
# equal the general score.
no_schools = (
    valid_general
    & (g["school_count"] == 0)
)

assert np.allclose(
    g.loc[
        no_schools,
        "score_education",
    ],
    g.loc[
        no_schools,
        "score_general",
    ],
)


# If there is no healthcare facility, healthcare
# score should equal the general score.
no_health = (
    valid_general
    & (g["healthcare_count"] == 0)
)

assert np.allclose(
    g.loc[
        no_health,
        "score_healthcare",
    ],
    g.loc[
        no_health,
        "score_general",
    ],
)


print("Exposure-model logic: PASS")


# --------------------------------------------------
# 6. Create Top 5 flags
#
# Top 5 = highest continuous impact score.
# Evidence quality does NOT silently alter impact rank.
# --------------------------------------------------

for mode in MODES:

    score = f"score_{mode}"
    flag = f"top5_{mode}"

    g[flag] = False

    top_indices = (
        g[g[score].notna()]
        .sort_values(
            [
                score,
                "network_tests",
                "cell_id",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .head(5)
        .index
    )

    g.loc[
        top_indices,
        flag,
    ] = True


# --------------------------------------------------
# 7. Evidence validation flag
# --------------------------------------------------

high_any_mode = (
    (g["priority_general"] == "High")
    | (g["priority_education"] == "High")
    | (g["priority_healthcare"] == "High")
)

weak_evidence = g[
    "evidence_label"
].isin(
    [
        "Limited",
        "Moderate",
    ]
)

g["validation_needed"] = (
    high_any_mode
    & weak_evidence
)


# --------------------------------------------------
# 8. Evidence-gap shortlist
# --------------------------------------------------

g["top5_evidence_gap"] = False

gap_indices = (
    g[
        g["analysis_status"]
        == "Evidence Gap"
    ]
    .sort_values(
        [
            "evidence_gap_exposure",
            "population",
            "cell_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )
    .head(5)
    .index
)

g.loc[
    gap_indices,
    "top5_evidence_gap",
] = True


# --------------------------------------------------
# 9. Print Top 5 for dashboard
# --------------------------------------------------

records = []

for mode in MODES:

    score = f"score_{mode}"
    priority = f"priority_{mode}"
    flag = f"top5_{mode}"

    top = (
        g[g[flag]]
        .sort_values(
            score,
            ascending=False,
        )
        .copy()
    )

    print()
    print(
        f"--- TOP 5 {mode.upper()} ---"
    )

    print(
        top[
            [
                "cell_id",
                score,
                priority,
                "connectivity_gap_score",
                "population",
                "school_count",
                "healthcare_count",
                "network_tests",
                "evidence_label",
            ]
        ]
        .to_string(index=False)
    )

    for rank, (_, row) in enumerate(
        top.iterrows(),
        start=1,
    ):

        records.append(
            {
                "mode": mode,
                "rank": rank,
                "cell_id": row["cell_id"],
                "impact_score": row[score],
                "priority": row[priority],
                "population": row["population"],
                "school_count": row["school_count"],
                "healthcare_count": row["healthcare_count"],
                "network_tests": row["network_tests"],
                "evidence_label": row["evidence_label"],
            }
        )


# --------------------------------------------------
# 10. Print evidence-gap Top 5
# --------------------------------------------------

print()
print("--- TOP 5 EVIDENCE GAPS ---")

gap_top = (
    g[g["top5_evidence_gap"]]
    .sort_values(
        "evidence_gap_exposure",
        ascending=False,
    )
)

print(
    gap_top[
        [
            "cell_id",
            "population",
            "school_count",
            "healthcare_count",
            "evidence_gap_exposure",
        ]
    ]
    .to_string(index=False)
)


# --------------------------------------------------
# 11. Dashboard-summary statistics
# --------------------------------------------------

print()
print("--- FINAL BASELINE SUMMARY ---")

print(
    "Observed cells:",
    int(measured.sum()),
)

print(
    "Evidence-gap cells:",
    int(unmeasured.sum()),
)

print(
    "High-priority cells needing validation:",
    int(
        g["validation_needed"].sum()
    ),
)

for mode in MODES:

    print(
        f"{mode.title()} High:",
        int(
            (
                g[f"priority_{mode}"]
                == "High"
            ).sum()
        ),
    )


# --------------------------------------------------
# 12. Top-5 overlap analysis
# --------------------------------------------------

general_top = set(
    g.loc[
        g["top5_general"],
        "cell_id",
    ]
)

education_top = set(
    g.loc[
        g["top5_education"],
        "cell_id",
    ]
)

health_top = set(
    g.loc[
        g["top5_healthcare"],
        "cell_id",
    ]
)

print()
print("--- TOP 5 OVERLAP ---")

print(
    "General ∩ Education:",
    len(
        general_top
        & education_top
    ),
)

print(
    "General ∩ Healthcare:",
    len(
        general_top
        & health_top
    ),
)

print(
    "Education ∩ Healthcare:",
    len(
        education_top
        & health_top
    ),
)

print(
    "In all three:",
    len(
        general_top
        & education_top
        & health_top
    ),
)


# --------------------------------------------------
# 13. Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

g.to_file(
    OUTPUT_PATH,
    driver="GeoJSON",
)

top5_df = pd.DataFrame(
    records
)

top5_df.to_csv(
    TOP5_PATH,
    index=False,
)


print()
print("Saved final baseline:")
print(OUTPUT_PATH)

print()
print("Saved Top 5 table:")
print(TOP5_PATH)

print()
print("=== PHASE 3 COMPLETE ===")