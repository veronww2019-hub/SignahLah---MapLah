"""
Signal Lah - Phase 3.3
Create relative priority tiers and evidence-strength labels.

Priority tiers are based on the distribution of observed
impact scores within Kuala Selangor.

Evidence strength remains separate from impact.

Input:
    data/interim/grid_baseline_scored.geojson

Output:
    data/interim/grid_priority_labelled.geojson
"""

from pathlib import Path

import geopandas as gpd
import numpy as np


INPUT_PATH = Path(
    "data/interim/grid_baseline_scored.geojson"
)

OUTPUT_PATH = Path(
    "data/interim/grid_priority_labelled.geojson"
)


# --------------------------------------------------
# 1. Load
# --------------------------------------------------

g = gpd.read_file(INPUT_PATH)

measured = g["network_data_present"] == True

print("=== SIGNAL LAH PRIORITY LABELS ===")
print()
print("Analysis cells:", len(g))
print("Observed cells:", int(measured.sum()))
print("Evidence gaps:", int((~measured).sum()))


# --------------------------------------------------
# 2. Evidence thresholds from observed distribution
# --------------------------------------------------

tests = g.loc[
    measured,
    "network_tests",
]

test_q25 = float(
    tests.quantile(0.25)
)

test_q50 = float(
    tests.quantile(0.50)
)

test_q75 = float(
    tests.quantile(0.75)
)

print()
print("--- EVIDENCE THRESHOLDS ---")
print("P25:", test_q25)
print("P50:", test_q50)
print("P75:", test_q75)


# --------------------------------------------------
# 3. Evidence-strength labels
# --------------------------------------------------

g["evidence_label"] = "Evidence Gap"

g.loc[
    measured
    & (
        g["network_tests"]
        < test_q25
    ),
    "evidence_label",
] = "Limited"

g.loc[
    measured
    & (
        g["network_tests"]
        >= test_q25
    )
    & (
        g["network_tests"]
        < test_q50
    ),
    "evidence_label",
] = "Moderate"

g.loc[
    measured
    & (
        g["network_tests"]
        >= test_q50
    )
    & (
        g["network_tests"]
        < test_q75
    ),
    "evidence_label",
] = "Good"

g.loc[
    measured
    & (
        g["network_tests"]
        >= test_q75
    ),
    "evidence_label",
] = "Strong"


# --------------------------------------------------
# 4. Priority labels
#
# Observed score distribution:
# < P25        = Low
# P25 to P75   = Medium
# >= P75       = High
#
# No observed network data = Evidence Gap
# --------------------------------------------------

MODES = [
    "general",
    "education",
    "healthcare",
]


for mode in MODES:

    score_col = f"score_{mode}"
    priority_col = f"priority_{mode}"

    valid = g[
        score_col
    ].notna()

    scores = g.loc[
        valid,
        score_col,
    ]

    q25 = float(
        scores.quantile(0.25)
    )

    q75 = float(
        scores.quantile(0.75)
    )

    print()
    print(
        f"--- {mode.upper()} PRIORITY THRESHOLDS ---"
    )

    print(
        "Low <",
        round(q25, 2),
    )

    print(
        "Medium:",
        round(q25, 2),
        "to",
        round(q75, 2),
    )

    print(
        "High >=",
        round(q75, 2),
    )

    g[priority_col] = "Evidence Gap"

    g.loc[
        valid
        & (
            g[score_col]
            < q25
        ),
        priority_col,
    ] = "Low"

    g.loc[
        valid
        & (
            g[score_col]
            >= q25
        )
        & (
            g[score_col]
            < q75
        ),
        priority_col,
    ] = "Medium"

    g.loc[
        valid
        & (
            g[score_col]
            >= q75
        ),
        priority_col,
    ] = "High"


# --------------------------------------------------
# 5. Decision-support labels
#
# High priority is not automatically high certainty.
# --------------------------------------------------

for mode in MODES:

    priority_col = f"priority_{mode}"
    decision_col = f"decision_{mode}"

    g[decision_col] = "Evidence Gap"

    # Low
    g.loc[
        g[priority_col] == "Low",
        decision_col,
    ] = "Lower relative priority"

    # Medium
    g.loc[
        g[priority_col] == "Medium",
        decision_col,
    ] = "Monitor / investigate"

    # High + weak evidence
    g.loc[
        (g[priority_col] == "High")
        & (
            g["evidence_label"].isin(
                [
                    "Limited",
                    "Moderate",
                ]
            )
        ),
        decision_col,
    ] = "High priority - validate evidence"

    # High + stronger evidence
    g.loc[
        (g[priority_col] == "High")
        & (
            g["evidence_label"].isin(
                [
                    "Good",
                    "Strong",
                ]
            )
        ),
        decision_col,
    ] = "High priority - supported"


# --------------------------------------------------
# 6. Evidence-gap ranking
# --------------------------------------------------

g["evidence_gap_rank"] = np.nan

gap_mask = ~measured

g.loc[
    gap_mask,
    "evidence_gap_rank",
] = (
    g.loc[
        gap_mask,
        "evidence_gap_exposure",
    ]
    .rank(
        method="min",
        ascending=False,
    )
)


# --------------------------------------------------
# 7. Audit labels
# --------------------------------------------------

print()
print("--- EVIDENCE LABEL COUNTS ---")

print(
    g["evidence_label"]
    .value_counts()
    .to_string()
)


for mode in MODES:

    priority_col = f"priority_{mode}"

    print()
    print(
        f"--- {mode.upper()} PRIORITY COUNTS ---"
    )

    print(
        g[priority_col]
        .value_counts()
        .to_string()
    )


# --------------------------------------------------
# 8. High-priority cells by evidence
# --------------------------------------------------

for mode in MODES:

    score_col = f"score_{mode}"
    priority_col = f"priority_{mode}"
    decision_col = f"decision_{mode}"

    high = g[
        g[priority_col] == "High"
    ].copy()

    high = high.sort_values(
        score_col,
        ascending=False,
    )

    print()
    print(
        f"--- TOP HIGH PRIORITY: {mode.upper()} ---"
    )

    print(
        high[
            [
                "cell_id",
                score_col,
                "connectivity_gap_score",
                "population",
                "school_count",
                "healthcare_count",
                "network_tests",
                "evidence_label",
                decision_col,
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


# --------------------------------------------------
# 9. High priority but low/moderate evidence
# --------------------------------------------------

print()
print(
    "--- HIGH PRIORITY CELLS NEEDING MORE VALIDATION ---"
)

needs_validation = g[
    (
        (
            g["priority_general"] == "High"
        )
        |
        (
            g["priority_education"] == "High"
        )
        |
        (
            g["priority_healthcare"] == "High"
        )
    )
    &
    (
        g["evidence_label"].isin(
            [
                "Limited",
                "Moderate",
            ]
        )
    )
].copy()

needs_validation = (
    needs_validation
    .sort_values(
        "connectivity_gap_score",
        ascending=False,
    )
)

print(
    needs_validation[
        [
            "cell_id",
            "network_tests",
            "evidence_label",
            "score_general",
            "score_education",
            "score_healthcare",
            "priority_general",
            "priority_education",
            "priority_healthcare",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# --------------------------------------------------
# 10. Hard checks
# --------------------------------------------------

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
        "Schools changed."
    )

if int(
    g["healthcare_count"].sum()
) != 64:
    raise ValueError(
        "Healthcare changed."
    )

for mode in MODES:

    priority_col = f"priority_{mode}"

    if (
        g.loc[
            ~measured,
            priority_col,
        ]
        != "Evidence Gap"
    ).any():
        raise ValueError(
            f"Unmeasured cells received "
            f"{mode} priority labels."
        )


# --------------------------------------------------
# 11. Save
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