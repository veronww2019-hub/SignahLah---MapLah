"""
Signal Lah - Phase 5.2B
Evaluate empirical intervention presets across observed cells.

Uses upgrade-only logic:
- download can only increase
- upload can only increase
- latency can only decrease

Outputs:
    data/interim/scenario_preset_evaluation.csv
"""

from pathlib import Path
import json

import geopandas as gpd
import pandas as pd

from simulate_intervention import (
    build_percentile_reference,
    percentile_for_value,
    simulate_connectivity_gap,
    get_priority_thresholds,
    priority_from_score,
)


DATA_PATH = Path(
    "data/processed/signal_lah_cells_v2.geojson"
)

PRESET_PATH = Path(
    "data/processed/scenario_presets.json"
)

OUTPUT_PATH = Path(
    "data/interim/scenario_preset_evaluation.csv"
)


# --------------------------------------------------
# 1. Load
# --------------------------------------------------

g = gpd.read_file(DATA_PATH)

preset_data = json.loads(
    PRESET_PATH.read_text(
        encoding="utf-8"
    )
)

presets = preset_data[
    "presets"
]


observed = g[
    g["network_data_present"] == True
].copy()


print(
    "=== SIGNAL LAH PRESET BENEFIT EVALUATION ==="
)

print()
print(
    "Observed cells:",
    len(observed),
)


# --------------------------------------------------
# 2. Build baseline empirical references
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
# 3. Fixed baseline priority thresholds
# --------------------------------------------------

thresholds = {}

for mode in [
    "general",
    "education",
    "healthcare",
]:

    thresholds[mode] = (
        get_priority_thresholds(
            g,
            f"score_{mode}",
        )
    )


# --------------------------------------------------
# 4. Evaluate each preset
# --------------------------------------------------

results = []


for preset_key, preset in presets.items():

    print()
    print(
        "======================================"
    )

    print(
        preset["label"]
    )

    print(
        "======================================"
    )


    scenario_rows = []


    for _, cell in observed.iterrows():

        # ------------------------------------------
        # Upgrade-only target
        # ------------------------------------------

        scenario_download = max(
            float(
                cell[
                    "avg_download_mbps"
                ]
            ),
            float(
                preset[
                    "download_mbps"
                ]
            ),
        )

        scenario_upload = max(
            float(
                cell[
                    "avg_upload_mbps"
                ]
            ),
            float(
                preset[
                    "upload_mbps"
                ]
            ),
        )

        scenario_latency = min(
            float(
                cell[
                    "avg_latency_ms"
                ]
            ),
            float(
                preset[
                    "latency_ms"
                ]
            ),
        )


        scenario_gap = (
            simulate_connectivity_gap(
                scenario_download,
                scenario_upload,
                scenario_latency,
                download_reference,
                upload_reference,
                latency_reference,
            )
        )


        row = {
            "cell_id":
                cell[
                    "cell_id"
                ],

            "population":
                cell[
                    "population"
                ],

            "school_count":
                cell[
                    "school_count"
                ],

            "healthcare_count":
                cell[
                    "healthcare_count"
                ],

            "before_gap":
                cell[
                    "connectivity_gap_score"
                ],

            "after_gap":
                scenario_gap[
                    "connectivity_gap_score"
                ],
        }


        for mode in [
            "general",
            "education",
            "healthcare",
        ]:

            exposure_column = {
                "general":
                    "general_exposure",

                "education":
                    "education_exposure",

                "healthcare":
                    "healthcare_exposure_mode",
            }[
                mode
            ]

            after_score = (
                scenario_gap[
                    "connectivity_gap"
                ]
                * float(
                    cell[
                        exposure_column
                    ]
                )
                * 100
            )


            after_priority = (
                priority_from_score(
                    after_score,
                    thresholds[
                        mode
                    ],
                )
            )


            row[
                f"before_score_{mode}"
            ] = cell[
                f"score_{mode}"
            ]

            row[
                f"after_score_{mode}"
            ] = after_score

            row[
                f"before_priority_{mode}"
            ] = cell[
                f"priority_{mode}"
            ]

            row[
                f"after_priority_{mode}"
            ] = after_priority


        scenario_rows.append(
            row
        )


    scenario = pd.DataFrame(
        scenario_rows
    )


    # ----------------------------------------------
    # Preset-wide audit
    # ----------------------------------------------

    if (
        scenario[
            "after_gap"
        ]
        >
        scenario[
            "before_gap"
        ]
        + 1e-8
    ).any():

        raise ValueError(
            f"{preset_key} increased connectivity "
            "gap despite upgrade-only logic."
        )


    for mode in [
        "general",
        "education",
        "healthcare",
    ]:

        before_priority_col = (
            f"before_priority_{mode}"
        )

        after_priority_col = (
            f"after_priority_{mode}"
        )

        before_score_col = (
            f"before_score_{mode}"
        )

        after_score_col = (
            f"after_score_{mode}"
        )


        high_before = (
            scenario[
                before_priority_col
            ]
            == "High"
        )


        high_count = int(
            high_before.sum()
        )


        moved_out_high = (
            high_before
            &
            (
                scenario[
                    after_priority_col
                ]
                != "High"
            )
        )


        moved_to_low = (
            high_before
            &
            (
                scenario[
                    after_priority_col
                ]
                == "Low"
            )
        )


        mean_reduction = (
            scenario.loc[
                high_before,
                before_score_col,
            ]
            -
            scenario.loc[
                high_before,
                after_score_col,
            ]
        ).mean()


        median_reduction = (
            scenario.loc[
                high_before,
                before_score_col,
            ]
            -
            scenario.loc[
                high_before,
                after_score_col,
            ]
        ).median()


        context_population = (
            scenario.loc[
                moved_out_high,
                "population",
            ]
            .sum()
        )


        context_schools = int(
            scenario.loc[
                moved_out_high,
                "school_count",
            ]
            .sum()
        )


        context_healthcare = int(
            scenario.loc[
                moved_out_high,
                "healthcare_count",
            ]
            .sum()
        )


        result = {
            "preset_key":
                preset_key,

            "preset_label":
                preset["label"],

            "mode":
                mode,

            "high_before":
                high_count,

            "moved_out_of_high":
                int(
                    moved_out_high.sum()
                ),

            "moved_high_to_low":
                int(
                    moved_to_low.sum()
                ),

            "pct_high_moved_out":
                (
                    moved_out_high.sum()
                    / high_count
                    * 100
                    if high_count
                    else 0
                ),

            "mean_score_reduction_high":
                mean_reduction,

            "median_score_reduction_high":
                median_reduction,

            # Context only:
            # these are NOT claimed beneficiaries.
            "population_in_transition_cells":
                context_population,

            "schools_in_transition_cells":
                context_schools,

            "healthcare_in_transition_cells":
                context_healthcare,
        }


        results.append(
            result
        )


        print()
        print(
            mode.upper()
        )

        print(
            "High before:",
            high_count,
        )

        print(
            "Moved out of High:",
            int(
                moved_out_high.sum()
            ),
        )

        print(
            "Moved High -> Low:",
            int(
                moved_to_low.sum()
            ),
        )

        print(
            "% of High moved out:",
            round(
                result[
                    "pct_high_moved_out"
                ],
                1,
            ),
        )

        print(
            "Mean score reduction:",
            round(
                mean_reduction,
                2,
            ),
        )


# --------------------------------------------------
# 5. Save summary
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print()
print("=== SUMMARY TABLE ===")

print(
    results_df[
        [
            "preset_label",
            "mode",
            "high_before",
            "moved_out_of_high",
            "moved_high_to_low",
            "pct_high_moved_out",
            "mean_score_reduction_high",
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


print()
print("Saved:")
print(OUTPUT_PATH)

print()
print("=== PHASE 5.2B COMPLETE ===")