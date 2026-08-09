"""
Signal Lah - Phase 5.3C
Audit GeoAI profile changes under the empirical presets.

Output:
    data/interim/scenario_geoai_shifts.csv
"""

from pathlib import Path
import json

import pandas as pd

from simulator_core import (
    SignalLahSimulator,
)


PRESET_PATH = Path(
    "data/processed/scenario_presets.json"
)

OUTPUT_PATH = Path(
    "data/interim/scenario_geoai_shifts.csv"
)


simulator = (
    SignalLahSimulator()
)

preset_data = json.loads(
    PRESET_PATH.read_text(
        encoding="utf-8"
    )
)

presets = preset_data[
    "presets"
]


observed = simulator.data[
    simulator.data[
        "network_data_present"
    ]
    == True
]


print(
    "=== SIGNAL LAH GEOAI SCENARIO SHIFT AUDIT ==="
)

print()
print(
    "Observed cells:",
    len(observed),
)


records = []


for preset_key, preset in presets.items():

    print()
    print(
        "========================================"
    )

    print(
        preset[
            "label"
        ]
    )

    print(
        "========================================"
    )


    for cell_id in observed[
        "cell_id"
    ]:

        result = simulator.simulate(
            cell_id=cell_id,

            download=preset[
                "download_mbps"
            ],

            upload=preset[
                "upload_mbps"
            ],

            latency=preset[
                "latency_ms"
            ],

            upgrade_only=True,
        )


        records.append(
            {
                "preset_key":
                    preset_key,

                "preset_label":
                    preset[
                        "label"
                    ],

                "cell_id":
                    cell_id,

                "before_profile":
                    result[
                        "baseline"
                    ][
                        "ai_profile"
                    ],

                "after_profile":
                    result[
                        "geoai"
                    ][
                        "scenario_profile"
                    ],

                "profile_changed":
                    result[
                        "delta"
                    ][
                        "geoai_profile_changed"
                    ],

                "assignment_margin":
                    result[
                        "geoai"
                    ][
                        "assignment_margin"
                    ],

                "before_gap":
                    result[
                        "baseline"
                    ][
                        "connectivity_gap_score"
                    ],

                "after_gap":
                    result[
                        "scenario"
                    ][
                        "connectivity_gap_score"
                    ],
            }
        )


results = pd.DataFrame(
    records
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

for preset_key, preset in presets.items():

    subset = results[
        results[
            "preset_key"
        ]
        == preset_key
    ]


    changed = int(
        subset[
            "profile_changed"
        ].sum()
    )


    print()
    print(
        preset[
            "label"
        ]
    )

    print(
        "Profiles changed:",
        changed,
        "/",
        len(subset),
    )

    print(
        "Percentage:",
        round(
            changed
            / len(subset)
            * 100,
            1,
        ),
        "%",
    )


    transitions = (
        subset[
            subset[
                "profile_changed"
            ]
        ]
        .groupby(
            [
                "before_profile",
                "after_profile",
            ]
        )
        .size()
        .reset_index(
            name="cells"
        )
        .sort_values(
            "cells",
            ascending=False,
        )
    )


    print()
    print(
        "PROFILE TRANSITIONS"
    )


    if len(transitions) == 0:

        print(
            "No profile changes."
        )

    else:

        print(
            transitions
            .to_string(
                index=False
            )
        )


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    OUTPUT_PATH,
    index=False,
)


print()
print("Saved:")
print(OUTPUT_PATH)

print()
print(
    "=== PHASE 5.3C COMPLETE ==="
)