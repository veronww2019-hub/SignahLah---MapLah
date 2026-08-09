"""
Signal Lah - Phase 5.2A
Generate empirical connectivity intervention presets.

Presets are based on the distribution of observed
Ookla measurements in the study area.

They are scenario targets, NOT regulatory standards.

Input:
    data/processed/signal_lah_cells_v2.geojson

Output:
    data/processed/scenario_presets.json
"""

from pathlib import Path
import json

import geopandas as gpd


INPUT_PATH = Path(
    "data/processed/signal_lah_cells_v2.geojson"
)

OUTPUT_PATH = Path(
    "data/processed/scenario_presets.json"
)


# --------------------------------------------------
# 1. Load observed cells
# --------------------------------------------------

g = gpd.read_file(INPUT_PATH)

observed = g[
    g["network_data_present"] == True
].copy()


print("=== SIGNAL LAH SCENARIO PRESETS ===")
print()
print("Observed cells:", len(observed))


# --------------------------------------------------
# 2. Quantiles
# --------------------------------------------------

download = observed[
    "avg_download_mbps"
]

upload = observed[
    "avg_upload_mbps"
]

latency = observed[
    "avg_latency_ms"
]


quantiles = {
    "download": {
        "p50": float(
            download.quantile(0.50)
        ),
        "p75": float(
            download.quantile(0.75)
        ),
        "p90": float(
            download.quantile(0.90)
        ),
    },

    "upload": {
        "p50": float(
            upload.quantile(0.50)
        ),
        "p75": float(
            upload.quantile(0.75)
        ),
        "p90": float(
            upload.quantile(0.90)
        ),
    },

    # Lower latency is better.
    "latency": {
        "p50": float(
            latency.quantile(0.50)
        ),
        "p25": float(
            latency.quantile(0.25)
        ),
        "p10": float(
            latency.quantile(0.10)
        ),
    },
}


# --------------------------------------------------
# 3. Build presets
# --------------------------------------------------

presets = {

    "typical_observed": {
        "label":
            "Typical Observed Target",

        "description": (
            "Hypothetical connectivity outcome based on "
            "approximately median observed performance "
            "across measured Kuala Selangor cells."
        ),

        "download_mbps":
            quantiles[
                "download"
            ]["p50"],

        "upload_mbps":
            quantiles[
                "upload"
            ]["p50"],

        "latency_ms":
            quantiles[
                "latency"
            ]["p50"],

        "basis": {
            "download":
                "P50 observed download",

            "upload":
                "P50 observed upload",

            "latency":
                "P50 observed latency",
        },
    },


    "strong_observed": {
        "label":
            "Strong Observed Target",

        "description": (
            "Hypothetical stronger connectivity outcome "
            "using upper-quartile observed speeds and "
            "lower-quartile observed latency."
        ),

        "download_mbps":
            quantiles[
                "download"
            ]["p75"],

        "upload_mbps":
            quantiles[
                "upload"
            ]["p75"],

        "latency_ms":
            quantiles[
                "latency"
            ]["p25"],

        "basis": {
            "download":
                "P75 observed download",

            "upload":
                "P75 observed upload",

            "latency":
                "P25 observed latency",
        },
    },


    "top_observed": {
        "label":
            "Top Observed Target",

        "description": (
            "Hypothetical high-performance connectivity "
            "outcome using P90 observed speeds and P10 "
            "observed latency."
        ),

        "download_mbps":
            quantiles[
                "download"
            ]["p90"],

        "upload_mbps":
            quantiles[
                "upload"
            ]["p90"],

        "latency_ms":
            quantiles[
                "latency"
            ]["p10"],

        "basis": {
            "download":
                "P90 observed download",

            "upload":
                "P90 observed upload",

            "latency":
                "P10 observed latency",
        },
    },
}


# --------------------------------------------------
# 4. Full output
# --------------------------------------------------

output = {
    "version": "1.0.0",

    "study_area":
        "Kuala Selangor",

    "method": (
        "Empirical scenario targets derived from "
        "observed cell-level Ookla performance. "
        "These are hypothetical what-if targets, "
        "not regulatory service standards."
    ),

    "upgrade_only": True,

    "observed_cells":
        len(observed),

    "quantiles":
        quantiles,

    "presets":
        presets,
}


# --------------------------------------------------
# 5. Print
# --------------------------------------------------

print()
print("--- GENERATED PRESETS ---")

for key, preset in presets.items():

    print()
    print(
        preset["label"]
    )

    print(
        "Download:",
        round(
            preset[
                "download_mbps"
            ],
            2,
        ),
        "Mbps",
    )

    print(
        "Upload:",
        round(
            preset[
                "upload_mbps"
            ],
            2,
        ),
        "Mbps",
    )

    print(
        "Latency:",
        round(
            preset[
                "latency_ms"
            ],
            2,
        ),
        "ms",
    )


# --------------------------------------------------
# 6. Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        output,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("Saved:")
print(OUTPUT_PATH)

print()
print("=== PHASE 5.2A COMPLETE ===")