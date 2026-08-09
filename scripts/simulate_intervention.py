"""
Signal Lah - Phase 5.1
What-If Connectivity Intervention Simulator.

Example:

python scripts/simulate_intervention.py ^
    --cell KS_248 ^
    --download 150 ^
    --upload 30 ^
    --latency 20

The simulator changes connectivity only.
Population and essential-service exposure remain fixed.

For evidence-gap cells:
- a hypothetical scenario may still be evaluated when
  population/exposure data are available;
- however, no before-to-after improvement is claimed because
  baseline connectivity is unknown.
"""

from pathlib import Path
import argparse

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_PATH = Path(
    "data/processed/signal_lah_cells_v2.geojson"
)

MODEL_PATH = Path(
    "models/geoai_kmeans_k3.joblib"
)


# --------------------------------------------------
# Percentile mapping
#
# Reproduces the empirical percentile logic used by
# the transparent baseline model.
# --------------------------------------------------

def build_percentile_reference(series):

    values = pd.Series(
        series
    ).dropna().astype(float)

    ranks = values.rank(
        method="average",
        pct=True,
    )

    reference = pd.DataFrame(
        {
            "value": values.values,
            "rank": ranks.values,
        }
    )

    # If duplicate metric values exist, use their
    # average percentile rank.
    reference = (
        reference
        .groupby(
            "value",
            as_index=False,
        )
        ["rank"]
        .mean()
        .sort_values("value")
    )

    return reference


def percentile_for_value(
    value,
    reference,
):

    x = reference[
        "value"
    ].to_numpy(
        dtype=float
    )

    y = reference[
        "rank"
    ].to_numpy(
        dtype=float
    )

    # Values outside the observed range may still be
    # simulated. Better than the observed maximum maps
    # toward percentile 1; worse than minimum toward 0.
    result = np.interp(
        float(value),
        x,
        y,
        left=0.0,
        right=1.0,
    )

    return float(result)


# --------------------------------------------------
# Priority thresholds
# --------------------------------------------------

def get_priority_thresholds(
    data,
    score_column,
):

    scores = (
        data[
            score_column
        ]
        .dropna()
    )

    return {
        "q25": float(
            scores.quantile(0.25)
        ),
        "q75": float(
            scores.quantile(0.75)
        ),
    }


def priority_from_score(
    score,
    thresholds,
):

    if pd.isna(score):
        return "Unavailable"

    if score < thresholds["q25"]:
        return "Low"

    if score < thresholds["q75"]:
        return "Medium"

    return "High"


# --------------------------------------------------
# Connectivity-gap simulation
# --------------------------------------------------

def simulate_connectivity_gap(
    download,
    upload,
    latency,
    download_reference,
    upload_reference,
    latency_reference,
):

    download_rank = percentile_for_value(
        download,
        download_reference,
    )

    upload_rank = percentile_for_value(
        upload,
        upload_reference,
    )

    latency_rank = percentile_for_value(
        latency,
        latency_reference,
    )

    # Higher speeds are good, so invert.
    download_gap = (
        1.0 - download_rank
    )

    upload_gap = (
        1.0 - upload_rank
    )

    # Higher latency is bad, so do not invert.
    latency_gap = latency_rank

    connectivity_gap = np.mean(
        [
            download_gap,
            upload_gap,
            latency_gap,
        ]
    )

    return {
        "download_gap": float(
            download_gap
        ),
        "upload_gap": float(
            upload_gap
        ),
        "latency_gap": float(
            latency_gap
        ),
        "connectivity_gap":
            float(
                connectivity_gap
            ),
        "connectivity_gap_score":
            float(
                connectivity_gap * 100
            ),
    }


# --------------------------------------------------
# Impact calculation
# --------------------------------------------------

def simulate_impact_scores(
    cell,
    connectivity_gap,
):

    exposures = {
        "general":
            cell[
                "general_exposure"
            ],

        "education":
            cell[
                "education_exposure"
            ],

        "healthcare":
            cell[
                "healthcare_exposure_mode"
            ],
    }

    scores = {}

    for mode, exposure in exposures.items():

        if pd.isna(exposure):
            scores[
                mode
            ] = np.nan

        else:
            scores[
                mode
            ] = (
                connectivity_gap
                * float(exposure)
                * 100
            )

    return scores


# --------------------------------------------------
# GeoAI scenario prediction
# --------------------------------------------------

def predict_scenario_profile(
    cell,
    download,
    upload,
    latency,
    bundle,
):

    if pd.isna(
        cell["population"]
    ):
        return {
            "cluster_id": None,
            "profile": (
                "Unclassified - "
                "Population Evidence Gap"
            ),
            "nearest_distance": None,
            "second_distance": None,
            "assignment_margin": None,
        }

    features = pd.DataFrame(
        [
            {
                "avg_download_mbps":
                    float(download),

                "avg_upload_mbps":
                    float(upload),

                "avg_latency_ms":
                    float(latency),

                "log_population":
                    np.log1p(
                        float(
                            cell[
                                "population"
                            ]
                        )
                    ),

                "school_count":
                    float(
                        cell[
                            "school_count"
                        ]
                    ),

                "healthcare_count":
                    float(
                        cell[
                            "healthcare_count"
                        ]
                    ),
            }
        ],
        columns=bundle["features"],
    )

    scaled = (
        bundle["scaler"]
        .transform(
            features
        )
    )

    model = bundle["model"]

    cluster_id = int(
        model.predict(
            scaled
        )[0]
    )

    distances = (
        model.transform(
            scaled
        )[0]
    )

    sorted_distances = np.sort(
        distances
    )

    nearest = float(
        sorted_distances[0]
    )

    second = float(
        sorted_distances[1]
    )

    # This is NOT probability.
    # It is simply a geometric separation measure.
    if second > 0:

        margin = (
            second - nearest
        ) / second

    else:

        margin = 0.0

    profile = (
        bundle[
            "profile_names"
        ][
            cluster_id
        ]
    )

    return {
        "cluster_id":
            cluster_id,

        "profile":
            profile,

        "nearest_distance":
            nearest,

        "second_distance":
            second,

        "assignment_margin":
            float(margin),
    }


# --------------------------------------------------
# Main simulation
# --------------------------------------------------

def simulate(
    cell_id,
    proposed_download,
    proposed_upload,
    proposed_latency,
):

    # ----------------------------------------------
    # Validate proposed values
    # ----------------------------------------------

    if proposed_download < 0:
        raise ValueError(
            "Download speed cannot be negative."
        )

    if proposed_upload < 0:
        raise ValueError(
            "Upload speed cannot be negative."
        )

    if proposed_latency <= 0:
        raise ValueError(
            "Latency must be greater than zero."
        )


    # ----------------------------------------------
    # Load data/model
    # ----------------------------------------------

    data = gpd.read_file(
        DATA_PATH
    )

    bundle = joblib.load(
        MODEL_PATH
    )


    matches = data[
        data["cell_id"]
        == cell_id
    ]

    if len(matches) != 1:

        raise ValueError(
            f"Expected exactly one cell '{cell_id}', "
            f"found {len(matches)}."
        )

    cell = matches.iloc[0]


    # ----------------------------------------------
    # Build empirical references from observed cells
    # ----------------------------------------------

    observed = data[
        data[
            "network_data_present"
        ]
        == True
    ]

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


    # ----------------------------------------------
    # Simulate proposed connectivity
    # ----------------------------------------------

    scenario_gap = (
        simulate_connectivity_gap(
            proposed_download,
            proposed_upload,
            proposed_latency,
            download_reference,
            upload_reference,
            latency_reference,
        )
    )


    scenario_scores = (
        simulate_impact_scores(
            cell,
            scenario_gap[
                "connectivity_gap"
            ],
        )
    )


    # ----------------------------------------------
    # Priority thresholds remain anchored to baseline
    # ----------------------------------------------

    thresholds = {}

    scenario_priorities = {}

    for mode in [
        "general",
        "education",
        "healthcare",
    ]:

        score_column = (
            f"score_{mode}"
        )

        thresholds[
            mode
        ] = get_priority_thresholds(
            data,
            score_column,
        )

        scenario_priorities[
            mode
        ] = priority_from_score(
            scenario_scores[
                mode
            ],
            thresholds[
                mode
            ],
        )


    # ----------------------------------------------
    # AI scenario profile
    # ----------------------------------------------

    ai_scenario = (
        predict_scenario_profile(
            cell,
            proposed_download,
            proposed_upload,
            proposed_latency,
            bundle,
        )
    )


    # ----------------------------------------------
    # Baseline state
    # ----------------------------------------------

    baseline_available = bool(
        cell[
            "network_data_present"
        ]
    )


    print(
        "=== SIGNAL LAH WHAT-IF SIMULATOR ==="
    )

    print()
    print(
        "Cell:",
        cell_id,
    )

    print(
        "Population:",
        (
            round(
                float(
                    cell[
                        "population"
                    ]
                ),
                2,
            )
            if not pd.isna(
                cell[
                    "population"
                ]
            )
            else "Unknown"
        ),
    )

    print(
        "Schools:",
        int(
            cell[
                "school_count"
            ]
        ),
    )

    print(
        "Healthcare facilities:",
        int(
            cell[
                "healthcare_count"
            ]
        ),
    )


    # ----------------------------------------------
    # Connectivity comparison
    # ----------------------------------------------

    print()
    print(
        "--- CONNECTIVITY ---"
    )

    if baseline_available:

        print(
            "Download:",
            round(
                float(
                    cell[
                        "avg_download_mbps"
                    ]
                ),
                2,
            ),
            "->",
            round(
                proposed_download,
                2,
            ),
            "Mbps",
        )

        print(
            "Upload:",
            round(
                float(
                    cell[
                        "avg_upload_mbps"
                    ]
                ),
                2,
            ),
            "->",
            round(
                proposed_upload,
                2,
            ),
            "Mbps",
        )

        print(
            "Latency:",
            round(
                float(
                    cell[
                        "avg_latency_ms"
                    ]
                ),
                2,
            ),
            "->",
            round(
                proposed_latency,
                2,
            ),
            "ms",
        )

    else:

        print(
            "Baseline connectivity: "
            "Evidence Gap"
        )

        print(
            "Proposed download:",
            round(
                proposed_download,
                2,
            ),
            "Mbps",
        )

        print(
            "Proposed upload:",
            round(
                proposed_upload,
                2,
            ),
            "Mbps",
        )

        print(
            "Proposed latency:",
            round(
                proposed_latency,
                2,
            ),
            "ms",
        )


    # ----------------------------------------------
    # Gap comparison
    # ----------------------------------------------

    print()
    print(
        "--- CONNECTIVITY GAP ---"
    )

    if baseline_available:

        baseline_gap = float(
            cell[
                "connectivity_gap_score"
            ]
        )

        after_gap = (
            scenario_gap[
                "connectivity_gap_score"
            ]
        )

        print(
            "Before:",
            round(
                baseline_gap,
                2,
            ),
        )

        print(
            "After:",
            round(
                after_gap,
                2,
            ),
        )

        print(
            "Reduction:",
            round(
                baseline_gap
                - after_gap,
                2,
            ),
            "points",
        )

    else:

        print(
            "Before: Unknown"
        )

        print(
            "Scenario:",
            round(
                scenario_gap[
                    "connectivity_gap_score"
                ],
                2,
            ),
        )


    # ----------------------------------------------
    # Impact modes
    # ----------------------------------------------

    print()
    print(
        "--- IMPACT RESULTS ---"
    )

    for mode in [
        "general",
        "education",
        "healthcare",
    ]:

        title = mode.title()

        after_score = (
            scenario_scores[
                mode
            ]
        )

        after_priority = (
            scenario_priorities[
                mode
            ]
        )

        print()
        print(title)

        if baseline_available:

            before_score = float(
                cell[
                    f"score_{mode}"
                ]
            )

            before_priority = (
                cell[
                    f"priority_{mode}"
                ]
            )

            print(
                "Score:",
                round(
                    before_score,
                    2,
                ),
                "->",
                round(
                    after_score,
                    2,
                ),
            )

            print(
                "Priority:",
                before_priority,
                "->",
                after_priority,
            )

            print(
                "Impact reduction:",
                round(
                    before_score
                    - after_score,
                    2,
                ),
                "points",
            )

        else:

            print(
                "Baseline score: Unknown"
            )

            print(
                "Scenario score:",
                (
                    round(
                        after_score,
                        2,
                    )
                    if not pd.isna(
                        after_score
                    )
                    else "Unavailable"
                ),
            )

            print(
                "Scenario priority:",
                after_priority,
            )


    # ----------------------------------------------
    # GeoAI profile comparison
    # ----------------------------------------------

    print()
    print(
        "--- GEOAI PROFILE ---"
    )

    if baseline_available:

        print(
            "Before:",
            cell[
                "ai_profile"
            ],
        )

    else:

        print(
            "Before:",
            "Unclassified - Evidence Gap",
        )

    print(
        "Scenario:",
        ai_scenario[
            "profile"
        ],
    )

    if (
        ai_scenario[
            "assignment_margin"
        ]
        is not None
    ):

        print(
            "Scenario assignment margin:",
            round(
                ai_scenario[
                    "assignment_margin"
                ],
                3,
            ),
        )

        print(
            "(geometric cluster separation, "
            "not a probability)"
        )


    # ----------------------------------------------
    # Decision-support summary
    # ----------------------------------------------

    print()
    print(
        "--- DECISION SUPPORT ---"
    )

    if baseline_available:

        general_change = (
            float(
                cell[
                    "score_general"
                ]
            )
            - scenario_scores[
                "general"
            ]
        )

        if general_change > 0.01:

            print(
                "The proposed intervention reduces "
                "the estimated connectivity impact."
            )

        elif general_change < -0.01:

            print(
                "The proposed scenario increases "
                "estimated connectivity impact."
            )

        else:

            print(
                "The proposed scenario produces "
                "little change in estimated impact."
            )

    else:

        print(
            "Baseline network performance is unknown, "
            "so Signal Lah does not claim an improvement "
            "amount for this cell."
        )


    print()
    print(
        "=== SIMULATION COMPLETE ==="
    )


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Signal Lah what-if connectivity simulator"
        )
    )

    parser.add_argument(
        "--cell",
        required=True,
        help="Signal Lah cell ID, e.g. KS_248",
    )

    parser.add_argument(
        "--download",
        required=True,
        type=float,
        help="Proposed download speed in Mbps",
    )

    parser.add_argument(
        "--upload",
        required=True,
        type=float,
        help="Proposed upload speed in Mbps",
    )

    parser.add_argument(
        "--latency",
        required=True,
        type=float,
        help="Proposed latency in milliseconds",
    )

    args = parser.parse_args()

    simulate(
        cell_id=args.cell,
        proposed_download=args.download,
        proposed_upload=args.upload,
        proposed_latency=args.latency,
    )