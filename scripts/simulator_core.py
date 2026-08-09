"""
Signal Lah - Phase 5.3
Reusable structured what-if simulation engine.

This module produces machine-readable simulation results
for later use by the web dashboard and validation scripts.
"""

from pathlib import Path
import argparse
import json

import geopandas as gpd
import joblib
import pandas as pd

from simulate_intervention import (
    build_percentile_reference,
    simulate_connectivity_gap,
    simulate_impact_scores,
    get_priority_thresholds,
    priority_from_score,
    predict_scenario_profile,
)


DATA_PATH = Path(
    "data/processed/signal_lah_cells_v2.geojson"
)

MODEL_PATH = Path(
    "models/geoai_kmeans_k3.joblib"
)


class SignalLahSimulator:

    def __init__(self):

        self.data = gpd.read_file(
            DATA_PATH
        )

        self.bundle = joblib.load(
            MODEL_PATH
        )

        self.observed = self.data[
            self.data[
                "network_data_present"
            ]
            == True
        ].copy()

        self.download_reference = (
            build_percentile_reference(
                self.observed[
                    "avg_download_mbps"
                ]
            )
        )

        self.upload_reference = (
            build_percentile_reference(
                self.observed[
                    "avg_upload_mbps"
                ]
            )
        )

        self.latency_reference = (
            build_percentile_reference(
                self.observed[
                    "avg_latency_ms"
                ]
            )
        )

        self.thresholds = {}

        for mode in [
            "general",
            "education",
            "healthcare",
        ]:

            self.thresholds[mode] = (
                get_priority_thresholds(
                    self.data,
                    f"score_{mode}",
                )
            )


    def simulate(
        self,
        cell_id,
        download,
        upload,
        latency,
        upgrade_only=False,
    ):

        if download < 0:
            raise ValueError(
                "Download cannot be negative."
            )

        if upload < 0:
            raise ValueError(
                "Upload cannot be negative."
            )

        if latency <= 0:
            raise ValueError(
                "Latency must be greater than zero."
            )


        matches = self.data[
            self.data[
                "cell_id"
            ]
            == cell_id
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected one cell '{cell_id}', "
                f"found {len(matches)}."
            )

        cell = matches.iloc[0]

        baseline_available = bool(
            cell[
                "network_data_present"
            ]
        )


        # ------------------------------------------
        # Requested scenario
        # ------------------------------------------

        requested_download = float(
            download
        )

        requested_upload = float(
            upload
        )

        requested_latency = float(
            latency
        )


        # ------------------------------------------
        # Upgrade-only logic
        # ------------------------------------------

        applied_download = (
            requested_download
        )

        applied_upload = (
            requested_upload
        )

        applied_latency = (
            requested_latency
        )


        if (
            upgrade_only
            and baseline_available
        ):

            applied_download = max(
                float(
                    cell[
                        "avg_download_mbps"
                    ]
                ),
                requested_download,
            )

            applied_upload = max(
                float(
                    cell[
                        "avg_upload_mbps"
                    ]
                ),
                requested_upload,
            )

            applied_latency = min(
                float(
                    cell[
                        "avg_latency_ms"
                    ]
                ),
                requested_latency,
            )


        # ------------------------------------------
        # Scenario connectivity gap
        # ------------------------------------------

        gap = simulate_connectivity_gap(
            applied_download,
            applied_upload,
            applied_latency,
            self.download_reference,
            self.upload_reference,
            self.latency_reference,
        )


        # ------------------------------------------
        # Scenario impact scores
        # ------------------------------------------

        scores = simulate_impact_scores(
            cell,
            gap[
                "connectivity_gap"
            ],
        )


        priorities = {}

        for mode in [
            "general",
            "education",
            "healthcare",
        ]:

            priorities[mode] = (
                priority_from_score(
                    scores[mode],
                    self.thresholds[
                        mode
                    ],
                )
            )


        # ------------------------------------------
        # Scenario GeoAI profile
        # ------------------------------------------

        scenario_ai = (
            predict_scenario_profile(
                cell,
                applied_download,
                applied_upload,
                applied_latency,
                self.bundle,
            )
        )


        # ------------------------------------------
        # Context
        # ------------------------------------------

        population = (
            None
            if pd.isna(
                cell["population"]
            )
            else float(
                cell["population"]
            )
        )


        result = {

            "cell_id":
                cell_id,

            "context": {
                "population":
                    population,

                "school_count":
                    int(
                        cell[
                            "school_count"
                        ]
                    ),

                "healthcare_count":
                    int(
                        cell[
                            "healthcare_count"
                        ]
                    ),

                "hospital_count":
                    int(
                        cell[
                            "hospital_count"
                        ]
                    ),

                "clinic_count":
                    int(
                        cell[
                            "clinic_count"
                        ]
                    ),
            },


            "baseline_available":
                baseline_available,


            "requested_scenario": {
                "download_mbps":
                    requested_download,

                "upload_mbps":
                    requested_upload,

                "latency_ms":
                    requested_latency,

                "upgrade_only":
                    bool(
                        upgrade_only
                    ),
            },


            "applied_scenario": {
                "download_mbps":
                    applied_download,

                "upload_mbps":
                    applied_upload,

                "latency_ms":
                    applied_latency,
            },


            "scenario": {

                "connectivity_gap_score":
                    float(
                        gap[
                            "connectivity_gap_score"
                        ]
                    ),

                "general": {
                    "score":
                        (
                            None
                            if pd.isna(
                                scores[
                                    "general"
                                ]
                            )
                            else float(
                                scores[
                                    "general"
                                ]
                            )
                        ),

                    "priority":
                        priorities[
                            "general"
                        ],
                },

                "education": {
                    "score":
                        (
                            None
                            if pd.isna(
                                scores[
                                    "education"
                                ]
                            )
                            else float(
                                scores[
                                    "education"
                                ]
                            )
                        ),

                    "priority":
                        priorities[
                            "education"
                        ],
                },

                "healthcare": {
                    "score":
                        (
                            None
                            if pd.isna(
                                scores[
                                    "healthcare"
                                ]
                            )
                            else float(
                                scores[
                                    "healthcare"
                                ]
                            )
                        ),

                    "priority":
                        priorities[
                            "healthcare"
                        ],
                },
            },


            "geoai": {
                "scenario_cluster_id":
                    scenario_ai[
                        "cluster_id"
                    ],

                "scenario_profile":
                    scenario_ai[
                        "profile"
                    ],

                "assignment_margin":
                    scenario_ai[
                        "assignment_margin"
                    ],

                "assignment_margin_is_probability":
                    False,
            },
        }


        # ------------------------------------------
        # Baseline + changes
        # ------------------------------------------

        if baseline_available:

            result[
                "baseline"
            ] = {

                "download_mbps":
                    float(
                        cell[
                            "avg_download_mbps"
                        ]
                    ),

                "upload_mbps":
                    float(
                        cell[
                            "avg_upload_mbps"
                        ]
                    ),

                "latency_ms":
                    float(
                        cell[
                            "avg_latency_ms"
                        ]
                    ),

                "connectivity_gap_score":
                    float(
                        cell[
                            "connectivity_gap_score"
                        ]
                    ),

                "general": {
                    "score":
                        float(
                            cell[
                                "score_general"
                            ]
                        ),

                    "priority":
                        cell[
                            "priority_general"
                        ],
                },

                "education": {
                    "score":
                        float(
                            cell[
                                "score_education"
                            ]
                        ),

                    "priority":
                        cell[
                            "priority_education"
                        ],
                },

                "healthcare": {
                    "score":
                        float(
                            cell[
                                "score_healthcare"
                            ]
                        ),

                    "priority":
                        cell[
                            "priority_healthcare"
                        ],
                },

                "ai_cluster_id":
                    int(
                        cell[
                            "ai_cluster_id"
                        ]
                    ),

                "ai_profile":
                    cell[
                        "ai_profile"
                    ],
            }


            result[
                "delta"
            ] = {

                "connectivity_gap_reduction":
                    float(
                        cell[
                            "connectivity_gap_score"
                        ]
                        -
                        gap[
                            "connectivity_gap_score"
                        ]
                    ),

                "general_score_reduction":
                    float(
                        cell[
                            "score_general"
                        ]
                        -
                        scores[
                            "general"
                        ]
                    ),

                "education_score_reduction":
                    float(
                        cell[
                            "score_education"
                        ]
                        -
                        scores[
                            "education"
                        ]
                    ),

                "healthcare_score_reduction":
                    float(
                        cell[
                            "score_healthcare"
                        ]
                        -
                        scores[
                            "healthcare"
                        ]
                    ),

                "general_priority_changed":
                    bool(
                        cell[
                            "priority_general"
                        ]
                        !=
                        priorities[
                            "general"
                        ]
                    ),

                "education_priority_changed":
                    bool(
                        cell[
                            "priority_education"
                        ]
                        !=
                        priorities[
                            "education"
                        ]
                    ),

                "healthcare_priority_changed":
                    bool(
                        cell[
                            "priority_healthcare"
                        ]
                        !=
                        priorities[
                            "healthcare"
                        ]
                    ),

                "geoai_profile_changed":
                    bool(
                        cell[
                            "ai_profile"
                        ]
                        !=
                        scenario_ai[
                            "profile"
                        ]
                    ),
            }

        else:

            result[
                "baseline"
            ] = None

            result[
                "delta"
            ] = None


        return result


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cell",
        required=True,
    )

    parser.add_argument(
        "--download",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--upload",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--latency",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--upgrade-only",
        action="store_true",
    )

    args = parser.parse_args()

    simulator = (
        SignalLahSimulator()
    )

    result = simulator.simulate(
        cell_id=args.cell,
        download=args.download,
        upload=args.upload,
        latency=args.latency,
        upgrade_only=args.upgrade_only,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )