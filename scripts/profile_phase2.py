"""
Signal Lah - Phase 3.1
Profile the completed Phase 2 dataset before designing
the transparent connectivity-impact model.
"""

import geopandas as gpd
import numpy as np


DATA_PATH = "data/interim/grid_complete_raw.geojson"

g = gpd.read_file(DATA_PATH)

network = g["network_data_present"]
population_data = g["population_data_present"]

measured = g[network].copy()
pop_valid = g[population_data].copy()


def quantiles(series):
    q = series.dropna().quantile(
        [0.10, 0.25, 0.50, 0.75, 0.90]
    )

    return (
        f"P10={q.loc[0.10]:.2f}, "
        f"P25={q.loc[0.25]:.2f}, "
        f"P50={q.loc[0.50]:.2f}, "
        f"P75={q.loc[0.75]:.2f}, "
        f"P90={q.loc[0.90]:.2f}"
    )


print("=== SIGNAL LAH PHASE 3 DATA PROFILE ===")

print()
print("--- CORE DATA ---")
print("Analysis cells:", len(g))
print("Network measured:", int(network.sum()))
print("Network unmeasured:", int((~network).sum()))
print("Population data:", int(population_data.sum()))
print("Population NoData:", int((~population_data).sum()))
print("Schools:", int(g["school_count"].sum()))
print("Healthcare:", int(g["healthcare_count"].sum()))


print()
print("--- NETWORK DISTRIBUTIONS ---")

print(
    "Download Mbps:",
    quantiles(measured["avg_download_mbps"]),
)

print(
    "Upload Mbps:",
    quantiles(measured["avg_upload_mbps"]),
)

print(
    "Latency ms:",
    quantiles(measured["avg_latency_ms"]),
)

print(
    "Tests per measured cell:",
    quantiles(measured["network_tests"]),
)


# District-level test-weighted averages
weights = measured["network_tests"].astype(float)

weighted_download = np.average(
    measured["avg_download_mbps"],
    weights=weights,
)

weighted_upload = np.average(
    measured["avg_upload_mbps"],
    weights=weights,
)

weighted_latency = np.average(
    measured["avg_latency_ms"],
    weights=weights,
)

print()
print("--- TEST-WEIGHTED DISTRICT PERFORMANCE ---")
print("Download:", round(weighted_download, 2), "Mbps")
print("Upload:", round(weighted_upload, 2), "Mbps")
print("Latency:", round(weighted_latency, 2), "ms")


print()
print("--- POPULATION DISTRIBUTION ---")

print(
    "Population per valid cell:",
    quantiles(pop_valid["population"]),
)

print(
    "Total population:",
    round(g["population"].sum(), 2),
)


# --------------------------------------------------
# Data blind spots
# --------------------------------------------------

pop_no_network = (
    population_data
    & (~network)
)

school_no_network = (
    (g["school_count"] > 0)
    & (~network)
)

health_no_network = (
    (g["healthcare_count"] > 0)
    & (~network)
)

both_data = (
    population_data
    & network
)

network_no_population = (
    network
    & (~population_data)
)


print()
print("--- DATA COVERAGE OVERLAP ---")

print(
    "Cells with both network + population data:",
    int(both_data.sum()),
)

print(
    "Population-data cells WITHOUT network measurements:",
    int(pop_no_network.sum()),
)

print(
    "Population living in those unmeasured cells:",
    round(
        g.loc[
            pop_no_network,
            "population",
        ].sum(),
        2,
    ),
)

print(
    "School-containing cells WITHOUT network measurements:",
    int(school_no_network.sum()),
)

print(
    "Schools in unmeasured network cells:",
    int(
        g.loc[
            ~network,
            "school_count",
        ].sum()
    ),
)

print(
    "Healthcare-containing cells WITHOUT network measurements:",
    int(health_no_network.sum()),
)

print(
    "Healthcare facilities in unmeasured network cells:",
    int(
        g.loc[
            ~network,
            "healthcare_count",
        ].sum()
    ),
)

print(
    "Measured-network cells WITHOUT population data:",
    int(network_no_population.sum()),
)


# --------------------------------------------------
# Measurement density
# --------------------------------------------------

print()
print("--- NETWORK EVIDENCE DENSITY ---")

for limit in [1, 5, 10, 20, 50, 100]:

    count = int(
        (
            network
            & (g["network_tests"] < limit)
        ).sum()
    )

    print(
        f"Measured cells with fewer than {limit} tests:",
        count,
    )


# --------------------------------------------------
# Most important blind spots
# --------------------------------------------------

blind = g[
    pop_no_network
].copy()

blind = blind.sort_values(
    "population",
    ascending=False,
)

print()
print("--- TOP POPULATED CELLS WITHOUT NETWORK DATA ---")

if len(blind) == 0:
    print("None")

else:
    print(
        blind[
            [
                "cell_id",
                "population",
                "school_count",
                "hospital_count",
                "clinic_count",
                "healthcare_count",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


print()
print("--- FACILITY-RICH CELLS WITHOUT NETWORK DATA ---")

facility_blind = g[
    (~network)
    & (
        (g["school_count"] > 0)
        | (g["healthcare_count"] > 0)
    )
].copy()

facility_blind["essential_facilities"] = (
    facility_blind["school_count"]
    + facility_blind["healthcare_count"]
)

facility_blind = facility_blind.sort_values(
    [
        "essential_facilities",
        "population",
    ],
    ascending=[False, False],
)

if len(facility_blind) == 0:
    print("None")

else:
    print(
        facility_blind[
            [
                "cell_id",
                "population",
                "school_count",
                "hospital_count",
                "clinic_count",
                "healthcare_count",
                "essential_facilities",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )


print()
print("=== PROFILE COMPLETE ===")