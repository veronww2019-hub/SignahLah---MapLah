"""
Signal Lah - Phase 2.2
Aggregate Ookla mobile-performance tiles into the Signal Lah analysis grid.

Inputs:
    data/raw/ookla/extracted/gps_mobile_tiles.shp
    data/raw/boundary/extracted/geoBoundaries-MYS-ADM2.geojson
    data/interim/analysis_grid.geojson

Output:
    data/interim/grid_with_network.geojson
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OOKLA_PATH = Path(
    "data/raw/ookla/extracted/gps_mobile_tiles.shp"
)

BOUNDARY_PATH = Path(
    "data/raw/boundary/extracted/geoBoundaries-MYS-ADM2.geojson"
)

GRID_PATH = Path(
    "data/interim/analysis_grid.geojson"
)

OUTPUT_PATH = Path(
    "data/interim/grid_with_network.geojson"
)

DISTRICT_NAME = "Kuala Selangor"

WORKING_CRS = "EPSG:32647"


# --------------------------------------------------
# 1. Load district and analysis grid
# --------------------------------------------------

boundary = gpd.read_file(BOUNDARY_PATH)

district = boundary[
    boundary["shapeName"] == DISTRICT_NAME
].copy()

if len(district) != 1:
    raise ValueError(
        f"Expected one {DISTRICT_NAME} boundary, found {len(district)}"
    )

district = district.to_crs("EPSG:4326")

grid = gpd.read_file(GRID_PATH)

if grid.crs is None:
    raise ValueError("Analysis grid has no CRS.")

grid = grid.to_crs("EPSG:4326")

print("=== SIGNAL LAH OOKLA AGGREGATION ===")
print()
print("Analysis cells:", len(grid))


# --------------------------------------------------
# 2. Read Ookla tiles in Kuala Selangor bbox
# --------------------------------------------------

bbox = tuple(district.total_bounds)

ookla = gpd.read_file(
    OOKLA_PATH,
    bbox=bbox,
    engine="pyogrio",
)

print("Ookla tiles in district bbox:", len(ookla))


# --------------------------------------------------
# 3. Select tiles whose centres lie inside district
# --------------------------------------------------

ookla_utm = ookla.to_crs(WORKING_CRS)

district_utm = district.to_crs(WORKING_CRS)

tile_centres_utm = (
    ookla_utm.geometry.centroid
)

inside_mask = tile_centres_utm.within(
    district_utm.geometry.iloc[0]
)

ookla = ookla.loc[
    inside_mask.values
].copy()

tile_centres_utm = (
    tile_centres_utm.loc[
        inside_mask.values
    ]
)

print(
    "Ookla tile centres inside district:",
    len(ookla),
)

print(
    "Raw tests inside district:",
    int(ookla["tests"].sum()),
)


# --------------------------------------------------
# 4. Convert tile centres into point features
# --------------------------------------------------

centre_points = gpd.GeoDataFrame(
    ookla[
        [
            "quadkey",
            "avg_d_kbps",
            "avg_u_kbps",
            "avg_lat_ms",
            "tests",
            "devices",
        ]
    ].reset_index(drop=True),
    geometry=gpd.GeoSeries(
        tile_centres_utm.reset_index(drop=True),
        crs=WORKING_CRS,
    ),
    crs=WORKING_CRS,
)

centre_points = centre_points.to_crs(
    "EPSG:4326"
)


# --------------------------------------------------
# 5. Assign each Ookla tile centre to one grid cell
# --------------------------------------------------

joined = gpd.sjoin(
    centre_points,
    grid[
        [
            "cell_id",
            "geometry",
        ]
    ],
    how="left",
    predicate="within",
)

unassigned = joined[
    joined["cell_id"].isna()
].copy()

if len(unassigned) > 0:
    print()
    print(
        "WARNING: unassigned tile centres:",
        len(unassigned),
    )

    print(
        "Tests in unassigned tiles:",
        int(unassigned["tests"].sum()),
    )

else:
    print(
        "All tile centres assigned to grid: True"
    )


# --------------------------------------------------
# 6. Ensure numeric fields
# --------------------------------------------------

numeric_columns = [
    "avg_d_kbps",
    "avg_u_kbps",
    "avg_lat_ms",
    "tests",
    "devices",
]

for column in numeric_columns:
    joined[column] = pd.to_numeric(
        joined[column],
        errors="coerce",
    )


# Only aggregate successfully assigned records.
assigned = joined[
    joined["cell_id"].notna()
].copy()


# --------------------------------------------------
# 7. Build weighted components
# --------------------------------------------------

assigned["download_weighted"] = (
    assigned["avg_d_kbps"]
    * assigned["tests"]
)

assigned["upload_weighted"] = (
    assigned["avg_u_kbps"]
    * assigned["tests"]
)

assigned["latency_weighted"] = (
    assigned["avg_lat_ms"]
    * assigned["tests"]
)


# --------------------------------------------------
# 8. Aggregate by Signal Lah cell
# --------------------------------------------------

aggregated = assigned.groupby(
    "cell_id",
    as_index=False,
).agg(
    network_tiles=(
        "quadkey",
        "count",
    ),
    network_tests=(
        "tests",
        "sum",
    ),
    tile_device_count_sum=(
        "devices",
        "sum",
    ),
    download_weighted=(
        "download_weighted",
        "sum",
    ),
    upload_weighted=(
        "upload_weighted",
        "sum",
    ),
    latency_weighted=(
        "latency_weighted",
        "sum",
    ),
)


# --------------------------------------------------
# 9. Calculate test-weighted network performance
# --------------------------------------------------

aggregated["avg_download_mbps"] = (
    aggregated["download_weighted"]
    / aggregated["network_tests"]
    / 1000
)

aggregated["avg_upload_mbps"] = (
    aggregated["upload_weighted"]
    / aggregated["network_tests"]
    / 1000
)

aggregated["avg_latency_ms"] = (
    aggregated["latency_weighted"]
    / aggregated["network_tests"]
)

aggregated = aggregated.drop(
    columns=[
        "download_weighted",
        "upload_weighted",
        "latency_weighted",
    ]
)


# --------------------------------------------------
# 10. Join results back to all grid cells
# --------------------------------------------------

output = grid.merge(
    aggregated,
    on="cell_id",
    how="left",
)

count_columns = [
    "network_tiles",
    "network_tests",
    "tile_device_count_sum",
]

for column in count_columns:
    output[column] = (
        output[column]
        .fillna(0)
        .astype(int)
    )

output["network_data_present"] = (
    output["network_tests"] > 0
)


# --------------------------------------------------
# 11. Validation
# --------------------------------------------------

raw_tests = int(
    ookla["tests"].sum()
)

assigned_tests = int(
    assigned["tests"].sum()
)

output_tests = int(
    output["network_tests"].sum()
)

cells_with_data = int(
    output["network_data_present"].sum()
)

cells_without_data = (
    len(output) - cells_with_data
)

print()
print("--- NETWORK AGGREGATION AUDIT ---")

print(
    "Raw district tiles:",
    len(ookla),
)

print(
    "Assigned tiles:",
    len(assigned),
)

print(
    "Raw district tests:",
    raw_tests,
)

print(
    "Assigned tests:",
    assigned_tests,
)

print(
    "Output tests:",
    output_tests,
)

print(
    "Tests preserved:",
    raw_tests == output_tests,
)

print()
print(
    "Cells with network data:",
    cells_with_data,
)

print(
    "Cells without network data:",
    cells_without_data,
)

print(
    "Network-data coverage:",
    round(
        cells_with_data / len(output) * 100,
        2,
    ),
    "%",
)

if cells_with_data > 0:
    measured = output[
        output["network_data_present"]
    ]

    print()
    print("--- CELL PERFORMANCE SUMMARY ---")

    print(
        "Median download:",
        round(
            measured["avg_download_mbps"].median(),
            2,
        ),
        "Mbps",
    )

    print(
        "Median upload:",
        round(
            measured["avg_upload_mbps"].median(),
            2,
        ),
        "Mbps",
    )

    print(
        "Median latency:",
        round(
            measured["avg_latency_ms"].median(),
            2,
        ),
        "ms",
    )

    print(
        "Median tests per measured cell:",
        round(
            measured["network_tests"].median(),
            1,
        ),
    )

    print(
        "Maximum tests in one cell:",
        int(
            measured["network_tests"].max()
        ),
    )


# --------------------------------------------------
# 12. Hard validation
# --------------------------------------------------

if raw_tests != output_tests:
    raise ValueError(
        "Network test count was not preserved during aggregation."
    )

if output["cell_id"].duplicated().any():
    raise ValueError(
        "Duplicate cell IDs detected after network aggregation."
    )


# --------------------------------------------------
# 13. Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.to_file(
    OUTPUT_PATH,
    driver="GeoJSON",
)

print()
print("Saved:")
print(OUTPUT_PATH)

print()
print("=== COMPLETE ===")