"""
Signal Lah - Phase 2.3
Aggregate the population raster into the Signal Lah analysis grid.

Method:
- read only the raster window around Kuala Selangor
- use population pixel centres
- retain only valid pixel centres inside Kuala Selangor
- assign each population pixel to exactly one analysis cell
- sum population by cell

Inputs:
    data/raw/population/mys_pop_2025_CN_100m_R2025A_v1.tif
    data/raw/boundary/extracted/geoBoundaries-MYS-ADM2.geojson
    data/interim/grid_with_network.geojson

Output:
    data/interim/grid_with_network_population.geojson
"""

from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import tifffile
import zarr


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RASTER_PATH = Path(
    "data/raw/population/mys_pop_2025_CN_100m_R2025A_v1.tif"
)

BOUNDARY_PATH = Path(
    "data/raw/boundary/extracted/geoBoundaries-MYS-ADM2.geojson"
)

GRID_PATH = Path(
    "data/interim/grid_with_network.geojson"
)

OUTPUT_PATH = Path(
    "data/interim/grid_with_network_population.geojson"
)

DISTRICT_NAME = "Kuala Selangor"


# --------------------------------------------------
# 1. Load boundary and current grid
# --------------------------------------------------

boundary = gpd.read_file(BOUNDARY_PATH)

district = boundary[
    boundary["shapeName"] == DISTRICT_NAME
].copy()

if len(district) != 1:
    raise ValueError(
        f"Expected exactly one {DISTRICT_NAME} boundary, "
        f"found {len(district)}"
    )

district = district.to_crs("EPSG:4326")
district_geom = district.geometry.iloc[0]

grid = gpd.read_file(GRID_PATH)

if grid.crs is None:
    raise ValueError("Grid has no CRS.")

grid = grid.to_crs("EPSG:4326")

print("=== SIGNAL LAH POPULATION AGGREGATION ===")
print()
print("Analysis cells:", len(grid))


# --------------------------------------------------
# 2. Open GeoTIFF and read its georeferencing
# --------------------------------------------------

tif = tifffile.TiffFile(RASTER_PATH)

page = tif.pages[0]

metadata = tif.geotiff_metadata

pixel_scale = metadata["ModelPixelScale"]
tiepoint = metadata["ModelTiepoint"]

pixel_width = float(pixel_scale[0])
pixel_height = float(pixel_scale[1])

origin_x = float(tiepoint[3])
origin_y = float(tiepoint[4])

height, width = page.shape

nodata = float(
    page.tags["GDAL_NODATA"].value
)

print("Raster shape:", page.shape)
print("Raster dtype:", page.dtype)
print("Pixel width:", pixel_width)
print("Pixel height:", pixel_height)
print("NoData:", nodata)


# --------------------------------------------------
# 3. Calculate raster window around district
# --------------------------------------------------

minx, miny, maxx, maxy = district.total_bounds

col_start = max(
    0,
    math.floor(
        (minx - origin_x) / pixel_width
    ) - 1,
)

col_end = min(
    width,
    math.ceil(
        (maxx - origin_x) / pixel_width
    ) + 1,
)

row_start = max(
    0,
    math.floor(
        (origin_y - maxy) / pixel_height
    ) - 1,
)

row_end = min(
    height,
    math.ceil(
        (origin_y - miny) / pixel_height
    ) + 1,
)

print()
print(
    "Raster window:",
    f"rows {row_start}:{row_end},",
    f"columns {col_start}:{col_end}",
)


# --------------------------------------------------
# 4. Read only the required raster window
# --------------------------------------------------

store = tif.aszarr()

z = zarr.open(
    store,
    mode="r",
)

array = np.asarray(
    z[
        row_start:row_end,
        col_start:col_end,
    ]
)

print("Window shape:", array.shape)


# --------------------------------------------------
# 5. Calculate centre coordinates of raster pixels
# --------------------------------------------------

columns = np.arange(
    col_start,
    col_end,
)

rows = np.arange(
    row_start,
    row_end,
)

xs = (
    origin_x
    + (columns + 0.5) * pixel_width
)

ys = (
    origin_y
    - (rows + 0.5) * pixel_height
)

xx, yy = np.meshgrid(
    xs,
    ys,
)


# --------------------------------------------------
# 6. Select valid raster values
# --------------------------------------------------

valid = (
    np.isfinite(array)
    & (array != nodata)
    & (array >= 0)
)

valid_values = array[
    valid
].astype(float)

valid_points = shapely.points(
    xx[valid],
    yy[valid],
)

pixels = gpd.GeoDataFrame(
    {
        "pixel_id": np.arange(
            len(valid_values)
        ),
        "population_value": valid_values,
    },
    geometry=valid_points,
    crs="EPSG:4326",
)

print()
print(
    "Valid population pixels in bbox:",
    len(pixels),
)


# --------------------------------------------------
# 7. Retain pixel centres inside Kuala Selangor
# --------------------------------------------------

inside_mask = pixels.geometry.within(
    district_geom
)

district_pixels = pixels.loc[
    inside_mask
].copy()

raw_population = float(
    district_pixels[
        "population_value"
    ].sum()
)

print(
    "Valid population pixels inside district:",
    len(district_pixels),
)

print(
    "Raw district population:",
    round(raw_population, 2),
)


# --------------------------------------------------
# 8. Assign population pixels to analysis cells
# --------------------------------------------------

joined = gpd.sjoin(
    district_pixels,
    grid[
        [
            "cell_id",
            "geometry",
        ]
    ],
    how="left",
    predicate="within",
)


# --------------------------------------------------
# 9. Handle rare pixel centres exactly on grid boundaries
# --------------------------------------------------

unassigned_mask = joined[
    "cell_id"
].isna()

unassigned_count = int(
    unassigned_mask.sum()
)

print()
print(
    "Initially unassigned district pixels:",
    unassigned_count,
)

if unassigned_count > 0:

    fallback_input = joined.loc[
        unassigned_mask,
        [
            "pixel_id",
            "population_value",
            "geometry",
        ],
    ].copy()

    fallback = gpd.sjoin(
        fallback_input,
        grid[
            [
                "cell_id",
                "geometry",
            ]
        ],
        how="left",
        predicate="intersects",
    )

    # If a point lies exactly on a shared cell boundary,
    # assign it deterministically to the lowest cell ID.
    fallback = fallback.sort_values(
        [
            "pixel_id",
            "cell_id",
        ]
    )

    fallback = fallback.drop_duplicates(
        subset="pixel_id",
        keep="first",
    )

    cell_lookup = fallback.set_index(
        "pixel_id"
    )["cell_id"]

    joined.loc[
        unassigned_mask,
        "cell_id",
    ] = joined.loc[
        unassigned_mask,
        "pixel_id",
    ].map(cell_lookup)


# --------------------------------------------------
# 10. Final assignment check
# --------------------------------------------------

still_unassigned = joined[
    "cell_id"
].isna()

if still_unassigned.any():

    missing_population = joined.loc[
        still_unassigned,
        "population_value",
    ].sum()

    raise ValueError(
        "Population pixels remain unassigned. "
        f"Count={int(still_unassigned.sum())}, "
        f"population={missing_population}"
    )


# --------------------------------------------------
# 11. Aggregate population by cell
# --------------------------------------------------

aggregated = joined.groupby(
    "cell_id",
    as_index=False,
).agg(
    population=(
        "population_value",
        "sum",
    ),
    population_pixels=(
        "pixel_id",
        "count",
    ),
)


# --------------------------------------------------
# 12. Join population to all analysis cells
# --------------------------------------------------

output = grid.merge(
    aggregated,
    on="cell_id",
    how="left",
)

# Preserve missing population data as NaN.
# NoData must not automatically be interpreted as zero population.
output["population"] = pd.to_numeric(
    output["population"],
    errors="coerce",
).astype(float)

output["population_pixels"] = (
    output["population_pixels"]
    .fillna(0)
    .astype(int)
)

output["population_data_present"] = (
    output["population_pixels"] > 0
)


# --------------------------------------------------
# 13. Validate population preservation
# --------------------------------------------------

output_population = float(
    output["population"].sum()
)

population_difference = abs(
    raw_population - output_population
)

cells_with_pixels = int(
    output[
        "population_data_present"
    ].sum()
)

cells_with_population = int(
    (
        (output["population"] > 0)
        & output["population_data_present"]
    ).sum()
)

cells_without_population_data = int(
    (~output["population_data_present"]).sum()
)

print()
print("--- POPULATION AGGREGATION AUDIT ---")

print(
    "Raw district population:",
    round(raw_population, 2),
)

print(
    "Output population:",
    round(output_population, 2),
)

print(
    "Difference:",
    round(population_difference, 6),
)

print(
    "Population preserved:",
    bool(
        np.isclose(
            raw_population,
            output_population,
            rtol=0,
            atol=0.1,
        )
    ),
)

print()
print(
    "Cells with valid population pixels:",
    cells_with_pixels,
)

print(
    "Cells with population > 0:",
    cells_with_population,
)

print(
    "Cells without population data:",
    cells_without_population_data,
)

print()
print(
    "Median population per cell:",
    round(
        output["population"].median(),
        2,
    ),
)

print(
    "Mean population per cell:",
    round(
        output["population"].mean(),
        2,
    ),
)

print(
    "Maximum population in one cell:",
    round(
        output["population"].max(),
        2,
    ),
)


# --------------------------------------------------
# 14. Hard validation
# --------------------------------------------------

if not np.isclose(
    raw_population,
    output_population,
    rtol=0,
    atol=0.1,
):
    raise ValueError(
        "Population was not preserved during aggregation."
    )

if output["cell_id"].duplicated().any():
    raise ValueError(
        "Duplicate cell IDs detected."
    )


# --------------------------------------------------
# 15. Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output.to_file(
    OUTPUT_PATH,
    driver="GeoJSON",
)

store.close()
tif.close()

print()
print("Saved:")
print(OUTPUT_PATH)

print()
print("=== COMPLETE ===")