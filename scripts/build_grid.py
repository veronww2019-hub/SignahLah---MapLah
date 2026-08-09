"""
Signal Lah - Phase 2.1
Build a reproducible ~2 km analysis grid for Kuala Selangor.

Small boundary slivers below 0.20 km² are merged into the
adjacent cell with which they share the longest boundary.

Input:
    data/raw/boundary/extracted/geoBoundaries-MYS-ADM2.geojson

Output:
    data/interim/analysis_grid.geojson
"""

from pathlib import Path
import math

import geopandas as gpd
from shapely.geometry import box


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BOUNDARY_PATH = Path(
    "data/raw/boundary/extracted/geoBoundaries-MYS-ADM2.geojson"
)

OUTPUT_PATH = Path(
    "data/interim/analysis_grid.geojson"
)

DISTRICT_NAME = "Kuala Selangor"

WORKING_CRS = "EPSG:32647"

GRID_SIZE_M = 2000

# Normal full cell = 4 km².
# Anything below 5% of a full cell is treated as a
# boundary sliver and merged into a neighbouring cell.
MIN_CELL_AREA_KM2 = 0.20


# --------------------------------------------------
# 1. Read boundary
# --------------------------------------------------

boundary = gpd.read_file(BOUNDARY_PATH)

district = boundary[
    boundary["shapeName"] == DISTRICT_NAME
].copy()

if len(district) != 1:
    raise ValueError(
        f"Expected exactly one {DISTRICT_NAME} feature, "
        f"found {len(district)}"
    )

district = district.to_crs(WORKING_CRS)

district_geom = district.geometry.iloc[0]

print("=== SIGNAL LAH GRID BUILDER ===")
print()
print("District:", DISTRICT_NAME)
print("Working CRS:", district.crs)
print(
    "District area:",
    round(district_geom.area / 1_000_000, 2),
    "km²",
)


# --------------------------------------------------
# 2. Create aligned 2 km grid
# --------------------------------------------------

minx, miny, maxx, maxy = district.total_bounds

start_x = math.floor(minx / GRID_SIZE_M) * GRID_SIZE_M
start_y = math.floor(miny / GRID_SIZE_M) * GRID_SIZE_M

end_x = math.ceil(maxx / GRID_SIZE_M) * GRID_SIZE_M
end_y = math.ceil(maxy / GRID_SIZE_M) * GRID_SIZE_M

cells = []

y = start_y

while y < end_y:
    x = start_x

    while x < end_x:
        square = box(
            x,
            y,
            x + GRID_SIZE_M,
            y + GRID_SIZE_M,
        )

        if square.intersects(district_geom):
            clipped = square.intersection(district_geom)

            if not clipped.is_empty and clipped.area > 0:
                cells.append(clipped)

        x += GRID_SIZE_M

    y += GRID_SIZE_M


grid = gpd.GeoDataFrame(
    geometry=cells,
    crs=WORKING_CRS,
)

print()
print("Initial clipped cells:", len(grid))


# --------------------------------------------------
# 3. Merge tiny boundary slivers
# --------------------------------------------------

minimum_area_m2 = MIN_CELL_AREA_KM2 * 1_000_000

merged_count = 0
unmerged_isolated = 0

while True:

    areas = grid.geometry.area

    tiny_indices = areas[
        areas < minimum_area_m2
    ].index.tolist()

    if not tiny_indices:
        break

    # Process the smallest sliver first.
    tiny_idx = min(
        tiny_indices,
        key=lambda idx: grid.at[idx, "geometry"].area,
    )

    tiny_geom = grid.at[
        tiny_idx,
        "geometry",
    ]

    candidates = grid.drop(
        index=tiny_idx
    )

    touching = candidates[
        candidates.geometry.touches(tiny_geom)
    ].copy()

    if touching.empty:
        # This may occur for a genuinely isolated island.
        # Keep it rather than deleting geography.
        grid.at[tiny_idx, "_protected"] = True

        protected = grid.get(
            "_protected",
            False,
        )

        other_tiny = [
            idx
            for idx in tiny_indices
            if idx != tiny_idx
            and not bool(
                grid.at[idx, "_protected"]
                if "_protected" in grid.columns
                else False
            )
        ]

        if not other_tiny:
            unmerged_isolated += 1
            break

        continue

    # Measure the boundary shared with each neighbour.
    touching["shared_boundary"] = (
        touching.geometry.boundary.intersection(
            tiny_geom.boundary
        ).length
    )

    target_idx = touching[
        "shared_boundary"
    ].idxmax()

    target_geom = grid.at[
        target_idx,
        "geometry",
    ]

    # Merge the sliver into its strongest neighbour.
    grid.at[
        target_idx,
        "geometry",
    ] = target_geom.union(tiny_geom)

    grid = grid.drop(
        index=tiny_idx
    ).reset_index(drop=True)

    merged_count += 1


if "_protected" in grid.columns:
    grid = grid.drop(columns="_protected")


print(
    "Boundary slivers merged:",
    merged_count,
)

print(
    "Remaining cells:",
    len(grid),
)


# --------------------------------------------------
# 4. Generate stable cell IDs
# --------------------------------------------------

representative = (
    grid.geometry.representative_point()
)

grid["sort_y"] = representative.y
grid["sort_x"] = representative.x

grid = grid.sort_values(
    ["sort_y", "sort_x"],
    ascending=[False, True],
).reset_index(drop=True)

grid["cell_id"] = [
    f"KS_{i:03d}"
    for i in range(1, len(grid) + 1)
]


# --------------------------------------------------
# 5. Geometry attributes
# --------------------------------------------------

grid["cell_area_km2"] = (
    grid.geometry.area / 1_000_000
)

representative = (
    grid.geometry.representative_point()
)

grid["center_easting"] = representative.x
grid["center_northing"] = representative.y

centres_geo = gpd.GeoSeries(
    representative,
    crs=WORKING_CRS,
).to_crs("EPSG:4326")

grid["center_lon"] = centres_geo.x
grid["center_lat"] = centres_geo.y

grid = grid.drop(
    columns=["sort_x", "sort_y"]
)


# --------------------------------------------------
# 6. Validate
# --------------------------------------------------

grid_area = grid.geometry.area.sum()
district_area = district_geom.area

area_difference = abs(
    grid_area - district_area
)

full_cells = (
    grid["cell_area_km2"] >= 3.999
).sum()

partial_cells = (
    len(grid) - full_cells
)

tiny_remaining = (
    grid["cell_area_km2"]
    < MIN_CELL_AREA_KM2
).sum()


print()
print("--- FINAL GRID AUDIT ---")

print("Grid cells:", len(grid))

print(
    "Full ~4 km² cells:",
    int(full_cells),
)

print(
    "Boundary / merged cells:",
    int(partial_cells),
)

print(
    "Grid area:",
    round(grid_area / 1_000_000, 4),
    "km²",
)

print(
    "District area:",
    round(district_area / 1_000_000, 4),
    "km²",
)

print(
    "Area difference:",
    round(area_difference, 2),
    "m²",
)

print(
    "Minimum cell area:",
    round(
        grid["cell_area_km2"].min(),
        4,
    ),
    "km²",
)

print(
    "Maximum cell area:",
    round(
        grid["cell_area_km2"].max(),
        4,
    ),
    "km²",
)

print(
    "Cells below minimum threshold:",
    int(tiny_remaining),
)

print(
    "All geometries valid:",
    bool(
        grid.geometry.is_valid.all()
    ),
)

print(
    "Duplicate cell IDs:",
    int(
        grid["cell_id"]
        .duplicated()
        .sum()
    ),
)


# --------------------------------------------------
# 7. Save
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output = grid.to_crs(
    "EPSG:4326"
)

output.to_file(
    OUTPUT_PATH,
    driver="GeoJSON",
)

print()
print("Saved:")
print(OUTPUT_PATH)

print()
print(
    "Output CRS:",
    output.crs,
)

print(
    "=== COMPLETE ==="
)