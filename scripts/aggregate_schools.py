"""
Signal Lah - Phase 2.4
Build a deduplicated school facility layer from OSM and
aggregate schools into the Signal Lah analysis grid.

Inputs:
    data/raw/osm/extracted/gis_osm_pois_free_1.shp
    data/raw/osm/extracted/gis_osm_pois_a_free_1.shp
    data/raw/boundary/extracted/geoBoundaries-MYS-ADM2.geojson
    data/interim/grid_with_network_population.geojson

Outputs:
    data/interim/canonical_schools.geojson
    data/interim/grid_with_network_population_education.geojson
"""

from pathlib import Path
import re
import unicodedata

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union


# --------------------------------------------------
# Configuration
# --------------------------------------------------

POINT_PATH = Path(
    "data/raw/osm/extracted/gis_osm_pois_free_1.shp"
)

AREA_PATH = Path(
    "data/raw/osm/extracted/gis_osm_pois_a_free_1.shp"
)

BOUNDARY_PATH = Path(
    "data/raw/boundary/extracted/geoBoundaries-MYS-ADM2.geojson"
)

GRID_PATH = Path(
    "data/interim/grid_with_network_population.geojson"
)

SCHOOL_OUTPUT = Path(
    "data/interim/canonical_schools.geojson"
)

GRID_OUTPUT = Path(
    "data/interim/grid_with_network_population_education.geojson"
)

DISTRICT_NAME = "Kuala Selangor"

WORKING_CRS = "EPSG:32647"

# Used only when two records already have the same
# normalized school name.
SAME_NAME_DISTANCE_M = 250


# --------------------------------------------------
# Name normalization
# --------------------------------------------------

def normalize_name(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if not text:
        return None

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text or None


# --------------------------------------------------
# 1. Load district and grid
# --------------------------------------------------

boundary = gpd.read_file(
    BOUNDARY_PATH
)

district = boundary[
    boundary["shapeName"] == DISTRICT_NAME
].copy()

if len(district) != 1:
    raise ValueError(
        f"Expected one {DISTRICT_NAME} boundary, "
        f"found {len(district)}"
    )

district = district.to_crs(
    "EPSG:4326"
)

grid = gpd.read_file(
    GRID_PATH
).to_crs("EPSG:4326")

bbox = tuple(
    district.total_bounds
)

print("=== SIGNAL LAH SCHOOL AGGREGATION ===")
print()
print("Analysis cells:", len(grid))


# --------------------------------------------------
# 2. Load OSM school points and polygons
# --------------------------------------------------

points = gpd.read_file(
    POINT_PATH,
    bbox=bbox,
    engine="pyogrio",
)

areas = gpd.read_file(
    AREA_PATH,
    bbox=bbox,
    engine="pyogrio",
)

district_geom = district.geometry.iloc[0]

points = points[
    points.within(district_geom)
].copy()

areas_utm_all = areas.to_crs(
    WORKING_CRS
)

district_utm = district.to_crs(
    WORKING_CRS
)

area_centres = (
    areas_utm_all.geometry
    .representative_point()
)

area_inside = area_centres.within(
    district_utm.geometry.iloc[0]
)

areas = areas.loc[
    area_inside.values
].copy()

points = points[
    points["fclass"] == "school"
].copy()

areas = areas[
    areas["fclass"] == "school"
].copy()

print(
    "Raw school points:",
    len(points),
)

print(
    "Raw school polygons:",
    len(areas),
)

print(
    "Raw combined school records:",
    len(points) + len(areas),
)


# --------------------------------------------------
# 3. Prepare projected geometries
# --------------------------------------------------

points = points.to_crs(
    WORKING_CRS
)

areas = areas.to_crs(
    WORKING_CRS
)

points["norm_name"] = (
    points["name"]
    .apply(normalize_name)
)

areas["norm_name"] = (
    areas["name"]
    .apply(normalize_name)
)


# --------------------------------------------------
# 4. Deduplicate school polygons with same name
# --------------------------------------------------

area_records = []

used = set()

for idx, row in areas.iterrows():

    if idx in used:
        continue

    name = row["norm_name"]

    if name is None:
        group_indices = [idx]

    else:
        candidates = areas[
            areas["norm_name"] == name
        ]

        reference_point = (
            row.geometry.representative_point()
        )

        group_indices = []

        for other_idx, other in candidates.iterrows():

            if other_idx in used:
                continue

            other_point = (
                other.geometry.representative_point()
            )

            distance = (
                reference_point.distance(
                    other_point
                )
            )

            if distance <= SAME_NAME_DISTANCE_M:
                group_indices.append(
                    other_idx
                )

        if not group_indices:
            group_indices = [idx]

    group = areas.loc[
        group_indices
    ]

    used.update(
        group_indices
    )

    merged_geometry = unary_union(
        group.geometry.tolist()
    )

    display_names = (
        group["name"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    display_name = (
        display_names.iloc[0]
        if len(display_names) > 0
        else None
    )

    area_records.append(
        {
            "name": display_name,
            "norm_name": name,
            "source_type": "area",
            "source_records": len(group),
            "geometry": merged_geometry,
        }
    )


canonical_areas = gpd.GeoDataFrame(
    area_records,
    geometry="geometry",
    crs=WORKING_CRS,
)

print()
print(
    "School polygons after same-name dedup:",
    len(canonical_areas),
)


# --------------------------------------------------
# 5. Remove point records duplicated by polygons
# --------------------------------------------------

kept_points = []

point_duplicates = 0

for _, point in points.iterrows():

    geom = point.geometry
    name = point["norm_name"]

    duplicate = False

    # Strongest rule:
    # a school point inside a school polygon is assumed
    # to describe the same facility.
    containing = canonical_areas[
        canonical_areas.geometry.contains(
            geom
        )
    ]

    if len(containing) > 0:
        duplicate = True

    # Second rule:
    # same normalized name within 250 m.
    if (
        not duplicate
        and name is not None
    ):

        same_name = canonical_areas[
            canonical_areas[
                "norm_name"
            ] == name
        ]

        if len(same_name) > 0:

            distances = (
                same_name.geometry
                .representative_point()
                .distance(geom)
            )

            if (
                distances
                <= SAME_NAME_DISTANCE_M
            ).any():
                duplicate = True

    if duplicate:
        point_duplicates += 1
        continue

    kept_points.append(
        point
    )


if kept_points:

    remaining_points = gpd.GeoDataFrame(
        kept_points,
        geometry="geometry",
        crs=WORKING_CRS,
    )

else:

    remaining_points = gpd.GeoDataFrame(
        columns=points.columns,
        geometry="geometry",
        crs=WORKING_CRS,
    )


print(
    "Point records duplicated by polygons:",
    point_duplicates,
)

print(
    "Remaining school points:",
    len(remaining_points),
)


# --------------------------------------------------
# 6. Deduplicate remaining point records
#    by same name + proximity
# --------------------------------------------------

point_records = []

used = set()

for idx, row in remaining_points.iterrows():

    if idx in used:
        continue

    name = row["norm_name"]

    if name is None:
        group_indices = [idx]

    else:

        candidates = remaining_points[
            remaining_points[
                "norm_name"
            ] == name
        ]

        group_indices = []

        for other_idx, other in candidates.iterrows():

            if other_idx in used:
                continue

            distance = (
                row.geometry.distance(
                    other.geometry
                )
            )

            if distance <= SAME_NAME_DISTANCE_M:
                group_indices.append(
                    other_idx
                )

        if not group_indices:
            group_indices = [idx]

    group = remaining_points.loc[
        group_indices
    ]

    used.update(
        group_indices
    )

    display_names = (
        group["name"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    display_name = (
        display_names.iloc[0]
        if len(display_names) > 0
        else None
    )

    # Use first point as deterministic representative.
    representative_geometry = (
        group.geometry.iloc[0]
    )

    point_records.append(
        {
            "name": display_name,
            "norm_name": name,
            "source_type": "point",
            "source_records": len(group),
            "geometry": representative_geometry,
        }
    )


canonical_points = gpd.GeoDataFrame(
    point_records,
    geometry="geometry",
    crs=WORKING_CRS,
)

print(
    "Remaining point facilities after dedup:",
    len(canonical_points),
)


# --------------------------------------------------
# 7. Combine canonical schools
# --------------------------------------------------

canonical = pd.concat(
    [
        canonical_areas,
        canonical_points,
    ],
    ignore_index=True,
)

canonical = gpd.GeoDataFrame(
    canonical,
    geometry="geometry",
    crs=WORKING_CRS,
)

canonical["school_id"] = [
    f"SCH_{i:03d}"
    for i in range(
        1,
        len(canonical) + 1
    )
]

canonical["school_name"] = (
    canonical["name"]
)

canonical["named"] = (
    canonical["school_name"]
    .notna()
)


# --------------------------------------------------
# 8. Create one representative point per school
# --------------------------------------------------

school_points = canonical.copy()

school_points["geometry"] = (
    school_points.geometry
    .representative_point()
)


# --------------------------------------------------
# 9. Assign canonical schools to grid cells
# --------------------------------------------------

grid_utm = grid.to_crs(
    WORKING_CRS
)

assigned = gpd.sjoin(
    school_points,
    grid_utm[
        [
            "cell_id",
            "geometry",
        ]
    ],
    how="left",
    predicate="within",
)

unassigned = assigned[
    assigned["cell_id"].isna()
]

if len(unassigned) > 0:
    raise ValueError(
        f"{len(unassigned)} canonical schools "
        "could not be assigned to a grid cell."
    )


# --------------------------------------------------
# 10. Aggregate schools by cell
# --------------------------------------------------

counts = assigned.groupby(
    "cell_id",
    as_index=False,
).agg(
    school_count=(
        "school_id",
        "count",
    ),
)


output = grid.merge(
    counts,
    on="cell_id",
    how="left",
)

output["school_count"] = (
    output["school_count"]
    .fillna(0)
    .astype(int)
)

output["school_present"] = (
    output["school_count"] > 0
)


# --------------------------------------------------
# 11. Audit
# --------------------------------------------------

print()
print("--- SCHOOL DEDUPLICATION AUDIT ---")

print(
    "Raw point records:",
    len(points),
)

print(
    "Raw area records:",
    len(areas),
)

print(
    "Raw total records:",
    len(points) + len(areas),
)

print(
    "Canonical unique schools:",
    len(canonical),
)

print(
    "Named canonical schools:",
    int(canonical["named"].sum()),
)

print(
    "Unnamed canonical schools:",
    int((~canonical["named"]).sum()),
)

print(
    "OSM records represented:",
    int(
        canonical[
            "source_records"
        ].sum()
        + point_duplicates
    ),
)

print()
print(
    "Cells containing schools:",
    int(
        output[
            "school_present"
        ].sum()
    ),
)

print(
    "Total schools in grid:",
    int(
        output[
            "school_count"
        ].sum()
    ),
)

print(
    "Maximum schools in one cell:",
    int(
        output[
            "school_count"
        ].max()
    ),
)

print()
print(
    "Population still preserved:",
    round(
        output["population"].sum(),
        2,
    ),
)

print(
    "Network tests still preserved:",
    int(
        output["network_tests"].sum()
    ),
)


# --------------------------------------------------
# 12. Hard validation
# --------------------------------------------------

if (
    int(output["school_count"].sum())
    != len(canonical)
):
    raise ValueError(
        "Canonical school count was not "
        "preserved during grid aggregation."
    )

if int(
    output["network_tests"].sum()
) != 8441:
    raise ValueError(
        "Network test count changed."
    )


# --------------------------------------------------
# 13. Save canonical schools and updated grid
# --------------------------------------------------

SCHOOL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

canonical_save = canonical[
    [
        "school_id",
        "school_name",
        "named",
        "source_type",
        "source_records",
        "geometry",
    ]
].copy()

canonical_save = canonical_save.to_crs(
    "EPSG:4326"
)

canonical_save.to_file(
    SCHOOL_OUTPUT,
    driver="GeoJSON",
)

output.to_file(
    GRID_OUTPUT,
    driver="GeoJSON",
)

print()
print("Saved canonical schools:")
print(SCHOOL_OUTPUT)

print()
print("Saved updated grid:")
print(GRID_OUTPUT)

print()
print("=== COMPLETE ===")