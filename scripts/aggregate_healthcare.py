"""
Signal Lah - Phase 2.5
Build a deduplicated healthcare facility layer from OSM
and aggregate healthcare facilities into the Signal Lah grid.

Healthcare categories retained separately:
- hospital
- clinic
- doctors

Inputs:
    data/raw/osm/extracted/gis_osm_pois_free_1.shp
    data/raw/osm/extracted/gis_osm_pois_a_free_1.shp
    data/raw/boundary/extracted/geoBoundaries-MYS-ADM2.geojson
    data/interim/grid_with_network_population_education.geojson

Outputs:
    data/interim/canonical_healthcare.geojson
    data/interim/grid_complete_raw.geojson
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
    "data/interim/grid_with_network_population_education.geojson"
)

HEALTH_OUTPUT = Path(
    "data/interim/canonical_healthcare.geojson"
)

GRID_OUTPUT = Path(
    "data/interim/grid_complete_raw.geojson"
)

DISTRICT_NAME = "Kuala Selangor"

WORKING_CRS = "EPSG:32647"

HEALTH_CLASSES = [
    "hospital",
    "clinic",
    "doctors",
]

# Only records already sharing the same normalized
# name may be merged using this proximity threshold.
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
# 1. Load district and analysis grid
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

district_geom = district.geometry.iloc[0]

grid = gpd.read_file(
    GRID_PATH
).to_crs("EPSG:4326")

bbox = tuple(
    district.total_bounds
)

print("=== SIGNAL LAH HEALTHCARE AGGREGATION ===")
print()
print("Analysis cells:", len(grid))


# --------------------------------------------------
# 2. Load current OSM healthcare records
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

# Exact point filtering
points = points[
    points.within(district_geom)
].copy()

# Area filtering using representative points
areas_utm_all = areas.to_crs(
    WORKING_CRS
)

district_utm = district.to_crs(
    WORKING_CRS
)

area_reps = (
    areas_utm_all.geometry
    .representative_point()
)

area_inside = area_reps.within(
    district_utm.geometry.iloc[0]
)

areas = areas.loc[
    area_inside.values
].copy()

points = points[
    points["fclass"].isin(
        HEALTH_CLASSES
    )
].copy()

areas = areas[
    areas["fclass"].isin(
        HEALTH_CLASSES
    )
].copy()


print(
    "Raw healthcare points:",
    len(points),
)

print(
    "Raw healthcare polygons:",
    len(areas),
)

print()
print("POINT CLASSES:")
print(
    points["fclass"]
    .value_counts()
    .to_string()
)

print()
print("AREA CLASSES:")
print(
    areas["fclass"]
    .value_counts()
    .to_string()
)


# --------------------------------------------------
# 3. Prepare projected records
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
# 4. Deduplicate polygons
#
# Conservative rule:
# Same normalized name + same facility type +
# within 250 m.
#
# Unnamed features remain separate.
# --------------------------------------------------

area_records = []
used_area_indices = set()

for idx, row in areas.iterrows():

    if idx in used_area_indices:
        continue

    name = row["norm_name"]
    facility_type = row["fclass"]

    if name is None:

        group_indices = [idx]

    else:

        candidates = areas[
            (areas["norm_name"] == name)
            & (
                areas["fclass"]
                == facility_type
            )
        ]

        reference = (
            row.geometry
            .representative_point()
        )

        group_indices = []

        for other_idx, other in candidates.iterrows():

            if other_idx in used_area_indices:
                continue

            other_rep = (
                other.geometry
                .representative_point()
            )

            if (
                reference.distance(
                    other_rep
                )
                <= SAME_NAME_DISTANCE_M
            ):
                group_indices.append(
                    other_idx
                )

        if not group_indices:
            group_indices = [idx]

    group = areas.loc[
        group_indices
    ]

    used_area_indices.update(
        group_indices
    )

    merged_geometry = unary_union(
        group.geometry.tolist()
    )

    names = (
        group["name"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    display_name = (
        names.iloc[0]
        if len(names) > 0
        else None
    )

    area_records.append(
        {
            "facility_name": display_name,
            "norm_name": name,
            "facility_type": facility_type,
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
    "Healthcare polygons after dedup:",
    len(canonical_areas),
)


# --------------------------------------------------
# 5. Remove point records duplicated by polygons
#
# We deliberately DO NOT remove every healthcare point
# merely because it lies inside a healthcare polygon.
#
# A clinic located within a hospital campus can be a
# separate real service.
#
# We require matching normalized names.
# --------------------------------------------------

kept_points = []
point_polygon_duplicates = 0

for _, point in points.iterrows():

    point_name = point["norm_name"]

    duplicate = False

    if point_name is not None:

        same_name_areas = canonical_areas[
            canonical_areas[
                "norm_name"
            ] == point_name
        ]

        for _, area in same_name_areas.iterrows():

            area_rep = (
                area.geometry
                .representative_point()
            )

            if (
                area.geometry.contains(
                    point.geometry
                )
                or
                area_rep.distance(
                    point.geometry
                )
                <= SAME_NAME_DISTANCE_M
            ):

                duplicate = True
                break

    if duplicate:
        point_polygon_duplicates += 1
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
    point_polygon_duplicates,
)

print(
    "Healthcare points remaining:",
    len(remaining_points),
)


# --------------------------------------------------
# 6. Deduplicate remaining points
#
# Same normalized name + same category + proximity.
# --------------------------------------------------

point_records = []
used_point_indices = set()

for idx, row in remaining_points.iterrows():

    if idx in used_point_indices:
        continue

    name = row["norm_name"]
    facility_type = row["fclass"]

    if name is None:

        group_indices = [idx]

    else:

        candidates = remaining_points[
            (
                remaining_points[
                    "norm_name"
                ] == name
            )
            &
            (
                remaining_points[
                    "fclass"
                ] == facility_type
            )
        ]

        group_indices = []

        for other_idx, other in candidates.iterrows():

            if other_idx in used_point_indices:
                continue

            if (
                row.geometry.distance(
                    other.geometry
                )
                <= SAME_NAME_DISTANCE_M
            ):

                group_indices.append(
                    other_idx
                )

        if not group_indices:
            group_indices = [idx]

    group = remaining_points.loc[
        group_indices
    ]

    used_point_indices.update(
        group_indices
    )

    names = (
        group["name"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    display_name = (
        names.iloc[0]
        if len(names) > 0
        else None
    )

    point_records.append(
        {
            "facility_name": display_name,
            "norm_name": name,
            "facility_type": facility_type,
            "source_type": "point",
            "source_records": len(group),
            "geometry": group.geometry.iloc[0],
        }
    )


canonical_points = gpd.GeoDataFrame(
    point_records,
    geometry="geometry",
    crs=WORKING_CRS,
)

print(
    "Point facilities after dedup:",
    len(canonical_points),
)


# --------------------------------------------------
# 7. Combine canonical facilities
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

canonical["facility_id"] = [
    f"HLT_{i:03d}"
    for i in range(
        1,
        len(canonical) + 1
    )
]

canonical["named"] = (
    canonical["facility_name"]
    .notna()
)


# --------------------------------------------------
# 8. Representative point for grid assignment
# --------------------------------------------------

facility_points = canonical.copy()

facility_points["geometry"] = (
    facility_points.geometry
    .representative_point()
)


# --------------------------------------------------
# 9. Assign healthcare facilities to grid cells
# --------------------------------------------------

grid_utm = grid.to_crs(
    WORKING_CRS
)

assigned = gpd.sjoin(
    facility_points,
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
        f"{len(unassigned)} healthcare facilities "
        "could not be assigned to grid cells."
    )


# --------------------------------------------------
# 10. Count each healthcare category per grid
# --------------------------------------------------

pivot = (
    assigned
    .groupby(
        [
            "cell_id",
            "facility_type",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


for category in HEALTH_CLASSES:

    if category not in pivot.columns:
        pivot[category] = 0


pivot = pivot[
    HEALTH_CLASSES
].reset_index()

pivot = pivot.rename(
    columns={
        "hospital": "hospital_count",
        "clinic": "clinic_count",
        "doctors": "doctor_count",
    }
)


# --------------------------------------------------
# 11. Join counts to analysis grid
# --------------------------------------------------

output = grid.merge(
    pivot,
    on="cell_id",
    how="left",
)

health_count_columns = [
    "hospital_count",
    "clinic_count",
    "doctor_count",
]

for column in health_count_columns:

    output[column] = (
        output[column]
        .fillna(0)
        .astype(int)
    )


output["healthcare_count"] = (
    output["hospital_count"]
    + output["clinic_count"]
    + output["doctor_count"]
)

output["healthcare_present"] = (
    output["healthcare_count"] > 0
)


# --------------------------------------------------
# 12. Audit
# --------------------------------------------------

print()
print("--- HEALTHCARE DEDUPLICATION AUDIT ---")

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
    "Canonical unique healthcare facilities:",
    len(canonical),
)

print(
    "Named facilities:",
    int(
        canonical["named"].sum()
    ),
)

print(
    "Unnamed facilities:",
    int(
        (~canonical["named"]).sum()
    ),
)

print()
print("CANONICAL FACILITY TYPES:")
print(
    canonical[
        "facility_type"
    ]
    .value_counts()
    .to_string()
)

print()
print(
    "Hospitals:",
    int(
        output[
            "hospital_count"
        ].sum()
    ),
)

print(
    "Clinics:",
    int(
        output[
            "clinic_count"
        ].sum()
    ),
)

print(
    "Doctors:",
    int(
        output[
            "doctor_count"
        ].sum()
    ),
)

print(
    "Total healthcare facilities:",
    int(
        output[
            "healthcare_count"
        ].sum()
    ),
)

print(
    "Cells with healthcare:",
    int(
        output[
            "healthcare_present"
        ].sum()
    ),
)

print(
    "Maximum healthcare facilities in one cell:",
    int(
        output[
            "healthcare_count"
        ].max()
    ),
)

print()
print(
    "Schools still preserved:",
    int(
        output[
            "school_count"
        ].sum()
    ),
)

print(
    "Population still preserved:",
    round(
        output[
            "population"
        ].sum(),
        2,
    ),
)

print(
    "Network tests still preserved:",
    int(
        output[
            "network_tests"
        ].sum()
    ),
)


# --------------------------------------------------
# 13. Hard validation
# --------------------------------------------------

if (
    int(
        output[
            "healthcare_count"
        ].sum()
    )
    != len(canonical)
):
    raise ValueError(
        "Healthcare facility count was not preserved."
    )

if int(
    output[
        "school_count"
    ].sum()
) != 105:
    raise ValueError(
        "School count changed."
    )

if int(
    output[
        "network_tests"
    ].sum()
) != 8441:
    raise ValueError(
        "Network test count changed."
    )


# --------------------------------------------------
# 14. Save
# --------------------------------------------------

HEALTH_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

canonical_save = canonical[
    [
        "facility_id",
        "facility_name",
        "facility_type",
        "named",
        "source_type",
        "source_records",
        "geometry",
    ]
].copy()

canonical_save = (
    canonical_save
    .to_crs("EPSG:4326")
)

canonical_save.to_file(
    HEALTH_OUTPUT,
    driver="GeoJSON",
)

output.to_file(
    GRID_OUTPUT,
    driver="GeoJSON",
)

print()
print("Saved canonical healthcare:")
print(HEALTH_OUTPUT)

print()
print("Saved complete raw grid:")
print(GRID_OUTPUT)

print()
print("=== COMPLETE ===")