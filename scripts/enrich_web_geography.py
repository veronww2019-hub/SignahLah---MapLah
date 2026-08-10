from pathlib import Path

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASE_CELLS = ROOT / "data" / "processed" / "signal_lah_cells_v2.geojson"
BOUNDARY = ROOT / "data" / "raw" / "boundary" / "extracted" / "geoBoundaries-MYS-ADM2.geojson"
PLACES = ROOT / "data" / "raw" / "osm" / "places_extracted" / "gis_osm_places_free_1.shp"

SCHOOLS = ROOT / "data" / "interim" / "canonical_schools.geojson"
HEALTHCARE = ROOT / "data" / "interim" / "canonical_healthcare.geojson"

PROCESSED_OUT = ROOT / "data" / "processed" / "signal_lah_cells_v2_enriched.geojson"

WEB_CELLS = ROOT / "web" / "signal_lah_cells_v2.geojson"
WEB_LOCALITIES = ROOT / "web" / "localities.geojson"
WEB_SCHOOLS = ROOT / "web" / "schools.geojson"
WEB_HEALTHCARE = ROOT / "web" / "healthcare.geojson"

WORKING_CRS = "EPSG:32647"

ALL_LOCALITY_TYPES = {
    "city",
    "town",
    "village",
    "suburb",
    "hamlet",
    "locality",
}

MAJOR_LOCALITY_TYPES = {
    "city",
    "town",
    "village",
    "suburb",
}


def get_district():
    boundary = gpd.read_file(BOUNDARY)

    district = boundary[
        boundary["shapeName"].astype(str).str.casefold().eq("kuala selangor")
    ].copy()

    if len(district) != 1:
        raise RuntimeError(
            f"Expected exactly one Kuala Selangor boundary, found {len(district)}"
        )

    return district


def get_localities(district):
    places = gpd.read_file(PLACES).to_crs(district.crs)

    places = places[
        places["name"].notna()
        & places["fclass"].isin(ALL_LOCALITY_TYPES)
    ].copy()

    # Keep named OSM settlements whose point lies inside Kuala Selangor.
    places = gpd.sjoin(
        places,
        district[["geometry"]],
        predicate="within",
        how="inner",
    )

    places = places.drop(columns=["index_right"], errors="ignore")

    # Remove exact duplicate OSM records / duplicate geometries where possible.
    places = (
        places.sort_values(["name", "fclass", "osm_id"])
        .drop_duplicates(
            subset=["osm_id"],
            keep="first",
        )
        .copy()
    )

    return places


def nearest_places(cell_points, places, major_only=False):
    allowed = MAJOR_LOCALITY_TYPES if major_only else ALL_LOCALITY_TYPES

    candidates = places[
        places["fclass"].isin(allowed)
    ][["osm_id", "name", "fclass", "geometry"]].copy()

    candidates = candidates.to_crs(WORKING_CRS)

    joined = gpd.sjoin_nearest(
        cell_points,
        candidates,
        how="left",
        distance_col="distance_m",
    )

    # sjoin_nearest may produce >1 row for exact-distance ties.
    # Resolve ties deterministically.
    hierarchy = {
        "city": 0,
        "town": 1,
        "village": 2,
        "suburb": 3,
        "hamlet": 4,
        "locality": 5,
    }

    joined["_rank"] = joined["fclass"].map(hierarchy).fillna(99)

    joined = (
        joined.sort_values(
            ["cell_id", "distance_m", "_rank", "name"],
            na_position="last",
        )
        .drop_duplicates("cell_id", keep="first")
        .copy()
    )

    return joined


def main():
    print("=== Signal Lah Phase 6B Geography Enrichment ===")

    cells = gpd.read_file(BASE_CELLS)

    if cells["cell_id"].duplicated().any():
        raise RuntimeError("Duplicate cell_id found in base dataset.")

    original_count = len(cells)

    district = get_district()
    places = get_localities(district)

    print(f"Base cells: {original_count}")
    print(f"Named Kuala Selangor localities: {len(places)}")

    cells_metric = cells.to_crs(WORKING_CRS)

    # Keep same centroid-distance method used during Phase 6B audit.
    cell_points = cells_metric[["cell_id", "geometry"]].copy()
    cell_points["geometry"] = cell_points.geometry.centroid

    nearest = nearest_places(
        cell_points,
        places,
        major_only=False,
    )

    nearest_major = nearest_places(
        cell_points,
        places,
        major_only=True,
    )

    locality_fields = nearest[
        ["cell_id", "name", "fclass", "distance_m"]
    ].rename(
        columns={
            "name": "nearest_locality",
            "fclass": "nearest_locality_type",
            "distance_m": "nearest_locality_distance_m",
        }
    )

    major_fields = nearest_major[
        ["cell_id", "name", "fclass", "distance_m"]
    ].rename(
        columns={
            "name": "nearest_major_locality",
            "fclass": "nearest_major_locality_type",
            "distance_m": "nearest_major_locality_distance_m",
        }
    )

    enriched = cells.merge(
        locality_fields,
        on="cell_id",
        how="left",
        validate="one_to_one",
    )

    enriched = enriched.merge(
        major_fields,
        on="cell_id",
        how="left",
        validate="one_to_one",
    )

    enriched["nearest_locality_distance_km"] = (
        enriched["nearest_locality_distance_m"] / 1000
    ).round(2)

    enriched["nearest_major_locality_distance_km"] = (
        enriched["nearest_major_locality_distance_m"] / 1000
    ).round(2)

    enriched = enriched.drop(
        columns=[
            "nearest_locality_distance_m",
            "nearest_major_locality_distance_m",
        ]
    )

    enriched["locality_source"] = "OpenStreetMap / Geofabrik"

    # ---------- Integrity checks ----------

    if len(enriched) != original_count:
        raise RuntimeError(
            f"Cell count changed: {original_count} -> {len(enriched)}"
        )

    if enriched["cell_id"].duplicated().any():
        raise RuntimeError("Duplicate cell_id produced during enrichment.")

    # Make sure core analytical values were not altered.
    critical_fields = [
        "cell_id",
        "avg_download_mbps",
        "avg_upload_mbps",
        "avg_latency_ms",
        "network_tests",
        "population",
        "school_count",
        "healthcare_count",
        "score_general",
        "score_education",
        "score_healthcare",
        "priority_general",
        "priority_education",
        "priority_healthcare",
        "ai_profile",
    ]

    for field in critical_fields:
        if field in cells.columns:
            left = cells[field].reset_index(drop=True)
            right = enriched[field].reset_index(drop=True)

            if not left.equals(right):
                raise RuntimeError(
                    f"Core analytical field changed unexpectedly: {field}"
                )

    # ---------- Facility assets ----------

    schools = gpd.read_file(SCHOOLS).to_crs("EPSG:4326")
    healthcare = gpd.read_file(HEALTHCARE).to_crs("EPSG:4326")

    if len(schools) != 105:
        raise RuntimeError(
            f"Expected 105 schools, found {len(schools)}"
        )

    if len(healthcare) != 64:
        raise RuntimeError(
            f"Expected 64 healthcare facilities, found {len(healthcare)}"
        )

    # ---------- Web locality layer ----------

    locality_web = places[
        ["osm_id", "fclass", "population", "name", "geometry"]
    ].copy()

    locality_web = locality_web.to_crs("EPSG:4326")

    # ---------- Export ----------

    PROCESSED_OUT.parent.mkdir(parents=True, exist_ok=True)
    WEB_CELLS.parent.mkdir(parents=True, exist_ok=True)

    enriched.to_crs("EPSG:4326").to_file(
        PROCESSED_OUT,
        driver="GeoJSON",
    )

    enriched.to_crs("EPSG:4326").to_file(
        WEB_CELLS,
        driver="GeoJSON",
    )

    locality_web.to_file(
        WEB_LOCALITIES,
        driver="GeoJSON",
    )

    schools.to_file(
        WEB_SCHOOLS,
        driver="GeoJSON",
    )

    healthcare.to_file(
        WEB_HEALTHCARE,
        driver="GeoJSON",
    )

    # ---------- Summary ----------

    near_dist = enriched["nearest_locality_distance_km"]
    major_dist = enriched["nearest_major_locality_distance_km"]

    print()
    print("=== OUTPUT SUMMARY ===")
    print(f"Enriched cells: {len(enriched)}")
    print(f"Locality browser features: {len(locality_web)}")
    print(f"Schools: {len(schools)}")
    print(f"Healthcare: {len(healthcare)}")

    print()
    print("Nearest locality distance:")
    print(f"  Median: {near_dist.median():.2f} km")
    print(f"  <= 5 km: {(near_dist <= 5).sum()} cells")
    print(f"  <= 10 km: {(near_dist <= 10).sum()} cells")

    print()
    print("Nearest major locality distance:")
    print(f"  Median: {major_dist.median():.2f} km")
    print(f"  <= 5 km: {(major_dist <= 5).sum()} cells")
    print(f"  <= 10 km: {(major_dist <= 10).sum()} cells")

    print()
    print("Example enriched cells:")
    print(
        enriched[
            [
                "cell_id",
                "nearest_locality",
                "nearest_locality_type",
                "nearest_locality_distance_km",
                "nearest_major_locality",
                "nearest_major_locality_type",
                "nearest_major_locality_distance_km",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("Phase 6B enrichment completed successfully.")


if __name__ == "__main__":
    main()