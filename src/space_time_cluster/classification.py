from __future__ import annotations

"""Classify summarized clusters against optional city and lake reference layers."""

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import geopandas as gpd


def classify_clusters_against_polygons(
    cluster_summary: pl.DataFrame,
    city_path: str | None,
    lake_path: str | None,
) -> tuple["gpd.GeoDataFrame | None", "gpd.GeoDataFrame | None"]:
    """Join buffered cluster summaries against optional city and lake layers.

    Inputs:
        cluster_summary: Per-cluster summary table with center and radius columns.
        city_path: Optional path to a city polygon layer.
        lake_path: Optional path to a lake polygon layer.

    Returns:
        A pair of GeoDataFrames for city hits and lake hits. Each item is ``None``
        when the corresponding layer was not requested or no classification was run.
    """
    if city_path is None and lake_path is None:
        return None, None

    import geopandas as gpd

    pdf = cluster_summary.to_pandas()
    if pdf.empty:
        return None, None

    centers = gpd.GeoDataFrame(
        pdf,
        geometry=gpd.points_from_xy(pdf["center_lon"], pdf["center_lat"]),
        crs="EPSG:4326",
    )
    local_crs = centers.estimate_utm_crs()
    centers_local = centers.to_crs(local_crs).copy()
    centers_local["geometry"] = centers_local.geometry.buffer(centers_local["radius_m"].to_numpy())

    city_hits = None
    lake_hits = None
    if city_path is not None:
        cities = gpd.read_file(city_path).to_crs(local_crs)
        city_hits = gpd.sjoin(centers_local, cities, how="left", predicate="intersects")
    if lake_path is not None:
        lakes = gpd.read_file(lake_path).to_crs(local_crs)
        lake_hits = gpd.sjoin(centers_local, lakes, how="left", predicate="intersects")
    return city_hits, lake_hits
