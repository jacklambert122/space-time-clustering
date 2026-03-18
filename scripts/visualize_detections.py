from __future__ import annotations

"""Plot clustered detections with hvPlot, centroids, ellipses, and nearby reference features."""

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import holoviews as hv
import hvplot.pandas  # noqa: F401
import hvplot.polars  # noqa: F401
import numpy as np
import pandas as pd
import polars as pl
from holoviews.element import tiles
from shapely import affinity
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

WEB_MERCATOR_LIMIT_LAT = 85.05112878
WEB_MERCATOR_ORIGIN_SHIFT = 20037508.34
WEB_MERCATOR_WORLD_EXTENT_M = 20_037_508.34


def default_existing_path(candidates: Iterable[str]) -> str | None:
    """Return the first existing path from a candidate list.

    Inputs:
        candidates: Candidate filesystem paths in priority order.

    Returns:
        The first existing path as a string, or ``None`` when none exist.
    """
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def resolve_basemap(name: str) -> hv.Element | None:
    """Resolve a user-facing basemap name to a HoloViews tile source.

    Inputs:
        name: Basemap selector name.

    Returns:
        A tile element for the requested basemap, or ``None`` when no basemap should be drawn.
    """
    normalized = name.strip().lower()
    basemaps: dict[str, hv.Element | None] = {
        "none": None,
        "osm": tiles.OSM(),
        "cartolight": tiles.CartoLight(),
        "cartodark": tiles.CartoDark(),
        "esriimagery": tiles.EsriImagery(),
        "esristreet": tiles.EsriStreet(),
        "esriterrain": tiles.EsriTerrain(),
        "opentopo": tiles.OpenTopoMap(),
    }
    if normalized not in basemaps:
        choices = ", ".join(sorted(basemaps))
        raise ValueError(f"Unknown basemap '{name}'. Choose from: {choices}")
    return basemaps[normalized]


def graticule_paths(step_degrees: int = 30) -> pd.DataFrame:
    """Build a simple global latitude/longitude graticule.

    Inputs:
        step_degrees: Spacing between graticule lines in degrees.

    Returns:
        A DataFrame of path rows in lon/lat coordinates.
    """
    rows: list[dict[str, float | str]] = []
    path_index = 0
    for lat in range(-60, 61, step_degrees):
        for lon in range(-180, 181, 2):
            rows.append({"lon": float(lon), "lat": float(lat), "path_id": f"lat_{path_index}"})
        path_index += 1
    for lon in range(-180, 181, step_degrees):
        for lat in range(-80, 81, 2):
            rows.append({"lon": float(lon), "lat": float(lat), "path_id": f"lon_{path_index}"})
        path_index += 1
    return pd.DataFrame(rows)


def world_frame_paths() -> pd.DataFrame:
    """Build a simple world extent frame in lon/lat coordinates.

    Inputs:
        None.

    Returns:
        A DataFrame containing the outer Web Mercator-safe geographic frame.
    """
    return pd.DataFrame(
        {
            "lon": [-180.0, 180.0, 180.0, -180.0, -180.0],
            "lat": [-80.0, -80.0, 80.0, 80.0, -80.0],
            "path_id": ["world_frame"] * 5,
        }
    )


def load_pipeline_outputs(assignments_path: str, summary_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load point assignments and cluster summary parquet outputs.

    Inputs:
        assignments_path: Path to the point assignments parquet file.
        summary_path: Path to the cluster summary parquet file.

    Returns:
        A pair of Polars DataFrames for assignments and cluster summaries.
    """
    return pl.read_parquet(assignments_path), pl.read_parquet(summary_path)


def pick_label_column(columns: Iterable[str]) -> str | None:
    """Choose a reasonable label column name from a dataset.

    Inputs:
        columns: Column names available in the dataset.

    Returns:
        The selected label column name, or ``None`` if no suitable label exists.
    """
    normalized = {column.lower(): column for column in columns}
    for candidate in ("name", "ascii_name", "city_name", "lake_name", "label", "id"):
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_reference_layer(path: str | None) -> gpd.GeoDataFrame | None:
    """Load a city or lake reference layer from vector or parquet input.

    Inputs:
        path: Optional filesystem path to the reference layer.

    Returns:
        A GeoDataFrame in ``EPSG:4326``, or ``None`` when no readable path was provided.
    """
    if path is None:
        return None

    source_path = Path(path)
    if not source_path.exists():
        print(f"Reference layer not found, skipping: {source_path}")
        return None

    if source_path.suffix.lower() == ".parquet":
        try:
            gdf = gpd.read_parquet(source_path)
        except Exception:
            pdf = pl.read_parquet(source_path).to_pandas()
            if {"lon", "lat"}.issubset(pdf.columns):
                gdf = gpd.GeoDataFrame(
                    pdf,
                    geometry=gpd.points_from_xy(pdf["lon"], pdf["lat"]),
                    crs="EPSG:4326",
                )
            else:
                raise ValueError(f"Parquet reference layer must contain geometry or lat/lon columns: {source_path}")
    else:
        gdf = gpd.read_file(source_path)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    label_column = pick_label_column(gdf.columns)
    gdf = gdf.copy()
    if label_column is None:
        gdf["_label"] = [f"feature_{idx}" for idx in range(len(gdf))]
    else:
        gdf["_label"] = gdf[label_column].astype(str)
    return gdf


def meters_per_lon_degree(lat_deg: float) -> float:
    """Estimate meters represented by one longitude degree at a latitude.

    Inputs:
        lat_deg: Latitude in degrees.

    Returns:
        Approximate meters per degree of longitude.
    """
    return 111_320.0 * max(np.cos(np.radians(lat_deg)), 1e-6)


def lonlat_to_web_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert longitude/latitude arrays into Web Mercator coordinates.

    Inputs:
        lon: Longitudes in degrees.
        lat: Latitudes in degrees.

    Returns:
        A pair of NumPy arrays containing Web Mercator x and y coordinates in meters.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.clip(np.asarray(lat, dtype=float), -WEB_MERCATOR_LIMIT_LAT, WEB_MERCATOR_LIMIT_LAT)
    x = lon * WEB_MERCATOR_ORIGIN_SHIFT / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    y = y * WEB_MERCATOR_ORIGIN_SHIFT / 180.0
    return x, y


def with_web_mercator_polars(df: pl.DataFrame, lon_col: str = "lon", lat_col: str = "lat") -> pl.DataFrame:
    """Append Web Mercator coordinates to a Polars DataFrame.

    Inputs:
        df: Polars DataFrame containing longitude and latitude columns.
        lon_col: Name of the longitude column in degrees.
        lat_col: Name of the latitude column in degrees.

    Returns:
        A Polars DataFrame with added ``x`` and ``y`` columns in Web Mercator meters.
    """
    x, y = lonlat_to_web_mercator(df[lon_col].to_numpy(), df[lat_col].to_numpy())
    return df.with_columns([pl.Series("x", x), pl.Series("y", y)])


def with_web_mercator_pandas(df: pd.DataFrame, lon_col: str = "lon", lat_col: str = "lat") -> pd.DataFrame:
    """Append Web Mercator coordinates to a pandas DataFrame.

    Inputs:
        df: pandas DataFrame containing longitude and latitude columns.
        lon_col: Name of the longitude column in degrees.
        lat_col: Name of the latitude column in degrees.

    Returns:
        A pandas DataFrame with added ``x`` and ``y`` columns in Web Mercator meters.
    """
    out = df.copy()
    x, y = lonlat_to_web_mercator(out[lon_col].to_numpy(), out[lat_col].to_numpy())
    out["x"] = x
    out["y"] = y
    return out


def ellipse_parameters_for_cluster(
    cluster_points: pl.DataFrame,
    center_lat: float,
    center_lon: float,
    fallback_radius_m: float,
) -> tuple[float, float, float]:
    """Estimate ellipse axes and rotation for one cluster.

    Inputs:
        cluster_points: DataFrame containing the cluster's detection points.
        center_lat: Cluster center latitude in degrees.
        center_lon: Cluster center longitude in degrees.
        fallback_radius_m: Radius to use when the point cloud is too small or degenerate.

    Returns:
        Major axis length in meters, minor axis length in meters, and rotation angle in degrees.
    """
    if cluster_points.height < 3:
        diameter_m = max(fallback_radius_m * 2.0, 1.0)
        return diameter_m, diameter_m, 0.0

    x_m = (cluster_points["lon"].to_numpy() - center_lon) * meters_per_lon_degree(center_lat)
    y_m = (cluster_points["lat"].to_numpy() - center_lat) * 111_320.0
    coords = np.column_stack([x_m, y_m])

    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    scale = 2.0
    major_m = max(scale * np.sqrt(eigvals[0]) * 2.0, fallback_radius_m * 2.0, 1.0)
    minor_m = max(scale * np.sqrt(eigvals[1]) * 2.0, min(fallback_radius_m * 2.0, major_m), 1.0)
    angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return major_m, minor_m, angle_deg


def ellipse_geometry(center_lon: float, center_lat: float, major_m: float, minor_m: float, angle_deg: float) -> Point:
    """Build an ellipse polygon around a cluster center.

    Inputs:
        center_lon: Cluster center longitude in degrees.
        center_lat: Cluster center latitude in degrees.
        major_m: Ellipse major-axis length in meters.
        minor_m: Ellipse minor-axis length in meters.
        angle_deg: Ellipse rotation angle in degrees.

    Returns:
        A shapely polygon approximating the ellipse in ``EPSG:4326``.
    """
    unit_circle = Point(center_lon, center_lat).buffer(1.0, quad_segs=96)
    ellipse = affinity.scale(
        unit_circle,
        xfact=(major_m / 2.0) / meters_per_lon_degree(center_lat),
        yfact=(minor_m / 2.0) / 111_320.0,
        origin=(center_lon, center_lat),
    )
    return affinity.rotate(ellipse, angle_deg, origin=(center_lon, center_lat))


def geometry_to_path_rows(geometry: BaseGeometry, group: str) -> list[dict[str, float | str | int]]:
    """Convert a geometry into a row set consumable by ``hvplot.paths``.

    Inputs:
        geometry: Shapely geometry to convert.
        group: Group label to assign to the generated path rows.

    Returns:
        A list of dictionaries with path coordinates and path group identifiers.
    """
    rows: list[dict[str, float | str | int]] = []

    def append_line(coords: Iterable[tuple[float, float]], path_id: int) -> None:
        for lon, lat in coords:
            rows.append({"lon": float(lon), "lat": float(lat), "group": group, "path_id": path_id})

    if geometry.geom_type == "Polygon":
        append_line(list(geometry.exterior.coords), 0)
    elif geometry.geom_type == "MultiPolygon":
        for path_id, polygon in enumerate(geometry.geoms):
            append_line(list(polygon.exterior.coords), path_id)
    elif geometry.geom_type == "LineString":
        append_line(list(geometry.coords), 0)
    elif geometry.geom_type == "MultiLineString":
        for path_id, line in enumerate(geometry.geoms):
            append_line(list(line.coords), path_id)
    return rows


def geodataframe_to_paths(gdf: gpd.GeoDataFrame, group: str) -> pd.DataFrame:
    """Convert a GeoDataFrame of linear or polygon features into path rows.

    Inputs:
        gdf: GeoDataFrame containing geometries in ``EPSG:4326``.
        group: Group label to assign to the generated paths.

    Returns:
        A pandas DataFrame with ``lon``, ``lat``, ``group``, and ``path_id`` columns.
    """
    rows: list[dict[str, float | str | int]] = []
    for idx, geometry in enumerate(gdf.geometry):
        for row in geometry_to_path_rows(geometry, group):
            row["path_id"] = f"{idx}_{row['path_id']}"
            rows.append(row)
    return pd.DataFrame(rows)


def find_nearby_features(
    gdf: gpd.GeoDataFrame | None,
    center_lat: float,
    center_lon: float,
    search_radius_m: float,
) -> gpd.GeoDataFrame | None:
    """Select reference features near a cluster center.

    Inputs:
        gdf: Optional reference GeoDataFrame in ``EPSG:4326``.
        center_lat: Cluster center latitude in degrees.
        center_lon: Cluster center longitude in degrees.
        search_radius_m: Search radius around the cluster center in meters.

    Returns:
        A GeoDataFrame containing only nearby features, or ``None`` when no layer was provided.
    """
    if gdf is None or gdf.empty:
        return None

    center = gpd.GeoSeries([Point(center_lon, center_lat)], crs="EPSG:4326")
    local_crs = center.estimate_utm_crs()
    center_local = center.to_crs(local_crs)
    layer_local = gdf.to_crs(local_crs)
    search_geometry = center_local.buffer(search_radius_m).iloc[0]

    if layer_local.geom_type.isin(["Point", "MultiPoint"]).all():
        distances = layer_local.distance(center_local.iloc[0])
        nearby = layer_local.loc[distances <= search_radius_m]
    else:
        nearby = layer_local.loc[layer_local.intersects(search_geometry)]
    if nearby.empty:
        return nearby
    return nearby.to_crs("EPSG:4326")


def label_positions(features: gpd.GeoDataFrame | None) -> pd.DataFrame:
    """Prepare label positions for a reference layer.

    Inputs:
        features: GeoDataFrame of nearby features in ``EPSG:4326``.

    Returns:
        A DataFrame with ``lon``, ``lat``, and ``label`` columns for annotations.
    """
    if features is None or features.empty:
        return pd.DataFrame(columns=["lon", "lat", "label"])

    anchors = features.geometry.apply(lambda geom: geom.centroid if geom.geom_type != "Point" else geom)
    return pd.DataFrame(
        {
            "lon": anchors.x,
            "lat": anchors.y,
            "label": features["_label"].astype(str).to_list(),
        }
    )


def polygon_overlay(
    features: gpd.GeoDataFrame | None,
    color: str,
    label: str,
    line_width: float,
    alpha: float,
    dash: str = "solid",
) -> hv.Element | None:
    """Convert polygon or line reference features into a path overlay.

    Inputs:
        features: Nearby feature layer in ``EPSG:4326``.
        color: Stroke color for the overlay.
        label: Legend label for the overlay.
        line_width: Stroke width in screen pixels.
        alpha: Stroke transparency.
        dash: Line dash style.

    Returns:
        A HoloViews element when drawable polygon or line features exist, otherwise ``None``.
    """
    if features is None or features.empty:
        return None
    drawable = features.loc[features.geom_type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])]
    if drawable.empty:
        return None
    path_df = geodataframe_to_paths(drawable, group=label)
    if path_df.empty:
        return None
    return with_web_mercator_pandas(path_df).hvplot.paths(
        x="x",
        y="y",
        by="path_id",
        color=color,
        line_width=line_width,
        alpha=alpha,
        line_dash=dash,
        label=label,
    )


def build_overlay(
    assignments: pl.DataFrame,
    summary: pl.DataFrame,
    land: gpd.GeoDataFrame | None,
    cities: gpd.GeoDataFrame | None,
    lakes: gpd.GeoDataFrame | None,
    nearby_radius_km: float,
    max_clusters: int | None,
    include_noise: bool,
    basemap_name: str,
) -> hv.Overlay:
    """Build an hvPlot overlay for clustered detections and nearby features.

    Inputs:
        assignments: Point assignments from the pipeline output.
        summary: Cluster summary from the pipeline output.
        land: Optional land polygon layer for offline context.
        cities: Optional city reference layer.
        lakes: Optional lake reference layer.
        nearby_radius_km: Extra search radius around each cluster in kilometers.
        max_clusters: Optional maximum number of clusters to plot.
        include_noise: Whether to draw unclustered detections.
        basemap_name: Named tile basemap style, or ``none`` for no remote tiles.

    Returns:
        A HoloViews overlay representing the visualization.
    """
    if summary.is_empty():
        raise ValueError("Cluster summary is empty; nothing to plot.")

    summary = summary.sort(["n_points", "cluster_id"], descending=[True, False])
    if max_clusters is not None:
        summary = summary.head(max_clusters)

    cluster_ids = set(summary["cluster_id"].to_list())
    clustered_points = assignments.filter(pl.col("cluster_id").is_in(list(cluster_ids)))
    noise_points = assignments.filter(pl.col("cluster_id") == -1)

    overlays: list[hv.Element] = []

    if basemap_name == "none":
        graticule = with_web_mercator_pandas(graticule_paths()).hvplot.paths(
            x="x",
            y="y",
            by="path_id",
            color="#c9d2d9",
            alpha=0.6,
            line_width=0.8,
            legend=False,
        )
        world_frame = with_web_mercator_pandas(world_frame_paths()).hvplot.paths(
            x="x",
            y="y",
            by="path_id",
            color="#8e9aa3",
            alpha=0.9,
            line_width=1.2,
            legend=False,
        )
        overlays.extend([graticule, world_frame])
        land_overlay = polygon_overlay(
            land,
            color="#97a97c",
            label="Land",
            line_width=1.0,
            alpha=0.55,
        )
        if land_overlay is not None:
            overlays.append(land_overlay)

    if include_noise and not noise_points.is_empty():
        overlays.append(
            with_web_mercator_polars(noise_points).hvplot.points(
                x="x",
                y="y",
                color="lightgray",
                alpha=0.35,
                size=8,
                label="Unclustered detections",
            )
        )

    palette = ["#0f4c5c", "#e36414", "#5f0f40", "#6a994e", "#3a86ff", "#ff006e", "#8338ec", "#2a9d8f", "#f4a261", "#264653"]
    plotted_city_label = False
    plotted_lake_label = False

    for idx, cluster in enumerate(summary.iter_rows(named=True)):
        color = palette[idx % len(palette)]
        cluster_id = int(cluster["cluster_id"])
        center_lat = float(cluster["center_lat"])
        center_lon = float(cluster["center_lon"])
        radius_m = float(cluster["radius_m"])
        cluster_points = clustered_points.filter(pl.col("cluster_id") == cluster_id).with_columns(
            pl.lit(f"Cluster {cluster_id}").alias("cluster_name")
        )

        overlays.append(
            with_web_mercator_polars(cluster_points).hvplot.points(
                x="x",
                y="y",
                color="white",
                alpha=0.9,
                size=16,
                marker="circle",
                legend=False,
                hover=False,
            )
        )
        overlays.append(
            with_web_mercator_polars(cluster_points).hvplot.points(
                x="x",
                y="y",
                color=color,
                alpha=0.95,
                size=10,
                label=f"Cluster {cluster_id}",
                hover_cols=["id", "time", "cluster_id", "neighbor_count"],
            )
        )

        centroid_df = pl.DataFrame(
            {
                "lon": [center_lon],
                "lat": [center_lat],
                "label": [f"Cluster {cluster_id} centroid"],
            },
        )
        overlays.append(
            with_web_mercator_polars(centroid_df).hvplot.points(
                x="x",
                y="y",
                marker="x",
                color=color,
                size=90,
                line_width=2,
                legend=False,
                hover_cols=["label"],
            )
        )

        major_m, minor_m, angle_deg = ellipse_parameters_for_cluster(
            cluster_points=cluster_points,
            center_lat=center_lat,
            center_lon=center_lon,
            fallback_radius_m=radius_m,
        )
        ellipse_df = pl.DataFrame(
            geometry_to_path_rows(
                ellipse_geometry(center_lon, center_lat, major_m, minor_m, angle_deg),
                group=f"ellipse_{cluster_id}",
            )
        )
        overlays.append(
            with_web_mercator_polars(ellipse_df).hvplot.paths(
                x="x",
                y="y",
                by="group",
                color=color,
                alpha=0.9,
                line_width=2,
                legend=False,
            )
        )

        search_radius_m = radius_m + nearby_radius_km * 1000.0
        nearby_cities = find_nearby_features(cities, center_lat, center_lon, search_radius_m)
        nearby_lakes = find_nearby_features(lakes, center_lat, center_lon, search_radius_m)

        if nearby_cities is not None and not nearby_cities.empty:
            city_polygon_paths = polygon_overlay(
                nearby_cities,
                color="#111111",
                label="Nearby city polygons" if not plotted_city_label else "",
                line_width=1.8,
                alpha=0.7,
                dash="dashed",
            )
            if city_polygon_paths is not None:
                overlays.append(city_polygon_paths)
            city_labels = with_web_mercator_pandas(label_positions(nearby_cities))
            city_points = pl.from_pandas(city_labels)
            overlays.append(
                city_points.hvplot.points(
                    x="x",
                    y="y",
                    marker="triangle",
                    color="black",
                    size=65,
                    label="Nearby cities" if not plotted_city_label else "",
                )
            )
            plotted_city_label = True
            overlays.append(hv.Labels(city_labels, ["x", "y"], "label").opts(text_color="black", text_font_size="8pt"))

        if nearby_lakes is not None and not nearby_lakes.empty:
            lake_polygon_paths = polygon_overlay(
                nearby_lakes,
                color="royalblue",
                label="Nearby lake polygons" if not plotted_lake_label else "",
                line_width=2.0,
                alpha=0.85,
            )
            if lake_polygon_paths is not None:
                overlays.append(lake_polygon_paths)
            if nearby_lakes.geom_type.isin(["Point", "MultiPoint"]).all():
                lake_points = pl.from_pandas(with_web_mercator_pandas(label_positions(nearby_lakes)))
                overlays.append(
                    lake_points.hvplot.points(
                        x="x",
                        y="y",
                        marker="diamond",
                        color="royalblue",
                        size=55,
                        label="Nearby lakes" if not plotted_lake_label else "",
                    )
                )
            plotted_lake_label = True
            overlays.append(hv.Labels(with_web_mercator_pandas(label_positions(nearby_lakes)), ["x", "y"], "label").opts(text_color="royalblue", text_font_size="8pt"))

    base = resolve_basemap(basemap_name)
    if base is not None:
        base = base.opts(alpha=0.8)
        composed: hv.Element = base * hv.Overlay(overlays)
    else:
        composed = hv.Overlay(overlays)
    overlay = composed.opts(
        width=1200,
        height=800,
        title="Detection Clusters with Centroids, Ellipses, Nearby Features, and Basemap",
        xlabel="Web Mercator X",
        ylabel="Web Mercator Y",
        legend_position="right",
        show_grid=True,
        active_tools=["pan", "wheel_zoom"],
        padding=0.08,
    )
    return overlay


def plot_detections(
    assignments_path: str,
    summary_path: str,
    output_path: str,
    land_path: str | None,
    city_path: str | None,
    lake_path: str | None,
    nearby_radius_km: float,
    max_clusters: int | None,
    include_noise: bool,
    basemap_name: str = "cartolight",
) -> None:
    """Render an hvPlot visualization of clustered detections and nearby features.

    Inputs:
        assignments_path: Path to ``point_assignments.parquet``.
        summary_path: Path to ``cluster_summary.parquet``.
        output_path: Destination HTML path.
        land_path: Optional land polygon layer path for offline world context.
        city_path: Optional city reference layer path.
        lake_path: Optional lake reference layer path.
        nearby_radius_km: Extra search radius around each cluster in kilometers.
        max_clusters: Optional maximum number of clusters to plot, sorted by size.
        include_noise: Whether to draw unclustered detections in the background.
        basemap_name: Named basemap style, or ``none`` for no remote tiles.

    Returns:
        None. The function writes an interactive HTML visualization to disk.
    """
    assignments, summary = load_pipeline_outputs(assignments_path, summary_path)
    land = load_reference_layer(land_path)
    cities = load_reference_layer(city_path)
    lakes = load_reference_layer(lake_path)
    overlay = build_overlay(assignments, summary, land, cities, lakes, nearby_radius_km, max_clusters, include_noise, basemap_name)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    hv.save(overlay, output, backend="bokeh")
    print(f"Wrote plot → {output}")


def main() -> None:
    """Parse CLI arguments and render an interactive cluster visualization.

    Inputs:
        None. Arguments are read from the command line.

    Returns:
        None. The function writes an HTML visualization to disk.
    """
    parser = argparse.ArgumentParser(
        description="Plot clustered detections with hvPlot, centroids, ellipses, and nearby cities/lakes."
    )
    parser.add_argument("--assignments", default="out/point_assignments.parquet", help="Path to point assignments parquet")
    parser.add_argument("--summary", default="out/cluster_summary.parquet", help="Path to cluster summary parquet")
    parser.add_argument("--output", default="images/detection_clusters.html", help="Output HTML path")
    parser.add_argument(
        "--land-path",
        default=default_existing_path(
            [
                "data/unpacked/ne_10m_land/ne_10m_land.shp",
                "data/land.gpkg",
            ]
        ),
        help="Optional land polygon layer path for offline context when basemap=none",
    )
    parser.add_argument(
        "--city-path",
        default=default_existing_path(["data/cities_global.gpkg", "data/cities_global.parquet"]),
        help="Optional city layer path; defaults to a known local dataset when available",
    )
    parser.add_argument(
        "--lake-path",
        default=default_existing_path(
            [
                "data/lakes.parquet",
                "data/unpacked/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp",
            ]
        ),
        help="Optional lake layer path; defaults to a known local dataset when available",
    )
    parser.add_argument("--nearby-radius-km", type=float, default=50.0, help="Extra radius around each cluster for nearby feature search")
    parser.add_argument("--max-clusters", type=int, default=None, help="Optional maximum number of largest clusters to plot")
    parser.add_argument("--hide-noise", action="store_true", help="Hide detections with cluster_id = -1")
    parser.add_argument(
        "--basemap",
        default="none",
        choices=["none", "osm", "cartolight", "cartodark", "esriimagery", "esristreet", "esriterrain", "opentopo"],
        help="Basemap style for the interactive plot. 'none' is the default to avoid 403 tile failures.",
    )
    args = parser.parse_args()

    plot_detections(
        assignments_path=args.assignments,
        summary_path=args.summary,
        output_path=args.output,
        land_path=args.land_path,
        city_path=args.city_path,
        lake_path=args.lake_path,
        nearby_radius_km=args.nearby_radius_km,
        max_clusters=args.max_clusters,
        include_noise=not args.hide_noise,
        basemap_name=args.basemap,
    )


if __name__ == "__main__":
    main()
