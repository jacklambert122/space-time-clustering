from __future__ import annotations

"""Summarize surviving cluster assignments into per-cluster output records."""

import numpy as np
import polars as pl

from .config import ClusterConfig
from .kalman import constant_position_kalman_feature_dict
from .spatial import haversine_matrix_m


def meters_per_lon_degree(lat_deg: float) -> float:
    """Estimate meters represented by one longitude degree at a latitude.

    Inputs:
        lat_deg: Latitude in degrees.

    Returns:
        Approximate meters per degree of longitude.
    """
    return 111_320.0 * max(float(np.cos(np.radians(lat_deg))), 1e-6)


def project_to_local_m(
    lat: np.ndarray,
    lon: np.ndarray,
    ref_lat: float,
    ref_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat coordinates into a local tangent-plane approximation in meters.

    Inputs:
        lat: Latitudes in degrees.
        lon: Longitudes in degrees.
        ref_lat: Reference latitude in degrees.
        ref_lon: Reference longitude in degrees.

    Returns:
        Arrays of local x and y coordinates in meters relative to the reference point.
    """
    x_m = (lon - ref_lon) * meters_per_lon_degree(ref_lat)
    y_m = (lat - ref_lat) * 111_320.0
    return x_m, y_m


def ellipse_spread_features(x_m: np.ndarray, y_m: np.ndarray) -> tuple[float, float, float]:
    """Compute spatial spread features from local cluster coordinates.

    Inputs:
        x_m: Local x coordinates in meters.
        y_m: Local y coordinates in meters.

    Returns:
        Major-axis standard deviation in meters, minor-axis standard deviation in meters,
        and axis ratio.
    """
    if len(x_m) < 2:
        return 0.0, 0.0, 1.0
    cov = np.cov(np.column_stack([x_m, y_m]), rowvar=False)
    eigvals = np.clip(np.linalg.eigvalsh(cov), a_min=0.0, a_max=None)
    eigvals.sort()
    minor_std = float(np.sqrt(eigvals[0]))
    major_std = float(np.sqrt(eigvals[1]))
    axis_ratio = major_std / max(minor_std, 1e-6)
    return major_std, minor_std, axis_ratio


def step_distance_features(lat: np.ndarray, lon: np.ndarray, time_s: np.ndarray) -> dict[str, float]:
    """Compute path and speed features from time-ordered detections.

    Inputs:
        lat: Latitudes in degrees.
        lon: Longitudes in degrees.
        time_s: Detection times in unix seconds.

    Returns:
        A dictionary of path-length, displacement, step-distance, and speed features.
    """
    if len(lat) < 2:
        return {
            "path_length_m": 0.0,
            "net_displacement_m": 0.0,
            "mean_step_distance_m": 0.0,
            "max_step_distance_m": 0.0,
            "mean_speed_mps": 0.0,
            "max_speed_mps": 0.0,
            "meander_ratio": 1.0,
        }

    order = np.argsort(time_s)
    lat_sorted = lat[order]
    lon_sorted = lon[order]
    time_sorted = time_s[order]
    step_distances = haversine_matrix_m(
        lat_sorted[:-1],
        lon_sorted[:-1],
        lat_sorted[1:],
        lon_sorted[1:],
    ).diagonal()
    dt = np.maximum(np.diff(time_sorted), 1e-6)
    speeds = step_distances / dt
    net_displacement = float(
        haversine_matrix_m(
            np.array([lat_sorted[0]]),
            np.array([lon_sorted[0]]),
            np.array([lat_sorted[-1]]),
            np.array([lon_sorted[-1]]),
        )[0, 0]
    )
    path_length = float(step_distances.sum())
    return {
        "path_length_m": path_length,
        "net_displacement_m": net_displacement,
        "mean_step_distance_m": float(step_distances.mean()),
        "max_step_distance_m": float(step_distances.max()),
        "mean_speed_mps": float(speeds.mean()),
        "max_speed_mps": float(speeds.max()),
        "meander_ratio": path_length / max(net_displacement, 1.0),
    }


def summarize_clusters(
    time_s: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    labels: np.ndarray,
    cfg: ClusterConfig,
) -> pl.DataFrame:
    """Aggregate surviving cluster labels into a per-cluster summary table.

    Inputs:
        time_s: Detection timestamps in unix seconds.
        lat: Detection latitudes in degrees.
        lon: Detection longitudes in degrees.
        labels: Final cluster labels for each detection, with ``-1`` for dropped rows.
        cfg: Pipeline configuration supplying summary parameters.

    Returns:
        A Polars DataFrame with one row per surviving cluster.
    """
    empty_columns = {
        "cluster_id": [],
        "n_points": [],
        "start_time": [],
        "end_time": [],
        "duration_s": [],
        "center_lat": [],
        "center_lon": [],
        "radius_m": [],
        "center_distance_mean_m": [],
        "center_distance_std_m": [],
        "spatial_major_std_m": [],
        "spatial_minor_std_m": [],
        "spatial_axis_ratio": [],
        "bbox_width_m": [],
        "bbox_height_m": [],
        "path_length_m": [],
        "net_displacement_m": [],
        "mean_step_distance_m": [],
        "max_step_distance_m": [],
        "mean_speed_mps": [],
        "max_speed_mps": [],
        "meander_ratio": [],
        "cp_kf_mean_innovation_m": [],
        "cp_kf_max_innovation_m": [],
        "cp_kf_rmse_m": [],
        "cp_kf_final_sigma_m": [],
        "cp_kf_mean_nis": [],
    }
    keep = labels != -1
    time_s = time_s[keep]
    lat = lat[keep]
    lon = lon[keep]
    labels = labels[keep]
    if len(labels) == 0:
        return pl.DataFrame(empty_columns)
    rows = []
    for cid in np.unique(labels):
        mask = labels == cid
        lat_c = lat[mask]
        lon_c = lon[mask]
        time_c = time_s[mask]
        center_lat = float(np.median(lat_c))
        center_lon = float(np.median(lon_c))
        dist = haversine_matrix_m(lat_c, lon_c, np.array([center_lat]), np.array([center_lon]))[:, 0]
        radius_m = float(np.quantile(dist, cfg.radius_quantile) + cfg.guard_band_m)
        x_m, y_m = project_to_local_m(lat_c, lon_c, center_lat, center_lon)
        major_std_m, minor_std_m, axis_ratio = ellipse_spread_features(x_m, y_m)
        path_features = step_distance_features(lat_c, lon_c, time_c)
        kalman_features = constant_position_kalman_feature_dict(
            x_m,
            y_m,
            process_noise_var_m2=cfg.kf_process_noise_var_m2,
            measurement_noise_var_m2=cfg.kf_measurement_noise_var_m2,
        )
        rows.append({
            "cluster_id": int(cid),
            "n_points": int(mask.sum()),
            "start_time": float(time_c.min()),
            "end_time": float(time_c.max()),
            "duration_s": float(time_c.max() - time_c.min()),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_m": radius_m,
            "center_distance_mean_m": float(dist.mean()),
            "center_distance_std_m": float(dist.std(ddof=0)),
            "spatial_major_std_m": major_std_m,
            "spatial_minor_std_m": minor_std_m,
            "spatial_axis_ratio": axis_ratio,
            "bbox_width_m": float((lon_c.max() - lon_c.min()) * meters_per_lon_degree(center_lat)),
            "bbox_height_m": float((lat_c.max() - lat_c.min()) * 111_320.0),
            **path_features,
            **kalman_features,
        })
    return pl.DataFrame(rows).sort("cluster_id")
