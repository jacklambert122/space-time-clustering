from __future__ import annotations

"""Kalman-filter and single-satellite observation helpers for tracking analysis."""

import math
from typing import Literal

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371000.0
WEB_MERCATOR_LIMIT_LAT = 85.05112878
WEB_MERCATOR_ORIGIN_SHIFT = 20037508.34
KalmanMotionModel = Literal["constant_position", "constant_velocity", "constant_acceleration"]


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


def local_m_to_latlon(
    x_m: np.ndarray,
    y_m: np.ndarray,
    ref_lat: float,
    ref_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local tangent-plane coordinates back to latitude/longitude.

    Inputs:
        x_m: Local x coordinates in meters.
        y_m: Local y coordinates in meters.
        ref_lat: Reference latitude in degrees.
        ref_lon: Reference longitude in degrees.

    Returns:
        Arrays of latitudes and longitudes in degrees.
    """
    lat = ref_lat + y_m / 111_320.0
    lon = ref_lon + x_m / meters_per_lon_degree(ref_lat)
    return lat, lon


def lonlat_to_web_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert longitude/latitude arrays into Web Mercator coordinates.

    Inputs:
        lon: Longitudes in degrees.
        lat: Latitudes in degrees.

    Returns:
        Arrays of Web Mercator x and y coordinates in meters.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.clip(np.asarray(lat, dtype=float), -WEB_MERCATOR_LIMIT_LAT, WEB_MERCATOR_LIMIT_LAT)
    x = lon * WEB_MERCATOR_ORIGIN_SHIFT / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    y = y * WEB_MERCATOR_ORIGIN_SHIFT / 180.0
    return x, y


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert a geodetic point on a spherical Earth to ECEF coordinates.

    Inputs:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        alt_m: Altitude above the spherical Earth in meters.

    Returns:
        A length-3 NumPy vector of ECEF coordinates in meters.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    radius = EARTH_RADIUS_M + alt_m
    return np.array(
        [
            radius * math.cos(lat) * math.cos(lon),
            radius * math.cos(lat) * math.sin(lon),
            radius * math.sin(lat),
        ],
        dtype=float,
    )


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[float, float, float]:
    """Convert a spherical-Earth ECEF point back to geodetic coordinates.

    Inputs:
        xyz: Length-3 ECEF coordinate vector in meters.

    Returns:
        Latitude in degrees, longitude in degrees, and altitude in meters.
    """
    radius = float(np.linalg.norm(xyz))
    lat = math.degrees(math.asin(xyz[2] / radius))
    lon = math.degrees(math.atan2(xyz[1], xyz[0]))
    alt = radius - EARTH_RADIUS_M
    return lat, lon, alt


def enu_rotation_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Build the rotation matrix from ECEF coordinates to local ENU coordinates.

    Inputs:
        lat_deg: Observer latitude in degrees.
        lon_deg: Observer longitude in degrees.

    Returns:
        A 3x3 NumPy rotation matrix mapping ECEF vectors into ENU space.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array(
        [
            -math.sin(lat) * math.cos(lon),
            -math.sin(lat) * math.sin(lon),
            math.cos(lat),
        ]
    )
    up = np.array(
        [
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ]
    )
    return np.stack([east, north, up], axis=0)


def azel_from_satellite(
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_m: float,
    target_lat_deg: float,
    target_lon_deg: float,
    target_alt_m: float = 0.0,
) -> tuple[float, float]:
    """Compute azimuth/elevation from a single satellite to a ground target.

    Inputs:
        sat_lat_deg: Satellite latitude in degrees.
        sat_lon_deg: Satellite longitude in degrees.
        sat_alt_m: Satellite altitude in meters.
        target_lat_deg: Target latitude in degrees.
        target_lon_deg: Target longitude in degrees.
        target_alt_m: Target altitude in meters.

    Returns:
        Azimuth in degrees and elevation in degrees in the satellite ENU frame.
    """
    sat_ecef = geodetic_to_ecef(sat_lat_deg, sat_lon_deg, sat_alt_m)
    target_ecef = geodetic_to_ecef(target_lat_deg, target_lon_deg, target_alt_m)
    los_ecef = target_ecef - sat_ecef
    los_enu = enu_rotation_matrix(sat_lat_deg, sat_lon_deg) @ los_ecef
    east, north, up = los_enu
    az_deg = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    horizontal = math.hypot(east, north)
    el_deg = math.degrees(math.atan2(up, horizontal))
    return az_deg, el_deg


def line_of_sight_ecef(
    sat_lat_deg: float,
    sat_lon_deg: float,
    az_deg: float,
    el_deg: float,
) -> np.ndarray:
    """Convert satellite azimuth/elevation angles into an ECEF line-of-sight vector.

    Inputs:
        sat_lat_deg: Satellite latitude in degrees.
        sat_lon_deg: Satellite longitude in degrees.
        az_deg: Azimuth angle in degrees.
        el_deg: Elevation angle in degrees.

    Returns:
        A unit-length ECEF line-of-sight vector.
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    los_enu = np.array(
        [
            math.cos(el) * math.sin(az),
            math.cos(el) * math.cos(az),
            math.sin(el),
        ],
        dtype=float,
    )
    rotation = enu_rotation_matrix(sat_lat_deg, sat_lon_deg)
    los_ecef = rotation.T @ los_enu
    return los_ecef / np.linalg.norm(los_ecef)


def azel_from_los_ecef(
    sat_lat_deg: float,
    sat_lon_deg: float,
    los_ecef: np.ndarray,
) -> tuple[float, float]:
    """Convert an ECEF line-of-sight vector into azimuth/elevation angles.

    Inputs:
        sat_lat_deg: Satellite latitude in degrees.
        sat_lon_deg: Satellite longitude in degrees.
        los_ecef: Line-of-sight direction vector in ECEF coordinates.

    Returns:
        Azimuth in degrees and elevation in degrees in the satellite ENU frame.
    """
    los_unit = np.asarray(los_ecef, dtype=float)
    los_unit = los_unit / np.linalg.norm(los_unit)
    los_enu = enu_rotation_matrix(sat_lat_deg, sat_lon_deg) @ los_unit
    east, north, up = los_enu
    az_deg = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    horizontal = math.hypot(east, north)
    el_deg = math.degrees(math.atan2(up, horizontal))
    return az_deg, el_deg


def target_line_of_sight_ecef(
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_m: float,
    target_lat_deg: float,
    target_lon_deg: float,
    target_alt_m: float = 0.0,
) -> np.ndarray:
    """Build the unit ECEF line of sight from a satellite to a target point.

    Inputs:
        sat_lat_deg: Satellite latitude in degrees.
        sat_lon_deg: Satellite longitude in degrees.
        sat_alt_m: Satellite altitude in meters.
        target_lat_deg: Target latitude in degrees.
        target_lon_deg: Target longitude in degrees.
        target_alt_m: Target altitude in meters.

    Returns:
        A unit-length ECEF vector from the satellite toward the target.
    """
    sat_ecef = geodetic_to_ecef(sat_lat_deg, sat_lon_deg, sat_alt_m)
    target_ecef = geodetic_to_ecef(target_lat_deg, target_lon_deg, target_alt_m)
    los_ecef = target_ecef - sat_ecef
    return los_ecef / np.linalg.norm(los_ecef)


def cross_boresight_basis(los_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal basis perpendicular to a boresight vector.

    Inputs:
        los_ecef: Unit or non-unit boresight vector in ECEF coordinates.

    Returns:
        Two orthonormal ECEF vectors spanning the cross-boresight tangent plane.
    """
    boresight = np.asarray(los_ecef, dtype=float)
    boresight = boresight / np.linalg.norm(boresight)
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(helper, boresight))) > 0.95:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
    axis_1 = np.cross(helper, boresight)
    axis_1 = axis_1 / np.linalg.norm(axis_1)
    axis_2 = np.cross(boresight, axis_1)
    axis_2 = axis_2 / np.linalg.norm(axis_2)
    return axis_1, axis_2


def satellite_view_basis(
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a satellite-centric orthonormal basis for viewing the Earth disk.

    Inputs:
        sat_lat_deg: Satellite latitude in degrees.
        sat_lon_deg: Satellite longitude in degrees.
        sat_alt_m: Satellite altitude in meters.

    Returns:
        Three ECEF unit vectors: view-x, view-y, and nadir direction.
    """
    sat_ecef = geodetic_to_ecef(sat_lat_deg, sat_lon_deg, sat_alt_m)
    nadir = -sat_ecef / np.linalg.norm(sat_ecef)
    view_x, view_y = cross_boresight_basis(nadir)
    return view_x, view_y, nadir


def project_los_to_satellite_view(
    los_ecef: np.ndarray,
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_m: float,
) -> tuple[float, float, float]:
    """Project a line-of-sight vector into the satellite Earth-view frame.

    Inputs:
        los_ecef: Line-of-sight direction vector in ECEF coordinates.
        sat_lat_deg: Satellite latitude in degrees.
        sat_lon_deg: Satellite longitude in degrees.
        sat_alt_m: Satellite altitude in meters.

    Returns:
        Projected x, y, z coordinates in the satellite-centric view frame.
    """
    los_unit = np.asarray(los_ecef, dtype=float)
    los_unit = los_unit / np.linalg.norm(los_unit)
    view_x, view_y, nadir = satellite_view_basis(sat_lat_deg, sat_lon_deg, sat_alt_m)
    return (
        float(np.dot(los_unit, view_x)),
        float(np.dot(los_unit, view_y)),
        float(np.dot(los_unit, nadir)),
    )


def earth_limb_radius_in_view(sat_alt_m: float) -> float:
    """Return the projected Earth-limb radius in the satellite view plane.

    Inputs:
        sat_alt_m: Satellite altitude in meters.

    Returns:
        Radius of the visible Earth disk in the satellite-centric orthographic view.
    """
    sat_radius = EARTH_RADIUS_M + sat_alt_m
    return float(EARTH_RADIUS_M / sat_radius)


def perturb_line_of_sight_cross_boresight(
    los_ecef: np.ndarray,
    sigma_axis_1_deg: float,
    sigma_axis_2_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply orthonormal cross-boresight angular noise to a line of sight.

    Inputs:
        los_ecef: Nominal line-of-sight unit vector in ECEF coordinates.
        sigma_axis_1_deg: One-sigma angular noise for the first cross-boresight axis.
        sigma_axis_2_deg: One-sigma angular noise for the second cross-boresight axis.
        rng: NumPy random generator used to sample the perturbation.

    Returns:
        A perturbed unit-length ECEF line-of-sight vector.
    """
    boresight = np.asarray(los_ecef, dtype=float)
    boresight = boresight / np.linalg.norm(boresight)
    axis_1, axis_2 = cross_boresight_basis(boresight)
    delta_1 = rng.normal(0.0, math.radians(sigma_axis_1_deg))
    delta_2 = rng.normal(0.0, math.radians(sigma_axis_2_deg))
    perturbed = boresight + delta_1 * axis_1 + delta_2 * axis_2
    return perturbed / np.linalg.norm(perturbed)


def ground_point_from_los_ecef(
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_m: float,
    los_ecef: np.ndarray,
    target_alt_m: float = 0.0,
) -> tuple[float, float] | None:
    """Intersect an ECEF line of sight from a satellite with the Earth surface.

    Inputs:
        sat_lat_deg: Satellite latitude in degrees.
        sat_lon_deg: Satellite longitude in degrees.
        sat_alt_m: Satellite altitude in meters.
        los_ecef: Line-of-sight direction vector in ECEF coordinates.
        target_alt_m: Target altitude above the spherical Earth in meters.

    Returns:
        Target latitude/longitude in degrees, or ``None`` if no Earth intersection exists.
    """
    sat_ecef = geodetic_to_ecef(sat_lat_deg, sat_lon_deg, sat_alt_m)
    los_unit = np.asarray(los_ecef, dtype=float)
    los_unit = los_unit / np.linalg.norm(los_unit)
    target_radius = EARTH_RADIUS_M + target_alt_m

    a = float(np.dot(los_unit, los_unit))
    b = float(2.0 * np.dot(sat_ecef, los_unit))
    c = float(np.dot(sat_ecef, sat_ecef) - target_radius**2)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None

    sqrt_disc = math.sqrt(disc)
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    positive_roots = [root for root in roots if root > 0.0]
    if not positive_roots:
        return None

    target_ecef = sat_ecef + min(positive_roots) * los_unit
    lat_deg, lon_deg, _ = ecef_to_geodetic(target_ecef)
    return lat_deg, lon_deg


def ground_point_from_azel(
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_m: float,
    az_deg: float,
    el_deg: float,
    target_alt_m: float = 0.0,
) -> tuple[float, float] | None:
    """Intersect a satellite line of sight with the Earth surface.

    Inputs:
        sat_lat_deg: Satellite latitude in degrees.
        sat_lon_deg: Satellite longitude in degrees.
        sat_alt_m: Satellite altitude in meters.
        az_deg: Azimuth angle in degrees.
        el_deg: Elevation angle in degrees.
        target_alt_m: Target altitude above the spherical Earth in meters.

    Returns:
        Target latitude/longitude in degrees, or ``None`` if no Earth intersection exists.
    """
    los_ecef = line_of_sight_ecef(sat_lat_deg, sat_lon_deg, az_deg, el_deg)
    return ground_point_from_los_ecef(
        sat_lat_deg=sat_lat_deg,
        sat_lon_deg=sat_lon_deg,
        sat_alt_m=sat_alt_m,
        los_ecef=los_ecef,
        target_alt_m=target_alt_m,
    )


def is_ground_point_visible_from_satellite(
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_m: float,
    target_lat_deg: float,
    target_lon_deg: float,
    target_alt_m: float = 0.0,
    tolerance_m: float = 1.0,
) -> bool:
    """Return whether a ground target is on the visible Earth limb from the satellite.

    Inputs:
        sat_lat_deg: Satellite latitude in degrees.
        sat_lon_deg: Satellite longitude in degrees.
        sat_alt_m: Satellite altitude in meters.
        target_lat_deg: Target latitude in degrees.
        target_lon_deg: Target longitude in degrees.
        target_alt_m: Target altitude in meters.
        tolerance_m: Absolute tolerance for comparing intersection distance to target range.

    Returns:
        ``True`` when the target is the first Earth-surface intersection along the line of sight.
    """
    sat_ecef = geodetic_to_ecef(sat_lat_deg, sat_lon_deg, sat_alt_m)
    target_ecef = geodetic_to_ecef(target_lat_deg, target_lon_deg, target_alt_m)
    los = target_ecef - sat_ecef
    target_range_m = float(np.linalg.norm(los))
    if target_range_m == 0.0:
        return False
    los_unit = los / target_range_m
    target_radius = EARTH_RADIUS_M + target_alt_m

    a = float(np.dot(los_unit, los_unit))
    b = float(2.0 * np.dot(sat_ecef, los_unit))
    c = float(np.dot(sat_ecef, sat_ecef) - target_radius**2)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False

    sqrt_disc = math.sqrt(disc)
    roots = sorted(root for root in [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)] if root > 0.0)
    if not roots:
        return False
    return abs(roots[0] - target_range_m) <= tolerance_m


def kalman_model_matrices(
    model: KalmanMotionModel,
    dt_s: float,
    process_noise_var_m2: float,
    measurement_noise_var_m2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build state-space matrices for the selected kinematic Kalman model.

    Inputs:
        model: Motion model name.
        dt_s: Time step in seconds.
        process_noise_var_m2: Scalar driving-noise variance.
        measurement_noise_var_m2: Scalar measurement-noise variance on x/y position.

    Returns:
        State transition, process covariance, measurement matrix, and measurement covariance.
    """
    dt = float(dt_s)
    q = float(process_noise_var_m2)
    r = float(measurement_noise_var_m2)

    if model == "constant_position":
        f_1d = np.array([[1.0]], dtype=float)
        q_1d = np.array([[q]], dtype=float)
        h_1d = np.array([[1.0]], dtype=float)
    elif model == "constant_velocity":
        f_1d = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        q_1d = q * np.array(
            [
                [dt**4 / 4.0, dt**3 / 2.0],
                [dt**3 / 2.0, dt**2],
            ],
            dtype=float,
        )
        h_1d = np.array([[1.0, 0.0]], dtype=float)
    elif model == "constant_acceleration":
        f_1d = np.array(
            [
                [1.0, dt, 0.5 * dt**2],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        q_1d = q * np.array(
            [
                [dt**6 / 36.0, dt**5 / 12.0, dt**4 / 6.0],
                [dt**5 / 12.0, dt**4 / 4.0, dt**3 / 2.0],
                [dt**4 / 6.0, dt**3 / 2.0, dt**2],
            ],
            dtype=float,
        )
        h_1d = np.array([[1.0, 0.0, 0.0]], dtype=float)
    else:
        raise ValueError(f"Unsupported Kalman motion model: {model}")

    n_state_1d = f_1d.shape[0]
    f_mat = np.block(
        [
            [f_1d, np.zeros((n_state_1d, n_state_1d), dtype=float)],
            [np.zeros((n_state_1d, n_state_1d), dtype=float), f_1d],
        ]
    )
    q_mat = np.block(
        [
            [q_1d, np.zeros((n_state_1d, n_state_1d), dtype=float)],
            [np.zeros((n_state_1d, n_state_1d), dtype=float), q_1d],
        ]
    )
    h_mat = np.block(
        [
            [h_1d, np.zeros((1, n_state_1d), dtype=float)],
            [np.zeros((1, n_state_1d), dtype=float), h_1d],
        ]
    )
    r_mat = np.eye(2, dtype=float) * r
    return f_mat, q_mat, h_mat, r_mat


def acceleration_from_schedule(
    time_s: float,
    acceleration_schedule: list[dict[str, float]] | None,
) -> tuple[float, float]:
    """Return the scheduled east/north acceleration active at a time.

    Inputs:
        time_s: Simulation time in seconds.
        acceleration_schedule: Optional list of schedule rows with ``start_time_s``,
            ``end_time_s``, ``east_accel_mps2``, and ``north_accel_mps2`` keys.

    Returns:
        East and north acceleration in meters per second squared.
    """
    east_accel = 0.0
    north_accel = 0.0
    if not acceleration_schedule:
        return east_accel, north_accel
    for event in acceleration_schedule:
        if float(event["start_time_s"]) <= time_s < float(event["end_time_s"]):
            east_accel += float(event["east_accel_mps2"])
            north_accel += float(event["north_accel_mps2"])
    return east_accel, north_accel


def initial_kinematic_state(
    model: KalmanMotionModel,
    x0_m: float,
    y0_m: float,
) -> np.ndarray:
    """Build the initial state vector for a selected kinematic model.

    Inputs:
        model: Motion model name.
        x0_m: Initial x position in meters.
        y0_m: Initial y position in meters.

    Returns:
        Initial state vector.
    """
    if model == "constant_position":
        return np.array([x0_m, y0_m], dtype=float)
    if model == "constant_velocity":
        return np.array([x0_m, 0.0, y0_m, 0.0], dtype=float)
    if model == "constant_acceleration":
        return np.array([x0_m, 0.0, 0.0, y0_m, 0.0, 0.0], dtype=float)
    raise ValueError(f"Unsupported Kalman motion model: {model}")


def extract_kinematic_state_columns(
    model: KalmanMotionModel,
    states: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract position, velocity, and acceleration series from state histories.

    Inputs:
        model: Motion model name.
        states: Filtered state history with shape ``(n_steps, n_state)``.

    Returns:
        Arrays for filtered x/y position and optional velocity/acceleration terms.
    """
    n_steps = states.shape[0]
    nan_values = np.full(n_steps, np.nan, dtype=float)
    if model == "constant_position":
        return {
            "filtered_x_m": states[:, 0],
            "filtered_y_m": states[:, 1],
            "filtered_vx_mps": nan_values,
            "filtered_vy_mps": nan_values,
            "filtered_ax_mps2": nan_values,
            "filtered_ay_mps2": nan_values,
        }
    if model == "constant_velocity":
        return {
            "filtered_x_m": states[:, 0],
            "filtered_vx_mps": states[:, 1],
            "filtered_y_m": states[:, 2],
            "filtered_vy_mps": states[:, 3],
            "filtered_ax_mps2": nan_values,
            "filtered_ay_mps2": nan_values,
        }
    if model == "constant_acceleration":
        return {
            "filtered_x_m": states[:, 0],
            "filtered_vx_mps": states[:, 1],
            "filtered_ax_mps2": states[:, 2],
            "filtered_y_m": states[:, 3],
            "filtered_vy_mps": states[:, 4],
            "filtered_ay_mps2": states[:, 5],
        }
    raise ValueError(f"Unsupported Kalman motion model: {model}")


def position_state_indices(model: KalmanMotionModel) -> tuple[int, int]:
    """Return the state-vector indices for x and y position for a model.

    Inputs:
        model: Motion model name.

    Returns:
        Indices of the x-position and y-position states.
    """
    if model == "constant_position":
        return 0, 1
    if model == "constant_velocity":
        return 0, 2
    if model == "constant_acceleration":
        return 0, 3
    raise ValueError(f"Unsupported Kalman motion model: {model}")


def run_kinematic_kalman(
    x_m: np.ndarray,
    y_m: np.ndarray,
    dt_s: float,
    process_noise_var_m2: float,
    measurement_noise_var_m2: float,
    model: KalmanMotionModel,
) -> dict[str, np.ndarray]:
    """Run a selectable 2D kinematic Kalman filter on position measurements.

    Inputs:
        x_m: Measured x coordinates in meters.
        y_m: Measured y coordinates in meters.
        dt_s: Time step in seconds.
        process_noise_var_m2: Scalar driving-noise variance.
        measurement_noise_var_m2: Scalar measurement-noise variance.
        model: Motion model name.

    Returns:
        Filtered state summaries including position and optional velocity/acceleration columns.
    """
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    if len(x_m) == 0:
        empty = np.empty(0, dtype=float)
        return {
            "filtered_x_m": empty,
            "filtered_y_m": empty,
            "filtered_vx_mps": empty,
            "filtered_vy_mps": empty,
            "filtered_ax_mps2": empty,
            "filtered_ay_mps2": empty,
            "innovation_m": empty,
            "residual_m": empty,
            "nis": empty,
            "sigma_m": empty,
        }

    f_mat, q_mat, h_mat, r_mat = kalman_model_matrices(model, dt_s, process_noise_var_m2, measurement_noise_var_m2)
    state = initial_kinematic_state(model, float(x_m[0]), float(y_m[0]))
    cov = np.eye(len(state), dtype=float) * max(float(measurement_noise_var_m2), 1.0)

    filtered_states: list[np.ndarray] = []
    innovations: list[float] = []
    residuals: list[float] = []
    nis_values: list[float] = []
    sigma_values: list[float] = []

    x_idx, y_idx = position_state_indices(model)

    for x_obs, y_obs in zip(x_m, y_m):
        state_pred = f_mat @ state
        cov_pred = f_mat @ cov @ f_mat.T + q_mat
        z = np.array([x_obs, y_obs], dtype=float)
        innovation = z - h_mat @ state_pred
        s_mat = h_mat @ cov_pred @ h_mat.T + r_mat
        k_gain = cov_pred @ h_mat.T @ np.linalg.inv(s_mat)
        state = state_pred + k_gain @ innovation
        cov = (np.eye(len(state), dtype=float) - k_gain @ h_mat) @ cov_pred

        filtered_states.append(state.copy())
        residual = z - h_mat @ state
        innovations.append(float(np.linalg.norm(innovation)))
        residuals.append(float(np.linalg.norm(residual)))
        nis_values.append(float(innovation.T @ np.linalg.inv(s_mat) @ innovation))
        sigma_values.append(float(np.sqrt((cov[x_idx, x_idx] + cov[y_idx, y_idx]) / 2.0)))

    state_columns = extract_kinematic_state_columns(model, np.asarray(filtered_states, dtype=float))
    return {
        **state_columns,
        "innovation_m": np.asarray(innovations, dtype=float),
        "residual_m": np.asarray(residuals, dtype=float),
        "nis": np.asarray(nis_values, dtype=float),
        "sigma_m": np.asarray(sigma_values, dtype=float),
    }


def run_constant_position_kalman(
    x_m: np.ndarray,
    y_m: np.ndarray,
    process_noise_var_m2: float,
    measurement_noise_var_m2: float,
) -> dict[str, np.ndarray]:
    """Run a constant-position Kalman filter on 2D measurements.

    Inputs:
        x_m: Measured x coordinates in meters.
        y_m: Measured y coordinates in meters.
        process_noise_var_m2: Scalar process-noise variance in square meters.
        measurement_noise_var_m2: Scalar measurement-noise variance in square meters.

    Returns:
        A dictionary containing filtered states, innovations, NIS values, and covariance sigmas.
    """
    return run_kinematic_kalman(
        x_m=x_m,
        y_m=y_m,
        dt_s=1.0,
        process_noise_var_m2=process_noise_var_m2,
        measurement_noise_var_m2=measurement_noise_var_m2,
        model="constant_position",
    )


def constant_position_kalman_feature_dict(
    x_m: np.ndarray,
    y_m: np.ndarray,
    process_noise_var_m2: float,
    measurement_noise_var_m2: float,
) -> dict[str, float]:
    """Summarize constant-position Kalman behavior for feature engineering.

    Inputs:
        x_m: Measured x coordinates in meters.
        y_m: Measured y coordinates in meters.
        process_noise_var_m2: Scalar process-noise variance in square meters.
        measurement_noise_var_m2: Scalar measurement-noise variance in square meters.

    Returns:
        A dictionary of residual and uncertainty feature values.
    """
    result = run_constant_position_kalman(x_m, y_m, process_noise_var_m2, measurement_noise_var_m2)
    if len(result["innovation_m"]) == 0:
        return {
            "cp_kf_mean_innovation_m": 0.0,
            "cp_kf_max_innovation_m": 0.0,
            "cp_kf_rmse_m": 0.0,
            "cp_kf_final_sigma_m": 0.0,
            "cp_kf_mean_nis": 0.0,
        }
    return {
        "cp_kf_mean_innovation_m": float(result["innovation_m"].mean()),
        "cp_kf_max_innovation_m": float(result["innovation_m"].max()),
        "cp_kf_rmse_m": float(np.sqrt(np.mean(np.square(result["residual_m"])))),
        "cp_kf_final_sigma_m": float(result["sigma_m"][-1]),
        "cp_kf_mean_nis": float(result["nis"].mean()),
    }


def simulate_single_satellite_tracking(
    n_steps: int,
    dt_s: float,
    sat_lat_deg: float,
    sat_lon_deg: float,
    sat_alt_km: float,
    target_lat_deg: float,
    target_lon_deg: float,
    east_velocity_mps: float,
    north_velocity_mps: float,
    wander_sigma_m: float,
    az_noise_deg: float,
    el_noise_deg: float,
    process_noise_var_m2: float,
    measurement_noise_var_m2: float,
    seed: int,
    kalman_model: KalmanMotionModel = "constant_position",
    acceleration_schedule: list[dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Simulate a single-satellite tracking scenario with azimuth/elevation measurements.

    Inputs:
        n_steps: Number of simulated time steps.
        dt_s: Time step size in seconds.
        sat_lat_deg: Satellite latitude in degrees.
        sat_lon_deg: Satellite longitude in degrees.
        sat_alt_km: Satellite altitude in kilometers.
        target_lat_deg: Initial target latitude in degrees.
        target_lon_deg: Initial target longitude in degrees.
        east_velocity_mps: Constant eastward target velocity in meters per second.
        north_velocity_mps: Constant northward target velocity in meters per second.
        wander_sigma_m: Per-step Gaussian maneuver noise in meters.
        az_noise_deg: One-sigma cross-boresight angular noise for the first tangent-plane axis, in degrees.
        el_noise_deg: One-sigma cross-boresight angular noise for the second tangent-plane axis, in degrees.
        process_noise_var_m2: Kalman process-noise variance in square meters.
        measurement_noise_var_m2: Kalman measurement-noise variance in square meters.
        seed: Random seed for deterministic simulation.
        kalman_model: Kinematic Kalman model used to filter reconstructed measurements.
        acceleration_schedule: Optional piecewise-constant acceleration schedule for the truth track.

    Returns:
        A pandas DataFrame containing true, measured, and filtered track quantities.
    """
    rng = np.random.default_rng(seed)
    sat_alt_m = sat_alt_km * 1000.0

    true_x = np.zeros(n_steps, dtype=float)
    true_y = np.zeros(n_steps, dtype=float)
    true_vx = np.zeros(n_steps, dtype=float)
    true_vy = np.zeros(n_steps, dtype=float)
    true_ax = np.zeros(n_steps, dtype=float)
    true_ay = np.zeros(n_steps, dtype=float)
    true_vx[0] = east_velocity_mps
    true_vy[0] = north_velocity_mps
    for idx in range(1, n_steps):
        time_prev_s = float((idx - 1) * dt_s)
        east_accel_mps2, north_accel_mps2 = acceleration_from_schedule(time_prev_s, acceleration_schedule)
        true_ax[idx - 1] = east_accel_mps2
        true_ay[idx - 1] = north_accel_mps2
        true_x[idx] = (
            true_x[idx - 1]
            + true_vx[idx - 1] * dt_s
            + 0.5 * east_accel_mps2 * dt_s**2
            + rng.normal(0.0, wander_sigma_m)
        )
        true_y[idx] = (
            true_y[idx - 1]
            + true_vy[idx - 1] * dt_s
            + 0.5 * north_accel_mps2 * dt_s**2
            + rng.normal(0.0, wander_sigma_m)
        )
        true_vx[idx] = true_vx[idx - 1] + east_accel_mps2 * dt_s
        true_vy[idx] = true_vy[idx - 1] + north_accel_mps2 * dt_s
    if n_steps > 0:
        final_time_s = float((n_steps - 1) * dt_s)
        true_ax[-1], true_ay[-1] = acceleration_from_schedule(final_time_s, acceleration_schedule)

    true_lat, true_lon = local_m_to_latlon(true_x, true_y, target_lat_deg, target_lon_deg)

    true_az: list[float] = []
    true_el: list[float] = []
    meas_lat: list[float] = []
    meas_lon: list[float] = []
    meas_az: list[float] = []
    meas_el: list[float] = []
    visible: list[bool] = []

    for lat_deg, lon_deg in zip(true_lat, true_lon):
        is_visible = is_ground_point_visible_from_satellite(
            sat_lat_deg=sat_lat_deg,
            sat_lon_deg=sat_lon_deg,
            sat_alt_m=sat_alt_m,
            target_lat_deg=float(lat_deg),
            target_lon_deg=float(lon_deg),
        )
        los_ecef = target_line_of_sight_ecef(
            sat_lat_deg=sat_lat_deg,
            sat_lon_deg=sat_lon_deg,
            sat_alt_m=sat_alt_m,
            target_lat_deg=float(lat_deg),
            target_lon_deg=float(lon_deg),
        )
        az_deg, el_deg = azel_from_los_ecef(sat_lat_deg, sat_lon_deg, los_ecef)
        noisy_los_ecef = perturb_line_of_sight_cross_boresight(
            los_ecef=los_ecef,
            sigma_axis_1_deg=az_noise_deg,
            sigma_axis_2_deg=el_noise_deg,
            rng=rng,
        )
        noisy_az, noisy_el = azel_from_los_ecef(sat_lat_deg, sat_lon_deg, noisy_los_ecef)
        measured = ground_point_from_los_ecef(
            sat_lat_deg=sat_lat_deg,
            sat_lon_deg=sat_lon_deg,
            sat_alt_m=sat_alt_m,
            los_ecef=noisy_los_ecef,
        )
        if measured is None:
            measured = (float(lat_deg), float(lon_deg))
            noisy_los_ecef = los_ecef
            noisy_az, noisy_el = az_deg, el_deg

        true_az.append(az_deg)
        true_el.append(el_deg)
        meas_az.append(noisy_az)
        meas_el.append(noisy_el)
        meas_lat.append(measured[0])
        meas_lon.append(measured[1])
        visible.append(is_visible)

    meas_x, meas_y = project_to_local_m(np.asarray(meas_lat), np.asarray(meas_lon), target_lat_deg, target_lon_deg)
    filter_result = run_kinematic_kalman(
        meas_x,
        meas_y,
        dt_s=dt_s,
        process_noise_var_m2=process_noise_var_m2,
        measurement_noise_var_m2=measurement_noise_var_m2,
        model=kalman_model,
    )
    filt_lat, filt_lon = local_m_to_latlon(
        filter_result["filtered_x_m"],
        filter_result["filtered_y_m"],
        target_lat_deg,
        target_lon_deg,
    )

    true_err = np.sqrt((meas_x - true_x) ** 2 + (meas_y - true_y) ** 2)
    filt_err = np.sqrt((filter_result["filtered_x_m"] - true_x) ** 2 + (filter_result["filtered_y_m"] - true_y) ** 2)

    df = pd.DataFrame(
        {
            "step": np.arange(n_steps, dtype=int),
            "time_s": np.arange(n_steps, dtype=float) * dt_s,
            "true_lat": true_lat,
            "true_lon": true_lon,
            "true_vx_mps": true_vx,
            "true_vy_mps": true_vy,
            "true_ax_mps2": true_ax,
            "true_ay_mps2": true_ay,
            "measurement_lat": np.asarray(meas_lat, dtype=float),
            "measurement_lon": np.asarray(meas_lon, dtype=float),
            "filtered_lat": filt_lat,
            "filtered_lon": filt_lon,
            "true_az_deg": np.asarray(true_az, dtype=float),
            "true_el_deg": np.asarray(true_el, dtype=float),
            "true_visible": np.asarray(visible, dtype=bool),
            "measurement_az_deg": np.asarray(meas_az, dtype=float),
            "measurement_el_deg": np.asarray(meas_el, dtype=float),
            "kalman_model": [kalman_model] * n_steps,
            "innovation_m": filter_result["innovation_m"],
            "residual_m": filter_result["residual_m"],
            "sigma_m": filter_result["sigma_m"],
            "nis": filter_result["nis"],
            "measurement_error_m": true_err,
            "filtered_error_m": filt_err,
            "filtered_vx_mps": filter_result["filtered_vx_mps"],
            "filtered_vy_mps": filter_result["filtered_vy_mps"],
            "filtered_ax_mps2": filter_result["filtered_ax_mps2"],
            "filtered_ay_mps2": filter_result["filtered_ay_mps2"],
        }
    )
    map_x_true, map_y_true = lonlat_to_web_mercator(df["true_lon"].to_numpy(), df["true_lat"].to_numpy())
    map_x_meas, map_y_meas = lonlat_to_web_mercator(df["measurement_lon"].to_numpy(), df["measurement_lat"].to_numpy())
    map_x_filt, map_y_filt = lonlat_to_web_mercator(df["filtered_lon"].to_numpy(), df["filtered_lat"].to_numpy())
    df["true_x_web"] = map_x_true
    df["true_y_web"] = map_y_true
    df["measurement_x_web"] = map_x_meas
    df["measurement_y_web"] = map_y_meas
    df["filtered_x_web"] = map_x_filt
    df["filtered_y_web"] = map_y_filt
    return df
