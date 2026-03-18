from __future__ import annotations

"""Kalman-filter and single-satellite observation helpers for tracking analysis."""

import math

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371000.0
WEB_MERCATOR_LIMIT_LAT = 85.05112878
WEB_MERCATOR_ORIGIN_SHIFT = 20037508.34


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
    sat_ecef = geodetic_to_ecef(sat_lat_deg, sat_lon_deg, sat_alt_m)
    los_ecef = line_of_sight_ecef(sat_lat_deg, sat_lon_deg, az_deg, el_deg)
    target_radius = EARTH_RADIUS_M + target_alt_m

    a = float(np.dot(los_ecef, los_ecef))
    b = float(2.0 * np.dot(sat_ecef, los_ecef))
    c = float(np.dot(sat_ecef, sat_ecef) - target_radius**2)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None

    sqrt_disc = math.sqrt(disc)
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    positive_roots = [root for root in roots if root > 0.0]
    if not positive_roots:
        return None

    target_ecef = sat_ecef + min(positive_roots) * los_ecef
    lat_deg, lon_deg, _ = ecef_to_geodetic(target_ecef)
    return lat_deg, lon_deg


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
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    if len(x_m) == 0:
        empty = np.empty(0, dtype=float)
        return {
            "filtered_x_m": empty,
            "filtered_y_m": empty,
            "innovation_m": empty,
            "residual_m": empty,
            "nis": empty,
            "sigma_m": empty,
        }

    q_mat = np.eye(2, dtype=float) * float(process_noise_var_m2)
    r_mat = np.eye(2, dtype=float) * float(measurement_noise_var_m2)
    state = np.array([x_m[0], y_m[0]], dtype=float)
    cov = r_mat.copy()

    filtered_x: list[float] = []
    filtered_y: list[float] = []
    innovations: list[float] = []
    residuals: list[float] = []
    nis_values: list[float] = []
    sigma_values: list[float] = []

    for x_obs, y_obs in zip(x_m, y_m):
        state_pred = state
        cov_pred = cov + q_mat
        z = np.array([x_obs, y_obs], dtype=float)
        innovation = z - state_pred
        s_mat = cov_pred + r_mat
        k_gain = cov_pred @ np.linalg.inv(s_mat)
        state = state_pred + k_gain @ innovation
        cov = (np.eye(2, dtype=float) - k_gain) @ cov_pred

        filtered_x.append(float(state[0]))
        filtered_y.append(float(state[1]))
        innovations.append(float(np.linalg.norm(innovation)))
        residuals.append(float(np.linalg.norm(z - state)))
        nis_values.append(float(innovation.T @ np.linalg.inv(s_mat) @ innovation))
        sigma_values.append(float(np.sqrt(np.trace(cov) / 2.0)))

    return {
        "filtered_x_m": np.asarray(filtered_x, dtype=float),
        "filtered_y_m": np.asarray(filtered_y, dtype=float),
        "innovation_m": np.asarray(innovations, dtype=float),
        "residual_m": np.asarray(residuals, dtype=float),
        "nis": np.asarray(nis_values, dtype=float),
        "sigma_m": np.asarray(sigma_values, dtype=float),
    }


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
        az_noise_deg: Azimuth measurement noise standard deviation in degrees.
        el_noise_deg: Elevation measurement noise standard deviation in degrees.
        process_noise_var_m2: Kalman process-noise variance in square meters.
        measurement_noise_var_m2: Kalman measurement-noise variance in square meters.
        seed: Random seed for deterministic simulation.

    Returns:
        A pandas DataFrame containing true, measured, and filtered track quantities.
    """
    rng = np.random.default_rng(seed)
    sat_alt_m = sat_alt_km * 1000.0

    true_x = np.zeros(n_steps, dtype=float)
    true_y = np.zeros(n_steps, dtype=float)
    for idx in range(1, n_steps):
        true_x[idx] = true_x[idx - 1] + east_velocity_mps * dt_s + rng.normal(0.0, wander_sigma_m)
        true_y[idx] = true_y[idx - 1] + north_velocity_mps * dt_s + rng.normal(0.0, wander_sigma_m)

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
        az_deg, el_deg = azel_from_satellite(
            sat_lat_deg=sat_lat_deg,
            sat_lon_deg=sat_lon_deg,
            sat_alt_m=sat_alt_m,
            target_lat_deg=float(lat_deg),
            target_lon_deg=float(lon_deg),
        )
        noisy_az = az_deg + rng.normal(0.0, az_noise_deg)
        noisy_el = el_deg + rng.normal(0.0, el_noise_deg)
        measured = ground_point_from_azel(
            sat_lat_deg=sat_lat_deg,
            sat_lon_deg=sat_lon_deg,
            sat_alt_m=sat_alt_m,
            az_deg=noisy_az,
            el_deg=noisy_el,
        )
        if measured is None:
            measured = (float(lat_deg), float(lon_deg))
            noisy_az, noisy_el = az_deg, el_deg

        true_az.append(az_deg)
        true_el.append(el_deg)
        meas_az.append(noisy_az)
        meas_el.append(noisy_el)
        meas_lat.append(measured[0])
        meas_lon.append(measured[1])
        visible.append(is_visible)

    meas_x, meas_y = project_to_local_m(np.asarray(meas_lat), np.asarray(meas_lon), target_lat_deg, target_lon_deg)
    filter_result = run_constant_position_kalman(
        meas_x,
        meas_y,
        process_noise_var_m2=process_noise_var_m2,
        measurement_noise_var_m2=measurement_noise_var_m2,
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
            "measurement_lat": np.asarray(meas_lat, dtype=float),
            "measurement_lon": np.asarray(meas_lon, dtype=float),
            "filtered_lat": filt_lat,
            "filtered_lon": filt_lon,
            "true_az_deg": np.asarray(true_az, dtype=float),
            "true_el_deg": np.asarray(true_el, dtype=float),
            "true_visible": np.asarray(visible, dtype=bool),
            "measurement_az_deg": np.asarray(meas_az, dtype=float),
            "measurement_el_deg": np.asarray(meas_el, dtype=float),
            "innovation_m": filter_result["innovation_m"],
            "residual_m": filter_result["residual_m"],
            "sigma_m": filter_result["sigma_m"],
            "nis": filter_result["nis"],
            "measurement_error_m": true_err,
            "filtered_error_m": filt_err,
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
