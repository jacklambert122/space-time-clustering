"""Test reusable Kalman and single-satellite observation helpers."""

import numpy as np

from space_time_cluster.kalman import (
    azel_from_satellite,
    constant_position_kalman_feature_dict,
    ground_point_from_azel,
    run_constant_position_kalman,
    simulate_single_satellite_tracking,
)


def test_azimuth_elevation_round_trip_recovers_target() -> None:
    """Verify azimuth/elevation conversion can recover the original ground point.

    Inputs:
        None.

    Returns:
        None. The test asserts the line-of-sight round-trip stays near the target.
    """
    az_deg, el_deg = azel_from_satellite(
        sat_lat_deg=35.0,
        sat_lon_deg=-105.0,
        sat_alt_m=700000.0,
        target_lat_deg=39.7,
        target_lon_deg=-105.0,
    )
    target = ground_point_from_azel(
        sat_lat_deg=35.0,
        sat_lon_deg=-105.0,
        sat_alt_m=700000.0,
        az_deg=az_deg,
        el_deg=el_deg,
    )

    assert target is not None
    assert abs(target[0] - 39.7) < 1e-3
    assert abs(target[1] + 105.0) < 1e-3


def test_run_constant_position_kalman_returns_expected_shapes() -> None:
    """Verify the Kalman runner returns one estimate per measurement.

    Inputs:
        None.

    Returns:
        None. The test asserts output array lengths match the input sequence.
    """
    result = run_constant_position_kalman(
        x_m=np.array([0.0, 5.0, 2.0], dtype=float),
        y_m=np.array([0.0, -1.0, 1.0], dtype=float),
        process_noise_var_m2=25.0,
        measurement_noise_var_m2=100.0,
    )

    assert len(result["filtered_x_m"]) == 3
    assert len(result["innovation_m"]) == 3
    assert result["sigma_m"][-1] > 0.0


def test_constant_position_kalman_features_are_finite() -> None:
    """Verify Kalman feature extraction returns finite summary values.

    Inputs:
        None.

    Returns:
        None. The test asserts feature values are finite and non-negative.
    """
    features = constant_position_kalman_feature_dict(
        x_m=np.array([0.0, 2.0, -1.0, 1.0], dtype=float),
        y_m=np.array([0.0, -1.0, 1.5, 0.5], dtype=float),
        process_noise_var_m2=10.0,
        measurement_noise_var_m2=100.0,
    )

    assert features["cp_kf_mean_innovation_m"] >= 0.0
    assert features["cp_kf_rmse_m"] >= 0.0
    assert features["cp_kf_final_sigma_m"] > 0.0


def test_simulate_single_satellite_tracking_returns_track_columns() -> None:
    """Verify the single-satellite simulator returns true, measured, and filtered tracks.

    Inputs:
        None.

    Returns:
        None. The test asserts the simulator output includes the expected tracking columns.
    """
    df = simulate_single_satellite_tracking(
        n_steps=10,
        dt_s=5.0,
        sat_lat_deg=0.0,
        sat_lon_deg=-100.0,
        sat_alt_km=700.0,
        target_lat_deg=39.7,
        target_lon_deg=-105.0,
        east_velocity_mps=0.0,
        north_velocity_mps=0.0,
        wander_sigma_m=5.0,
        az_noise_deg=0.1,
        el_noise_deg=0.1,
        process_noise_var_m2=25.0,
        measurement_noise_var_m2=10000.0,
        seed=42,
    )

    assert df.shape[0] == 10
    for column in [
        "true_lat",
        "measurement_lat",
        "filtered_lat",
        "innovation_m",
        "sigma_m",
        "measurement_error_m",
        "filtered_error_m",
    ]:
        assert column in df.columns
