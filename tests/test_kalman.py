"""Test reusable Kalman and single-satellite observation helpers."""

import numpy as np

from space_time_cluster.kalman import (
    azel_from_satellite,
    cross_boresight_basis,
    constant_position_kalman_feature_dict,
    earth_limb_radius_in_view,
    ground_point_from_azel,
    is_ground_point_visible_from_satellite,
    perturb_line_of_sight_cross_boresight,
    project_los_to_satellite_view,
    run_kinematic_kalman,
    run_constant_position_kalman,
    simulate_single_satellite_tracking,
    target_line_of_sight_ecef,
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


def test_run_kinematic_kalman_supports_velocity_and_acceleration_models() -> None:
    """Verify the generic Kalman runner supports CV and CA motion models.

    Inputs:
        None.

    Returns:
        None. The test asserts model-specific state columns are populated.
    """
    x_m = np.array([0.0, 5.0, 10.0], dtype=float)
    y_m = np.array([0.0, 0.0, 0.0], dtype=float)

    cv_result = run_kinematic_kalman(
        x_m=x_m,
        y_m=y_m,
        dt_s=1.0,
        process_noise_var_m2=1.0,
        measurement_noise_var_m2=10.0,
        model="constant_velocity",
    )
    ca_result = run_kinematic_kalman(
        x_m=x_m,
        y_m=y_m,
        dt_s=1.0,
        process_noise_var_m2=1.0,
        measurement_noise_var_m2=10.0,
        model="constant_acceleration",
    )

    assert len(cv_result["filtered_x_m"]) == 3
    assert np.isfinite(cv_result["filtered_vx_mps"]).all()
    assert np.isnan(cv_result["filtered_ax_mps2"]).all()
    assert len(ca_result["filtered_y_m"]) == 3
    assert np.isfinite(ca_result["filtered_vx_mps"]).all()
    assert np.isfinite(ca_result["filtered_ax_mps2"]).all()


def test_visibility_check_distinguishes_visible_and_obstructed_targets() -> None:
    """Verify line-of-sight visibility distinguishes near-side and far-side targets.

    Inputs:
        None.

    Returns:
        None. The test asserts only the near-side target is visible from the satellite.
    """
    assert is_ground_point_visible_from_satellite(
        sat_lat_deg=35.0,
        sat_lon_deg=-105.0,
        sat_alt_m=700000.0,
        target_lat_deg=39.7,
        target_lon_deg=-105.0,
    )
    assert not is_ground_point_visible_from_satellite(
        sat_lat_deg=0.0,
        sat_lon_deg=-105.0,
        sat_alt_m=700000.0,
        target_lat_deg=39.7,
        target_lon_deg=-105.0,
    )


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


def test_cross_boresight_basis_is_orthonormal() -> None:
    """Verify the cross-boresight basis is orthonormal to the boresight.

    Inputs:
        None.

    Returns:
        None. The test asserts the tangent-plane axes form an orthonormal basis.
    """
    los_ecef = target_line_of_sight_ecef(
        sat_lat_deg=0.0,
        sat_lon_deg=0.0,
        sat_alt_m=700000.0,
        target_lat_deg=0.0,
        target_lon_deg=0.0,
    )
    axis_1, axis_2 = cross_boresight_basis(los_ecef)

    assert np.isclose(np.linalg.norm(axis_1), 1.0)
    assert np.isclose(np.linalg.norm(axis_2), 1.0)
    assert np.isclose(np.dot(axis_1, los_ecef), 0.0)
    assert np.isclose(np.dot(axis_2, los_ecef), 0.0)
    assert np.isclose(np.dot(axis_1, axis_2), 0.0)


def test_cross_boresight_noise_perturbs_nadir_case_in_two_dimensions() -> None:
    """Verify cross-boresight noise remains well-defined at nadir.

    Inputs:
        None.

    Returns:
        None. The test asserts the perturbed direction stays finite at nadir geometry.
    """
    rng = np.random.default_rng(42)
    los_ecef = target_line_of_sight_ecef(
        sat_lat_deg=0.0,
        sat_lon_deg=0.0,
        sat_alt_m=700000.0,
        target_lat_deg=0.0,
        target_lon_deg=0.0,
    )
    perturbed = perturb_line_of_sight_cross_boresight(
        los_ecef=los_ecef,
        sigma_axis_1_deg=0.15,
        sigma_axis_2_deg=0.15,
        rng=rng,
    )

    assert np.isfinite(perturbed).all()
    assert np.isclose(np.linalg.norm(perturbed), 1.0)
    assert not np.allclose(perturbed, los_ecef)


def test_satellite_view_projection_places_nadir_at_plot_center() -> None:
    """Verify the nadir look direction projects to the center of the Earth-view plot.

    Inputs:
        None.

    Returns:
        None. The test asserts the nadir line of sight maps to the view origin.
    """
    los_ecef = target_line_of_sight_ecef(
        sat_lat_deg=0.0,
        sat_lon_deg=0.0,
        sat_alt_m=700000.0,
        target_lat_deg=0.0,
        target_lon_deg=0.0,
    )
    view_x, view_y, view_z = project_los_to_satellite_view(
        los_ecef=los_ecef,
        sat_lat_deg=0.0,
        sat_lon_deg=0.0,
        sat_alt_m=700000.0,
    )

    assert abs(view_x) < 1e-9
    assert abs(view_y) < 1e-9
    assert view_z > 0.99


def test_earth_limb_radius_in_view_is_between_zero_and_one() -> None:
    """Verify the projected Earth-disk radius is bounded for a satellite above Earth.

    Inputs:
        None.

    Returns:
        None. The test asserts the visible Earth disk radius is physically bounded.
    """
    limb_radius = earth_limb_radius_in_view(700000.0)

    assert 0.0 < limb_radius < 1.0


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
        "true_visible",
        "kalman_model",
        "innovation_m",
        "sigma_m",
        "measurement_error_m",
        "filtered_error_m",
    ]:
        assert column in df.columns


def test_simulate_single_satellite_tracking_supports_constant_velocity_model() -> None:
    """Verify the simulator can use the constant-velocity Kalman model.

    Inputs:
        None.

    Returns:
        None. The test asserts velocity state outputs are populated for the CV model.
    """
    df = simulate_single_satellite_tracking(
        n_steps=10,
        dt_s=5.0,
        sat_lat_deg=35.0,
        sat_lon_deg=-105.0,
        sat_alt_km=700.0,
        target_lat_deg=39.7,
        target_lon_deg=-105.0,
        east_velocity_mps=5.0,
        north_velocity_mps=0.0,
        wander_sigma_m=1.0,
        az_noise_deg=0.1,
        el_noise_deg=0.1,
        process_noise_var_m2=25.0,
        measurement_noise_var_m2=10000.0,
        seed=42,
        kalman_model="constant_velocity",
    )

    assert (df["kalman_model"] == "constant_velocity").all()
    assert df["filtered_vx_mps"].notna().all()
    assert df["filtered_vy_mps"].notna().all()


def test_simulate_single_satellite_tracking_applies_scheduled_acceleration_changes() -> None:
    """Verify the truth track responds to scheduled acceleration events.

    Inputs:
        None.

    Returns:
        None. The test asserts scheduled accelerations change the truth velocity history.
    """
    df = simulate_single_satellite_tracking(
        n_steps=8,
        dt_s=1.0,
        sat_lat_deg=35.0,
        sat_lon_deg=-105.0,
        sat_alt_km=700.0,
        target_lat_deg=39.7,
        target_lon_deg=-105.0,
        east_velocity_mps=0.0,
        north_velocity_mps=0.0,
        wander_sigma_m=0.0,
        az_noise_deg=0.01,
        el_noise_deg=0.01,
        process_noise_var_m2=25.0,
        measurement_noise_var_m2=10000.0,
        seed=42,
        kalman_model="constant_acceleration",
        acceleration_schedule=[
            {
                "start_time_s": 2.0,
                "end_time_s": 5.0,
                "east_accel_mps2": 1.0,
                "north_accel_mps2": -0.5,
            }
        ],
    )

    assert "true_vx_mps" in df.columns
    assert "true_ax_mps2" in df.columns
    assert df.loc[df["time_s"] < 2.0, "true_ax_mps2"].eq(0.0).all()
    assert df.loc[(df["time_s"] >= 2.0) & (df["time_s"] < 5.0), "true_ax_mps2"].eq(1.0).all()
    assert df.loc[(df["time_s"] >= 2.0) & (df["time_s"] < 5.0), "true_ay_mps2"].eq(-0.5).all()
    assert df["true_vx_mps"].iloc[-1] > 0.0
