from __future__ import annotations

"""Interactive hvPlot/Panel playground for a single-satellite constant-position Kalman filter."""

import argparse
import math

from holoviews.element import tiles
import hvplot.pandas  # noqa: F401
import numpy as np
import pandas as pd
import panel as pn

from space_time_cluster.kalman import (
    earth_limb_radius_in_view,
    project_los_to_satellite_view,
    simulate_single_satellite_tracking,
    target_line_of_sight_ecef,
)

pn.extension("tabulator")


def build_dashboard(
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
    kalman_model: str,
    accel_event_1_start_frac: float,
    accel_event_1_end_frac: float,
    accel_event_1_east_mps2: float,
    accel_event_1_north_mps2: float,
    accel_event_2_start_frac: float,
    accel_event_2_end_frac: float,
    accel_event_2_east_mps2: float,
    accel_event_2_north_mps2: float,
) -> pn.Column:
    """Build the interactive dashboard body for the Kalman playground.

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
        az_noise_deg: One-sigma cross-boresight angular noise on the first tangent-plane axis, in degrees.
        el_noise_deg: One-sigma cross-boresight angular noise on the second tangent-plane axis, in degrees.
        process_noise_var_m2: Kalman process-noise variance in square meters.
        measurement_noise_var_m2: Kalman measurement-noise variance in square meters.
        seed: Random seed for the simulation.
        kalman_model: Selected kinematic Kalman motion model.
        accel_event_1_start_frac: Start fraction for acceleration event 1.
        accel_event_1_end_frac: End fraction for acceleration event 1.
        accel_event_1_east_mps2: East acceleration for event 1.
        accel_event_1_north_mps2: North acceleration for event 1.
        accel_event_2_start_frac: Start fraction for acceleration event 2.
        accel_event_2_end_frac: End fraction for acceleration event 2.
        accel_event_2_east_mps2: East acceleration for event 2.
        accel_event_2_north_mps2: North acceleration for event 2.

    Returns:
        A Panel column containing map, diagnostics, and summary views.
    """
    total_time_s = max((n_steps - 1) * dt_s, 0.0)
    acceleration_schedule = [
        {
            "start_time_s": min(accel_event_1_start_frac, accel_event_1_end_frac) * total_time_s,
            "end_time_s": max(accel_event_1_start_frac, accel_event_1_end_frac) * total_time_s,
            "east_accel_mps2": accel_event_1_east_mps2,
            "north_accel_mps2": accel_event_1_north_mps2,
        },
        {
            "start_time_s": min(accel_event_2_start_frac, accel_event_2_end_frac) * total_time_s,
            "end_time_s": max(accel_event_2_start_frac, accel_event_2_end_frac) * total_time_s,
            "east_accel_mps2": accel_event_2_east_mps2,
            "north_accel_mps2": accel_event_2_north_mps2,
        },
    ]
    df = simulate_single_satellite_tracking(
        n_steps=n_steps,
        dt_s=dt_s,
        sat_lat_deg=sat_lat_deg,
        sat_lon_deg=sat_lon_deg,
        sat_alt_km=sat_alt_km,
        target_lat_deg=target_lat_deg,
        target_lon_deg=target_lon_deg,
        east_velocity_mps=east_velocity_mps,
        north_velocity_mps=north_velocity_mps,
        wander_sigma_m=wander_sigma_m,
        az_noise_deg=az_noise_deg,
        el_noise_deg=el_noise_deg,
        process_noise_var_m2=process_noise_var_m2,
        measurement_noise_var_m2=measurement_noise_var_m2,
        seed=seed,
        kalman_model=kalman_model,
        acceleration_schedule=acceleration_schedule,
    )

    sat_alt_m = sat_alt_km * 1000.0

    map_overlay = (
        tiles.OSM().opts(alpha=0.8)
        * df.hvplot.paths(x="true_x_web", y="true_y_web", color="#1b9e77", line_width=3, label="True track")
        * df.hvplot.scatter(x="measurement_x_web", y="measurement_y_web", color="#d95f02", size=35, alpha=0.65, label="Measured positions")
        * df.hvplot.paths(x="filtered_x_web", y="filtered_y_web", color="#7570b3", line_width=3, label="Filtered track")
    ).opts(
        width=900,
        height=520,
        title="Single-satellite tracking map",
        active_tools=["pan", "wheel_zoom"],
        legend_position="right",
    )

    diagnostics = df[["time_s", "innovation_m", "sigma_m", "measurement_error_m", "filtered_error_m"]].hvplot.line(
        x="time_s",
        y=["innovation_m", "sigma_m", "measurement_error_m", "filtered_error_m"],
        width=900,
        height=300,
        title="Innovation, uncertainty, and error metrics",
        xlabel="Time (s)",
        ylabel="Meters",
        shared_axes=False,
    )

    kinematic_cols = ["time_s", "true_vx_mps", "true_vy_mps"]
    if df["filtered_vx_mps"].notna().any():
        kinematic_cols.extend(["filtered_vx_mps", "filtered_vy_mps"])
    if df["true_ax_mps2"].notna().any():
        kinematic_cols.extend(["true_ax_mps2", "true_ay_mps2"])
    if df["filtered_ax_mps2"].notna().any():
        kinematic_cols.extend(["filtered_ax_mps2", "filtered_ay_mps2"])
    kinematics = None
    if len(kinematic_cols) > 1:
        kinematics = df[kinematic_cols].hvplot.line(
            x="time_s",
            y=kinematic_cols[1:],
            width=900,
            height=260,
            title="True and filtered kinematic state",
            xlabel="Time (s)",
            ylabel="State value",
            shared_axes=False,
        )

    look_angles = df[["time_s", "measurement_az_deg", "measurement_el_deg"]].hvplot.line(
        x="time_s",
        y=["measurement_az_deg", "measurement_el_deg"],
        width=900,
        height=260,
        title="Measured azimuth/elevation from the satellite",
        xlabel="Time (s)",
        ylabel="Degrees",
        shared_axes=False,
    )

    def view_projection_rows(lat_col: str, lon_col: str, label: str) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for lat_deg, lon_deg in zip(df[lat_col].to_numpy(), df[lon_col].to_numpy()):
            los_ecef = target_line_of_sight_ecef(
                sat_lat_deg=sat_lat_deg,
                sat_lon_deg=sat_lon_deg,
                sat_alt_m=sat_alt_m,
                target_lat_deg=float(lat_deg),
                target_lon_deg=float(lon_deg),
            )
            view_x, view_y, _ = project_los_to_satellite_view(
                los_ecef=los_ecef,
                sat_lat_deg=sat_lat_deg,
                sat_lon_deg=sat_lon_deg,
                sat_alt_m=sat_alt_m,
            )
            rows.append({"view_x": view_x, "view_y": view_y, "series": label})
        return pd.DataFrame(rows)

    true_view = view_projection_rows("true_lat", "true_lon", "True")
    meas_view = view_projection_rows("measurement_lat", "measurement_lon", "Measured")
    filt_view = view_projection_rows("filtered_lat", "filtered_lon", "Filtered")
    limb_radius = earth_limb_radius_in_view(sat_alt_m)
    limb_angles = np.linspace(0.0, 2.0 * math.pi, 361)
    earth_disk = pd.DataFrame(
        {
            "view_x": limb_radius * np.cos(limb_angles),
            "view_y": limb_radius * np.sin(limb_angles),
            "path_id": ["earth_limb"] * len(limb_angles),
        }
    )
    crosshair = pd.DataFrame(
        {
            "view_x": [-limb_radius, limb_radius, 0.0, 0.0],
            "view_y": [0.0, 0.0, -limb_radius, limb_radius],
            "path_id": ["horizon_x", "horizon_x", "horizon_y", "horizon_y"],
        }
    )
    sphere_view = (
        earth_disk.hvplot.paths(x="view_x", y="view_y", by="path_id", color="#4c566a", line_width=2, legend=False)
        * crosshair.hvplot.paths(x="view_x", y="view_y", by="path_id", color="#d8dee9", line_width=1, alpha=0.6, legend=False)
        * true_view.hvplot.paths(x="view_x", y="view_y", color="#1b9e77", line_width=3, label="True LOS")
        * meas_view.hvplot.scatter(x="view_x", y="view_y", color="#d95f02", alpha=0.65, size=30, label="Measured LOS")
        * filt_view.hvplot.paths(x="view_x", y="view_y", color="#7570b3", line_width=3, label="Filtered LOS")
    ).opts(
        frame_width=440,
        frame_height=440,
        title="Satellite Earth-view sphere projection",
        xlabel="Cross-boresight X",
        ylabel="Cross-boresight Y",
        data_aspect=1,
        xlim=(-1.05 * limb_radius, 1.05 * limb_radius),
        ylim=(-1.05 * limb_radius, 1.05 * limb_radius),
        padding=0.0,
        axiswise=True,
        legend_position="right",
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "mean_measurement_error_m",
                "mean_filtered_error_m",
                "mean_innovation_m",
                "final_sigma_m",
                "mean_nis",
                "visible_fraction",
                "kalman_model",
                "event_1_window_s",
                "event_2_window_s",
            ],
            "value": [
                float(df["measurement_error_m"].mean()),
                float(df["filtered_error_m"].mean()),
                float(df["innovation_m"].mean()),
                float(df["sigma_m"].iloc[-1]),
                float(df["nis"].mean()),
                float(df["true_visible"].mean()),
                kalman_model,
                f"{acceleration_schedule[0]['start_time_s']:.1f} to {acceleration_schedule[0]['end_time_s']:.1f}",
                f"{acceleration_schedule[1]['start_time_s']:.1f} to {acceleration_schedule[1]['end_time_s']:.1f}",
            ],
        }
    )
    if (~df["true_visible"]).any():
        visibility_note = (
            "Some target states are Earth-obstructed from the satellite. In those intervals, "
            "the target is not the first Earth intersection along the line of sight, so the "
            "reconstructed ground point is not physically the same target."
        )
    else:
        visibility_note = (
            "All simulated target states remain Earth-visible from the satellite, so the "
            "azimuth/elevation to ground-point reconstruction is physically consistent."
        )

    return pn.Column(
        pn.pane.Markdown(
            "## Constant-position Kalman playground\n"
            "Tune the satellite geometry, cross-boresight angular noise, truth-track acceleration events, "
            "and Kalman parameters to compare motion models."
        ),
        pn.pane.Markdown(f"**Visibility check:** {visibility_note}"),
        pn.Row(map_overlay, sphere_view),
        diagnostics,
        *( [kinematics] if kinematics is not None else [] ),
        look_angles,
        pn.widgets.Tabulator(summary, show_index=False, disabled=True, sizing_mode="stretch_width"),
    )


def create_app() -> pn.Row:
    """Create the interactive Panel application for the Kalman playground.

    Inputs:
        None.

    Returns:
        A Panel layout containing the parameter controls and dashboard views.
    """
    widgets = {
        "n_steps": pn.widgets.IntSlider(name="Steps", start=10, end=300, value=120),
        "dt_s": pn.widgets.FloatSlider(name="dt (s)", start=1.0, end=120.0, step=1.0, value=10.0),
        "sat_lat_deg": pn.widgets.FloatSlider(name="Satellite Lat", start=-85.0, end=85.0, step=0.1, value=35.0),
        "sat_lon_deg": pn.widgets.FloatSlider(name="Satellite Lon", start=-180.0, end=180.0, step=0.1, value=-105.0),
        "sat_alt_km": pn.widgets.FloatSlider(name="Satellite Alt (km)", start=300.0, end=36000.0, step=10.0, value=700.0),
        "target_lat_deg": pn.widgets.FloatSlider(name="Target Lat", start=-80.0, end=80.0, step=0.1, value=39.7),
        "target_lon_deg": pn.widgets.FloatSlider(name="Target Lon", start=-180.0, end=180.0, step=0.1, value=-105.0),
        "east_velocity_mps": pn.widgets.FloatSlider(name="East Velocity (m/s)", start=-100.0, end=100.0, step=1.0, value=0.0),
        "north_velocity_mps": pn.widgets.FloatSlider(name="North Velocity (m/s)", start=-100.0, end=100.0, step=1.0, value=0.0),
        "wander_sigma_m": pn.widgets.FloatSlider(name="Maneuver Sigma (m)", start=0.0, end=500.0, step=5.0, value=15.0),
        "az_noise_deg": pn.widgets.FloatSlider(name="Cross-Boresight Noise 1 (deg)", start=0.0, end=2.0, step=0.001, value=0.15),
        "el_noise_deg": pn.widgets.FloatSlider(name="Cross-Boresight Noise 2 (deg)", start=0.0, end=2.0, step=0.001, value=0.15),
        "process_noise_var_m2": pn.widgets.FloatInput(name="KF Process Var (m^2)", value=25.0, step=5.0, start=0.0),
        "measurement_noise_var_m2": pn.widgets.FloatInput(name="KF Measurement Var (m^2)", value=10000.0, step=1000.0, start=0.0),
        "seed": pn.widgets.IntInput(name="Seed", value=42, step=1),
        "kalman_model": pn.widgets.Select(
            name="Kalman Model",
            options={
                "Constant Position": "constant_position",
                "Constant Velocity": "constant_velocity",
                "Constant Acceleration": "constant_acceleration",
            },
            value="constant_position",
        ),
        "accel_event_1_start_frac": pn.widgets.FloatSlider(name="Accel Event 1 Start", start=0.0, end=1.0, step=0.01, value=0.25),
        "accel_event_1_end_frac": pn.widgets.FloatSlider(name="Accel Event 1 End", start=0.0, end=1.0, step=0.01, value=0.45),
        "accel_event_1_east_mps2": pn.widgets.FloatSlider(name="Accel Event 1 East (m/s^2)", start=-5.0, end=5.0, step=0.1, value=0.0),
        "accel_event_1_north_mps2": pn.widgets.FloatSlider(name="Accel Event 1 North (m/s^2)", start=-5.0, end=5.0, step=0.1, value=0.0),
        "accel_event_2_start_frac": pn.widgets.FloatSlider(name="Accel Event 2 Start", start=0.0, end=1.0, step=0.01, value=0.60),
        "accel_event_2_end_frac": pn.widgets.FloatSlider(name="Accel Event 2 End", start=0.0, end=1.0, step=0.01, value=0.80),
        "accel_event_2_east_mps2": pn.widgets.FloatSlider(name="Accel Event 2 East (m/s^2)", start=-5.0, end=5.0, step=0.1, value=0.0),
        "accel_event_2_north_mps2": pn.widgets.FloatSlider(name="Accel Event 2 North (m/s^2)", start=-5.0, end=5.0, step=0.1, value=0.0),
    }

    dashboard = pn.bind(build_dashboard, **widgets)
    controls = pn.WidgetBox(
        "## Controls",
        *widgets.values(),
        sizing_mode="stretch_height",
        width=320,
    )
    return pn.Row(controls, pn.Column(dashboard, sizing_mode="stretch_both"), sizing_mode="stretch_both")


def main() -> None:
    """Serve the Kalman playground as a local Panel application.

    Inputs:
        None. Arguments are read from the command line.

    Returns:
        None. The function starts a local Panel server.
    """
    parser = argparse.ArgumentParser(description="Run an interactive Kalman filter playground for single-satellite tracking.")
    parser.add_argument("--port", type=int, default=5007, help="Port for the Panel server")
    parser.add_argument("--show", action="store_true", help="Open a browser tab automatically")
    args = parser.parse_args()

    pn.serve(create_app(), port=args.port, show=args.show, title="Kalman Playground")


if __name__ == "__main__":
    main()
