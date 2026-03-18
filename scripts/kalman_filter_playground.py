from __future__ import annotations

"""Interactive hvPlot/Panel playground for a single-satellite constant-position Kalman filter."""

import argparse

from holoviews.element import tiles
import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn

from space_time_cluster.kalman import simulate_single_satellite_tracking

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
        az_noise_deg: Azimuth measurement noise standard deviation in degrees.
        el_noise_deg: Elevation measurement noise standard deviation in degrees.
        process_noise_var_m2: Kalman process-noise variance in square meters.
        measurement_noise_var_m2: Kalman measurement-noise variance in square meters.
        seed: Random seed for the simulation.

    Returns:
        A Panel column containing map, diagnostics, and summary views.
    """
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
    )

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
    )

    look_angles = df[["time_s", "measurement_az_deg", "measurement_el_deg"]].hvplot.line(
        x="time_s",
        y=["measurement_az_deg", "measurement_el_deg"],
        width=900,
        height=260,
        title="Measured azimuth/elevation from the satellite",
        xlabel="Time (s)",
        ylabel="Degrees",
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
            ],
            "value": [
                float(df["measurement_error_m"].mean()),
                float(df["filtered_error_m"].mean()),
                float(df["innovation_m"].mean()),
                float(df["sigma_m"].iloc[-1]),
                float(df["nis"].mean()),
                float(df["true_visible"].mean()),
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
            "Tune the satellite geometry, azimuth/elevation noise, and Kalman parameters "
            "to see how the stationary filter behaves."
        ),
        pn.pane.Markdown(f"**Visibility check:** {visibility_note}"),
        map_overlay,
        diagnostics,
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
        "az_noise_deg": pn.widgets.FloatSlider(name="Az Noise (deg)", start=0.0, end=2.0, step=0.001, value=0.15),
        "el_noise_deg": pn.widgets.FloatSlider(name="El Noise (deg)", start=0.0, end=2.0, step=0.001, value=0.15),
        "process_noise_var_m2": pn.widgets.FloatInput(name="KF Process Var (m^2)", value=25.0, step=5.0, start=0.0),
        "measurement_noise_var_m2": pn.widgets.FloatInput(name="KF Measurement Var (m^2)", value=10000.0, step=1000.0, start=0.0),
        "seed": pn.widgets.IntInput(name="Seed", value=42, step=1),
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
