"""Smoke-test hvPlot cluster visualization output generation."""

from pathlib import Path

import polars as pl

from conftest import load_script_module

visualize_detections = load_script_module("visualize_detections.py")


def test_plot_detections_writes_output_html(tmp_path: Path) -> None:
    """Verify the visualization script can render a basic cluster plot.

    Inputs:
        tmp_path: Pytest-provided temporary base directory.

    Returns:
        None. The test asserts that a non-empty output HTML file is written.
    """
    assignments_path = tmp_path / "point_assignments.parquet"
    summary_path = tmp_path / "cluster_summary.parquet"
    output_path = tmp_path / "cluster_plot.html"

    pl.DataFrame(
        {
            "id": [1, 2, 3],
            "time": [1000.0, 1010.0, 1020.0],
            "lat": [40.0, 40.0005, 39.9998],
            "lon": [-105.0, -105.0004, -104.9997],
            "cluster_id": [0, 0, 0],
            "neighbor_count": [2, 2, 2],
            "kept_after_neighbor_filter": [True, True, True],
            "h3": ["cell_a", "cell_a", "cell_a"],
        }
    ).write_parquet(assignments_path)

    pl.DataFrame(
        {
            "cluster_id": [0],
            "n_points": [3],
            "start_time": [1000.0],
            "end_time": [1020.0],
            "center_lat": [40.0],
            "center_lon": [-105.0],
            "radius_m": [300.0],
        }
    ).write_parquet(summary_path)

    visualize_detections.plot_detections(
        assignments_path=str(assignments_path),
        summary_path=str(summary_path),
        output_path=str(output_path),
        land_path=None,
        city_path=None,
        lake_path=None,
        nearby_radius_km=5.0,
        max_clusters=None,
        include_noise=True,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
