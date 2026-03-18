"""Test end-to-end pipeline execution and optional classification output writes."""

from pathlib import Path

import pandas as pd
import polars as pl

from space_time_cluster.config import ClusterConfig
from space_time_cluster.pipeline import run_pipeline


def test_run_pipeline_writes_assignment_and_summary_outputs(tmp_path: Path, capsys) -> None:
    """Verify the pipeline writes its core parquet outputs for a small input.

    Inputs:
        tmp_path: Pytest-provided temporary base directory.
        capsys: Pytest fixture for capturing stdout.

    Returns:
        None. The test asserts the core pipeline outputs are written and populated.
    """
    parquet_path = tmp_path / "detections.parquet"
    out_dir = tmp_path / "out"
    pl.DataFrame(
        {
            "id": [1, 2, 3],
            "time": [1000.0, 1010.0, 2000.0],
            "lat": [40.0, 40.0005, 41.0],
            "lon": [-105.0, -105.0005, -105.0],
        }
    ).write_parquet(parquet_path)

    cfg = ClusterConfig(
        parquet_path=str(parquet_path),
        start_time=900.0,
        end_time=2100.0,
        max_time_delta_s=30.0,
        max_distance_m=200.0,
        min_cluster_size=2,
        out_dir=str(out_dir),
    )

    run_pipeline(cfg)
    captured = capsys.readouterr()

    assignments = pl.read_parquet(out_dir / "point_assignments.parquet")
    summary = pl.read_parquet(out_dir / "cluster_summary.parquet")

    assert "Loaded rows: 3" in captured.out
    assert assignments.shape == (3, 8)
    assert assignments["cluster_id"].to_list() == [0, 0, -1]
    assert assignments["neighbor_count"].to_list() == [1, 1, 0]
    assert summary.shape[0] == 1
    assert summary["n_points"].to_list() == [2]
    assert "cp_kf_rmse_m" in summary.columns
    assert "path_length_m" in summary.columns
    assert summary["duration_s"].to_list() == [10.0]


def test_run_pipeline_writes_optional_classification_outputs(tmp_path: Path, monkeypatch) -> None:
    """Verify optional city and lake classification outputs are written when requested.

    Inputs:
        tmp_path: Pytest-provided temporary base directory.
        monkeypatch: Pytest fixture for replacing the classification call.

    Returns:
        None. The test asserts optional classification parquet outputs are written.
    """
    parquet_path = tmp_path / "detections.parquet"
    out_dir = tmp_path / "out"
    pl.DataFrame(
        {
            "id": [1, 2],
            "time": [1000.0, 1010.0],
            "lat": [40.0, 40.0005],
            "lon": [-105.0, -105.0005],
        }
    ).write_parquet(parquet_path)

    def fake_classification(cluster_summary: pl.DataFrame, city_path: str | None, lake_path: str | None):
        return (
            pd.DataFrame({"cluster_id": [0], "city_name": ["Test City"]}),
            pd.DataFrame({"cluster_id": [0], "lake_name": ["Test Lake"]}),
        )

    monkeypatch.setattr("space_time_cluster.pipeline.classify_clusters_against_polygons", fake_classification)

    cfg = ClusterConfig(
        parquet_path=str(parquet_path),
        start_time=900.0,
        end_time=1100.0,
        max_time_delta_s=30.0,
        max_distance_m=200.0,
        min_cluster_size=2,
        out_dir=str(out_dir),
        city_vector_path="cities.geojson",
        lake_vector_path="lakes.geojson",
    )

    run_pipeline(cfg)

    city_hits = pd.read_parquet(out_dir / "cluster_city_hits.parquet")
    lake_hits = pd.read_parquet(out_dir / "cluster_lake_hits.parquet")

    assert city_hits["city_name"].tolist() == ["Test City"]
    assert lake_hits["lake_name"].tolist() == ["Test Lake"]
