"""Test parquet loading and output-directory helpers used by the pipeline."""

from pathlib import Path

import polars as pl

from space_time_cluster.config import ClusterConfig
from space_time_cluster.io import ensure_out_dir, load_time_chunk


def test_ensure_out_dir_creates_missing_directory(tmp_path: Path) -> None:
    """Verify the output directory helper creates nested directories.

    Inputs:
        tmp_path: Pytest-provided temporary base directory.

    Returns:
        None. The test asserts the expected directory is created.
    """
    out_dir = tmp_path / "nested" / "out"

    path = ensure_out_dir(str(out_dir))

    assert path == out_dir
    assert path.exists()
    assert path.is_dir()


def test_load_time_chunk_filters_rows_and_casts_columns(tmp_path: Path) -> None:
    """Verify parquet loading filters rows and normalizes output dtypes.

    Inputs:
        tmp_path: Pytest-provided temporary base directory.

    Returns:
        None. The test asserts row filtering and output schema normalization.
    """
    parquet_path = tmp_path / "detections.parquet"
    pl.DataFrame(
        {
            "id": [1, 2, 3],
            "time": [50, 100, 150],
            "lat": [40, 41, 42],
            "lon": [-105, -106, -107],
        }
    ).write_parquet(parquet_path)

    cfg = ClusterConfig(
        parquet_path=str(parquet_path),
        start_time=75.0,
        end_time=125.0,
    )

    out = load_time_chunk(cfg)

    assert out.shape == (1, 4)
    assert out["id"].to_list() == [2]
    assert out.schema["id"] == pl.Int64
    assert out.schema["time"] == pl.Float64
    assert out.schema["lat"] == pl.Float64
    assert out.schema["lon"] == pl.Float64
