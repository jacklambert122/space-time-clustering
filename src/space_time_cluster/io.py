from __future__ import annotations

"""Load input detections and prepare filesystem locations for pipeline outputs."""

from pathlib import Path

import polars as pl

from .config import ClusterConfig


def load_time_chunk(cfg: ClusterConfig) -> pl.DataFrame:
    """Load and type-cast the configured time slice from the detections parquet.

    Inputs:
        cfg: Pipeline configuration describing source columns and time bounds.

    Returns:
        A Polars DataFrame containing the selected rows and normalized column types.
    """
    lf = (
        pl.scan_parquet(cfg.parquet_path)
        .select([cfg.id_col, cfg.time_col, cfg.lat_col, cfg.lon_col])
        .with_columns([
            pl.col(cfg.id_col).cast(pl.Int64),
            pl.col(cfg.time_col).cast(pl.Float64),
            pl.col(cfg.lat_col).cast(pl.Float64),
            pl.col(cfg.lon_col).cast(pl.Float64),
        ])
        .filter(
            (pl.col(cfg.time_col) >= float(cfg.start_time))
            & (pl.col(cfg.time_col) < float(cfg.end_time))
        )
    )
    return lf.collect()


def ensure_out_dir(out_dir: str) -> Path:
    """Create the output directory if needed and return it as a ``Path``.

    Inputs:
        out_dir: Path to the directory where outputs should be written.

    Returns:
        A ``Path`` object pointing to the ensured output directory.
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
