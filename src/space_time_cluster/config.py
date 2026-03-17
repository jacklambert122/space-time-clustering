from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class ClusterConfig:
    parquet_path: str
    time_col: str = "time"
    id_col: str = "id"
    lat_col: str = "lat"
    lon_col: str = "lon"

    # unix seconds stored as float64 in parquet
    start_time: float = 0.0
    end_time: float = 0.0

    # coarse partitioning
    h3_res: int = 7
    time_bin_seconds: float = 60.0

    # exact edge thresholds
    max_time_delta_s: float = 30.0
    max_distance_m: float = 1500.0

    # optional per-point support filter
    use_neighbor_count_filter: bool = False
    min_neighbors: int = 2

    # final cluster filter
    min_cluster_size: int = 3

    # uncertainty summary
    guard_band_m: float = 100.0
    radius_quantile: float = 0.95

    # outputs
    out_dir: str = "out"

    # optional spatial reference layers
    city_vector_path: str | None = None
    lake_vector_path: str | None = None


def load_config(path: str | Path) -> ClusterConfig:
    with open(path, "r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)
    return ClusterConfig(**payload)
