from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import random

import polars as pl


@dataclass
class ClusterSpec:
    center_lat: float
    center_lon: float
    start_time_s: float
    n_points: int
    duration_s: float
    spatial_sigma_m: float
    drift_m_per_s_lat: float = 0.0
    drift_m_per_s_lon: float = 0.0


def meters_to_lat_deg(meters: float) -> float:
    return meters / 111_320.0


def meters_to_lon_deg(meters: float, lat_deg: float) -> float:
    denom = 111_320.0 * max(math.cos(math.radians(lat_deg)), 1e-6)
    return meters / denom


def sample_cluster_points(spec: ClusterSpec, id_start: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(spec.n_points):
        dt_s = random.uniform(0.0, spec.duration_s)
        t = spec.start_time_s + dt_s
        drift_lat_deg = meters_to_lat_deg(spec.drift_m_per_s_lat * dt_s)
        drift_lon_deg = meters_to_lon_deg(spec.drift_m_per_s_lon * dt_s, spec.center_lat)
        noise_lat_deg = meters_to_lat_deg(random.gauss(0.0, spec.spatial_sigma_m))
        noise_lon_deg = meters_to_lon_deg(random.gauss(0.0, spec.spatial_sigma_m), spec.center_lat)
        rows.append({
            "id": id_start + i,
            "time": float(t),
            "lat": spec.center_lat + drift_lat_deg + noise_lat_deg,
            "lon": spec.center_lon + drift_lon_deg + noise_lon_deg,
        })
    return rows


def sample_global_noise(n_points: int, start_time_s: float, end_time_s: float, id_start: int) -> list[dict]:
    rows: list[dict] = []
    total_s = end_time_s - start_time_s
    for i in range(n_points):
        dt_s = random.uniform(0.0, total_s)
        rows.append({
            "id": id_start + i,
            "time": float(start_time_s + dt_s),
            "lat": random.uniform(-80.0, 80.0),
            "lon": random.uniform(-180.0, 180.0),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a mock detections parquet with unix-second float timestamps.")
    parser.add_argument("--output", default="detections.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-points", type=int, default=1000)
    args = parser.parse_args()

    random.seed(args.seed)

    t0 = datetime(2026, 3, 15, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    t1 = datetime(2026, 3, 15, 1, 0, 0, tzinfo=timezone.utc).timestamp()

    specs = [
        ClusterSpec(39.7392, -104.9903, t0 + 5 * 60, 250, 180.0, 180.0, 0.15, 0.35),
        ClusterSpec(40.7128, -74.0060, t0 + 20 * 60, 300, 240.0, 220.0, -0.10, 0.25),
        ClusterSpec(51.5074, -0.1278, t0 + 35 * 60, 220, 150.0, 150.0, 0.05, -0.15),
        ClusterSpec(34.0522, -118.2437, t0 + 45 * 60, 260, 300.0, 300.0, 0.20, 0.10),
    ]

    rows: list[dict] = []
    next_id = 0
    for spec in specs:
        pts = sample_cluster_points(spec, next_id)
        rows.extend(pts)
        next_id += len(pts)

    rows.extend(sample_global_noise(args.noise_points, t0, t1, next_id))

    df = pl.DataFrame(rows).with_columns([
        pl.col("id").cast(pl.Int64),
        pl.col("time").cast(pl.Float64),
        pl.col("lat").cast(pl.Float64),
        pl.col("lon").cast(pl.Float64),
    ]).sort("time")

    df.write_parquet(args.output)
    print(f"Wrote {args.output}")
    print(df.head(10))
    print(f"Rows: {df.height:,}")
    print(f"Time range: {df['time'].min()} to {df['time'].max()}")


if __name__ == "__main__":
    main()
