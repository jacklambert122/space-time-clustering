from __future__ import annotations

"""Coordinate the end-to-end clustering workflow from input load to output writes."""

import polars as pl

from .classification import classify_clusters_against_polygons
from .config import ClusterConfig
from .graph import (
    apply_optional_neighbor_filter,
    build_edges_for_points,
    compute_neighbor_counts,
    connected_component_labels,
    relabel_from_filters,
)
from .io import ensure_out_dir, load_time_chunk
from .spatial import assign_h3_cells
from .summary import summarize_clusters
from .time_utils import time_bin_index_us, unix_seconds_float_to_us


def run_pipeline(cfg: ClusterConfig) -> None:
    """Run the end-to-end clustering workflow for a configured input slice.

    Inputs:
        cfg: Pipeline configuration covering input columns, thresholds, and outputs.

    Returns:
        None. The function writes output files and prints a brief run summary.
    """
    out_dir = ensure_out_dir(cfg.out_dir)
    df = load_time_chunk(cfg)
    if df.is_empty():
        print("No rows found in requested time range.")
        return
    ids = df[cfg.id_col].to_numpy()
    lat = df[cfg.lat_col].to_numpy()
    lon = df[cfg.lon_col].to_numpy()
    time_s = df[cfg.time_col].to_numpy()
    time_us = unix_seconds_float_to_us(time_s)
    h3_cells = assign_h3_cells(lat, lon, cfg.h3_res)
    time_bins = time_bin_index_us(time_us, cfg.time_bin_seconds)

    src, dst = build_edges_for_points(
        time_us=time_us,
        lat=lat,
        lon=lon,
        h3_cells=h3_cells,
        time_bins=time_bins,
        max_time_delta_s=cfg.max_time_delta_s,
        max_distance_m=cfg.max_distance_m,
    )
    kept_mask, src_f, dst_f = apply_optional_neighbor_filter(
        n_points=len(ids),
        src=src,
        dst=dst,
        use_filter=cfg.use_neighbor_count_filter,
        min_neighbors=cfg.min_neighbors,
    )
    labels_raw = connected_component_labels(len(ids), src_f, dst_f)
    labels_final = relabel_from_filters(labels_raw, kept_mask, cfg.min_cluster_size)
    neighbor_counts = compute_neighbor_counts(len(ids), src, dst)

    assigned = df.with_columns([
        pl.Series("cluster_id", labels_final),
        pl.Series("neighbor_count", neighbor_counts),
        pl.Series("kept_after_neighbor_filter", kept_mask),
        pl.Series("h3", h3_cells.tolist()),
    ])
    assigned.write_parquet(out_dir / "point_assignments.parquet")

    summary = summarize_clusters(time_s=time_s, lat=lat, lon=lon, labels=labels_final, cfg=cfg)
    summary.write_parquet(out_dir / "cluster_summary.parquet")

    if cfg.city_vector_path or cfg.lake_vector_path:
        city_hits, lake_hits = classify_clusters_against_polygons(summary, cfg.city_vector_path, cfg.lake_vector_path)
        if city_hits is not None:
            city_hits.to_parquet(out_dir / "cluster_city_hits.parquet")
        if lake_hits is not None:
            lake_hits.to_parquet(out_dir / "cluster_lake_hits.parquet")

    print(f"Loaded rows: {df.height:,}")
    print(f"Edges before neighbor filter: {len(src):,}")
    if cfg.use_neighbor_count_filter:
        print(f"Points kept after neighbor filter: {int(kept_mask.sum()):,} / {len(ids):,}")
        print(f"Edges after neighbor filter: {len(src_f):,}")
    print(summary.head(10))
