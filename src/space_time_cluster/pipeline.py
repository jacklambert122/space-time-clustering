from __future__ import annotations

from pathlib import Path

import h3
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import ClusterConfig

EARTH_RADIUS_M = 6371000.0
TIME_SCALE_US = 1_000_000


def haversine_matrix_m(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(lat1_deg)[:, None]
    lon1 = np.radians(lon1_deg)[:, None]
    lat2 = np.radians(lat2_deg)[None, :]
    lon2 = np.radians(lon2_deg)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return EARTH_RADIUS_M * c


def assign_h3_cells(lat: np.ndarray, lon: np.ndarray, res: int) -> np.ndarray:
    return np.array([h3.latlng_to_cell(float(la), float(lo), res) for la, lo in zip(lat, lon)], dtype=object)


def unix_seconds_float_to_us(time_s: np.ndarray) -> np.ndarray:
    return np.round(time_s * TIME_SCALE_US).astype(np.int64)


def time_bin_index_us(time_us: np.ndarray, bin_seconds: float) -> np.ndarray:
    bin_us = np.int64(round(bin_seconds * TIME_SCALE_US))
    return time_us // bin_us


def neighbor_cells(cell: str) -> set[str]:
    return set(h3.grid_disk(cell, 1))


def load_time_chunk(cfg: ClusterConfig) -> pl.DataFrame:
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


def build_edges_for_points(
    time_us: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    h3_cells: np.ndarray,
    time_bins: np.ndarray,
    max_time_delta_s: float,
    max_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(time_us)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    bucket_to_rows: dict[tuple[int, str], list[int]] = {}
    for i, (tb, hc) in enumerate(zip(time_bins, h3_cells)):
        bucket_to_rows.setdefault((int(tb), str(hc)), []).append(i)

    all_src: list[np.ndarray] = []
    all_dst: list[np.ndarray] = []
    processed_pairs: set[tuple[tuple[int, str], tuple[int, str]]] = set()
    max_time_delta_us = np.int64(round(max_time_delta_s * TIME_SCALE_US))

    for key_a, rows_a_list in bucket_to_rows.items():
        tb_a, cell_a = key_a
        rows_a = np.asarray(rows_a_list, dtype=np.int64)

        candidate_keys_b: list[tuple[int, str]] = []
        for dt_bin in (-1, 0, 1):
            tb_b = tb_a + dt_bin
            for cell_b in neighbor_cells(cell_a):
                key_b = (tb_b, cell_b)
                if key_b in bucket_to_rows:
                    candidate_keys_b.append(key_b)

        for key_b in candidate_keys_b:
            ordered = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
            if ordered in processed_pairs:
                continue
            processed_pairs.add(ordered)

            rows_b = np.asarray(bucket_to_rows[key_b], dtype=np.int64)
            ta = time_us[rows_a]
            tb = time_us[rows_b]
            dt = np.abs(ta[:, None] - tb[None, :])
            time_mask = dt <= max_time_delta_us
            if not np.any(time_mask):
                continue

            dist = haversine_matrix_m(lat[rows_a], lon[rows_a], lat[rows_b], lon[rows_b])
            mask = time_mask & (dist <= max_distance_m)

            if key_a == key_b:
                ia, ib = np.where(mask)
                keep = rows_a[ia] < rows_b[ib]
                ia = ia[keep]
                ib = ib[keep]
            else:
                ia, ib = np.where(mask)

            if len(ia) == 0:
                continue

            all_src.append(rows_a[ia])
            all_dst.append(rows_b[ib])

    if not all_src:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def compute_neighbor_counts(n_points: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    counts = np.zeros(n_points, dtype=np.int64)
    if len(src) == 0:
        return counts
    np.add.at(counts, src, 1)
    np.add.at(counts, dst, 1)
    return counts


def apply_optional_neighbor_filter(
    n_points: int,
    src: np.ndarray,
    dst: np.ndarray,
    use_filter: bool,
    min_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not use_filter:
        kept = np.ones(n_points, dtype=bool)
        return kept, src, dst
    counts = compute_neighbor_counts(n_points, src, dst)
    kept = counts >= min_neighbors
    if len(src) == 0:
        return kept, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    edge_keep = kept[src] & kept[dst]
    return kept, src[edge_keep], dst[edge_keep]


def connected_component_labels(n_points: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if n_points == 0:
        return np.empty(0, dtype=np.int64)
    if len(src) == 0:
        return np.arange(n_points, dtype=np.int64)
    data = np.ones(len(src) * 2, dtype=np.uint8)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    graph = coo_matrix((data, (row, col)), shape=(n_points, n_points)).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int64)


def relabel_from_filters(labels: np.ndarray, kept_mask: np.ndarray, min_cluster_size: int) -> np.ndarray:
    out = labels.copy()
    out[~kept_mask] = -1
    valid_mask = out != -1
    if not np.any(valid_mask):
        return out
    valid_labels = out[valid_mask]
    unique, counts = np.unique(valid_labels, return_counts=True)
    small = set(unique[counts < min_cluster_size])
    if small:
        out[np.isin(out, list(small))] = -1
    valid_final = np.unique(out[out != -1])
    remap = {old: new for new, old in enumerate(valid_final)}
    if remap:
        idx = out != -1
        out[idx] = np.array([remap[x] for x in out[idx]], dtype=np.int64)
    return out


def summarize_clusters(
    time_s: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    labels: np.ndarray,
    cfg: ClusterConfig,
) -> pl.DataFrame:
    keep = labels != -1
    time_s = time_s[keep]
    lat = lat[keep]
    lon = lon[keep]
    labels = labels[keep]
    if len(labels) == 0:
        return pl.DataFrame({
            "cluster_id": [],
            "n_points": [],
            "start_time": [],
            "end_time": [],
            "center_lat": [],
            "center_lon": [],
            "radius_m": [],
        })
    rows = []
    for cid in np.unique(labels):
        mask = labels == cid
        lat_c = lat[mask]
        lon_c = lon[mask]
        time_c = time_s[mask]
        center_lat = float(np.median(lat_c))
        center_lon = float(np.median(lon_c))
        dist = haversine_matrix_m(lat_c, lon_c, np.array([center_lat]), np.array([center_lon]))[:, 0]
        radius_m = float(np.quantile(dist, cfg.radius_quantile) + cfg.guard_band_m)
        rows.append({
            "cluster_id": int(cid),
            "n_points": int(mask.sum()),
            "start_time": float(time_c.min()),
            "end_time": float(time_c.max()),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_m": radius_m,
        })
    return pl.DataFrame(rows).sort("cluster_id")


def classify_clusters_against_polygons(cluster_summary: pl.DataFrame, city_path: str | None, lake_path: str | None):
    if city_path is None and lake_path is None:
        return None, None
    import geopandas as gpd
    pdf = cluster_summary.to_pandas()
    if pdf.empty:
        return None, None
    centers = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf["center_lon"], pdf["center_lat"]), crs="EPSG:4326")
    local_crs = centers.estimate_utm_crs()
    centers_local = centers.to_crs(local_crs).copy()
    centers_local["geometry"] = centers_local.geometry.buffer(centers_local["radius_m"].to_numpy())
    city_hits = None
    lake_hits = None
    if city_path is not None:
        cities = gpd.read_file(city_path).to_crs(local_crs)
        city_hits = gpd.sjoin(centers_local, cities, how="left", predicate="intersects")
    if lake_path is not None:
        lakes = gpd.read_file(lake_path).to_crs(local_crs)
        lake_hits = gpd.sjoin(centers_local, lakes, how="left", predicate="intersects")
    return city_hits, lake_hits


def run_pipeline(cfg: ClusterConfig) -> None:
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
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
    assigned.write_parquet(Path(cfg.out_dir) / "point_assignments.parquet")

    summary = summarize_clusters(time_s=time_s, lat=lat, lon=lon, labels=labels_final, cfg=cfg)
    summary.write_parquet(Path(cfg.out_dir) / "cluster_summary.parquet")

    if cfg.city_vector_path or cfg.lake_vector_path:
        city_hits, lake_hits = classify_clusters_against_polygons(summary, cfg.city_vector_path, cfg.lake_vector_path)
        if city_hits is not None:
            city_hits.to_parquet(Path(cfg.out_dir) / "cluster_city_hits.parquet")
        if lake_hits is not None:
            lake_hits.to_parquet(Path(cfg.out_dir) / "cluster_lake_hits.parquet")

    print(f"Loaded rows: {df.height:,}")
    print(f"Edges before neighbor filter: {len(src):,}")
    if cfg.use_neighbor_count_filter:
        print(f"Points kept after neighbor filter: {int(kept_mask.sum()):,} / {len(ids):,}")
        print(f"Edges after neighbor filter: {len(src_f):,}")
    print(summary.head(10))
