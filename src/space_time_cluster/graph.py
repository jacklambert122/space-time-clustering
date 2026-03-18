from __future__ import annotations

"""Build the clustering graph and apply graph-based filtering and relabeling steps."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .constants import TIME_SCALE_US
from .spatial import haversine_matrix_m, neighbor_cells


def build_edges_for_points(
    time_us: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    h3_cells: np.ndarray,
    time_bins: np.ndarray,
    max_time_delta_s: float,
    max_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build undirected candidate edges that satisfy time and distance thresholds.

    Inputs:
        time_us: Detection timestamps in integer microseconds.
        lat: Detection latitudes in degrees.
        lon: Detection longitudes in degrees.
        h3_cells: H3 cell ids for each detection.
        time_bins: Coarse time-bin indices for each detection.
        max_time_delta_s: Maximum allowed time separation, in seconds.
        max_distance_m: Maximum allowed spatial separation, in meters.

    Returns:
        Two integer arrays containing the source and destination indices for each
        surviving undirected edge.
    """
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
    """Count incident edges for each point in an undirected graph.

    Inputs:
        n_points: Total number of points in the graph.
        src: Source indices for each edge.
        dst: Destination indices for each edge.

    Returns:
        An integer array giving the degree of each point.
    """
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
    """Filter points and edges by minimum neighbor count when enabled.

    Inputs:
        n_points: Total number of points in the graph.
        src: Source indices for each edge.
        dst: Destination indices for each edge.
        use_filter: Whether to apply the neighbor-count filter.
        min_neighbors: Minimum number of neighbors required to keep a point.

    Returns:
        A tuple of the kept-point mask and the filtered source and destination
        edge arrays.
    """
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
    """Return connected-component labels for the undirected edge list.

    Inputs:
        n_points: Total number of points in the graph.
        src: Source indices for each edge.
        dst: Destination indices for each edge.

    Returns:
        An integer label array with one connected-component id per point.
    """
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
    """Apply point and cluster-size filters, then compact labels to 0..N-1.

    Inputs:
        labels: Raw connected-component labels for each point.
        kept_mask: Boolean mask showing which points survive point-level filtering.
        min_cluster_size: Minimum number of points required for a cluster to survive.

    Returns:
        An integer label array where removed points and undersized clusters are set
        to ``-1`` and surviving clusters are renumbered densely.
    """
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
