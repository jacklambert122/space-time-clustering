"""Test graph construction, neighbor filtering, and cluster relabeling behavior."""

import numpy as np
from h3 import latlng_to_cell

from space_time_cluster.graph import (
    apply_optional_neighbor_filter,
    build_edges_for_points,
    compute_neighbor_counts,
    connected_component_labels,
    relabel_from_filters,
)


def test_build_edges_for_points_respects_time_and_distance_thresholds() -> None:
    """Verify edge construction applies time and distance thresholds.

    Inputs:
        None.

    Returns:
        None. The test asserts only the nearby point pair is connected.
    """
    time_us = np.array([0, 10_000_000, 50_000_000], dtype=np.int64)
    lat = np.array([40.0, 40.0005, 41.0])
    lon = np.array([-105.0, -105.0005, -105.0])
    h3_cells = np.array([latlng_to_cell(float(la), float(lo), 7) for la, lo in zip(lat, lon)], dtype=object)
    time_bins = np.array([0, 0, 0], dtype=np.int64)

    src, dst = build_edges_for_points(
        time_us=time_us,
        lat=lat,
        lon=lon,
        h3_cells=h3_cells,
        time_bins=time_bins,
        max_time_delta_s=30.0,
        max_distance_m=200.0,
    )

    assert src.tolist() == [0]
    assert dst.tolist() == [1]


def test_compute_neighbor_counts_counts_both_endpoints() -> None:
    """Verify neighbor counts are incremented for both edge endpoints.

    Inputs:
        None.

    Returns:
        None. The test asserts the expected per-point degrees.
    """
    counts = compute_neighbor_counts(
        n_points=4,
        src=np.array([0, 1, 1], dtype=np.int64),
        dst=np.array([1, 2, 3], dtype=np.int64),
    )
    assert counts.tolist() == [1, 3, 1, 1]


def test_apply_optional_neighbor_filter_filters_edges_by_min_neighbors() -> None:
    """Verify neighbor filtering drops points and edges below the threshold.

    Inputs:
        None.

    Returns:
        None. The test asserts the expected kept mask and filtered edges.
    """
    kept, src_f, dst_f = apply_optional_neighbor_filter(
        n_points=4,
        src=np.array([0, 1, 1], dtype=np.int64),
        dst=np.array([1, 2, 3], dtype=np.int64),
        use_filter=True,
        min_neighbors=2,
    )

    assert kept.tolist() == [False, True, False, False]
    assert src_f.tolist() == []
    assert dst_f.tolist() == []


def test_connected_component_labels_groups_connected_points() -> None:
    """Verify connected-component labels group linked points together.

    Inputs:
        None.

    Returns:
        None. The test asserts separate graph components receive different labels.
    """
    labels = connected_component_labels(
        n_points=5,
        src=np.array([0, 1, 3], dtype=np.int64),
        dst=np.array([1, 2, 4], dtype=np.int64),
    )

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def test_relabel_from_filters_removes_small_clusters_and_reindexes() -> None:
    """Verify relabeling drops small clusters and compacts surviving labels.

    Inputs:
        None.

    Returns:
        None. The test asserts filtered labels are renumbered as expected.
    """
    labels = np.array([5, 5, 7, 9, 9], dtype=np.int64)
    kept_mask = np.array([True, True, True, False, False])

    out = relabel_from_filters(labels, kept_mask, min_cluster_size=2)

    assert out.tolist() == [0, 0, -1, -1, -1]
