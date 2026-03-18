"""Test H3 assignment and distance helpers used by spatial clustering logic."""

import h3
import numpy as np

from space_time_cluster.spatial import assign_h3_cells, haversine_matrix_m, neighbor_cells


def test_haversine_matrix_m_returns_zero_for_identical_points() -> None:
    """Verify zero distance for identical coordinates.

    Inputs:
        None.

    Returns:
        None. The test asserts that identical points produce a zero distance.
    """
    dist = haversine_matrix_m(
        np.array([40.0]),
        np.array([-105.0]),
        np.array([40.0]),
        np.array([-105.0]),
    )
    assert dist.shape == (1, 1)
    assert dist[0, 0] == 0.0


def test_haversine_matrix_m_is_symmetric() -> None:
    """Verify haversine distance is symmetric for reversed inputs.

    Inputs:
        None.

    Returns:
        None. The test asserts equal forward and reverse distances.
    """
    forward = haversine_matrix_m(
        np.array([40.0]),
        np.array([-105.0]),
        np.array([40.01]),
        np.array([-105.0]),
    )
    reverse = haversine_matrix_m(
        np.array([40.01]),
        np.array([-105.0]),
        np.array([40.0]),
        np.array([-105.0]),
    )
    assert np.isclose(forward[0, 0], reverse[0, 0])
    assert 1000.0 < forward[0, 0] < 1200.0


def test_assign_h3_cells_matches_library_output() -> None:
    """Verify H3 assignment matches direct library calls.

    Inputs:
        None.

    Returns:
        None. The test asserts the helper returns the expected H3 cells.
    """
    lat = np.array([40.0, 40.1])
    lon = np.array([-105.0, -105.1])
    out = assign_h3_cells(lat, lon, 7)

    assert out.tolist() == [
        h3.latlng_to_cell(40.0, -105.0, 7),
        h3.latlng_to_cell(40.1, -105.1, 7),
    ]


def test_neighbor_cells_includes_origin_cell() -> None:
    """Verify the neighbor set includes the origin cell.

    Inputs:
        None.

    Returns:
        None. The test asserts the origin cell is present in the neighbor disk.
    """
    cell = h3.latlng_to_cell(40.0, -105.0, 7)
    neighbors = neighbor_cells(cell)
    assert cell in neighbors
    assert len(neighbors) >= 1
