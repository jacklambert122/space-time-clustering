from __future__ import annotations

"""Provide spatial helpers for H3 partitioning and great-circle distance checks."""

import h3
import numpy as np

from .constants import EARTH_RADIUS_M


def haversine_matrix_m(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    """Compute pairwise great-circle distances, in meters, between two point sets.

    Inputs:
        lat1_deg: Latitudes for the first point set, in degrees.
        lon1_deg: Longitudes for the first point set, in degrees.
        lat2_deg: Latitudes for the second point set, in degrees.
        lon2_deg: Longitudes for the second point set, in degrees.

    Returns:
        A 2D array of pairwise distances in meters.
    """
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
    """Assign each latitude/longitude pair to an H3 cell at the given resolution.

    Inputs:
        lat: Point latitudes in degrees.
        lon: Point longitudes in degrees.
        res: H3 resolution to use for assignment.

    Returns:
        An object-dtype NumPy array of H3 cell ids.
    """
    return np.array([h3.latlng_to_cell(float(la), float(lo), res) for la, lo in zip(lat, lon)], dtype=object)


def neighbor_cells(cell: str) -> set[str]:
    """Return the H3 disk of radius 1 around a cell, including the cell itself.

    Inputs:
        cell: H3 cell id for the center cell.

    Returns:
        A set containing the center cell and its immediate H3 neighbors.
    """
    return set(h3.grid_disk(cell, 1))
