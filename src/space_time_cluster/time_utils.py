from __future__ import annotations

"""Convert source timestamps and assign them to fixed-width clustering time bins."""

import numpy as np

from .constants import TIME_SCALE_US


def unix_seconds_float_to_us(time_s: np.ndarray) -> np.ndarray:
    """Convert unix-second floats to rounded integer microseconds.

    Inputs:
        time_s: NumPy array of unix timestamps stored as floating-point seconds.

    Returns:
        A NumPy array of rounded integer microseconds.
    """
    return np.round(time_s * TIME_SCALE_US).astype(np.int64)


def time_bin_index_us(time_us: np.ndarray, bin_seconds: float) -> np.ndarray:
    """Map integer microsecond timestamps into fixed-width time bins.

    Inputs:
        time_us: NumPy array of integer microsecond timestamps.
        bin_seconds: Width of each bin in seconds.

    Returns:
        A NumPy array of integer bin indices.
    """
    bin_us = np.int64(round(bin_seconds * TIME_SCALE_US))
    return time_us // bin_us
