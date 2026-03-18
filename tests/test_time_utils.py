"""Test timestamp conversion and fixed-width time binning helpers."""

import numpy as np
from space_time_cluster.time_utils import time_bin_index_us, unix_seconds_float_to_us


def test_unix_seconds_float_to_us_rounds_consistently() -> None:
    """Verify unix-second floats are rounded consistently to microseconds.

    Inputs:
        None.

    Returns:
        None. The test asserts the expected rounded microsecond values.
    """
    arr = np.array([1000.0, 1000.125, 1000.9999994])
    out = unix_seconds_float_to_us(arr)
    assert out.tolist() == [1000000000, 1000125000, 1000999999]


def test_time_bin_index_us() -> None:
    """Verify integer microsecond timestamps map to the expected bins.

    Inputs:
        None.

    Returns:
        None. The test asserts the expected time-bin assignments.
    """
    arr = np.array([0, 59_999_999, 60_000_000, 120_000_000], dtype=np.int64)
    bins = time_bin_index_us(arr, 60.0)
    assert bins.tolist() == [0, 0, 1, 2]
