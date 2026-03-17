from space_time_cluster.pipeline import unix_seconds_float_to_us, time_bin_index_us
import numpy as np


def test_unix_seconds_float_to_us_rounds_consistently():
    arr = np.array([1000.0, 1000.125, 1000.9999994])
    out = unix_seconds_float_to_us(arr)
    assert out.tolist() == [1000000000, 1000125000, 1000999999]


def test_time_bin_index_us():
    arr = np.array([0, 59_999_999, 60_000_000, 120_000_000], dtype=np.int64)
    bins = time_bin_index_us(arr, 60.0)
    assert bins.tolist() == [0, 0, 1, 2]
