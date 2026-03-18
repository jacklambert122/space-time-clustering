"""Test per-cluster summary generation for surviving detection assignments."""

import numpy as np

from space_time_cluster.config import ClusterConfig
from space_time_cluster.summary import summarize_clusters


def test_summarize_clusters_builds_expected_rows() -> None:
    """Verify cluster summaries contain the expected aggregate fields.

    Inputs:
        None.

    Returns:
        None. The test asserts the generated summary rows match expectations.
    """
    cfg = ClusterConfig(parquet_path="unused", guard_band_m=10.0, radius_quantile=1.0)
    time_s = np.array([100.0, 110.0, 200.0, 210.0])
    lat = np.array([40.0, 40.001, 41.0, 41.0])
    lon = np.array([-105.0, -105.0, -106.0, -106.001])
    labels = np.array([0, 0, 1, -1], dtype=np.int64)

    summary = summarize_clusters(time_s=time_s, lat=lat, lon=lon, labels=labels, cfg=cfg)
    rows = summary.to_dicts()

    assert len(rows) == 2
    assert rows[0]["cluster_id"] == 0
    assert rows[0]["n_points"] == 2
    assert rows[0]["start_time"] == 100.0
    assert rows[0]["end_time"] == 110.0
    assert rows[0]["duration_s"] == 10.0
    assert rows[0]["radius_m"] > 10.0
    assert rows[0]["center_distance_mean_m"] > 0.0
    assert rows[0]["spatial_major_std_m"] >= rows[0]["spatial_minor_std_m"]
    assert rows[0]["path_length_m"] > 0.0
    assert rows[0]["cp_kf_rmse_m"] >= 0.0
    assert rows[0]["cp_kf_final_sigma_m"] > 0.0
    assert rows[1]["cluster_id"] == 1
    assert rows[1]["n_points"] == 1
    assert rows[1]["path_length_m"] == 0.0
    assert rows[1]["cp_kf_max_innovation_m"] == 0.0


def test_summarize_clusters_returns_empty_frame_when_all_points_removed() -> None:
    """Verify summarization returns an empty schema-preserving DataFrame when needed.

    Inputs:
        None.

    Returns:
        None. The test asserts the empty output retains the expected columns.
    """
    cfg = ClusterConfig(parquet_path="unused")
    summary = summarize_clusters(
        time_s=np.array([100.0]),
        lat=np.array([40.0]),
        lon=np.array([-105.0]),
        labels=np.array([-1], dtype=np.int64),
        cfg=cfg,
    )

    assert summary.is_empty()
    assert summary.columns == [
        "cluster_id",
        "n_points",
        "start_time",
        "end_time",
        "duration_s",
        "center_lat",
        "center_lon",
        "radius_m",
        "center_distance_mean_m",
        "center_distance_std_m",
        "spatial_major_std_m",
        "spatial_minor_std_m",
        "spatial_axis_ratio",
        "bbox_width_m",
        "bbox_height_m",
        "path_length_m",
        "net_displacement_m",
        "mean_step_distance_m",
        "max_step_distance_m",
        "mean_speed_mps",
        "max_speed_mps",
        "meander_ratio",
        "cp_kf_mean_innovation_m",
        "cp_kf_max_innovation_m",
        "cp_kf_rmse_m",
        "cp_kf_final_sigma_m",
        "cp_kf_mean_nis",
    ]
