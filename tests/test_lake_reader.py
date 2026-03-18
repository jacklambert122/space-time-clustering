"""Test lake conversion helpers used for simplified reference-layer outputs."""

from pathlib import Path

import geopandas as gpd
import polars as pl
from shapely.geometry import Polygon

from conftest import load_script_module

prepare_water_land = load_script_module("prepare_water_land.py")
convert_lakes = prepare_water_land.convert_lakes


def test_convert_lakes_writes_centroid_parquet(tmp_path: Path) -> None:
    """Verify lake polygons are converted into centroid-based parquet records.

    Inputs:
        tmp_path: Pytest-provided temporary base directory.

    Returns:
        None. The test asserts the written parquet contains the expected lake fields.
    """
    source_path = tmp_path / "lakes.geojson"
    output_path = tmp_path / "lakes.parquet"
    gdf = gpd.GeoDataFrame(
        {
            "Lake_name": ["Test Lake"],
            "Area_km2": [12.5],
        },
        geometry=[
            Polygon(
                [
                    (-105.0, 40.0),
                    (-104.9, 40.0),
                    (-104.9, 40.1),
                    (-105.0, 40.1),
                ]
            )
        ],
        crs="EPSG:4326",
    )
    gdf.to_file(source_path, driver="GeoJSON")

    convert_lakes(str(source_path), str(output_path))
    out = pl.read_parquet(output_path)

    assert out.shape == (1, 5)
    assert out["name"].to_list() == ["Test Lake"]
    assert out["area_km2"].to_list() == [12.5]
    assert out["type"].to_list() == ["lake"]
    assert 40.0 < out["lat"][0] < 40.1
    assert -105.0 < out["lon"][0] < -104.9
