"""Test GeoNames and Natural Earth city readers used for reference data prep."""

from pathlib import Path
import zipfile

import geopandas as gpd
import polars as pl
from shapely.geometry import Point

from conftest import load_script_module

prepare_global_cities = load_script_module("prepare_global_cities.py")
geonames_to_polars = prepare_global_cities.geonames_to_polars
natural_earth_to_polars = prepare_global_cities.natural_earth_to_polars


def test_geonames_to_polars_filters_and_normalizes_rows(tmp_path: Path) -> None:
    """Verify the GeoNames reader filters invalid rows and normalizes columns.

    Inputs:
        tmp_path: Pytest-provided temporary base directory.

    Returns:
        None. The test asserts the normalized GeoNames output matches expectations.
    """
    zip_path = tmp_path / "cities500.zip"
    rows = [
        "1\tDenver\tDenver\t\t39.7392\t-104.9903\tP\tPPLA2\tUS\t\tCO\t001\t\t\t715522\t\t\tAmerica/Denver\t2024-01-01",
        "2\tBadCoords\tBadCoords\t\t95.0\t-104.0\tP\tPPL\tUS\t\tCO\t001\t\t\t2000\t\t\tAmerica/Denver\t2024-01-01",
        "3\tSmallTown\tSmallTown\t\t40.0\t-105.0\tP\tPPL\tUS\t\tCO\t001\t\t\t50\t\t\tAmerica/Denver\t2024-01-01",
    ]
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("cities500.txt", "\n".join(rows))

    out = geonames_to_polars(str(zip_path), min_population=1000)

    assert out.shape == (1, 13)
    assert out["city_id"].to_list() == [1]
    assert out["name"].to_list() == ["Denver"]
    assert out["lat"].to_list() == [39.7392]
    assert out["lon"].to_list() == [-104.9903]
    assert out["population"].to_list() == [715522]


def test_natural_earth_to_polars_reads_vector_and_applies_scalerank_filter(tmp_path: Path) -> None:
    """Verify the Natural Earth reader loads vector data and filters by scalerank.

    Inputs:
        tmp_path: Pytest-provided temporary base directory.

    Returns:
        None. The test asserts the normalized Natural Earth output matches expectations.
    """
    source_path = tmp_path / "places.geojson"
    gdf = gpd.GeoDataFrame(
        {
            "name": ["Boulder", "Tiny Place"],
            "adm0name": ["United States", "United States"],
            "adm1name": ["Colorado", "Colorado"],
            "pop_max": [100000, 1000],
            "featurecla": ["Admin-1 capital", "Populated place"],
            "timezone": ["America/Denver", "America/Denver"],
            "scalerank": [3, 10],
        },
        geometry=[Point(-105.2705, 40.015), Point(-105.3, 40.02)],
        crs="EPSG:4326",
    )
    gdf.to_file(source_path, driver="GeoJSON")

    out = natural_earth_to_polars(str(source_path), min_scalerank=5)

    assert out.shape[0] == 1
    assert out["name"].to_list() == ["Boulder"]
    assert out["country_code"].to_list() == ["United States"]
    assert out["admin1_code"].to_list() == ["Colorado"]
    assert out["population"].to_list() == [100000]
    assert out["scalerank"].to_list() == [3]
