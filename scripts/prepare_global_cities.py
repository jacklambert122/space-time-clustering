from __future__ import annotations

"""Prepare normalized global city reference datasets from GeoNames or Natural Earth."""

import argparse
import io
from pathlib import Path
import zipfile

import polars as pl


GEONAMES_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code", "admin2_code",
    "admin3_code", "admin4_code", "population", "elevation", "dem", "timezone", "modification_date",
]


def maybe_add_h3(df: pl.DataFrame, h3_res: int | None) -> pl.DataFrame:
    """Attach an H3 column when a target resolution is provided.

    Inputs:
        df: Normalized city DataFrame with ``lat`` and ``lon`` columns.
        h3_res: Optional H3 resolution to compute.

    Returns:
        The original DataFrame or a copy with an added ``h3`` column.
    """
    if h3_res is None:
        return df
    import h3
    cells = [h3.latlng_to_cell(float(la), float(lo), h3_res) for la, lo in zip(df["lat"].to_list(), df["lon"].to_list())]
    return df.with_columns(pl.Series("h3", cells))


def geonames_to_polars(input_zip: str, min_population: int) -> pl.DataFrame:
    """Load a GeoNames zip archive into the normalized city schema.

    Inputs:
        input_zip: Path to a GeoNames zip archive containing a tab-separated text file.
        min_population: Minimum population threshold for retained rows.

    Returns:
        A normalized Polars DataFrame of city records.
    """
    with zipfile.ZipFile(input_zip, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".txt")]
        if not members:
            raise ValueError(f"No .txt member found in {input_zip}")
        with zf.open(members[0], "r") as f:
            raw = f.read()
    df = pl.read_csv(
        io.BytesIO(raw),
        separator="\t",
        has_header=False,
        new_columns=GEONAMES_COLUMNS,
        infer_schema_length=10000,
        null_values=["", "NULL"],
        ignore_errors=True,
    )
    out = (
        df.select([
            pl.col("geonameid").cast(pl.Int64, strict=False).alias("city_id"),
            pl.col("name").cast(pl.Utf8).alias("name"),
            pl.col("asciiname").cast(pl.Utf8).alias("ascii_name"),
            pl.col("latitude").cast(pl.Float64, strict=False).alias("lat"),
            pl.col("longitude").cast(pl.Float64, strict=False).alias("lon"),
            pl.col("country_code").cast(pl.Utf8).alias("country_code"),
            pl.col("admin1_code").cast(pl.Utf8).alias("admin1_code"),
            pl.col("admin2_code").cast(pl.Utf8).alias("admin2_code"),
            pl.col("feature_class").cast(pl.Utf8).alias("feature_class"),
            pl.col("feature_code").cast(pl.Utf8).alias("feature_code"),
            pl.col("population").cast(pl.Int64, strict=False).fill_null(0).alias("population"),
            pl.col("timezone").cast(pl.Utf8).alias("timezone"),
            pl.col("modification_date").cast(pl.Utf8).alias("modification_date"),
        ])
        .filter(
            pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
            & (pl.col("lat") >= -90.0) & (pl.col("lat") <= 90.0)
            & (pl.col("lon") >= -180.0) & (pl.col("lon") <= 180.0)
            & (pl.col("population") >= min_population)
        )
        .sort(["country_code", "population", "name"], descending=[False, True, False])
    )
    return out


def natural_earth_to_polars(input_vector: str, min_scalerank: int | None) -> pl.DataFrame:
    """Load Natural Earth populated places into the normalized city schema.

    Inputs:
        input_vector: Path to a Natural Earth populated places vector file.
        min_scalerank: Optional maximum scalerank value to retain.

    Returns:
        A normalized Polars DataFrame of city records.
    """
    import geopandas as gpd

    gdf = gpd.read_file(input_vector)
    if gdf.crs is None:
        raise ValueError("Input vector has no CRS")
    if str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    cols = {c.lower(): c for c in gdf.columns}

    def pick(*names: str) -> str | None:
        """Pick the first available source column name from a list of candidates.

        Inputs:
            *names: Candidate column names in priority order.

        Returns:
            The matching source column name, or ``None`` if none are present.
        """
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    name_col = pick("name", "nameascii", "name_en")
    scalerank_col = pick("scalerank")
    adm0_col = pick("adm0name", "admin", "sov0name")
    adm1_col = pick("adm1name")
    pop_col = pick("pop_max", "gn_pop", "pop_min")
    feature_col = pick("featurecla")
    timezone_col = pick("timezone")

    pdf = pl.DataFrame({
        "city_id": list(range(len(gdf))),
        "name": gdf[name_col].astype(str).tolist() if name_col else [""] * len(gdf),
        "ascii_name": gdf[name_col].astype(str).tolist() if name_col else [""] * len(gdf),
        "lat": gdf.geometry.y.astype(float).tolist(),
        "lon": gdf.geometry.x.astype(float).tolist(),
        "country_code": gdf[adm0_col].astype(str).tolist() if adm0_col else [None] * len(gdf),
        "admin1_code": gdf[adm1_col].astype(str).tolist() if adm1_col else [None] * len(gdf),
        "admin2_code": [None] * len(gdf),
        "feature_class": ["P"] * len(gdf),
        "feature_code": gdf[feature_col].astype(str).tolist() if feature_col else [None] * len(gdf),
        "population": pl.Series(gdf[pop_col].tolist()).cast(pl.Int64, strict=False).fill_null(0).to_list() if pop_col else [0] * len(gdf),
        "timezone": gdf[timezone_col].astype(str).tolist() if timezone_col else [None] * len(gdf),
        "modification_date": [None] * len(gdf),
        "scalerank": pl.Series(gdf[scalerank_col].tolist()).cast(pl.Int64, strict=False).to_list() if scalerank_col else [None] * len(gdf),
    })

    out = pdf.filter(
        pl.col("lat").is_not_null() & pl.col("lon").is_not_null()
        & (pl.col("lat") >= -90.0) & (pl.col("lat") <= 90.0)
        & (pl.col("lon") >= -180.0) & (pl.col("lon") <= 180.0)
    )
    if min_scalerank is not None and "scalerank" in out.columns:
        out = out.filter(pl.col("scalerank").is_not_null() & (pl.col("scalerank") <= min_scalerank))
    return out.sort(["population", "name"], descending=[True, False])


def write_vector_outputs(df: pl.DataFrame, out_base: str, layer_name: str) -> None:
    """Write the normalized city dataset to parquet and vector formats.

    Inputs:
        df: Normalized city DataFrame with coordinate columns.
        out_base: Output path prefix without file extension.
        layer_name: Layer name to use in vector outputs.

    Returns:
        None. The function writes parquet and vector files to disk.
    """
    import geopandas as gpd
    from pyogrio import list_drivers

    Path(out_base).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(f"{out_base}.parquet")

    pdf = df.to_pandas()
    gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf["lon"], pdf["lat"]), crs="EPSG:4326")
    gdf.to_file(f"{out_base}.gpkg", layer=layer_name, driver="GPKG")

    drivers = list_drivers(read=False)
    if "OpenFileGDB" in drivers and drivers["OpenFileGDB"]:
        gdb_path = f"{out_base}.gdb"
        gdf.to_file(gdb_path, layer=layer_name, driver="OpenFileGDB")
        print(f"Wrote {gdb_path}")
    else:
        print("OpenFileGDB write driver not available in this GDAL build; wrote .parquet and .gpkg instead.")


def main() -> None:
    """Parse CLI arguments and generate prepared city reference data.

    Inputs:
        None. Arguments are read from the command line.

    Returns:
        None. The function writes normalized city reference outputs and prints a summary.
    """
    parser = argparse.ArgumentParser(description="Prepare a global cities dataset for the clustering pipeline.")
    sub = parser.add_subparsers(dest="source", required=True)

    p_geo = sub.add_parser("geonames")
    p_geo.add_argument("--input", required=True, help="Path to GeoNames zip such as cities500.zip or allCountries.zip")
    p_geo.add_argument("--out-base", required=True, help="Output base path without extension")
    p_geo.add_argument("--min-population", type=int, default=0)
    p_geo.add_argument("--h3-res", type=int, default=None)

    p_ne = sub.add_parser("naturalearth")
    p_ne.add_argument("--input", required=True, help="Path to Natural Earth populated places shp/gpkg")
    p_ne.add_argument("--out-base", required=True, help="Output base path without extension")
    p_ne.add_argument("--min-scalerank", type=int, default=None)
    p_ne.add_argument("--h3-res", type=int, default=None)

    args = parser.parse_args()

    if args.source == "geonames":
        df = geonames_to_polars(args.input, args.min_population)
    else:
        df = natural_earth_to_polars(args.input, args.min_scalerank)

    df = maybe_add_h3(df, args.h3_res)
    write_vector_outputs(df, args.out_base, layer_name="cities")
    print(df.head(10))
    print(f"Rows: {df.height:,}")


if __name__ == "__main__":
    main()
