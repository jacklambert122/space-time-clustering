"""Convert water and land layers into simplified parquet point datasets."""

import argparse
import geopandas as gpd
import numpy as np
import polars as pl


def convert_lakes(input_path: str, output_parquet: str) -> None:
    """Convert a lake polygon layer into centroid-based parquet records.

    Inputs:
        input_path: Path to the source lake polygon vector layer.
        output_parquet: Destination parquet path for centroid records.

    Returns:
        None. The function writes the converted parquet file to disk.
    """
    gdf = gpd.read_file(input_path)

    gdf = gdf.to_crs("EPSG:4326")

    # centroid for fast lookup (you can keep polygons too if needed)
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x

    df = pl.DataFrame({
        "name": gdf.get("Lake_name", [""] * len(gdf)),
        "lat": gdf["lat"].to_numpy(),
        "lon": gdf["lon"].to_numpy(),
        "area_km2": gdf.get("Area_km2", np.zeros(len(gdf))),
        "type": ["lake"] * len(gdf),
    })

    df.write_parquet(output_parquet)
    print(f"Wrote lakes → {output_parquet}")


def convert_land(input_path: str, output_parquet: str) -> None:
    """Sample a coarse grid of points that fall within the land polygons.

    Inputs:
        input_path: Path to the source land polygon vector layer.
        output_parquet: Destination parquet path for sampled point records.

    Returns:
        None. The function writes the converted parquet file to disk.
    """
    gdf = gpd.read_file(input_path).to_crs("EPSG:4326")

    # sample points on polygon grid (fast classification approach)
    minx, miny, maxx, maxy = gdf.total_bounds

    lats = np.linspace(miny, maxy, 1000)
    lons = np.linspace(minx, maxx, 1000)

    points: list[tuple[float, float]] = []

    for lat in lats:
        for lon in lons:
            pt = gpd.points_from_xy([lon], [lat])[0]
            if gdf.contains(pt).any():
                points.append((lat, lon))

    df = pl.DataFrame({
        "lat": [p[0] for p in points],
        "lon": [p[1] for p in points],
        "type": ["land"] * len(points),
    })

    df.write_parquet(output_parquet)
    print(f"Wrote land → {output_parquet}")


def main() -> None:
    """Parse CLI arguments and run a water or land conversion task.

    Inputs:
        None. Arguments are read from the command line.

    Returns:
        None. The function runs the selected conversion and writes parquet output.
    """
    parser = argparse.ArgumentParser(
        description="Convert lake or land reference layers into simplified parquet datasets."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    lakes = sub.add_parser("lakes", help="Convert a lake polygon layer to centroid parquet records")
    lakes.add_argument(
        "--input",
        default="data/unpacked/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp",
        help="Path to the source lake polygon layer",
    )
    lakes.add_argument(
        "--output",
        default="data/lakes.parquet",
        help="Destination parquet path",
    )

    land = sub.add_parser("land", help="Convert a land polygon layer to sampled point parquet records")
    land.add_argument(
        "--input",
        default="data/unpacked/ne_10m_land/ne_10m_land.shp",
        help="Path to the source land polygon layer",
    )
    land.add_argument(
        "--output",
        default="data/land_points.parquet",
        help="Destination parquet path",
    )

    args = parser.parse_args()

    if args.command == "lakes":
        convert_lakes(args.input, args.output)
    else:
        convert_land(args.input, args.output)


if __name__ == "__main__":
    main()
