import geopandas as gpd
import polars as pl
import numpy as np


def convert_lakes(input_path: str, output_parquet: str):
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


def convert_land(input_path: str, output_parquet: str):
    gdf = gpd.read_file(input_path).to_crs("EPSG:4326")

    # sample points on polygon grid (fast classification approach)
    minx, miny, maxx, maxy = gdf.total_bounds

    lats = np.linspace(miny, maxy, 1000)
    lons = np.linspace(minx, maxx, 1000)

    points = []

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