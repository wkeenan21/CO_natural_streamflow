import geopandas as gpd
from pathlib import Path
import os
from os.path import join
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.strtree import STRtree
from pygeohydro import NWIS
nwis = NWIS()

###############
# 1 download DEM (R script)
# 2 find gages with good streamflow data in Upper Col basin
# 3 delineate basins
# 4 download climate reanlysis data
# 5 harmonize it

cwd = os.getcwd()


################## STEP 1 #####################

# query basins
ucol = gpd.read_file(join(cwd, r"data\shapefiles\UCOL\globalwatershed.shp")).set_crs('4326')
ucol_geom = ucol.union_all()  # single shapely geometry for spatial filtering

def make_tiles(gdf, n_tiles_x=3, n_tiles_y=3):
    """Split the bounding box of a GeoDataFrame into an n x m grid of sub-bboxes."""
    minx, miny, maxx, maxy = gdf.total_bounds
    xs = [minx + i * (maxx - minx) / n_tiles_x for i in range(n_tiles_x + 1)]
    ys = [miny + i * (maxy - miny) / n_tiles_y for i in range(n_tiles_y + 1)]
    tiles = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            tiles.append((x0, y0, x1, y1))
    return tiles

tiles = make_tiles(ucol, n_tiles_x=3, n_tiles_y=3)
print(f"Querying NWIS across {len(tiles)} tiles…")

# ── Query each tile and collect results ───────────────────────────────────────
records = []

for i, (x0, y0, x1, y1) in enumerate(tiles):
    try:
        chunk = nwis.get_info(
            {
                "bBox":          f"{x0:.6f},{y0:.6f},{x1:.6f},{y1:.6f}",
                "siteType":      "ST",
                "siteStatus":    "active",
                "hasDataTypeCd": "dv",
            }
        )
        print(f"  tile {i+1}/{len(tiles)}: {len(chunk)} gages found")
        records.append(chunk)
    except Exception as e:
        print(f"  tile {i+1}/{len(tiles)}: no data or error — {e}")

# ── Combine and deduplicate across tile boundaries ────────────────────────────
info = pd.concat(records, ignore_index=True)
info = info.drop_duplicates(subset="site_no")
print(f"\n{len(info)} unique gages found across all tiles")

# ── Spatial filter: keep only gages inside the basin polygon ──────────────────
info_gdf = gpd.GeoDataFrame(info, geometry="geometry", crs="EPSG:4326")
print(f"Found {len(info)} gages")
inside_mask  = info_gdf.geometry.within(ucol_geom)
pps = info_gdf[inside_mask].copy().reset_index(drop=True)
print(f"{len(pps)} gages fall within the basin boundary")

# info is already a GeoDataFrame with point geometry
gages_gdf = pps.set_crs("EPSG:4326")

# Save to file
# gages_gdf.to_file(join(cwd, "ucol_gages.gpkg"), driver="GPKG")
# print(gages_gdf.head())

# find gages with good streamflow data
# Date range of interest
dates = ("1999-10-01", "2026-09-30")

station_ids = pps["site_no"].tolist()

pps.to_file(os.path.join(cwd, r'data\shapefiles\UCOL_basins.shp'))

# # Returns a pandas DataFrame: columns = station IDs, index = dates
# # Units are cubic meters per second (cms)
# flow = NWIS.get_streamflow(station_ids, dates, freq="dv") # this takes time
# print(flow.shape)   # (n_days, n_stations)

# # Station metadata is stored in .attrs
# for sid, meta in flow.attrs.items():
#     print(sid, meta.get("station_nm"), meta.get("dec_lat_va"))

