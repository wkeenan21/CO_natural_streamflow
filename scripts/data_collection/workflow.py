import geopandas as gpd
import rioxarray
from pathlib import Path
import os
from os.path import join
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pygeohydro import NWIS
import numpy as np
from shapely.strtree import STRtree
from pynhd import NLDI
import pygeoogc

# Clear the web request cache
pygeoogc.clear_cache()

# Now try running your code again
from pynhd import NLDI
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
nwis    = NWIS()
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

# Returns a pandas DataFrame: columns = station IDs, index = dates
# Units are cubic meters per second (cms)
flow = NWIS.get_streamflow(station_ids, dates, freq="dv") # this takes time
print(flow.shape)   # (n_days, n_stations)

# Station metadata is stored in .attrs
for sid, meta in flow.attrs.items():
    print(sid, meta.get("station_nm"), meta.get("dec_lat_va"))

###################### STEP 2: Delineate watersheds from NLDI
from pynhd import NLDI, WaterData, NHDPlusHR
import pynhd as nhd
nldi  = NLDI()
basin = nldi.get_basins(feature_ids=station_ids) # watershed polygons
basin['gage_used'] = basin.index.str[-8:] # make gage column

# # merge with CSU_Flow25
flow25 = gpd.read_file(os.path.join(cwd, r"data\CSU_Flow25\watersheds_shapefile_20250624.shp"))
flow25.geometry = flow25.geometry.to_crs(4326)

mask = flow25.geometry.within(ucol_geom)
flow25 = flow25[mask].copy().reset_index(drop=True)
# fix gage Ids
def fix_gage_id(id_val):
    id_str = str(id_val).strip()
    # Only pad if it is a 7-digit numeric string
    if len(id_str) == 7 and id_str.isdigit():
        return id_str.zfill(8)

    return id_str

# Apply the logic to the gage_used column
flow25['gage_used'] = flow25['gage_used'].apply(fix_gage_id)
flow25['gage_used'] = np.where(flow25['gage_used'].str.contains('E+', regex=False), flow25['usgs_id'], flow25['gage_used']) # the pesky long ones

basin = pd.concat([basin, flow25[['gage_used', 'geometry']].drop_duplicates(subset='gage_used')], ignore_index=True)
basin = basin.drop_duplicates(subset='gage_used')

############# REMOVE NESTED BASINS
def remove_nested_basins(gdf, area_col=None):
    gdf = gdf.copy().reset_index(drop=True)

    rep_points = gdf.geometry.representative_point()

    is_geographic = gdf.crs.is_geographic if gdf.crs else True
    buffer_dist   = 0.001 if is_geographic else 100.0
    buffered      = gdf.geometry.buffer(-buffer_dist)

    tree = STRtree(buffered.values)

    # A basin is a PARENT (downstream, non-headwater) if ANY other basin's
    # representative point falls inside it. Flag those for removal.
    parent_idx = set()

    for i, pt in enumerate(rep_points):
        candidates = tree.query(pt)
        for j in candidates:
            if j == i:
                continue
            if buffered.iloc[j].contains(pt):
                # basin j contains basin i's point → j is a parent, not a headwater
                parent_idx.add(j)

    headwater_mask = ~gdf.index.isin(parent_idx)
    headwaters     = gdf[headwater_mask].copy().reset_index(drop=True)

    print(f"Total basins       : {len(gdf)}")
    print(f"Parents (removed)  : {len(parent_idx)}")
    print(f"Headwaters kept    : {len(headwaters)}")

    return headwaters

# just headwaters
basins = remove_nested_basins(basin)
#headwaters.explore()

