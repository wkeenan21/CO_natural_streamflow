import geopandas as gpd
from pathlib import Path
import os
from os.path import join
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.strtree import STRtree
from pynhd import NLDI



###############
# 1 download DEM (R script)
# 2 find gages with good streamflow data in Upper Col basin
# 3 delineate basins
# 4 download climate reanlysis data
# 5 harmonize it

cwd = os.getcwd()

pps = gpd.read_file(os.path.join(cwd, r'data\shapefiles\UCOL_basins.shp'))
station_ids = pps['site_no'].to_list()
###################### STEP 2: Delineate watersheds from NLDI #####################
nldi  = NLDI()
basins = []
for id in station_ids:
    try:
        basin = nldi.get_basins(id) # watershed polygons
        basins.append(basin)
    except:
        continue
basins = pd.concat(basins)

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
