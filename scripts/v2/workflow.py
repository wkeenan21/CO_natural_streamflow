import geopandas as gpd
import rioxarray
from pathlib import Path
import os
from os.path import join
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.strtree import STRtree
from pynhd import NLDI
import cdsspy
nldi = NLDI()
from pygeohydro import NWIS
nwis = NWIS()
from shapely.geometry import MultiPolygon, Polygon
import re
from pysheds.grid import Grid
from shapely.geometry import shape
from rasterstats import zonal_stats
import rasterio

###############
# 1 download DEM (R script) # don't need, can just grab basins from NLDI
# 2 find gages with good streamflow data in Upper Col basin
# 3 delineate basins # don't need, can just grab basins from NLDI
# 4 download climate reanlysis data
# 5 harmonize it

# parameters
#Date range of interest
dates = ("2003-10-01", "2025-09-30")
cwd = os.getcwd()

def diagnose_gage_quality(df, completion_threshold=0.90):
    """
    Analyzes and prints the data quality profile for all streamgages in the DataFrame
    by checking how many water years meet the completeness threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame where index is DatetimeIndex and columns are gage IDs.
    completion_threshold : float
        The fraction of daily data (0.0 to 1.0) required to consider a water year "good".
    """
    # Ensure the index is datetime format
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # Calculate the Water Year (Oct 1 - Sep 30)
    water_years = df.index.year + (df.index.month >= 10).astype(int)
    
    # Count total expected days per water year based on what's in the index
    days_per_wy = pd.Series(water_years).value_counts()
    total_wy_count = len(days_per_wy)
    
    print("=" * 60)
    print(f"GAGE DATA QUALITY DIAGNOSTIC (Threshold: {completion_threshold*100:.1f}%)")
    print(f"Total Gages Analyzed: {df.shape[1]}")
    print(f"Total Water Years in Period: {total_wy_count} ({water_years.min()} to {water_years.max()})")
    print("=" * 60)
    print(f"{'Gage ID':<15} | {'Good Water Years':<18} | {'Total Water Years':<18} | {'Status':<10}")
    print("-" * 60)
    
    good_gages_count = 0
    
    for gage in df.columns:
        # Count non-null flow values per water year
        wy_counts = df[gage].notna().groupby(water_years).sum()
        
        # Align expected days with the years present
        expected_days = days_per_wy.loc[wy_counts.index]
        
        # Calculate completion percentage
        wy_completion = wy_counts / expected_days
        
        # Count how many water years pass the threshold
        good_years_count = (wy_completion >= completion_threshold).sum()
        
        # Label it for quick scanning
        # If it has 0 good years, flag it; if it's perfectly complete, label it excellent
        if good_years_count == total_wy_count:
            status = "Excellent"
            good_gages_count += 1
        elif good_years_count >= 10:  # Using your previous benchmark as a baseline reference
            status = "Passing"
            good_gages_count += 1
        else:
            status = "Poor"
            
        print(f"{str(gage):<15} | {good_years_count:<18} | {total_wy_count:<18} | {status:<10}")
        
    print("=" * 60)
    print(f"Diagnostic complete. {good_gages_count}/{df.shape[1]} gages have at least 10 quality years.")


def extract_all_snodas(gdf, date_range, snodas_folder, gage_col='gage'):
    """
    Opens each daily CONUS SNODAS file exactly once and extracts SWE sums 
    for all watersheds simultaneously.
    """
    # 1. Get a list of all target dates
    start_date, end_date = date_range
    date_list = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 2. Open just one file to get the exact SNODAS CRS, then reproject your whole GDF
    # SNODAS typically uses an unprojected cylindrical or custom NSIDC grid

    first_date_str = date_list[0].strftime('%Y%m%dd')
    sample_file = os.path.join(snodas_folder, f"SNODAS_SWE_{first_date_str}.tif")
    
    if os.path.exists(sample_file):
        with rasterio.open(sample_file) as src:
            snodas_crs = src.crs
        gdf_reproj = gdf.to_crs(snodas_crs)
    else:
        raise FileNotFoundError(f"Could not find sample SNODAS file: {sample_file}")

    # Dictionary to hold arrays: { date: [sum_gage1, sum_gage2, ...] }
    master_swe_data = {}
    
    print(f"Extracting SNODAS data across {len(date_list)} days for all watersheds...")
    for date_val in date_list:
        date_str = date_val.strftime('%Y%m%dd')
        filename = f"SNODAS_SWE_{date_str}.tif"
        file_path = os.path.join(snodas_folder, filename)
        
        if os.path.exists(file_path):
            # zonal_stats extracts data for ALL polygons in the GDF simultaneously 
            # while the file is held open in a low-level C-buffer.
            stats = zonal_stats(gdf_reproj, file_path, stats="sum", nodata=np.nan)
            # Extract just the raw sum values into a list matching GDF order
            master_swe_data[date_val] = [s['sum'] if s['sum'] is not None else np.nan for s in stats]
        else:
            # Fallback for missing tracking dates
            master_swe_data[date_val] = [np.nan] * len(gdf_reproj)
            
    # Convert to a master lookup dataframe
    # Rows = DatetimeIndex, Columns = Gage IDs
    snodas_master_df = pd.DataFrame(master_swe_data, index=gdf[gage_col]).T
    snodas_master_df.index.name = 'time'
    
    return snodas_master_df

print("SNODAS COLLECTED")


def prepare_dem_inputs(raw_dem_path, output_folder):
    """
    Conditions a raw DEM, computes D8 flow direction and accumulation, 
    and saves the resulting rasters to a folder.
    
    Parameters:
    -----------
    raw_dem_path : str
        Path to the raw input DEM file.
    output_folder : str
        Directory path where the output rasters will be saved.
        
    Returns:
    --------
    tuple : (filled_path, fdir_path, acc_path)
        Paths to the three newly created raster files.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Initialize Pysheds grid and read raw DEM
    grid = Grid.from_raster(raw_dem_path)
    dem = grid.read_raster(raw_dem_path)
    
    # 1. Condition the DEM
    print("Filling pits and depressions...")
    pit_filled = grid.fill_pits(dem)
    dep_filled = grid.fill_depressions(pit_filled)
    filled_dem = grid.resolve_flats(dep_filled)
    
    # 2. Calculate Routing
    print("Calculating D8 flow direction and accumulation...")
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(filled_dem, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)
    
    # 3. Define output filenames
    filled_path = os.path.join(output_folder, "dem_filled.tif")
    fdir_path = os.path.join(output_folder, "flow_direction.tif")
    acc_path = os.path.join(output_folder, "flow_accumulation.tif")
    
    # 4. Export rasters to disk
    print(f"Saving outputs to {output_folder}...")
    grid.to_raster(filled_dem, filled_path)
    grid.to_raster(fdir, fdir_path)
    grid.to_raster(acc, acc_path)
    
    print("Pre-processing complete.")
    return filled_path, fdir_path, acc_path

def delineate_watersheds_preprocessed(fdir_path, acc_path, points_gdf, gage_col='gage', snap_threshold=1000):
    """
    Delineates watersheds using pre-calculated flow direction and accumulation rasters.
    
    Parameters:
    -----------
    fdir_path : str
        Path to the pre-calculated D8 flow direction raster.
    acc_path : str
        Path to the pre-calculated flow accumulation raster.
    points_gdf : gpd.GeoDataFrame
        GeoDataFrame containing streamgage points.
    gage_col : str
        Column name holding the unique gage ID.
    snap_threshold : int
        Minimum accumulation cell count to snap the point to.
        
    Returns:
    --------
    gpd.GeoDataFrame
        A GeoDataFrame containing the polygon watersheds with matching gage IDs.
    """
    # Initialize grids using the flow direction layout as the primary coordinate framework
    grid = Grid.from_raster(fdir_path)
    fdir = grid.read_raster(fdir_path)
    acc = grid.read_raster(acc_path)
    
    # D8 direction mapping structure used during generation
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    
    # Ensure points match the raster coordinate system
    dem_crs = grid.crs
    if points_gdf.crs != dem_crs:
        print(f"Reprojecting points to match raster CRS: {dem_crs}")
        points_gdf = points_gdf.to_crs(dem_crs)
        
    watershed_records = []
    print(f"Delineating {len(points_gdf)} watersheds from pre-processed surfaces...")
    
    for idx, row in points_gdf.iterrows():
        gage_id = row[gage_col]
        geom = row.geometry
        x, y = geom.x, geom.y
        
        try:
            # Snap point to high-accumulation channel flow path
            x_snap, y_snap = grid.snap_to_mask(acc > snap_threshold, (x, y))
            
            # Extract catchment array
            catchment = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')
            catchment = catchment.astype(np.int32)
            
            # Vectorize boundary arrays into shapely geometry shapes
            shapes_generator = grid.polygonize(catchment)
            polygons = [shape(s) for s, val in shapes_generator if val > 0]
            
            if polygons:
                combined_polygon = polygons[0] if len(polygons) == 1 else polygons[0].union(polygons[1:])
                watershed_records.append({
                    gage_col: gage_id,
                    'geometry': combined_polygon
                })
            else:
                print(f"Warning: No polygon generated for gage {gage_id}")
                
        except Exception as e:
            print(f"Failed to delineate gage {gage_id}: {e}")
            
    # Build final GeoDataFrame output
    result_gdf = gpd.GeoDataFrame(watershed_records, crs=dem_crs)
    return result_gdf


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


def filter_gages_by_quality(df, min_years=2, completion_threshold=0.50):
    """
    Filters out streamgages (columns) that do not have enough complete water years.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame where index is DatetimeIndex and columns are gage IDs.
    min_years : int
        Minimum number of valid water years required to keep a gage.
    completion_threshold : float
        The fraction of daily data (0.0 to 1.0) required to consider a water year "complete".
        
    Returns:
    --------
    pd.DataFrame
        The filtered DataFrame containing only the qualifying gages.
    """
    # Ensure the index is datetime format
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # Calculate the Water Year: If month is Oct, Nov, Dec, it belongs to next year's WY
    water_years = df.index.year + (df.index.month >= 10).astype(int)
    
    # Track which gages to keep
    valid_gages = []
    
    # Count expected days per water year to handle leap years accurately
    # Grouping the index by water year to see how many days are actually in each WY
    days_per_wy = pd.Series(water_years).value_counts()

    for gage in df.columns:
        # Group by water year and count non-null daily flow values
        # (Using .notna() counts actual data points, ignoring NaNs)
        wy_counts = df[gage].notna().groupby(water_years).sum()
        
        # Align expected days with the years present in this gage's data
        expected_days = days_per_wy.loc[wy_counts.index]
        
        # Calculate completion percentage for each water year
        wy_completion = wy_counts / expected_days
        
        # Count how many water years meet or exceed the threshold
        good_years_count = (wy_completion >= completion_threshold).sum()
        
        if good_years_count >= min_years:
            valid_gages.append(gage)
            
    # Calculate how many were removed
    original_count = df.shape[1]
    filtered_df = df[valid_gages]
    removed_count = original_count - filtered_df.shape[1]
    
    print(f"Removed {removed_count} out of {original_count} gages due to poor data quality.")
    print(f"Retained {filtered_df.shape[1]} gages.")
    
    return filtered_df


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


def fix_gage_id(id_val):
    id_str = str(id_val).strip()
    # Only pad if it is a 7-digit numeric string
    if len(id_str) == 7 and id_str.isdigit():
        return id_str.zfill(8)

    return id_str
################## STEP 1 #####################

# This step searchs the Ucol watershed for USGS gages and finds point locations


# query basins
ucol = gpd.read_file(join(cwd, r"data\shapefiles\UCOL\UCOL.shp")).set_crs('4326')
ucol_geom = ucol.union_all()  # single shapely geometry for spatial filtering
if isinstance(ucol_geom, MultiPolygon):
    polygon = ucol_geom.geoms[0]
else:
    polygon = ucol_geom  # Fallback if it's already a Polygon

ucol = gpd.GeoDataFrame(geometry=[polygon], crs=4326)


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
info = info.rename(columns = {'station_nm':'name', 'site_no':'gage'})

# ── Spatial filter: keep only gages inside the basin polygon ──────────────────
info_gdf = gpd.GeoDataFrame(info, geometry="geometry", crs="EPSG:4326")
print(f"Found {len(info)} gages")
inside_mask  = info_gdf.geometry.within(ucol_geom)
pps = info_gdf[inside_mask].copy().reset_index(drop=True)
print(f"{len(pps)} gages fall within the basin boundary")

# info is already a GeoDataFrame with point geometry
gages_gdf = pps.set_crs("EPSG:4326")
gage_ids = gages_gdf['gage'].to_list()

#Save to file
gages_gdf.to_file(join(cwd, "ucol_gages.gpkg"), driver="GPKG")
print(gages_gdf.head())


flow = NWIS.get_streamflow(gage_ids, dates, freq="dv") # this takes time
print(flow.shape)   # (n_days, n_stations)
 # filter them
flow_f = filter_gages_by_quality(flow)

pattern = r"USGS-(\d+)"
gage_ids = [
    re.search(pattern, col).group(1) if re.search(pattern, col) else col 
    for col in flow_f.columns
]

###################### STEP 2: Delineate watersheds from NLDI #####################
nldi  = NLDI()
basins = []
no_work = []
for id in gage_ids:
    try:
        basin = nldi.get_basins(id) # watershed polygons
        basins.append(basin)
    except:
        no_work.append(id)
        continue
# the polygons
basins = pd.concat(basins)

############# REMOVE NESTED BASINS ##############

# just headwaters
basins.geometry = basins.geometry.to_crs(3857)
basins['gage'] = basins.index
headwaters = remove_nested_basins(basins)
headwaters.explore()
headwaters.to_file(os.path.join(cwd, r'data\shapefiles\UCOL_headwaters.shp'))

flow_f2 = flow[headwaters.gage.to_list()]

# TODO make this delineation work

manual_delineate_list = gages_gdf[gages_gdf['gage'].isin(no_work)]
dem_path = os.path.join(cwd, r'data\terrain\dem_ucol_30m.tif')
terrain_folder = os.path.join(cwd, r'data\terrain\dem_ucol')

# Run the raw data once and save the states
filled_path, fdir_path, acc_path = prepare_dem_inputs(
    raw_dem_path=dem_path, 
    output_folder=terrain_folder
)

# Step 2: Pass those paths into your delineation routine whenever needed
watershed_polygons_gdf = delineate_watersheds_preprocessed(
    fdir_path=fdir_path,
    acc_path=acc_path,
    points_gdf=manual_delineate_list,
    gage_col='gage'
)

#### NOTE TO SELF ####
# Seems like most of the no work list is canals or gages that only measure water quality (no cfs)

basins.geometry = basins.geometry.to_crs(4326)

import os
import geopandas as gpd
import pandas as pd
import pygridmet as gridmet
import rioxarray
from shapely.geometry import mapping

basins['gage'] = basins.index

gdf = basins
date_range = dates
gage_col='gage'
    
# Variables requested from GRIDMET service
variables = ['pr', 'srad', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd', 'pet']

# Define aggregation types per variable
mean_vars = ['srad', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd',]
sum_vars = ['pet', 'pr']

# 1. Run the SNODAS extraction ONCE up front
snodas_folder = r"N:\Research\Kampf\Private\KeenanW\SNODAS"
snodas_lookup = extract_all_snodas(gdf, date_range, snodas_folder, gage_col)

# Loop over each individual watershed
for idx, row in gdf.iterrows():
    gage_id = str(row[gage_col])
    print(f"Processing gage: {gage_id}...")
    
    # Isolate the single watershed geometry and wrap it into a standalone GeoDataFrame
    single_gdf = gpd.GeoDataFrame([row], crs=gdf.crs)

    # 1. Fetch gridded
    ds = gridmet.get_bygeom(
        geometry=single_gdf.geometry.iloc[0],
        dates=date_range,
        crs=single_gdf.crs,
        variables=variables,
    )
    
    # 2. Write spatial dimensions to ensure rioxarray can parse coordinates
    if "rio" not in ds.dims:
        ds = ds.rio.write_crs(ds.crs)
        
    # 3. Clip the bounding box netCDF precisely down to the polygon edge mask
    # geometry selection ensures we don't include exterior square cells
    clipped_ds = ds.rio.clip(
        single_gdf.geometry.apply(mapping), 
        crs=single_gdf.crs, 
        all_touched=True
    )
    
    daily_records = []
    
    # 4. Spatially aggregate over grid coordinates (x and y dimensions)
    # Means across pixels
    ds_mean = clipped_ds[mean_vars].mean(dim=["lon", "lat"], skipna=True)
    # Sums across pixels
    ds_sum = clipped_ds[sum_vars].sum(dim=["lon", "lat"], skipna=True)
    
    # Combine aggregated structures into a single pandas DataFrame
    df_mean = ds_mean.to_dataframe().reset_index()
    df_sum = ds_sum.to_dataframe().reset_index()
    
    # Merge mean and sum frames along the 'time' index
    gr_df = pd.merge(df_mean, df_sum, on='time')
    
    # Clean up headers to show aggregation style explicitly (Optional)
    rename_dict = {v: f"{v}_mean" for v in mean_vars}
    rename_dict.update({v: f"{v}_sum" for v in sum_vars})
    final_df = gr_df.rename(columns=rename_dict)

    # Set the time index
    final_df = final_df.set_index('time')
    
    # EXTRACTION FIX: Instead of opening files, instantly pull the column from memory!
    # Pull the pre-calculated time series matching this specific gage ID
    gage_swe_series = snodas_lookup[row[gage_col]] 
    
    # Map the series back to the main dataframe
    gr_df = gr_df.merge(gage_swe_series, left_on='time', right_index=True, how='left')
    gr_df = gr_df.rename(columns={row[gage_col]: 'swe_sum'})



















