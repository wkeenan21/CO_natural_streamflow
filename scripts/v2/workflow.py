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
nldi = NLDI()
from pygeohydro import NWIS
nwis = NWIS()
from shapely.geometry import MultiPolygon, Polygon
import re
from pysheds.grid import Grid
from shapely.geometry import shape
from rasterstats import zonal_stats
import rasterio
os.environ["API_USGS_PAT"] = "ePZbH4mtoakm2VYSZDE78clCxGDmKxRWd7nWYzpt"
from dataretrieval import waterdata
import pyarrow

###############
# 1 download DEM (R script) # don't need, can just grab basins from NLDI
# 2 find gages with good streamflow data in Upper Col basin
# 3 delineate basins # don't need, can just grab basins from NLDI
# 4 download climate reanlysis data
# 5 harmonize it

cwd = os.getcwd()
ncwd = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow'
appcwd = os.path.join(cwd, r'shiny-app\ucol_natural')


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


def extract_all_snodas(gdf, date_range, snodas_folder, export_folder, gage_col='gage'):
    """
    Opens daily CONUS SNODAS files and extracts SWE sums for all watersheds.
    Saves a separate CSV file for each water year (Oct 1 - Sep 30) in the date range.
    """
    start_date, end_date = date_range
    date_list = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 1. Get the exact SNODAS CRS from the first file and reproject the GDF
    first_date_str = date_list[0].strftime('%Y%m%d')
    sample_file = os.path.join(snodas_folder, f"SNODAS_SWE_{first_date_str}.tif")
    
    if os.path.exists(sample_file):
        with rasterio.open(sample_file) as src:
            snodas_crs = src.crs
        gdf_reproj = gdf.to_crs(snodas_crs)
    else:
        raise FileNotFoundError(f"Could not find sample SNODAS file: {sample_file}")

    if not os.path.exists(export_folder):
        raise FileNotFoundError(f'could not find {export_folder}')

    # --- WATER YEAR ADJUSTMENT ---
    # Calculate initial water year: if month >= 10, it's the next calendar year
    first_date = date_list[0]
    current_water_year = first_date.year + 1 if first_date.month >= 10 else first_date.year
    year_swe_data = {}
    
    print(f"Extracting SNODAS data across {len(date_list)} days...")
    
    for i, date_val in enumerate(date_list):

        # Calculate water year for the current loop date
        this_water_year = date_val.year + 1 if date_val.month >= 10 else date_val.year
        
        # If we've hit a new water year, export the accumulated data from the previous water year
        if this_water_year != current_water_year:
            # Note: You may want to rename '_export_yearly_csv' or pass a prefix like f"WY{current_water_year}" 
            # if your internal helper function appends the year to the filename.
            _export_yearly_csv(year_swe_data, f"WY{current_water_year}", gdf[gage_col], export_folder)
            
            # Reset for the new water year
            current_water_year = this_water_year
            year_swe_data = {}

        print(date_val.strftime('%Y-%m-%d'))
        date_str = date_val.strftime('%Y%m%d')
        filename = f"SNODAS_SWE_{date_str}.tif"
        file_path = os.path.join(snodas_folder, filename)
        
        if os.path.exists(file_path):
            stats = zonal_stats(gdf_reproj, file_path, stats="sum", nodata=-9999)
            year_swe_data[date_val] = [s['sum'] if s['sum'] is not None else np.nan for s in stats]
        else:
            print(f'missing {file_path} filling na')
            year_swe_data[date_val] = [np.nan] * len(gdf_reproj)

    # Export the final water year's data after the loop finishes
    if year_swe_data:
        _export_yearly_csv(year_swe_data, f"WY{current_water_year}", gdf[gage_col], export_folder)

    print("SNODAS PROCESSING COMPLETE. ALL FILES EXPORTED.")


def _export_yearly_csv(year_data, year, gage_ids, export_folder):
    """Helper function to format and save the yearly data to CSV."""
    print(f"--- Exporting CSV for year {year} ---")
    df = pd.DataFrame(year_data, index=gage_ids).T
    df.index.name = 'date'
    
    export_path = os.path.join(export_folder, f"SNODAS_SWE_{year}.csv")
    df.to_csv(export_path)
    print(f"Saved: {export_path}")


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
    #result_gdf = gpd.GeoDataFrame(watershed_records, crs=dem_crs)
    return watershed_records


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

    if 'USGS-' in id_str:
        id_str = id_str.replace('USGS-', '')

    return id_str

def add_usgs_prefix(data):
    """
    Adds 'USGS-' to the beginning of a single string or every string in a list.
    """
    if isinstance(data, str):
        return f"USGS-{data}"
    elif isinstance(data, list):
        return [f"USGS-{item}" for item in data]
    else:
        raise TypeError("Input must be a string or a list of strings")

from shapely.geometry import Polygon, MultiPolygon

def df_to_geodataframe(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    crs: str = "EPSG:4326",
    drop_latlon: bool = False,
) -> gpd.GeoDataFrame:
    """
    Convert a pandas DataFrame with latitude/longitude columns to a GeoDataFrame of points.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing coordinate columns.
    lat_col : str
        Name of the latitude column. Default is 'lat'.
    lon_col : str
        Name of the longitude column. Default is 'lon'.
    crs : str
        Coordinate reference system for the output GeoDataFrame. Default is 'EPSG:4326' (WGS84).
    drop_latlon : bool
        If True, drop the original lat/lon columns from the output. Default is False.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with a Point geometry column.

    Raises
    ------
    KeyError
        If lat_col or lon_col are not found in the DataFrame.
    ValueError
        If lat/lon columns contain non-numeric or all-null values.
    """
    missing = [c for c in [lat_col, lon_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not found in DataFrame: {missing}")

    for col in [lat_col, lon_col]:
        if df[col].isna().all():
            raise ValueError(f"Column '{col}' contains all null values.")

    geometry = gpd.points_from_xy(df[lon_col].astype(float), df[lat_col].astype(float))

    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)

    if drop_latlon:
        gdf = gdf.drop(columns=[lat_col, lon_col])

    return gdf


def remove_polygon_holes(geometry):
    """
    Removes holes from a Shapely Polygon or MultiPolygon.
    
    Parameters:
    geometry (Polygon or MultiPolygon): The input geometry with holes.
    
    Returns:
    Polygon or MultiPolygon: The filled geometry without holes.
    """
    if isinstance(geometry, Polygon):
        # Create a new polygon using only the exterior linear ring
        return Polygon(geometry.exterior)
        
    elif isinstance(geometry, MultiPolygon):
        # Iterate through each sub-polygon, fix it, and bundle back into a MultiPolygon
        filled_polygons = [Polygon(poly.exterior) for poly in geometry.geoms]
        return MultiPolygon(filled_polygons)
        
    else:
        # Return the geometry untouched if it's a Point, LineString, etc.
        return geometry

################## STEP 1 #####################

# This step searchs the Ucol watershed for USGS gages and finds point locations
ucol_gage = '09379900'
ucol = nldi.get_basins(ucol_gage)['geometry'].iloc[0]
ucol_geom = remove_polygon_holes(ucol)
ucol = gpd.GeoDataFrame(geometry=[ucol_geom], crs=4326)
ucol.to_file(os.path.join(ncwd, r'data\shapefiles\UCOL.parquet'))
ucol.to_file(os.path.join(appcwd, r'spatial_data\UCOL.parquet'))

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
inside_mask = info_gdf.geometry.within(ucol_geom)
pps = info_gdf[inside_mask].copy().reset_index(drop=True)
print(f"{len(pps)} gages fall within the basin boundary")

# info is already a GeoDataFrame with point geometry
gages_gdf = pps.set_crs("EPSG:4326")
gage_ids = gages_gdf['gage'].to_list()
gages_gdf.to_file(os.path.join(ncwd, r'data/shapefiles/all_UCOL_gages.gpkg'))
gages_gdf.to_parquet(os.path.join(appcwd, r'spatial_data/all_UCOL_gages.parquet'))

###################### STEP 2: Delineate watersheds from NLDI #####################
nldi  = NLDI()
basins_list = []
no_work = []
for id in gage_ids:
    try:
        basin = nldi.get_basins(id) # watershed polygons
        basins_list.append(basin)
    except:
        no_work.append(id)
        continue
# the polygons
basins = pd.concat(basins_list)

# TODO get a bigger DEM for all of UCOL :( and delineate some more


# manually delineate the ones that NLDI didn't get
mdl = gages_gdf[gages_gdf['gage'].isin(no_work)]
print(len(mdl))
man_delin = gpd.read_file(os.path.join(ncwd, r'data/shapefiles/man_delin.shp'))

mdl = mdl[
    ~mdl['name'].str.lower().str.contains('canal') & 
    ~mdl['name'].str.lower().str.contains('tunnel')
]
print(len(mdl))
# check if we already did these
mdl = mdl[mdl.gage.isin(man_delin.gage)]
print(len(mdl))

dem_path = os.path.join(ncwd, r'data\terrain\dem_ucol_south\output_USGS30m.tif')
terrain_folder = os.path.join(ncwd, r'data\terrain\dem_ucol_south')

# Run the raw data once and save the states
filled_path = os.path.join(terrain_folder, r'dem_filled.tif')
acc_path = os.path.join(terrain_folder, r'flow_accumulation.tif')
fdir_path = os.path.join(terrain_folder, r'flow_direction.tif')

if not os.path.exists(acc_path):
    filled_path, fdir_path, acc_path = prepare_dem_inputs(
        raw_dem_path=dem_path, 
        output_folder=terrain_folder
    )

# Step 2: Pass those paths into your delineation routine whenever needed
watershed_polygons_gdf = delineate_watersheds_preprocessed(
    fdir_path=fdir_path,
    acc_path=acc_path,
    points_gdf=mdl,
    gage_col='gage'
)

man_delin = gpd.GeoDataFrame().from_dict(watershed_polygons_gdf)

# drop the multipolygon
man_delin['geometry'] = man_delin['geometry'].apply(
    lambda x: x[0] if isinstance(x, np.ndarray) else x
)

# Now set the geometry
man_delin = man_delin.set_geometry('geometry')
man_delin = man_delin.set_crs(4326)
#man_delin.to_file(os.path.join(ncwd, r'data/shapefiles/man_delin.shp'))
#man_delin = gpd.read_file(os.path.join(ncwd, r'data/shapefiles/man_delin.shp'))

basins.geometry = basins.geometry.to_crs(4326)
basins['gage'] = basins.index
basins['gage'] = basins['gage'].apply(fix_gage_id)
basins.reset_index(drop=True, inplace=True)
basins = pd.concat([basins, man_delin])
basins = basins[basins.gage.isin(gages_gdf.gage)]

# save
basins.to_file(os.path.join(ncwd, r'data/shapefiles/all_UCOL_basins.gpkg'))
basins.to_parquet(os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet'))

#############################################################################
# GET STREAMFLOW AND MERGE DIVERSION DATA
#############################################################################
from collections import defaultdict

# skip previous steps by reading these
gages_gdf = gpd.read_file(os.path.join(ncwd, r'data/shapefiles/ucol_gages.gpkg'))
basins = gpd.read_file(os.path.join(ncwd, r'data/shapefiles/all_UCOL_basins.gpkg'))
# make dict for names
name_dict = gages_gdf.set_index('gage')['name'].to_dict()
basins['name'] = basins['gage'].map(name_dict)
# Get all streamflow
date_range = ("2003-10-01", "2025-09-30")
flow = NWIS.get_streamflow(gages_gdf['gage'].to_list(), date_range, freq="dv")

# fix timeseries index
flow.index = flow.index.normalize().tz_localize(None)

# rename columns
rename_dict = {}
for usgsgage in flow.columns:
    gage = fix_gage_id(usgsgage)
    rename_dict[usgsgage] = gage
flow = flow.rename(columns=rename_dict)

# just doesn't work for some?
nwis_fail = gages_gdf[~gages_gdf['gage'].isin(flow.columns)]
nwis_fail = add_usgs_prefix(nwis_fail['gage'].to_list())

df, metadata = waterdata.get_daily(
    monitoring_location_id=nwis_fail,
    parameter_code='00060',
    time=f'{date_range[0]}/{date_range[1]}'
)

# get basin area
basins.geometry = basins.geometry.to_crs(5070)
basins['area_m2'] = basins.area
# add all the ones that didn't work
for usgsgage in df['monitoring_location_id'].unique():
    gage = fix_gage_id(usgsgage)
    sub_df = df[df['monitoring_location_id']==usgsgage][['time', 'value']]
    sub_df = sub_df.rename(columns={'value':gage})
    sub_df = sub_df.set_index('time')
    flow[gage] = sub_df[gage] * 0.0283168

# DIVERSIONS
# COLUMN = siteID
dvrs = gpd.read_file(os.path.join(ncwd, r"data\diversion\input\ucrb_diversion_master_table.csv"))

# this code clarifies intrabasin transfers. If the intrabasin tranfer delivers to another subbasin within UCOL,
# this should be considered a transbasin import for that subbasin. For a large sub-basin that has a intrabasin diversion
# that delivers somewhere within the basin, the consumptive use of the intrabasin diversion will cancel out with
# the transbasin import
dvrs_intra = dvrs[dvrs['siteUse']=='intrabasin'].copy()
dvrs_intra['siteUse'] = 'transbasin'
dvrs_intra['origin_decLat'] = dvrs_intra['decLat']
dvrs_intra['origin_decLong'] = dvrs_intra['decLong']
dvrs_intra['decLat'] = dvrs_intra['dest_decLat']
dvrs_intra['decLong'] = dvrs_intra['dest_decLong']

dvrs_intra['siteID'] = dvrs_intra['siteID'].str.replace('intrabasin', 'transbasin')
# add rows for the transbasin
dvrs = pd.concat([dvrs, dvrs_intra])
# need to rebuild the geometry after this
dvrs = df_to_geodataframe(dvrs, lat_col='decLat', lon_col='decLong')
#dvrs.to_file(os.path.join(ncwd, r"data\shapefiles\ucrb_diversion_master_table.gpkg"))

# WATERSHEDS
# gageID column = gage
# WIDE CSV FOR DIVERSION DATA IN CFS
# COLUMN HEADERS = siteID
dvrsFlow = pd.read_csv(os.path.join(ncwd, "data\diversion\processed\processed_data\combined_diversion_records_filtered_filled_cfs_fill_years.csv"))
dvrsFlow = dvrsFlow.rename(columns={'Date':'date'})
dvrsFlow['date'] = pd.to_datetime(dvrsFlow['date'])
# merge with 2022 to 2025
dvrsFlow2 = pd.read_csv(os.path.join(ncwd, "data\diversion\will_processed\combined_diversion_records_filtered_filled_cfs_fill_years.csv"))
dvrsFlow2 = dvrsFlow2.rename(columns={'datetime':'date'})
dvrsFlow2['date'] = pd.to_datetime(dvrsFlow2['date'])
dvrsFlow = pd.concat([dvrsFlow, dvrsFlow2])

# Ensure date columns are datetime objects for proper merging
dvrsFlow['date'] = pd.to_datetime(dvrsFlow['date'])
dvrsFlow = dvrsFlow.set_index('date')

# dvrsFlow doesn't have any "transbasin" flags since we just made them up
for col in dvrsFlow.columns:
    if 'intrabasin' in col:
        trans_dup = col.replace('intrabasin', 'transbasin')
        dvrsFlow[trans_dup] = dvrsFlow[col]

# OUT DIRECTORY FOR STREAMFLOW WITH DIVERSIONS
wdvrsDir = os.path.join(ncwd, r'data\timeseries\selected_w_diversion') # has everything
appDir = os.path.join(appcwd, r'timeseries') # does not have each individual diversion

# 1. Spatial Join: Find which diversions are in which watersheds
# 'inner' join keeps only points that fall inside a polygon
# 'within' ensures the point is geometrically inside the watershed boundary
basins.geometry = basins.geometry.to_crs(4326)
joined = gpd.sjoin(dvrs, basins, how="inner", predicate="within")

# send to csvs
for gage in flow.columns:

    df = flow[[gage]]
    df = df.rename(columns={gage:'Q_cms'})
    df['Q_cfs'] = df['Q_cms'] * 35.3147
    # add area and mmd
    area = basins[basins['gage']==gage]['area_m2'].iloc[0]
    df['Q_mmd'] = (df['Q_cms'] * 86400000) / area
    df['area_m2'] = area
    # make sure it's got every day
    df = df.asfreq('D')
    # add name and gage as columns
    df['name'] = name_dict[gage]
    df['gage'] = gage
    
    # Identify siteIDs for diversions located in this specific watershed
    target_diversions = joined[joined['gage'] == gage]['siteID'].unique()
    print(gage, f'diversions: {len(target_diversions)}')

    # loop through the diversions
    if len(target_diversions) > 0:

        # 2. Group the full column names by their base div_id
        from collections import defaultdict
        id_groups = defaultdict(list)

        for col in target_diversions:
            # Extracts 'div_0001' from 'div_0001_intrabasin'
            base_id = "_".join(col.split("_")[:2]) 
            id_groups[base_id].append(col)

        # 3. Identify base_ids that contain BOTH 'intrabasin' and 'transbasin'
        ids_to_remove = set()
        for base_id, cols in id_groups.items():
            # Check if any column in this group ends with or contains the specific types
            has_intra = any('intrabasin' in c for c in cols)
            has_trans = any('transbasin' in c for c in cols)
            
            if has_intra and has_trans:
                ids_to_remove.add(base_id)

        # 4. Filter them out of your final valid_cols list
        valid_cols = [
            col for col in target_diversions 
            if "_".join(col.split("_")[:2]) not in ids_to_remove
        ]
            
        if valid_cols:
            # 1. Extract the raw data
            subset_dvrs = dvrsFlow[valid_cols].copy()
            # 2. Convert raw diversions to Consumptive Use (CU)
            useDict = {
                'irrigation': -0.6, 'municipal': -0.3, 'interbasin': -1, 
                'industrial': -1, 'hydropower': 0, 'intrabasin': -1, 'transbasin': 1
            }

            # INTERBASIN = all water leaves UCOL
            # INTRABASIN = water may or may not leave sub-basin. Does not leave UCOL.
            # TRANSBASIN = water imported from a different sub-basin within UCOL.
            
            # do it for each unit
            units = ['cfs', 'cms', 'mmd']
            # aggregate diversions by type
            cu_types = useDict.keys()
            for unit in units:
                # Create a temporary list to hold the names of the new CU columns
                cu_cols = []
                for col in valid_cols:
                    # Determine the multiplier by checking the end of the siteID string
                    multiplier = 0 # Default if no match is found
                    for usage, val in useDict.items():
                        if col.lower().endswith(usage):
                            multiplier = val
                            break

                    mmd_scale = 2446575.5461 / area # goes from cfs to mmd
                    unit_multiplier = {'cfs':1, 'cms':0.0283168, 'mmd':mmd_scale}
                    # Calculate CU for this specific diversion
                    cu_col_name = f"{col}_CU_{unit}"
                    subset_dvrs[cu_col_name] = subset_dvrs[col] * multiplier * unit_multiplier[unit]
                    cu_cols.append(cu_col_name)

                for cu_type in cu_types:
                    matched_cols = [col for col in cu_cols if f'_{cu_type}_CU_{unit}' in col]
                    if matched_cols:
                        subset_dvrs[f'{cu_type}_{unit}'] = subset_dvrs[matched_cols].sum(axis=1)
                    else:
                        subset_dvrs[f'{cu_type}_{unit}'] = 0.0
                
                # 3. Aggregate: Sum all CU columns to get the total impact on the watershed
                subset_dvrs[f'Q_CU_{unit}'] = subset_dvrs[cu_cols].sum(axis=1)
        
        # 4. merge
        combined_df = pd.merge(
            df, 
            subset_dvrs, 
            right_index=True,
            left_index=True, 
            how='inner'
        )

        # 5. aggregate by type
            # Aggregate columns matching each diversion type
        for unit in units:
            combined_df[f'Q_NAT_{unit}'] = combined_df[f'Q_{unit}'] - combined_df[f'Q_CU_{unit}']

    else:
        # If no diversions found, we still save the original flow (or skip)
        combined_df = df
        
    # 4. Save the new CSV
    if len(combined_df) < 1:
        raise Exception('merge failed: df has no rows')

    out_path = os.path.join(wdvrsDir, f"{gage}.csv")
    combined_df.to_csv(out_path, index_label='date')

    # make a smaller df without all the diversions
    cu_types2 = []
    for unit in units:
        for cu_type in cu_types:
            cu_types2.append(f'{cu_type}_{unit}')

    columns = [col for col in combined_df.columns if 'Q' in col or col in cu_types2]
    small_df = combined_df[columns]
    out_path_small = os.path.join(appDir, f"{gage}.csv")
    small_df.to_csv(out_path_small, index_label='date')

    print(f"Processed gage {gage}: Added {len(target_diversions)} diversion columns.")

# Now we need to select basins for training, testing, and implementation
from scipy.optimize import milp, LinearConstraint, Bounds

def select_max_water_years_global(basins_gdf, flow_df, gage_col='gage'):
    """
    Selects a non-nested subset of basins within the Upper Colorado River Basin
    using True Global Optimization (ILP) to maximize total basin-water-years.
    """
    num_basins = len(basins_gdf)
    # -------------------------------------------------------------------------
    # 2. Calculate Water Year Completeness Weights
    # -------------------------------------------------------------------------
    water_years = flow_df.index.to_series().apply(lambda d: d.year + 1 if d.month >= 10 else d.year)
    
    good_years_dict = {}
    for col in flow_df.columns:
        if col in basins_gdf[gage_col].values:
            yearly_complete = flow_df[col].groupby(water_years).apply(lambda x: x.notna().mean() >= 0.70) # Using your 0.80 threshold
            good_years_dict[col] = yearly_complete.sum()

    basins_gdf['good_wy'] = basins_gdf[gage_col].map(good_years_dict).fillna(0)

    # -------------------------------------------------------------------------
    # 3. Build Conflict Matrix (Nesting Constraints)
    # -------------------------------------------------------------------------
    print("Analyzing spatial relationships to build global constraints...")
    
    # We need to construct an inequality matrix A where each row represents a conflict.
    # If Basin i and Basin j intersect, we append a constraint: x_i + x_j <= 1
    # This prevents the solver from selecting both.
    constraints_list = []
    
    # Define a minimum overlap area threshold in square meters.
    # For a 100m misalignment along a typical boundary, we can look at the total overlap area.
    # Alternatively, you can check if the overlap constitutes more than e.g., 1% of the smaller basin.
    AREA_CUSHION_M2 = 100 * 100  # 10,000 m² (equivalent to a 100m x 100m grid cell)

    basins_list = basins_gdf.to_dict('records')
    for i in range(num_basins):
        poly_i = basins_list[i]['geometry']
        
        for j in range(i + 1, num_basins):
            poly_j = basins_list[j]['geometry']
            
            # First do a fast check: do they even touch/overlap?
            if poly_i.intersects(poly_j):
                # Calculate the exact geometric intersection polygon
                intersection_poly = poly_i.intersection(poly_j)
                
                # Only flag as a conflict if the intersection area exceeds our cushion
                # (Note: Your GeoDataFrame must be in a metric projected CRS like UTM or Albers)
                if intersection_poly.area > AREA_CUSHION_M2:
                    
                    # Optional secondary check: Ensure it's not just a long, thin sliver 
                    # by checking if it represents more than 1% of either basin's total area.
                    min_basin_area = min(poly_i.area, poly_j.area)
                    if intersection_poly.area / min_basin_area > 0.01:
                        
                        row = np.zeros(num_basins)
                        row[i] = 1
                        row[j] = 1
                        constraints_list.append(row)

    # -------------------------------------------------------------------------
    # 4. Set Up and Solve the Integer Linear Program (ILP)
    # -------------------------------------------------------------------------
    print("Solving Global Optimization Problem...")
    
    # The solver *minimizes* c^T * x. To maximize, we invert the weights (negative values)
    c = -basins_gdf['good_wy'].values
    
    # Define bounds: each basin variable x must be 0 or 1 (Binary integer)
    bounds = Bounds(0, 1)
    integrality = np.ones(num_basins) # 1 means integer variable
    
    if len(constraints_list) > 0:
        A = np.array(constraints_list)
        # For every conflict row, the sum of selections must be less than or equal to 1
        ub = np.ones(A.shape[0])
        lb = np.zeros(A.shape[0]) # can be 0 or 1
        constraints = LinearConstraint(A, lb, ub)
    else:
        constraints = []

    # Run the Mixed-Integer Linear Programming solver
    res = milp(c=c, bounds=bounds, constraints=constraints, integrality=integrality)
    
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
        
    # Extracted selections (rounded cleanly to 0 or 1)
    selected_indices = np.where(np.round(res.x) == 1)[0]
    selected_gages = basins_gdf.iloc[selected_indices][gage_col].tolist()

    # -------------------------------------------------------------------------
    # 5. Map results back to the original DataFrame
    # -------------------------------------------------------------------------
    basins_gdf['model_category'] = 'no-model (nested or excluded)'
    basins_gdf.loc[basins_gdf[gage_col].isin(selected_gages), 'model_category'] = 'train_test'
    basins_gdf['good_water_years'] = basins_gdf[gage_col].map(good_years_dict).fillna(0)
    
    total_selected_years = basins_gdf[basins_gdf['model_category'] == 'train_test']['good_water_years'].sum()
    
    print(f"\nOptimization Complete!")
    print(f"Selected {len(selected_gages)} non-nested basins.")
    print(f"True Maximized High-Quality Basin-Water-Years: {int(total_selected_years)} years.")
    
    return basins_gdf

basins.geometry = basins.geometry.to_crs(9822)
results_df = select_max_water_years_global(basins, flow)
results_df.to_file(os.path.join(ncwd, r'data/shapefiles/basin_selection_results.gpkg'))
results_df.to_file(os.path.join(ncwd, r'data/shapefiles/basin_selection_results.shp'))

model_basins = results_df[results_df.model_category.str.contains('train_test')]
model_basins.explore()

############# REMOVE NESTED BASINS ##############
headwaters = remove_nested_basins(model_basins)
headwaters.explore()
headwaters_path = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\data\shapefiles\UCOL_headwaters_sheds.gpkg'
headwaters.to_file(headwaters_path, driver='GPKG')
headwaters_path2 = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\data\shapefiles\UCOL_headwaters_sheds2.shp'
headwaters.to_file(headwaters_path2)
headwaters = gpd.read_file(headwaters_path)

# Save to file
gages_headwaters = gages_gdf[gages_gdf.index.isin(headwaters['gage'])]
gages_headwaters.to_file(join(ncwd, r"data\shapefiles\UCOL_headwater_gages.gpkg"), driver="GPKG")

# add in camels
# maybeee

# maybe later we will grab data for these
parents = basins[~basins.gage.isin(headwaters.gage)]


#### NOTE TO SELF ####
# Seems like most of the no work list is canals or gages that only measure water quality (no cfs)
import pygridmet as gridmet
import rioxarray
from shapely.geometry import mapping

gdf = headwaters

# Variables requested from GRIDMET service
variables = ['pr', 'srad', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd', 'pet']

# Define aggregation types per variable
mean_vars = ['srad', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd',]
sum_vars = ['pet', 'pr']

# Run the SNODAS extraction ONCE up front

#############
# You ran this for data\shapefiles\UCOL_headwaters_sheds.shp already. In snodas_processed
#############
keep = gpd.read_file(r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\data\shapefiles\all_UCOL_basins.gpkg')
rmv = gpd.read_file(r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\data\shapefiles\UCOL_headwaters_sheds.gpkg')
rmv2 = gpd.read_file(r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\data\shapefiles\basin_selection_results.gpkg')
print(len(keep))
keep = keep[~keep['gage'].isin(rmv['gage'])]
print(len(keep))
keep = keep[~keep['gage'].isin(rmv2['gage'])]
print(len(keep))


snodas_folder = r"N:\Research\Kampf\Private\KeenanW\SNODAS"
snodas_export_folder = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\snodas_processed2'
extract_all_snodas(gdf, date_range, snodas_folder, snodas_export_folder, 'gage')

snodas_dfs = []
years = list(range(pd.to_datetime(date_range[0]).year, pd.to_datetime(date_range[1]).year + 1))
for year in years:
    snodas_dfs.append(pd.read_csv(os.path.join(snodas_export_folder, f'SNODAS_SWE_{year}.csv')))

snodas_lookup = pd.concat(snodas_dfs)
snodas_lookup['time'] = pd.to_datetime(snodas_lookup['time'])
snodas_lookup = snodas_lookup.rename(columns={'time':'date'})
snodas_lookup = snodas_lookup.set_index('date')

gdf = keep
# Loop over each individual watershed
for idx, row in gdf.iterrows():
    gage_id = str(row['gage'])
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
        ds = ds.rio.write_crs(gdf.crs)
        
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
    rename_dict['time'] = 'date'
    rename_dict.update({v: f"{v}_sum" for v in sum_vars})
    gr_df = gr_df.rename(columns=rename_dict)

    # Set the time index
    gr_df = gr_df.set_index('date')
    
    # EXTRACTION FIX: Instead of opening files, instantly pull the column from memory!
    # Pull the pre-calculated time series matching this specific gage ID
    gage_swe_series = snodas_lookup[row[gage_col]] 
    
    # Map the series back to the main dataframe
    gr_df['swe'] = gage_swe_series
    gr_df = gr_df.rename(columns={'swe':'swe_sum'})

    # get streamflow
    flow_series = flow[f'USGS-{gage_id}']
    gr_df['Q_cfs'] = flow_series

    gr_df.to_csv(fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\unfilled\{gage_id}.csv')





