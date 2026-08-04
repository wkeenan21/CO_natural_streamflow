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
appDir = os.path.join(appcwd, r'timeseries') # does not have each individual diversion

# Now we need to select basins for training, testing, and implementation

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
    
    print(f"Total Water Years in Period: {total_wy_count} ({water_years.min()} to {water_years.max()})")
    
    for gage in df.columns:
        # Count non-null flow values per water year
        wy_counts = df[gage].notna().groupby(water_years).sum()
        
        # Align expected days with the years present
        expected_days = days_per_wy.loc[wy_counts.index]
        
        # Calculate completion percentage
        wy_completion = wy_counts / expected_days
        
        # Count how many water years pass the threshold
        good_years_count = (wy_completion >= completion_threshold).sum()
        
        print(f"good_years_count: {good_years_count:<18}")



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
    # 1. Convert to string base representation without precision loss
    if isinstance(id_val, float):
        id_str = str(int(round(id_val)))
    elif isinstance(id_val, str) and '.' in id_val:
        try:
            id_str = str(int(round(float(id_val))))
        except ValueError:
            id_str = id_val.strip()
    else:
        id_str = str(id_val).strip()

    # 2. Clean prefixes
    if 'USGS-' in id_str:
        id_str = id_str.replace('USGS-', '')

    # 3. Pad 7-digit IDs with a leading zero
    if len(id_str) == 7 and id_str.isdigit():
        return id_str.zfill(8)
    elif id_str == '93710009':
        return id_str.zfill(9)

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
skip = True
if not skip:
    ucol_gage = '09380000'
    ucol = nldi.get_basins(ucol_gage)['geometry'].iloc[0]
    ucol_geom = remove_polygon_holes(ucol)
    ucol = gpd.GeoDataFrame(geometry=[ucol_geom], crs=4326)
    ucol.to_parquet(os.path.join(ncwd, r'spatial_data\UCOL.parquet'))
    ucol.to_parquet(os.path.join(appcwd, r'spatial_data\UCOL.parquet'))

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
    pps = pps[
        ~pps['name'].str.lower().str.contains('canal') & 
        ~pps['name'].str.lower().str.contains('tunnel') &
        ~pps['name'].str.lower().str.contains('diversion') &
        ~pps['name'].str.lower().str.contains('ditch')
    ]
    print(f"{len(pps)} gages that aren't tunnels or canals or diversions")

    # info is already a GeoDataFrame with point geometry
    gages = pps.set_crs("EPSG:4326")

    ###################### STEP 2: Delineate watersheds from NLDI #####################
    nldi  = NLDI()
    basins_list = []
    for id in gages['gage'].to_list():
        try:
            basin = nldi.get_basins(id) # watershed polygons
            basins_list.append(basin)
        except:
            print(gages[gages.gage==id]['name'])
            continue
    # the polygons
    basins = pd.concat(basins_list)
    basins['gage'] = basins.index
    basins['gage'] = basins['gage'].apply(fix_gage_id)

    # NLDI didn't get:
    mdl = gages[~gages['gage'].isin(basins.gage)]
    print(len(mdl))

    # read these we already delineated
    man_delin = gpd.read_file(os.path.join(ncwd, r'data/shapefiles/man_delin.shp'))
    man_delin = pd.merge(left=man_delin, right=gages[['name','gage']], on='gage', how='left').dropna()

    # Now join the manually delineated ones to the basins list
    man_delin = man_delin.set_geometry('geometry')
    man_delin = man_delin.set_crs(4326)
    basins = pd.merge(left=basins, right=gages[['name','gage']], on='gage', how='left')
    basins = pd.concat([basins, man_delin])

    # check if we already did these
    mdl = mdl[~mdl.gage.isin(man_delin.gage)]
    print(len(mdl), mdl.gage.unique())

    #############################################################################
    # GET STREAMFLOW AND MERGE DIVERSION DATA
    #############################################################################
    from collections import defaultdict

    # make dict for names
    name_dict = gages.set_index('gage')['name'].to_dict()
    # Get all streamflow
    date_range = ("1979-10-01", "2025-09-30")
    #flow = NWIS.get_streamflow(gages['gage'].to_list(), date_range, freq="dv") # comes in cubic meters per second

    # send to csv so you don't have to run that again
    big_flow_path = os.path.join(ncwd,r'data\timeseries\big_flow.csv')
    #flow.to_csv(big_flow_path, index_label='date')
    flow = pd.read_csv(big_flow_path).set_index('date')
    flow.index = pd.to_datetime(flow.index)

    # fix timeseries index
    flow.index = flow.index.normalize().tz_localize(None)

    # rename columns
    rename_dict = {}
    for usgsgage in flow.columns:
        gage = fix_gage_id(usgsgage)
        rename_dict[usgsgage] = gage
    flow = flow.rename(columns=rename_dict)

    # just doesn't work for some?
    nwis_fail = gages[~gages['gage'].isin(flow.columns)]
    nwis_fail = add_usgs_prefix(nwis_fail['gage'].to_list())

    df, metadata = waterdata.get_daily(
        monitoring_location_id=nwis_fail,
        parameter_code='00060',
        time=f'{date_range[0]}/{date_range[1]}'
    )

    # add all the ones that didn't work
    for usgsgage in df['monitoring_location_id'].unique():
        gage = fix_gage_id(usgsgage)
        sub_df = df[df['monitoring_location_id']==usgsgage][['time', 'value']]
        sub_df = sub_df.rename(columns={'value':gage})
        sub_df = sub_df.set_index('time')
        flow[gage] = sub_df[gage] * 0.0283168 # convert to cms

    # get basin area
    basins.geometry = basins.geometry.to_crs(5070)
    basins['area_m2'] = basins.area

    # save all the basins and gages
    print(f'{len(basins)} basins, {len(gages)} gages')

    # check for ones with streamflow for the app
    keep = []
    for gage in flow.columns:
        if flow[gage].notna().sum() >= 365:
            keep.append(gage)

    basins = basins[basins['gage'].isin(keep)]
    gages = gages[gages['gage'].isin(keep)]
    print(f'{len(basins)} basins, {len(gages)} gages')
    print('basins with no gage:', set(basins.gage).difference(set(gages.gage)))
    need_basin = set(gages.gage).difference(set(basins.gage))
    print('gages with no basin:', set(gages.gage).difference(set(basins.gage))) # should be none

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

    need_basin = gages[gages.gage.isin(need_basin)]

    # delineate
    # watershed_polygons_gdf = delineate_watersheds_preprocessed(
    #     fdir_path=fdir_path,
    #     acc_path=acc_path,
    #     points_gdf=need_basin,
    #     gage_col='gage'
    # )
    # man_delin = gpd.GeoDataFrame().from_dict(watershed_polygons_gdf)
    # man_delin.geometry.crs = 4326
    # man_delin = pd.merge(left=man_delin, right=gages[['gage', 'name']], on='gage', how='left')
    #man_delin.to_file(os.path.join(ncwd, r'data/shapefiles/man_delin2.shp'))
    man_delin = gpd.read_file(os.path.join(ncwd, r'data/shapefiles/man_delin2.shp'))

    # add to basins...
    basins = basins.to_crs(4326)
    print(len(basins))
    basins = pd.concat([basins, man_delin])
    print(len(basins))

    # remove holes from geometry:
    # Fill holes by re-constructing Polygons using only their exterior boundary
    from shapely.geometry import Polygon
    def remove_holes_and_force_polygon(geom):
        if geom is None or geom.is_empty:
            return geom

        # 1. If it's a MultiPolygon, select the largest polygon component by area
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda p: p.area)

        # 2. Re-create the Polygon using only its exterior boundary (removes holes)
        if geom.geom_type == "Polygon":
            return Polygon(geom.exterior)

        return geom

    # Apply to the GeoDataFrame
    basins["geometry"] = basins["geometry"].apply(remove_holes_and_force_polygon)

    # add area m2
    basins.geometry = basins.geometry.to_crs(9822)
    basins['area_m2'] = basins.area
    basins['area_km2'] = basins['area_m2'] / 1000000

    # Enforce that they are the same
    basins_path = os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet')
    gages_path = os.path.join(appcwd, r'spatial_data/all_UCOL_gages.parquet')

    basins = basins.sort_values(by='gage').reset_index(drop=True)
    gages = gages.sort_values(by='gage').reset_index(drop=True)
    if all(basins.gage == gages.gage):
        basins.to_parquet(basins_path)
        gages.to_parquet(gages_path)
        basins.to_parquet(os.path.join(ncwd, r'spatial_data/all_UCOL_basins.parquet'))
        gages.to_parquet(os.path.join(ncwd, r'spatial_data/all_UCOL_gages.parquet'))

    basins[basins.gage==ucol_gage].explore()
    # DIVERSIONS
    # COLUMN = siteID

skip2 = False
if not skip2:
    ###################
    # SKIP STEPS BY READING HERE
    basins_path = os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet')
    gages_path = os.path.join(appcwd, r'spatial_data/all_UCOL_gages.parquet')
    basins = gpd.read_parquet(basins_path)
    gages = gpd.read_parquet(gages_path)
    big_flow_path = os.path.join(ncwd,r'data\timeseries\big_flow.csv')
    flow = pd.read_csv(big_flow_path).set_index('date')
    flow.index = pd.to_datetime(flow.index)
        # fix timeseries index
    flow.index = flow.index.normalize().tz_localize(None)
    # rename columns
    rename_dict = {}
    for usgsgage in flow.columns:
        gage = fix_gage_id(usgsgage)
        rename_dict[usgsgage] = gage
    flow = flow.rename(columns=rename_dict)
    name_dict = gages.set_index('gage')['name'].to_dict()
    date_range = ("1979-10-01", "2025-09-30")
    #####################

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

    # 1980 to 2025
    dvrsFlow = pd.read_csv(os.path.join(ncwd, "data\diversion\will_processed\combined_diversion_records_filtered_filled_cfs_fill_years.csv"))
    dvrsFlow = dvrsFlow.rename(columns={'datetime':'date'})
    dvrsFlow['date'] = pd.to_datetime(dvrsFlow['date'])
    #dvrsFlow = pd.concat([dvrsFlow, dvrsFlow])

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

    import time
    current_time = time.time()
    seconds_in_1_hours = 60 * 60
    # send to csvs
    for gage in basins.gage:

        out_path_small = os.path.join(appDir, f"{gage}.csv")
        out_path = os.path.join(wdvrsDir, f"{gage}.csv")
        
        # Check if the file exists first
        if os.path.exists(out_path_small):
            df = pd.read_csv(out_path_small, parse_dates=['date'], index_col='date')
            df_cleaned = df[~df.index.duplicated(keep='first')]
            df_cleaned = df_cleaned.asfreq('D')
            df_cleaned.to_csv(out_path_small, index_label='date')

        if os.path.exists(out_path):
            df = pd.read_csv(out_path, parse_dates=['date'], index_col='date')
            df_cleaned = df[~df.index.duplicated(keep='first')]
            df_cleaned = df_cleaned.asfreq('D')
            df_cleaned.to_csv(out_path, index_label='date')

            # Get the last modification time of the file
            #file_mod_time = os.path.getmtime(out_path_small)
            
            # If the file was modified less than 24 hours ago, skip it
            # if (current_time - file_mod_time) < seconds_in_1_hours:
            #     print(f"Skipping {gage}.csv - updated within the last 24 hours.")
            #     continue

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
            if len(combined_df) < 1:
                raise Exception('merge failed: df has no rows')

            # 5. aggregate by type
                # Aggregate columns matching each diversion type
            for unit in units:
                combined_df[f'Q_NAT_{unit}'] = combined_df[f'Q_{unit}'] - combined_df[f'Q_CU_{unit}']

        else:
            # If no diversions found, we still save the original flow (or skip)
            combined_df = df
            
        combined_df = combined_df.asfreq('D')
        # 4. Save the new CSV
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

#### GRAB CLIMATE FORCING ####
basins_path = os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet')
gages_path = os.path.join(appcwd, r'spatial_data/all_UCOL_gages.parquet')
basins = gpd.read_parquet(basins_path)
gages = gpd.read_parquet(gages_path)

import pygridmet as gridmet
import rioxarray
from shapely.geometry import mapping

# smaller date range for this
date_range = ("2003-10-01", "2025-09-30")

# Variables requested from GRIDMET service
variables = ['pr', 'srad', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd', 'pet']

# Define aggregation types per variable
mean_vars = ['srad', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd',]
sum_vars = ['pet', 'pr']

# Run the SNODAS extraction ONCE up front

#############
# You ran this for data\shapefiles\UCOL_headwaters_sheds.shp already. In snodas_processed
#############
gages_done = []
for fol in [1,2]:
    snodas_export_folder = fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\snodas_processed{fol}'
    df = pd.read_csv(os.path.join(snodas_export_folder, 'SNODAS_SWE_WY2025.csv'))
    gages_done = gages_done + df.columns.to_list()
        
gages_done.remove('date')
gages_done = set(gages_done)
print(f'{len(gages_done)} with snodas')
left2do = set(basins.gage).difference(gages_done)
print(f'{len(left2do)} without snodas')

basins2do = basins[basins.gage.isin(left2do)]

snodas_folder = r"N:\Research\Kampf\Private\KeenanW\SNODAS"
snodas_export_folder = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\snodas_processed3'
#extract_all_snodas(basins2do, date_range, snodas_folder, snodas_export_folder, 'gage')

snodas_dfs = []
years = list(range(pd.to_datetime(date_range[0]).year+1, pd.to_datetime(date_range[1]).year + 1))

for fol in [1,2,3]:
    snodas_export_folder = fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\snodas_processed{fol}'
    for year in years:
        snodas_dfs.append(pd.read_csv(os.path.join(snodas_export_folder, f'SNODAS_SWE_WY{year}.csv')))

snodas_lookup = pd.concat(snodas_dfs)
snodas_lookup['date'] = pd.to_datetime(snodas_lookup['date'])
snodas_lookup = snodas_lookup.set_index('date')
snodas_lookup = snodas_lookup.sort_values(by='date')

# add area km2
basins = basins.to_crs(4326)
# prepare rename dict
rename_dict = {v: f"{v}_mean" for v in mean_vars}
rename_dict['time'] = 'date'
rename_dict.update({v: f"{v}_sum" for v in sum_vars})
appDir = os.path.join(appcwd, r'timeseries') # does not have each individual diversion
nodata = []
# Loop over each individual watershed
for gage in basins.gage:
    #gage = '09303400'
    outpath = fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\gr_snodas\{gage}.csv'
    outpath2 = fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\gr_snodas_flow7\{gage}.csv'

    # 1. Skip if both files already exist
    # if os.path.exists(outpath) and os.path.exists(outpath2):
    #     print(f'{gage} already done')
    #     continue

    # 2. Check for SNODAS data
    try:
        gage_swe_series = snodas_lookup[gage].dropna()
        if gage_swe_series.empty or any(gage_swe_series.isna()):
            raise Exception('no snow data')
    except Exception as e:
        print(f'no snodas for {gage}')
        continue

    # Isolate watershed geometry once
    single_gdf = basins[basins.gage == gage]
    name = single_gdf['name'].iloc[0]
    areakm2 = single_gdf['area_km2'].iloc[0]

    # 3. Generate gridMET file (outpath) if missing
    if True: #not os.path.exists(outpath):
        print(f"downloading gridmet: {gage}...")

        if areakm2 < 10:
            print(f"getting by coords {gage}")
            try:
                gr_df = gridmet.get_bycoords(
                    coords=[single_gdf.geometry.iloc[0].centroid.x, single_gdf.geometry.iloc[0].centroid.y], 
                    dates=date_range, 
                    crs=single_gdf.crs, 
                    variables=variables
                )
                gr_df.columns = gr_df.columns.str.split(' ').str[0]
                gr_df = gr_df.rename(columns=rename_dict)

                area_factor = areakm2 / 16
                for var in ['pr', 'pet']:
                    if f'{var}_sum' in gr_df.columns:
                        gr_df[f'{var}_sum'] = gr_df[f'{var}_sum'] * area_factor

                gr_df['swe'] = gage_swe_series
                gr_df = gr_df.rename(columns={'swe': 'swe_sum'})
                gr_df.to_csv(outpath, index_label='date')
            except Exception as e:
                print(f"Failed getting coords for {gage}: {e}")

        elif areakm2 < 25000:
            print(f"area: {areakm2:.2f} km2")
            try:
                ds = gridmet.get_bygeom(geometry=single_gdf.geometry.iloc[0], dates=date_range, crs=single_gdf.crs, variables=variables)
                ds_mean = ds[mean_vars].mean(dim=["lon", "lat"], skipna=True)
                ds_sum = ds[sum_vars].sum(dim=["lon", "lat"], skipna=True)
                df_mean = ds_mean.to_dataframe().reset_index()
                df_sum = ds_sum.to_dataframe().reset_index()
                gr_df = pd.merge(df_mean, df_sum, on='time')
                gr_df = gr_df.rename(columns=rename_dict)
                gr_df = gr_df.set_index('date', drop=True)
                gr_df['swe'] = gage_swe_series
                gr_df = gr_df.rename(columns={'swe': 'swe_sum'})
                gr_df.to_csv(outpath, index_label='date')
            except Exception as e:
                print(f'{gage} error during get_bygeom: {e}')
                if 'unable to allocate' not in str(e).lower():
                    print(f"getting by coords fallback for {gage}")
                    try:
                        gr_df = gridmet.get_bycoords(
                            coords=[single_gdf.geometry.iloc[0].centroid.x, single_gdf.geometry.iloc[0].centroid.y], 
                            dates=date_range, 
                            crs=single_gdf.crs, 
                            variables=variables
                        )
                        gr_df.columns = gr_df.columns.str.split(' ').str[0]
                        gr_df = gr_df.rename(columns=rename_dict)

                        area_factor = areakm2 / 16
                        for var in ['pr', 'pet']:
                            if f'{var}_sum' in gr_df.columns:
                                gr_df[f'{var}_sum'] = gr_df[f'{var}_sum'] * area_factor

                        gr_df['swe'] = gage_swe_series
                        if gr_df['swe'].isna().sum() == len(gr_df):
                            fuck
                        gr_df = gr_df.rename(columns={'swe': 'swe_sum'})
                        gr_df.to_csv(outpath, index_label='date')
                    except Exception as fallback_e:
                        print(f"Fallback failed for {gage}: {fallback_e}")
                else:
                    print(f"{gage} {name} too big (memory allocation issue)")
        else:
            print(f"{gage} {name} too big, didn't try")

#############
# Merge with flow with 0 interpolation
#############
for gage in train_eligible_rdf[train_eligible_rdf['diversion_frac'].isna()].gage:

    outpath = fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\gr_snodas\{gage}.csv'
    outpath2 = fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\gr_snodas_flow0\{gage}.csv'

    if True: #os.path.exists(outpath) and not os.path.exists(outpath2):
        print(f'merging flow for {gage}...')

        gr_df = pd.read_csv(outpath, parse_dates=['date'], index_col='date')
        if 'spatial_ref_x' in gr_df.columns and 'spatial_ref_y' in gr_df.columns:
            gr_df = gr_df.drop(columns=['spatial_ref_x', 'spatial_ref_y'])

        flow_path = os.path.join(appDir, f"{gage}.csv")
        flow_df = pd.read_csv(flow_path, parse_dates=['date'], index_col='date')
        flow_df = flow_df[~flow_df.index.duplicated(keep='first')]

        Q_col = 'Q_cfs'
        first_valid = flow_df[Q_col].first_valid_index()
        last_valid = flow_df[Q_col].last_valid_index()
        flow_df = flow_df.loc[first_valid:last_valid]

        if last_valid.year < 2003:
            print(f'no data after {last_valid}, not generating {gage}')
            continue

        #flow_df = flow_df.interpolate(method='linear', limit=7)
        #diagnose_gage_quality(flow_df[[Q_col]])

        gr_df = pd.merge(left=gr_df, right=flow_df, how='left', left_index=True, right_index=True)
        gr_df = gr_df.asfreq('D')

        gr_df.to_csv(outpath2, index_label='date')
        print(f'Successfully generated {outpath2}')


############# REMOVE BASINS WITH TOO MUCH DAM STORAGE OR DIVERSION ##############
# SKIP STEPS BY READING HERE
basins_path = os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet')
gages_path = os.path.join(appcwd, r'spatial_data/all_UCOL_gages.parquet')
basins = gpd.read_parquet(basins_path)
gages = gpd.read_parquet(gages_path)

# get attributes
attrs_path = os.path.join(appcwd, r'attributes\all_UCOL_attributes.csv')
attrs = pd.read_csv(attrs_path, dtype={'gage': np.float64})
attrs['gage'] = attrs['gage'].apply(fix_gage_id)
attrs = attrs.set_index('gage')

# add names to attrs
basins = basins.set_index('gage')
attrs['name'] = basins.name
name_col = attrs.pop("name")
attrs.insert(0, "name", name_col)
attrs.to_parquet(os.path.join(appcwd, r'attributes\all_UCOL_attributes.parquet'))

# check for data completeness
results = []
Q_col = 'Q_cfs'
for gage in basins.index.to_list():
    rd = {'gage':gage}
    # access the csv
    csv = os.path.join(ncwd, fr'timeseries\gr_snodas_flow0\{gage}.csv')
    try:
        df = pd.read_csv(csv, index_col='date', parse_dates=['date'])
        rd['data'] = True
    except:
        print('No timeseries', gage)
        rd['data'] = False
        results.append(rd)
        continue
    
    # valid dates
    first_valid = df[Q_col].first_valid_index()
    last_valid = df[Q_col].last_valid_index()

    # limit to valid dates
    df = df.loc[first_valid:last_valid]

    # calculate mean vars
    variables = [Q_col, 'pr_sum', 'swe_sum', 'pet_sum', 'tmmx_mean', 'tmmn_mean']
    for var in variables:
        rd[f'{var}_mean'] = df[var].mean()

    if np.isnan(rd[f'{Q_col}_mean'])
        rd['data'] = False
    # check for NAs
    NAs = 0.1
    NA_ratio = df[Q_col].isna().sum() / len(df)
    rd['NA_ratio'] = NA_ratio

    # check length
    days = 365*2
    period = len(df)

    rd['firstday'] = first_valid
    rd['lastday'] = last_valid
    rd['period'] = period

    # CONSUMPTIVE USE
    CU_vars = ['irrigation' , 'municipal', 'intrabasin', 'interbasin', 'industrial']
    CU = 0
    for var in CU_vars:
        try: # try because the columns don't exist sometimes
            CU += df[f'{var}_cfs'].sum()
        except:  # noqa: E722, S110
            pass
    CU = CU * -1
    try:
        CU += df['transbasin_cfs'].sum()
    except:  # noqa: E722, S110
        pass
    Q = df['Q_cfs'].sum()
    divert_ratio = 0.1
    divert = CU/Q
    rd['diversion_frac'] = divert

    results.append(rd)

rdf = pd.DataFrame().from_dict(results)
# marge with the geometry and area
rdf = pd.merge(left=basins, right=rdf, on='gage')
# merge with the attributes
rdf = pd.merge(left=rdf, right=attrs, on=['gage', 'name'])

############# BASIN SELECTION SCHEME ##################
rdf = rdf[rdf.data]

############ visualize basin size
import seaborn as sns
var='diversion_frac'
sns.histplot(data=rdf, x=var, bins=20, log_scale=True, color='skyblue', edgecolor='black')
plt.title(f'Histogram of {var}')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.5)
plt.show()

from sklearn.model_selection import train_test_split
# which have enough data to be suitable for training and testing?
NA_thresh = 0.15
period_thresh = 365*1
rdf2 = rdf[(rdf.NA_ratio < NA_thresh) & (rdf.period > period_thresh)]

# We select the testing basins from the purely natural.
strat = 'Q_cfs_mean'
strat_col = f'strata_{strat}'
bins=10

rdf2[strat_col] = pd.cut(rdf2[strat], bins=10)
natty = rdf2[(rdf2.diversion_frac==0) & (rdf2.dor_pc_pva == 0)]
test_size = 10/len(natty) # I want 10 basins

_, test_df = train_test_split(natty, test_size=test_size, stratify=natty[strat_col], random_state=42)
test_df[['name', 'geometry']].explore()

################# Nested matrix ##################
# 1. Ensure a projected CRS for accurate area calculations (reproject if geographic)
if basins.crs is not None and basins.crs.is_geographic:
    basins_proj = basins.to_crs(basins.estimate_utm_crs())
else:
    basins_proj = basins.copy()

# Extract geometries, areas, and gage IDs
gages = basins_proj.index.values
geoms = basins_proj.geometry.values
areas = basins_proj.geometry.area.values
n = len(basins_proj)

# 2. Set an overlap threshold (e.g., 85% of the smaller geometry's area)
OVERLAP_THRESHOLD = 0.85 

# 3. Initialize the 286x286 DataFrame
nested_matrix = pd.DataFrame(False, index=gages, columns=gages)

# 4. Use spatial index (sindex) to efficiently check overlapping candidates
sindex = basins_proj.sindex

for i in range(n):
    geom_i = geoms[i]
    area_i = areas[i]
    
    # Fast bounding-box filtering
    possible_matches = list(sindex.intersection(geom_i.bounds))
    
    for j in possible_matches:
        if i >= j:  # Avoid redundant pairs and self-comparison
            continue
            
        geom_j = geoms[j]
        
        # Calculate actual intersection if bounding boxes overlap
        if geom_i.intersects(geom_j):
            intersection_area = geom_i.intersection(geom_j).area
            smaller_area = min(area_i, areas[j])
            
            # If the intersection makes up most of the smaller watershed's area, they are nested
            if (intersection_area / smaller_area) >= OVERLAP_THRESHOLD:
                nested_matrix.iat[i, j] = True
                nested_matrix.iat[j, i] = True
# ==========================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==========================================
TEST_SIZE = 10
TRAIN_SIZE = 50
NUM_TRAIN_SETS = 10

AREA_TOLERANCE_FRAC = 0.20 
MAX_ATTEMPTS = 5
SEED = 42
np.random.seed(SEED)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def sample_non_nested_subset(pool_df, size, nested_mat, target_area=None, area_tol=0.20, max_attempts=1000):
    """
    Randomly/greedily samples a mutually non-nested subset of specified size.
    Enforces target mean area if specified.
    """
    pool_gages = pool_df['gage'].values
    
    for _ in range(max_attempts):
        shuffled = np.random.choice(pool_gages, size=len(pool_gages), replace=False)
        selected = []
        
        for gage in shuffled:
            # Check for mutual nesting among already selected gages in this set
            if not nested_mat.loc[selected, gage].any() if selected else True:
                selected.append(gage)
                if len(selected) == size:
                    break
        
        if len(selected) == size:
            if target_area is not None:
                mean_area = pool_df.loc[pool_df['gage'].isin(selected), 'area_km2'].mean()
                if abs(mean_area - target_area) / target_area <= area_tol:
                    return selected
            else:
                return selected

    raise ValueError(f"Could not find a valid non-nested subset of size {size} within constraints.")

# ==========================================
# 3. STRATEGIC TESTING SET SELECTION
# ==========================================
# Count total nested connections for each gage (excluding self)
nesting_counts = nested_matrix.sum(axis=1) - nested_matrix.values.diagonal().astype(int)
rdf['nesting_degree'] = rdf['gage'].map(nesting_counts)

# Filter pristine watersheds
pristine_pool = rdf[(rdf['diversion_frac'] == 0) & (rdf['dor_pc_pva'] == 0)].copy()

# Sort pristine pool: prioritize large area (descending) and low nesting degree (ascending)
pristine_pool = pristine_pool.sort_values(
    by=['nesting_degree', 'area_km2'], 
    ascending=[True, False]
)

# Pick the test set prioritizing top candidates
test_gages = sample_non_nested_subset(
    pool_df=pristine_pool.head(30), # Top 30 candidate pool balancing area & low nesting
    size=TEST_SIZE,
    nested_mat=nested_matrix,
    max_attempts=MAX_ATTEMPTS
)

test_set = rdf[rdf['gage'].isin(test_gages)].copy()
target_mean_area = test_set['area_km2'].mean()

print(f"=== TEST SET SELECTED ({len(test_set)} gages) ===")
print(f"Mean Area: {target_mean_area:.2f} km²")
print(f"Mean Nesting Degree: {test_set['nesting_degree'].mean():.1f} connections")

# ==========================================
# 4. EXCLUDE TEST GAGES & ALL NESTED RELATIVES
# ==========================================
# Find all gages nested with ANY test set gage
test_and_nested_mask = nested_matrix.loc[test_gages].any(axis=0)
blocked_gages = nested_matrix.columns[test_and_nested_mask].tolist()

# Define candidate pool purely isolated from the test set
train_eligible_rdf = rdf[~rdf['gage'].isin(blocked_gages)].copy()

print(f"\nTotal Watersheds: {len(rdf)}")
print(f"Blocked (Test + Nested with Test): {len(blocked_gages)}")
print(f"Eligible Training Watersheds: {len(train_eligible_rdf)}")

# ==========================================
# 5. SELECT 10 TRAINING SETS
# ==========================================
train_eligible_rdf['reg_score'] = train_eligible_rdf['diversion_frac'] + (train_eligible_rdf['dor_pc_pva'] / 1000)
target_percentiles = np.linspace(5, 95, NUM_TRAIN_SETS)
target_reg_values = np.percentile(train_eligible_rdf['reg_score'], target_percentiles)

training_sets = {}

for k, target_reg in enumerate(target_reg_values, 1):
    train_eligible_rdf['score_diff'] = (train_eligible_rdf['reg_score'] - target_reg).abs()
    
    # Take closest candidates to the target regulation level
    pool_k = train_eligible_rdf.sort_values('score_diff').head(100)
    
    train_gages = sample_non_nested_subset(
        pool_df=pool_k,
        size=TRAIN_SIZE,
        nested_mat=nested_matrix,
        target_area=target_mean_area,
        area_tol=AREA_TOLERANCE_FRAC,
        max_attempts=MAX_ATTEMPTS
    )
    
    train_df = rdf[rdf['gage'].isin(train_gages)].copy()
    training_sets[f'train_set_{k}'] = train_df
    
    mean_div = train_df['diversion_frac'].mean()
    mean_dor = train_df['dor_pc_pva'].mean()
    mean_area = train_df['area_km2'].mean()
    print(f"Train Set {k:02d} | Mean Div: {mean_div:.3f} | Mean DOR: {mean_dor:.3f} | Mean Area: {mean_area:.1f} km²")

############# REMOVE NESTED BASINS ##############
headwaters = remove_nested_basins(basins)
headwaters.explore()
headwaters_path = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\data\shapefiles\UCOL_headwaters_sheds.gpkg'
headwaters.to_file(headwaters_path, driver='GPKG')
headwaters_path2 = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\data\shapefiles\UCOL_headwaters_sheds2.shp'
headwaters.to_file(headwaters_path2)
headwaters = gpd.read_file(headwaters_path)

# Save to file
gages_headwaters = gages[gages.index.isin(headwaters['gage'])]
gages_headwaters.to_file(join(ncwd, r"data\shapefiles\UCOL_headwater_gages.gpkg"), driver="GPKG")

# add in camels
# maybeee

# maybe later we will grab data for these
parents = basins[~basins.gage.isin(headwaters.gage)]





