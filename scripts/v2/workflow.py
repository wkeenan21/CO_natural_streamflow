import geopandas as gpd
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.strtree import STRtree
from pynhd import NLDI
nldi = NLDI()
from pygeohydro import NWIS
nwis = NWIS()
from shapely.geometry import MultiPolygon, Polygon
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
def count_complete_water_years(df: pd.DataFrame, col: str) -> int:
    """
    Counts the number of water years (Oct 1 to Sep 30) with 100% valid data.
    
    A water year is considered complete only if every required calendar day 
    (365 days, or 366 in a leap year) is present and non-null in the column.
    """
    if col not in df.columns or df.empty:
        return 0
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # 1. Assign Water Year (WY): Oct-Dec belong to the next calendar year's WY
    water_years = df.index.year + (df.index.month >= 10).astype(int)

    # 2. Count non-null days per water year
    valid_counts = df[col].notna().groupby(water_years).sum()

    # 3. Verify against expected days (accounting for leap years)
    complete_count = 0
    for wy, valid_days in valid_counts.items():
        wy_start = pd.Timestamp(year=wy - 1, month=10, day=1)
        wy_end = pd.Timestamp(year=wy, month=9, day=30)
        expected_days = (wy_end - wy_start).days + 1

        if valid_days == expected_days:
            complete_count += 1

    return complete_count

def plot_hydrograph(df: pd.DataFrame, Q='Q_cfs', title: str = "Hydrograph") -> None:
    """Plots a hydrograph for a DataFrame with a datetime index and 'Q' column.

    Restricts the display range between the first and last valid 'Q' values.
    """
    # Identify non-null Q entries
    valid_q = df[Q].dropna()

    if valid_q.empty:
        raise ValueError("No valid 'Q' data found in the DataFrame.")

    # Determine start and end datetime bounds
    first_valid = valid_q.index[0]
    last_valid = valid_q.index[-1]

    # Slice the DataFrame to the valid date range
    df_sliced = df.loc[first_valid:last_valid]

    # Plot creation
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df_sliced.index,
        df_sliced[Q],
        color="tab:blue",
        linewidth=1.5,
        label="Discharge (Q)",
    )

    # Formatting
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Discharge ($Q$)", fontsize=11)
    ax.set_xlim(first_valid, last_valid)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

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
skip = False
if not skip:
    ucol_gage = '09380000'
    ucol = nldi.get_basins(ucol_gage)['geometry'].iloc[0]
    ucol_geom = remove_polygon_holes(ucol)
    ucol = gpd.GeoDataFrame(geometry=[ucol_geom], crs=4326)
    ucol.to_file(os.path.join(ncwd, r'spatial_data\UCOL.shp'))
    # ucol.to_parquet(os.path.join(ncwd, r'spatial_data\UCOL.parquet'))
    # ucol.to_parquet(os.path.join(appcwd, r'spatial_data\UCOL.parquet'))

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

    # nldi  = NLDI()
    # basins_list = []
    # for id in gages['gage'].to_list():
    #     try:
    #         basin = nldi.get_basins(id) # watershed polygons
    #         basins_list.append(basin)
    #     except:
    #         print(gages[gages.gage==id]['name'])
    #         continue
    # # the polygons
    # basins = pd.concat(basins_list)
    # basins['gage'] = basins.index
    # basins['gage'] = basins['gage'].apply(fix_gage_id)

    # # NLDI didn't get:
    # mdl = gages[~gages['gage'].isin(basins.gage)]
    # print(len(mdl))

    # # read these we already delineated
    # man_delin = gpd.read_file(os.path.join(ncwd, r'data/shapefiles/man_delin.shp'))
    # man_delin = pd.merge(left=man_delin, right=gages[['name','gage']], on='gage', how='left').dropna()

    # # Now join the manually delineated ones to the basins list
    # man_delin = man_delin.set_geometry('geometry')
    # man_delin = man_delin.set_crs(4326)
    # basins = pd.merge(left=basins, right=gages[['name','gage']], on='gage', how='left')
    # basins = pd.concat([basins, man_delin])

    # # check if we already did these
    # mdl = mdl[~mdl.gage.isin(man_delin.gage)]
    # print(len(mdl), mdl.gage.unique())

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


    # use the NLDI gages to approximate basin boundaries
    # (if its not in here, we don't want it)
    old_basins_path = os.path.join(appcwd, r'spatial_data\v1\all_UCOL_basins.parquet')
    old_basins = gpd.read_parquet(old_basins_path).to_crs(5070)
    keep = []
    keep = keep + old_basins.gage.to_list()
    
    # check for ones with streamflow for the app
    has_flow = []
    for gage in flow.columns:
        if flow[gage].notna().sum() >= 365:
            has_flow.append(gage)
    
    # remove other ones
    keep = set(keep).intersection(set(has_flow))
    gages = gages[gages['gage'].isin(keep)]
    old_basins = old_basins[old_basins['gage'].isin(gages.gage.to_list())]

    # delineate basins
    terrain_folder = os.path.join(ncwd, r'data\terrain')
    southDEM = os.path.join(terrain_folder, r'dem_ucol_south\south5070.tif')
    northDEM = os.path.join(terrain_folder, r'dem_ucol_north\north5070.tif')
    bigDEM = os.path.join(terrain_folder, r'dem_ucol_big\big5070.tif')
    dems = [southDEM, northDEM, bigDEM]

    # # grab bounding box of dems
    from shapely.geometry import box
    dem_geoms = {}
    for dem in dems:
        with rasterio.open(dem) as src:
            bounds = src.bounds  # Returns (left, bottom, right, top)
            crs = src.crs        # Coordinate Reference System
            geom = box(*bounds)
            dem_geoms[dem] = geom

    # 1. Create a GeoDataFrame for DEM bounding extents
    dem_gdf = gpd.GeoDataFrame(
        {"dem_name": ["south", "north"]},
        geometry=[dem_geoms[southDEM], dem_geoms[northDEM]],
        crs=5070,  # Set original CRS of DEMs
    ).to_crs(5070)

    # 2. Perform a spatial join checking containment
    joined = gpd.sjoin(
        old_basins.to_crs(5070), dem_gdf, how="left", predicate="within"
    )

    # drop ones that cover both
    joined = (
    joined.assign(_is_north=joined['dem_name'] == 'north')
    .sort_values(by='_is_north', ascending=False)
    .drop_duplicates(subset=['gage'], keep='first')
    .drop(columns=['_is_north'])
    )

    # 3. Map results back with fallback to 'big'
    joined = joined.set_index('gage').sort_index().to_crs(5070)
    old_basins = old_basins.set_index('gage').sort_index().to_crs(5070)
    gages = gages.set_index('gage').sort_index().to_crs(5070)
    old_basins["dem"] = joined["dem_name"].fillna("big50")
    gages["dem"] = joined["dem_name"].fillna("big50")

    # these gotta go with south
    redo_w_south = ['09118450', '09124500', '09146200', '09147000', '09147025', '09147500', '09149500']
    gages['dem'] = np.where(gages.index.isin(redo_w_south), 'south', gages['dem'])

    print('lees ferry is', gages[gages.index=='09380000']['dem'], 'keystone is:', gages[gages.index=='09047700']['dem'])

    import whitebox
    wbt = whitebox.WhiteboxTools()

    cats = []
    snap_threshold = 1000
    direction ='big50'
    directions = ['north', 'south', 'big50']
    missing_list = []
    for direction in directions:
        # filter to just this direction
        if not os.path.exists(os.path.join(terrain_folder, fr'dem_ucol_{direction}\merged_wsheds.shp')):
            print(f'starting {direction}')
            sub_gages = gages[gages['dem']==direction]
            print(f'doing {len(sub_gages)} gages for {direction}')
            # save as shp
            sg_path = os.path.join(terrain_folder, fr'dem_ucol_{direction}\pre_snap.shp')
            sub_gages.to_file(sg_path)
            # create filled, flow dir, and flow acc
            dem_path = os.path.join(terrain_folder, fr'dem_ucol_{direction}\{direction}5070.tif')
            print('filling DEM')
            filled_path = os.path.join(terrain_folder, fr'dem_ucol_{direction}\filled_dem.tif')
            filled_dem = wbt.fill_depressions(dem_path, output=filled_path, flat_increment=0.001)
            print('flow direction')
            flow_dir_path = os.path.join(terrain_folder, fr'dem_ucol_{direction}\flow_direction.tif')
            wbt.d8_pointer(filled_path, output=flow_dir_path)
            print('flow accumulation')
            flow_acc_path = os.path.join(terrain_folder, fr'dem_ucol_{direction}\flow_accum.tif')
            wbt.d8_flow_accumulation(filled_path, output=flow_acc_path, out_type='cells')
            print('extract streams')
            streams_path = os.path.join(terrain_folder, fr'dem_ucol_{direction}\streams.tif')
            wbt.extract_streams(flow_accum=flow_acc_path, output=streams_path,threshold=2000.0)
            print('snapping pour points')
            pp_path = os.path.join(terrain_folder, fr'dem_ucol_{direction}\snapped.shp')
            wbt.jenson_snap_pour_points(pour_pts=sg_path, streams=streams_path, output=pp_path, snap_dist=1000)
            pp = gpd.read_file(pp_path)
            pp = pp[['gage', 'geometry']]
            pp.to_file(pp_path)
            print('wshed delineation')
            wshed_raster_path = os.path.join(terrain_folder, fr'dem_ucol_{direction}\wshed_raster.tif')
            wbt.unnest_basins(d8_pntr=flow_dir_path, pour_pts=pp_path, output=wshed_raster_path)

            vector_cats = []
            for file in os.listdir(os.path.join(terrain_folder, fr'dem_ucol_{direction}')):
                if 'wshed_raster' in file:
                    path = os.path.join(terrain_folder, fr'dem_ucol_{direction}\{file}')
                    vector_name = file.replace('raster', 'vector')
                    vector_name = vector_name.replace('.tif', '.shp')
                    outpath = os.path.join(terrain_folder, fr'dem_ucol_{direction}\{vector_name}')
                else:
                    continue
                #if not os.path.exists(outpath):
                v = wbt.raster_to_vector_polygons(path, output=outpath)
                cats = gpd.read_file(outpath)
                print(len(cats))
                vector_cats.append(cats)

            # align value with the gage ID
            cats = pd.concat(vector_cats)
            cats['VALUE'] = cats['VALUE'].astype(int)
            cats = cats.set_index('VALUE')
            pp_path = os.path.join(terrain_folder, fr'dem_ucol_{direction}\snapped.shp')
            pp = gpd.read_file(pp_path)
            pp['VALUE'] = pp.index + 1
            pp = pp.set_index('VALUE')

            basins_dir = pd.merge(left=cats, right=pp[['gage']], left_index=True, right_index=True)
            basins_dir.geometry = basins_dir.geometry.to_crs(5070)
            basins_dir['geometry'] = basins_dir.geometry.simplify(tolerance=100)
            # save
            basins_dir.to_file(os.path.join(terrain_folder, fr'dem_ucol_{direction}\merged_wsheds.shp'))

    # concat the 3 directions
    bsns_list = []
    for direction in ['north', 'south']:
        basins_dir_path = os.path.join(terrain_folder, fr'dem_ucol_{direction}\merged_wsheds.shp')
        bsns = gpd.read_file(basins_dir_path)
        bsns['dem'] = direction
        bsns_list.append(bsns)
    bsns = pd.concat(bsns_list)
    # keep the duplicates from the south
    bsns = bsns.sort_values(by=['gage', 'dem'], ascending=[True, False])
    bsns = bsns.drop_duplicates(subset='gage', keep='first')
    bsns[bsns['gage']=='09146200'].explore() # check uncompagra
    print(f'{len(bsns)} basins from 30m North and South DEMs')

    # gunnison is messed up, drop them (they will be added back from the big DEM)
    use_big = ['09152500', '09144250']
    bsns = bsns[~bsns.gage.isin(use_big)].copy()

    bsns_big = gpd.read_file(os.path.join(terrain_folder, fr'dem_ucol_big50\merged_wsheds.shp'))
    bsns_big['dem'] = 'big50'

    # check what we are missing
    still_need = set(gages.index).difference(set(bsns['gage']))
    bsns_big = bsns_big[bsns_big['gage'].isin(still_need)]
    basins = pd.concat([bsns, bsns_big])
    print(f'Added {len(bsns_big)} basins from big DEM')

    # check what we are still missing
    still_need = set(gages.index).difference(set(basins['gage']))
    if len(still_need) > 0:
        nldi_fill = old_basins[old_basins.index.isin(still_need)]
        basins = pd.concat([bsns, nldi_fill])
        print(f'Added {len(nldi_fill)} basins from NLDI DEMs')

    print(f'{len(basins)} total basins, {len(gages)} total gages')

    # get basin area
    basins['area_m2'] = basins.area

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

    # add area km2
    basins['area_km2'] = basins['area_m2'] / 1000000

    # force gage index
    basins = basins.set_index('gage')

    # drop FID
    basins = basins.drop(columns=['FID'])
    
    # add name
    basins['name'] = gages.name

    # some of the manual delineations might be broken

    #basins_path = os.path.join(terrain_folder, f'dem_ucol_north\merged_wsheds.shp')
    #basins = gpd.read_file(basins_path).set_index('gage')
    #basins['area_m2'] = basins.area
    old_basins.set_index('gage', inplace=True)

    # send to file for inspection
    basins.to_file(os.path.join(terrain_folder, 'inspect_new_basins.shp'))
    basins.sort_index(inplace=True)
    for gage in basins.index:
        try:
            new_area = basins[basins.index==gage]['area_m2'].iloc[0]
            old_area = old_basins[old_basins.index==gage]['area_m2'].iloc[0]
            area_frac = new_area/old_area
            if area_frac > 1.1 or area_frac < 0.9:
                print(gage, f'{area_frac:.2f}')
        except:
            continue

    # use NLDI basins for these instead
    broken_basins = ['09019000', '09050700', '09027100', '09035700', '09036000', '09050100', '09110000', '09172500', '09330000','383926107593001']
    nldi_fill = old_basins[old_basins.index.isin(broken_basins)]
    print(f'geometry before:')
    basins[basins.index==broken_basins[0]]['geometry'].iloc[0]
    old_basins = old_basins[old_basins.index.isin(basins.index)]
    basins['geometry'] = np.where(basins.index.isin(broken_basins), old_basins['geometry'], basins['geometry'])
    print(f'geometry after:')
    basins[basins.index==broken_basins[0]]['geometry'].iloc[0]
    basins['dem'] = np.where(basins.index.isin(broken_basins), 'NLDI', basins['dem'])

    # simplify NLDI ones
    basins['geometry'] = basins.geometry.simplify(tolerance=100)
    


    basins = basins.sort_index()
    gages = gages.sort_index()

    # drop some columns
    basins = basins.drop(columns=['VALUE'])

    # make sure they look good
    basins[basins.index.isin(['09380000', '09330000', '09152500'])].explore()
    basins.explore()

    # Enforce that they are the same
    basins_path = os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet')
    gages_path = os.path.join(appcwd, r'spatial_data/all_UCOL_gages.parquet')
    basins_shp_path = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\spatial_data\all_UCOL_basins.shp'

    if all(basins.index == gages.index):
        basins.to_parquet(basins_path)
        basins.to_file(basins_shp_path)
        gages.to_parquet(gages_path)
        print(f'saved to {basins_path} and {gages_path}')
        lees_ferry = basins[basins.index=='09380000']
        lees_ferry.to_parquet(os.path.join(appcwd, r'spatial_data/UCOL.parquet'))
        #basins.to_parquet(os.path.join(ncwd, r'spatial_data/all_UCOL_basins.parquet'))
        #gages.to_parquet(os.path.join(ncwd, r'spatial_data/all_UCOL_gages.parquet'))

#### GRAB CLIMATE FORCING ####
basins_path = os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet')
gages_path = os.path.join(appcwd, r'spatial_data/all_UCOL_gages.parquet')
basins = gpd.read_parquet(basins_path)
gages = gpd.read_parquet(gages_path)
basins[basins.index.isin(['09380000', '09330000', '09152500'])].explore()

import pygridmet as gridmet
import rioxarray
from shapely.geometry import mapping

# smaller date range for this
date_range = ("2003-10-01", "2025-09-30")

# Variables requested from GRIDMET service
variables = ['pr', 'srad', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd', 'pet']

# Define aggregation types per variable
mean_vars = ['srad', 'rmax', 'rmin', 'tmmn', 'tmmx', 'vpd']
sum_vars = ['pet', 'pr']

# Run the SNODAS extraction ONCE up front
#############
gages_done = []
for fol in [1,2]:
    snodas_export_folder = fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\snodas_processed{fol}'
    df = pd.read_csv(os.path.join(snodas_export_folder, 'SNODAS_SWE_WY2025.csv'))
    gages_done = gages_done + df.columns.to_list()
        
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
basins = basins.to_crs(4326)

# prepare rename dict
rename_dict = {v: f"{v}_mean" for v in mean_vars}
rename_dict['time'] = 'date'
rename_dict.update({v: f"{v}_sum" for v in sum_vars})
appDir = os.path.join(appcwd, r'timeseries') # does not have each individual diversion
results = []
# Loop over each individual watershed

# run it just for the coords fallback fails
fails = basins[basins['data']=='coords fallback fail'].index.to_list()

for gage in fails: # basins.index to run all
    rd = {'gage':gage}
    gage = '09053500'
    outpath = fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\gr_snodas_new\{gage}.csv'

    # 2. Check for SNODAS data
    gage_swe_series = snodas_lookup[gage].dropna()

    # Isolate watershed geometry once
    single_gdf = basins[basins.index == gage]
    name = single_gdf['name'].iloc[0]
    areakm2 = single_gdf['area_km2'].iloc[0]

    # 3. Generate gridMET file (outpath) if missing
    if True: #not os.path.exists(outpath):
        print(f"downloading gridmet: {gage}...")

        if areakm2 < 8:
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
                gr_df['gridmet_method'] = 'coordinates'
                gr_df.to_csv(outpath, index_label='date')
                rd['data'] = 'coords'
            except Exception as e:
                print(f"Failed getting coords for {gage}: {e}")
                rd['data'] = 'failed by coords'
                

        elif areakm2 < 25000:
            print(f"area: {areakm2:.2f} km2, getting by geom")
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
                gr_df['gridmet_method'] = 'geometry'
                gr_df.to_csv(outpath, index_label='date')
                rd['data'] = 'geometry'
            except Exception as e:
                print(f'{gage} error during get_bygeom: {e}')
                continue
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
                        gr_df['gridmet_method'] = 'coordinates'
                        gr_df.to_csv(outpath, index_label='date')
                        rd['data':'coords fallback']
                    except Exception as fallback_e:
                        print(f"Fallback failed for {gage}: {fallback_e}")
                        rd['data'] = 'coords fallback fail'
                else:
                    print(f"{gage} {name} too big (memory allocation issue)")
                    rd['data'] = 'too big fail'
        else:
            print(f"{gage} {name} too big, didn't try")
            rd['data'] = 'too big no try'

    results.append(rd)

rdf = pd.DataFrame().from_dict(results)
rdf = rdf.set_index('gage')

basins['data'] = rdf['data']

#############################
# MERGE DIVERSIONS WITH STREAMFLOW
##############################
###################
# SKIP STEPS BY READING HERE
def fill_discharge_data(df: pd.DataFrame, col_names: tuple = ("Q_cms")) -> pd.DataFrame:
    """Fills missing values in a streamflow time series based on flow-rate and time gap thresholds.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with a DatetimeIndex and the target column.
    col_name : str
        Name of the discharge column.

    Returns:
    --------
    pd.DataFrame
        Dataframe with updated missing values and a boolean flag column.
    """
    df = df.copy()

    # Calculate 20% of the mean of valid measurements
    for col_name in col_names:
        q_mean = df[col_name].mean()
        low_flow_threshold = 0.20 * q_mean

        # Identify original non-NaN values and their positions
        is_valid = df[col_name].notna()

        # Track original start and end valid measurements (Rule 2)
        first_valid_idx = is_valid.idxmax() if is_valid.any() else None
        last_valid_idx = is_valid.idxmin() if is_valid.any() else None
        # Precise last index with valid data
        last_valid_idx = df[col_name].dropna().index[-1]

        # Create the helper columns: last valid value and last valid timestamp
        prev_val = df[col_name].ffill()
        last_valid_time = df.index.to_series().where(is_valid).ffill()

        # Calculate days elapsed since the last valid measurement
        days_since_last = (df.index - last_valid_time).dt.total_seconds() / (
            24 * 3600
        )

        # Condition 1: Low-flow (< 20% of mean) and gap < 120 days
        is_low_flow = prev_val < low_flow_threshold
        cond_low_flow = is_low_flow & (days_since_last < 120)

        # Condition 2: Regular/High-flow (>= 20% of mean) and gap < 7 days
        cond_high_flow = (~is_low_flow) & (days_since_last < 7)

        # Combined fill condition (excluding original valid points and outer edges)
        fill_mask = (
            df[col_name].isna()  # Only fill missing data
            & (cond_low_flow | cond_high_flow)  # Respect low/high flow gap rules
            & (df.index > first_valid_idx)  # Rule 2: Ignore leading edge
            & (df.index < last_valid_idx)  # Rule 2: Ignore trailing edge
        )

        # Apply fills and create flag column
        df[f"{col_name}_is_filled"] = fill_mask
        df.loc[fill_mask, col_name] = prev_val[fill_mask]

    return df

def glover_unit_response(t_days, a_ft, T_sqft_day, S):
    """
    Calculates the daily fraction of a discrete recharge pulse returning 
    to the stream at time t using the Glover analytical derivative.
    
    a_ft        : Distance from field/recharge area to stream (ft)
    T_sqft_day  : Aquifer Transmissivity (ft^2/day)
    S           : Specific Yield (dimensionless, e.g., 0.1 to 0.2)
    """
    t = np.asarray(t_days, dtype=float)
    q = np.zeros_like(t)
    valid = t > 0
    D = T_sqft_day / S
    t_v = t[valid]
    
    # Instantaneous unit response function q(t)
    q[valid] = (a_ft / (2 * np.sqrt(np.pi * D * (t_v**3)))) * np.exp(-(a_ft**2) / (4 * D * t_v))
    return q

reservoir_gdf = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\all_UCOL_reservoirs.parquet").to_crs(epsg=4326)
reservoir_gdf = reservoir_gdf[reservoir_gdf.areasqkm > 3]
basins_path = os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet')
gages_path = os.path.join(appcwd, r'spatial_data/all_UCOL_gages.parquet')
basins = gpd.read_parquet(basins_path)
basins.geometry = basins.geometry.to_crs(4326) # need this for spatial joins
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
# interpolate in the same way you interpolate the in the loop below. This is necessary for data quality check in the inflows calculation
flow = fill_discharge_data(flow, col_names=flow.columns)


name_dict = gages['name'].to_dict()
date_range = ("1979-10-01", "2025-09-30")
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
dvrs_intra['siteName'] = dvrs_intra['siteName'] + '_delivery_point'
# add rows for the transbasin
dvrs = pd.concat([dvrs, dvrs_intra])
# need to rebuild the geometry after this
dvrs = df_to_geodataframe(dvrs, lat_col='decLat', lon_col='decLong')
dvrs.to_parquet(os.path.join(appcwd, r"spatial_data\ucrb_diversion_master_table.parquet"))

# RESERVOIR EVAP
evap = gpd.read_parquet(r"N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\evap\ucol_reservoirs.parquet")
evap['RES_NAME'] = evap['RES_NAME'].str.replace(' ', '_')
evap_joined = gpd.sjoin(evap, basins, how="inner", predicate="within")

# WATERSHEDS
# gageID column = gage
# WIDE CSV FOR DIVERSION DATA IN CFS
# COLUMN HEADERS = siteID

# 1980 to 2025
dvrsFlow = pd.read_csv(os.path.join(ncwd, r"data\diversion\will_processed\combined_diversion_records_filtered_filled_cfs_fill_years.csv"))
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
joined = gpd.sjoin(dvrs, basins, how="inner", predicate="within")

import time
current_time = time.time()
seconds_in_1_hours = 60 * 60
# send to csvs
basins_subset = ['09211200', '09209400', '09205000', '09188500', '09210500', '09201500', '09196500', '09195000']
basins_subset = ['09041090', '09041400']
basins_subset = ['09034250', '09010500', '09014050', '09015000', '09019000', '09019500', '09021000', '09022000', '09024000', '09025000', '09025300', '09026500', '09027100', '09032000', '09032050', '09032100', '09032200', '09032300', '09032400', '09032990', '09033100', '09033300', '401530105475401']
basins_sub = basins[basins.index.isin(basins_subset)].sort_values('area_m2')
for gage in basins_sub.index:
    #gage = '09034250'
    out_path_small = os.path.join(appDir, f"{gage}.csv")
    out_path = os.path.join(wdvrsDir, f"{gage}.csv")
    
    # # Check if the file exists first
    # if os.path.exists(out_path_small):
    #     df = pd.read_csv(out_path_small, parse_dates=['date'], index_col='date')
    #     df_cleaned = df[~df.index.duplicated(keep='first')]
    #     df_cleaned = df_cleaned.asfreq('D')
    #     df_cleaned.to_csv(out_path_small, index_label='date')

    # if os.path.exists(out_path):
    #     df = pd.read_csv(out_path, parse_dates=['date'], index_col='date')
    #     df_cleaned = df[~df.index.duplicated(keep='first')]
    #     df_cleaned = df_cleaned.asfreq('D')
    #     df_cleaned.to_csv(out_path, index_label='date')

    df = flow[[gage]]
    df = df.rename(columns={gage:'Q_cms'})

    # INTERPOLATE MISSING Q
    df = fill_discharge_data(df, col_names=['Q_cms'])

    df['Q_cfs'] = df['Q_cms'] * 35.3147
    # add area and mmd
    area = basins[basins.index==gage]['area_m2'].iloc[0]
    df['Q_mmd'] = (df['Q_cms'] * 86400000) / area
    df['area_m2'] = area
    # make sure it's got every day
    df = df.asfreq('D')
    # add name and gage as columns
    df['name'] = name_dict[gage]
    df['gage'] = gage
    # filter to 1980 and up
    df = df[df.index.year>1980].copy()
    
    # Identify siteIDs for diversions located in this specific watershed
    target_diversions = joined[joined['gage'] == gage]['siteID'].unique()
    target_evaps = evap_joined[evap_joined['gage']==gage]['RES_NAME'].unique()
    
    frac_in_cols = []
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

        # do it for each unit
        units = ['cfs', 'mmd', 'cms']
            
        if valid_cols:
            # 1. Extract the raw data
            subset_dvrs = dvrsFlow[valid_cols].copy()
            # 2. Convert raw diversions to Consumptive Use (CU)
            useDict = {
                'irrigation': -0.6, 'municipal': -0.3, 'interbasin': -1, 
                'industrial': -1, 'hydropower': 0, 'intrabasin': -1, 'transbasin': 1,
            }

            # INTERBASIN = all water leaves UCOL
            # INTRABASIN = water may or may not leave sub-basin. Does not leave UCOL.
            # TRANSBASIN = water imported from a different sub-basin within UCOL.
            
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
        
        # get evaps
        if len(target_evaps) > 0:
            dfs = []
            for res in target_evaps:
                evap = pd.read_parquet(fr"N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\evap\{res}.parquet")
                evap['date'] = pd.to_datetime(evap['start_date'])
                evap = evap.set_index('date')
                evap_col = f'evap_{res}'
                evap = evap.rename(columns={'E_volume (m3)': evap_col})[[evap_col]]
                dfs.append(evap)
        
            evap_df = pd.concat(dfs, axis=1)

            # convert to negative
            mmd_scale = (2446575.5461 / area) * -1 # goes from cfs to mmd
            CONVERSION_FACTORS = {
                'cms': -1 / 86400,
                'cfs': -35.3146667 / 86400,
                'mmd': mmd_scale,
            }

            # Convert the whole DataFrame to cfs
            evap_df = pd.concat(
                [evap_df.mul(CONVERSION_FACTORS[u]).add_suffix(f'_{u}') for u in units],
                axis=1
            )
            # sum total evap
            for unit in units:
                unit_cols = evap_df.filter(like=f'_{unit}').columns
                evap_df[f'evap_{unit}'] = evap_df[unit_cols].sum(axis=1)

            # merge diversions and evaporation
            subset_dvrs = pd.merge(left=subset_dvrs, right=evap_df, how='left', left_index=True, right_index=True)

            # add evaporation to Q_CU
            for unit in units:
                subset_dvrs[f'Q_CU_{unit}'] = subset_dvrs[f'Q_CU_{unit}'] + subset_dvrs[f'evap_{unit}']

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
            
        ###### return flow proof of concept
        if 'irrigation_cfs' in combined_df.columns:

            combined_df['irrigation_diversion_cfs'] = combined_df['irrigation_cfs'] / useDict['irrigation'] * -1 # convert CU to diversion
            combined_df['irrigation_rf_cfs'] = combined_df['irrigation_diversion_cfs'] * (1-useDict['irrigation']*-1)  # convert diversion to return flow
            # diversion should equal return flow + consumptive use
            #test = combined_df['irrigation_diversion_cfs'].abs().sum() / (combined_df['irrigation_rf_cfs'].abs().sum() + combined_df['irrigation_cfs'].abs().sum())

            # 1. Define physical aquifer parameters
            a = 1000.0       # Distance to stream in feet
            T = 5000.0       # Transmissivity in ft^2/day
            S = 0.20         # Specific yield

            # 2. Generate Glover response kernel (e.g., 5-year lag window)
            max_lag_days = 365 * 5
            t_axis = np.arange(max_lag_days)
            kernel = glover_unit_response(t_axis, a, T, S)

            # Normalize kernel so total mass sums to 1.0
            if np.sum(kernel) > 0:
                kernel = kernel / np.sum(kernel)

            # 3. Convolve daily unconsumed return flows with the Glover kernel
            rf_series = combined_df['irrigation_rf_cfs'].fillna(0).values
            lagged_rf = np.convolve(rf_series, kernel, mode='full')[:len(combined_df)]

            # 4. Compute estimated Natural Streamflow
            combined_df['lagged_rf_cfs'] = lagged_rf
            # first undo the previous step which added irrigation CU to create natural flow
            combined_df['Q_NAT_noag'] = combined_df[f'Q_NAT_cfs'] + combined_df[f'irrigation_cfs']
            # now subtract diversions and add return flow
            combined_df['Q_NAT2_cfs'] = combined_df['Q_NAT_noag'] - combined_df['irrigation_diversion_cfs'] + combined_df['lagged_rf_cfs']

            fig, ax = plt.subplots()
            df2 = combined_df.loc['2015-10-01':'2016-10-01']
            ax.plot(df2.index, df2['Q_cfs'], label='observed Q')
            ax.plot(df2.index, df2['lagged_rf_cfs'], label='lagged return flow')
            ax.plot(df2.index, df2['Q_NAT2_cfs'], label='natural Q', linestyle='--')
            ax.plot(df2.index, df2['irrigation_cfs'], label='irrigation')
            ax.plot(df2.index, df2['Q_NAT_cfs'], label='observed + CU Q')
            ax.legend()
            plt.show()

        ##### reservoir naturalization proof of concept #####
        basin = basins[basins.index==gage]
        # is this basin impacted by a major reservoir(s)?
        basin_res = gpd.sjoin(basin, reservoir_gdf, how='inner', predicate='intersects')
        # does this basin have a decent data record?
        days = combined_df['Q_cfs'].notna().sum() # need 10 years of data
        if len(basin_res) > 0 and days > 3650: 
            print('processing inflows')
            # find the largest non-overlapping sub-basins within this basin (use basins gdf)
            target_geom = basin.geometry.iloc[0]
            contained_basins = basins[(basins.index != gage) & (basins.centroid.within(target_geom))].copy()
            contained_basins = contained_basins[contained_basins['area_m2'] < area] # using the centroid can put big basins inside small basins
            # remove basins with bad data: 
            contained_gages = contained_basins.index.unique().to_list()
            gages = contained_gages + [gage]
            flow_subset = flow[gages]
            flow_subset = flow_subset[flow_subset.index.year > 1999].copy() # I don't care if we are missing a lot of early data

            # Filter mask for rows where the target column is valid (not NA)
            valid_target_mask = flow_subset[gage].notna()
            total_valid_target = valid_target_mask.sum()

            # Calculate the overlapping valid percentage for each gauge
            keep = []
            for col in contained_gages:
                common_valid_count = (valid_target_mask & flow_subset[col].notna()).sum()
                pct_overlap = (
                    (common_valid_count / total_valid_target) * 100
                    if total_valid_target > 0
                    else 0
                )
                print(col, pct_overlap)
                if pct_overlap > 90: # 90% overlap so we don't have cascading nans
                    keep.append(col)
                
            contained_basins = contained_basins[contained_basins.index.isin(keep)]
            # find the area of the total watershed covered
            if not contained_basins.empty:

                # Sort by area descending so we evaluate larger sub-basins first
                contained_basins = contained_basins.sort_values(by='area_m2', ascending=False)
                
                # 2. Iteratively select non-overlapping sub-basins
                selected_subbasins = []
                union_geom = None
                
                for idx, subbasin in contained_basins.iterrows():
                    geom = subbasin.geometry
                    # Check if this sub-basin overlaps significantly with already selected ones
                    if union_geom is None:
                        selected_subbasins.append(idx)
                        union_geom = geom
                    elif not geom.overlaps(union_geom) and not geom.within(union_geom):
                        # Alternatively, use area intersection check if geometries slightly touch/overlap edges:
                        if geom.intersection(union_geom).area / geom.area < 0.01:
                            selected_subbasins.append(idx)
                            union_geom = union_geom.union(geom)

                # GeoDataFrame of the largest non-overlapping sub-basins
                non_overlapping_gdf = basins.loc[selected_subbasins]

                subbasins_union = non_overlapping_gdf.unary_union
                missing_geom = target_geom.difference(subbasins_union)
                missing_area_gdf = gpd.GeoDataFrame({'gage': [gage], 'geometry': [missing_geom]}, crs=basins.crs)
                missing_area_gdf.to_file(fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\spatial_data\missing_geometries\{gage}.shp')

                # 3. Find the area of the total watershed covered
                # Expressed in the CRS units (e.g., m² or km² depending on projection)
                covered_area = union_geom.area
                
                # Percentage of the main target basin covered
                basin_area = basin.geometry.iloc[0].area
                frac_covered = covered_area / basin_area
            else:
                covered_area = 0.0
                frac_covered = 0.0

            inflow_gages = non_overlapping_gdf.index.to_list()
            inflows = pd.DataFrame()
            for igage in inflow_gages:
                idf = os.path.join(appDir, f"{igage}.csv")
                idf = pd.read_csv(idf, parse_dates=['date'], index_col='date')
                inflows[igage] = idf['Q_NAT2_cfs'] 
            # sum all the natural inflows
            inflows['inflow_sum'] = inflows[inflow_gages].sum(axis=1, skipna=False)
            inflows = inflows[['inflow_sum']].dropna()
            overlap_dates = inflows.index.intersection(combined_df[['Q_NAT2_cfs']].index)

            # compare total inflow vs total outflow for the period of record
            frac_in = inflows.loc[overlap_dates]['inflow_sum'].sum() / combined_df.loc[overlap_dates]['Q_NAT2_cfs'].sum()
            combined_df['frac_inflow'] = frac_in
            combined_df['frac_gaged_inflow_area'] = frac_covered

            if frac_in < 1:
                # assume the unaccounted area had similar temporal behavior, add it back in  
                scale = 1+(1-frac_in)
                combined_df['Q_NAT2_cfs'] = inflows['inflow_sum'] * scale

            if frac_in > 1: # the reservoir is losing a lot of water to seepage or unaccounted evap. Ignore that.
                combined_df['Q_NAT2_cfs'] = inflows['inflow_sum']
            
            print(f'inflow gages: {inflow_gages}, frac_in {frac_in}, % basin gaged {frac_covered}')
            frac_in_cols = ['frac_inflow','frac_gaged_inflow_area']

    else:
        # If no diversions found, we still save the original flow (or skip)
        combined_df = df
        combined_df['Q_NAT2_cfs'] = combined_df['Q_cfs']
        combined_df['Q_NAT_cfs'] = combined_df['Q_cfs']
    
    combined_df = combined_df.asfreq('D')
    # 4. Save the new CSV
    combined_df.to_csv(out_path, index_label='date')

    # make a smaller df without all the diversions and interpolation
    cu_types2 = []
    cu_types = list(cu_types) + ['evap']
    for unit in units:
        for cu_type in cu_types:
            cu_types2.append(f'{cu_type}_{unit}')
    columns = [col for col in combined_df.columns if 'Q' in col or col in cu_types2]
    # I wanna look at frac in
    columns += frac_in_cols
    small_df = combined_df[columns]
    out_path_small = os.path.join(appDir, f"{gage}.csv")
    small_df.to_csv(out_path_small, index_label='date')

    print(f"Processed gage {gage}: Added {len(target_diversions)} diversion columns.")

#############
# Merge with flow with 0 interpolation
#############
for gage in basins.index:

    inpath = fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\gr_snodas_new\{gage}.csv'
    outpath2 = fr'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\gr_snodas_flow0\{gage}.csv'

    if os.path.exists(inpath): #os.path.exists(outpath) and not os.path.exists(outpath2):
        print(f'merging flow for {gage}...')

        gr_df = pd.read_csv(inpath, parse_dates=['date'], index_col='date')
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
# check for data completeness and prepare for modelling
cwd = os.getcwd()
ncwd = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow'
appcwd = os.path.join(cwd, r'shiny-app\ucol_natural')
gcwd = r'G:\My Drive\natural_streamflow_colab' # where to save things on your machine
dcwd = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\gr_snodas_flow0'
basins_path = os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet')
gages_path = os.path.join(appcwd, r'spatial_data/all_UCOL_gages.parquet')
basins = gpd.read_parquet(basins_path)
gages = gpd.read_parquet(gages_path)

# get attributes
attrs_path = os.path.join(r"G:\My Drive\colab_out\attributes\all_UCOL_attributes.csv")
attrs = pd.read_csv(attrs_path)
attrs.rename(columns={'gauge_id':'gage'}, inplace=True)
attrs['gage'] = attrs['gage'].apply(fix_gage_id)
attrs = attrs.set_index('gage')

# get reservoirs
res = gpd.read_parquet(os.path.join(appcwd, fr'spatial_data/all_UCOL_reservoirs.parquet')).to_crs(basins.crs)
res['gt_05'] = res['areasqkm'] > 0.5
joined = gpd.sjoin(basins, res[["areasqkm", "gt_05", "geometry"]], how="left", predicate="intersects")
# Group by the joined index (which matches the original basins index)
agg_df = joined.groupby(joined.index).agg(reservoirs=("areasqkm", "count"), res_sqkm=("areasqkm", "sum"), res_gt_05=("gt_05", "sum"))
# Assign the calculated totals back to the original basins GeoDataFrame
basins[["reservoirs", "res_sqkm", "res_gt_05"]] = agg_df[["reservoirs", "res_sqkm", "res_gt_05"]]
# Fill NaN values for basins that contain no reservoirs
basins["reservoirs"] = basins["reservoirs"].fillna(0).astype(int)
basins["res_sqkm"] = basins["res_sqkm"].fillna(0.0)
basins["res_gt_05"] = basins["res_gt_05"].fillna(0).astype(int)
basins.sort_values(by='area_km2', inplace=True)

# add names to attrs
attrs['name'] = basins.name
name_col = attrs.pop("name")
attrs.insert(0, "name", name_col)
attrs['gage'] = attrs.index
attrs.reset_index(drop=True, inplace=True)
attrs.to_parquet(os.path.join(appcwd, r'attributes\all_UCOL_attributes.parquet'))
attrs.to_parquet(os.path.join(gcwd, r'timeseries\basinCharacteristics.parquet'))
attrs = pd.read_parquet(os.path.join(appcwd, r'attributes\all_UCOL_attributes.parquet'))

units = ['cfs', 'mmd', 'cms']
Q_cols = []
for unit in units:
    Q_cols.append(f'Q_{unit}')
    Q_cols.append(f'Q_NAT_{unit}')
    Q_cols.append(f'Q_CU_{unit}')

results = []
Q_col = 'Q_cfs'
for gage in basins.index.to_list():

    print(gage)
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
    df = df.loc[first_valid:last_valid].copy()
    df = df.asfreq('D')
    period_length = len(df)

    if period_length<1:
        rd['data'] = False
        continue

    # calculate mean vars
    variables = [Q_col, 'Q_mmd', 'pr_sum', 'swe_sum', 'pet_sum', 'tmmx_mean', 'tmmn_mean']
    for var in variables:
        rd[f'{var}_mean'] = df[var].mean()

    # check for NAs
    NA_ratio = df[Q_col].isna().sum() / len(df)
    rd['NA_ratio'] = NA_ratio

    # check length
    rd['firstday'] = first_valid
    rd['lastday'] = last_valid
    rd['period'] = period_length

    # CONSUMPTIVE USE
    CU_vars = ['irrigation' , 'municipal', 'intrabasin', 'interbasin', 'industrial', 'evap']
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
    rd['cu_frac'] = divert

    try:
        evapmean = df[f'evap_cfs'].mean()
        rd['mean_daily_evap_cfs'] = evapmean
    except:
        rd['mean_daily_evap_cfs'] = 0

    # Ensure Q_NAT_cfs exists; if missing, mirror Q_cfs
    for Q in Q_cols:
        if 'CU' not in Q and 'NAT' not in Q:
            Q_actual = Q_col
        if Q not in df.columns and 'NAT' in Q: # make natural columns match observed
            df[Q] = df[Q_actual]
        if Q not in df.columns and 'CU' in Q: # make CU columns zero
            df[Q] = 0

    # Track missing Q values prior to interpolation
    q_was_na = df[Q_col].isna()

    # Interpolate Q columns (limit of 30 days)
    limit=14
    df[Q_cols] = df[Q_cols].ffill(limit=limit)


    # Flag days where any Q column was successfully filled via interpolation
    q_is_now_valid = df[Q_col].notna()
    df['is_Q_interpolated'] = np.where(q_is_now_valid & q_was_na, True, False)

    # Interpolate all remaining numeric columns (limit of 30 days)
    other_cols = [col for col in df.columns if col not in Q_cols and col != 'is_Q_interpolated']
    df[other_cols] = df[other_cols].interpolate(method='time', limit=limit)

    # Calculate metrics
    num_interpolated_days = df['is_Q_interpolated'].sum()
    frac_interpolated = num_interpolated_days / period_length if period_length > 0 else 0
    
    # Days with complete Q data after interpolation
    days_with_data_after = df[Q_cols].notna().all(axis=1).sum()
    frac_data_after = days_with_data_after / period_length if period_length > 0 else 0


    cols = ['date', 'gage', 'srad_mean', 'rmax_mean', 'rmin_mean', 'tmmn_mean', 'tmmx_mean', 'vpd_mean', 'pet_sum', 'pr_sum', 'swe_sum', 'Q_cms', 'Q_cfs',
       'Q_mmd', 'Q_CU_cfs', 'Q_CU_cms', 'Q_CU_mmd', 'Q_NAT_cfs', 'Q_NAT_cms', 'Q_NAT_mmd', 'is_Q_interpolated']
    df['gage'] = gage
    df['date'] = df.index
    df = df[cols]
    rm = set(['gage', 'date', 'is_Q_interpolated'])
    cols_to_null = list(set(cols) - rm)
    df.loc[df['Q_cfs'].isna(), cols_to_null] = np.nan

    df.reset_index(drop=True, inplace=True)

    # Store results
    wys = count_complete_water_years(df, 'Q_cfs')

    rd.update({
        'period_start': first_valid,
        'period_end': last_valid,
        'num_interpolated_days': num_interpolated_days,
        'frac_interpolated_days': frac_interpolated,
        'frac_data_after_interp': frac_data_after,
        'wys': wys
    })

    results.append(rd)

    save = False
    if save:
        df.to_parquet(os.path.join(gcwd, rf'timeseries\{gage}.parquet'))

rdf = pd.DataFrame().from_dict(results)

# marge with other attributes
rdf = rdf.set_index('gage')
rdf = pd.merge(left=basins, right=rdf, left_index=True, right_index=True)
# recalculate area
rdf['area_m2'] = rdf.area
rdf['area_km2'] = rdf['area_m2'] * 10E-6
# merge with the attributes
attrs = pd.read_parquet(os.path.join(appcwd, r'attributes\all_UCOL_attributes.parquet'))
attrs_vars = ['dor_pc_pva', 'rev_mc_usu', 'dis_m3_pyr']
attrs = attrs.set_index('gage')
rdf = pd.merge(left=rdf, right=attrs[attrs_vars], left_index=True, right_index=True)
rdf['dor_pc_pva'] = rdf['dor_pc_pva'] / 1000

############# BASIN SELECTION SCHEME ##################
rdf = rdf[rdf.data] # gotta have streamflow
rdf = rdf[rdf.wys>5] # gotta have at least 5 good wys

# fix taylor park reservoir
tp = '09109000'
meanq = rdf[rdf.index==tp]['Q_cfs_mean'].iloc[0] * 31556926 # mean cfs * seconds in a year = mean yearly Q in cubic feet
damstorage = 106200 * 43559.9 # acre feet storage to cubic feet
dor = damstorage / meanq
rdf['dor_pc_pva'] = np.where(rdf.index==tp, dor, rdf['dor_pc_pva'])

# calculate the fraction of watershed area covered by reservoirs
rdf['res_area_frac'] = rdf['res_sqkm'] / rdf['area_km2']
# fix nas
rdf['cu_frac'] = rdf['cu_frac'].fillna(0)
rdf['dor_pc_pva'] = rdf['dor_pc_pva'].fillna(0)

# add a score for regulation
# Normalize values strictly between 0 and 1
cu_norm = (rdf['cu_frac'] - rdf['cu_frac'].min()) / (rdf['cu_frac'].max() - rdf['cu_frac'].min())
res_norm = (rdf['res_area_frac'] - rdf['res_area_frac'].min()) / (rdf['res_area_frac'].max() - rdf['res_area_frac'].min())

# Apply weights (e.g., 0.8 / 0.2 split)
rdf['reg_score'] = (0.8 * cu_norm) + (0.2 * res_norm)

basin_sum_path = os.path.join(ncwd, 'basin_summarys.parquet')
rdf.to_parquet(basin_sum_path)
### skip some steps by reading here
rdf = pd.read_parquet(basin_sum_path)


################# Nested matrix ##################
# 1. Ensure a projected CRS for accurate area calculations (reproject if geographic)
if rdf.crs is not None and rdf.crs.is_geographic:
    basins_proj = rdf.to_crs(basins.estimate_utm_crs())
else:
    basins_proj = rdf.copy()

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

np.fill_diagonal(nested_matrix.values, True) # basins are nested with themselves
# ==========================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==========================================
TEST_SIZE = 10
TRAIN_SIZE = 50
NUM_TRAIN_SETS = 10

MAX_ATTEMPTS = 50
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
rdf['nesting_degree'] = rdf.index.map(nesting_counts)

# Filter pristine watersheds
pristine_pool = rdf[(rdf['reg_score'] == 0)].copy()

# Sort pristine pool: prioritize large area (descending) and low nesting degree (ascending)
pristine_pool = pristine_pool.sort_values(
    by=['nesting_degree'], 
    ascending=[True]
)
print(f'pristine basins: {len(pristine_pool)}')
pristine_pool.iloc[0:15][['name', 'geometry']].explore()
# # Mannually pick the test set
# for gage in pristine_pool.iloc[0:15].index:
#     df = pd.read_csv(os.path.join(appDir, f'{gage}.csv'), parse_dates=['date'], index_col='date')
#     plot_hydrograph(df, 'Q_cfs', gage)

# experiment 1 test gages
# test_gages = [
#     '09266500', 
#     '09210500', 
#     '09223000', 
#     '09312600',
#     '09310700', 
#     '09253000', # slater
#     '383926107593001', 
#     '09123450', 
#     '09217900',
#     '09081600']

test_gages = ['09081600', '09217900', '09123450', '09312600', '09210500', '09266500', '09223000']
test_set = rdf[rdf.index.isin(test_gages)].copy()
target_mean_area = test_set['area_km2'].median()
target_mean_Q = test_set['Q_cfs_mean'].median()

print(f"=== TEST SET SELECTED ({len(test_set)} gages) ===")
print(f"Mean Area: {target_mean_area:.2f} km², mean Q: {target_mean_Q:.2f}")
print(test_set['name'])
test_set[['name', 'geometry']].explore()
# ==========================================
# 4. EXCLUDE TEST GAGES & ALL NESTED RELATIVES
# ==========================================
# Find all gages nested with ANY test set gage
test_and_nested_mask = nested_matrix.loc[test_gages].any(axis=0)
blocked_gages = nested_matrix.columns[test_and_nested_mask].tolist()

# Define candidate pool purely isolated from the test set
train_eligible_rdf = rdf[~rdf.index.isin(blocked_gages)].copy()

print(f"\nTotal Watersheds: {len(rdf)}")
print(f"Blocked (Test + Nested with Test): {len(blocked_gages)}")
print(f"Eligible Training Watersheds: {len(train_eligible_rdf)}")

# Pristine watersheds excluding test set and any watersheds nested with test set
pristine_pool_rm = pristine_pool[~pristine_pool.index.isin(blocked_gages)].copy()
# print(f'pristine options left {len(pristine_pool_rm)}')
# print(f"pristine mean Q: {pristine_pool_rm['Q_cfs_mean'].mean()}, mean area {pristine_pool_rm['area_km2'].mean()}")

# Pool of all candidate modified watersheds
modified_pool = train_eligible_rdf[rdf['reg_score'] > 0].copy()

# Standardize feature scales for distance matching (Area & Flow)
match1 = 'area_km2'
match2 = 'pr_sum_mean'
mean_q, std_q = rdf[match1].mean(), rdf[match1].std()
mean_a, std_a = rdf[match2].mean(), rdf[match2].std()

def calc_distance(df1, df2):
    """Calculates standardized Euclidean distance in (Q_cfs_mean, area_km2) space."""
    q_diff = (df1[match1] - df2[match1]) / std_q
    a_diff = (df1[match2] - df2[match2]) / std_a
    return np.sqrt(q_diff**2 + a_diff**2)

# ==========================================
# 2. CONTINUOUS INCREMENTAL SWAPPING ALGORITHM
# ==========================================
training_sets = {}

# add 5 of the testing gages back to every training set.
five_unseen = test_gages # ['09081600', '09266500', '09253000', '09312600', '09223000']
five_seen = list(set(test_gages) - set(five_unseen))
five_seen_df = test_set[test_set.index.isin(five_seen)]

# Set 0: Purely pristine starting baseline
current_gages = pristine_pool_rm.index.tolist()
train_df = rdf[rdf.index.isin(current_gages)].copy()
train_df = pd.concat([train_df, five_seen_df]) # add 5 back
train_df['set'] = 0
training_sets['train_set_0'] = train_df

train_set0 = training_sets['train_set_0']
train_set0[['geometry', 'name']].explore()

used_modified_gages = set()

NUM_STEPS = 10
REPLACEMENTS_PER_STEP = 10

print(f"TEST SET: Mean Area: {target_mean_area:.2f} km², mean Q: {target_mean_Q:.2f}")
for step in range(0, NUM_STEPS + 1):

    if step == 0:
        num_pristine = (train_df['reg_score'] == 0).sum()
        num_mod = len(train_df) - num_pristine
        mean_q_val = train_df['Q_cfs_mean'].median()
        mean_a_val = train_df['area_km2'].median()
        mean_reg = train_df['reg_score'].median()
        days = int(train_df['period'].sum())
        
        print(f"Train Set {step:02d} | Pristine: {num_pristine:02d} | Modified: {num_mod:02d} | Days: {days} | "
            f"Mean Reg Score: {mean_reg:.3f} | Mean Q: {mean_q_val:.1f} cfs | Mean Area: {mean_a_val:.1f} km²")
        continue

    replacements_made = 0
    
    # Identify internal nesting conflicts within current_gages
    sub_matrix = nested_matrix.loc[current_gages, current_gages].values.copy()
    np.fill_diagonal(sub_matrix, False)
    nesting_counts = pd.Series(sub_matrix.sum(axis=1), index=current_gages)
    
    # SORT CANDIDATES TO REMOVE FROM CURRENT SET:
    # 1. Nesting conflicts first (highest count)
    # 2. Lowest regulation score (pristine first, then least-modified)
    # 3. Largest area
    current_df = rdf[rdf.index.isin(current_gages)].copy()
    current_df['nest_count'] = current_df.index.map(nesting_counts)
    
    removal_candidates = current_df.sort_values(
        by=['nest_count', 'reg_score', 'area_km2'],
        ascending=[False, True, False]
    ).index.tolist()

    for target_gage in removal_candidates:
        if replacements_made >= REPLACEMENTS_PER_STEP:
            break
            
        target_row = rdf[rdf.index == target_gage].iloc[0]
        
        # Filter available modified gages that have higher reg_score than target_row
        # (Ensures the set becomes progressively more modified over time)
        avail_mod = modified_pool[
            (~modified_pool.index.isin(used_modified_gages)) &
            (modified_pool['reg_score'] >= target_row['reg_score'])
        ].copy()
        
        # If no strictly higher modified gages exist, fallback to any unused modified gage
        if avail_mod.empty:
            avail_mod = modified_pool[~modified_pool.index.isin(used_modified_gages)].copy()
            if avail_mod.empty:
                break  # Exhausted candidate pool
            
        # Match by nearest (Q_cfs_mean, area_km2) distance
        avail_mod['dist'] = calc_distance(avail_mod, target_row)
        avail_mod = avail_mod.sort_values('dist')
        
        temp_set = [g for g in current_gages if g != target_gage]
        
        # Find best candidate with ZERO spatial nesting in the remaining set
        selected_mod_gage = None
        for m_gage in avail_mod.index:
            if not nested_matrix.loc[temp_set, m_gage].any():
                selected_mod_gage = m_gage
                break
                
        # Perform swap
        if selected_mod_gage is not None:
            current_gages.remove(target_gage)
            current_gages.append(selected_mod_gage)
            used_modified_gages.add(selected_mod_gage)
            replacements_made += 1

    # Store resulting training set
    train_df = rdf[rdf.index.isin(current_gages)].copy()
    train_df = pd.concat([train_df, five_seen_df]) # add 5 back
    train_df['set'] = step
    training_sets[f'train_set_{step}'] = train_df
    
    # Logging
    num_pristine = (train_df['reg_score'] == 0).sum()
    num_mod = len(train_df) - num_pristine
    mean_q_val = train_df['Q_cfs_mean'].median()
    mean_a_val = train_df['area_km2'].median()
    mean_reg = train_df['reg_score'].median()
    days = int(train_df['period'].sum())
    
    print(f"Train Set {step:02d} | Pristine: {num_pristine:02d} | Modified: {num_mod:02d} | Days: {days} | "
          f"Mean Reg Score: {mean_reg:.3f} | Mean Q: {mean_q_val:.1f} cfs | Mean Area: {mean_a_val:.1f} km²")

og = set(training_sets['train_set_0']['name'])
for i in range(1,11):
    print(i)
    cg = set(training_sets[f'train_set_{i}']['name'])
    added = cg.difference(og)
    subtracted = og.difference(cg)
    print(f'added: {added}')
    print(f'subtracted: {subtracted}')
    og = cg

set10 = training_sets['train_set_8']
set10[['name', 'geometry']].explore()

################
# PREP NH
################
import yaml
import pickle

# define paths
experiment = 2
ccwd = fr'/content/drive/MyDrive/natural_streamflow_colab/configs/experiment{experiment}' # how to write file paths so Collab can read them
pcwd = fr'G:\My Drive\natural_streamflow_colab\configs\experiment{experiment}\pickles' # where to save the pickles
gcwd = fr'G:\My Drive\natural_streamflow_colab\configs\experiment{experiment}' # where to save the configs

# 1. Save test_set.txt (list of gage IDs separated by line breaks)
test_set_path = os.path.join(gcwd, 'test_set.txt')
with open(test_set_path, 'w') as f:
    f.write('\n'.join(map(str, test_gages)))
test_set_gpath = f'{ccwd}/test_set.txt'

# Path to template config YAML
config_temp = os.path.join(gcwd, 'config_template_cuda.yml')

# 2. Iterate through training sets, save text files, and generate modified YAMLs
for i in range(11):
    key = f'train_set_{i}'
    tset = training_sets[key]
    tset_gages = tset.index.to_list()

    # Save the training set gage list to a .txt file
    tset_txt_path = os.path.join(gcwd, f'{key}.txt')
    with open(tset_txt_path, 'w') as f:
        f.write('\n'.join(map(str, tset_gages)))

    # Path to reference in YAML (using ccwd as specified)
    train_set_gpath = f"{ccwd}/{key}.txt"

    # Read config template
    with open(config_temp, 'r') as f:
        config_data = yaml.safe_load(f)

    # change experiment name
    config_data['experiment_name'] = f'experiment{experiment}'

    # Modify basin file paths to point to the current training set path
    config_data['validation_basin_file'] = test_set_gpath
    config_data['train_basin_file'] = train_set_gpath
    config_data['test_basin_file'] = test_set_gpath

     # the pristine one needs validation dates for hyperparameter tuning
    if i == 0:
        config_data['validation_start_date'] = '01/10/2022'
        config_data['validation_end_date'] = '30/09/2025'
        config_data['test_start_date'] = '01/10/2003'
        config_data['test_end_date'] = '30/09/2022'

    # add the per basin train periods
    trainpb = {}
    testpb = {}
    valpb = {}
    train_sd = pd.to_datetime(config_data['train_start_date'], dayfirst=True)
    train_ed = pd.to_datetime(config_data['train_end_date'], dayfirst=True)
    test_sd = pd.to_datetime(config_data['test_start_date'], dayfirst=True)
    test_ed = pd.to_datetime(config_data['test_end_date'], dayfirst=True)
    val_sd = pd.to_datetime(config_data['validation_start_date'], dayfirst=True)
    val_ed = pd.to_datetime(config_data['validation_end_date'], dayfirst=True)

    for gage in tset.index.to_list() + test_gages:
        if gage not in test_gages:
            # train these on all dates
            trainpb[gage] = {'start_dates': [train_sd], 'end_dates':[train_ed]}
        elif gage in five_seen:
            # train these gages on just half their period of record, test on the other half
            fd = test_set[test_set.index==gage]['firstday'].iloc[0]
            ld = test_set[test_set.index==gage]['lastday'].iloc[0]
            midpoint = fd + (ld - fd) / 2
            midpoint = midpoint.normalize()
            one_day_after = midpoint + pd.Timedelta(days=1)
            # validation and test dates should be the same for all but set 0
            trainpb[gage] = {'start_dates': [train_sd], 'end_dates':[midpoint]}
            testpb[gage] = {'start_dates': [one_day_after], 'end_dates':[test_ed]}
            valpb[gage] = {'start_dates': [val_sd], 'end_dates':[val_ed]}
        elif gage in five_unseen:
            # don't train on these at all
            testpb[gage] = {'start_dates': [test_sd], 'end_dates':[test_ed]}
            valpb[gage] = {'start_dates': [val_sd], 'end_dates':[val_ed]}
    
    test_pickle_path = os.path.join(pcwd, f'test_{i}.pkl')
    test_pickle_gpath = f'{ccwd}/pickles/test_{i}.pkl'
    train_pickle_path = os.path.join(pcwd, f'train_{i}.pkl')
    train_pickle_gpath = f'{ccwd}/pickles/train_{i}.pkl'
    val_pickle_path = os.path.join(pcwd, f'val_{i}.pkl')
    val_pickle_gpath = f'{ccwd}/pickles/val_{i}.pkl'

    # save files
    with open(test_pickle_path, "wb") as f:
        pickle.dump(testpb, f)
    with open(train_pickle_path, "wb") as f:
        pickle.dump(trainpb, f)
    with open(val_pickle_path, "wb") as f:
        pickle.dump(valpb, f)
    
    config_data['per_basin_test_periods_file'] = test_pickle_gpath
    config_data['per_basin_train_periods_file'] = train_pickle_gpath
    config_data['per_basin_validation_periods_file'] = val_pickle_gpath

    # delete global start and end dates
    for date in ['train_start_date', 'train_end_date', 'test_start_date', 'test_end_date', 'validation_start_date', 'validation_end_date']:
        del config_data[date]

    if i == 0:
        print(testpb)

    # Save modified configuration to new YAML file
    config_path = os.path.join(gcwd, f'configs/config_{key}.yml')
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_data, f, default_flow_style=False)

    # Add Consumptive Use predictors and create new yaml
    inputs = config_data['dynamic_inputs']
    inputs = inputs + ['Q_CU_cfs']
    config_data['dynamic_inputs'] = inputs
    config_data['experiment_name'] = f'experiment{experiment}_wCU'
    config_path = os.path.join(gcwd, f'configs/config_wCU_{key}.yml')
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_data, f, default_flow_style=False)


# Prep for tuning
tcwd = fr'G:\My Drive\natural_streamflow_colab\configs\tuning'

# Path to pristine config YAML
config_0 = os.path.join(gcwd, 'config_train_set_0.yml')

# prep config dir for hyperparameter tuning
hiddens = [64, 128, 256]
dropouts = [0.1, 0.2]
batchs = [128, 256]
epochs = [50]

for e in epochs:
    for h in hiddens:
        for d in dropouts:
            for b in batchs:
            # Read config template
                with open(config_0, 'r') as f:
                    config_data = yaml.safe_load(f)

                config_data['hidden_size'] = h
                config_data['output_dropout'] = d
                config_data['batch_size'] = b
                config_data['epochs'] = e

                d_str = str(d)[-1]

                # change experiment name
                config_data['experiment_name'] = f'experiment{experiment}_tuning'

                config_path = os.path.join(tcwd, f'batch{b}_hidden{h}_dropout{d_str}_epoch{e}.yml')
                with open(config_path, 'w') as f:
                    yaml.safe_dump(config_data, f, default_flow_style=False)