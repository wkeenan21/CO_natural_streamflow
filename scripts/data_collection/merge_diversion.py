import pandas as pd
import geopandas as gpd
import os
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
# take a look at diversions

ncwd = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow'

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

cwd = os.getcwd()

# DIVERSIONS
# COLUMN = siteID
dvrs = gpd.read_file(os.path.join(ncwd, r"data\diversion\input\ucrb_diversion_master_table.csv"))
dvrs = df_to_geodataframe(dvrs, lat_col='decLat', lon_col='decLong')

dvrs_intra = dvrs[dvrs['siteUse']=='intrabasin'].copy()
dvrs_intra['siteUse'] = 'transbasin'
dvrs_intra['decLat'] = dvrs_intra['dest_decLat']
dvrs_intra['decLong'] = dvrs_intra['dest_decLong']
dvrs_intra['siteID'] = dvrs_intra['siteID'].str.replace('intrabasin', 'transbasin')

dvrs = pd.concat([dvrs, dvrs_intra])

#dvrs.to_file(r"data\shapefiles\ucrb_diversion_master_table.shp")

# WATERSHEDS
# gageID column = gage

wsheds = gpd.read_file(os.path.join(ncwd, r'data\shapefiles\UCOL_headwaters_sheds.shp')).to_crs(dvrs.crs)

# DIRECTORY OF CSVS FOR STREAMFLOW
# FILE NAME FORMAT = gageId.csv
# flow column = Q_cfs
# date column = date
flowDir = os.path.join(cwd, r'data\NH_data\unfilled')

# WIDE CSV FOR DIVERSION DATA IN CFS
# COLUMN HEADERS = siteID
# date column = Date
dvrsFlow = pd.read_csv("data\diversion\processed\processed_data\combined_diversion_records_filtered_filled_cfs_fill_years.csv")

# OUT DIRECTORY FOR STREAMFLOW WITH DIVERSIONS
wdvrsDir = os.path.join(cwd, r'data\NH_data\w_diversion')

# 1. Spatial Join: Find which diversions are in which watersheds
# 'inner' join keeps only points that fall inside a polygon
# 'within' ensures the point is geometrically inside the watershed boundary
joined = gpd.sjoin(dvrs, wsheds, how="inner", predicate="within")

# 2. Ensure date columns are datetime objects for proper merging
dvrsFlow['Date'] = pd.to_datetime(dvrsFlow['Date'])

# 3. Process each watershed
for gage_id in wsheds['gage'].unique():
    
    # Identify siteIDs for diversions located in this specific watershed
    target_diversions = joined[joined['gage'] == gage_id]['siteID'].unique()
    
    # Path to the existing streamflow file
    flow_file_path = os.path.join(flowDir, f"{gage_id}.csv")

    print(gage_id, f'diversions: {len(target_diversions)}')
    
    if os.path.exists(flow_file_path):
        # Load streamflow data
        flow_df = pd.read_csv(flow_file_path)
        flow_df['date'] = pd.to_datetime(flow_df['date'])
        
        if len(target_diversions) > 0:
            # Filter diversion data for the relevant siteIDs
            valid_cols = [sid for sid in target_diversions if sid in dvrsFlow.columns]
            
            if valid_cols:
                # 1. Extract the raw data
                subset_dvrs = dvrsFlow[['Date'] + valid_cols].copy()
                
                # 2. Convert raw diversions to Consumptive Use (CU)
                useDict = {
                    'irrigation': -0.6, 'municipal': -0.3, 'interbasin': -1, 
                    'industrial': -1, 'hydropower': 0, 'intrabasin': -1, 'transbasin': 1
                }
                
                # Create a temporary list to hold the names of the new CU columns
                cu_cols = []
                
                for col in valid_cols:
                    # Determine the multiplier by checking the end of the siteID string
                    multiplier = 0 # Default if no match is found
                    for usage, val in useDict.items():
                        if col.lower().endswith(usage):
                            multiplier = val
                            break
                    
                    # Calculate CU for this specific diversion
                    cu_col_name = f"{col}_CU"
                    subset_dvrs[cu_col_name] = subset_dvrs[col] * multiplier
                    cu_cols.append(cu_col_name)
                
                # 3. Aggregate: Sum all CU columns to get the total impact on the watershed
                subset_dvrs['Q_cfs_cu'] = subset_dvrs[cu_cols].sum(axis=1)
            
            # 4. merge
            combined_df = pd.merge(
                flow_df, 
                subset_dvrs.rename(columns={'Date': 'date'}), 
                on='date', 
                how='inner'
            )

            if len(combined_df) < 1:
                raise Exception('merge failed')

            combined_df['Q_cfs_nat'] = combined_df['Q_cfs'] - combined_df['Q_cfs_cu']
        else:
            # If no diversions found, we still save the original flow (or skip)
            combined_df = flow_df
            
        # 4. Save the new CSV
        out_path = os.path.join(wdvrsDir, f"{gage_id}.csv")
        combined_df.to_csv(out_path, index=False)
        print(f"Processed gage {gage_id}: Added {len(target_diversions)} diversion columns.")
    else:
        print(f"Warning: Streamflow file for {gage_id} not found in {flowDir}.")


def plot_watershed_flows(df, gage_id, sdate='2000-01-01', edate='2026-01-01'):
    """
    Plots Observed, Consumptive Use, and Naturalized flows.
    """
    # Ensure date is the index for cleaner plotting
    plot_df = df.set_index('date').sort_index()
    plot_df.index = pd.to_datetime(plot_df.index)
    sdate = pd.Timestamp(sdate)
    edate = pd.Timestamp(edate)
    plot_df = plot_df[(plot_df.index > sdate) & (plot_df.index < edate)]

    plt.figure(figsize=(12, 6))

    # 1. Plot Naturalized Flow (usually the highest)
    plt.plot(plot_df.index, plot_df['Q_cfs_nat'], 
             label='Naturalized Flow (Q_nat)', color='forestgreen', alpha=0.8, linewidth=1.5)

    # 2. Plot Observed Flow
    plt.plot(plot_df.index, plot_df['Q_cfs'], 
             label='Observed Flow (Q_obs)', color='royalblue', alpha=0.7, linewidth=1.2)

    # 3. Plot Consumptive Use (Area fill is often better for CU)
    plt.fill_between(plot_df.index, 0, plot_df['Q_cfs_cu'], 
                     label='Consumptive Use (Q_cu)', color='red', alpha=0.3)

    # Formatting
    plt.title(f'Streamflow Components for Gage: {gage_id}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Flow (cfs)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Show or save
    # plt.savefig(f"{gage_id}_flow_plot.png")
    plt.show()


# lets look at the East River
gage = '09242500'
df = pd.read_csv(os.path.join(cwd, fr"data\NH_data\w_diversion\{gage}.csv"))
# Example usage inside your loop:
plot_watershed_flows(df, gage, sdate='2019-01-01')


# lets look for gages where natural flow may be lower than observed because of transbasin imports
imports = []
# gages where consumptive use is higher than natural flows
overallocated = []
for gage_id in wsheds['gage'].unique():
    try:
        df = pd.read_csv(os.path.join(cwd, fr"data\NH_data\w_diversion\{gage_id}.csv"))
        test = df['Q_cfs_nat']
        if any(df['Q_cfs_nat'] < df['Q_cfs']):
            imports.append(gage_id)
        elif any(df['Q_cfs_cu'] > df['Q_cfs']):
            overallocated.append(gage_id)
    except:
        continue

print(imports)
print(overallocated)

wsheds = gpd.read_file(os.path.join(cwd, r"data\CSU_Flow25\watersheds_shapefile_20250624.shp"))
def fix_gage_id(id_val):
    id_str = str(id_val).strip()
    # Only pad if it is a 7-digit numeric string
    if len(id_str) == 7 and id_str.isdigit():
        return id_str.zfill(8)

    return id_str

# Apply the logic to the gage column
wsheds['gage'] = wsheds['gage'].apply(fix_gage_id)
wsheds['gage'] = np.where(wsheds['gage'].str.contains('E+', regex=False), wsheds['usgs_id'], wsheds['gage']) # the pesky long ones

flow_ratios = {}

for gage_id in wsheds['gage'].unique():
    try:
        print(gage_id)
        # Load the newly created CSVs
        file_path = os.path.join(cwd, fr"data\NH_data\w_diversion\{gage_id}.csv")
        df = pd.read_csv(file_path)
        
        # Calculate totals for the period of record
        total_obs = df['Q_cfs'].sum()
        total_nat = df['Q_cfs_nat'].sum()
        
        if total_nat > 0:
            ratio = (total_obs / total_nat) * 100
        else:
            ratio = None # Avoid division by zero
            
        flow_ratios[gage_id] = ratio

    except FileNotFoundError:
        print(f"Skipping {gage_id}: CSV not found.")
    except Exception as e:
        print(f"Error processing {gage_id}: {e}")

# Map the results back to the GeoDataFrame
wsheds['flow_ratio_pct'] = wsheds['gage'].map(flow_ratios)

# --- MAP THE RESULTS ---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot the basins colored by flow_ratio_pct
# We use 'RdYlBu' so that basins with much less flow than natural (heavily diverted) 
# appear red/yellow, and naturalized basins appear blue.
wsheds.plot(column='flow_ratio_pct', 
            ax=ax, 
            legend=True, 
            cmap='RdYlBu')

### Insights on the Results:
ax.set_title("Watershed Impact: Observed vs. Natural Flow", fontsize=15)
ax.set_axis_off()

plt.show()