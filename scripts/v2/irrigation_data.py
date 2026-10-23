import pandas as pd
import geopandas as gpd
import numpy as np
import os
import glob
import re

cwd = os.getcwd()
ncwd = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow'
appcwd = os.path.join(cwd, r'shiny-app\ucol_natural')
appDir = os.path.join(appcwd, r'timeseries') # does not have each individual diversion
proj_crs = 5070

basins_path = os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet')
gages_path = os.path.join(appcwd, r'spatial_data/all_UCOL_gages.parquet')
basins = gpd.read_parquet(basins_path)
basins.geometry = basins.geometry.to_crs(proj_crs) # need this for spatial joins

# COLORADO DATA: subfolders are divided by watersheds
# loop through folders in main dir: N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\spatial_data\irrigation\CO
# each folder contains polygon shps with column IRRIG_TYPE and values DRIP, FLOOD, and SPRINKLER. Ignore all rows with other types
# there is a polygon shp for years 2000, 2005, 2010, 2015, 2020, and 2023. The name of the year is in the file name of the shp

# UTAH DATA: subfolders contain all data for the state but one for each year
# loop through folders in main dir: N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\spatial_data\irrigation\UT
# each folder contains polygon shps with column IRR_METHOD and values Flood, Sprinkler, Drip. Ignore all rows with other types

# first find which shapefiles you need to analyze each basin and save that info
# large basins will require joining multiple shapefiles and sometimes UTAH and Colorado data
# some basins in Wyoming and New Mexico will not have any data.
# then loop through each basin and find the % of irrigated land that is flood, sprinkler, and drip and save that info for each basin.

# -------------------------------------------------------------------------
# 1. Setup paths and projection
# -------------------------------------------------------------------------
cwd = os.getcwd()
ncwd = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow'
appcwd = os.path.join(cwd, r'shiny-app\ucol_natural')
appDir = os.path.join(appcwd, r'timeseries')
proj_crs = 26913

basins_path = os.path.join(appcwd, r'spatial_data\all_UCOL_basins.parquet')
gages_path = os.path.join(appcwd, r'spatial_data\all_UCOL_gages.parquet')

# Load basins and reproject
basins = gpd.read_parquet(basins_path)
basins = basins.to_crs(proj_crs)

# Define irrigation data roots
co_irr_dir = os.path.join(ncwd, r'spatial_data\irrigation\CO')
ut_irr_dir = os.path.join(ncwd, r'spatial_data\irrigation\UT')

# Standard method mapping dictionary
METHOD_MAP = {
    # Colorado values
    'DRIP': 'Drip',
    'FLOOD': 'Flood',
    'SPRINKLER': 'Sprinkler',
    # Utah values
    'Drip': 'Drip',
    'Flood': 'Flood',
    'Sprinkler': 'Sprinkler'
}

# -------------------------------------------------------------------------
# 2. Index all irrigation shapefiles and extract metadata
# -------------------------------------------------------------------------
irr_files = []

# Scan Colorado directory
for root, _, files in os.walk(co_irr_dir):
    for f in files:
        if f.endswith('.shp'):
            # Extract 4-digit year from filename
            match = re.search(r'(2023)', f)
            if match:
                irr_files.append({
                    'state': 'CO',
                    'year': int(match.group(1)),
                    'path': os.path.join(root, f),
                    'type_col': 'IRRIG_TYPE'
                })

# Scan Utah directory
for root, _, files in os.walk(ut_irr_dir):
    for f in files:
        if f.endswith('.shp'):
            match = re.search(r'(2023)', f)
            if match:
                irr_files.append({
                    'state': 'UT',
                    'year': int(match.group(1)),
                    'path': os.path.join(root, f),
                    'type_col': 'IRR_METHOD'
                })

df_irr_files = pd.DataFrame(irr_files)

# Get bounding box / total bounds for each shapefile to fast-index basin overlaps
shp_bounds = []
for idx, row in df_irr_files.iterrows():
    # Read bounding envelope only to save memory
    gdf_bbox = gpd.read_file(row['path'], rows=0)  # load schema/crs
    full_gdf_bbox = gpd.read_file(row['path'], bbox=None) # get total bounds
    bbox_geom = gpd.GeoSeries([full_gdf_bbox.unary_union.envelope], crs=full_gdf_bbox.crs).to_crs(proj_crs).iloc[0]
    shp_bounds.append(bbox_geom)

gdf_shp_index = gpd.GeoDataFrame(df_irr_files, geometry=shp_bounds, crs=proj_crs)

# -------------------------------------------------------------------------
# 3. Find which shapefiles overlap each basin
# -------------------------------------------------------------------------
# Spatial join between basin envelopes and indexed shapefiles
basin_shp_map = gpd.sjoin(
    basins[['geometry']].reset_index(),
    gdf_shp_index,
    how='inner',
    predicate='intersects'
)

# Save the spatial mapping file list
mapping_output_path = os.path.join(ncwd, r'spatial_data\basin_irrigation_shapefiles_index.csv')
basin_shp_map[['gage', 'state', 'year', 'path']].to_csv(mapping_output_path, index=False)
print(f"Saved basin-shapefile spatial index to: {mapping_output_path}")

# -------------------------------------------------------------------------
# 4. Calculate % of Irrigated Land per Basin by Method and Year
# -------------------------------------------------------------------------
results = []

basins = basins.reset_index()
basin_id_col = 'gage'

basins_sub = basins.sample(10)
for b_idx, basin_row in basins.iterrows():
    gage = basin_row[basin_id_col]
    b_geom = gpd.GeoSeries([basin_row.geometry], crs=proj_crs)
    
    # Get shapefiles that intersect this basin
    matched_shps = gdf_shp_index[gdf_shp_index.geometry.intersects(basin_row.geometry)]
    
    if matched_shps.empty:
        # Basin is likely in WY, NM, or has no coverage
        continue
    
    # Group matched shapefiles by year
    years = matched_shps['year'].unique()
    
    for yr in years:
        print(gage, yr)
        yr_shps = matched_shps[matched_shps['year'] == yr]
        
        clipped_polys = []
        
        for _, shp_row in yr_shps.iterrows():
            # Read shapefile using spatial bounding box filter for speed
            gdf_irr = gpd.read_file(shp_row['path'], bbox = basin_row.geometry) #bbox=basin_row.geometry
            #gdf_irr = gdf_irr.clip(basin_row.geometry)
            
            if gdf_irr.empty:
                #print(shp_row['path'], "empty")
                continue
                
            # Reproject
            gdf_irr = gdf_irr.to_crs(proj_crs)
            
            # Filter for valid irrigation methods
            col_name = shp_row['type_col']
            if col_name in gdf_irr.columns:
                gdf_irr = gdf_irr[gdf_irr[col_name].isin(METHOD_MAP.keys())].copy()
                gdf_irr['method_std'] = gdf_irr[col_name].map(METHOD_MAP)
                
                # Clip to basin geometry
                clipped = gpd.clip(gdf_irr, b_geom)
                if not clipped.empty:
                    clipped_polys.append(clipped[['method_std', 'geometry']])
        
        if clipped_polys:
            gdf_basin_irr = pd.concat(clipped_polys, ignore_index=True)
            
            # Compute area in equal-area projection units (sq meters)
            gdf_basin_irr['area_sqm'] = gdf_basin_irr.geometry.area
            
            # Group by method
            method_areas = gdf_basin_irr.groupby('method_std')['area_sqm'].sum()
            total_irr_area = method_areas.sum()
            
            if total_irr_area > 0:
                flood_pct = (method_areas.get('Flood', 0.0) / total_irr_area) * 100
                sprinkler_pct = (method_areas.get('Sprinkler', 0.0) / total_irr_area) * 100
                drip_pct = (method_areas.get('Drip', 0.0) / total_irr_area) * 100
            else:
                flood_pct = sprinkler_pct = drip_pct = 0.0
                
            results.append({
                'basin_id': gage,
                'year': yr,
                'total_irrigated_sqkm': total_irr_area/1E6,
                'pct_flood': flood_pct,
                'pct_sprinkler': sprinkler_pct,
                'pct_drip': drip_pct
            })
        else:
            print('no farms in:', gage)

# Convert results to DataFrame and save
df_results = pd.DataFrame(results)


# join with basins
df_results.set_index('basin_id', inplace=True)
basins.set_index('gage', inplace=True)
irr_df = pd.merge(df_results, basins[['geometry']], how='outer', left_index=True, right_index=True)
irr_df = gpd.GeoDataFrame(irr_df)

flood_eff = 0.5
sprink_eff = 0.8
drip_eff = 0.9

irr_df['ie'] = ((irr_df['pct_flood'] / 100) * flood_eff) + ((irr_df['pct_sprinkler'] / 100) * sprink_eff) + ((irr_df['pct_drip'] / 100) * drip_eff)
med_ie = irr_df.ie.median()
irr_df['ie'] = irr_df['ie'].fillna(med_ie)

results_output_path = os.path.join(ncwd, r'spatial_data\basin_irrigation_percentages.parquet')
irr_df.to_parquet(results_output_path, index=True)
print(f"Analysis complete. Results saved to: {results_output_path}")


from pynhd import NLDI, WaterData, NHDPlusHR
import pynhd as nhd
nldi = NLDI()
station_id = "09380000"

basin = nldi.get_basins(station_id)

flw_main = nldi.navigate_byid(
    fsource="nwissite",
    fid=f"USGS-{station_id}",
    navigation="upstreamMain",
    source="flowlines",
    distance=1000,
)

flw_trib = nldi.navigate_byid(
    fsource="nwissite",
    fid=f"USGS-{station_id}",
    navigation="upstreamTributaries",
    source="flowlines",
    distance=300,
)