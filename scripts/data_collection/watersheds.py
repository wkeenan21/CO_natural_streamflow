"""
Download spatial data like watershed polygons, gage locations
"""
import sys
import pandas as pd
import os
import geopandas as gpd
from os.path import join
import ee
from shapely.geometry import box, Point
import dataretrieval.nwis as nwis
import sys

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts/data_collection'))
from special_functions import *

# read watersheds from flow25, gage points
flow25 = gpd.read_file(join(cwd, r'data\CSU_Flow25\site_coordinates_20250731.shp'))
flow25['gauge_id'] = fix_gage_series(flow25['gage_used'])

"""
Make a box
"""
# Define bounding box coordinates: (minx, miny, maxx, maxy)
bounds = (-109.5, 36, -101, 42)
# Create a shapely polygon from the bounding box
polygon = box(*bounds)
# Put it into a GeoDataFrame
gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs="EPSG:4326")
# Save as a shapefile
gdf.to_file(join(cwd, r"data\shapefiles\bounding_box_CO.shp"))

"""
Download DEM for the box
"""
# Initialize the Earth Engine API
# Trigger the authentication flow.
ee.Authenticate()
# Initialize the library.
ee.Initialize(project='ee-wkeenan21')
# create a shapefile of Colorado
# Load your Colorado shapefile asset (replace with your actual asset ID)
# Example: 'users/your_username/Colorado'
co = ee.FeatureCollection("projects/ee-wkeenan21/assets/CSU/bounding_box_CO")
# Load the USGS 3DEP 10m DEM
dem = ee.ImageCollection("USGS/3DEP/10m_collection")
dem = dem.mosaic()  # Merge into a single image
# Clip DEM to Colorado boundary
dem_co = dem.clip(co)
# Set up export task to Google Drive
task = ee.batch.Export.image.toDrive(
    image=dem_co,
    description="CO_DEM_30m",
    folder="EarthEngineExports",   # Folder in Google Drive (creates if doesn’t exist)
    fileNamePrefix="CO_DEM_10m",   # Output file name
    region=co.geometry(),
    scale=30,                      # Match 10 m resolution
    crs="EPSG:4326",               # WGS84 (you can change to match your needs)
    maxPixels=1e13                 # Allow large exports
)
# Start the export
task.start()

"""
Find all the gages in CO and flow25, check which ones are already have in Caravan
"""
parameterCode = '00060'
# get
sites = nwis.get_info(
    parameterCd=parameterCode,
    stateCd='CO',  # optional: limit to Colorado
    period='P10W',
)[0]

#Extract relevant metadata
co_df = sites[['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va', 'drain_area_va']]

#Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(co_df.dec_long_va, co_df.dec_lat_va)]
co_df = gpd.GeoDataFrame(co_df, geometry=geometry, crs="EPSG:4326")
#gage_sites_gdf.to_file(join(cwd, r'data\shapefiles\watersheds_CO_active.shp'))
co_df.rename(columns={'site_no':'gauge_id'}, inplace=True)

# loop through the Caravans data and see what we got
caravan = r"C:\Users\C830645719\Downloads\Caravan\Caravan\timeseries\csv"

camels = []
for dataset in ['camels']:
    for file in os.listdir(join(caravan, dataset)):
        name, gage = file.split('_')
        gage, file = gage.split('.')
        camels.append(gage)

# get a shapefile of the pour points for camels basins
sites = nwis.get_info(
    sites=camels,
    parameterCd=parameterCode,
)[0]

camels_df = sites[['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va', 'drain_area_va']].copy()
camels_df.rename(columns={'site_no':'gauge_id'}, inplace=True)
#Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(camels_df.dec_long_va, camels_df.dec_lat_va)]
camels_df = gpd.GeoDataFrame(camels_df, geometry=geometry, crs="EPSG:4326")
camels_df.rename(columns={'site_no':'gauge_id'}, inplace=True)

# merge flow 25 and the new ones
flow25 = flow25.to_crs(4326)

gage_sites_gdf = pd.concat([co_df[['geometry', 'gauge_id']], flow25[['geometry', 'gauge_id']], camels_df[['geometry', 'gauge_id']]])
gage_sites_gdf = gage_sites_gdf.to_crs(3857)
gage_sites_gdf = gage_sites_gdf.drop_duplicates(subset='gauge_id')

"""
Delineate watersheds using whitebox
"""
# Paths
work_dir = join(cwd, r'data\terrain')
dem = join(work_dir, "CO_DEM_30m.tif")

# filter by what we need
pour_points = gage_sites_gdf[gage_sites_gdf['gauge_id'].isin(co_df['gauge_id'])].copy()
pourPointsPath = join(work_dir, r'pour_points_camels_co_flow25.shp')
pour_points.to_file(pourPointsPath)
import rasterio
from whitebox import WhiteboxTools
import os

# Initialize WhiteboxTools
wbt = WhiteboxTools()
wbt.verbose = True  # See progress in console

# Reproject
import rioxarray
# Open raster
dem = rioxarray.open_rasterio(dem)
# Reproject to EPSG:3857
dem_3857 = dem.rio.reproject("EPSG:3857")
# Save to GeoTIFF
dem_3857_path = join(work_dir, r'dem_3857.tif')
dem_3857.rio.to_raster(dem_3857_path)

# Step 1: Fill sinks
dem_filled = os.path.join(work_dir, "dem_filled.tif")
wbt.fill_depressions(dem_3857_path, dem_filled)

# Step 2: Flow direction (D8)
flow_dir = os.path.join(work_dir, "flow_dir.tif")
wbt.d8_pointer(dem_filled, flow_dir)

# Step 3: (Optional) Snap pour points to high flow accumulation cells
flow_acc = os.path.join(work_dir, "flow_acc.tif")
wbt.d8_flow_accumulation(dem_filled, flow_acc, out_type="cells")

snapped_pp = os.path.join(work_dir, "pour_points_snapped.shp")
wbt.snap_pour_points(pourPointsPath, flow_acc, snapped_pp, snap_dist=300) # snap_dist in map units (e.g. meters)

# Step 4: Watershed delineation
watersheds_raster = os.path.join(work_dir, "watersheds.tif")
wbt.watershed(flow_dir, snapped_pp, watersheds_raster)

# Step 5: Raster to vector polygons
watersheds_vec = os.path.join(work_dir, "watersheds.shp")
wbt.raster_to_vector_polygons(watersheds_raster, watersheds_vec)

# Step 6: Load into GeoDataFrame
watersheds_gdf = gpd.read_file(watersheds_vec)

# merge with pour points to get the name
pour_points['VALUE'] = pour_points.index
watersheds_gdf = pd.merge(left=watersheds_gdf, right=pour_points[['VALUE','gauge_id']], left_on='VALUE', right_on='VALUE')
watersheds_gdf.to_file(join(cwd, r'data\shapefiles\wsheds_co_gages.shp'))

# just merge 3 shapefiles
# flow25
flow25 = gpd.read_file(r'C:\Users\C830645719\OneDrive - Colostate\flow_prediction\data\0-FINAL-DELIVERABLES\watersheds_shapefile_20250624.shp').to_crs(3857)
flow25['gauge_id'] = fix_gage_series(flow25['gage_used'])
flow25.loc[74, 'gauge_id'] = '402114105350101'
# camels
camels = gpd.read_file(r'C:\Users\C830645719\Downloads\Caravan\Caravan\shapefiles\camels\camels_basin_shapes.shp').to_crs(3857)
camels['gauge_id'] = camels['gauge_id'].str[7:]
# colorado
co_df = gpd.read_file(join(cwd, r'data\shapefiles\wsheds_co_gages.shp')).to_crs(3857)

cols = ['gauge_id', 'geometry']

df = pd.concat([camels[cols], flow25[cols]])
df = df.sort_values(by='gauge_id')
df = df.drop_duplicates(subset=['gauge_id'])
df.to_file(join(cwd, r'data\shapefiles\wsheds_co_camels_flow25_2.shp'))

def write_lines_to_file(strings, filename):
    """
    Write a list of strings to a text file, one per line.
    
    Parameters
    ----------
    strings : list of str
        The lines you want to write.
    filename : str
        The path to the output text file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for s in strings:
            f.write(s + "\n")

# write the flow25 gages to a txt file
write_lines_to_file(list(flow25['gauge_id']), os.path.join(cwd, r'scripts/configs/flow25_gages.txt'))
write_lines_to_file(list(camels))


