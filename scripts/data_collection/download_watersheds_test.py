import requests
import geopandas as gpd
import sys
import os
from shapely.geometry import Polygon, MultiPolygon
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts/data_collection'))
from special_functions import *

gages = gpd.read_file(r'C:\Users\C830645719\OneDrive - Colostate\documents\GitHub\CO_natural_streamflow\data\shapefiles\gages_CO_active.shp')

feature_collection = {
    "type": "FeatureCollection",
    "features": []
}

for gage in gages['site_no']:
    try:
        response = requests.get(url=fr'https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{gage}/basin')
        responseJson = response.json()
        feature = response.json()['features'][0]
        feature['properties']['gauge_id'] = gage
        feature_collection["features"].append(feature)
        print(f'got {gage}')
    except:
        print(f'failed {gage}')

gdf = gpd.GeoDataFrame.from_features(feature_collection["features"], crs=4326)
gdf = gdf.drop_duplicates(subset='gauge_id', keep='last')


# get rid of holes and little slivers

def to_single(geom):
    if geom.geom_type == "MultiPolygon":
        # Take the largest polygon (by area) if there are multiple parts
        geom = max(geom.geoms, key=lambda a: a.area)
    return geom

def remove_holes(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    elif geom.geom_type == "MultiPolygon":
        # Apply recursively to all parts
        parts = [Polygon(p.exterior) for p in geom.geoms]
        return max(parts, key=lambda a: a.area)  # keep largest
    else:
        return geom

gdf["geometry"] = gdf.geometry.apply(to_single)
gdf["geometry"] = gdf.geometry.apply(remove_holes)

gdf.to_file(r'C:\Users\C830645719\OneDrive - Colostate\documents\GitHub\CO_natural_streamflow\data\shapefiles\wsheds_CO_NLDI_rip2.shp')

flow25 = gpd.read_file(r'C:\Users\C830645719\OneDrive - Colostate\flow_prediction\data\0-FINAL-DELIVERABLES\watersheds_shapefile_20250624.shp').to_crs(3857)
flow25['gauge_id'] = fix_gage_series(flow25['gage_used'])
flow25.loc[74, 'gauge_id'] = '402114105350101'
# camels
camels = gpd.read_file(r'C:\Users\C830645719\Downloads\Caravan\Caravan\shapefiles\camels\camels_basin_shapes.shp').to_crs(3857)
camels['gauge_id'] = camels['gauge_id'].str[7:]
# Colorado influenced gages
co_df = gdf.to_crs(3857)

cols = ['gauge_id', 'geometry']

df = pd.concat([camels[cols], flow25[cols], co_df[cols]])
df = df.sort_values(by='gauge_id')
df = df.drop_duplicates(subset=['gauge_id'], keep='first')

df.to_file(os.path.join(cwd, r'data\shapefiles\wsheds_co_camels_flow25_3.shp'))