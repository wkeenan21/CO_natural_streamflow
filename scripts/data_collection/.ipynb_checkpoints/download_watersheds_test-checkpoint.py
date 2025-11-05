import requests
import geopandas as gpd
import contextily as cx
import matplotlib
matplotlib.use(backend='TkAgg')
import matplotlib.pyplot as plt


response = requests.get(url=r'https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-07124410/basin')

responseJson = response.json()

feature = response.json()['features']

print(feature)

geojson_obj = feature[0]  # extract the GeoJSON dictionary

# Handles both FeatureCollections and lists of features
if geojson_obj["type"] == "FeatureCollection":
    gdf = gpd.GeoDataFrame.from_features(geojson_obj["features"])
elif geojson_obj["type"] == "Feature":
    from shapely.geometry import shape
    gdf = gpd.GeoDataFrame(
        [geojson_obj["properties"]],
        geometry=[shape(geojson_obj["geometry"])],
        crs="EPSG:4326"
    )
else:
    raise ValueError("Unsupported GeoJSON type.")

gdf = gdf.to_crs(epsg=3857)


