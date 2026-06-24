import geopandas as gpd
import ipyleaflet as L
from shiny.express import input, render, ui
from shinywidgets import render_widget
import os

print(os.getcwd())

# 1. Load the shapefile and calculate its bounding box extent
# (Assumes your shapefile is in a 'data' folder relative to app.py)
gdf = gpd.read_file(r"shiny-app\map-distance\data\bounding_box_CO.shp")

# Ensure it's in WGS84 (Lat/Lon) to match ipyleaflet coordinates
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)

# Get the total bounds: (minx, miny, maxx, maxy) -> (min_lon, min_lat, max_lon, max_lat)
minx, miny, maxx, maxy = gdf.total_bounds
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2
bounds = [[miny, minx], [maxy, maxx]]

# 2. Setup the Shiny Page (Clean layout, no sidebar or tiles)
ui.page_opts(title="Colorado Map View", fillable=True)

with ui.card():
    ui.card_header("Colorado Boundary Map")

    @render_widget
    def map():
        # Initialize map centered on the shapefile's center
        m = L.Map(center=(center_lat, center_lon))
        
        # Define the two requested basemaps
        osm_layer = L.basemap_to_tiles(L.basemaps.OpenStreetMap.Mapnik)
        satellite_layer = L.basemap_to_tiles(L.basemaps.Esri.WorldImagery)
        
        # Give them user-friendly names for the LayerControl toggle
        osm_layer.name = "Open Street Maps"
        satellite_layer.name = "Satellite"
        
        # Add layers to the map
        m.add_layer(osm_layer)
        m.add_layer(satellite_layer)
        
        # Add the Shapefile data layer to the map
        geo_data = L.GeoData(
            geo_dataframe=gdf,
            style={
                "color": "blue",
                "opacity": 1,
                "weight": 2,
                "fillOpacity": 0.2,
                "fillColor": "blue",
            },
            name="CO Bounding Box",
        )
        m.add_layer(geo_data)
        
        # Add a LayerControl so users can toggle between basemaps
        control = L.LayersControl(position="topright")
        m.add_control(control)
        
        # Force the map to fit the exact bounding box extent of the shapefile
        m.fit_bounds(bounds)
        
        return m