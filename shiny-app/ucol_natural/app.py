import geopandas as gpd
import ipyleaflet as L
import os
from shiny.express import input, render, ui
from shinywidgets import render_widget

# 1. Load and project the shapefile correctly
gdf = gpd.read_file(r"shiny-app\ucol_natural\spatial_data\UCOL.shp")

if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)

# Calculate bounds and center
minx, miny, maxx, maxy = gdf.total_bounds
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2
bounds = [[miny, minx], [maxy, maxx]]

# 2. Setup the Shiny Page
ui.page_opts(title="Colorado Map View", fillable=True)

with ui.card():
    ui.card_header("Colorado Boundary Map")

    @render_widget
    def map():
        # Setup the layers first
        osm_layer = L.basemap_to_tiles(L.basemaps.OpenStreetMap.Mapnik)
        osm_layer.name = "Open Street Maps"
        
        satellite_layer = L.basemap_to_tiles(L.basemaps.Esri.WorldImagery)
        satellite_layer.name = "Satellite"

        geo_data = L.GeoData(
            geo_dataframe=gdf,
            style={
                "color": "black",
                "opacity": 1,
                "weight": 2,
                "fillOpacity": 0.0,
                "fillColor": "black",
            },
            name="Upper Colorado River Basin",
        )

        # Initialize the map with the layers already loaded inside it
        # This prevents the 'state_change' frontend race condition!
        m = L.Map(
            center=(center_lat, center_lon), 
            zoom=6, 
            layers=[osm_layer, satellite_layer, geo_data]
        )

        # Add the layer control safely now that the layers exist
        control = L.LayersControl(position="topright")
        m.add_control(control)

        # Fit map to bounds
        m.fit_bounds(bounds)

        return m