import geopandas as gpd
import ipyleaflet as L
import os
from shiny.express import input, render, ui
from shinywidgets import render_widget

# 1. Load and project the shapefile correctly
ucol = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\UCOL.parquet").to_crs(epsg=4326)
gages = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\all_UCOL_gages.parquet").to_crs(epsg=4326)

ts_dir = r'shiny-app\ucol_natural\timeseries' # 09163500.csv

# Calculate bounds and center
minx, miny, maxx, maxy = ucol.total_bounds
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2
bounds = [[miny, minx], [maxy, maxx]]

# 2. Setup the Shiny Page
ui.page_opts(title="Colorado Map View", fillable=True)

with ui.card():
    ui.card_header("Upper Colorado River Basin Naturalized Streamflow")

    @render_widget
    def map():
        # Setup the layers first
        osm_layer = L.basemap_to_tiles(L.basemaps.OpenStreetMap.Mapnik)
        osm_layer.name = "Open Street Maps"

        ucol_layer = L.GeoData(
            geo_dataframe=ucol,
            style={
                "color": "black",
                "opacity": 1,
                "weight": 2,
                "fillOpacity": 0.0,
                "fillColor": "black",
            },
            name="Upper Colorado River Basin",
        )

        # Style points as black circles instead of blue markers
        gages_layer = L.GeoData(
            geo_dataframe=gages,
            style={
                "color": "black",
                "fillColor": "black",
                "opacity": 1,
                "weight": 1,
                "fillOpacity": 1,
            },
            point_style={
                "radius": 5,
                "color": "black",
                "fillColor": "black",
                "fillOpacity": 1,
                "weight": 1,
            },
            name="Streamflow Gages",
        )

        # 1. Initialize the map with scroll_wheel_zoom=True
        m = L.Map(
            center=(center_lat, center_lon), 
            zoom=10, 
            layers=[osm_layer, ucol_layer, gages_layer],
            scroll_wheel_zoom=True
        )

        # 3. Add dynamic popup interaction on click
        def handle_click(event, feature, **kwargs):
            # Extract geometry coordinates for popup placement
            coords = feature['geometry']['coordinates']
            # GeoJSON coordinates are [lon, lat], Leaflet wants [lat, lon]
            lat_lon = [coords[1], coords[0]]
            
            # Build an HTML string out of the point's attributes
            props = feature['properties']
            html_content = ui.HTML(
                f"<h4>Gage Info</h4>" + 
                "".join(f"<b>{k}:</b> {v}<br>" for k, v in props.items())
            )
            
            # Create and add the popup to the map
            popup = L.Popup(
                location=lat_lon,
                child=html_content,
                close_button=True,
                auto_close=True,
                close_on_escape_key=True
            )
            m.add_layer(popup)

        # Attach click event listener to the gages layer
        gages_layer.on_click(handle_click)

        # Add the layer control safely now that the layers exist
        control = L.LayersControl(position="topright")
        m.add_control(control)

        # Fit map to bounds
        m.fit_bounds(bounds)

        return m