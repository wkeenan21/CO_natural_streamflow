import geopandas as gpd
import ipyleaflet as L
import ipywidgets as widgets
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
            interactive=False,
            name="Upper Colorado River Basin",
        )

        # Style points as black circles instead of blue markers
        gages_layer = L.GeoData(
            geo_dataframe=gages,
            point_style={
                "radius": 4,
                "color": "blue",
                "fillColor": "grey",
                "fillOpacity": 0.5,
                "weight": 1,
            },
            name="Streamflow Gages",
        )

        # 1. Initialize the map with scroll_wheel_zoom=True
        m = L.Map(
            center=(center_lat, center_lon), 
            zoom=10, 
            layers=[gages_layer, ucol_layer, osm_layer],
            scroll_wheel_zoom=True
        )

        # 2. Make popups for gages
        # 1. Initialize an empty Popup and attach it to the map
        popup = L.Popup(
            location=[41.107166, -104.970417],
            child=widgets.HTML(value="Click a gage"),
            close_button=True,
            auto_close=True,
            close_on_escape_key=True
        )
        m.add(popup)

       # 2. Setup your click handler function (remove the global popup variable)
        def gage_click(event=None, feature=None, id=None, **kwargs):
            # Extract properties from the clicked feature safely
            properties = feature.get('properties', {})
            name = properties.get('name', 'Unknown')
            value = properties.get('gage', 'No data')
            
            # Extract coordinates and flip from [lon, lat] to [lat, lon]
            coords = feature['geometry']['coordinates']
            lat_lon = [coords[1], coords[0]]
            
            # Create a fresh Popup instance on every single click
            new_popup = L.Popup(
                location=lat_lon,
                child=widgets.HTML(value=f"<b>{name}</b><br>Gage:<b>USGS-{value}</b><br>"),
                close_button=True,
                auto_close=True,
                close_on_escape_key=True
            )
            
            # Add the fresh popup to the map
            m.add(new_popup)
            print(f'clicking: {coords, name, value}')

        # 3. Attach the click event to your GeoData layer
        gages_layer.on_click(gage_click)

        # Add the layer control safely now that the layers exist
        control = L.LayersControl(position="topright")
        m.add_control(control)

        # Fit map to bounds
        m.fit_bounds(bounds)

        return m