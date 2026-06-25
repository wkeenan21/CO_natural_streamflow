import geopandas as gpd
import ipyleaflet as L
import ipywidgets as widgets
import os
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_widget

# 1. Load data
ucol = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\UCOL.parquet").to_crs(epsg=4326)
gages = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\all_UCOL_gages.parquet").to_crs(epsg=4326)
basins = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\all_UCOL_basins.parquet").to_crs(epsg=4326) # <-- New Basins Layer

ts_dir = r'shiny-app\ucol_natural\timeseries'

# Calculate bounds and center
minx, miny, maxx, maxy = ucol.total_bounds
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2
bounds = [[miny, minx], [maxy, maxx]]

# Track the selected gage reactively
selected_gage = reactive.Value(None)

# 2. Setup Page Layout
ui.page_opts(title="Upper Colorado River Basin Naturalized Streamflow", fillable=True)

with ui.layout_column_wrap(width=1/2):
    
    # LEFT PANEL: Map
    with ui.card():
        ui.card_header("USGS Streamflow Gages and Watersheds")

        @render_widget
        def map():
            osm_layer = L.basemap_to_tiles(L.basemaps.OpenStreetMap.Mapnik)
            osm_layer.name = "Open Street Maps"

            ucol_layer = L.GeoData(
                geo_dataframe=ucol,
                style={"color": "black", "opacity": 1, "weight": 2, "fillOpacity": 0.0, "fillColor": "black", 'interactive':False},
                interactive=False,
                name="Upper Colorado River Basin",
            )

            gages_layer = L.GeoData(
                geo_dataframe=gages,
                point_style={"radius": 4, "color": "blue", "fillColor": "grey", "fillOpacity": 0.5, "weight": 1},
                name="Streamflow Gages",
            )

            m = L.Map(center=(center_lat, center_lon), zoom=10, layers=[gages_layer, ucol_layer, osm_layer], scroll_wheel_zoom=True)

            def gage_click(event=None, feature=None, id=None, **kwargs):
                properties = feature.get('properties', {})
                name = properties['name']
                gage = properties['gage']
                
                selected_gage.set({"gage": gage, "name": name})
                
                # -----------------------------------------------------------
                # DYNAMIC WATERSHED HANDLING
                # -----------------------------------------------------------
                # Look for an existing watershed layer and clear it out first
                for layer in list(m.layers):
                    if layer.name == "Active Watershed":
                        m.remove_layer(layer)
                
                # Filter the basins dataframe for the clicked gage ID
                selected_basin_df = basins[basins['gage'] == gage]
                
                if not selected_basin_df.empty:
                    # Construct a new GeoData layer for the polygon outline
                    watershed_layer = L.GeoData(
                        geo_dataframe=selected_basin_df,
                        style={
                            "color": "blue",         # Blue outline
                            "weight": 2,
                            "fillColor": "lightblue", # Very light blue fill
                            "fillOpacity": 0.25,
                            'interactive':False,
                        },
                        interactive=False,          # Allows users to still click items underneath it
                        name="Active Watershed"
                    )
                    # Add it to the map canvas dynamically
                    m.add_layer(watershed_layer)
                # -----------------------------------------------------------
                
                coords = feature['geometry']['coordinates']
                lat_lon = [coords[1], coords[0]]
                
                new_popup = L.Popup(
                    location=lat_lon,
                    child=widgets.HTML(value=f"<b>{name}</b><br>Gage:<b>USGS-{gage}</b><br>"),
                    close_button=True, auto_close=True, close_on_escape_key=True
                )
                m.add(new_popup)

            gages_layer.on_click(gage_click)
            control = L.LayersControl(position="topright")
            m.add_control(control)
            m.fit_bounds(bounds)
            return m

    # RIGHT PANEL: Date, Units, Plot & Export
    with ui.card():
        ui.card_header("Streamflow and Consumptive Use Timeseries")
        
        with ui.div(style="display: flex; justify-content: space-between; align-items: flex-start; gap: 15px; margin-bottom: 10px;"):
            with ui.div(style="display: flex; gap: 15px; flex-grow: 1;"):
                ui.input_date_range(
                    "date_range", 
                    "Select Date Range:", 
                    start="2021-10-01", 
                    end="2022-09-30"
                )
                
                ui.input_select(
                    "units", 
                    "Select Flow Units:", 
                    choices={"cfs": "cfs", "cms": "cms", "mmd": "mmd (millimeters / day)"},
                    selected="cfs"
                )
            
            with ui.div(style="margin-top: 24px;"):
                @render.download(
                    label="Export CSV", 
                    filename=lambda: f"USGS-{selected_gage.get()['gage'] if selected_gage.get() else 'data'}_extracted.csv"
                )
                def download_data():
                    df_filtered = get_filtered_data()
                    if df_filtered is not None:
                        return df_filtered.to_csv(index=False)

        @reactive.calc
        def get_filtered_data():
            gage_info = selected_gage.get()
            if gage_info is None:
                return None
                
            gage = gage_info["gage"]
            csv_path = os.path.join(ts_dir, f"{gage}.csv")
            
            if not os.path.exists(csv_path):
                return None
                
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])

            unit_choice = input.units()
            unit_cols = [col for col in df.columns if col.endswith(f'_{unit_choice}')]
            columns = ['date'] + unit_cols
            
            start_date, end_date = input.date_range()
            mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
            filtered_df = df.loc[mask].copy()[columns]
            
            filtered_df.index = filtered_df['date']
            return filtered_df

        @render_widget
        def plot_flows():
            CU_COLORS = {
                'irrigation': '#e6ab02', 'municipal': '#7570b3', 'interbasin': '#d95f02',
                'industrial': '#666666', 'hydropower': '#1b7837', 'intrabasin': '#a6761d', 'transbasin': '#e7298a'
            }
            
            gage_info = selected_gage.get()
            df = get_filtered_data()
            unit = input.units()
            
            if gage_info is None or df is None:
                fig = go.Figure()
                fig.add_annotation(
                    text="Click a stream gage on the map to view its interactive hydrograph.",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="gray")
                )
                fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                return fig
            elif df.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data for these dates at this stream gage",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="gray")
                )
                fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                return fig
            
            name = gage_info['name']
            gage = gage_info['gage']
            
            fig = go.Figure()

            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            print(df['date'].min())

            cu_types = ['irrigation', 'municipal', 'interbasin', 'industrial', 'hydropower', 'intrabasin', 'transbasin']
            cu_labels = {
                'irrigation':'irrigation', 
                'municipal':'municipal', 
                'interbasin':'transbasin exported (outside CRB)', 
                'industrial':'industrial', 
                'hydropower':'hydropower', 
                'intrabasin':'transbasin exported (within CRB)', 
                'transbasin':'transbasin imported'
            }

            for cu_type in cu_types:
                col_name = f'{cu_type}_{unit}'
                if col_name in df.columns and df[col_name].sum() != 0:
                    
                    # Determine the correct stack group based on whether it's an import or export
                    if cu_type == 'transbasin':
                        stack_name = 'positive_stack'
                    else:
                        stack_name = 'negative_stack'
                        
                    fig.add_trace(go.Scatter(
                        x=df['date'], y=df[col_name],
                        name=cu_labels[cu_type], mode='lines',
                        line=dict(width=0.5, color=CU_COLORS.get(cu_type, '#cccccc')),
                        
                        # Use the dynamically assigned stack group here
                        stackgroup=stack_name, 
                        
                        fillcolor=CU_COLORS.get(cu_type, '#cccccc'), opacity=0.6,
                        hovertemplate='<b>%{hovertext}</b><br>Flow: %{y:.1f} ' + unit + '<extra></extra>',
                        hovertext=[cu_labels[cu_type]] * len(df)
                    ))

            q_nat = f'Q_NAT_{unit}'
            if q_nat in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df[q_nat], name=f'Naturalized Flow ({unit})', mode='lines',
                    line=dict(color='forestgreen', width=2),
                    hovertemplate='<b>Naturalized Flow</b><br>Date: %{x}<br>Flow: %{y:.1f} ' + unit + '<extra></extra>'
                ))

            q_obs = f'Q_{unit}'
            if q_obs in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df[q_obs], name=f'Observed Flow ({unit})', mode='lines',
                    line=dict(color='royalblue', width=1.5),
                    hovertemplate='<b>Observed Flow</b><br>Date: %{x}<br>Flow: %{y:.1f} ' + unit + '<extra></extra>'
                ))

            fig.update_layout(
                title=dict(text=f'{name} (Gage: {gage})', font=dict(size=16)),
                xaxis=dict(title='Date', showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(title=f'Streamflow ({unit})', showgrid=True, gridcolor='#f0f0f0'),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis_type='date',
                hovermode="x unified",
                template="plotly_white"
            )

            fig.update_xaxes(
                dtick="M1",      # Tick label every month
            )

            return fig