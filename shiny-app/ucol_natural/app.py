import geopandas as gpd
import ipyleaflet as L
import os
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_widget
import ipywidgets as widgets
import numpy as np
import asyncio

# 1. Load data
ucol = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\UCOL.parquet").to_crs(epsg=4326)
gages = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\all_UCOL_gages.parquet").to_crs(epsg=4326)
basins = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\all_UCOL_basins.parquet").to_crs(epsg=4326)
dvrs = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\ucrb_diversion_master_table.parquet").to_crs(epsg=4326) 

# make gage column
gages['gage'] = gages.index
gages.reset_index(inplace=True, drop=True)
basins['gage'] = basins.index
basins.reset_index(inplace=True, drop=True)

# Build search choices mapping: {gage_id: "gage_id - gage_name"}
gages = gages.sort_values(by='name', ascending=True)
gage_choices = {
    str(row['gage']): f"{row['gage']} - {row['name']}" 
    for _, row in gages.iterrows()
}

ts_dir = r'shiny-app\ucol_natural\timeseries'

# Global Color Mapping for Diversions / Consumptive Use
CU_COLORS = {
    'irrigation': '#7aab74', 'municipal': '#7570b3', 'interbasin': '#d95f02',
    'industrial': '#666666', 'hydropower': '#1b7837', 'intrabasin': '#e7298a', 'transbasin': '#e7298a',
    'evap':'#e6ab02'
}

# Calculate bounds and center
minx, miny, maxx, maxy = ucol.total_bounds
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2
bounds = [[miny, minx], [maxy, maxx]]

# Track the selected gage reactively and search status
selected_gage = reactive.Value(None)
search_error_msg = reactive.Value("")

# 2. Setup Page Layout
ui.page_opts(title=ui.div(
        "Upper Colorado River Basin Streamflow, Diversions, and Consumptive Use",
        style="text-align: center; width: 100%; font-size: 1.4rem; font-weight: 700;"
    ), fillable=True)

with ui.layout_column_wrap(width=1/2):
    
    # LEFT PANEL: Map
    with ui.card():
        ui.card_header("Select a gage on the map or search by gage ID or name")

        # Top-left search controls using Selectize for autocomplete
        with ui.div(style="margin-bottom: 2px; width: 100%;"):
            ui.input_selectize(
                "gage_search_id", 
                label=None, 
                choices=gage_choices, 
                selected=None,
                width='100%',
                options={
                    "placeholder": "Search for a streamflow gage by name or ID #",
                    "allowEmptyOption": True
                }
            )

        @render.text
        def search_error():
            err = search_error_msg.get()
            return err if err else ""

        @render_widget
        def map():
            # Initialize Base Map 
            osm_layer = L.basemap_to_tiles(L.basemaps.OpenStreetMap.Mapnik)
            osm_layer.name = "Open Street Maps"
            ucol_style = {"color": "black", "opacity": 1, "weight": 2, "fillOpacity": 0.0, "fillColor": "black", 'interactive': False}
            ucol_layer = L.GeoData(
                geo_dataframe=ucol,
                style=ucol_style,
                interactive=False,
                name="Upper Colorado River Basin",
            )

            gage_style = {"radius": 5, "color": "blue", "fillColor": "grey", "fillOpacity": 0.8, "weight": 1}
            gages_layer = L.GeoData(
                geo_dataframe=gages,
                point_style=gage_style,
                name="Streamflow Gages",
            )

            wshed_style={"color": "blue", "weight": 2, "fillColor": "lightblue", "fillOpacity": 0.25, 'interactive': False}
            diversion_legend_style = {"radius": 4, "fillOpacity": 0.5, "weight": 1, "color": 'gray', 'fillColor':'gray'}

            m = L.Map(center=(center_lat, center_lon), zoom=6, scroll_wheel_zoom=True, layers=[osm_layer, gages_layer, ucol_layer])

            # Centralized Selection Handler (Map click & Text input)
            def select_gage_by_id(gage_id):
                gage_str = str(gage_id).strip()
                match = gages[gages['gage'].astype(str).str.strip() == gage_str]
                
                if match.empty:
                    search_error_msg.set("invalid gage ID")
                    return False

                search_error_msg.set("")
                row = match.iloc[0]
                gage = row['gage']
                name = row['name']
                lat, lon = row.geometry.y, row.geometry.x

                selected_gage.set({"gage": gage, "name": name})

                # Clear existing Watershed and Diversions layers
                names = list(CU_COLORS.keys())
                names.append('Active Watershed')
                for layer in list(m.layers):
                    if layer.name in names:
                        m.remove_layer(layer)

                # Watershed Handling
                selected_basin_df = basins[basins['gage'] == gage]
                if not selected_basin_df.empty:
                    watershed_layer = L.GeoData(
                        geo_dataframe=selected_basin_df,
                        style=wshed_style,
                        interactive=False,
                        name="Active Watershed"
                    )
                    m.add_layer(watershed_layer)

                    # Diversions Handling
                    selected_dvrs = gpd.clip(dvrs, selected_basin_df).copy()
                    if not selected_dvrs.empty:
                        for cu_type in selected_dvrs['siteUse'].unique():
                            type_dvrs = selected_dvrs[selected_dvrs['siteUse']==cu_type].copy()

                            color = CU_COLORS[cu_type]
                            
                            diversion_style = {"radius": 4, "fillOpacity": 0.5, "weight": 1, "color": color}
                            diversions_layer = L.GeoData(
                                geo_dataframe=type_dvrs,
                                point_style={"type": "circle"}, 
                                style=diversion_style,
                                interactive=False,
                                name=cu_type
                            )
                            diversions_layer.on_click(div_click)
                            m.add_layer(diversions_layer)

                m.center = (lat, lon)

                return True

            def div_click(event=None, feature=None, **kwargs):
                if not feature or "properties" not in feature:
                    return

                props = feature.get("properties", {})
                point_name = props.get("siteName", "Unknown Location")
                coords = feature.get("geometry", {}).get("coordinates", [])

                if len(coords) < 2:
                    return

                lat, lon = coords[1], coords[0]

                for layer in list(m.layers):
                    if isinstance(layer, L.Popup):
                        m.remove_layer(layer)

                popup_content = widgets.HTML(
                    value=f"{point_name}"
                )

                popup = L.Popup(
                    location=[lat, lon],
                    child=popup_content,
                    close_button=True,
                    auto_close=True,
                )
                m.add_layer(popup)

            # Map Click Event Handler
            def gage_click(event=None, feature=None, id=None, **kwargs):
                if feature and 'properties' in feature:
                    gage = feature['properties'].get('gage')
                    if gage:
                        select_gage_by_id(gage)

            gages_layer.on_click(gage_click)

            # Reactive Observer on selectize drop-down change
            @reactive.effect
            @reactive.event(input.gage_search_id)
            def _():
                gage_id = input.gage_search_id()
                if gage_id:
                    select_gage_by_id(gage_id)

            for layer in list(m.layers):
                print('layer:', layer.name)
            for control in list(m.controls):
                print('control:', control)

# HTML content representing each layer's style
            legend_html = f"""
            <div style="
                background-color: white; 
                padding: 10px; 
                border-radius: 5px; 
                border: 1px solid #ccc;
                box-shadow: 0 0 5px rgba(0,0,0,0.2);
                font-family: Arial, sans-serif;
                font-size: 12px;
                line-height: 20px;
            ">
                <b style="font-size: 13px;">Legend</b><br/>
                
                <!-- Lines / Polygons -->
                <div style="display: flex; align-items: center; margin-top: 5px;">
                    <span style="display: inline-block; width: 18px; height: 3px; background-color: {ucol_style['color']}; margin-right: 8px;"></span>
                    <span>Upper Colorado Basin</span>
                </div>
                
                <div style="display: flex; align-items: center; margin-top: 4px;">
                    <span style="display: inline-block; width: 16px; height: 12px; border: 2px solid {wshed_style['color']}; background-color: rgba(173, 216, 230, 0.25); margin-right: 8px;"></span>
                    <span>Active Watershed</span>
                </div>

                <!-- Points -->
                <div style="display: flex; align-items: center; margin-top: 4px;">
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; border: 1px solid {gage_style['color']}; background-color: {gage_style['fillColor']}; margin-right: 11px; margin-left: 3px;"></span>
                    <span>Streamflow Gage</span>
                </div>

                <div style="display: flex; align-items: center; margin-top: 4px;">
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; border: 1px solid {diversion_legend_style['color']}; background-color: {diversion_legend_style['fillColor']}; margin-right: 11px; margin-left: 3px;"></span>
                    <span>Diversion</span>
                </div>
            </div>
            """

            # Create an ipywidgets HTML control and wrap it as a map control
            legend_widget = widgets.HTML(value=legend_html)
            legend_control = L.WidgetControl(widget=legend_widget, position="bottomright")
            
            m.add_control(legend_control)

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
                @render.download_button(
                    label="Export CSV", 
                    filename=lambda: f"USGS-{selected_gage.get()['gage'] if selected_gage.get() else 'data'}_extracted.csv"
                )
                async def download_data():
                    results = get_filtered_data()
                    await asyncio.sleep(0.25)
                    if results is not None:
                        df_filtered, _ = results
                        yield df_filtered.to_csv(index=False)

        @reactive.calc
        def get_filtered_data():
            gage_info = selected_gage.get()
            unit_choice = input.units()
            start_date, end_date = input.date_range()

            if gage_info is None:
                return None
            gage = gage_info["gage"]
            csv_path = os.path.join(ts_dir, f"{gage}.csv")
            if not os.path.exists(csv_path):
                return None

            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df.index = df['date']

            Q_col = f'Q_{unit_choice}'
            first_valid = df[Q_col].first_valid_index()
            last_valid = df[Q_col].last_valid_index()

            if first_valid is not None and last_valid is not None:
                df = df.loc[first_valid:last_valid].copy()
            else:
                df = df.iloc[0:0]
            
            mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
            filtered_df = df.loc[mask].copy()

            first_valid_str = first_valid.strftime('%Y-%m-%d')
            last_valid_str = last_valid.strftime('%Y-%m-%d')

            if filtered_df.empty:
                return filtered_df, (first_valid_str, last_valid_str)

            unit_cols = [col for col in df.columns if col.endswith(f'_{unit_choice}')]
            columns = ['date'] + unit_cols
            filtered_df = filtered_df[columns]

            return filtered_df, (first_valid_str, last_valid_str)

        @render_widget
        def plot_flows():
            gage_info = selected_gage.get()
            unit = input.units()
            
            if gage_info is None:
                fig = go.Figure()
                fig.add_annotation(
                    text="Click a stream gage on the map to view its interactive hydrograph.",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="gray")
                )
                fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                return fig

            df, valid_range = get_filtered_data()
            df = df.copy()
            
            if df.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"No data for these dates at this stream gage. First date with flow data: {valid_range[0]}. Last date with flow data: {valid_range[1]}",
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

            # order it by precedence then reverse it so the labels work
            cu_types = ['irrigation', 'intrabasin', 'evap', 'interbasin', 'transbasin', 'municipal', 'industrial', 'hydropower']
            cu_types = reversed(cu_types)
            cu_labels = {
                'irrigation':'irrigation', 
                'municipal':'municipal', 
                'interbasin':'transbasin exported (outside CRB)', 
                'industrial':'industrial', 
                'hydropower':'hydropower', 
                'intrabasin':'transbasin exported (within CRB)', 
                'transbasin':'transbasin imported',
                'evap':'reservoir evaporation'
            }

            for cu_type in cu_types:
                col_name = f'{cu_type}_{unit}'
                if col_name in df.columns and df[col_name].sum() != 0:
                    if cu_type == 'transbasin':
                        stack_name = 'positive_stack'
                    else:
                        stack_name = 'negative_stack'
                        
                    fig.add_trace(go.Scatter(
                        x=df['date'], y=df[col_name],
                        name=cu_labels[cu_type], mode='lines',
                        line=dict(width=0.5, color=CU_COLORS.get(cu_type, '#cccccc')),
                        stackgroup=stack_name, 
                        fillcolor=CU_COLORS.get(cu_type, '#cccccc'), opacity=0.6,
                        hovertemplate='<b>%{hovertext}</b><br>%{y:.1f} ' + unit + '<extra></extra>',
                        hovertext=[cu_labels[cu_type]] * len(df)
                    ))

            q_nat = f'Q_NAT_{unit}'
            if q_nat in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df[q_nat], name='observed flow + consumptive use', mode='lines',
                    line=dict(color='forestgreen', width=2),
                    hovertemplate='<b>observed flow + consumptive use</b><br> %{y:.1f} ' + unit + '<extra></extra>'
                ))

            q_obs = f'Q_{unit}'
            if q_obs in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df[q_obs], name='observed flow', mode='lines',
                    line=dict(color='royalblue', width=1.5),
                    hovertemplate='<b>observed flow</b><br>%{y:.1f} ' + unit + '<extra></extra>'
                ))
            
            # check for negative natural flows
            if q_nat in df.columns:
                negQ = np.any(df[q_nat] < 0)
                if negQ:
                    subtitle=dict(text='NOTE: Water imports and subsequent dam storage may produce negative observed flow + consumptive use', font=dict(size=12))
                else:
                    subtitle=dict()
            else:
                subtitle=dict()

            fig.update_layout(
                title=dict(text=f'{name} (USGS-{gage}), Period of record {valid_range[0]} to {valid_range[1]}', font=dict(size=16), subtitle=subtitle),
                xaxis=dict(title='Date', showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(title=f'Streamflow ({unit})', showgrid=True, gridcolor='#f0f0f0'),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis_type='date',
                hovermode="x unified",
                template="plotly_white"
            )
            # go again
            return fig