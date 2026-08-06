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
basins = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\all_UCOL_basins.parquet").to_crs(epsg=4326)
dvrs = gpd.read_parquet(r"shiny-app\ucol_natural\spatial_data\ucrb_diversion_master_table.parquet").to_crs(epsg=4326) 

ts_dir = r'shiny-app\ucol_natural\timeseries'

# Global Color Mapping for Diversions / Consumptive Use
CU_COLORS = {
    'irrigation': '#e6ab02', 'municipal': '#7570b3', 'interbasin': '#d95f02',
    'industrial': '#666666', 'hydropower': '#1b7837', 'intrabasin': '#a6761d', 'transbasin': '#e7298a'
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
ui.page_opts(title="Upper Colorado River Basin Naturalized Streamflow", fillable=True)

with ui.layout_column_wrap(width=1/2):
    
    # LEFT PANEL: Map
    with ui.card():
        ui.card_header("USGS Streamflow Gages and Watersheds")

        # Top-left search controls
        ui.p("Select a gage or enter USGS gage ID here:", style="margin-bottom: 5px; font-weight: 500;")
        with ui.div(style="display: flex; gap: 10px; align-items: center; margin-bottom: 5px;"):
            ui.input_text("gage_search_id", None, placeholder="e.g. 09010500")
            ui.input_action_button("btn_search_gage", "See data")

        @render.text
        def search_error():
            err = search_error_msg.get()
            return err if err else ""

        # Initialize Base Map
        m = L.Map(center=(center_lat, center_lon), zoom=10, scroll_wheel_zoom=True)
        osm_layer = L.basemap_to_tiles(L.basemaps.OpenStreetMap.Mapnik)
        osm_layer.name = "Open Street Maps"

        ucol_layer = L.GeoData(
            geo_dataframe=ucol,
            style={"color": "black", "opacity": 1, "weight": 2, "fillOpacity": 0.0, "fillColor": "black", 'interactive': False},
            interactive=False,
            name="Upper Colorado River Basin",
        )

        gages_layer = L.GeoData(
            geo_dataframe=gages,
            point_style={"radius": 4, "color": "blue", "fillColor": "grey", "fillOpacity": 0.5, "weight": 1},
            name="Streamflow Gages",
        )

        m.add_layer(osm_layer)
        m.add_layer(gages_layer)
        m.add_layer(ucol_layer)

        control = L.LayersControl(position="topright")
        m.add_control(control)
        m.fit_bounds(bounds)

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

            # Clear existing Watershed, Diversions, and Popup layers
            for layer in list(m.layers):
                if layer.name in ["Active Watershed", "Diversions"] or isinstance(layer, L.Popup):
                    m.remove_layer(layer)

            # Watershed Handling
            selected_basin_df = basins[basins['gage'] == gage]
            if not selected_basin_df.empty:
                watershed_layer = L.GeoData(
                    geo_dataframe=selected_basin_df,
                    style={"color": "blue", "weight": 2, "fillColor": "lightblue", "fillOpacity": 0.25, 'interactive': False},
                    interactive=False,
                    name="Active Watershed"
                )
                m.add_layer(watershed_layer)

                # Diversions Handling
                selected_dvrs = gpd.clip(dvrs, selected_basin_df)
                if not selected_dvrs.empty:
                    selected_dvrs = selected_dvrs.copy()
                    selected_dvrs['fillColor'] = selected_dvrs['siteUse'].map(CU_COLORS).fillna('#666666')
                    selected_dvrs['color'] = selected_dvrs['fillColor']
                    selected_dvrs['radius'] = 4
                    selected_dvrs['fillOpacity'] = 0.7
                    selected_dvrs['weight'] = 1

                    diversions_layer = L.GeoData(
                        geo_dataframe=selected_dvrs,
                        point_style={"type": "circle"}, 
                        style={"radius": 4, "fillOpacity": 0.2, "weight": 1, "color": "#666666"},
                        name="Diversions"
                    )

                    # Hover event for Diversion Points (siteName Popup)
                    dvr_hover_popup = L.Popup(child=widgets.HTML(), close_button=False, auto_close=False)

                    def dvr_hover(event=None, feature=None, **kwargs):
                        if feature and 'properties' in feature:
                            site_name = feature['properties'].get('siteName', 'N/A')
                            coords = feature.get('geometry', {}).get('coordinates', [])
                            if event == 'mouseover' and len(coords) >= 2:
                                dvr_hover_popup.location = [coords[1], coords[0]]
                                dvr_hover_popup.child.value = f"<div style='font-size:11px; padding:2px;'><b>{site_name}</b></div>"
                                if dvr_hover_popup not in m.layers:
                                    m.add_layer(dvr_hover_popup)
                            elif event == 'mouseout':
                                if dvr_hover_popup in m.layers:
                                    m.remove_layer(dvr_hover_popup)

                    diversions_layer.on_hover(dvr_hover)
                    m.add_layer(diversions_layer)

            # Create fresh Popup for selected gage location (Fixes location bug)
            popup_html = widgets.HTML(value=f"<b>{name}<br>USGS-{gage}</b><br>")
            new_popup = L.Popup(
                location=[lat, lon],
                child=popup_html,
                close_button=True,
                auto_close=True,
                close_on_escape_key=True
            )
            m.add_layer(new_popup)
            m.center = (lat, lon)
            return True

        # Map Click Event Handler
        def gage_click(event=None, feature=None, id=None, **kwargs):
            if feature and 'properties' in feature:
                gage = feature['properties'].get('gage')
                if gage:
                    select_gage_by_id(gage)

        gages_layer.on_click(gage_click)

        # Search Button Reactive Observer
        @reactive.effect
        @reactive.event(input.btn_search_gage)
        def _():
            gage_id = input.gage_search_id()
            if gage_id:
                select_gage_by_id(gage_id)

        @render_widget
        def map():
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
                    results = get_filtered_data()
                    if results is not None:
                        df_filtered, _ = results
                        return df_filtered.to_csv(index=False)

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
                title=dict(text=f'{name} (Gage: {gage}), Period of record {valid_range[0]} to {valid_range[1]}', font=dict(size=16)),
                xaxis=dict(title='Date', showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(title=f'Streamflow ({unit})', showgrid=True, gridcolor='#f0f0f0'),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis_type='date',
                hovermode="x unified",
                template="plotly_white"
            )

            return fig