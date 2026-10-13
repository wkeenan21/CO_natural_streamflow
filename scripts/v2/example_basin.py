
import os
import ast
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import contextily as cx

cwd = os.getcwd()
ncwd = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow'
appcwd = os.path.join(cwd, r'shiny-app\ucol_natural')
appDir = os.path.join(appcwd, r'timeseries') # does not have each individual diversion
proj_crs = 5070
basins_path = os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet')
basins = gpd.read_parquet(basins_path)
basins.geometry = basins.geometry.to_crs(proj_crs) # need this for spatial joins
wdvrsDir = os.path.join(ncwd, r'data\timeseries\selected_w_diversion') # has everything
allDir = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\gr_snodas_flow0'
res = gpd.read_parquet(fr"{cwd}\shiny-app\ucol_natural\spatial_data\all_UCOL_reservoirs.parquet").to_crs(epsg=4326)
res_centroid = gpd.read_parquet(fr"{cwd}\shiny-app\ucol_natural\spatial_data\all_UCOL_reservoirs_centroids.parquet").to_crs(epsg=4326)
# filter reservoirs...
res = res[res.areasqkm>0.5].copy()
res_centroid = res_centroid[res_centroid.areasqkm>0.5].copy()
# fix geometry
res.geometry = res.geometry.make_valid()

def fix_gage_id(id_val):
    # 1. Convert to string base representation without precision loss
    if isinstance(id_val, float):
        id_str = str(int(round(id_val)))
    elif isinstance(id_val, str) and '.' in id_val:
        try:
            id_str = str(int(round(float(id_val))))
        except ValueError:
            id_str = id_val.strip()
    else:
        id_str = str(id_val).strip()

    # 2. Clean prefixes
    if 'USGS-' in id_str:
        id_str = id_str.replace('USGS-', '')

    # 3. Pad 7-digit IDs with a leading zero
    if len(id_str) == 7 and id_str.isdigit():
        return id_str.zfill(8)
    elif id_str == '93710009':
        return id_str.zfill(9)

    return id_str


# --- Shared Setup & Parameters ---
lfs = 11
tfs = 14
xdim = 14  # Slightly widened for a 2x2 grid
ydim = 12  # Height expanded for 2 rows
year = 2012

# Load shared spatial data once
gages = gpd.read_parquet(rf"{cwd}\shiny-app\ucol_natural\spatial_data\all_UCOL_gages.parquet").to_crs(epsg=4326)
basins = gpd.read_parquet(rf"{cwd}\shiny-app\ucol_natural\spatial_data\all_UCOL_basins.parquet").to_crs(epsg=4326)
dvrs = gpd.read_parquet(rf"{cwd}\shiny-app\ucol_natural\spatial_data\ucrb_diversion_master_table.parquet").to_crs(epsg=4326)
res = gpd.read_parquet(rf"{cwd}\shiny-app\ucol_natural\spatial_data\all_UCOL_reservoirs.parquet").to_crs(epsg=4326)
ucol = gpd.read_parquet(fr"{cwd}\shiny-app\ucol_natural\spatial_data\UCOL.parquet").to_crs(epsg=4326)

CU_COLORS = {
    'irrigation': '#7aab74', 'municipal': '#7570b3', 'interbasin': '#d95f02',
    'industrial': '#666666', 'hydropower': '#1b7837', 'intrabasin': '#e7298a', 'transbasin': '#e7298a',
    'evap':'#e6ab02'
}

name_dict = basins['name'].to_dict()

# =============================================================
# DATA PREPARATION: DIVERSION (Panels A & B)
# =============================================================
gage_div = '09346400'
start_div = f'{year - 1}-10-01'
end_div = f'{year}-10-01'

df_div = pd.read_csv(os.path.join(wdvrsDir, f"{gage_div}.csv"), index_col='date', parse_dates=['date'])
df_div = df_div.loc[start_div:end_div]
name_div = name_dict.get(gage_div, gage_div)

target_gage_div = gages.loc[[gage_div]] if gage_div in gages.index else gages[gages['gage'] == gage_div]
target_basin_div = basins.loc[[gage_div]] if gage_div in basins.index else basins[basins['gage'] == gage_div]

dvrs_in_basin = gpd.sjoin(dvrs, target_basin_div, how='inner', predicate='within')

target_basin_div_3857 = target_basin_div.to_crs(epsg=3857)
target_gage_div_3857 = target_gage_div.to_crs(epsg=3857)
dvrs_in_basin_3857 = dvrs_in_basin.to_crs(epsg=3857)

# =============================================================
# DATA PREPARATION: RESERVOIR (Panels C & D)
# =============================================================
start_res = f'{year - 1}-10-01'
end_res = f'{year}-09-30'
outflowgage = '09355500'
name_res = name_dict.get(outflowgage, outflowgage)

rdf = pd.read_csv(r"C:\Users\C830645719\Documents\github-repos\CO_natural_streamflow\data\nat_results.csv")
rdf['gage'] = rdf['gage'].apply(fix_gage_id)
rdf = rdf[rdf['gage'] == outflowgage].copy()

inflow_col = f'{year}_inflows'
rdf[inflow_col] = rdf[inflow_col].apply(ast.literal_eval)
igages = rdf[inflow_col].iloc[0]

target_basin_res = basins.loc[[outflowgage]] if outflowgage in basins.index else basins[basins['gage'] == outflowgage]
outflow_gage_pt = gages.loc[[outflowgage]] if outflowgage in gages.index else gages[gages['gage'] == outflowgage]

inflow_basins = basins.loc[basins.index.isin(igages)] if igages[0] in basins.index else basins[basins['gage'].isin(igages)]
inflow_gages_pts = gages.loc[gages.index.isin(igages)] if igages[0] in gages.index else gages[gages['gage'].isin(igages)]

inflow_union = inflow_basins.unary_union
ungaged_area = target_basin_res.overlay(gpd.GeoDataFrame(geometry=[inflow_union], crs=target_basin_res.crs), how='difference')

res_in_basin = gpd.sjoin(res, target_basin_res[['geometry']], how='inner', predicate='intersects')
res_in_basin['calc_area'] = res_in_basin.to_crs(epsg=3857).geometry.area
largest_res = res_in_basin.sort_values(by='calc_area', ascending=False).iloc[[0]]

crs_3857 = 'EPSG:3857'
target_basin_res_3857 = target_basin_res.to_crs(crs_3857)
ungaged_area_3857 = ungaged_area.to_crs(crs_3857)
inflow_basins_3857 = inflow_basins.to_crs(crs_3857)
outflow_gage_3857 = outflow_gage_pt.to_crs(crs_3857)
inflow_gages_3857 = inflow_gages_pts.to_crs(crs_3857)
largest_res_3857 = largest_res.to_crs(crs_3857)

df_out = pd.read_csv(os.path.join(wdvrsDir, f"{outflowgage}.csv"), index_col='date', parse_dates=['date'])
df_out = df_out.loc[start_res:end_res]

greens = cm.get_cmap('Greens')(np.linspace(0.4, 0.85, len(igages)))
color_dict = dict(zip(igages, greens))

inflow_dfs = {}
inflow_sum = pd.Series(0.0, index=df_out.index)

for g_id in igages:
    inflow_file = os.path.join(wdvrsDir, f"{g_id}.csv")
    if os.path.exists(inflow_file):
        df_in = pd.read_csv(inflow_file, index_col='date', parse_dates=['date']).loc[start_res:end_res]
        inflow_dfs[g_id] = df_in
        inflow_sum = inflow_sum.add(df_in['Q_NAT2_cfs'], fill_value=0)

ungaged_inflow = df_out['Q_NAT2_cfs'] - inflow_sum


# =============================================================
# COMBINED 4-PANEL PLOTTING (2x2 Grid)
# =============================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(xdim, ydim), layout="constrained")

# Single main title for the entire combined figure
fig.suptitle(f'Upper San Juan Example Naturalization', fontsize=tfs + 2, fontweight='bold')

# -------------------------------------------------------------
# Panel A: Diversion Watershed Map
# -------------------------------------------------------------
target_basin_div_3857.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=2, label='Watershed Boundary')
target_gage_div_3857.plot(ax=ax1, color='blue', marker='^', markersize=120, zorder=5, label=f'Gage {gage_div}')

irrigation_dvrs = dvrs_in_basin_3857[dvrs_in_basin_3857['siteUse'] == 'irrigation']
transbasin_dvrs = dvrs_in_basin_3857[dvrs_in_basin_3857['siteUse'] == 'interbasin']

irrigation_dvrs.plot(ax=ax1, color='orange', marker='o', markersize=40, alpha=0.8, zorder=4, label='Irrigation Diversion')
transbasin_dvrs.plot(ax=ax1, color='red', marker='s', markersize=40, alpha=0.8, zorder=4, label='Transbasin Diversion')

cx.add_basemap(ax1, source=cx.providers.USGS.USTopo)
ax1.set_axis_off()
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=lfs)
ax1.set_title(f'A: {gage_div}: Diversion Impacted', loc='left', fontsize=tfs, fontweight='bold')
ax1.set_box_aspect(None)

cx.add_basemap(ax1, source=cx.providers.Esri.WorldTerrain)
usgs_hydro_wms = (
    "https://basemap.nationalmap.gov/arcgis/rest/services/"
    "USGSHydroCached/MapServer/tile/{z}/{y}/{x}"
)
# Overlay hydrography onto your plot
cx.add_basemap(ax1, source=usgs_hydro_wms)

# -------------------------------------------------------------
# Panel B: Diversion Naturalization Graph
# -------------------------------------------------------------
ax2.plot(df_div.index, df_div['Q_cfs'], label='Observed Q', color='blue')
ax2.plot(df_div.index, df_div['Q_NAT2_cfs'], label='Estimated Natural Q', color='green')
ax2.plot(df_div.index, df_div['lagged_rf_cfs'] * -1, label='Return Flow Q', color='green', linestyle=':')
ax2.plot(df_div.index, df_div['irrigation_diversion_cfs'], label='Irrigation Q', color='orange')
ax2.plot(df_div.index, df_div['interbasin_cfs'], label='Transbasin Diversion Q', color='red')

ax2.set_xlabel('Date')
ax2.set_ylabel('Q (cfs)')
ax2.grid(True)
ax2.legend(fontsize=lfs)
ax2.set_title(f'B', loc='left', fontsize=tfs, fontweight='bold')

# -------------------------------------------------------------
# Panel C: Reservoir Watershed Map
# -------------------------------------------------------------
ungaged_area_3857.plot(ax=ax3, facecolor='orange', alpha=0.3, edgecolor='darkorange', linewidth=1, label='Ungaged Inflow Area')
target_basin_res_3857.plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=2, label='Total Basin Boundary')

count = 0
for g_id, row in inflow_basins_3857.iterrows():
    inflow_basins_3857.loc[[g_id]].plot(
        ax=ax3, 
        facecolor='green', 
        alpha=0.4, 
        edgecolor='darkgreen', 
        linewidth=1.5, 
    )

largest_res_3857.plot(ax=ax3, color='cyan', edgecolor='blue', linewidth=0.8, zorder=5, label='Navajo Reservoir')
outflow_gage_3857.plot(ax=ax3, color='black', marker='^', markersize=130, zorder=6, label=f'Gage {outflowgage}')
inflow_gages_3857.plot(ax=ax3, color='green', marker='o', markersize=50, zorder=6, label='Inflow Gages')

cx.add_basemap(ax3, source=cx.providers.Esri.WorldTerrain)
usgs_hydro_wms = (
    "https://basemap.nationalmap.gov/arcgis/rest/services/"
    "USGSHydroCached/MapServer/tile/{z}/{y}/{x}"
)
# Overlay hydrography onto your plot
cx.add_basemap(ax3, source=usgs_hydro_wms)
ax3.set_axis_off()

# Build custom legend entries for Panel C
legend_handles = [
    mpatches.Patch(facecolor='none', edgecolor='black', linewidth=2, label=f'Watershed Boundary'),
    mpatches.Patch(facecolor='green', alpha=0.3, edgecolor='darkgreen', label='Gaged Inflows'),
    mpatches.Patch(facecolor='orange', alpha=0.3, edgecolor='darkorange', label='Ungaged Inflows'),
    mpatches.Patch(color='cyan', label='Navajo Reservoir'),
]

point_handles, point_labels = ax3.get_legend_handles_labels()
for h, l in zip(point_handles, point_labels):
    if l not in [p.get_label() for p in legend_handles]:
        legend_handles.append(h)

ax3.legend(handles=legend_handles, loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=lfs)
ax3.set_title(f'C: {outflowgage}: Reservoir Impacted', loc='left', fontsize=tfs, fontweight='bold')
ax3.set_box_aspect(None)

# -------------------------------------------------------------
# Panel D: Reservoir Hydrographs
# -------------------------------------------------------------
ax4.plot(df_out.index, df_out['Q_cfs'], label='Outflow Observed Q', color='blue', linewidth=1.8)
ax4.plot(df_out.index, df_out['Q_NAT2_cfs'], label='Outflow Natural Q', color='darkgreen', linewidth=2.5)

for g_id, df_in in inflow_dfs.items():
    count += 1
    if count == 1:
        label='Gaged Inflow Q'
    else:
        label=''
    ax4.plot(
        df_in.index, 
        df_in['Q_NAT2_cfs'], 
        label=label, 
        color='green', 
        alpha=0.6, 
        linewidth=1.8
    )

ax4.plot(df_out.index, ungaged_inflow, label='Ungaged Inflow Q', color='orange', alpha=0.5, linewidth=2.0)

if 'Q_NATUSBR_cfs' in df_out.columns:
    ax4.plot(df_out.index, df_out['Q_NATUSBR_cfs'], label='USBR Natural Q', color='black', linestyle='--', linewidth=1.8)

ax4.set_xlabel('Date')
ax4.set_ylabel('Q (cfs)')
ax4.grid(True)
ax4.legend(loc='upper right', fontsize=lfs)
ax4.set_title(f'D', loc='left', fontsize=tfs, fontweight='bold')

plt.show()

####################
# 2 panel
####################
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(xdim, ydim), layout="constrained")

fig1.suptitle(f'Upper San Juan Example Naturalization: Diversions', fontsize=tfs + 2, fontweight='bold')

# -------------------------------------------------------------
# Panel A: Diversion Watershed Map
# -------------------------------------------------------------
target_basin_div_3857.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=2, label='Watershed Boundary')
target_gage_div_3857.plot(ax=ax1, color='blue', marker='^', markersize=120, zorder=5, label=f'Gage {gage_div}')

irrigation_dvrs = dvrs_in_basin_3857[dvrs_in_basin_3857['siteUse'] == 'irrigation']
transbasin_dvrs = dvrs_in_basin_3857[dvrs_in_basin_3857['siteUse'] == 'interbasin']

irrigation_dvrs.plot(ax=ax1, color='orange', marker='o', markersize=40, alpha=0.8, zorder=4, label='Irrigation Diversion')
transbasin_dvrs.plot(ax=ax1, color='red', marker='s', markersize=40, alpha=0.8, zorder=4, label='Transbasin Diversion')

cx.add_basemap(ax1, source=cx.providers.USGS.USTopo)
ax1.set_axis_off()
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=lfs)
ax1.set_title(f'A: {gage_div}: Diversion Impacted', loc='left', fontsize=tfs, fontweight='bold')
ax1.set_box_aspect(None)

cx.add_basemap(ax1, source=cx.providers.Esri.WorldTerrain)
usgs_hydro_wms = (
    "https://basemap.nationalmap.gov/arcgis/rest/services/"
    "USGSHydroCached/MapServer/tile/{z}/{y}/{x}"
)
cx.add_basemap(ax1, source=usgs_hydro_wms)

# -------------------------------------------------------------
# Panel B: Diversion Naturalization Graph
# -------------------------------------------------------------
ax2.plot(df_div.index, df_div['Q_cfs'], label='Observed Q', color='blue')
ax2.plot(df_div.index, df_div['Q_NAT2_cfs'], label='Estimated Natural Q', color='green')
ax2.plot(df_div.index, df_div['lagged_rf_cfs'] * -1, label='Return Flow Q', color='green', linestyle=':')
ax2.plot(df_div.index, df_div['irrigation_diversion_cfs'], label='Irrigation Q', color='orange')
ax2.plot(df_div.index, df_div['interbasin_cfs'], label='Transbasin Diversion Q', color='red')

ax2.set_xlabel('Date')
ax2.set_ylabel('Q (cfs)')
ax2.grid(True)
ax2.legend(fontsize=lfs)
ax2.set_title(f'B', loc='left', fontsize=tfs, fontweight='bold')

plt.show()
###############################################
# BIG MAP
################################################
# --- Setup Parameters ---
lfs = 10
tfs = 14
fig_dim = (12, 10)

# --- Load Data (using your existing variables/paths) ---
gages = gpd.read_parquet(rf"{cwd}\shiny-app\ucol_natural\spatial_data\all_UCOL_gages.parquet").to_crs(epsg=4326)
dvrs = gpd.read_parquet(rf"{cwd}\shiny-app\ucol_natural\spatial_data\ucrb_diversion_master_table.parquet").to_crs(epsg=4326)
ucol = gpd.read_parquet(rf"{cwd}\shiny-app\ucol_natural\spatial_data\UCOL.parquet").to_crs(epsg=4326)

CU_COLORS = {
    'irrigation': '#003b02', 'municipal': '#7570b3', 'interbasin':'#e7298a', 'industrial': '#666666'
}

CU_LABELS = {'irrigation':'irrigation', 'municipal':'municipal', 'industrial':'industrial', 'interbasin':'transbasin', 'intrabasin':'transbasin'}

# --- Project Data to Web Mercator for Basemap & Metric Distance Math ---
crs_3857 = 'EPSG:3857'
ucol_3857 = ucol.to_crs(crs_3857)
gages_3857 = gages.to_crs(crs_3857)
dvrs_3857 = dvrs.to_crs(crs_3857)
res_3857 = res.to_crs(crs_3857)

# make all the drvs
dvrs_3857['siteUse'] = np.where(dvrs_3857['siteUse'] == 'intrabasin', 'interbasin', dvrs_3857['siteUse'])

# Ensure spatial intersection filter for diversions within UCOL
dvrs_ucol_3857 = gpd.sjoin(dvrs_3857, ucol_3857[['geometry']], how='inner', predicate='within')

# --- Plot Initialization ---
fig, ax = plt.subplots(1, 1, figsize=fig_dim, layout="constrained")

# 1. Basin Boundary Plot
ucol_3857.plot(
    ax=ax, 
    facecolor='none', 
    edgecolor='black', 
    linewidth=2.0, 
    zorder=3, 
    label='Upper Colorado Basin'
)

# 2. Plot Diversions by Type with Specific Colors & Transparency
for site_use, color in CU_COLORS.items():
    subset = dvrs_ucol_3857[dvrs_ucol_3857['siteUse'] == site_use]
    label = CU_LABELS[site_use]
    if not subset.empty:
        subset.plot(
            ax=ax,
            color=color,
            marker='o',
            markersize=15,
            alpha=0.9,  # Alpha transparency for dense points
            zorder=4,
            label=f'{label.capitalize()} diversion'
        )

# 3. Plot Stream Gages
gages_3857.plot(
    ax=ax, 
    color='blue', 
    marker='^', 
    markersize=45, 
    edgecolor='darkblue',
    linewidth=0.5,
    zorder=5, 
    label='Stream Gages'
)

# plot reservoirs
res_3857.plot(
    ax=ax,
    edgecolor='cyan',
    facecolor='lightblue',
    linewidth=0.5,
    alpha=0.5,
    label='Reservoirs'
)

# 4. Add Matching USGS Topo Basemap
cx.add_basemap(ax, source=cx.providers.Esri.WorldTerrain)
usgs_hydro_wms = (
    "https://basemap.nationalmap.gov/arcgis/rest/services/"
    "USGSHydroCached/MapServer/tile/{z}/{y}/{x}"
)
# Overlay hydrography onto your plot
cx.add_basemap(ax, source=usgs_hydro_wms)

# 5. Add North Arrow
ax.annotate(
    'N', xy=(0.04, 0.94), xytext=(0.04, 0.87),
    arrowprops=dict(facecolor='black', width=3, headwidth=9),
    ha='center', va='center', xycoords='axes fraction',
    fontsize=12, fontweight='bold',
    bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor='black', alpha=0.8)
)

# 6. Add Scale Bar (100 km / ~62 mi)
scalebar_length_m = 100_000  # 100 km in Web Mercator units
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# Position near lower left inside the map
sb_x0 = x_min + (x_max - x_min) * 0.05
sb_x1 = sb_x0 + scalebar_length_m
sb_y = y_min + (y_max - y_min) * 0.05

ax.plot([sb_x0, sb_x1], [sb_y, sb_y], color='black', linewidth=3, zorder=6)
ax.text(
    (sb_x0 + sb_x1) / 2, sb_y + (y_max - y_min) * 0.012, 
    '100 km', ha='center', va='bottom', fontsize=10, fontweight='bold',
    bbox=dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7),
    zorder=6
)

# --- Formatting & Legend ---
ax.set_axis_off()
ax.set_title('Upper Colorado River Basin \nStream Gages, Diversions, and Reservoirs', fontsize=tfs, fontweight='bold', pad=10)

# Build custom legend entries for Panel C
legend_handles = [
    mpatches.Patch(facecolor='none', edgecolor='black', linewidth=2, label='Upper Colorado Watershed'),
    mpatches.Patch(facecolor='lightblue', edgecolor='cyan', label='Major Reservoir'),
]

point_handles, point_labels = ax.get_legend_handles_labels()
for h, l in zip(point_handles, point_labels):
    if l not in [p.get_label() for p in legend_handles]:
        legend_handles.append(h)

ax.legend(handles=legend_handles, loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=lfs)

plt.show()


###########
# RETURN FLOW GRAPH
###########
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

gage_div = '09237450'
year = 2016
start_div = f'{year}-05-01'
end_div = f'{year}-12-31'

df = pd.read_csv(os.path.join(wdvrsDir, f"{gage_div}.csv"), index_col='date', parse_dates=['date'])


df_div = df.loc[start_div:end_div]
name_div = name_dict.get(gage_div, gage_div)

doy_to_month = {
    1: "1 Jan",
    32: "1 Feb",
    60: "1 Mar",
    91: "1 Apr",
    121: "1 May",
    152: "1 Jun",
    182: "1 Jul",
    213: "1 Aug",
    244: "1 Sep",
    274: "1 Oct",
    305: "1 Nov",
    335: "1 Dec",
}

fig, ax = plt.subplots()

# 2. Plot using df_div.index.dayofyear as the x-axis
ax.plot(
    df_div.index.dayofyear, df_div["Q_cfs"], label="Observed Q", color="blue"
)
ax.plot(
    df_div.index.dayofyear,
    df_div["Q_NAT2_cfs"],
    label="Estimated Natural Q",
    color="green",
)
ax.plot(
    df_div.index.dayofyear,
    df_div["lagged_rf_cfs"] * -1,
    label="Return Flow Q",
    color="green",
    linestyle=":",
)
ax.plot(
    df_div.index.dayofyear,
    df_div["irrigation_diversion_cfs"],
    label="Irrigation Q",
    color="orange",
)

# 3. Filter DOY ticks to only those present within your data range
ticks = [
    doy
    for doy in doy_to_month.keys()
    if doy >= df_div.index.dayofyear.min()
    and doy <= df_div.index.dayofyear.max()
]
labels = [doy_to_month[doy] for doy in ticks]

# 4. Set the fixed locators and tick labels
ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))

ax.axhline(y=0, color='black')

ax.set_xlabel("Date")
ax.set_ylabel("Q (cfs)")
ax.legend(fontsize=lfs)
name = name_dict[gage_div]
ax.set_title(name, loc="left")


##########
# GRAND CANYON GRAPH
##########
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

gage_div = '09380000'
syear = 2020
eyear = 2025
start_div = f'{syear}-10-01'
end_div = f'{eyear}-09-30'

df = pd.read_csv(os.path.join(wdvrsDir, f"{gage_div}.csv"), index_col='date', parse_dates=['date'])
# 1. Assign Water Year to each row (Oct-Dec belong to the next calendar year)
df["water_year"] = df.index.year + (df.index.month >= 10).astype(int)

# 2. Group by water year and calculate min and max for each year
yearly_stats = df.groupby("water_year")[["Q_cfs", "Q_NAT2_cfs"]].agg(
    ["min", "max"]
)

# 3. Calculate and print the average yearly min and max
avg_stats = yearly_stats.mean()

print("Average Yearly Min and Max across all Water Years:")
print(f"Q_cfs - Average Min: {avg_stats[('Q_cfs', 'min')]:.2f}")
print(f"Q_cfs - Average Max: {avg_stats[('Q_cfs', 'max')]:.2f}")
print(f"Q_NAT2_cfs - Average Min: {avg_stats[('Q_NAT2_cfs', 'min')]:.2f}")
print(f"Q_NAT2_cfs - Average Max: {avg_stats[('Q_NAT2_cfs', 'max')]:.2f}")

df_div = df_div.loc[start_div:end_div]
name_div = name_dict.get(gage_div, gage_div)

fig, ax = plt.subplots()

# 2. Plot using df_div.index.dayofyear as the x-axis
ax.plot(
    df_div.index, df_div["Q_cfs"], label="Observed Q", color="blue"
)
ax.plot(
    df_div.index,
    df_div["Q_NAT2_cfs"],
    label="Estimated Natural Q",
    color="green",
)
ax.plot(
    df_div.index,
    df_div["lagged_rf_cfs"] * -1,
    label="Return Flow Q",
    color="green",
    linestyle=":",
)
ax.plot(
    df_div.index,
    df_div["irrigation_diversion_cfs"],
    label="Irrigation Q",
    color="orange",
)

ax.axhline(y=0, color='black')

ax.set_xlabel("Date")
ax.set_ylabel("Q (cfs)")
ax.legend(fontsize=lfs)
ax.set_title(f"", loc="left", fontsize=tfs)


###########
# CALIBRATE CONSUMPTIVE USE RATIO
###########
def align_and_intersect(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aligns two DataFrames on their combined DatetimeIndex and retains data only

    where both DataFrames have complete (non-NaN) rows across all columns.

    Missing or invalid rows are preserved in the index as full NaN rows.
    """
    # 1. Combine indexes to ensure both DataFrames cover the complete union of timestamps
    union_idx = df1.index.union(df2.index)
    df1_reindexed = df1.reindex(union_idx)
    df2_reindexed = df2.reindex(union_idx)

    # 2. Identify rows where ALL columns are non-NaN for each DataFrame
    valid_df1 = df1_reindexed.notna().all(axis=1)
    valid_df2 = df2_reindexed.notna().all(axis=1)

    # 3. Find the intersection of valid rows across both DataFrames
    both_valid_mask = valid_df1 & valid_df2

    # 4. Mask out rows where either DataFrame is missing data, keeping the index intact
    res1 = df1_reindexed.where(both_valid_mask, np.nan)
    res2 = df2_reindexed.where(both_valid_mask, np.nan)

    return res1, res2

# select gages, read data, align
b4gage =  '09146020' #'09218500' #'09303400'
afgage =   '09146200' #'09219200' ##'09304000'
b4 = pd.read_csv(os.path.join(allDir, f"{b4gage}.csv"), index_col='date', parse_dates=['date']) # Q_cfs = upstream observed
af = pd.read_csv(os.path.join(allDir, f"{afgage}.csv"), index_col='date', parse_dates=['date']) # Q_cfs = downstream observed, irrigation_cfs
af['irrigation_diversion'] = af['irrigation_cfs'] / 0.6 # total diverted water
b4, af = align_and_intersect(b4[['Q_cfs', 'pr_sum']], af[['Q_cfs', 'irrigation_diversion', 'pr_sum']])
b4 = b4.dropna()
af = af.dropna()

# scale for the missing precip
# b4area = basins[basins.index==b4gage]['area_km2'].iloc[0]
# afarea = basins[basins.index==afgage]['area_km2'].iloc[0]
# scale = b4area/afarea
b4precip = b4['pr_sum'].sum()
afprecip = af['pr_sum'].sum()
scale = b4precip / afprecip
b4['Q_cfs_scaled'] = b4['Q_cfs'] / scale 
print(f'scale {scale}')

# calculate CU
Qup = b4['Q_cfs'].sum()
Qdown = af['Q_cfs'].sum()
Qdivert = af['irrigation_diversion'].sum() * -1

# CU ratio
cu = (Qup - Qdown) / Qdivert
print(f'cu {cu}')
