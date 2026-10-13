import geopandas as gpd
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.strtree import STRtree
from pynhd import NLDI
nldi = NLDI()
from pygeohydro import NWIS
nwis = NWIS()
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import shape
from rasterstats import zonal_stats
import rasterio
os.environ["API_USGS_PAT"] = "ePZbH4mtoakm2VYSZDE78clCxGDmKxRWd7nWYzpt"
from dataretrieval import waterdata
import pyarrow
from scipy.optimize import minimize_scalar

def route_linear_reservoir(inflow: np.ndarray, k: float) -> np.ndarray:
    """Routes an inflow time series through a linear reservoir."""
    n = len(inflow)
    outflow = np.zeros(n)
    alpha = np.exp(-1.0 / k) if k > 0 else 0.0
    
    for t in range(1, n):
        outflow[t] = outflow[t - 1] * alpha + inflow[t] * (1.0 - alpha)
        
    return outflow

def q_obs_rmse_objective(k: float, df: pd.DataFrame) -> float:
    if k <= 0:
        return 1e9
    
    # Route only the unconsumed/return portion
    R_pred = route_linear_reservoir(df['return_supply_cfs'].values, k)
    
    # Subtract TOTAL diversion at headgate, add back routed return flow
    Q_pred = df['Q_NAT_cfs'].values - df['diversion_cfs'].values + R_pred
    
    rmse = np.sqrt(np.mean((Q_pred - df['Q_cfs'].values) ** 2))
    return rmse

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

cwd = os.getcwd()
ncwd = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow'
appcwd = os.path.join(cwd, r'shiny-app\ucol_natural')
appDir = os.path.join(appcwd, r'timeseries') # does not have each individual diversion

basins_path = os.path.join(appcwd, r'spatial_data/all_UCOL_basins.parquet')
gages_path = os.path.join(appcwd, r'spatial_data/all_UCOL_gages.parquet')
basins = gpd.read_parquet(basins_path)
basins.geometry = basins.geometry.to_crs(4326) # need this for spatial joins
gages = gpd.read_parquet(gages_path)

# find gages for calibration
for gage in basins.index:
    gage = '09209400'
    file = os.path.join(appDir, f'{gage}.csv')
    df = pd.read_csv(file)
    df.index = pd.to_datetime(df['date'])

    df['irrigation_cfs'] = df['irrigation_cfs']*-1
    df['diversion_cfs'] = df['irrigation_cfs'] / 0.6
    df['return_supply_cfs'] = df['diversion_cfs'] * 0.4
    
    # --- Calibration Execution ---
    # Set search range for k in days (e.g., 1 to 365 days)
    result = minimize_scalar(
        q_obs_rmse_objective, 
        args=(df,), 
        bounds=(1.0, 50000), 
        method='bounded'
    )

    calibrated_k = result.x
    print(f"Calibrated k: {calibrated_k:.2f} days")

    # Generate final predicted streamflow series
    df['R_modeled_cfs'] = route_linear_reservoir(df['return_supply_cfs'].values, 10)
    df['Q_obs_modeled'] = df['Q_NAT_cfs'] - df['irrigation_cfs'] + df['R_modeled_cfs']
    stop


fig, ax = plt.subplots()
df = df.loc['10-01-2023':'10-01-2024']
ax.plot(df.index, df['Q_cfs'], label='Q obs')
ax.plot(df.index, df['Q_NAT_cfs'], label='Q nat')
#ax.plot(df.index, df['diversion_cfs'], label='Q diversion')
#ax.plot(df.index, df['irrigation_cfs'], label='Q irrigation')
ax.plot(df.index, df['R_modeled_cfs'], label='return flow')
ax.plot(df.index, df['Q_obs_modeled'], label='best fit')
ax.legend()
plt.show()