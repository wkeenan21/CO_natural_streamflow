"""
Shared paths, settings and helpers for the UCRB natural-flow data pipeline.

The pipeline has two entry points, both run from anywhere (paths are resolved
relative to this file, not the working directory):

    python pipeline/update_inputs.py   # download/refresh streamflow, diversions, evap
    python pipeline/naturalize.py      # build natural-flow timeseries the app reads

API keys are read from environment variables (never hard-code them):
    API_USGS_PAT        USGS Water Data API token (used by `dataretrieval.waterdata`)
    CDSS_API_KEY        Colorado DWR / CDSS REST API token
    DRI_EVAP_API_KEY    DRI OpenET reservoir evaporation API key
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PIPELINE_DIR = Path(__file__).resolve().parent
APP_DIR = PIPELINE_DIR.parent                      # shiny-app/ucol_natural
SPATIAL_DIR = APP_DIR / "spatial_data"             # static spatial inputs (read by app too)
TIMESERIES_DIR = APP_DIR / "timeseries"            # naturalize.py output, read by the app

INPUT_DIR = PIPELINE_DIR / "inputs"                # static inputs that are not downloaded
DIVERSION_INPUT_DIR = INPUT_DIR / "diversions"     # master table + historical records

DATA_DIR = PIPELINE_DIR / "data"                   # downloaded / intermediate data (not committed)
STREAMFLOW_PATH = DATA_DIR / "streamflow_cms.parquet"
DIVERSION_WORK_DIR = DATA_DIR / "diversions"
DIVERSION_RAW_DIR = DIVERSION_WORK_DIR / "raw"           # per-source, per-site downloads
DIVERSION_PROC_DIR = DIVERSION_WORK_DIR / "processed"    # combined / filtered / filled tables
DIVERSIONS_PATH = DIVERSION_PROC_DIR / "diversions_filtered_filled_cfs.parquet"
EVAP_DIR = DATA_DIR / "evap"
NATURALIZED_FULL_DIR = DATA_DIR / "naturalized_full"     # every diversion column, for checking
NAT_RESULTS_PATH = DATA_DIR / "nat_results.csv"
INFLOW_VALIDATION_PATH = DATA_DIR / "inflow_validation.csv"            # see naturalize.py
INFLOW_VALIDATION_YEARLY_PATH = DATA_DIR / "inflow_validation_by_year.csv"
VALIDATION_PLOT_DIR = DATA_DIR / "plots"                                  # plot_inflow_validation.py

# Static inputs in spatial_data/
BASINS_PATH = SPATIAL_DIR / "all_UCOL_basins.parquet"
GAGES_PATH = SPATIAL_DIR / "all_UCOL_gages.parquet"
UCOL_PATH = SPATIAL_DIR / "UCOL.parquet"
RESERVOIRS_PATH = SPATIAL_DIR / "all_UCOL_reservoirs.parquet"        # NHD polygons
EVAP_RESERVOIRS_PATH = SPATIAL_DIR / "ucol_reservoirs.parquet"       # DRI evap reservoir points
DIVERSION_MASTER_PARQUET = SPATIAL_DIR / "ucrb_diversion_master_table.parquet"  # app copy
DIVERSION_MASTER_CSV = DIVERSION_INPUT_DIR / "ucrb_diversion_master_table.csv"
IRRIGATION_PATH = SPATIAL_DIR / "basin_irrigation_percentages.csv"
DIST_MATRIX_PATH = SPATIAL_DIR / "dist_matrix.csv"
USBR_NAT_FLOWS_PATH = SPATIAL_DIR / "usbr_nat_flows.csv"

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
START_DATE = pd.Timestamp("1979-10-01")
PROJ_CRS = 5070                 # projected CRS for areas and spatial joins
CMS_TO_CFS = 35.3147


def end_date() -> pd.Timestamp:
    """Last day to request data for (today)."""
    return pd.Timestamp.today().normalize()


def ensure_dirs():
    for d in [DATA_DIR, DIVERSION_RAW_DIR, DIVERSION_PROC_DIR, EVAP_DIR,
              NATURALIZED_FULL_DIR, TIMESERIES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def env_key(name: str):
    """Return an API key from the environment, or None (with a warning)."""
    val = os.environ.get(name)
    if not val:
        print(f"WARNING: environment variable {name} is not set")
    return val


# --------------------------------------------------------------------------
# Helpers (from scripts/v2/workflow.py)
# --------------------------------------------------------------------------
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


def add_usgs_prefix(data):
    """Adds 'USGS-' to the beginning of a single string or every string in a list."""
    if isinstance(data, str):
        return f"USGS-{data}"
    elif isinstance(data, list):
        return [f"USGS-{item}" for item in data]
    else:
        raise TypeError("Input must be a string or a list of strings")


def df_to_geodataframe(df, lat_col="lat", lon_col="lon", crs="EPSG:4326", drop_latlon=False):
    """Convert a DataFrame with latitude/longitude columns to a GeoDataFrame of points."""
    import geopandas as gpd

    missing = [c for c in [lat_col, lon_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not found in DataFrame: {missing}")
    for col in [lat_col, lon_col]:
        if df[col].isna().all():
            raise ValueError(f"Column '{col}' contains all null values.")

    geometry = gpd.points_from_xy(df[lon_col].astype(float), df[lat_col].astype(float))
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)
    if drop_latlon:
        gdf = gdf.drop(columns=[lat_col, lon_col])
    return gdf


def fill_discharge_data(df: pd.DataFrame, col_names=("Q_cms",)) -> pd.DataFrame:
    """Fills missing values in a streamflow time series between the first and last valid dates:
    - Winter months (Oct-Apr): Filled only if gap size <= 182 days.
    - Non-winter months (May-Sep): Filled only if gap size < 7 days.
    Adds a boolean `<col>_filled` flag column for each column processed.
    """
    df = df.copy()
    new_flags = {}
    for col_name in col_names:
        is_valid = df[col_name].notna()
        if not is_valid.any():
            new_flags[f"{col_name}_filled"] = pd.Series(False, index=df.index)
            continue

        first_valid = df[col_name].first_valid_index()
        last_valid = df[col_name].last_valid_index()
        valid_range_mask = (df.index >= first_valid) & (df.index <= last_valid)

        is_winter = df.index.month.isin([10, 11, 12, 1, 2, 3, 4])
        is_summer = ~is_winter
        is_missing = df[col_name].isna() & valid_range_mask

        # Group consecutive missing values to measure total continuous gap length
        gap_id = (~is_missing).cumsum()
        gap_sizes = df.groupby(gap_id)[col_name].transform('size')

        winter_missing = is_missing & is_winter & (gap_sizes <= 182)
        summer_missing = is_missing & is_summer & (gap_sizes < 7)
        to_fill = winter_missing | summer_missing

        filled_series = df[col_name].interpolate(method='time')
        new_flags[f"{col_name}_filled"] = to_fill
        df.loc[to_fill, col_name] = filled_series[to_fill]

    # add flag columns in one step (avoids a fragmented DataFrame with ~300 gages)
    flags = pd.DataFrame(new_flags, index=df.index)
    df = df.drop(columns=[c for c in flags.columns if c in df.columns])
    return pd.concat([df, flags], axis=1)


def merge_window(old: pd.DataFrame, new: pd.DataFrame, start, end) -> pd.DataFrame:
    """Replace `old` with `new` inside [start, end] for the columns in `new`.

    Used for incremental downloads: the re-downloaded window overwrites the cached
    values (so provisional data that was revised or removed is updated), while
    data outside the window, and columns that failed to download, are kept.
    """
    if old is None or old.empty:
        return new.sort_index()
    out = old.copy()
    idx = out.index.union(new.index)
    out = out.reindex(idx)
    in_window = (idx >= start) & (idx <= end)
    for col in new.columns:
        if col not in out.columns:
            out[col] = np.nan
        out.loc[in_window, col] = new[col].reindex(idx[in_window])
    return out.sort_index()
