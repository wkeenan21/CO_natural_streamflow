
#####################
# get reservoir evap
#####################
import os
import requests
import pandas as pd
import geopandas as gpd

# Define paths and API configuration
basedir = r'C:\Users\willy\Documents\GitHub\CO_natural_streamflow'
ucol_path = fr"{basedir}\shiny-app\ucol_natural\spatial_data\UCOL.parquet"
evap_dir = r"N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\timeseries\evap"

# Ensure output directory exists
os.makedirs(evap_dir, exist_ok=True)

BASE_URL = "https://operevap.dri.edu"
key = "2fdddbf5-9893-4cf8-885e-563de2ae7bc9"
HEADERS = {"api-key": key}

# 1. Load UCOL boundary
ucol = gpd.read_parquet(ucol_path).to_crs(4326)
ucol_geom = ucol.unary_union  # Combine multi-polygons if applicable

# 2. API Helper Functions
def get_reservoirs():
    """Get list of all available reservoirs."""
    url = f"{BASE_URL}/info/list_RES_NAMES"
    response = requests.post(url, headers=HEADERS)
    if response.status_code == 200 and "RES_NAMES" in response.json():
        return response.json()["RES_NAMES"]
    print(f"Error fetching reservoir list: {response.status_code}")
    return []

def get_reservoir_metadata(reservoir_names, chunk_size=50):
    """Get metadata in batches to avoid URL length limitations."""
    url = f"{BASE_URL}/metadata/reservoirs"
    all_metadata = []
    
    for i in range(0, len(reservoir_names), chunk_size):
        chunk = reservoir_names[i:i + chunk_size]
        params = {
            "RES_NAMES": ",".join(chunk),
            "output_format": "json"
        }
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code == 200:
            all_metadata.extend(response.json())
        else:
            print(f"Error fetching metadata batch {i}: {response.status_code}")
            
    return all_metadata

def get_reservoir_timeseries(reservoir_name, start_date="1979-10-01", end_date="2025-09-30"):
    """Get daily timeseries data for a reservoir."""
    url = f"{BASE_URL}/timeseries/daily/reservoirs/daterange"
    params = {
        "RES_NAMES": reservoir_name,
        "datasets": "nete-volume-calcs",
        "variables": "NetE,E_volume",
        "start_date": start_date,
        "end_date": end_date,
        "units": "metric",
        "output_format": "json"
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json()
    return None

# 3. Retrieve metadata & filter spatially
print("Fetching reservoir list...")
res_names = get_reservoirs()

print(f"Fetching metadata for {len(res_names)} reservoirs...")
meta_data = get_reservoir_metadata(res_names)

df_meta = pd.DataFrame(meta_data)

# Ensure numeric lat/lon
df_meta["LAT"] = pd.to_numeric(df_meta["LAT"], errors="coerce")
df_meta["LON"] = pd.to_numeric(df_meta["LON"], errors="coerce")
df_meta = df_meta.dropna(subset=["LAT", "LON"])

# Build GeoDataFrame
gdf_res = gpd.GeoDataFrame(
    df_meta,
    geometry=gpd.points_from_xy(df_meta["LON"], df_meta["LAT"]),
    crs="EPSG:4326"
)

# Spatial filter: Reservoirs inside UCOL boundary
ucol_reservoirs = gdf_res[gdf_res.geometry.within(ucol_geom)].copy()

print(f"Found {len(ucol_reservoirs)} reservoirs within the UCOL boundary.")

# Save filtered spatial locations as GeoParquet
geoparquet_path = os.path.join(evap_dir, "ucol_reservoirs.parquet")
ucol_reservoirs.to_parquet(geoparquet_path)
print(f"Saved reservoir locations to: {geoparquet_path}")

# 4. Download and save daily timeseries for each filtered reservoir
print("Downloading daily evaporation timeseries...")
for idx, row in ucol_reservoirs.iterrows():
    res_name = row["RES_NAME"]
    print(f" - Downloading: {res_name}")
    
    ts_data = get_reservoir_timeseries(res_name)
    if ts_data:
        # Convert response JSON to DataFrame
        df_ts = pd.DataFrame(ts_data)

        
        # Clean up column names and save
        df_ts = df_ts.drop(columns=['end_date'])
        safe_filename = res_name.replace(' ', '_')
        out_file = os.path.join(evap_dir, f"{safe_filename}.parquet")
        
        df_ts.to_parquet(out_file, index=False)
    else:
        print(f"   Failed to retrieve data for {res_name}")

print("Processing complete!")