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

def fill_discharge_data(df: pd.DataFrame, col_names: tuple = ("Q_cms")) -> pd.DataFrame:
    """Fills missing values in a streamflow time series based on flow-rate and time gap thresholds.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with a DatetimeIndex and the target column.
    col_name : str
        Name of the discharge column.

    Returns:
    --------
    pd.DataFrame
        Dataframe with updated missing values and a boolean flag column.
    """
    df = df.copy()

    # Calculate 20% of the mean of valid measurements
    for col_name in col_names:
        q_mean = df[col_name].mean()
        low_flow_threshold = 0.20 * q_mean

        # Identify original non-NaN values and their positions
        is_valid = df[col_name].notna()

        # Track original start and end valid measurements (Rule 2)
        first_valid_idx = is_valid.idxmax() if is_valid.any() else None
        last_valid_idx = is_valid.idxmin() if is_valid.any() else None
        # Precise last index with valid data
        last_valid_idx = df[col_name].dropna().index[-1]

        # Create the helper columns: last valid value and last valid timestamp
        prev_val = df[col_name].ffill()
        last_valid_time = df.index.to_series().where(is_valid).ffill()

        # Calculate days elapsed since the last valid measurement
        days_since_last = (df.index - last_valid_time).dt.total_seconds() / (
            24 * 3600
        )

        # Condition 1: Low-flow (< 20% of mean) and gap < 120 days
        is_low_flow = prev_val < low_flow_threshold
        cond_low_flow = is_low_flow & (days_since_last < 120)

        # Condition 2: Regular/High-flow (>= 20% of mean) and gap < 7 days
        cond_high_flow = (~is_low_flow) & (days_since_last < 7)

        # Combined fill condition (excluding original valid points and outer edges)
        fill_mask = (
            df[col_name].isna()  # Only fill missing data
            & (cond_low_flow | cond_high_flow)  # Respect low/high flow gap rules
            & (df.index > first_valid_idx)  # Rule 2: Ignore leading edge
            & (df.index < last_valid_idx)  # Rule 2: Ignore trailing edge
        )

        # Apply fills and create flag column
        df[f"{col_name}_is_filled"] = fill_mask
        df.loc[fill_mask, col_name] = prev_val[fill_mask]

    return df

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


import time
import geopandas as gpd
import numpy as np
import rasterio
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import os

cwd = os.getcwd()
ncwd = r'N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\data\terrain\dem_ucol_big50'
appcwd = os.path.join(cwd, r'shiny-app\ucol_natural')
appDir = os.path.join(appcwd, r'timeseries') # does not have each individual diversion

# 1. Load data
raster_path = os.path.join(ncwd, "streams.tif")
shp_path = os.path.join(ncwd, "snapped.shp")

start_time = time.time()

gages = gpd.read_file(shp_path)
gages.set_index('gage', inplace=True)

with rasterio.open(raster_path) as src:
  stream_arr = src.read(1)
  transform = src.transform
  crs = src.crs
  cell_size_x = abs(transform[0])
  cell_size_y = abs(transform[4])


gages = gages.to_crs(5070)

# 2. Identify stream pixels and map them to sequential integer IDs (0 to N-1)
rows, cols = np.where((stream_arr == 1) | (stream_arr == 1.0))
num_nodes = len(rows)
print(f"Found {num_nodes:,} stream pixels.")

# Create a fast lookup grid for node IDs (-1 means no stream)
grid_h, grid_w = stream_arr.shape
node_grid = np.full((grid_h, grid_w), -1, dtype=np.int32)
node_grid[rows, cols] = np.arange(num_nodes)

# Get spatial coordinates for all stream nodes
xs, ys = rasterio.transform.xy(transform, rows, cols)
node_coords = np.column_stack((xs, ys))

# 3. Build a sparse adjacency matrix efficiently using COO format
print("Building sparse graph connections...")
diag_dist = np.hypot(cell_size_x, cell_size_y)
orth_dist = cell_size_x

src_list = []
dst_list = []
weight_list = []

# Check 4 orthogonal and 4 diagonal directions
# To avoid duplicate edges, we only look at down/right directions
neighbor_offsets = [
    (1, 0, orth_dist),
    (0, 1, orth_dist),
    (1, 1, diag_dist),
    (1, -1, diag_dist),
]

for dr, dc, w in neighbor_offsets:
  nr = rows + dr
  nc = cols + dc

  # Filter valid bounds
  valid = (nr >= 0) & (nr < grid_h) & (nc >= 0) & (nc < grid_w)
  if not np.any(valid):
    continue

  # Find neighbors that are also stream pixels
  valid_r = rows[valid]
  valid_c = cols[valid]
  valid_nr = nr[valid]
  valid_nc = nc[valid]

  neighbor_ids = node_grid[valid_nr, valid_nc]
  is_stream = neighbor_ids != -1

  if np.any(is_stream):
    u_nodes = node_grid[valid_r[is_stream], valid_c[is_stream]]
    v_nodes = neighbor_ids[is_stream]

    # Add bidirectional edges
    src_list.extend(u_nodes)
    dst_list.extend(v_nodes)
    weight_list.extend([w] * len(u_nodes))

    src_list.extend(v_nodes)
    dst_list.extend(u_nodes)
    weight_list.extend([w] * len(u_nodes))

# Create SciPy Compressed Sparse Row (CSR) graph matrix
adjacency_matrix = coo_matrix(
    (weight_list, (src_list, dst_list)), shape=(num_nodes, num_nodes)
).tocsr()

# 4. Snap gages to nearest stream nodes using cKDTree
print("Snapping gages to stream network...")
tree = cKDTree(node_coords)
gage_coords = np.array([[geom.x, geom.y] for geom in gages.geometry])
_, gage_node_indices = tree.query(gage_coords)

# 5. Compute shortest paths from gage nodes using SciPy's C-optimized Dijkstra
print("Calculating shortest paths between gages...")
num_gages = len(gages)
dist_matrix_km = np.zeros((num_gages, num_gages))

for i, src_idx in enumerate(gage_node_indices):
  # Compute shortest paths from a single source node to all stream nodes (1D array)
  distances = dijkstra(
      adjacency_matrix,
      directed=False,
      indices=src_idx,
      return_predecessors=False,
  )

  # Extract distances from this source to only the other gage nodes
  dist_matrix_km[i, :] = distances[gage_node_indices] / 1000.0

print(f"Done in {time.time() - start_time:.2f} seconds!")
print("\nDistance Matrix (Kilometers):")
print(np.round(dist_matrix_km, 3))

import pandas as pd

# Convert to Pandas DataFrame using the gage indices
df_dist = pd.DataFrame(
    dist_matrix_km, index=gages.index, columns=gages.index
)

df_dist.to_csv(os.path.join(ncwd, 'dist_matrix.csv'))
