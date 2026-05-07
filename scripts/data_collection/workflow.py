import geopandas as gpd
import rioxarray
from pathlib import Path
import os
from os.path import join
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pygeohydro import NWIS
import numpy as np
from shapely.strtree import STRtree
###############
# 1 download DEM (R script)
# 2 find gages with good streamflow data in Upper Col basin
# 3 delineate basins
# 4 download climate reanlysis data
# 5 harmonize it

cwd = os.getcwd()


################## STEP 1


# query basins
ucol = gpd.read_file(join(cwd, r"data\shapefiles\UCOL\globalwatershed.shp")).set_crs('4326')
ucol_geom = ucol.union_all()  # single shapely geometry for spatial filtering

def make_tiles(gdf, n_tiles_x=3, n_tiles_y=3):
    """Split the bounding box of a GeoDataFrame into an n x m grid of sub-bboxes."""
    minx, miny, maxx, maxy = gdf.total_bounds
    xs = [minx + i * (maxx - minx) / n_tiles_x for i in range(n_tiles_x + 1)]
    ys = [miny + i * (maxy - miny) / n_tiles_y for i in range(n_tiles_y + 1)]
    tiles = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            tiles.append((x0, y0, x1, y1))
    return tiles

tiles = make_tiles(ucol, n_tiles_x=3, n_tiles_y=3)
print(f"Querying NWIS across {len(tiles)} tiles…")

# ── Query each tile and collect results ───────────────────────────────────────
nwis    = NWIS()
records = []

for i, (x0, y0, x1, y1) in enumerate(tiles):
    try:
        chunk = nwis.get_info(
            {
                "bBox":          f"{x0:.6f},{y0:.6f},{x1:.6f},{y1:.6f}",
                "siteType":      "ST",
                "siteStatus":    "active",
                "hasDataTypeCd": "dv",
            }
        )
        print(f"  tile {i+1}/{len(tiles)}: {len(chunk)} gages found")
        records.append(chunk)
    except Exception as e:
        print(f"  tile {i+1}/{len(tiles)}: no data or error — {e}")

# ── Combine and deduplicate across tile boundaries ────────────────────────────
info = pd.concat(records, ignore_index=True)
info = info.drop_duplicates(subset="site_no")
print(f"\n{len(info)} unique gages found across all tiles")

# ── Spatial filter: keep only gages inside the basin polygon ──────────────────
info_gdf = gpd.GeoDataFrame(info, geometry="geometry", crs="EPSG:4326")
print(f"Found {len(info)} gages")
inside_mask  = info_gdf.geometry.within(ucol_geom)
pps = info_gdf[inside_mask].copy().reset_index(drop=True)
print(f"{len(pps)} gages fall within the basin boundary")

# info is already a GeoDataFrame with point geometry
gages_gdf = pps.set_crs("EPSG:4326")

# Save to file
# gages_gdf.to_file(join(cwd, "ucol_gages.gpkg"), driver="GPKG")
# print(gages_gdf.head())

# find gages with good streamflow data
# Date range of interest
dates = ("1999-10-01", "2026-09-30")

station_ids = pps["site_no"].tolist()

# Returns a pandas DataFrame: columns = station IDs, index = dates
# Units are cubic meters per second (cms)
flow = NWIS.get_streamflow(station_ids, dates, freq="dv")
print(flow.shape)   # (n_days, n_stations)

# Station metadata is stored in .attrs
for sid, meta in flow.attrs.items():
    print(sid, meta.get("station_nm"), meta.get("dec_lat_va"))

###################### STEP 2: Delineate watersheds from NLDI
from pynhd import NLDI

nldi  = NLDI()
basin = nldi.get_basins(station_ids) # watershed polygons
basin['gage_used'] = basin.index.str[-8:] # make gage column

# # merge with CSU_Flow25
flow25 = gpd.read_file(os.path.join(cwd, r"data\CSU_Flow25\watersheds_shapefile_20250624.shp"))
flow25.geometry = flow25.geometry.to_crs(4326)

mask = flow25.geometry.within(ucol_geom)
flow25 = flow25[mask].copy().reset_index(drop=True)
# fix gage Ids
def fix_gage_id(id_val):
    id_str = str(id_val).strip()
    # Only pad if it is a 7-digit numeric string
    if len(id_str) == 7 and id_str.isdigit():
        return id_str.zfill(8)

    return id_str

# Apply the logic to the gage_used column
flow25['gage_used'] = flow25['gage_used'].apply(fix_gage_id)
flow25['gage_used'] = np.where(flow25['gage_used'].str.contains('E+', regex=False), flow25['usgs_id'], flow25['gage_used']) # the pesky long ones

basin = pd.concat([basin, flow25[['gage_used', 'geometry']].drop_duplicates(subset='gage_used')], ignore_index=True)
basin = basin.drop_duplicates(subset='gage_used')


############# REMOVE NESTED BASINS
def remove_nested_basins(gdf, area_col=None):
    gdf = gdf.copy().reset_index(drop=True)

    rep_points = gdf.geometry.representative_point()

    is_geographic = gdf.crs.is_geographic if gdf.crs else True
    buffer_dist   = 0.001 if is_geographic else 100.0
    buffered      = gdf.geometry.buffer(-buffer_dist)

    tree = STRtree(buffered.values)

    # A basin is a PARENT (downstream, non-headwater) if ANY other basin's
    # representative point falls inside it. Flag those for removal.
    parent_idx = set()

    for i, pt in enumerate(rep_points):
        candidates = tree.query(pt)
        for j in candidates:
            if j == i:
                continue
            if buffered.iloc[j].contains(pt):
                # basin j contains basin i's point → j is a parent, not a headwater
                parent_idx.add(j)

    headwater_mask = ~gdf.index.isin(parent_idx)
    headwaters     = gdf[headwater_mask].copy().reset_index(drop=True)

    print(f"Total basins       : {len(gdf)}")
    print(f"Parents (removed)  : {len(parent_idx)}")
    print(f"Headwaters kept    : {len(headwaters)}")

    return headwaters

# just headwaters
basins = remove_nested_basins(basin)
#headwaters.explore()

############## SNODAS ############
import requests
import gzip
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray
import xarray as xr
from rasterio.transform import from_origin
from rasterio.mask import mask as rio_mask
from datetime import date, timedelta
from pathlib import Path
from shapely.geometry import mapping
import warnings

OUT_DIR      = Path(join(cwd, 'data\SNODAS'))
TMP_DIR      = OUT_DIR / "tmp"
OUT_NETCDF   = OUT_DIR / "basin_swe.nc"
OUT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

START_DATE   = date(2025, 9, 30)
END_DATE     = date.today()
PRODUCT_CODE = "1034"       # SWE
SCALE        = 1000.0       # stored as mm × 1000 → divide for metres
NODATA_RAW   = -9999

# SNODAS fixed grid parameters (masked CONUS product)
NCOLS        = 6935
NROWS        = 3351
XLL          = -124.733749999999
YLL          =   24.949583333333
CELLSIZE     =    0.00833333333

# ── Reproject headwater basins to WGS84 ────────────────────────────────────────
basins    = basins.to_crs("EPSG:4326").copy().reset_index(drop=True)
basin_ids = basins.gage_used.tolist()   # use integer index; swap for a name col if you have one
n_basins  = len(basins)
print(f"Basins: {n_basins}  |  Days: {(END_DATE - START_DATE).days + 1}")

# ── SNODAS URL builder ─────────────────────────────────────────────────────────
def build_snodas_url(d: date) -> tuple[str, str]:
    ym   = d.strftime("%Y%m")
    ymd  = d.strftime("%Y%m%d")
    stem = f"us_ssmv1{PRODUCT_CODE}tS__T0001TTNATS{ymd}05HP001"
    base = f"https://nohrsc.noaa.gov/snowfall/data/{ym}/{stem}"
    return f"{base}.dat.gz", f"{base}.Hdr.gz"

# ── Download + decompress one file ────────────────────────────────────────────
def fetch_gz(url: str, out_path: Path) -> bool:
    gz = out_path.with_suffix(".gz")

    r = requests.get(url, stream=True, timeout=60)
    print(r)
    r.raise_for_status()
    with open(gz, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    with gzip.open(gz, "rb") as g, open(out_path, "wb") as f:
        shutil.copyfileobj(g, f)
    gz.unlink()
    return True


fetch_gz(dat_url, dat_path)

# ── Read .dat binary → masked numpy array (SWE in metres) ─────────────────────
def read_snodas_dat(dat_path: Path) -> np.ndarray:
    raw = np.frombuffer(dat_path.read_bytes(), dtype=">i2").reshape(NROWS, NCOLS)
    swe = np.where(raw == NODATA_RAW, np.nan, raw.astype(np.float32) / SCALE)
    return swe

# ── Build rasterio transform (same every day) ──────────────────────────────────
TRANSFORM = from_origin(
    west  = XLL,
    north = YLL + NROWS * CELLSIZE,
    xsize = CELLSIZE,
    ysize = CELLSIZE,
)

# ── Precompute basin shapes as GeoJSON-like dicts for rasterio ─────────────────
basin_shapes = [mapping(geom) for geom in basins.geometry]

# ── Compute basin-average SWE for one day's array ─────────────────────────────
def basin_mean_swe(swe_array: np.ndarray) -> list[float]:
    """Clip raster to each basin polygon and return mean SWE (m)."""
    means = []
    # Write to an in-memory rasterio dataset
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=NROWS, width=NCOLS,
            count=1, dtype=rasterio.float32,
            crs="EPSG:4326", transform=TRANSFORM, nodata=np.nan,
        ) as dataset:
            dataset.write(swe_array.astype(np.float32), 1)

        with memfile.open() as dataset:
            for shape in basin_shapes:
                try:
                    clipped, _ = rio_mask(
                        dataset, [shape],
                        crop=True, nodata=np.nan, filled=True
                    )
                    vals = clipped[0]
                    valid = vals[~np.isnan(vals)]
                    means.append(float(np.mean(valid)) if len(valid) > 0 else np.nan)
                except Exception:
                    means.append(np.nan)
    return means

# ── Main loop ──────────────────────────────────────────────────────────────────
dates_list  = []
swe_records = []   # list of lists: [n_days × n_basins]

d = START_DATE
while d <= END_DATE:
    dat_url, hdr_url = build_snodas_url(d)
    dat_path = TMP_DIR / f"snodas_{d.strftime('%Y%m%d')}.dat"

    # Download (skip header — grid params are fixed)
    success = fetch_gz(dat_url, dat_path)

    if success:
        swe_array = read_snodas_dat(dat_path)
        day_means = basin_mean_swe(swe_array)
        dat_path.unlink()   # delete tmp file immediately to save disk
    else:
        print(f"  WARNING: missing {d.isoformat()}")
        day_means = [np.nan] * n_basins

    dates_list.append(pd.Timestamp(d))
    swe_records.append(day_means)

    if d.timetuple().tm_yday % 30 == 0:   # progress every ~30 days
        print(f"  processed through {d.isoformat()}")

    d += timedelta(days=1)

print("Download complete. Building output dataset…")

# ── Assemble DataFrame ─────────────────────────────────────────────────────────
swe_df = pd.DataFrame(
    swe_records,
    index   = pd.DatetimeIndex(dates_list, name="date"),
    columns = basin_ids,
)
swe_df.columns.name = "basin_id"
swe_df.index.name   = "date"

print(swe_df)

# ── Save to NetCDF via xarray ──────────────────────────────────────────────────
ds = xr.Dataset(
    {"swe": (["date", "basin_id"], swe_df.values)},
    coords={
        "date":     swe_df.index,
        "basin_id": basin_ids,
    },
)
ds["swe"].attrs = {"units": "metres", "long_name": "Basin-average SWE from SNODAS"}
ds.attrs = {
    "source":    "NOHRSC SNODAS masked product 1034",
    "created":   pd.Timestamp.now().isoformat(),
    "crs":       "EPSG:4326",
}
ds.to_netcdf(OUT_NETCDF)
print(f"\nSaved → {OUT_NETCDF.resolve()}")

# ── Also save as CSV if preferred ─────────────────────────────────────────────
swe_df.to_csv(OUT_DIR / "basin_swe.csv")
print(f"Saved → {(OUT_DIR / 'basin_swe.csv').resolve()}")