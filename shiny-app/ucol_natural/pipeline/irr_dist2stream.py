"""
Mean distance from irrigated land to the nearest stream, per basin (simpler replacement for
dist2stream.py, which ran yearly IrrMapper on Earth Engine).

One-off static input for the Glover return-flow lag in naturalize.py. Run locally:

    python pipeline/irr_dist2stream.py
    python pipeline/irr_dist2stream.py --gages 09149500 09251000   # quick test

Data
  Irrigated land  GMIA-NEXT irrigation probability, water year 2024, ~30 m, EPSG:4326
                  (spatial_data/GMIA-NEXT_prob_NAM1_2024_v0.tif). 2024 is taken as
                  representative of the study period. Irrigated = probability >= 50%.
  Streams         NHD HR flowlines from download_nhd.py (Lee's Ferry drainage = the whole
                  UCRB). Default: stream_perennial and artificial_path (wide-river and
                  reservoir centerlines); --classes adds others, e.g. canal_ditch. Saved as a
                  feature class, pipeline/data/nhd/streams_<tag>.gpkg.

Method
  Each irrigated pixel centre is projected to EPSG:5070 and given its distance to the nearest
  selected flowline (exact point-to-line distance). The stream network is the whole UCRB, so a
  field near a divide can be measured to a stream in the next basin. For each basin polygon
  (its whole drainage, so nested basins include all upstream irrigated land):
      mean_dist = sum(area * d) / sum(area)
  The plain area-weighted mean treats every irrigated acre (each drop of applied water)
  equally, giving the average return-flow response of the basin. Pixel area varies with
  latitude in the geographic grid, hence the weights.

Output
  spatial_data/basin_dist2stream_gmia_<tag>.csv   (tag: art-per, art-can-per, ...)
      gage, name, n_pixels, irrigated_km2, mean_dist_m, median_dist_m, p90_dist_m, mean_dist_ft
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from rasterio.windows import Window, from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

GMIA_PATH = C.SPATIAL_DIR / "GMIA-NEXT_prob_NAM1_2024_v0.tif"
FLOWLINE_PATH = C.NHD_DIR / "basins" / "09380000.parquet"
DEFAULT_CLASSES = ['artificial_path', 'stream_perennial']
SHORT = {'artificial_path': 'art', 'canal_ditch': 'can', 'stream_perennial': 'per',
         'stream_intermittent': 'int', 'stream_ephemeral': 'eph'}

IRRIGATED_MIN_PROB = 50      # percent
STRIP_ROWS = 2048            # raster rows read at a time
CHUNK_PTS = 500_000          # points per nearest-stream query
EARTH_R = 6_371_008.8        # m, for pixel areas
M_TO_FT = 3.28084


def load_streams(classes, path):
    fl = gpd.read_parquet(FLOWLINE_PATH)
    streams = fl[fl['flow_class'].isin(classes)].to_crs(C.PROJ_CRS)
    streams = streams[~streams.geometry.is_empty & streams.geometry.notna()]
    path.parent.mkdir(parents=True, exist_ok=True)
    streams.to_file(path, layer='streams', driver='GPKG')
    print(f"{len(streams):,} flowlines ({', '.join(classes)}) -> {path}")
    return streams


def irrigated_pixels(area_geom_ll):
    """x, y (EPSG:5070) and area (m2) of irrigated pixel centres inside area_geom_ll (EPSG:4326)."""
    shapely.prepare(area_geom_ll)
    to_proj = Transformer.from_crs(4326, C.PROJ_CRS, always_xy=True)
    xs, ys, areas = [], [], []
    with rasterio.open(GMIA_PATH) as src:
        win = from_bounds(*area_geom_ll.bounds, src.transform).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        t = src.transform
        for r0 in range(0, win.height, STRIP_ROWS):
            strip = Window(win.col_off, win.row_off + r0, win.width, min(STRIP_ROWS, win.height - r0))
            a = src.read(1, window=strip)
            rows, cols = np.nonzero((a >= IRRIGATED_MIN_PROB) & (a != src.nodata))
            if not len(rows):
                continue
            lon = t.c + (strip.col_off + cols + 0.5) * t.a
            lat = t.f + (strip.row_off + rows + 0.5) * t.e
            keep = shapely.contains_xy(area_geom_ll, lon, lat)
            lon, lat = lon[keep], lat[keep]
            x, y = to_proj.transform(lon, lat)
            xs.append(x)
            ys.append(y)
            areas.append(EARTH_R ** 2 * np.radians(t.a) * np.radians(-t.e) * np.cos(np.radians(lat)))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(areas)


def nearest_stream_dist(x, y, streams):
    tree = shapely.STRtree(streams.geometry.values)
    d = np.empty(len(x))
    for i in range(0, len(x), CHUNK_PTS):
        pts = shapely.points(x[i:i + CHUNK_PTS], y[i:i + CHUNK_PTS])
        (pi, _), dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
        d[i + pi] = dist
        print(f"  distances {min(i + CHUNK_PTS, len(x)):,} / {len(x):,}", end='\r')
    print()
    return d


def basin_stats(basins, x, y, area, d):
    rows = []
    for gage, geom in basins.geometry.items():
        xmin, ymin, xmax, ymax = geom.bounds
        sel = np.flatnonzero((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax))
        shapely.prepare(geom)
        sel = sel[shapely.contains_xy(geom, x[sel], y[sel])]
        row = {'gage': gage, 'name': basins.at[gage, 'name'], 'n_pixels': len(sel),
               'irrigated_km2': area[sel].sum() / 1e6}
        if len(sel):
            w, ds = area[sel], d[sel]
            row['mean_dist_m'] = np.average(ds, weights=w)
            row['median_dist_m'] = np.median(ds)
            row['p90_dist_m'] = np.percentile(ds, 90)
        rows.append(row)
    out = pd.DataFrame(rows).reindex(columns=['gage', 'name', 'n_pixels', 'irrigated_km2', 'mean_dist_m',
                                              'median_dist_m', 'p90_dist_m'])
    out['mean_dist_ft'] = out['mean_dist_m'] * M_TO_FT
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gages', nargs='+', help='only these basins (output is not written)')
    ap.add_argument('--classes', nargs='+', default=DEFAULT_CLASSES, choices=list(SHORT),
                    help='NHD flow_class values that count as streams')
    args = ap.parse_args()
    tag = '-'.join(sorted(SHORT[c] for c in args.classes))
    out_path = C.SPATIAL_DIR / f"basin_dist2stream_gmia_{tag}.csv"

    basins = gpd.read_parquet(C.BASINS_PATH).to_crs(C.PROJ_CRS)
    basins.index = [C.fix_gage_id(g) for g in basins.index]
    if args.gages:
        basins = basins.loc[[C.fix_gage_id(g) for g in args.gages]]

    streams = load_streams(args.classes, C.NHD_DIR / f"streams_{tag}.gpkg")
    area_ll = basins.to_crs(4326).geometry.union_all()
    x, y, area = irrigated_pixels(area_ll)
    print(f"{len(x):,} irrigated pixels ({area.sum() / 1e6:,.0f} km2, prob >= {IRRIGATED_MIN_PROB}%)")
    d = nearest_stream_dist(x, y, streams)
    out = basin_stats(basins, x, y, area, d)

    irr = out[out['n_pixels'] > 0]
    print(f"{len(irr)} of {len(out)} basins have irrigated land; mean distance "
          f"median {irr['mean_dist_m'].median():,.0f} m, range {irr['mean_dist_m'].min():,.0f}"
          f"-{irr['mean_dist_m'].max():,.0f} m")
    if args.gages:
        print(out.to_string(index=False))
    else:
        out.round(3).to_csv(out_path, index=False)
        print(f"-> {out_path}")


if __name__ == '__main__':
    main()
