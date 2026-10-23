"""
Download NHD (high resolution, 1:24k) flowlines for the UCRB and cut them to each basin.

One-off static input for the return-flow distance work (irrigation to nearest flowline).
All flowline types are kept (streams by hydrographic category, canals/ditches, artificial
paths, connectors, pipelines, ...) and labelled in `flow_class`, so later tests can choose
which ones count as "the stream".

    python pipeline/download_nhd.py                      # all UCRB HU4s, all basins
    python pipeline/download_nhd.py --gages 09251000     # per-basin files for these only
    python pipeline/download_nhd.py --huc4 1401 1405     # other HU4s (default 1401-1408)

Source
  USGS NHD Best Resolution HU4 staged products (GeoPackage, falling back to Shapefile):
  https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU4/
  NHD was retired on 2023-10-01 (no further edits; its successor is 3DHP), but these
  downloads are still hosted. Zips are cached, so re-runs don't download again.

Outputs (pipeline/data/nhd/, gitignored: too large for git; re-create with this script)
  raw/NHD_H_<huc4>_HU4_*.zip         cached downloads
  hu4/flowlines_<huc4>.parquet       flowlines per HU4 (EPSG:5070, 2-D)
  hu4/wbdhu4_<huc4>.parquet          HU4 boundary (coverage check)
  nhd_flowlines_ucrb.parquet         all HU4s combined
  basins/<gage>.parquet              flowlines clipped to each basin (its whole drainage)
  basin_flowline_km.csv              km of flowline per basin and flow_class

Columns: permanent_identifier, reachcode, gnis_name, ftype, fcode, flow_class,
visibilityfilter (smallest map scale the feature is shown at; can emulate a coarser network),
innetwork, lengthkm (NHD length of the whole feature), huc4, geometry.

flow_class (from FType / FCode, NHD data dictionary)
  stream_perennial 46006 | stream_intermittent 46003 | stream_ephemeral 46007 |
  stream_unspecified 46000 | canal_ditch 33600 | canal_aqueduct 33601 |
  canal_stormwater 33603 | artificial_path 558 | connector 334 | pipeline 428 |
  underground_conduit 420 | coastline 566 | other
  Note: artificial paths are centerlines through waterbodies AND through wide (2-bank)
  rivers, so large mainstem reaches (e.g. Colorado, Green, San Juan) are artificial_path,
  not stream_perennial. Keep them in any "perennial" stream set.
"""
import argparse
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

UCRB_HUC4 = ['1401', '1402', '1403', '1404', '1405', '1406', '1407', '1408']
BASE_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/HU4"
FORMATS = [('GPKG', 'GPKG'), ('Shape', 'Shape')]          # (folder, file suffix), in order tried
NHD_DIR = C.NHD_DIR
RAW_DIR, HU4_DIR, BASIN_DIR = NHD_DIR / "raw", NHD_DIR / "hu4", NHD_DIR / "basins"
REGION_PATH = NHD_DIR / "nhd_flowlines_ucrb.parquet"
KM_TABLE_PATH = NHD_DIR / "basin_flowline_km.csv"
COVERAGE_TOL = 0.005    # warn if more than 0.5% of a basin lies outside the downloaded HU4s

# lower-cased NHD field names (GeoPackage and truncated shapefile names) -> output names
COLMAP = {
    'permanent_identifier': 'permanent_identifier', 'permanent_': 'permanent_identifier',
    'reachcode': 'reachcode', 'gnis_name': 'gnis_name', 'ftype': 'ftype', 'fcode': 'fcode',
    'visibilityfilter': 'visibilityfilter', 'visibility': 'visibilityfilter',
    'innetwork': 'innetwork', 'lengthkm': 'lengthkm',
}
OUT_COLS = ['permanent_identifier', 'reachcode', 'gnis_name', 'ftype', 'fcode', 'flow_class',
            'visibilityfilter', 'innetwork', 'lengthkm', 'huc4']

FLOW_CLASSES = [  # (ftype, fcode or None, name); first match wins
    (460, 46006, 'stream_perennial'), (460, 46003, 'stream_intermittent'),
    (460, 46007, 'stream_ephemeral'), (460, None, 'stream_unspecified'),
    (336, 33601, 'canal_aqueduct'), (336, 33603, 'canal_stormwater'), (336, None, 'canal_ditch'),
    (558, None, 'artificial_path'), (334, None, 'connector'), (428, None, 'pipeline'),
    (420, None, 'underground_conduit'), (566, None, 'coastline'),
]


def classify(ftype, fcode):
    """flow_class from NHD FType / FCode."""
    ft = pd.to_numeric(pd.Series(ftype), errors='coerce').to_numpy(dtype=float)
    fc = pd.to_numeric(pd.Series(fcode), errors='coerce').to_numpy(dtype=float)
    conds = [(ft == t) & (fc == c) if c is not None else (ft == t) for t, c, _ in FLOW_CLASSES]
    return np.select(conds, [n for _, _, n in FLOW_CLASSES], default='other')


def normalize(fl, huc4):
    """Rename NHD fields, add flow_class / huc4, keep OUT_COLS + geometry."""
    fl = fl.rename(columns={c: COLMAP[c.lower()] for c in fl.columns
                            if c != fl.geometry.name and c.lower() in COLMAP})
    for col in ('ftype', 'fcode', 'visibilityfilter', 'innetwork'):
        if col in fl.columns:
            fl[col] = pd.to_numeric(fl[col], errors='coerce').astype('Int64')
    fl['flow_class'] = classify(fl['ftype'], fl['fcode'])
    fl['huc4'] = huc4
    keep = [c for c in OUT_COLS if c in fl.columns]
    return fl[keep + [fl.geometry.name]]


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download(huc4, force=False):
    """Download the HU4 zip (GeoPackage, else Shapefile) into RAW_DIR; return its path."""
    import requests
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for folder, suffix in FORMATS:
        name = f"NHD_H_{huc4}_HU4_{suffix}.zip"
        dest, url = RAW_DIR / name, f"{BASE_URL}/{folder}/{name}"
        head = requests.head(url, timeout=60)
        if head.status_code == 404 or head.status_code == 403:   # S3 answers 403 for missing keys
            continue
        head.raise_for_status()
        size = int(head.headers.get('Content-Length', 0))
        if dest.exists() and not force and (size == 0 or dest.stat().st_size == size):
            print(f"  {name}: cached")
            return dest
        for attempt in range(3):
            try:
                fetch(url, dest, size)
                return dest
            except Exception as e:  # network hiccup: retry
                print(f"  {name}: download failed ({e}), retry {attempt + 1}/3")
                time.sleep(10)
        raise SystemExit(f"could not download {url}")
    raise SystemExit(f"no NHD HU4 product found for {huc4} under {BASE_URL}")


def fetch(url, dest, size):
    import requests
    part = dest.with_suffix('.part')
    done, next_report = 0, 0.1
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(part, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
                done += len(chunk)
                if size and done / size >= next_report:
                    print(f"  {dest.name}: {100 * done / size:.0f}% of {size / 1e6:.0f} MB")
                    next_report += 0.1
    if size and part.stat().st_size != size:
        raise IOError(f"got {part.stat().st_size} bytes, expected {size}")
    part.replace(dest)


# --------------------------------------------------------------------------
# Read
# --------------------------------------------------------------------------
def list_layers(path):
    try:
        import pyogrio
        return [str(n) for n in pyogrio.list_layers(path)[:, 0]]
    except ImportError:
        import fiona
        return list(fiona.listlayers(path))


def read_layer(path, layer_name):
    import geopandas as gpd
    layers = {l.lower(): l for l in list_layers(path)}
    if layer_name.lower() not in layers:
        raise KeyError(f"{layer_name} not in {path.name} (layers: {sorted(layers.values())})")
    return gpd.read_file(path, layer=layers[layer_name.lower()])


def to_2d(gdf):
    import geopandas as gpd
    import shapely
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf.geometry = gpd.GeoSeries(shapely.force_2d(gdf.geometry.values), index=gdf.index, crs=gdf.crs)
    return gdf


def read_hu4(zip_path, huc4):
    """Flowlines and HU4 boundary from a staged-product zip, in EPSG:5070."""
    import geopandas as gpd
    with tempfile.TemporaryDirectory(dir=NHD_DIR) as tmp, zipfile.ZipFile(zip_path) as z:
        tmp = Path(tmp)
        names = z.namelist()

        def extract(member):  # flatten paths so nothing is written outside tmp
            target = tmp / Path(member).name
            with z.open(member) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
            return target

        gpkg = [n for n in names if n.lower().endswith('.gpkg')]
        if gpkg:
            src = extract(gpkg[0])
            fl = read_layer(src, 'NHDFlowline')
            wbd = read_layer(src, 'WBDHU4')
        else:  # shapefile product; big layers may be split into NHDFlowline_0.shp, _1, ...
            def read_shps(pattern):
                stems = sorted({n[:-4] for n in names if re.search(pattern, Path(n).name, re.I)})
                if not stems:
                    raise KeyError(f"no {pattern} in {zip_path.name}")
                parts = []
                for stem in stems:
                    for n in names:
                        if n[:-4] == stem and n.lower()[-4:] in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
                            extract(n)
                    parts.append(gpd.read_file(tmp / (Path(stem).name + '.shp')))
                return pd.concat(parts, ignore_index=True)
            fl = read_shps(r'^NHDFlowline(_\d+)?\.shp$')
            wbd = read_shps(r'^WBDHU4\.shp$')
    fl = normalize(to_2d(fl), huc4).to_crs(C.PROJ_CRS)
    wbd = to_2d(wbd).to_crs(C.PROJ_CRS)
    hucol = next((c for c in wbd.columns if c.lower() == 'huc4'), None)
    if hucol is not None:
        wbd = wbd[wbd[hucol].astype(str) == huc4]
    return fl, wbd[[wbd.geometry.name]].assign(huc4=huc4)


def hu4_flowlines(huc4, force=False):
    """Cached per-HU4 parquet (download + read on first use)."""
    import geopandas as gpd
    HU4_DIR.mkdir(parents=True, exist_ok=True)
    fl_path, wbd_path = HU4_DIR / f"flowlines_{huc4}.parquet", HU4_DIR / f"wbdhu4_{huc4}.parquet"
    if fl_path.exists() and wbd_path.exists() and not force:
        print(f"HU4 {huc4}: cached")
        return gpd.read_parquet(fl_path), gpd.read_parquet(wbd_path)
    print(f"HU4 {huc4}: downloading")
    zip_path = download(huc4, force)
    print(f"HU4 {huc4}: reading {zip_path.name}")
    fl, wbd = read_hu4(zip_path, huc4)
    fl.to_parquet(fl_path, index=False)
    wbd.to_parquet(wbd_path, index=False)
    print(f"HU4 {huc4}: {len(fl):,} flowlines")
    return fl, wbd


# --------------------------------------------------------------------------
# Basins
# --------------------------------------------------------------------------
def load_basins(gages=None):
    import geopandas as gpd
    import shapely
    basins = gpd.read_parquet(C.BASINS_PATH)
    basins.index = [C.fix_gage_id(g) for g in basins.index]
    basins = basins.to_crs(C.PROJ_CRS)
    basins.geometry = gpd.GeoSeries(shapely.make_valid(basins.geometry.values),
                                    index=basins.index, crs=basins.crs)
    if gages:
        wanted = [C.fix_gage_id(g) for g in gages]
        missing = set(wanted) - set(basins.index)
        if missing:
            raise SystemExit(f"unknown gages: {sorted(missing)}")
        basins = basins.loc[wanted]
    return basins


def check_coverage(basins, wbd):
    """Warn about basins that extend outside the downloaded HU4s."""
    import shapely
    covered = shapely.union_all(wbd.geometry.values)
    outside = basins.geometry.difference(covered).area / basins.geometry.area
    bad = outside[outside > COVERAGE_TOL].sort_values(ascending=False)
    if len(bad):
        print(f"WARNING: {len(bad)} basins extend more than {COVERAGE_TOL:.1%} outside HU4s "
              f"{sorted(wbd['huc4'].unique())}; their networks are incomplete there "
              f"(add HU4s with --huc4):")
        for g, f in bad.head(15).items():
            print(f"  {g}: {f:.1%} outside")


def lines_only(geoms):
    """Keep the line parts of clip results (drops points where a line touches the edge)."""
    import shapely
    out = np.empty(len(geoms), dtype=object)
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            out[i] = None
        elif g.geom_type in ('LineString', 'MultiLineString'):
            out[i] = g
        else:
            parts = []
            for p in shapely.get_parts(g):
                if p.geom_type == 'LineString':
                    parts.append(p)
                elif p.geom_type == 'MultiLineString':
                    parts.extend(p.geoms)
            out[i] = shapely.MultiLineString(parts) if parts else None
    return out


def clip_to_basin(flow, poly):
    """Flowlines in a basin polygon; features crossing the boundary are cut at it."""
    import geopandas as gpd
    import shapely
    hit = np.sort(flow.sindex.query(poly, predicate='intersects'))
    inside = flow.sindex.query(poly, predicate='contains')   # poly.contains(flowline)
    sub = flow.iloc[hit].copy()
    edge = ~np.isin(hit, inside)
    if edge.any():
        geoms = np.asarray(sub.geometry.values, dtype=object)
        geoms[edge] = lines_only(shapely.intersection(geoms[edge], poly))
        sub = sub.set_geometry(gpd.GeoSeries(geoms, index=sub.index, crs=sub.crs))
        sub = sub[sub.geometry.notna()]
    return sub


def write_basins(flow, basins):
    BASIN_DIR.mkdir(parents=True, exist_ok=True)
    rows, total_bytes = [], 0
    for n, (gage, poly) in enumerate(basins.geometry.items(), 1):
        sub = clip_to_basin(flow, poly)
        path = BASIN_DIR / f"{gage}.parquet"
        sub.to_parquet(path, index=False)
        total_bytes += path.stat().st_size
        km = (sub.geometry.length / 1000).groupby(sub['flow_class']).sum()
        rows.append({'gage': gage, 'basin_km2': poly.area / 1e6, 'n_flowlines': len(sub),
                     'total_km': km.sum(), **km.to_dict()})
        if n % 10 == 0 or n == len(basins):
            print(f"  {n}/{len(basins)} basins written ({total_bytes / 1e9:.2f} GB so far)")
    table = pd.DataFrame(rows).fillna(0)
    classes = [n for _, _, n in FLOW_CLASSES] + ['other']
    table = table[['gage', 'basin_km2', 'n_flowlines', 'total_km'] + [c for c in classes if c in table]]
    table['drainage_density_km_per_km2'] = table['total_km'] / table['basin_km2']
    return table


# --------------------------------------------------------------------------
def main():
    import geopandas as gpd
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--huc4', nargs='+', default=UCRB_HUC4, help='HU4 codes to download (default 1401-1408)')
    p.add_argument('--gages', nargs='+', help='only write per-basin files for these gages')
    p.add_argument('--force', action='store_true', help='re-download and re-read the HU4s')
    p.add_argument('--no-basins', dest='basins', action='store_false', help='skip per-basin files')
    args = p.parse_args()
    NHD_DIR.mkdir(parents=True, exist_ok=True)

    fls, wbds = zip(*[hu4_flowlines(h, args.force) for h in args.huc4])
    flow = pd.concat(fls, ignore_index=True)
    wbd = pd.concat(wbds, ignore_index=True)
    flow = gpd.GeoDataFrame(flow, geometry=fls[0].geometry.name, crs=fls[0].crs)
    wbd = gpd.GeoDataFrame(wbd, geometry=wbds[0].geometry.name, crs=wbds[0].crs)
    if 'permanent_identifier' in flow.columns:
        n0 = len(flow)
        flow = flow.drop_duplicates('permanent_identifier').reset_index(drop=True)
        if len(flow) < n0:
            print(f"dropped {n0 - len(flow)} flowlines repeated across HU4s")
    flow.to_parquet(REGION_PATH, index=False)
    km = (flow.geometry.length / 1000).groupby(flow['flow_class']).sum().sort_values(ascending=False)
    print(f"\n{len(flow):,} flowlines, {km.sum():,.0f} km -> {REGION_PATH}")
    print((km.round(0).astype(int).map('{:,} km'.format)
           + (100 * km / km.sum()).map(' ({:.1f}%)'.format)).to_string())

    if not args.basins:
        return
    basins = load_basins(args.gages)
    check_coverage(basins, wbd)
    print(f"\nclipping to {len(basins)} basins -> {BASIN_DIR}")
    table = write_basins(flow, basins)
    if args.gages and KM_TABLE_PATH.exists():  # update rows for these gages only
        old = pd.read_csv(KM_TABLE_PATH, dtype={'gage': str})
        table = pd.concat([old[~old['gage'].isin(table['gage'])], table], ignore_index=True)
    table.sort_values('gage').to_csv(KM_TABLE_PATH, index=False, float_format='%.3f')
    print(f"km of flowline by basin and class -> {KM_TABLE_PATH}")


if __name__ == "__main__":
    main()
