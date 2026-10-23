"""
Effective distance from irrigated land to the nearest stream, per basin and year.

One-off static input for the Glover return-flow lag in naturalize.py (distance to stream,
GLOVER_A_FT). Runs on Google Earth Engine: fill in your login details below, then

    python pipeline/dist2stream.py submit        # upload streams + basins, start yearly exports
    python pipeline/dist2stream.py status        # check progress
    python pipeline/dist2stream.py collect       # build tables + plots

    python pipeline/dist2stream.py submit --years 2020          # quick test with one year
    python pipeline/dist2stream.py submit --classes stream_perennial artificial_path

Data
  Irrigated land  IrrMapper v1.2 (UMT/Climate/IrrMapper_RF/v1_2), 30 m, annual 1986-2024
                  (irrigated = 1, everything else masked). Unmapped years use the basin median.
  Streams         NHD HR flowlines from download_nhd.py, default classes stream_perennial,
                  canal_ditch and artificial_path (wide-river and reservoir centerlines).
                  Read from pipeline/data/nhd/basins/09380000.parquet (Lee's Ferry = the whole
                  UCRB) or the combined nhd_flowlines_ucrb.parquet. Each class is simplified
                  (SIMPLIFY_STREAM_M) and uploaded to Earth Engine once, so other class sets can
                  be tested later by uploading only the new classes.

Method
  For each basin polygon (its whole drainage, so nested basins include all upstream
  irrigated land) and year, over irrigated pixels:
      effective distance = sqrt(mean(d^2))
  where d is the distance to the nearest selected flowline. Glover's time scale is a^2 S / T,
  so this preserves the basin-mean time scale. The stream network is the whole UCRB, so a
  field near a divide can be measured to a stream in the next basin. Distances are capped at
  SEARCH_RADIUS_M (the share of capped pixels is reported).

Outputs (named by the class set, e.g. tag "art-can-per")
  spatial_data/basin_dist2stream_<tag>.csv          per basin: median effective distance (+ range)
  spatial_data/basin_dist2stream_<tag>_by_year.csv  gage, year, irrigated_km2, eff_dist_m/ft,
                                                    mean_dist_m, frac_capped, source
  pipeline/data/plots/dist2stream_<tag>_*.png       map, histogram, vs area, by year
"""
# ---------------------------------------------------------------------------
# Fill in your Google Earth Engine details
# ---------------------------------------------------------------------------
GEE_PROJECT = "your-cloud-project-id"   # Google Cloud project registered for Earth Engine
GEE_SERVICE_ACCOUNT = None              # optional: "name@project.iam.gserviceaccount.com"
GEE_KEY_FILE = None                     # optional: path to that service account's JSON key
GEE_ASSET_FOLDER = None                 # optional: default projects/<GEE_PROJECT>/assets/ucrb_dist2stream
# With no service account, the first run opens a browser to log in (ee.Authenticate) and
# caches the credentials, so later runs don't ask again.
# ---------------------------------------------------------------------------

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

IRRMAPPER = "UMT/Climate/IrrMapper_RF/v1_2"
DEFAULT_CLASSES = ['stream_perennial', 'canal_ditch', 'artificial_path']
SHORT = {  # flow_class -> short name used in file and asset names
    'stream_perennial': 'per', 'stream_intermittent': 'int', 'stream_ephemeral': 'eph',
    'stream_unspecified': 'uns', 'canal_ditch': 'can', 'canal_aqueduct': 'aqu',
    'canal_stormwater': 'sto', 'artificial_path': 'art', 'connector': 'con',
    'pipeline': 'pip', 'underground_conduit': 'und', 'coastline': 'coa', 'other': 'oth',
}
FLOWLINE_SOURCES = [C.NHD_DIR / "basins" / "09380000.parquet", C.NHD_DIR / "nhd_flowlines_ucrb.parquet"]

SCALE_M = 30                 # IrrMapper resolution
SEARCH_RADIUS_M = 10_000     # distances beyond this are set to it (and counted)
SIMPLIFY_STREAM_M = 15       # flowline simplification before upload (half a pixel)
SIMPLIFY_BASIN_M = 30        # basin outline simplification before upload
CHUNK_COORDS = 100_000       # vertices per uploaded stream asset
CHUNK_BYTES = 3_000_000      # GeoJSON size per basin upload task
FIRST_YEAR = C.START_DATE.year
M_TO_FT = 3.28084
STAT_COLS = ['n_pix', 'sum_d', 'sum_d2', 'n_capped']


def tag_for(classes):
    return '-'.join(sorted(SHORT[c] for c in classes))


def out_paths(tag):
    return (C.SPATIAL_DIR / f"basin_dist2stream_{tag}.csv",
            C.SPATIAL_DIR / f"basin_dist2stream_{tag}_by_year.csv",
            C.DATA_DIR / f"dist2stream_{tag}_raw.csv")


# --------------------------------------------------------------------------
# Earth Engine setup
# --------------------------------------------------------------------------
def init_ee():
    import ee
    if not GEE_PROJECT or GEE_PROJECT.startswith('your-'):
        raise SystemExit("set GEE_PROJECT (and optionally the service account) at the top of dist2stream.py")
    if GEE_SERVICE_ACCOUNT:
        creds = ee.ServiceAccountCredentials(GEE_SERVICE_ACCOUNT, str(GEE_KEY_FILE))
        ee.Initialize(creds, project=GEE_PROJECT)
    else:
        try:
            ee.Initialize(project=GEE_PROJECT)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=GEE_PROJECT)
    return ee


def asset_folder():
    return GEE_ASSET_FOLDER or f"projects/{GEE_PROJECT}/assets/ucrb_dist2stream"


def ensure_folder(ee, folder):
    if not asset_exists(ee, folder):
        ee.data.createAsset({'type': 'FOLDER'}, folder)


def asset_exists(ee, asset_id):
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def list_assets(ee, folder, prefix=''):
    try:
        assets = ee.data.listAssets({'parent': folder}).get('assets', [])
    except Exception:
        return []
    return sorted(a['name'] for a in assets if a['name'].split('/')[-1].startswith(prefix))


def task_list(ee):
    """[{'description', 'state', 'error_message'}] for this project's recent tasks."""
    try:
        return ee.data.getTaskList()
    except Exception:
        state_map = {'PENDING': 'READY', 'SUCCEEDED': 'COMPLETED'}
        out = []
        for op in ee.data.listOperations():
            m = op.get('metadata', {})
            out.append({'description': m.get('description', ''),
                        'state': state_map.get(m.get('state'), m.get('state')),
                        'error_message': op.get('error', {}).get('message', '')})
        return out


def active_tasks(ee):
    """Descriptions of tasks that are queued or running (so they aren't submitted twice)."""
    return {t['description'] for t in task_list(ee) if t['state'] in ('READY', 'RUNNING')}


def complete_set(ee, folder, prefix):
    """Assets named <prefix><i>_of_<n>, if all n exist; else None."""
    have = list_assets(ee, folder, prefix)
    ns = {a.rsplit('_of_', 1)[-1] for a in have}
    return have if have and len(ns) == 1 and len(have) == int(ns.pop()) else None


def wait(tasks, what, poll_s=30):
    while True:
        states = [t.status()['state'] for t in tasks]
        if all(s in ('COMPLETED', 'FAILED', 'CANCELLED') for s in states):
            break
        print(f"  {what}: {states.count('COMPLETED')}/{len(tasks)} done, waiting...")
        time.sleep(poll_s)
    failed = [t.status() for t in tasks if t.status()['state'] != 'COMPLETED']
    if failed:
        raise SystemExit(f"{what} failed: {[f.get('error_message') for f in failed]}")


def export_to_asset(ee, fc, asset_id):
    t = ee.batch.Export.table.toAsset(collection=fc, description=asset_id.split('/')[-1], assetId=asset_id)
    t.start()
    return t


# --------------------------------------------------------------------------
# Uploads (basins and streams), each done once
# --------------------------------------------------------------------------
def basin_chunks():
    """Basin polygons (whole drainage) as GeoJSON features, split into upload-sized chunks."""
    import geopandas as gpd
    basins = gpd.read_parquet(C.BASINS_PATH)
    basins.index = [C.fix_gage_id(g) for g in basins.index]
    geom = basins.geometry.to_crs(C.PROJ_CRS).simplify(SIMPLIFY_BASIN_M, preserve_topology=True)
    geom = gpd.GeoSeries(geom, crs=C.PROJ_CRS).to_crs(4326)
    chunks, cur, size = [], [], 0
    for gage, g in zip(basins.index, geom):
        gj = json.loads(gpd.GeoSeries([g]).to_json())['features'][0]['geometry']
        n = len(json.dumps(gj))
        if cur and size + n > CHUNK_BYTES:
            chunks.append(cur)
            cur, size = [], 0
        cur.append({'gage': gage, 'geometry': gj})
        size += n
    if cur:
        chunks.append(cur)
    return chunks


def upload_basins(ee, folder):
    done = complete_set(ee, folder, 'basins_')
    if done:
        return done
    chunks = basin_chunks()
    existing, running = set(list_assets(ee, folder, 'basins_')), active_tasks(ee)
    tasks = []
    for i, chunk in enumerate(chunks):
        aid = f"{folder}/basins_{i:02d}_of_{len(chunks):02d}"
        if aid in existing or aid.split('/')[-1] in running:
            continue
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry(f['geometry'], 'EPSG:4326', False), {'gage': f['gage']})
            for f in chunk])
        tasks.append(export_to_asset(ee, fc, aid))
    print(f"uploading {sum(map(len, chunks))} basins in {len(chunks)} assets")
    if tasks:
        wait(tasks, "basin upload")
    done = complete_set(ee, folder, 'basins_')
    if done is None:
        raise SystemExit(f"basin assets in {folder} are incomplete; delete basins_* and re-run")
    return done


def flowline_path(arg):
    paths = [Path(arg)] if arg else FLOWLINE_SOURCES
    for p in paths:
        if p.exists():
            return p
    raise SystemExit(f"no flowline file found ({', '.join(map(str, paths))}); run download_nhd.py first")


def stream_chunks(gdf):
    """Line parts as lon/lat coordinate lists, grouped into chunks of ~CHUNK_COORDS vertices."""
    import shapely
    geoms = gdf.geometry.to_crs(C.PROJ_CRS).simplify(SIMPLIFY_STREAM_M, preserve_topology=False)
    geoms = geoms.to_crs(4326)
    parts = shapely.get_parts(geoms.values[~(geoms.isna() | geoms.is_empty).values])
    chunks, cur, n = [], [], 0
    for p in parts:
        if p.geom_type != 'LineString':
            continue
        xy = np.round(shapely.get_coordinates(p), 5)            # ~1 m
        keep = np.r_[True, np.any(np.diff(xy, axis=0) != 0, axis=1)]  # drop repeated vertices
        xy = xy[keep]
        if len(xy) < 2:
            continue
        if cur and n + len(xy) > CHUNK_COORDS:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(xy.tolist())
        n += len(xy)
    if cur:
        chunks.append(cur)
    return chunks


def upload_streams(ee, folder, classes, src):
    """Upload each flow class (once) as MultiLineString assets; return all asset IDs."""
    import geopandas as gpd
    sub = f"{folder}/streams"
    ensure_folder(ee, sub)

    def complete(cls):  # assets are named <short>_<i>_of_<n>
        return complete_set(ee, sub, f"{SHORT[cls]}_")

    todo = [c for c in classes if complete(c) is None]
    if todo:
        print(f"reading {src.name} for {todo}")
        flow = gpd.read_parquet(src, columns=['flow_class', 'geometry'])
        running = active_tasks(ee)
        tasks = []
        for cls in todo:
            part = flow[flow['flow_class'] == cls]
            if part.empty:
                raise SystemExit(f"no '{cls}' flowlines in {src.name}")
            chunks = stream_chunks(part)
            km = part.geometry.length.sum() / 1000
            print(f"  {cls}: {len(part):,} flowlines, {km:,.0f} km -> {len(chunks)} assets")
            existing = set(list_assets(ee, sub, f"{SHORT[cls]}_"))
            for i, chunk in enumerate(chunks):
                aid = f"{sub}/{SHORT[cls]}_{i:03d}_of_{len(chunks):03d}"
                if aid in existing or aid.split('/')[-1] in running:
                    continue
                fc = ee.FeatureCollection([ee.Feature(
                    ee.Geometry.MultiLineString(chunk, 'EPSG:4326', False), {'flow_class': cls})])
                tasks.append(export_to_asset(ee, fc, aid))
        if tasks:
            wait(tasks, "stream upload")
    ids = []
    for cls in classes:
        have = complete(cls)
        if have is None:
            raise SystemExit(f"stream assets for '{cls}' are incomplete or from a different setting; "
                             f"delete {sub}/{SHORT[cls]}_* and re-run (or wait for running uploads)")
        ids += have
    return ids


# --------------------------------------------------------------------------
# Distance statistics
# --------------------------------------------------------------------------
def irrmapper_years(ee):
    years = ee.ImageCollection(IRRMAPPER).aggregate_array('system:time_start').map(
        lambda t: ee.Date(t).get('year')).distinct().sort()
    return [int(y) for y in years.getInfo()]


def distance_image(ee, streams):
    d = streams.distance(searchRadius=SEARCH_RADIUS_M, maxError=SCALE_M)
    capped = d.mask().Not()                     # farther than the search radius
    return d.unmask(SEARCH_RADIUS_M).toDouble(), capped.toDouble()


def year_stats(ee, basins, dist, capped, year):
    """Sum of d, d^2 and pixel counts over irrigated pixels, per basin, for one year."""
    irr = (ee.ImageCollection(IRRMAPPER)
           .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
           .select('classification').mosaic())
    img = (ee.Image.cat([ee.Image(1).toDouble(), dist, dist.multiply(dist), capped])
           .rename(STAT_COLS).updateMask(irr.eq(1)))
    stats = img.reduceRegions(collection=basins, reducer=ee.Reducer.sum(),
                              scale=SCALE_M, crs=f'EPSG:{C.PROJ_CRS}', tileScale=8)
    return stats.map(lambda f: ee.Feature(None, f.toDictionary(['gage'] + STAT_COLS)).set('year', year))


def cmd_submit(args):
    ee = init_ee()
    folder = asset_folder()
    ensure_folder(ee, folder)
    tag = tag_for(args.classes)
    basin_ids = upload_basins(ee, folder)
    stream_ids = upload_streams(ee, folder, args.classes, flowline_path(args.flowlines))
    basins = ee.FeatureCollection([ee.FeatureCollection(i) for i in basin_ids]).flatten()
    streams = ee.FeatureCollection([ee.FeatureCollection(i) for i in stream_ids]).flatten()
    dist, capped = distance_image(ee, streams)

    years = irrmapper_years(ee)
    if args.years:
        years = [y for y in years if y in args.years]
    print(f"stream set '{tag}' ({', '.join(args.classes)}); IrrMapper years {years[0]}-{years[-1]}")
    running = active_tasks(ee)
    n = 0
    for year in years:
        aid = f"{folder}/d2s_{tag}_{year}"
        if asset_exists(ee, aid) or aid.split('/')[-1] in running:
            continue
        export_to_asset(ee, year_stats(ee, basins, dist, capped, year), aid)
        n += 1
    print(f"submitted {n} yearly exports; check with `status`, then run `collect`")


def cmd_status(args):
    ee = init_ee()
    tasks = [t for t in task_list(ee)
             if t.get('description', '').startswith(('d2s_', 'basins_', *[f"{s}_" for s in SHORT.values()]))]
    if not tasks:
        print("no tasks found")
    counts = pd.Series([t['state'] for t in tasks]).value_counts()
    print(counts.to_string())
    for t in sorted(tasks, key=lambda t: t['description']):
        if t['state'] in ('FAILED', 'RUNNING', 'READY'):
            print(f"{t['description']:24s} {t['state']:10s} {t.get('error_message', '')}")


def cmd_collect(args):
    ee = init_ee()
    tag = tag_for(args.classes)
    rows = []
    for aid in list_assets(ee, asset_folder(), f"d2s_{tag}_"):
        rows += [f['properties'] for f in ee.FeatureCollection(aid).getInfo()['features']]
    if not rows:
        raise SystemExit(f"no d2s_{tag}_* assets yet; run `submit` and wait for the exports")
    stats = pd.DataFrame(rows)
    stats.to_csv(out_paths(tag)[2], index=False)
    finish(stats, tag, args.plots)


# --------------------------------------------------------------------------
# Tables and plots (no Earth Engine needed from here on)
# --------------------------------------------------------------------------
def build_tables(stats, first_year, last_year):
    """Per-year effective distances, with the basin median filled in for unmapped years."""
    s = stats.copy()
    s['gage'] = s['gage'].apply(C.fix_gage_id)
    s['year'] = s['year'].astype(int)
    s = s[s['n_pix'] > 0]                               # years with no irrigation in the basin
    s['irrigated_km2'] = s['n_pix'] * SCALE_M ** 2 / 1e6
    s['eff_dist_m'] = np.sqrt(s['sum_d2'] / s['n_pix'])
    s['mean_dist_m'] = s['sum_d'] / s['n_pix']
    s['frac_capped'] = s['n_capped'] / s['n_pix']
    s['source'] = 'irrmapper'
    s = s[['gage', 'year', 'irrigated_km2', 'eff_dist_m', 'mean_dist_m', 'frac_capped', 'source']]

    g = s.groupby('gage')
    summary = pd.DataFrame({
        'median_eff_dist_m': g['eff_dist_m'].median(),
        'min_eff_dist_m': g['eff_dist_m'].min(),
        'max_eff_dist_m': g['eff_dist_m'].max(),
        'median_mean_dist_m': g['mean_dist_m'].median(),
        'median_irrigated_km2': g['irrigated_km2'].median(),
        'max_frac_capped': g['frac_capped'].max(),
        'n_years': g.size(),
        'first_mapped': g['year'].min(),
        'last_mapped': g['year'].max(),
    })
    summary['median_eff_dist_ft'] = summary['median_eff_dist_m'] * M_TO_FT

    full = pd.MultiIndex.from_product([summary.index, range(first_year, last_year + 1)],
                                      names=['gage', 'year']).to_frame(index=False)
    by_year = full.merge(s, on=['gage', 'year'], how='left')
    fill = by_year['eff_dist_m'].isna()
    by_year.loc[fill, 'eff_dist_m'] = by_year.loc[fill, 'gage'].map(summary['median_eff_dist_m'])
    by_year.loc[fill, 'source'] = 'median'
    by_year['eff_dist_ft'] = by_year['eff_dist_m'] * M_TO_FT
    return by_year, summary.reset_index()


def gage_names():
    try:
        g = pd.read_parquet(C.GAGES_PATH, columns=['name'])
        return {C.fix_gage_id(k): v for k, v in g['name'].items()}
    except Exception:
        return {}


REF_LINES = [(2640 / M_TO_FT, 'current GLOVER_A_FT (2,640 ft)'),
             (1500 / M_TO_FT, 'CRDSS far (1,500 ft)'), (600 / M_TO_FT, 'CRDSS near (600 ft)')]


def make_plots(by_year, summary, tag, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    BLUE, INK2, GRID, SURF = "#2a78d6", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
                         "axes.axisbelow": True, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
    out_dir.mkdir(parents=True, exist_ok=True)
    s = summary[summary['median_irrigated_km2'] > 0]
    label = f"streams: {tag}"
    TICKS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    def plain_log(axis, lo, hi):
        t = [v for v in TICKS if lo <= v <= hi]
        axis.set_ticks(t, [f'{v:,}' for v in t])
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # 1. histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    lo = max(s['median_eff_dist_m'].min() * 0.9, 5)
    hi = max(s['median_eff_dist_m'].max() * 1.1, REF_LINES[0][0] * 1.1)
    bins = np.logspace(np.log10(min(lo, REF_LINES[-1][0] * 0.9)), np.log10(hi), 30)
    ax.hist(s['median_eff_dist_m'], bins=bins, color=BLUE, edgecolor=SURF, lw=1)
    ax.set_xscale('log')
    plain_log(ax.xaxis, bins[0], bins[-1])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    for v, lab in REF_LINES:
        ax.axvline(v, color=INK2, lw=0.8, ls=':')
        ax.text(v, 1.0, f' {lab}', transform=ax.get_xaxis_transform(), rotation=90,
                va='top', ha='right', fontsize=7.5, color=INK2)
    ax.set_xlabel('median effective distance, irrigated land to stream (m, log)')
    ax.set_ylabel('basins')
    ax.set_title(f'{len(s)} basins with irrigation ({label})', loc='left', fontsize=10.5)
    fig.savefig(out_dir / f'dist2stream_{tag}_hist.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. vs irrigated area
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(s['median_irrigated_km2'], s['median_eff_dist_m'], s=18, color=BLUE,
               edgecolor=SURF, lw=1, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ylo = min(s['median_eff_dist_m'].min(), REF_LINES[-1][0]) * 0.8
    yhi = max(s['median_eff_dist_m'].max(), REF_LINES[0][0]) * 1.25
    ax.set_ylim(ylo, yhi)
    plain_log(ax.yaxis, ylo, yhi)
    for v, lab in REF_LINES:
        ax.axhline(v, color=INK2, lw=0.8, ls=':')
        ax.text(ax.get_xlim()[1], v, f' {lab}', va='center', fontsize=7.5, color=INK2)
    ax.set_xlabel('median irrigated area in drainage (km², log)')
    ax.set_ylabel('median effective distance (m, log)')
    ax.set_title(f'Effective distance vs. irrigated area ({label})', loc='left', fontsize=10.5)
    fig.savefig(out_dir / f'dist2stream_{tag}_vs_area.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. by year, 8 most irrigated basins
    top = s.nlargest(8, 'median_irrigated_km2')['gage']
    names = gage_names()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharex=True, gridspec_kw={'hspace': 0.5})
    for ax, gage in zip(axes.flat, top):
        d = by_year[by_year['gage'] == gage]
        mapped = d['source'] == 'irrmapper'
        ax.plot(d.loc[mapped, 'year'], d.loc[mapped, 'eff_dist_m'], color=BLUE, lw=1.5)
        ax.plot(d.loc[~mapped, 'year'], d.loc[~mapped, 'eff_dist_m'], 'o', ms=3,
                color=INK2, label='median (no IrrMapper)')
        ax.set_title(f"{str(names.get(gage, ''))[:30].title()}\n{gage}", fontsize=8, loc='left')
        ax.set_ylim(0, d['eff_dist_m'].max() * 1.15)
    for ax in axes.flat[len(top):]:
        ax.set_visible(False)
    axes.flat[0].set_ylabel('effective distance (m)')
    axes.flat[4].set_ylabel('effective distance (m)')
    axes.flat[0].legend(frameon=False, fontsize=7, loc='lower left')
    fig.suptitle(f'Yearly effective distance, most-irrigated basins ({label})', x=0.01, ha='left', fontsize=10.5)
    fig.savefig(out_dir / f'dist2stream_{tag}_by_year.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4. map
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not available, skipping map")
        return
    basins = gpd.read_parquet(C.BASINS_PATH).to_crs(C.PROJ_CRS)
    basins.index = [C.fix_gage_id(g) for g in basins.index]
    basins = basins.join(summary.set_index('gage')['median_eff_dist_m'])
    basins = basins.assign(area=basins.area).sort_values('area', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 9))
    basins.plot(column='median_eff_dist_m', ax=ax, cmap='Blues', edgecolor=SURF, linewidth=0.3,
                legend=True, missing_kwds={'color': '#f1f0ec', 'label': 'no irrigation'},
                legend_kwds={'label': 'median effective distance (m)', 'shrink': 0.6})
    try:
        gpd.read_parquet(C.UCOL_PATH).to_crs(C.PROJ_CRS).boundary.plot(ax=ax, color=INK2, lw=0.8)
    except Exception:
        pass
    ax.set_axis_off()
    ax.set_title(f'Median effective distance from irrigated land to stream ({label})\n'
                 '(each basin = its whole drainage; small basins drawn on top)', loc='left', fontsize=10.5)
    fig.savefig(out_dir / f'dist2stream_{tag}_map.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def finish(stats, tag, plots=True):
    summary_path, by_year_path, _ = out_paths(tag)
    by_year, summary = build_tables(stats, FIRST_YEAR, C.end_date().year)
    summary.insert(1, 'name', summary['gage'].map(gage_names()))
    by_year.to_csv(by_year_path, index=False, float_format='%.4f')
    summary.to_csv(summary_path, index=False, float_format='%.4f')
    s = summary['median_eff_dist_m']
    print(f"{len(summary)} basins with irrigation; median effective distance "
          f"{s.median():.0f} m ({s.median() * M_TO_FT:.0f} ft), 10-90%: "
          f"{s.quantile(.1):.0f}-{s.quantile(.9):.0f} m")
    capped = summary[summary['max_frac_capped'] > 0.01]
    if len(capped):
        print(f"{len(capped)} basins have >1% of irrigated pixels beyond {SEARCH_RADIUS_M / 1000:.0f} km "
              f"(distance capped): {capped['gage'].tolist()[:10]}")
    print(f"wrote {summary_path.name} and {by_year_path.name} to {summary_path.parent}")
    if plots:
        make_plots(by_year, summary, tag, C.VALIDATION_PLOT_DIR)
        print(f"plots in {C.VALIDATION_PLOT_DIR}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['submit', 'status', 'collect'])
    p.add_argument('--classes', nargs='+', default=DEFAULT_CLASSES, choices=list(SHORT),
                   help='NHD flow classes that count as streams (default: %(default)s)')
    p.add_argument('--flowlines', help="flowline geoparquet (default: Lee's Ferry basin file, "
                                       "else nhd_flowlines_ucrb.parquet)")
    p.add_argument('--years', nargs='+', type=int, help='submit: only these years (e.g. a quick test)')
    p.add_argument('--no-plots', dest='plots', action='store_false', help='collect: skip plots')
    args = p.parse_args()
    C.ensure_dirs()
    {'submit': cmd_submit, 'status': cmd_status, 'collect': cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
