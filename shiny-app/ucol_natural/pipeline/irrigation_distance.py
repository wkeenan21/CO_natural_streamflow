"""
Effective distance from irrigated land to the nearest river, per basin and year.

One-off static input for the Glover return-flow lag in naturalize.py (distance to stream,
GLOVER_A_FT). Runs on Google Earth Engine; needs `earthengine-api` and an authenticated
Earth Engine Cloud project.

    python pipeline/irrigation_distance.py submit  --project <gee-project>   # start exports
    python pipeline/irrigation_distance.py status  --project <gee-project>   # check progress
    python pipeline/irrigation_distance.py collect --project <gee-project>   # build tables + plots

Data
  Irrigated land  IrrMapper v1.2 (UMT/Climate/IrrMapper_RF/v1_2), 30 m, annual 1986-2024
                  (irrigated = 1, everything else masked). Years outside the record use the
                  basin's median.
  Rivers          WWF/HydroSHEDS/v1/FreeFlowingRivers (the FeatureCollection; the
                  _FeatureView asset is display-only and can't be used in computations).
                  Built on 15 arc-second (~500 m) HydroSHEDS, so river lines are only located
                  to a few hundred metres and small streams and canals are not included.

Method
  For each basin polygon (its whole drainage, so nested basins include all upstream
  irrigated land) and year, over irrigated pixels:
      effective distance = sqrt(mean(d^2))
  where d is the distance to the nearest river. Glover's time scale is a^2 S / T, so this
  preserves the basin-mean time scale. Distances are capped at SEARCH_RADIUS_M (the share
  of capped pixels is reported).

Outputs (spatial_data/)
  basin_irrigation_distance_by_year.csv  gage, year, irrigated_km2, eff_dist_m/ft,
                                         mean_dist_m, frac_capped, source (irrmapper|median)
  basin_irrigation_distance.csv          per basin: median effective distance (+ range)
  pipeline/data/plots/irrigation_distance_*.png   (collect --plots, the default)
"""
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
RIVERS = "WWF/HydroSHEDS/v1/FreeFlowingRivers"
SCALE_M = 30                 # IrrMapper resolution
SEARCH_RADIUS_M = 20_000     # distances beyond this are set to it (and counted)
SIMPLIFY_M = 30              # basin outline simplification before upload (EPSG:5070)
CHUNK_BYTES = 3_000_000      # max GeoJSON size per basin upload task
FIRST_YEAR = C.START_DATE.year
M_TO_FT = 3.28084

STAT_COLS = ['n_pix', 'sum_d', 'sum_d2', 'n_capped']


# --------------------------------------------------------------------------
# Earth Engine
# --------------------------------------------------------------------------
def init_ee(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def ensure_folder(ee, folder):
    try:
        ee.data.getAsset(folder)
    except Exception:
        ee.data.createAsset({'type': 'FOLDER'}, folder)


def asset_exists(ee, asset_id):
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def basin_chunks():
    """Basin polygons (whole drainage) as GeoJSON features, split into upload-sized chunks."""
    import geopandas as gpd
    basins = gpd.read_parquet(C.BASINS_PATH)
    basins.index = [C.fix_gage_id(g) for g in basins.index]
    geom = basins.geometry.to_crs(C.PROJ_CRS).simplify(SIMPLIFY_M, preserve_topology=True)
    geom = gpd.GeoSeries(geom, crs=C.PROJ_CRS).to_crs(4326)
    chunks, cur, size = [], [], 0
    for gage, g in zip(basins.index, geom):
        feat = {'gage': gage, 'geometry': json.loads(gpd.GeoSeries([g]).to_json())['features'][0]['geometry']}
        n = len(json.dumps(feat['geometry']))
        if cur and size + n > CHUNK_BYTES:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(feat)
        size += n
    if cur:
        chunks.append(cur)
    return chunks, tuple(geom.total_bounds)


def upload_basins(ee, folder):
    """Export basin polygons to assets (once). Returns their asset IDs."""
    chunks, bounds = basin_chunks()
    ids = [f"{folder}/basins_{i:02d}" for i in range(len(chunks))]
    tasks = []
    for aid, chunk in zip(ids, chunks):
        if asset_exists(ee, aid):
            continue
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry(f['geometry'], 'EPSG:4326', False), {'gage': f['gage']})
            for f in chunk])
        t = ee.batch.Export.table.toAsset(collection=fc, description=aid.split('/')[-1], assetId=aid)
        t.start()
        tasks.append(t)
        print(f"uploading {len(chunk)} basins -> {aid}")
    if tasks:
        wait(tasks, "basin upload")
    return ids, bounds


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


def irrmapper_years(ee):
    ic = ee.ImageCollection(IRRMAPPER)
    years = ic.aggregate_array('system:time_start').map(
        lambda t: ee.Date(t).get('year')).distinct().sort()
    return [int(y) for y in years.getInfo()]


def year_stats(ee, basins, rivers, year):
    """Sum of d, d^2 and pixel counts over irrigated pixels, per basin, for one year."""
    irr = (ee.ImageCollection(IRRMAPPER)
           .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
           .select('classification').mosaic())
    irrigated = irr.eq(1).selfMask()
    d = rivers.distance(searchRadius=SEARCH_RADIUS_M, maxError=SCALE_M)
    capped = d.mask().Not()                       # farther than the search radius
    d = d.unmask(SEARCH_RADIUS_M).toDouble()
    img = (ee.Image.cat([ee.Image(1).toDouble(), d, d.multiply(d), capped.toDouble()])
           .rename(STAT_COLS).updateMask(irrigated))
    stats = img.reduceRegions(collection=basins, reducer=ee.Reducer.sum(),
                              scale=SCALE_M, crs=f'EPSG:{C.PROJ_CRS}', tileScale=8)
    return stats.map(lambda f: ee.Feature(None, f.toDictionary(['gage'] + STAT_COLS)).set('year', year))


def cmd_submit(args):
    ee = init_ee(args.project)
    ensure_folder(ee, args.folder)
    basin_ids, bounds = upload_basins(ee, args.folder)
    basins = ee.FeatureCollection([ee.FeatureCollection(i) for i in basin_ids]).flatten()
    ucrb = ee.Geometry.Rectangle(list(bounds), 'EPSG:4326', False).buffer(SEARCH_RADIUS_M)
    rivers = ee.FeatureCollection(RIVERS).filterBounds(ucrb)

    years = irrmapper_years(ee)
    if args.years:
        years = [y for y in years if y in args.years]
    print(f"IrrMapper years: {years[0]}-{years[-1]} ({len(years)})")
    n = 0
    for year in years:
        aid = f"{args.folder}/irr_dist_{year}"
        if asset_exists(ee, aid):
            continue
        t = ee.batch.Export.table.toAsset(collection=year_stats(ee, basins, rivers, year),
                                          description=f"irr_dist_{year}", assetId=aid)
        t.start()
        n += 1
    print(f"submitted {n} yearly exports to {args.folder}; check with `status`, then run `collect`")


def cmd_status(args):
    ee = init_ee(args.project)
    tasks = [t for t in ee.data.getTaskList() if t.get('description', '').startswith(('irr_dist_', 'basins_'))]
    if not tasks:
        print("no tasks found")
    for t in sorted(tasks, key=lambda t: t['description']):
        print(f"{t['description']:16s} {t['state']:10s} {t.get('error_message', '')}")


def fetch_stats(ee, folder):
    rows = []
    for a in ee.data.listAssets({'parent': folder})['assets']:
        if not a['name'].split('/')[-1].startswith('irr_dist_'):
            continue
        feats = ee.FeatureCollection(a['name']).getInfo()['features']
        rows += [f['properties'] for f in feats]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Tables (no Earth Engine needed from here on)
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
    keep = ['gage', 'year', 'irrigated_km2', 'eff_dist_m', 'mean_dist_m', 'frac_capped', 'source']
    s = s[keep]

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

    # every basin x year in the record; unmapped years get the median
    full = pd.MultiIndex.from_product([summary.index, range(first_year, last_year + 1)],
                                      names=['gage', 'year']).to_frame(index=False)
    by_year = full.merge(s, on=['gage', 'year'], how='left')
    fill = by_year['eff_dist_m'].isna()
    by_year.loc[fill, 'eff_dist_m'] = by_year.loc[fill, 'gage'].map(summary['median_eff_dist_m'])
    by_year.loc[fill, 'source'] = 'median'
    by_year['eff_dist_ft'] = by_year['eff_dist_m'] * M_TO_FT
    return by_year, summary.reset_index()


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
REF_LINES = [(2640 / M_TO_FT, 'current GLOVER_A_FT (2,640 ft)'),
             (1500 / M_TO_FT, 'CRDSS far (1,500 ft)'), (600 / M_TO_FT, 'CRDSS near (600 ft)')]


def make_plots(by_year, summary, out_dir):
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
    TICKS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

    def plain_log(axis, lo, hi):
        t = [v for v in TICKS if lo <= v <= hi]
        axis.set_ticks(t, [f'{v:,}' for v in t])
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    def refs(ax, axis='x'):
        for v, lab in REF_LINES:
            (ax.axvline if axis == 'x' else ax.axhline)(v, color=INK2, lw=0.8, ls=':')
            if axis == 'x':
                ax.text(v, 1.0, f' {lab}', transform=ax.get_xaxis_transform(), rotation=90,
                        va='top', ha='right', fontsize=7.5, color=INK2)

    # 1. distribution of median effective distance
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(np.log10(max(s['median_eff_dist_m'].min(), 10)),
                       np.log10(s['median_eff_dist_m'].max() * 1.05), 30)
    ax.hist(s['median_eff_dist_m'], bins=bins, color=BLUE, edgecolor=SURF, lw=1)
    ax.set_xscale('log')
    plain_log(ax.xaxis, bins[0], bins[-1])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)   # room for the reference labels
    refs(ax)
    ax.set_xlabel('median effective distance, irrigated land to river (m, log)')
    ax.set_ylabel('basins')
    ax.set_title(f'{len(s)} basins with irrigation', loc='left', fontsize=10.5)
    fig.savefig(out_dir / 'irrigation_distance_hist.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. distance vs irrigated area
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(s['median_irrigated_km2'], s['median_eff_dist_m'], s=18, color=BLUE,
               edgecolor=SURF, lw=1, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ylo = min(s['median_eff_dist_m'].min(), REF_LINES[-1][0]) * 0.8
    ax.set_ylim(ylo, s['median_eff_dist_m'].max() * 1.25)
    plain_log(ax.yaxis, ylo, s['median_eff_dist_m'].max() * 1.25)
    refs(ax, 'y')
    for v, lab in REF_LINES:
        ax.text(ax.get_xlim()[1], v, f' {lab}', va='center', fontsize=7.5, color=INK2)
    ax.set_xlabel('median irrigated area in drainage (km², log)')
    ax.set_ylabel('median effective distance (m, log)')
    ax.set_title('Effective distance vs. irrigated area', loc='left', fontsize=10.5)
    fig.savefig(out_dir / 'irrigation_distance_vs_area.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3. year-to-year change for the most irrigated basins
    top = s.nlargest(8, 'median_irrigated_km2')['gage']
    names = gage_names()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharex=True, gridspec_kw={'hspace': 0.5})
    for ax, gage in zip(axes.flat, top):
        d = by_year[by_year['gage'] == gage]
        mapped = d['source'] == 'irrmapper'
        ax.plot(d.loc[mapped, 'year'], d.loc[mapped, 'eff_dist_m'], color=BLUE, lw=1.5)
        ax.plot(d.loc[~mapped, 'year'], d.loc[~mapped, 'eff_dist_m'], 'o', ms=3,
                color=INK2, label='median (no IrrMapper)')
        ax.set_title(f"{str(names.get(gage, gage))[:30].title()}\n{gage}", fontsize=8, loc='left')
        ax.set_ylim(0, d['eff_dist_m'].max() * 1.15)
    axes.flat[0].set_ylabel('effective distance (m)')
    axes.flat[4].set_ylabel('effective distance (m)')
    axes.flat[0].legend(frameon=False, fontsize=7, loc='lower left')
    fig.suptitle('Yearly effective distance, 8 most-irrigated basins', x=0.01, ha='left', fontsize=10.5)
    fig.savefig(out_dir / 'irrigation_distance_by_year.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4. map (needs geopandas)
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not available, skipping map")
        return
    basins = gpd.read_parquet(C.BASINS_PATH).to_crs(C.PROJ_CRS)
    basins.index = [C.fix_gage_id(g) for g in basins.index]
    basins = basins.join(summary.set_index('gage')['median_eff_dist_m'])
    basins['area'] = basins.area
    basins = basins.sort_values('area', ascending=False)   # small basins drawn on top
    fig, ax = plt.subplots(figsize=(8, 9))
    basins.plot(column='median_eff_dist_m', ax=ax, cmap='Blues', edgecolor=SURF, linewidth=0.3,
                legend=True, missing_kwds={'color': '#f1f0ec', 'label': 'no irrigation'},
                legend_kwds={'label': 'median effective distance (m)', 'shrink': 0.6})
    try:
        gpd.read_parquet(C.UCOL_PATH).to_crs(C.PROJ_CRS).boundary.plot(ax=ax, color=INK2, lw=0.8)
    except Exception:
        pass
    ax.set_axis_off()
    ax.set_title('Median effective distance from irrigated land to river\n'
                 '(each basin = its whole drainage; small basins drawn on top)', loc='left', fontsize=10.5)
    fig.savefig(out_dir / 'irrigation_distance_map.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def gage_names():
    try:
        g = pd.read_parquet(C.GAGES_PATH, columns=['name'])
        return {C.fix_gage_id(k): v for k, v in g['name'].items()}
    except Exception:
        return {}


def cmd_collect(args):
    ee = init_ee(args.project)
    stats = fetch_stats(ee, args.folder)
    if stats.empty:
        raise SystemExit(f"no irr_dist_* assets in {args.folder}; run `submit` and wait for it to finish")
    stats.to_csv(C.DATA_DIR / 'irrigation_distance_raw.csv', index=False)
    finish(stats, args.plots)


def finish(stats, plots=True):
    by_year, summary = build_tables(stats, FIRST_YEAR, C.end_date().year)
    names = gage_names()
    summary.insert(1, 'name', summary['gage'].map(names))
    by_year.to_csv(C.IRR_DISTANCE_BY_YEAR_PATH, index=False)
    summary.to_csv(C.IRR_DISTANCE_PATH, index=False)
    s = summary['median_eff_dist_m']
    print(f"{len(summary)} basins with irrigation; median effective distance "
          f"{s.median():.0f} m ({s.median() * M_TO_FT:.0f} ft), 10-90%: "
          f"{s.quantile(.1):.0f}-{s.quantile(.9):.0f} m")
    capped = summary[summary['max_frac_capped'] > 0.01]
    if len(capped):
        print(f"{len(capped)} basins have >1% of irrigated pixels beyond {SEARCH_RADIUS_M / 1000:.0f} km "
              f"(distance capped): {capped['gage'].tolist()[:10]}")
    print(f"wrote {C.IRR_DISTANCE_PATH.name} and {C.IRR_DISTANCE_BY_YEAR_PATH.name} to {C.IRR_DISTANCE_PATH.parent}")
    if plots:
        make_plots(by_year, summary, C.VALIDATION_PLOT_DIR)
        print(f"plots in {C.VALIDATION_PLOT_DIR}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['submit', 'status', 'collect'])
    p.add_argument('--project', required=True, help='Earth Engine Cloud project ID')
    p.add_argument('--folder', help='asset folder (default projects/<project>/assets/ucrb_irrigation_distance)')
    p.add_argument('--years', nargs='+', type=int, help='submit: only these years (e.g. a quick test)')
    p.add_argument('--no-plots', dest='plots', action='store_false', help='collect: skip plots')
    args = p.parse_args()
    args.folder = args.folder or f"projects/{args.project}/assets/ucrb_irrigation_distance"
    C.ensure_dirs()
    {'submit': cmd_submit, 'status': cmd_status, 'collect': cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
