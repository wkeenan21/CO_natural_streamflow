"""
Download / refresh every time-varying input the naturalization step needs.

Meant to run once per day (e.g. a scheduled GitHub Action). Each dataset is
cached in pipeline/data/, and after the first full download only a recent
window is re-requested and merged in, so provisional values that get revised
are updated without re-downloading the whole record.

    python pipeline/update_inputs.py                  # everything
    python pipeline/update_inputs.py --only streamflow evap
    python pipeline/update_inputs.py --full           # ignore caches, download full records
    python pipeline/update_inputs.py --plots          # also write diversion QA PDFs

Outputs
    pipeline/data/streamflow_cms.parquet              daily mean Q (cms), one column per gage
    pipeline/data/diversions/processed/diversions_filtered_filled_cfs.parquet
    pipeline/data/evap/<RES_NAME>.parquet             daily reservoir evaporation (DRI)

Static inputs (basins, gages, reservoirs, irrigation, distance matrix, USBR natural
flows) are not downloaded here; they live in shiny-app/ucol_natural/spatial_data/.

Sources merged from scripts/v2/workflow.py (streamflow), scripts/v2/evap.py
(reservoir evaporation) and scripts/v3/ucrb_diversions.py (diversions).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import diversions

# Default re-download windows (days before the last cached date)
STREAMFLOW_LOOKBACK = 120     # USGS provisional data is revised for months
DIVERSION_LOOKBACK = 730      # diversion records are often reported a year or more late
EVAP_LOOKBACK = 365


# --------------------------------------------------------------------------
# Streamflow (USGS daily values)
# --------------------------------------------------------------------------
def _nwis_chunk(ids, start, end):
    from pygeohydro import NWIS
    flow = NWIS().get_streamflow(ids, (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
                                 freq="dv")      # cms
    flow.index = pd.to_datetime(flow.index).normalize().tz_localize(None)
    return flow.rename(columns={c: C.fix_gage_id(c) for c in flow.columns})


def _waterdata(ids, start, end):
    """Fallback for gages NWIS doesn't return (as in workflow.py). Needs API_USGS_PAT."""
    from dataretrieval import waterdata
    df, _ = waterdata.get_daily(monitoring_location_id=C.add_usgs_prefix(ids),
                                parameter_code='00060',
                                time=f'{start.strftime("%Y-%m-%d")}/{end.strftime("%Y-%m-%d")}')
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if "statistic_id" in df.columns:          # daily mean only
        df = df[df["statistic_id"] == "00003"]
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df["gage"] = df["monitoring_location_id"].apply(C.fix_gage_id)
    wide = df.pivot_table(index="time", columns="gage", values="value", aggfunc="mean")
    return wide * 0.0283168                   # cfs -> cms


def update_streamflow(lookback_days=STREAMFLOW_LOOKBACK, full=False, chunk=50):
    import geopandas as gpd
    gages = gpd.read_parquet(C.GAGES_PATH)
    ids = sorted({C.fix_gage_id(g) for g in gages.index})

    old = None if full or not C.STREAMFLOW_PATH.exists() else pd.read_parquet(C.STREAMFLOW_PATH)
    end = C.end_date()
    start = C.START_DATE if old is None else max(
        C.START_DATE, old.index.max() - pd.Timedelta(days=lookback_days))
    print(f"streamflow: {len(ids)} gages, {start.date()} to {end.date()}")

    parts = []
    for i in range(0, len(ids), chunk):
        sub = ids[i:i + chunk]
        try:
            parts.append(_nwis_chunk(sub, start, end))
        except Exception as e:
            print(f"  NWIS failed for gages {i}-{i + len(sub)}: {e}")
    new = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    new = new.loc[:, ~new.columns.duplicated()]

    missing = [g for g in ids if g not in new.columns]
    if missing:
        print(f"  {len(missing)} gages not returned by NWIS; trying the Water Data API")
        try:
            wd = _waterdata(missing, start, end)
            new = pd.concat([new, wd[[c for c in wd.columns if c not in new.columns]]], axis=1)
        except Exception as e:
            print(f"  Water Data API failed: {e}")

    if new.empty:
        raise RuntimeError("no streamflow downloaded; keeping the existing cache")
    new = new.sort_index()
    flow = C.merge_window(old, new, start, end)
    flow.index.name = "date"
    still_missing = [g for g in ids if g not in flow.columns]
    if still_missing:
        print(f"  no data at all for {len(still_missing)} gages: {still_missing}")
    flow.to_parquet(C.STREAMFLOW_PATH)
    print(f"streamflow: wrote {flow.shape[1]} gages through {flow.index.max().date()}")
    return flow


# --------------------------------------------------------------------------
# Reservoir evaporation (DRI OpenET reservoir evaporation API; from evap.py)
# --------------------------------------------------------------------------
EVAP_URL = "https://operevap.dri.edu"


def _evap_headers():
    return {"api-key": C.env_key("DRI_EVAP_API_KEY")}


def get_reservoir_timeseries(reservoir_name, start, end, headers):
    """Daily evaporation timeseries for one reservoir."""
    params = {"RES_NAMES": reservoir_name,
              "datasets": "nete-volume-calcs",
              "variables": "NetE,E_volume",
              "start_date": start.strftime("%Y-%m-%d"),
              "end_date": end.strftime("%Y-%m-%d"),
              "units": "metric",
              "output_format": "json"}
    r = requests.get(f"{EVAP_URL}/timeseries/daily/reservoirs/daterange",
                     headers=headers, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def refresh_evap_reservoir_list():
    """Re-query the DRI reservoir list and keep those inside UCOL (evap.py steps 1-3).

    Only needed if DRI adds reservoirs; writes spatial_data/ucol_reservoirs.parquet.
    """
    import geopandas as gpd
    headers = _evap_headers()
    ucol_geom = gpd.read_parquet(C.UCOL_PATH).to_crs(4326).union_all()

    r = requests.post(f"{EVAP_URL}/info/list_RES_NAMES", headers=headers, timeout=120)
    r.raise_for_status()
    names = r.json()["RES_NAMES"]
    meta = []
    for i in range(0, len(names), 50):       # batches to avoid URL length limits
        rr = requests.get(f"{EVAP_URL}/metadata/reservoirs", headers=headers, timeout=120,
                          params={"RES_NAMES": ",".join(names[i:i + 50]), "output_format": "json"})
        rr.raise_for_status()
        meta.extend(rr.json())
    df = pd.DataFrame(meta)
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["LON"], df["LAT"]), crs="EPSG:4326")
    gdf = gdf[gdf.geometry.within(ucol_geom)].copy()
    gdf.to_parquet(C.EVAP_RESERVOIRS_PATH)
    print(f"evap: {len(gdf)} DRI reservoirs inside UCOL written to {C.EVAP_RESERVOIRS_PATH}")


def update_evap(lookback_days=EVAP_LOOKBACK, full=False):
    import geopandas as gpd
    headers = _evap_headers()
    names = gpd.read_parquet(C.EVAP_RESERVOIRS_PATH)["RES_NAME"].tolist()
    end = C.end_date()
    print(f"evap: {len(names)} reservoirs")
    n_fail = 0
    for name in names:
        path = C.EVAP_DIR / f"{name.replace(' ', '_')}.parquet"
        old = None if full or not path.exists() else pd.read_parquet(path)
        if old is not None and len(old):
            old["date"] = pd.to_datetime(old["date"])
            start = max(C.START_DATE, old["date"].max() - pd.Timedelta(days=lookback_days))
        else:
            start = C.START_DATE
        try:
            new = pd.DataFrame(get_reservoir_timeseries(name, start, end, headers))
            if new.empty:
                continue
            new["date"] = pd.to_datetime(new["date"])
            if old is not None and len(old):
                new = pd.concat([old[old["date"] < start], new], ignore_index=True)
            new = new.drop_duplicates(subset="date", keep="last").sort_values("date")
            new.to_parquet(path, index=False)
        except Exception as e:
            n_fail += 1
            print(f"  evap download failed for {name}: {e}")
        time.sleep(0.2)
    print(f"evap: done ({n_fail} failed; failed reservoirs keep their cached data)")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=["streamflow", "diversions", "evap"],
                   help="update only these datasets")
    p.add_argument("--full", action="store_true", help="ignore caches and download full records")
    p.add_argument("--plots", action="store_true", help="write diversion QA plots (slow)")
    p.add_argument("--refresh-evap-reservoirs", action="store_true",
                   help="re-query the DRI reservoir list before downloading evap")
    args = p.parse_args()
    todo = args.only or ["streamflow", "diversions", "evap"]

    C.ensure_dirs()
    failures = []
    if "streamflow" in todo:
        try:
            update_streamflow(full=args.full)
        except Exception as e:
            failures.append(f"streamflow: {e}")
    if "diversions" in todo:
        try:
            diversions.update_diversions(lookback_days=(10 ** 5 if args.full else DIVERSION_LOOKBACK),
                                         make_plots=args.plots)
        except Exception as e:
            failures.append(f"diversions: {e}")
    if "evap" in todo:
        try:
            if args.refresh_evap_reservoirs:
                refresh_evap_reservoir_list()
            update_evap(full=args.full)
        except Exception as e:
            failures.append(f"evap: {e}")

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nall inputs updated")


if __name__ == "__main__":
    main()
