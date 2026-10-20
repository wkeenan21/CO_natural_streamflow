"""
Daily surface-water diversion records for the Upper Colorado River Basin.

Adapted from scripts/v3/ucrb_diversions.py (Jacob Knight & Samuel Lopez, USGS, 2024),
which accompanies the USGS data release "Database of surface water diversion sites
and daily withdrawals for the Upper Colorado River Basin, 1980-2022". The outlier
filter (`divfilter`) and gap filling (`fill_missing_diversion_records`) follow the
USGS logic; the changes are for running it repeatedly:

  * Downloads are cached per site in `pipeline/data/diversions/raw/<source>/`.
    CDSS, NMOSE and USGS sites only re-request the last `lookback_days`
    and merge that window into the cache; the other sources are one request per
    site for the full record, so they are re-downloaded in full.
  * If a site's download fails but a cached copy exists, the cached copy is used,
    so one website being down does not silently drop sites from the dataset.
  * Global variables (dates, EPSG, API key) are passed as arguments.
  * QA plots (PDFs) are optional (`make_plots=True`).
  * Arizona (ADWR) sites are skipped, as in the original __main__ block.

Required inputs (not downloaded; copy from the original `diversion/input` folder):
    pipeline/inputs/diversions/ucrb_diversion_master_table.csv
    pipeline/inputs/diversions/nmose_historical_data/
    pipeline/inputs/diversions/utdwr_historical_data/
    pipeline/inputs/diversions/cuwcd_historical_data/
    pipeline/inputs/diversions/wyseo_historical_data/
"""
import re
import shutil
from calendar import monthrange
from io import StringIO

import numpy as np
import pandas as pd
import requests

import common as C

MODEL_EPSG = 26912        # UTM 12N, used by the USGS site tables
TIMEOUT = 120             # seconds per web request

SITE_INFO_COLS = ["siteID", "siteName", "siteUse", "siteLat", "siteLong", "siteSource",
                  "siteFile", "startDate", "endDate", "noFillYears", "shortID",
                  "destinationCode", "destinationFlag", "siteFolder"]


# --------------------------------------------------------------------------
# Utilities from ucrb_diversions.py
# --------------------------------------------------------------------------
def divfilter(df, mindiff=25., minfact=2., sig=3.):
    """Identify time series outliers using an initial and secondary filter.

    First pass: records more than `mindiff` cfs AND more than `minfact` times the
    median annual maximum are flagged (mdflag). Second pass: records more than `sig`
    standard deviations (of the annual maxima not flagged in pass one) above the
    median annual maximum are flagged (sdflag).
    (Vectorized; same result as the original row-wise apply.)
    """
    df.loc[:, "year"] = df.loc[:, "datetime"].dt.year
    dfy = df.loc[df["discharge_cfs"] > 0.].groupby("year").max()
    aym = np.median(dfy.loc[:, "discharge_cfs"])

    q = df["discharge_cfs"]
    df.loc[:, "mdflag"] = ((q - aym > mindiff) & (q > minfact * aym)).astype(int)

    dfy = df.loc[(df["mdflag"] == 0) & (df["discharge_cfs"] > 0.)].groupby("year").max()
    aymstd = np.std(dfy.loc[:, "discharge_cfs"])

    df.loc[:, "sdflag"] = (q > (aym + sig * aymstd)).astype(int)
    return df, aym


def gaplengths(vals):
    """Length of the run of consecutive nulls that each null value belongs to."""
    i0 = 0
    x = 0
    cts = []
    for i, v in enumerate(vals):
        if np.isnan(v):
            x += 1
            cts.append(x)
        else:
            cts[i0: i] = [x] * (i - i0)
            i0 = i
            x = 0
            cts.append(x)
    return cts


def build_spdf(start, end, spfreq="M", tsfreq="D"):
    """Daily dataframe with year/month/day, monthly stress period and 1-indexed totim."""
    dts = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq=tsfreq)
    df = pd.DataFrame(data={"year": dts.year, "month": dts.month, "day": dts.day}, index=dts)
    if spfreq == "M":
        df.loc[:, "sp"] = df.groupby(["year", "month"]).ngroup()
        df.loc[:, "ts"] = df.groupby("sp").cumcount()
    df.loc[:, "totim"] = range(1, len(df) + 1)
    return df


def format_sites_df(df):
    """Add point geometry from decimal lat/long and UTM 12N x/y columns."""
    import geopandas as gpd
    from shapely.geometry import Point

    df = df.copy()
    df.loc[:, "geometry"] = df.apply(lambda x: Point(x.siteLong, x.siteLat), axis=1)
    df = gpd.GeoDataFrame(df, geometry="geometry", crs="epsg:4269")
    df.to_crs(epsg=MODEL_EPSG, inplace=True)
    df.loc[:, "utmX"] = df.geometry.x
    df.loc[:, "utmY"] = df.geometry.y
    return df


# --------------------------------------------------------------------------
# Per-source downloaders: each returns a DataFrame indexed by date with a
# single 'discharge_cfs' column.
# --------------------------------------------------------------------------
def _as_daily(df, date_col, value_col):
    df = df.rename(columns={value_col: "discharge_cfs"})
    df["date"] = pd.to_datetime(df[date_col])
    return df.set_index("date")[["discharge_cfs"]]


def fetch_cdss(row, start, end, api_key=None):
    params = {'format': "csvforced",
              'dateFormat': "dateOnly",
              'min-dataMeasDate': start.strftime("%m-%d-%Y"),
              'max-dataMeasDate': end.strftime("%m-%d-%Y"),
              'wcIdentifier': "*Total (Diversion)",
              'wdid': str(int(row.cdssID))}
    if api_key:
        params["apiKey"] = api_key
    rr = requests.get("https://dwr.state.co.us/Rest/GET/api/v2/structures/divrec/divrecday/",
                      params=params, timeout=TIMEOUT)
    rr.raise_for_status()
    df = pd.read_csv(StringIO(rr.text), skiprows=2)
    return _as_daily(df, "dataMeasDate", "dataValue")


def fetch_nmose(row, start, end, api_key=None):
    params = {"type": "S",
              "sDate": start.strftime("%m/%d/%Y"),
              "eDate": end.strftime("%m/%d/%Y"),
              "dischargeData": "davg",
              "rptFormat": "CSV",
              "sort": "asc",
              "id": row.nmoseID}
    rr = requests.get("http://meas.ose.state.nm.us/ReportProxy", params=params, timeout=TIMEOUT)
    rr.raise_for_status()
    df = pd.read_csv(StringIO(rr.text), skiprows=1)
    num = df._get_numeric_data()
    num[num < 0] = 0
    df[num.columns] = num
    return _as_daily(df, "Day", "DischargeAvg (cfs)")


def fetch_utdwr(row, start, end, api_key=None):
    year = pd.Timestamp.today().strftime("%Y")
    url = (f"https://www.waterrights.utah.gov/cgi-bin/dvrtview.exe?STATION_ID={row.utdwrID}"
           f"&RECORD_YEAR={year}&Modinfo=Daily_Comma")
    rr = requests.get(url, timeout=TIMEOUT)
    rr.raise_for_status()
    url_raw = []
    for line in StringIO(rr.text).readlines():
        if line.startswith("Daily comma delimited"):
            url_raw = re.findall('"([^"]*)"', line)
    url_full = "https://www.waterrights.utah.gov" + "".join(url_raw)
    rrr = requests.get(url_full, timeout=TIMEOUT)
    rrr.raise_for_status()
    df = pd.read_csv(StringIO(rrr.text))
    df.columns = ["year", "month", "day", "discharge_cfs"]
    df["date"] = pd.to_datetime(df[['year', 'month', 'day']])
    return df.set_index("date")[["discharge_cfs"]]


def fetch_wyseo(row, start, end, api_key=None):
    base = ("https://seoflow.wyo.gov/Export/BulkExport?DateRange=EntirePeriodOfRecord"
            "&TimeZone=0&Calendar=CALENDARYEAR&Interval=Daily&Step=1&ExportFormat=csv"
            "&TimeAligned=True&RoundData=False&IncludeGradeCodes=False&IncludeApprovalLevels=False"
            "&IncludeInterpolationTypes=False&Datasets[0].DatasetName=Discharge.Discharge%40{0}"
            "&Datasets[0].Calculation=Aggregate&Datasets[0].UnitId=208&_=").format(row.wyseoID)
    rr = requests.get(base, timeout=TIMEOUT)
    rr.raise_for_status()
    df = pd.read_csv(StringIO(rr.text), skiprows=4)
    if len(df) == 0:
        print(f"  no data for {row.siteName} {row.wyseoID}; retrying with 'Discharge (cfs)'")
        rr = requests.get(base.replace("Discharge.Discharge", "Discharge.Discharge (cfs)"),
                          timeout=TIMEOUT)
        df = pd.read_csv(StringIO(rr.text), skiprows=4)
    df['date'] = pd.to_datetime(df['Start of Interval (UTC)']).dt.normalize().dt.tz_localize(None)
    df = df.rename(columns={'Average (ft^3/s)': 'discharge_cfs'})
    return df.set_index("date")[["discharge_cfs"]]


def fetch_usbr(row, start, end, api_key=None):
    url = f"https://www.usbr.gov/uc/water/hydrodata/gage_data/{int(row.usbrID)}/csv/19.csv"
    rr = requests.get(url, timeout=TIMEOUT)
    rr.raise_for_status()
    df = pd.read_csv(StringIO(rr.text))
    df.columns = ["date", "discharge_cfs"]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["discharge_cfs"]]


def fetch_usgs(row, start, end, api_key=None):
    params = {"format": "rdb",
              "startDT": start.strftime("%Y-%m-%d"),
              "endDT": end.strftime("%Y-%m-%d"),
              "statCd": "00003",
              "parameterCd": "00060",
              "siteStatus": "all",
              "sites": "{0:08d}".format(int(str(row.usgsID).replace("USGS_", "")))}
    rr = requests.get("https://waterservices.usgs.gov/nwis/dv", params=params, timeout=TIMEOUT)
    rr.raise_for_status()
    temp = StringIO(rr.text)
    line, i = '#', -1
    while line[0] == '#':
        line = temp.readline()
        i += 1
    df = pd.read_csv(StringIO(rr.text), sep='\t', header=[i, i + 1])
    df.drop(df.columns[[0, 1, 4]], axis=1, inplace=True)
    df.columns = ["date", "discharge_cfs"]
    df = df.loc[df["discharge_cfs"] != "Ssn"]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["discharge_cfs"]]


# Source settings. `folder` is the folder the filter step reads each site's
# record from: the raw download, or the download combined with the manually
# collected historical record (as in ucrb_diversions.py, NMOSE uses the raw
# download even though a combined file is written).
SOURCES = {
    "CDSS":  dict(id_col="cdssID",  fetch=fetch_cdss,  incremental=True,  hist=None,                     folder="cdss_raw_data"),
    "NMOSE": dict(id_col="nmoseID", fetch=fetch_nmose, incremental=True,  hist=["nmose_historical_data"], folder="nmose_raw_data"),
    "USBR":  dict(id_col="usbrID",  fetch=fetch_usbr,  incremental=False, hist=None,                     folder="usbr_raw_data"),
    "USGS":  dict(id_col="usgsID",  fetch=fetch_usgs,  incremental=True,  hist=None,                     folder="usgs_raw_data"),
    "UTDWR": dict(id_col="utdwrID", fetch=fetch_utdwr, incremental=False, hist=["utdwr_historical_data", "cuwcd_historical_data"], folder="utdwr_combined_data"),
    "WYSEO": dict(id_col="wyseoID", fetch=fetch_wyseo, incremental=False, hist=["wyseo_historical_data"], folder="wyseo_combined_data"),
}


def _read_record(path):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df[["discharge_cfs"]]


def _combine_with_historical(auto, hist_path, sp_df, monthly=False):
    """Fill a manually-collected historical record with the downloaded record."""
    temp = pd.read_csv(hist_path)
    temp["date"] = pd.to_datetime(temp["date"])
    temp = sp_df.join(temp.set_index("date"), how="left")
    if monthly and "monthly_cfsd" in temp.columns:
        # Utah historical records: spread monthly cfs-days evenly over the month
        for _, rr in temp.loc[temp["monthly_cfsd"].notnull()].iterrows():
            ix = temp.loc[(temp["year"] == rr.year) & (temp["month"] == rr.month)].index
            temp.loc[ix, "discharge_cfs"] = rr.monthly_cfsd / monthrange(int(rr.year), int(rr.month))[1]
    if "discharge_cfs" not in temp.columns:
        temp["discharge_cfs"] = np.nan
    df = temp.join(auto.rename(columns={"discharge_cfs": "auto_cfs"}), how="left")
    df["discharge_cfs"] = df["discharge_cfs"].fillna(df["auto_cfs"])
    return df[["discharge_cfs"]]


def download_source(source, sites, sp_df, end, lookback_days, api_key=None):
    """Download/refresh every site of one data source. Returns the site info table."""
    cfg = SOURCES[source]
    raw_dir = C.DIVERSION_RAW_DIR / f"{source.lower()}_raw_data"
    comb_dir = C.DIVERSION_RAW_DIR / f"{source.lower()}_combined_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    comb_dir.mkdir(parents=True, exist_ok=True)

    sdf = sites.loc[sites["dataSource"] == source].copy()
    if source != "WYSEO":   # WY sites without an online ID come from the historical folder
        sdf = sdf.loc[sdf[cfg["id_col"]].notnull()]
    print(f"{source}: {len(sdf)} sites")

    info = []
    n_ok = n_cached = n_fail = 0
    for _, r in sdf.iterrows():
        site_file = f"{r.siteName}.csv"
        raw_path = raw_dir / site_file
        comb_path = comb_dir / site_file
        got_data = False

        wy_offline = source == "WYSEO" and len(str(r.wyseoID)) <= 4
        if wy_offline:
            # no online data: copy the historical record (as in ucrb_diversions.py)
            shutil.copy2(C.DIVERSION_INPUT_DIR / "wyseo_historical_data" / site_file, comb_path)
            got_data = True
        else:
            cached = _read_record(raw_path) if raw_path.exists() else None
            if cfg["incremental"] and cached is not None and not cached.empty:
                start = max(C.START_DATE, cached.index.max() - pd.Timedelta(days=lookback_days))
            else:
                start = C.START_DATE
            try:
                new = cfg["fetch"](r, start, end, api_key)
                new["discharge_cfs"] = pd.to_numeric(new["discharge_cfs"], errors="coerce")
                new = new[~new.index.duplicated(keep="last")]
                if cfg["incremental"] and cached is not None:
                    rec = C.merge_window(cached, new, start, end)
                else:
                    rec = new.sort_index()
                rec.to_csv(raw_path, index_label="date")
                got_data = True
                n_ok += 1
            except Exception as e:
                if cached is not None:
                    print(f"  download failed for {source} {r.siteName}; using cached copy ({e})")
                    rec = cached
                    got_data = True
                    n_cached += 1
                else:
                    print(f"  could not download {source} {r.siteName}: {e}")
                    n_fail += 1

            if got_data and cfg["hist"]:
                if r.historicalRecord == "y":
                    hist_path = None
                    for h in cfg["hist"]:
                        p = C.DIVERSION_INPUT_DIR / h / site_file
                        if p.exists():
                            hist_path = p
                            break
                    if hist_path is None:
                        raise FileNotFoundError(
                            f"historical record for {r.siteName} not found in "
                            f"{[str(C.DIVERSION_INPUT_DIR / h) for h in cfg['hist']]}")
                    combined = _combine_with_historical(rec, hist_path, sp_df,
                                                        monthly=(source == "UTDWR"))
                    combined.to_csv(comb_path, index_label="date")
                else:
                    rec.to_csv(comb_path, index_label="date")

        if got_data:
            info.append({"siteID": r[cfg["id_col"]], "siteName": r.siteName, "siteUse": r.siteUse,
                         "siteLat": r.decLat, "siteLong": r.decLong, "siteSource": r.dataSource,
                         "siteFile": site_file, "startDate": r.startDate, "endDate": r.endDate,
                         "noFillYears": r.no_fill_years, "shortID": r.shortID,
                         "destinationCode": r.destinationCode, "destinationFlag": r.destinationFlag,
                         "siteFolder": cfg["folder"]})

    print(f"{source}: {n_ok} downloaded, {n_cached} from cache, {n_fail} failed")
    return pd.DataFrame(info, columns=SITE_INFO_COLS)


# --------------------------------------------------------------------------
# Combine, filter and fill (USGS logic)
# --------------------------------------------------------------------------
def build_diversion_dataframe(site_tables):
    """Combine the per-source site tables (replaces build_diversion_dataframe)."""
    sdf = pd.concat([format_sites_df(t) for t in site_tables if len(t)], ignore_index=True)
    sdf.sort_values("shortID", inplace=True)
    sdf.reset_index(drop=True, inplace=True)

    missing = sdf.loc[sdf["utmY"].isnull()]
    missing.drop(columns="geometry").to_csv(C.DIVERSION_PROC_DIR / "diversion_sites_missing_loc.csv")
    sdf = sdf.drop(missing.index)

    sdf.loc[:, "tabFile"] = sdf.apply(lambda xx: "div_{0:04d}_{1}.tab".format(int(xx.shortID), xx.siteUse), axis=1)
    sdf.drop(columns="geometry").to_csv(C.DIVERSION_PROC_DIR / "combined_diversion_sites.csv")
    return pd.DataFrame(sdf.drop(columns="geometry"))


def _plot_pages(pdf_path, items, plot_fn):
    """QA plots, 8 per page (replaces the inline PdfPages blocks)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_path) as pdf:
        for p in range(0, len(items), 8):
            fig, axes = plt.subplots(4, 2, figsize=(8.5, 11), dpi=100)
            for ax, item in zip(axes.flat, items[p:p + 8]):
                plot_fn(ax, item)
                ax.set_xlim([pd.to_datetime("1980-01-01"), pd.Timestamp.today()])
                ax.xaxis.set_major_locator(mdates.YearLocator(10))
                ax.xaxis.set_minor_locator(mdates.YearLocator(1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            for ax in axes.flat[len(items[p:p + 8]):]:
                ax.axis("off")
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)


def filter_diversion_records(sdf, sp_df, make_plots=False):
    """Reduce high outliers in each site's daily record (filter_diversion_records)."""
    print("filtering diversion records")
    sdf = sdf.copy()
    sdf['startDate'] = pd.to_datetime(sdf['startDate'].replace('', pd.NaT)).fillna(pd.Timestamp('1900-01-01'))
    sdf['endDate'] = pd.to_datetime(sdf['endDate'].replace('', pd.NaT)).fillna(pd.Timestamp('2100-01-01'))

    filtered, original, plots, problem_sites = {}, {}, [], []
    for ct, (i, r) in enumerate(sdf.iterrows(), start=1):
        if ct % 100 == 0:
            print(f"filtering site {ct}/{len(sdf)}")
        df = pd.read_csv(C.DIVERSION_RAW_DIR / r.siteFolder / r.siteFile)
        df.loc[:, "discharge_cfs"] = pd.to_numeric(df.loc[:, "discharge_cfs"], errors='coerce')
        df.loc[df["discharge_cfs"] < 0., "discharge_cfs"] = np.nan
        df = df.loc[df["discharge_cfs"].notnull()].copy()

        if len(df.loc[df["discharge_cfs"] > 0.]) == 0:
            print(f"no data for site {r.siteName} {r.siteID}")
            problem_sites.append(i)
            continue

        df["date"] = pd.to_datetime(df["date"])
        df["datetime"] = df["date"]
        df = df.groupby("date").max()

        # identify outliers and decrease them to the median of the annual maxima
        df, aym = divfilter(df)
        df.loc[:, "discharge_cfs_orig"] = df.loc[:, "discharge_cfs"].copy()
        df.loc[(df["mdflag"] == 1) & (df["sdflag"] == 1), "discharge_cfs"] = aym
        if make_plots:
            plots.append((f"{r.siteName} shortID {int(r.shortID):04d}", df.copy()))

        df = df.drop(["datetime", "year"], axis=1)
        df = sp_df.join(df, how="outer")
        dt = pd.to_datetime(df.index.values)
        # enforce zero values before diversion start date and after diversion end date
        outside = (dt < r.startDate) | (dt > r.endDate)
        df.loc[outside, "discharge_cfs"] = 0.
        df.loc[outside, "discharge_cfs_orig"] = 0.

        df_out = df.loc[df["totim"].notnull()]
        colname = "div_{0:04d}_{1}".format(int(r.shortID), r.siteUse)
        filtered[colname] = df_out["discharge_cfs"]
        original[colname] = df_out["discharge_cfs_orig"]

    mdf = sp_df.filter([]).join(pd.DataFrame(filtered), how="left")
    odf = sp_df.filter([]).join(pd.DataFrame(original), how="left")
    odf.to_csv(C.DIVERSION_PROC_DIR / "combined_diversion_records_raw_cfs.csv")
    mdf.to_csv(C.DIVERSION_PROC_DIR / "combined_diversion_records_filtered_cfs.csv")
    if problem_sites:
        sdf.loc[problem_sites].to_csv(C.DIVERSION_PROC_DIR / "nodatafound_diversion_sites.csv")

    if make_plots:
        def plot_fn(ax, item):
            title, d = item
            ax.plot(d["datetime"], d["discharge_cfs_orig"], lw=0.5, c="k")
            ax.plot(d["datetime"], d["discharge_cfs_orig"], marker=".", lw=0, markersize=3, rasterized=True)
            ax.scatter(d.loc[d["mdflag"] == 1, "datetime"], d.loc[d["mdflag"] == 1, "discharge_cfs_orig"],
                       marker="^", facecolors="None", edgecolors="b", alpha=0.5)
            ax.scatter(d.loc[d["sdflag"] == 1, "datetime"], d.loc[d["sdflag"] == 1, "discharge_cfs_orig"],
                       marker=".", facecolors="None", edgecolors="r", alpha=0.5)
            ax.set_title(title, loc="left")
        _plot_pages(C.DIVERSION_PROC_DIR / "diversion_plots_filtered_cfs.pdf", plots, plot_fn)
    return mdf


def fill_missing_diversion_records(mdf, sdf, sp_df1, fill_all_missing=False, make_plots=False):
    """Interpolate and fill missing values in each site's record (USGS logic).

    1. Years listed in `no_fill_years` are set to 0 where missing.
    2. Gaps shorter than 90 days are linearly interpolated.
    3. Years with no records at all are filled with the site's monthly means.
    4. Everything still missing is set to 0 (or monthly means if fill_all_missing).
    """
    print("interpolating and filling missing diversion records")
    df = mdf.copy()
    df.index.name = "datetime"
    divcols = df.columns

    df = sp_df1.join(df, how="outer", on="datetime")
    df.index = df.pop("datetime")

    sdf = sdf.copy()
    sdf.index = sdf["shortID"].apply(lambda x: "{0:04d}".format(int(x)))
    sdf["divcol"] = sdf["tabFile"].str.replace(".tab", "", regex=False)

    # fill nan values with zero in years designated no-fill
    for _, r in sdf.loc[sdf["noFillYears"].notnull()].iterrows():
        if r.divcol not in df.columns:
            continue
        for yr in [int(float(x)) for x in str(r.noFillYears).split(";") if x.strip()]:
            df.loc[(df["year"] == yr) & (df[r.divcol].isnull()), r.divcol] = 0.

    # identify years for each site with no records
    ydf = df.groupby("year").max().filter(divcols)
    ydf = sp_df1.join(ydf, how="left", on="year")

    # monthly mean diversions
    mmean = df.groupby("month").mean().fillna(0.).filter(divcols)
    mmean = sp_df1.join(mmean, how="left", on="month")

    # interpolate linearly between records no more than 90 days apart
    df1 = df.copy()
    gaps = {}
    for k, div in enumerate(divcols, start=1):
        df1[div] = df1[div].interpolate(method="linear", limit=90,
                                        limit_direction="backward", limit_area="inside")
        gaps[div] = gaplengths(df[div].to_numpy(dtype=float))
        if k % 100 == 0:
            print(f"interpolating site {k}/{len(divcols)}")

    a = df1[divcols].to_numpy(dtype=float)
    m = pd.DataFrame(gaps, index=df.index)[divcols].to_numpy(dtype=float)
    a = np.where(m >= 90., np.nan, a)          # mask interpolation over gaps >= 90 days

    b = mmean[divcols].to_numpy(dtype=float)
    m = ydf[divcols].to_numpy(dtype=float)
    a = np.where(np.isnan(m), b, a)            # years with no data -> monthly means

    out = pd.DataFrame(a, index=df1.index, columns=divcols)
    if fill_all_missing:
        out = out.fillna(pd.DataFrame(b, index=df1.index, columns=divcols))
    out = out.fillna(0.)
    out.index.name = "date"

    if make_plots:
        names = sdf.set_index("divcol")["siteName"].to_dict()
        def plot_fn(ax, div):
            ax.plot(df.index, df[div], marker=".", lw=0, markersize=3, rasterized=True)
            ax.plot(out.index, out[div], c="m", lw=0.25)
            ax.set_title(f"{names.get(div, div)} {div}", loc="left")
        tag = "fill_all" if fill_all_missing else "fill_years"
        _plot_pages(C.DIVERSION_PROC_DIR / f"diversion_plots_filtered_filled_cfs_{tag}.pdf",
                    list(divcols), plot_fn)
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def update_diversions(lookback_days=730, make_plots=False):
    """Refresh all diversion records and write the filtered/filled daily table (cfs)."""
    if not C.DIVERSION_MASTER_CSV.exists():
        raise FileNotFoundError(
            f"{C.DIVERSION_MASTER_CSV} not found. Copy ucrb_diversion_master_table.csv and the "
            f"*_historical_data folders from the original diversion 'input' folder into "
            f"{C.DIVERSION_INPUT_DIR}")
    end = C.end_date()
    sites = pd.read_csv(C.DIVERSION_MASTER_CSV)
    api_key = C.env_key("CDSS_API_KEY")

    spdf = build_spdf(start=C.START_DATE, end=end)
    spdf1 = spdf.copy()
    spdf1["datetime"] = pd.to_datetime(spdf1.index.values)
    spdf1.index = spdf1.pop("totim")

    tables = []
    for source in SOURCES:
        tries = 5 if source == "NMOSE" else 1   # NM occasionally fails on the first try
        for attempt in range(1, tries + 1):
            try:
                tables.append(download_source(source, sites, spdf, end, lookback_days,
                                              api_key=api_key if source == "CDSS" else None))
                break
            except FileNotFoundError:
                raise
            except Exception as e:
                print(f"{source} retrieval failed ({e}); attempt {attempt}/{tries}")

    sdf = build_diversion_dataframe(tables)
    mdf = filter_diversion_records(sdf, spdf, make_plots=make_plots)
    out = fill_missing_diversion_records(mdf, sdf, spdf1, fill_all_missing=False, make_plots=make_plots)
    out.to_parquet(C.DIVERSIONS_PATH)
    print(f"wrote {out.shape[1]} diversion records through {out.index.max().date()} to {C.DIVERSIONS_PATH}")
    return out
