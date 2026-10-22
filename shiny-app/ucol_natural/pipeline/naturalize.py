"""
Estimate daily natural streamflow for every UCOL gage and write the files the app reads.

Port of the "MERGE DIVERSIONS WITH STREAMFLOW" section of scripts/v2/workflow.py.
Run pipeline/update_inputs.py first.

    python pipeline/naturalize.py
    python pipeline/naturalize.py --gages 09010500 09019500   # only these (plus nothing else)

For each basin, smallest to largest:
  Q_NAT_cfs   = Q_cfs - evap_cfs - Q_CU_cfs   (add back consumptive use and reservoir evap)
  Q_NAT2_cfs  = Q_NAT_cfs with irrigation diversions removed and Glover-lagged return
                flows added back (floored at 0). Below major reservoirs, Q_NAT2_cfs is
                rebuilt from naturalized inflow gages scaled by frac_in, and Q_NAT3_cfs
                is the same with Muskingum-routed inflows.
  Q_NATUSBR_cfs  USBR monthly natural flow spread to daily, for comparison.

Outputs
  shiny-app/ucol_natural/timeseries/<gage>.parquet        app file ('Q' columns + CU by type)
  pipeline/data/naturalized_full/<gage>.parquet           every diversion column
  pipeline/data/nat_results.csv                           per-gage diagnostics (NSE, PBIAS, ...)
  pipeline/data/inflow_validation.csv                     leave-one-out test of the below-reservoir
                                                          scaling: NSE/PBIAS per (basin, inflow gage)
  pipeline/data/inflow_validation_by_year.csv             the same per water year
  spatial_data/ucrb_diversion_master_table.parquet        diversion points for the app map
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

# Consumptive-use multipliers by diversion type. 'irrigation' is replaced per
# basin with -1 * irrigation efficiency.
#   INTERBASIN = all water leaves UCOL
#   INTRABASIN = water may or may not leave sub-basin. Does not leave UCOL.
#   TRANSBASIN = water imported from a different sub-basin within UCOL.
USE_DICT = {
    'irrigation': -0.6, 'municipal': -0.3, 'interbasin': -1,
    'industrial': -1, 'hydropower': 0, 'intrabasin': -1, 'transbasin': 1,
}

# Irrigation efficiency by method (from scripts/v2/irrigation_data.py)
FLOOD_EFF, SPRINK_EFF, DRIP_EFF = 0.5, 0.8, 0.9

# Glover return-flow aquifer parameters
GLOVER_A_FT = 2640        # distance to stream (ft)
GLOVER_T = 5000.0         # transmissivity (ft^2/day)
GLOVER_S = 0.20           # specific yield
GLOVER_MAX_LAG_DAYS = 365 * 5

# basins dropped from the analysis ("chipmunk outlet thing" etc.)
EXCLUDED_BASINS = ['09014050', '09060799', '09019000']

# leave-one-out validation below reservoirs: only evaluate an inflow gage in years when it
# carries at least this share of the basin's estimated natural flow (Q_NAT2)
MIN_VALIDATION_SHARE = 0.05


# --------------------------------------------------------------------------
# Hydrology helpers (from workflow.py)
# --------------------------------------------------------------------------
def glover_unit_response(t_days, a_ft, T_sqft_day, S):
    """Daily fraction of a recharge pulse returning to the stream at time t (Glover)."""
    t = np.asarray(t_days, dtype=float)
    q = np.zeros_like(t)
    valid = t > 0
    D = T_sqft_day / S
    t_v = t[valid]
    q[valid] = (a_ft / (2 * np.sqrt(np.pi * D * (t_v ** 3)))) * np.exp(-(a_ft ** 2) / (4 * D * t_v))
    return q


def apply_glover_warmup(rf_series, kernel, warmup_years=3, cycle_days=365):
    """Convolve return flows with the kernel, prepending repeated first-year cycles as warm-up."""
    cycle_length = min(cycle_days, len(rf_series))
    warmup_prefix = np.tile(rf_series[:cycle_length], warmup_years)
    extended = np.concatenate([warmup_prefix, rf_series])
    lagged = np.convolve(extended, kernel, mode='full')
    off = len(warmup_prefix)
    return lagged[off: off + len(rf_series)]


def mass_balance_compare(s1, s2):
    """sum(s1) / sum(s2) over dates where both are non-null."""
    combined = pd.concat([s1, s2], axis=1, join="inner").dropna()
    sum_s2 = combined.iloc[:, 1].sum()
    if sum_s2 == 0:
        return 'division by zero'
    return combined.iloc[:, 0].sum() / sum_s2


def route_streamflow_muskingum_cfs(s_cfs, upstream_gage, downstream_gage, dm, a=0.5, b=0.4, x=0.2):
    """Route a daily cfs series from an upstream to a downstream gage (hourly Muskingum).

    Velocity v = a * Q**b (m/s) uses the series mean; distances come from the
    in-stream distance matrix `dm` (km; rows upstream, columns downstream).
    """
    CFS_TO_CMS = 0.028316847
    CMS_TO_CFS = 1.0 / CFS_TO_CMS

    if upstream_gage == downstream_gage:
        return s_cfs.copy()
    try:
        distance_km = dm.loc[upstream_gage, downstream_gage]
    except KeyError:
        raise KeyError(f"Upstream gage '{upstream_gage}' and downstream gage '{downstream_gage}' "
                       f"not found in distance matrix.")
    if pd.isna(distance_km) or distance_km <= 0:
        raise ValueError(f"Invalid distance ({distance_km} km) between {upstream_gage} and {downstream_gage}.")

    distance_m = distance_km * 1000.0
    dt_sec = 3600.0

    hourly_s = s_cfs.resample('h').ffill(limit=24)
    routed_hourly = pd.Series(index=hourly_s.index, dtype=float)
    block_groups = hourly_s.groupby((hourly_s.isna() != hourly_s.isna().shift()).cumsum())

    for _, sub_series in block_groups:
        if sub_series.isna().any():
            continue
        inflow = (sub_series * CFS_TO_CMS).values
        n = len(inflow)
        outflow = np.zeros(n)
        if n > 0:
            outflow[0] = max(inflow[0], 0.0)

        q_mean = s_cfs.mean()  # mean velocity (note: workflow.py uses the cfs mean here)
        v = a * (q_mean ** b)
        K = distance_m / v
        for i in range(1, n):
            denom = 2 * K * (1 - x) + dt_sec
            if denom <= 0:
                C0, C1, C2 = 0.0, 1.0, 0.0
            else:
                C0 = (-2 * K * x + dt_sec) / denom
                C1 = (2 * K * x + dt_sec) / denom
                C2 = (2 * K * (1 - x) - dt_sec) / denom
            outflow[i] = max(0.0, C0 * inflow[i] + C1 * inflow[i - 1] + C2 * outflow[i - 1])
        routed_hourly.loc[sub_series.index] = outflow * CMS_TO_CFS

    return routed_hourly.resample('D').mean().reindex(s_cfs.index)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_diversion_sites(basins_crs):
    """Diversion points, with intrabasin deliveries duplicated as 'transbasin' imports.

    If an intrabasin transfer delivers to another sub-basin within UCOL, it is a
    transbasin import for that sub-basin. For a large basin containing both ends,
    the intrabasin consumptive use cancels out with the transbasin import.
    """
    import geopandas as gpd
    if C.DIVERSION_MASTER_CSV.exists():
        dvrs = pd.read_csv(C.DIVERSION_MASTER_CSV, dtype=str)   # all text, as gpd.read_file gave
        dvrs_intra = dvrs[dvrs['siteUse'] == 'intrabasin'].copy()
        dvrs_intra['siteUse'] = 'transbasin'
        dvrs_intra['origin_decLat'] = dvrs_intra['decLat']
        dvrs_intra['origin_decLong'] = dvrs_intra['decLong']
        dvrs_intra['decLat'] = dvrs_intra['dest_decLat']
        dvrs_intra['decLong'] = dvrs_intra['dest_decLong']
        dvrs_intra['siteID'] = dvrs_intra['siteID'].str.replace('intrabasin', 'transbasin')
        dvrs_intra['siteName'] = dvrs_intra['siteName'] + '_delivery_point'
        dvrs = pd.concat([dvrs, dvrs_intra])
        dvrs = C.df_to_geodataframe(dvrs, lat_col='decLat', lon_col='decLong').to_crs(epsg=C.PROJ_CRS)
        dvrs.to_parquet(C.DIVERSION_MASTER_PARQUET)      # app map layer
    else:
        print(f"note: {C.DIVERSION_MASTER_CSV} not found; using existing {C.DIVERSION_MASTER_PARQUET.name}")
        dvrs = gpd.read_parquet(C.DIVERSION_MASTER_PARQUET)
    return dvrs.to_crs(basins_crs)


def load_irrigation_efficiency(basin_ids):
    """Area-weighted irrigation efficiency per basin; basins without data get the median."""
    irr = pd.read_csv(C.IRRIGATION_PATH, dtype={'basin_id': str}).set_index('basin_id')
    irr['ie'] = ((irr['pct_flood'] / 100) * FLOOD_EFF + (irr['pct_sprinkler'] / 100) * SPRINK_EFF
                 + (irr['pct_drip'] / 100) * DRIP_EFF)
    med_ie = irr['ie'].median()
    return irr['ie'].reindex(basin_ids).fillna(med_ie)


def load_usbr_natural_flows():
    """USBR monthly natural flows (acre-feet/month) as mean cfs, WY1980-2020."""
    brdf = pd.read_csv(C.USBR_NAT_FLOWS_PATH, encoding='utf-8-sig')
    brdf = brdf.rename(columns={col: C.fix_gage_id(col) for col in brdf.columns})
    brdf["date"] = pd.to_datetime(brdf["date"], format="%m/%y")
    days_in_month = brdf['date'].dt.days_in_month
    water_cols = [col for col in brdf.columns if col != "date"]
    # (acre-feet * 43,560 sq ft/acre) / (days in month * 86,400 sec/day)
    brdf[water_cols] = brdf[water_cols].apply(lambda col: (col * 43560.0) / (days_in_month * 86400.0))
    brdf = brdf.drop_duplicates(subset='date', keep='last')
    brdf = brdf[(brdf.date.dt.year > 1979) & (brdf.date.dt.year < 2021)]
    return brdf.set_index('date')


def load_evap(res_name):
    path = C.EVAP_DIR / f"{res_name}.parquet"
    evap = pd.read_parquet(path)
    evap['date'] = pd.to_datetime(evap['date'])
    evap_col = f'evap_{res_name}'
    return evap.set_index('date').rename(columns={'E_volume (m3)': evap_col})[[evap_col]]


def load_inputs():
    import geopandas as gpd
    for p in [C.STREAMFLOW_PATH, C.DIVERSIONS_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run pipeline/update_inputs.py first")

    inp = {}
    reservoir_gdf = gpd.read_parquet(C.RESERVOIRS_PATH).to_crs(epsg=C.PROJ_CRS)
    inp['reservoirs'] = reservoir_gdf[reservoir_gdf.areasqkm > 3]

    basins = gpd.read_parquet(C.BASINS_PATH)
    basins.geometry = basins.geometry.to_crs(C.PROJ_CRS)
    basins['area_m2'] = basins.area
    basins['area_km2'] = basins['area_m2'] / 1E6
    gages = gpd.read_parquet(C.GAGES_PATH)
    inp['name_dict'] = gages['name'].to_dict()

    # streamflow (cms), gap-filled the same way as the per-gage loop
    flow = pd.read_parquet(C.STREAMFLOW_PATH)
    flow.index = pd.to_datetime(flow.index).normalize()
    flow = flow.rename(columns={c: C.fix_gage_id(c) for c in flow.columns})
    inp['flow'] = C.fill_discharge_data(flow, col_names=flow.columns)

    dvrs = load_diversion_sites(basins.crs)
    inp['joined'] = gpd.sjoin(dvrs, basins, how="inner", predicate="within")

    evap = gpd.read_parquet(C.EVAP_RESERVOIRS_PATH).to_crs(epsg=C.PROJ_CRS)
    evap['RES_NAME'] = evap['RES_NAME'].str.replace(' ', '_')
    inp['evap_joined'] = gpd.sjoin(evap, basins, how="inner", predicate="within")

    inp['ie'] = load_irrigation_efficiency(basins.index)

    dm = pd.read_csv(C.DIST_MATRIX_PATH, dtype={'gage': str})
    dm['gage'] = dm['gage'].apply(C.fix_gage_id)
    inp['dm'] = dm.set_index('gage')

    dvrsFlow = pd.read_parquet(C.DIVERSIONS_PATH)
    dvrsFlow.index = pd.to_datetime(dvrsFlow.index)
    # dvrsFlow has no "transbasin" columns since we made them up: duplicate intrabasin ones
    trans = {col.replace('intrabasin', 'transbasin'): dvrsFlow[col]
             for col in dvrsFlow.columns if 'intrabasin' in col}
    inp['dvrsFlow'] = pd.concat([dvrsFlow, pd.DataFrame(trans)], axis=1)

    inp['brdf'] = load_usbr_natural_flows()

    inp['basins'] = basins[~basins.index.isin(EXCLUDED_BASINS)]
    return inp


# --------------------------------------------------------------------------
# Per-gage naturalization
# --------------------------------------------------------------------------
def naturalize_gage(gage, inp, nat_store):
    """Naturalize one gage. `nat_store` holds final Q_NAT2_cfs of basins already done."""
    basins, flow, joined, dvrsFlow = inp['basins'], inp['flow'], inp['joined'], inp['dvrsFlow']
    evap_joined, brdf, dm, name_dict = inp['evap_joined'], inp['brdf'], inp['dm'], inp['name_dict']
    useDict = dict(USE_DICT)
    cu_types = useDict.keys()

    rd = {'gage': gage, 'name': name_dict[gage]}

    df = flow[[gage, f'{gage}_filled']].copy()
    df = df.rename(columns={gage: 'Q_cms', f'{gage}_filled': 'Q_filled_cfs'})
    # make another column that you won't fill
    df['Q_unfilled_cfs'] = np.where(df['Q_filled_cfs'], np.nan, df['Q_cms']) * C.CMS_TO_CFS
    df = C.fill_discharge_data(df, col_names=['Q_cms'])
    df['Q_cfs'] = df['Q_cms'] * C.CMS_TO_CFS

    # irrigation efficiency for this basin
    useDict['irrigation'] = inp['ie'].loc[gage] * -1

    area = basins.loc[gage, 'area_m2']
    df['Q_mmd'] = (df['Q_cms'] * 86400000) / area
    df['area_m2'] = area

    df = df.asfreq('D')
    df['name'] = name_dict[gage]
    df['gage'] = gage
    df = df[df.index.year > 1980].copy()

    # USBR natural flows, if they exist
    if gage in brdf.columns:
        last_valid = brdf[gage].last_valid_index() + pd.offsets.MonthEnd(1)
        df_br = brdf[[gage]].reindex(df.index, method='ffill')
        df_br = df_br.rename(columns={gage: 'Q_NATUSBR_cfs'})
        df_br['Q_NATUSBR_cfs'] = np.where(df_br.index > last_valid, np.nan, df_br['Q_NATUSBR_cfs'])
        df = df.join(df_br)
    else:
        df['Q_NATUSBR_cfs'] = np.nan

    target_diversions = joined[joined['gage'] == gage]['siteID'].unique()
    target_evaps = evap_joined[evap_joined['gage'] == gage]['RES_NAME'].unique()

    # ---- diversions -> consumptive use ----
    if len(target_diversions) > 0:
        # drop diversions with BOTH intrabasin and transbasin points in this basin (they cancel)
        id_groups = defaultdict(list)
        for col in target_diversions:
            id_groups["_".join(col.split("_")[:2])].append(col)
        ids_to_remove = {bid for bid, cols in id_groups.items()
                         if any('intrabasin' in c for c in cols) and any('transbasin' in c for c in cols)}
        valid_cols = [col for col in target_diversions
                      if "_".join(col.split("_")[:2]) not in ids_to_remove]
        missing = [c for c in valid_cols if c not in dvrsFlow.columns]
        if missing:
            print(f"  {gage}: {len(missing)} diversion sites have no records: {missing}")
            valid_cols = [c for c in valid_cols if c in dvrsFlow.columns]

        if valid_cols:
            subset_dvrs = dvrsFlow[valid_cols].copy()
            cu_cols = []
            for col in valid_cols:
                multiplier = 0
                for usage, val in useDict.items():
                    if col.lower().endswith(usage):
                        multiplier = val
                        break
                cu_col_name = f"{col}_CU_cfs"
                subset_dvrs[cu_col_name] = subset_dvrs[col] * multiplier
                cu_cols.append(cu_col_name)

            for cu_type in cu_types:
                matched = [col for col in cu_cols if f'_{cu_type}_CU_cfs' in col]
                subset_dvrs[f'{cu_type}_cfs'] = subset_dvrs[matched].sum(axis=1) if matched else 0.0

            subset_dvrs['Q_CU_cfs'] = subset_dvrs[cu_cols].sum(axis=1)
            df = pd.merge(df, subset_dvrs, right_index=True, left_index=True, how='inner')

            # return flow
            if 'irrigation_cfs' in df.columns:
                df['irrigation_diversion_cfs'] = df['irrigation_cfs'] / useDict['irrigation'] * -1  # CU -> diversion
                df['irrigation_rf_cfs'] = df['irrigation_diversion_cfs'] * (1 - useDict['irrigation'] * -1)  # -> return flow
                kernel = glover_unit_response(np.arange(GLOVER_MAX_LAG_DAYS), GLOVER_A_FT, GLOVER_T, GLOVER_S)
                if np.sum(kernel) > 0:
                    kernel = kernel / np.sum(kernel)
                rf_series = df['irrigation_rf_cfs'].fillna(0).values
                df['lagged_rf_cfs'] = apply_glover_warmup(rf_series, kernel, warmup_years=3, cycle_days=365)

    # ---- reservoir evaporation ----
    if len(target_evaps) > 0:
        dfs = []
        for res in target_evaps:
            try:
                dfs.append(load_evap(res))
            except FileNotFoundError:
                print(f"  {gage}: no evap file for {res}")
        if dfs:
            evap_df = pd.concat(dfs, axis=1)
            evap_df = evap_df * (-35.3146667 / 86400)        # m3/day -> negative cfs
            evap_df['evap_cfs'] = evap_df.sum(axis=1)
            df = pd.merge(left=df, right=evap_df, how='left', left_index=True, right_index=True)

    # ---- Q + CU ----
    cols = df.columns
    if 'evap_cfs' in cols and 'Q_CU_cfs' in cols:
        df['Q_NAT_cfs'] = df['Q_cfs'] - df['evap_cfs'] - df['Q_CU_cfs']
    elif 'evap_cfs' in cols:
        df['Q_NAT_cfs'] = df['Q_cfs'] - df['evap_cfs']
    elif 'Q_CU_cfs' in cols:
        df['Q_NAT_cfs'] = df['Q_cfs'] - df['Q_CU_cfs']
    else:
        df['Q_NAT_cfs'] = df['Q_cfs']

    # lagged return flows
    if 'irrigation_cfs' in cols:
        df['Q_NAT_noag'] = df['Q_NAT_cfs'] + df['irrigation_cfs']
        df['Q_NAT2_cfs'] = df['Q_NAT_noag'] - df['irrigation_diversion_cfs'] + df['lagged_rf_cfs']
        df['Q_NAT2_cfs'] = np.where(df['Q_NAT2_cfs'] < 0, 0, df['Q_NAT2_cfs'])
    else:
        df['Q_NAT2_cfs'] = df['Q_NAT_cfs']

    # ---- basins below major reservoirs ----
    import geopandas as gpd
    basin = basins[basins.index == gage]
    basin_res = gpd.sjoin(basin, inp['reservoirs'], how='inner', predicate='contains')
    days = df['Q_cfs'].notna().sum()  # need 10 years of data
    name = name_dict[gage].lower()
    if (len(basin_res) > 0 and days > 3650) or ('below' in name and 'reservoir' in name):
        reservoir_naturalize(gage, df, rd, basin, area, inp, nat_store)

    # ---- comparison with USBR ----
    monthly_df = df[['Q_NAT2_cfs', 'Q_NATUSBR_cfs']].dropna().resample('MS').mean()
    obs, sim = monthly_df['Q_NATUSBR_cfs'], monthly_df['Q_NAT2_cfs']
    with np.errstate(divide='ignore', invalid='ignore'):
        rd['nse'] = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
        rd['pbias'] = 100 * (np.sum(sim - obs) / np.sum(obs))
    rd['obs/nat'] = mass_balance_compare(df['Q_cfs'], df['Q_NAT_cfs'])
    rd['obs/nat2'] = mass_balance_compare(df['Q_cfs'], df['Q_NAT2_cfs'])
    rd['nat2/natusbr'] = mass_balance_compare(df['Q_cfs'], df['Q_NATUSBR_cfs'])
    rd['n_diversions'] = len(target_diversions)

    return df.asfreq('D'), rd


def reservoir_naturalize(gage, df, rd, basin, area, inp, nat_store):
    """Rebuild natural flow below reservoirs from naturalized inflow gages (in place on df)."""
    basins, dm = inp['basins'], inp['dm']
    print(f'  {gage}: processing inflows')
    rd['below_res'] = True

    # largest non-overlapping sub-basins within this basin
    target_geom = basin.geometry.iloc[0]
    contained = basins[(basins.index != gage) & (basins.centroid.within(target_geom))].copy()
    contained = contained[contained['area_m2'] < area]  # centroid can put big basins inside small ones
    contained = contained[contained.index.isin(list(nat_store))]

    if contained.empty:
        print(f'  {gage}: contained basins empty, cant naturalize')
        rd['contained_basins'] = False
        df['Q_NAT2_cfs'] = np.nan
        return

    rd['contained_basins'] = True
    contained = contained.sort_values(by='area_m2', ascending=False)
    contained_gages = contained.index.unique().to_list()
    # Q_NAT2 of contained basins (already processed, smallest first) + this gage's Q_NAT
    flow_subset = pd.concat([nat_store[c].rename(c) for c in contained_gages]
                            + [df['Q_NAT_cfs'].rename(gage)], axis=1, join='outer')

    # for each year, find the inflow gages with good data and use them to naturalize flow
    years = df['Q_cfs'].dropna().index.year.unique()
    all_inflows = []
    inflow_gages, frac_covered = [], 0.0
    for year in years:
        startwy = pd.Timestamp(year=year - 1, month=10, day=1)
        endwy = pd.Timestamp(year=year, month=9, day=30)
        ydf = flow_subset[contained_gages].loc[startwy:endwy].copy()

        use = []
        for cgage in ydf.columns:
            if ydf[cgage].isna().sum() < 37:  # 37 days, 10% of the year
                use.append(cgage)
                ydf[cgage] = ydf[cgage].interpolate(method='linear')

        if len(use) == 0:
            print(f'  no inflow data, cant reservoir naturalize for {year, gage}')
            df.loc[startwy:endwy, 'Q_NAT2_cfs'] = np.nan
            rd[f'{year}_inflows'] = False
            continue

        ydf = ydf[use]
        yb = contained[contained.index.isin(use)]

        # iteratively select non-overlapping sub-basins, biggest first
        inflow_gages = []
        union_geom = None
        for idx, subbasin in yb.iterrows():
            geom = subbasin.geometry
            if union_geom is None:
                inflow_gages.append(idx)
                union_geom = geom
            elif geom.intersection(union_geom).area / geom.area < 0.01:  # edge effects only
                inflow_gages.append(idx)
                union_geom = union_geom.union(geom)

        frac_covered = union_geom.area / basin.geometry.iloc[0].area
        if frac_covered < 0.33:  # not enough inflow gages, can't naturalize at this scale
            print(f'  frac covered: {frac_covered:.2f}, cant reservoir naturalize for {gage}')
            df.loc[startwy:endwy, 'Q_NAT2_cfs'] = np.nan
            rd[f'{year}_inflows'] = False
            continue

        rd[f'{year}_inflows'] = inflow_gages
        rd['inflows'] = inflow_gages
        all_inflows.append(ydf[inflow_gages].copy())

        # total natural inflow vs outflow -> share not accounted for by ungaged inlets
        overlap = flow_subset[inflow_gages + [gage]].dropna()
        gaged_total = overlap[inflow_gages].sum(axis=1).sum()
        frac_in = gaged_total / overlap[gage].sum()
        rd[f'{year}_frac_inflow'] = frac_in
        rd[f'{year}_frac_gaged_inflow_area'] = frac_covered

        # leave-one-out test of the ungaged-inflow scaling. The stored value is
        # frac_in_test / frac_in = share of gaged inflow from the other gages. It is computed
        # on the same dates as frac_in; giving the reduced set its own (often longer) overlap
        # let the outlet sums stop cancelling and biased the test.
        if len(inflow_gages) > 1 and frac_covered > 0.6:
            for igage in inflow_gages:
                inflow_copy = [g for g in inflow_gages if g != igage]
                rd[f'{year}_{igage}_frac_inflow'] = overlap[inflow_copy].sum(axis=1).sum() / gaged_total

    if len(all_inflows) == 0:
        return

    ai_df = pd.concat(all_inflows)
    raw_inflows = list(ai_df.columns)
    for igage in raw_inflows:
        ai_df[f'{igage}_routed'] = route_streamflow_muskingum_cfs(ai_df[igage], igage, gage, dm)
        df[f'test_{igage}'] = np.nan
        df[f'igage_{igage}'] = np.nan

    name_dict = inp['name_dict']
    yearly_rows = []
    df['Q_NAT3_cfs'] = np.nan
    for year in years:
        if not rd.get(f'{year}_inflows'):
            continue
        startwy = pd.Timestamp(year=year - 1, month=10, day=1)
        endwy = pd.Timestamp(year=year, month=9, day=30)
        ydf = ai_df.loc[startwy:endwy].copy()
        if ydf.empty:  # e.g. the water year before the record starts
            continue
        frac_in = rd[f'{year}_frac_inflow']
        y_inflows = rd[f'{year}_inflows']
        # sum within the year (summing ai_df and assigning would give an empty ydf the full index)
        ydf['inflow_sum_routed'] = ydf[[f'{g}_routed' for g in y_inflows]].sum(axis=1, skipna=False)
        ydf['inflow_sum'] = ydf[y_inflows].sum(axis=1, skipna=False)

        if frac_in > 0.5:
            # < 1: assume the unaccounted area behaves like the gaged inflows
            # > 1: the reservoir may be releasing stored flows
            df.loc[startwy:endwy, 'Q_NAT2_cfs'] = ydf['inflow_sum'] / frac_in
            df.loc[startwy:endwy, 'Q_NAT3_cfs'] = ydf['inflow_sum_routed'] / frac_in
        else:  # frac in <= 0.5
            df.loc[startwy:endwy, 'Q_NAT2_cfs'] = np.nan
            df.loc[startwy:endwy, 'Q_NAT3_cfs'] = np.nan
            continue  # scaling not applied this year, so nothing to validate

        # validation: hold out each inflow gage and predict it from the others. The keys
        # below only exist for years with >1 inflow gage and frac_covered > 0.6 (this
        # year's values, checked in the loop above).
        has_flow = ydf['inflow_sum'].notna()
        nat_total = ydf.loc[has_flow, 'inflow_sum'].sum() / frac_in  # this year's natural outflow
        for igage in y_inflows:
            key = f'{year}_{igage}_frac_inflow'
            if key not in rd:
                continue
            # skip gages that are a small piece of the basin (e.g. Escalante at Lees Ferry)
            share = ydf.loc[has_flow, igage].sum() / nat_total if nat_total > 0 else np.nan
            rd[f'{year}_{igage}_share'] = share
            if not share >= MIN_VALIDATION_SHARE:
                continue
            inflow_copy = [g for g in y_inflows if g != igage]
            others = ai_df.loc[startwy:endwy, inflow_copy].sum(axis=1, skipna=False)
            # bump flows up by frac in, subtract known inflows -> missing inflow
            test = others / rd[key] - others
            df.loc[startwy:endwy, f'test_{igage}'] = test
            df.loc[startwy:endwy, f'igage_{igage}'] = ydf[igage]
            yearly_rows.append({'gage': gage, 'name': rd['name'], 'inflow_gage': igage,
                                'inflow_name': name_dict.get(igage), 'water_year': year,
                                'share': share, 'frac_covered': rd[f'{year}_frac_gaged_inflow_area'],
                                **fit_stats(ydf[igage], test)})

    # how well does the ungaged-inflow scaling do? every held-out gage, all evaluated years
    summary_rows = []
    for igage in raw_inflows:
        stats = fit_stats(df[f'igage_{igage}'], df[f'test_{igage}'])
        if stats['n_days'] == 0:
            continue
        yrs = [r for r in yearly_rows if r['inflow_gage'] == igage]
        rd[f'{igage}_nse'], rd[f'{igage}_pbias'], rd[f'{igage}_r'] = stats['nse'], stats['pbias'], stats['r']
        summary_rows.append({'gage': gage, 'name': rd['name'], 'inflow_gage': igage,
                             'inflow_name': name_dict.get(igage), 'n_years': len(yrs),
                             'mean_share': np.mean([r['share'] for r in yrs]), **stats})
        print(f"  {gage}, {igage}, nse {stats['nse']:.2f}, pbias {stats['pbias']:.1f}, r {stats['r']:.2f}")
    rd['_validation'] = (summary_rows, yearly_rows)


VAL_SUMMARY_COLS = ['gage', 'name', 'inflow_gage', 'inflow_name', 'n_years', 'mean_share',
                    'n_days', 'nse', 'pbias', 'r']
VAL_YEARLY_COLS = ['gage', 'name', 'inflow_gage', 'inflow_name', 'water_year', 'share',
                   'frac_covered', 'n_days', 'nse', 'pbias', 'r']


def fit_stats(obs, sim):
    """NSE, PBIAS (%) and r of sim vs obs, on the days both have data."""
    m = pd.concat([obs.rename('obs'), sim.rename('sim')], axis=1).dropna()
    if m.empty:
        return {'n_days': 0, 'nse': np.nan, 'pbias': np.nan, 'r': np.nan}
    o, s = m['obs'], m['sim']
    with np.errstate(divide='ignore', invalid='ignore'):
        nse = 1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2)
        pbias = 100 * np.sum(s - o) / np.sum(o)
    return {'n_days': len(m), 'nse': nse, 'pbias': pbias, 'r': o.corr(s)}


# --------------------------------------------------------------------------
def app_columns(df):
    """Columns the app needs: anything with 'Q' plus consumptive use by type."""
    cu_cols = [f'{t}_cfs' for t in list(USE_DICT) + ['evap']]
    return [col for col in df.columns if 'Q' in col or col in cu_cols]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gages", nargs="+", help="only write these gages (all basins are still "
                   "processed up to the largest requested, so reservoir inflows are available)")
    args = p.parse_args()

    C.ensure_dirs()
    inp = load_inputs()
    basins = inp['basins'].sort_values(by='area_km2')
    order = list(basins.index)
    if args.gages:
        wanted = {C.fix_gage_id(g) for g in args.gages}
        unknown = wanted - set(order)
        if unknown:
            raise SystemExit(f"unknown gages: {sorted(unknown)}")
        last = max(order.index(g) for g in wanted)
        order = order[:last + 1]
    else:
        wanted = set(order)

    nat_store, results, skipped = {}, [], []
    val_summary, val_yearly = [], []
    for gage in order:
        if gage not in inp['flow'].columns:
            skipped.append(gage)
            continue
        df, rd = naturalize_gage(gage, inp, nat_store)
        nat_store[gage] = df['Q_NAT2_cfs']
        summary_rows, yearly_rows = rd.pop('_validation', ([], []))
        if gage in wanted:
            df.reset_index(names='date').to_parquet(C.NATURALIZED_FULL_DIR / f"{gage}.parquet", index=False)
            small = df[app_columns(df)].reset_index(names='date')
            small.to_parquet(C.TIMESERIES_DIR / f"{gage}.parquet", index=False)
            results.append(rd)
            val_summary += summary_rows
            val_yearly += yearly_rows
            print(f"processed {gage} ({rd['name']}): {rd['n_diversions']} diversions")

    if skipped:
        print(f"no streamflow for {len(skipped)} basins, skipped: {skipped}")
    pd.DataFrame(results).to_csv(C.NAT_RESULTS_PATH, index=False)
    pd.DataFrame(val_summary, columns=VAL_SUMMARY_COLS).to_csv(C.INFLOW_VALIDATION_PATH, index=False)
    pd.DataFrame(val_yearly, columns=VAL_YEARLY_COLS).to_csv(C.INFLOW_VALIDATION_YEARLY_PATH, index=False)
    print(f"inflow validation: {len(val_summary)} inflow gages in {C.INFLOW_VALIDATION_PATH}")
    print(f"wrote {len(results)} gages to {C.TIMESERIES_DIR}; diagnostics in {C.NAT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
