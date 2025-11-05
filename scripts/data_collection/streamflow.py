"""
Grab streamflow from USGS API and CDWR API
"""
import pandas as pd
from dataretrieval import nwis
import geopandas as gpd
import os
import cdsspy
import numpy as np
from os.path import join
import sys
from shapely.geometry import Polygon, MultiPolygon

def remove_holes(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    elif geom.geom_type == "MultiPolygon":
        # Apply recursively to all parts
        parts = [Polygon(p.exterior) for p in geom.geoms]
        return max(parts, key=lambda a: a.area)  # keep largest
    else:
        return geom

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts/data_collection'))
from special_functions import *

# define working directory
cwd = os.getcwd()
wsheds = gpd.read_file(join(cwd, r'data\shapefiles\wsheds_co_camels_flow25_3.shp'))
wsheds = wsheds.to_crs("EPSG:5070")

wsheds['geometry'] = wsheds.geometry.apply(to_single)
wsheds['geometry'] = wsheds.geometry.apply(remove_holes)

# DAYMET
dm = pd.read_csv(r"C:\Users\C830645719\OneDrive - Colostate\documents\GitHub\CO_natural_streamflow\data\timeseries\raw\daymet2000-2024.csv")
gm = pd.read_csv(r"C:\Users\C830645719\OneDrive - Colostate\documents\GitHub\CO_natural_streamflow\data\timeseries\raw\gridmet2000-2024.csv")
dm.index = pd.to_datetime(dm['date'])
gm.index = pd.to_datetime(gm['date'])

# pull data for this time period
startDate = '2000-01-01'
endDate = '2025-01-01'
parameterCode = '00060' # daily discharge

for gage, area in zip(wsheds['gauge_id'], wsheds['area']):
    # Daily discharge at "ANDDITCO" telemetry station
    camelsPath = join(cwd, fr'data\timeseries\camels\{gage}.csv')
    outCsv = join(cwd, fr'data\NH_data\unfilled\{gage}.csv')
    outCsv2 = join(cwd, fr'data\NH_data\filled\{gage}.csv')

    if not os.path.exists(outCsv):
        # if it's a camels gage, get the flow from there
        if os.path.exists(camelsPath):
            flow = pd.read_csv(camelsPath)
            flow.index = pd.to_datetime(flow['date'])
            flow = flow.rename({'streamflow':'Q_cfs'})
            flow.drop(columns=['date'], inplace=True)
            flow['Q_mmd'] = flow['Q_cfs'] * (0.0283168 * 86400) / (area * 1000000) * 1000
            flow['gage'] = gage
        else:
            if gage.isalpha():
                print(f'getting {gage}')
                #try:
                flow = cdsspy.get_telemetry_ts(
                    abbrev=gage,  # Site abbreviation from the outputs of get_telemetry_stations()
                    parameter="DISCHRG",  # Desired parameter, identified by the get_reference_tbl()
                    start_date=startDate,  # Starting date
                    end_date=endDate,  # Ending date
                    timescale="day"  # select daily timescale
                )
                flow['gage'] = gage
                flow.drop(columns=['abbrev', 'parameter', 'measUnit', 'modified'], inplace=True)
                flow.rename(columns={'measValue': 'Q_cfs', 'measDate': 'date'}, inplace=True)
                flow.index = pd.to_datetime(flow['date'])
                flow.index = flow.index.tz_localize(None)
                flow.drop(columns='date', inplace=True)
                flow = fill_missing_days(flow)
                flow.index.freq = 'D'  # make the frequency daily (weird pandas thing)

                flow['Q_mmd'] = (flow['Q_cfs'] * (0.0283168 * 86400) / (area * 1000000)) * 1000
                # except Exception as e:
                #     print(f'{gage} failed: {e}')
                #     continue
                # get some info about the station
                # info = cdsspy.get_telemetry_stations(abbrev=gage)

                # # flow['lat'] = info['latitude'].iloc[0]
                # # flow['lon'] = info['longitude'].iloc[0]
                # # flow['name'] = info['stationName'].iloc[0]
                # flow['gage'] = gage
                # flow.drop(columns=['abbrev', 'parameter', 'measUnit', 'modified'], inplace=True)
                # flow.rename(columns={'measValue': 'Q_cfs', 'measDate': 'date'}, inplace=True)
                # flow.index = pd.to_date(flow['date']).tz_localize(None)
                # flow.drop(columns='date', inplace=True)
                # flow = fill_missing_days(flow)
                # flow.index.freq = 'D'  # make the frequency daily (weird pandas thing)

            else:
                #try:
                print(f'getting {gage}')
                flow = nwis.get_dv(sites=gage, parameterCd=parameterCode, start=startDate, end=endDate)[0]  # this takes awhile
                if len(flow) < 10:
                    print(f'no data for {gage}')
                    continue
                info = nwis.get_info(sites=gage)[0]

                flow['date'] = flow.index
                flow['gage'] = gage
                flow['lat'] = info['dec_lat_va'].iloc[0]
                flow['lon'] = info['dec_long_va'].iloc[0]
                flow['name'] = info['station_nm'].iloc[0]
                flow.rename(columns={'00060_Mean': 'Q_cfs', '00060_Mean_cd': 'qc_code', '00060_loc 1_Mean':'Q_cfs'}, inplace=True)

                flow = fill_missing_days(flow)  # sometimes whole days are just missing
                flow.drop(columns='date', inplace=True)

                flow.index.freq = 'D'  # make the frequency daily (weird pandas thing)

                #flow['Q_mmd'] = flow['Q_cfs'] * ((0.0283168 * 86400) / (area * 1000000)) * 1000
                flow['Q_mmd'] = flow['Q_cfs'] * (2446.58 / area)
                # except Exception as e:
                #     print(f'{gage} failed: {e}')
                #     continue

            # add daymet to the flow df
            dm_sub = dm[dm['gauge_id'] == gage].copy()
            dm_sub = dm_sub.loc[startDate:endDate]

            # add gridmet to the flow df
            gm_sub = gm[gm['gauge_id'] == gage].copy()
            gm_sub = gm_sub.loc[startDate:endDate]

            # drop timezone, only have these columns
            flowCols = ['Q_mmd', 'Q_cfs', 'gage']
            flow.index = flow.index.tz_localize(None)
            flow = flow[flowCols].copy()

            vars = ['prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
            for var in vars:
                flow[var] = dm_sub[var].reindex(flow.index)

            # flow.index = pd.to_datetime(flow.index).tz_localize(None).normalize()
            # gm_sub.index = pd.to_datetime(gm_sub.index).tz_localize(None).normalize()

            vars = ['eto', 'vpd']
            for var in vars:
                flow[var] = gm_sub[var].reindex(flow.index)

            static_vars = ['gage']
            for var in static_vars:
                flow[var] = flow[var].iloc[0]

            flow.index.freq = 'D'
            flow.to_csv(outCsv)

            # gap fill
            vars = ['Q_mmd', 'Q_cfs']
            total = len(flow['Q_cfs'])
            valid = len(flow['Q_cfs'].dropna())
            for var in vars:
                flow[var] = flow[var].interpolate(limit=90)

            afterFill = len(flow['Q_cfs'].dropna())

            print(f'{gage} total: {total}, beforeFill: {valid}, afterFill: {afterFill}, %: {afterFill/total:.2f}')

            flow.to_csv(outCsv2)
