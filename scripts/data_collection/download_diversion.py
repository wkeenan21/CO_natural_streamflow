import pandas as pd
from dataretrieval import nwis
import geopandas as gpd
import os
import cdsspy
import numpy as np
from os.path import join
import sys
from shapely.geometry import Polygon, MultiPolygon
# define working directory
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts/data_collection'))
from special_functions import *

token = '7V+/s+lkHFgIgpIumq0UrRIIOtpwoVyE'

# get watershed shapefile
wsheds = gpd.read_file(join(cwd, r'data\shapefiles\wsheds_co_camels_flow25_3.shp'))
wsheds = wsheds.to_crs("EPSG:4326")

wsheds1 = wsheds[wsheds['gauge_id'] == '06752260']

# directory for streamflow and climate timeseries
csv_dir = join(cwd, r'data\NH_data\filled')

for wshed in wsheds1.iterrows():
    
    gage = wshed[1]['gauge_id']

    flow = pd.read_csv(join(csv_dir, fr'{gage}.csv'), index_col='date', parse_dates=True)

    structures = cdsspy.get_structures(
        aoi = wshed[1]['geometry'],)

    structures = structures[(structures["ciuCode"] == "A")]
    structures = structures.sort_values(by='waterSource')
    
    wdids = list(structures['wdid'])
    names = list(structures['structureName'])

    for wdid, name in zip(wdids,names):
        print(name)
        try:
            ts = cdsspy.get_structures_divrec_ts(wdid=wdid, start_date='2000-01-01', end_date= '2025-01-01', timescale='day', api_key=token)
            ts = ts[['dataValue', 'dataMeasDate', 'wdid', 'measUnits']]
            unit = ts['measUnits'].dropna().iloc[0]

            flow_col_name = f'{name}_{wdid}_{unit}'

            ts.rename(columns={'dataValue':flow_col_name}, inplace=True)
            
            ts.index = pd.to_datetime(ts['dataMeasDate'], utc=False)
            ts = fill_missing_days(ts)

            flow[flow_col_name] = ts[flow_col_name]
        except Exception as e:
            print(e)
