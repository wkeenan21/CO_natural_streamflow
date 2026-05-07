import pandas as pd
from dataretrieval import nwis
import geopandas as gpd
import os
import cdsspy
import numpy as np
from os.path import join
import sys
from shapely.geometry import Polygon, MultiPolygon

# pull data for this time period
startDate = '1907-01-01'
endDate = '2026-03-26'
parameterCode = '00060' # daily discharge

gage = '06724000'

flow = nwis.get_dv(sites=gage, parameterCd=parameterCode, start=startDate, end=endDate)[0]
flow = flow.rename(columns={'00060_Mean':'Q_cfs', '00060_Mean_cd':'quality_code'})
flow.to_csv(fr'N:\Research\Kampf\Private\KeenanW\flow25_gages_burned\saint_vrain.csv')