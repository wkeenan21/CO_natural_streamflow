"""
Decide what watersheds you already have from caravan, and what you need to get
"""

import pandas as pd
from dataretrieval import nwis
import geopandas as gpd
import os
import cdsspy
import numpy as np
from os.path import join

from jinja2.utils import missing

cwd = os.getcwd()

# this is all the active watersheds
allCO_gages = gpd.read_file(join(cwd, r'data\shapefiles\watersheds_CO_active.shp'))
flow25 = gpd.read_file(join(cwd, r'data\CSU_Flow25\site_coordinates_20250731.shp'))




