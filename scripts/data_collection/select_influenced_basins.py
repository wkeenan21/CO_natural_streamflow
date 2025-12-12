import pandas as pd
import os
import numpy as np
import geopandas as gpd
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

cwd = os.getcwd()

sys.path.append(os.path.join(cwd, 'scripts/data_collection'))
from special_functions import *

flow25 = gpd.read_file(os.path.join(cwd, 'data/CSU_Flow25/watersheds_shapefile_20250624.shp'))
df = gpd.read_file(os.path.join(cwd, 'data/shapefiles/wsheds_co_camels_flow25_3.shp'))
df['gage'] = fix_gage_series(df['gauge_id'])
filled_gage_dir = os.path.join(cwd, r'data/NH_data/filled')
camels = gpd.read_file(r'C:\Users\C830645719\Downloads\Caravan\Caravan\shapefiles\camels\camels_basin_shapes.shp').to_crs(3857)
camels['gauge_id'] = camels['gauge_id'].str[7:]
flow25['gage'] = fix_gage_series(flow25['gage_used'])

# make some plots of the average stats for flow25 (training) basins
results = []

for gage in df['gauge_id']:
    try:
        resultsDict = {'gage':gage}

        # check which dataset it's in
        if gage in list(flow25['gage']):
            resultsDict['dataset'] = 'natural'
        elif gage in list(camels['gauge_id']):
            resultsDict['dataset'] = 'camels'
        else:
            resultsDict['dataset'] = 'influenced'

        # check your area
        resultsDict['area'] = df[df['gage']==gage].iloc[0]['area']

        # load the timeseries
        gage_df = pd.read_csv(os.path.join(filled_gage_dir, f'{gage}.csv'))

        vars = ['Q_mmd', 'Q_cfs', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp', 'eto', 'vpd']
        for var in vars:
            resultsDict[f'mean_{var}'] = gage_df[var].mean()
            resultsDict[f'median_{var}'] = gage_df[var].median()
            resultsDict[f'frac_missing_{var}'] = len(gage_df[var].dropna()) / len(gage_df[var])

        results.append(resultsDict)
    except:
        print('no', gage)

meanDf = pd.DataFrame().from_dict(results)


# compare to some of the hydroatlas stats
ha = pd.read_csv(os.path.join(cwd, 'data/NH_data/filled/basinCharacteristics.csv'))
ha['gage'] = fix_gage_series(ha['gage'])

for dataset in ['natural', 'influenced']:
    meanDf_sub = meanDf[meanDf['dataset']==dataset]

    #print(dataset, meanDf_sub['median_Q_cfs'].median(), meanDf_sub['area'].median())
    # print(dataset)
    # print(meanDf_sub['median_Q_cfs'].quantile(.9))
    # print(meanDf_sub['area'].quantile(.9))

    ha_sub = ha[ha['gage'].isin(meanDf_sub['gage'])]
    stop

    print(ha_sub['dor_pc_pva'].mean())

    # meanDf_sub['area'].hist()
    # plt.show()

ha_sub_dor = ha_sub[['dor_pc_pva', 'gage']]
plt.show()
# < 43 median CFS and < 654 area
testBasins = meanDf[(meanDf['area'] < 654) & (meanDf['dataset']=='influenced') & (meanDf['median_Q_cfs'] < 43)]
testBasins['median_Q_cfs'].hist()
plt.show()

testBasinsgdf = df[df['gage'].isin(testBasins['gage'])]
testBasinsgdf.to_file(os.path.join(cwd, 'data/shapefiles/testBasins1.shp'))

write_lines_to_file(list(testBasins['gage']), os.path.join(cwd, r'scripts/configs/testbasins1.txt'))
write_lines_to_file(list(camels))


