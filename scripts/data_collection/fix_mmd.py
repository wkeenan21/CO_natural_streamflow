import pandas as pd
import geopandas as gpd
import os
import sys
import time

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'scripts/data_collection'))

with open(os.path.join(cwd, r'scripts/configs/flow25_gages.txt'), "r") as f:
    flow25gages = f.read().splitlines()

wsheds = gpd.read_file(os.path.join(cwd, r'data\shapefiles\wsheds_co_camels_flow25_3.shp'))
wsheds = wsheds.to_crs("EPSG:5070")

gage_dir = os.path.join(cwd, r'data/NH_data/filled')

for csv in os.listdir(gage_dir):
    file_path = os.path.join(gage_dir, csv)
    if csv != 'basinCharacteristics.csv':
        now = time.time()
        last_modified = os.path.getmtime(file_path)


        df = pd.read_csv(file_path)
        gage = csv[:-4]

        # rename datetime to date
        df.rename(columns={'datetime':'date', 'Unnamed: 0':'date'}, inplace=True)

        df['date'] = pd.to_datetime(df['date'])

        area = wsheds[wsheds['gauge_id'] == gage]['area'].iloc[0]
        df['Q_mmd'] = df['Q_cfs'] * (0.0283168 * 86400) / (area * 1000)

                # gap fill
        vars = ['Q_mmd', 'Q_cfs', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp', 'eto', 'vpd']
        for var in vars:
            df[var] = df[var].interpolate(limit=90)

    #     df.to_csv(os.path.join(gage_dir, csv), index=False)

    if gage in flow25gages:
        df.to_csv(os.path.join(cwd, 'data/NH_data/flow25', csv))
        
print('done')
