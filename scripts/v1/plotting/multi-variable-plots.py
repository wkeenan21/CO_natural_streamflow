import pandas as pd
import os
from os.path import join
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
cwd = os.getcwd()

df = pd.read_csv(r"C:\Users\C830645719\Downloads\Caravan\Caravan\timeseries\csv\camels\camels_09081600.csv")
df.index = pd.to_datetime(df['date'])

import pandas as pd
import matplotlib.pyplot as plt

# --- water year slice (Oct 2018 – Sep 2019) ---
wy_start = "2018-10-01"
wy_end   = "2019-09-30"
df_wy = df.loc[wy_start:wy_end].copy()

# variables of interest
colsDict = {
    "snow_depth_water_equivalent_mean":'SWE',
    "total_precipitation_sum":'Precip',
    "potential_evaporation_sum":'PET',
    "surface_net_solar_radiation_mean":'Solar Rad',
    "temperature_2m_mean": 'Temp',
    "volumetric_soil_water_layer_1_mean":'Soil Mois'
}
cols = list(colsDict.values())

df_wy = df_wy.rename(columns=colsDict)

# normalize each column to [0,1] manually
df_scaled = (df_wy[cols] - df_wy[cols].min()) / (df_wy[cols].max() - df_wy[cols].min())

# --- Panel plot ---
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(len(cols), 1, figsize=(12, 10), sharex=True)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default color cycle

for i, col in enumerate(cols):
    axes[i].plot(df_scaled.index, df_scaled[col], color=colors[i % len(colors)], lw=1.5)
    axes[i].set_ylabel(col, fontsize=14)
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel("Date")
#fig.suptitle("", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()