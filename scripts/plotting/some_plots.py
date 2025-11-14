import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()

dfPath = os.path.join(cwd, r"data\NH_data\basinCharacteristics.csv")

df = pd.read_csv(dfPath)

df['area'].hist(bins=40)
plt.show()
