import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run
from neuralhydrology.nh_run_scheduler import schedule_runs
import pyarrow
import yaml
import os
import pandas as pd
import numpy as np

cwd = r'G:\My Drive\natural_streamflow_colab'
runs = fr'{cwd}\run_dir'

#####
# open anaconda prompt
# cd G:\My Drive\natural_streamflow_colab\neuralhydrology

for folder in os.listdir(runs):
  if 'tuning' in folder:
    run_dir = Path(fr"{runs}/{folder}")
    eval_run(run_dir=run_dir, period="validation", epoch=10)

period = "test" # test train or validation
eval_run(run_dir=run_dir, period=period)

# examine a bunch to assess hyperparameters
resultslist = []
epochs = [10, 20, 30]
for epoch in epochs:
    epoch = str(epoch).zfill(3)

    period = "validation"
    models = {}
    for folder in os.listdir(runs):
        if 'tuning' in folder:
            run_dir = Path(fr"{runs}/{folder}")

            with open(run_dir / period / f"model_epoch{epoch}" / f"{period}_results.p", "rb") as fp:
                results = pickle.load(fp)

            # read config data
            config = yaml.safe_load(open(fr'{run_dir}\config.yml', 'r'))
            config['results'] = results
            config['epoch'] = epoch
            config['model'] = f'{folder}_{epoch}'
            config['folder'] = folder

            # evaluate model performanc3
            nses = []
            for gage in results.keys():
                nse = results[gage]['1D']['NSE']
                nses.append(nse)
            
            medianNSE = np.median(nses)
            meanNSE = np.mean(nses)
            config['medianNSE'] = medianNSE
            config['meanNSE'] = meanNSE
            resultslist.append(config)

rdf = pd.DataFrame().from_dict(resultslist)

# filter to 30 epochs
rdf = rdf[rdf.epoch=='030']
# view results
fig, ax = plt.subplots()
x = 'output_dropout'
y = 'medianNSE'
ax.scatter(rdf[x], rdf[y])
ax.set_xlabel(x)
ax.set_ylabel(y)
plt.show()

# see if overtraining
fig, ax = plt.subplots()

x = 'epoch'
y = 'medianNSE'

# Map each unique hidden_size to a unique color automatically
unique_sizes = rdf['hidden_size'].unique()
colors = plt.cm.tab10.colors  # Color palette
color_map = {size: colors[i % len(colors)] for i, size in enumerate(unique_sizes)}

for model in rdf.folder.unique():
    mdf = rdf[rdf.folder == model]
    
    # Extract the hidden_size for this specific model
    h_size = mdf['hidden_size'].iloc[0]
    
    ax.plot(
        mdf[x], 
        mdf[y], 
        label=f"{model} (h={h_size})", 
        color=color_map[h_size], 
        marker='o'
    )

ax.set_xlabel(x)
ax.set_ylabel(y)
plt.show()
