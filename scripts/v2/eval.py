import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
import torch
import pyarrow
import yaml
import os
import pandas as pd
import numpy as np
import seaborn as sns

cwd = r'G:\My Drive\natural_streamflow_colab'
runs = fr'{cwd}\run_dir'


# examine a bunch to assess hyperparameters
resultslist = []
epochs = [10, 20, 30, 40, 50]
for epoch in epochs:
    epoch = str(epoch).zfill(3)

    period = "validation"
    models = {}
    for folder in os.listdir(runs):
        if 'tuning' in folder:
            run_dir = Path(fr"{runs}/{folder}")

            try:
                with open(run_dir / period / f"model_epoch{epoch}" / f"{period}_results.p", "rb") as fp:
                    results = pickle.load(fp)
            except:
                continue

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
            config['NSEs'] = nses
            resultslist.append(config)

rdf = pd.DataFrame().from_dict(resultslist)

# filter to 30 epochs
rdf2 = rdf[rdf.epoch=='030']
# view results
fig, ax = plt.subplots()
x = 'output_dropout'
y = 'medianNSE'
ax.scatter(rdf2[x], rdf2[y])
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


rdf3 = rdf[['medianNSE', 'meanNSE', 'hidden_size', 'batch_size', 'output_dropout', 'epoch', 'NSEs']].sort_values(by='medianNSE', ascending=False)
fig, ax = plt.subplots()
ax.scatter(rdf3.epoch, rdf3.medianNSE)
ax.set_xlabel('epoch')
ax.set_ylabel('NSE')
plt.show()

# examine duplicates to assess reproducability

df = rdf3.copy()
# 1. Create a composite hyperparameter label for grouping
df['config'] = (
    'H:' + df['hidden_size'].astype(str) + 
    ' | B:' + df['batch_size'].astype(str) + 
    ' | D:' + df['output_dropout'].astype(str) + 
    ' | E:' + df['epoch'].astype(str)
)

# Optional: Sort configurations by median performance for clarity
order = df.groupby('config')['medianNSE'].median().sort_values(ascending=False).index

# 2. Plot the stochastic variability
plt.figure(figsize=(12, 6))

# Boxplot shows IQR, median, and overall spread per group
sns.boxplot(
    data=df, 
    x='config', 
    y='medianNSE', 
    order=order, 
    color='skyblue', 
    showfliers=False
)

# Stripplot overlays individual model seeds/runs
sns.stripplot(
    data=df, 
    x='config', 
    y='medianNSE', 
    order=order, 
    color='black', 
    alpha=0.7, 
    jitter=0.2, 
    size=6
)

plt.xticks(rotation=45, ha='right')
plt.title('Stochastic Variability of medianNSE Across Hyperparameter Sets')
plt.xlabel('Hyperparameter Configuration (Hidden | Batch | Dropout | Epoch)')
plt.ylabel('medianNSE')
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()
plt.show()

######################
# Evaluate the experiment
######################
five_unseen = ['09081600', '09266500', '09253000', '09312600', '09223000']
five_seen = ['09217900', '09123450', '09210500', '383926107593001', '09310700']

resultslist = []
epoch = '030'
period = "test"
models = {}
for folder in os.listdir(runs):
    if '_2608_' in folder:
        run_dir = Path(fr"{runs}/{folder}")

        try:
            with open(run_dir / period / f"model_epoch{epoch}" / f"{period}_results.p", "rb") as fp:
                results = pickle.load(fp)
        except:
            continue

        # read config data
        config = yaml.safe_load(open(fr'{run_dir}\config.yml', 'r'))
        config['results'] = results
        config['model'] = f'{folder}_{epoch}'
        config['folder'] = folder

        # evaluate model performance
        nses = []
        seen_nses = []
        unseen_nses = []
        for gage in results.keys():
            nse = results[gage]['1D']['NSE']
            nses.append(nse)
            if gage in five_seen:
                seen_nses.append(nse)
            elif gage == '09081600':
                config['crystal'] = nse
            else:
                unseen_nses.append(nse)
        
        medianNSE = np.median(nses)
        meanNSE = np.mean(nses)
        config['medianNSE'] = medianNSE
        config['meanNSE'] = meanNSE
        config['NSEs'] = nses
        config['seenNSE'] = np.median(seen_nses)
        config['unseenNSE'] = np.median(unseen_nses)
        resultslist.append(config)

rdf = pd.DataFrame().from_dict(resultslist)
rdf2 = rdf[['train_basin_file', 'medianNSE', 'meanNSE', 'NSEs', 'seenNSE', 'unseenNSE', 'crystal', 'folder']].copy()
rdf2['regulation'] = rdf2['train_basin_file'].str.extract(r'(\d+)\.txt$').astype(int)

rdf2.sort_values(by='regulation', inplace=True)

def scatter(df, x, y):
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.show()

scatter(rdf2, 'regulation', 'crystal')


df_exploded = rdf2.explode('NSEs')
df_exploded['NSEs'] = df_exploded['NSEs'].astype(float)

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_exploded, x='regulation', y='NSEs', palette='Set2')
plt.title('Distribution of NSEs by Regulation')
plt.xlabel('Regulation Status')
plt.ylabel('NSE')
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()


folder = 'experiment1_2608_184410'
run_dir = Path(fr"{runs}/{folder}")
with open(run_dir / period / f"model_epoch{epoch}" / f"{period}_results.p", "rb") as fp:
    results = pickle.load(fp)
gage = '09081600'
Qsim = f'Q_cfs_sim'
Qobs = 'Q_cfs_obs'
df = pd.DataFrame()

qobs = results[gage]['1D']['xr'][Qobs]
qsim = results[gage]['1D']['xr'][Qsim]
df['date'] = pd.to_datetime(qobs['date'])
df.index = df['date']
df[Qobs] = qobs
df[Qsim] = qsim

def hydrograph(df, obs, sim, start = '10-01-2024', end = '09-30-2025'):
    fig, ax = plt.subplots()
    df = df.loc[start:end]
    ax.plot(df.index, df[obs], label='Q obs')
    ax.plot(df.index, df[sim], label='Q sim')
    ax.set_xlabel('date')
    ax.set_ylabel('Q')
    ax.legend()
    plt.show()

hydrograph(df, Qsim, Qobs)
