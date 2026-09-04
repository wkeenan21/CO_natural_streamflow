import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pyarrow
import yaml
import os
import pandas as pd
import numpy as np
import seaborn as sns

cwd = r'G:\My Drive\natural_streamflow_colab'
runs = fr'{cwd}\run_dir'

five_unseen = ['09081600', '09266500', '09253000', '09312600', '09223000']
five_seen = ['09217900', '09123450', '09210500', '383926107593001', '09310700']

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
            seen_nses = []
            unseen_nses = []
            for gage in results.keys():
                nse = results[gage]['1D']['NSE']
                nses.append(nse)
                if gage in five_unseen:
                    unseen_nses.append(nse)
                else:
                    seen_nses.append(nse)
            
            medianNSE = np.median(nses)
            meanNSE = np.mean(nses)
            config['medianNSE'] = medianNSE
            config['meanNSE'] = meanNSE
            config['NSEs'] = nses
            config['seenNSE'] = np.median(seen_nses)
            config['unseenNSE'] = np.median(unseen_nses)
            resultslist.append(config)

adf = pd.DataFrame().from_dict(resultslist)

# filter to 30 epochs
adf2 = adf[adf.epoch=='030']
# view results
fig, ax = plt.subplots()
x = 'output_dropout'
y = 'unseenNSE'
ax.scatter(adf2[x], adf2[y])
ax.set_xlabel(x)
ax.set_ylabel(y)
plt.show()

# see if overtraining
fig, ax = plt.subplots()

x = 'epoch'
y = 'medianNSE'

# Map each unique hidden_size to a unique color automatically
unique_sizes = adf['hidden_size'].unique()
colors = plt.cm.tab10.colors  # Color palette
color_map = {size: colors[i % len(colors)] for i, size in enumerate(unique_sizes)}

for model in adf.folder.unique():
    mdf = adf[adf.folder == model]
    
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


adf3 = adf[['medianNSE', 'meanNSE', 'hidden_size', 'batch_size', 'output_dropout', 'epoch', 'NSEs']].sort_values(by='medianNSE', ascending=False)
fig, ax = plt.subplots()
ax.scatter(adf3.epoch, adf3.medianNSE)
ax.set_xlabel('epoch')
ax.set_ylabel('NSE')
plt.show()

# examine duplicates to assess reproducability

df = adf3.copy()
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
test_gages = ['09081600', '09217900', '09123450', '09312600', '09210500', '09266500', '09223000']

resultslist = []
epoch = '030'
period = "test"
models = {}

for folder in os.listdir(runs):
    if '_0209_' in folder or '_0109_' in folder:
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
        nses = {}
        pbiases = {}

        for gage in results.keys():
            # Extract NSE
            nse = results[gage]['1D']['NSE']
            nses[gage] = nse
            
            # Access Q_obs and Q_sim arrays/DataArrays
            q_obs = results[gage]['1D']['xr']['Q_cfs_obs']
            q_sim = results[gage]['1D']['xr']['Q_cfs_sim']

            # Calculate Percent Bias (%): positive value indicates overestimation
            sum_obs = np.sum(q_obs)
            if sum_obs != 0:
                pbias = 100 * (np.sum(q_sim - q_obs) / sum_obs)
            else:
                pbias = np.nan
            
            pbiases[gage] = pbias
        
        # Aggregate NSE stats
        justnses = list(nses.values())
        config['medianNSE'] = np.median(justnses)
        config['meanNSE'] = np.mean(justnses)
        config['NSEs'] = nses

        # Aggregate PBIAS stats
        justpbiases = [p for p in pbiases.values() if not np.isnan(p)]
        config['medianPBIAS'] = np.median(justpbiases) if justpbiases else np.nan
        config['meanPBIAS'] = np.mean(justpbiases) if justpbiases else np.nan
        config['PBIASes'] = pbiases

        resultslist.append(config)

adf = pd.DataFrame().from_dict(resultslist)
adf2 = adf[['train_basin_file', 'medianNSE', 'meanNSE', 'NSEs', 'PBIASes', 'meanPBIAS', 'medianPBIAS', 'folder']].copy()
adf2['regulation'] = adf2['train_basin_file'].str.extract(r'(\d+)\.txt$').astype(int)
adf2.sort_values(by='regulation', inplace=True)
adf_noCU = adf2[~adf2.folder.str.contains('wCU')]
adf_wCU = adf2[adf2.folder.str.contains('wCU')]

def scatter(df, x, y):
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.show()

scatter(adf_noCU, 'regulation', 'medianPBIAS')

plt.figure(figsize=(8, 5))
for gage in five_unseen:
    plt.plot(adf_noCU['regulation'], adf_noCU['NSEs'].apply(lambda x: x.get(gage)), marker='o', label=str(gage))

plt.xlabel('Regulation')
plt.ylabel('NSE (1 = perfect, <0 = worse than using mean)')
plt.ylim(-1, 1)
plt.grid(True)
plt.legend(title='Gage ID')
plt.tight_layout()
plt.show()


df_exploded = adf2.explode('NSEs')
df_exploded['NSEs'] = df_exploded['NSEs'].astype(float)

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_exploded, x='regulation', y='NSEs', palette='Set2')
plt.title('Distribution of NSEs by Regulation')
plt.xlabel('Regulation Status')
plt.ylabel('NSE')
plt.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()


for folder in adf2.folder.unique():
    reg = adf2[adf2.folder==folder]['regulation'].iloc[0]

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

    def hydrograph(df, obs, sim, start = '10-01-2024', end = '09-30-2025', title='title'):
        fig, ax = plt.subplots()
        df = df.loc[start:end]
        ax.plot(df.index, df[obs], label='Q obs')
        ax.plot(df.index, df[sim], label='Q sim')
        ax.set_xlabel('date')
        ax.set_ylabel('Q')
        ax.legend()
        ax.set_title(title)
        plt.show()

    hydrograph(df, Qobs, Qsim, title=reg)
