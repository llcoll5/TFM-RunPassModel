# %%
import os
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from train_test_creator import get_train_test_from_final_dataset

from dotenv import load_dotenv

load_dotenv()

FILE_NAME = "tracking_week_"
WEEKS = 9
TRACKING_FEATURES = [
    "gameId",
    "playId",
    "nflId",
    "frameId",
    "x",
    "y",
    "s",
    "a",
    "dis",
    "o",
    "dir"
]
REAL_FEATURES = [
    "x",
    "y",
    "s",
    "a",
    "dis",
    "o",
    "dir"    
]

dd_tracking = []
for week in range(1, WEEKS + 1):
    dd_tracking.append(dd.read_csv(os.path.join('..', '..', 'data', f'tracking_week_{week}.csv'), dtype={'jerseyNumber': 'float64',
    'nflId': 'float64'}))
dd_tracking = dd.concat(dd_tracking, axis=0)

#%%

with ProgressBar():
    means = dd_tracking[REAL_FEATURES].mean().compute()
    stds = dd_tracking[REAL_FEATURES].std().compute()

#%%
def standardize(df, means, stds):
    for col in means.index:
        df[col] = (df[col] - means[col]) / stds[col]
    return df
with ProgressBar():
    standardized_tracking = standardize(dd_tracking[REAL_FEATURES], means,  stds)

#%%

dd_tracking[REAL_FEATURES] = standardized_tracking

#%%

columns_to_drop = [col for col in dd_tracking.columns if col not in TRACKING_FEATURES]
dd_tracking = dd_tracking[dd_tracking["frameType"] == "BEFORE_SNAP"].copy()
dd_tracking = dd_tracking.drop(columns=columns_to_drop).copy()
dd_tracking = dd_tracking.fillna(0).copy()

#%%
dd_tracking.head(10)
    
#%%
with ProgressBar():

    dd_tracking_train, dd_tracking_test = get_train_test_from_final_dataset(dd_tracking)
    print(f"Saving tracking_{WEEKS}_weeks_train.parquet ...")
    dd_tracking_train.to_parquet(os.path.join('..', '..', 'data', 'final', f'tracking_{WEEKS}_weeks_train'), write_index=False, compute=True)
    print(f"Saving tracking_{WEEKS}_weeks_test.parquet ...")
    dd_tracking_test.to_parquet(os.path.join('..', '..', 'data', 'final', f'tracking_{WEEKS}_weeks_test'), write_index=False, compute=True)
