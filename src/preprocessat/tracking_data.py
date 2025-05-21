# %%
import os
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from train_test_creator import get_train_test_from_final_dataset

from dotenv import load_dotenv

load_dotenv()


FILE_NAME = "tracking_week_"
FINAL_FILE = pd.DataFrame()
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
DEBUG = False



#%%

dd_tracking = []
for week in range(1, WEEKS + 1):
    dd_tracking.append(dd.read_csv(os.path.join('..', '..', 'data', f'tracking_week_{week}.csv'), dtype={'jerseyNumber': 'float64',
    'nflId': 'float64'}))
dd_tracking = dd.concat(dd_tracking, axis=0)


columns_to_drop = [col for col in dd_tracking.columns if col not in TRACKING_FEATURES]
dd_tracking = dd_tracking[dd_tracking["frameType"] == "BEFORE_SNAP"].copy()
dd_tracking = dd_tracking.drop(columns=columns_to_drop).copy()
dd_tracking = dd_tracking.fillna(0).copy()


#%%
def get_mean_dict(df, mean_cols):
    """
    Get the mean of the columns in mean_cols grouped by the columns in groupby_cols.
    """
    mean_dict = {}
    for col in mean_cols:
        mean_dict[col] = df[col].mean().compute()
    return mean_dict

def get_std_dict(df, std_cols):
    """
    Get the std of the columns in std_cols grouped by the columns in groupby_cols.
    """
    std_dict = {}
    for col in std_cols:
        std_dict[col] = df[col].std().compute()
    return std_dict

def standarize_tensor(df, mean_dict, std_dict):
    """
    Standarize the tensor using the mean and std dictionaries.
    """
    for col in df.columns:
        if col in mean_dict.keys():
            df[col] = (df[col] - mean_dict[col]) / std_dict[col]
    return df

#%%
with ProgressBar():
    mean_dict = get_mean_dict(dd_tracking, REAL_FEATURES)
    std_dict = get_std_dict(dd_tracking, REAL_FEATURES)
    dd_tracking = standarize_tensor(dd_tracking, mean_dict, std_dict)
#%%
with ProgressBar():

    dd_tracking_train, dd_tracking_test = get_train_test_from_final_dataset(dd_tracking)
    print(f"Saving tracking_{WEEKS}_weeks_train.parquet ...")
    dd_tracking_train.to_parquet(os.path.join('data', 'final', f'tracking_{WEEKS}_weeks_train'), write_index=False, compute=True)
    print(f"Saving tracking_{WEEKS}_weeks_test.parquet ...")
    dd_tracking_test.to_parquet(os.path.join('data', 'final', f'tracking_{WEEKS}_weeks_test'), write_index=False, compute=True)
