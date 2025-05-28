from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import general_params as gp

SEED = gp['seed']


def read_data_df() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads in data."""

    ratings_df = pd.read_csv(gp['train_data_path'])

    ratings_df[["sid", "pid"]] = ratings_df["sid_pid"].str.split("_", expand=True)
    ratings_df = ratings_df.drop("sid_pid", axis=1)
    ratings_df["sid"] = ratings_df["sid"].astype(int)
    ratings_df["pid"] = ratings_df["pid"].astype(int)
    ratings_df = ratings_df[['sid','pid','rating']]
    
    wishlist_df = pd.read_csv(gp['train_tbr_path'])

    return ratings_df, wishlist_df

def split_df(df, split_size) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the data into training and validation sets with the given ratio."""
    train_df, valid_df = train_test_split(df, test_size=split_size, random_state=SEED)
    return train_df, valid_df


def read_data_matrix(df: pd.DataFrame) -> np.ndarray:
    """Returns matrix view of the training data, where columns are scientists (sid) and
    rows are papers (pid)."""

    return df.pivot(index="sid", columns="pid", values="rating").values

def read_sample_df():
    sub_df = pd.read_csv(gp['sample_submission_path'])
    sub_df[["sid","pid"]] = (
        sub_df.sid_pid
            .str.split("_", expand=True)
            .astype(int)
    )
    return sub_df