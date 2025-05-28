from typing import Tuple, Callable

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import os
from tqdm.auto import tqdm
from data_handler.data_handler import *
from config import general_params, als_params as params

factor = params['factor']
lambda_reg = params['lambda_reg']
num_iterations = params['num_iterations']

SEED = general_params['seed']
DATA_DIR = general_params['data_dir']
N_USERS = general_params['n_users']
N_PAPERS = general_params['n_papers']

torch.manual_seed(SEED)
np.random.seed(SEED)

def read_data_matrix(df: pd.DataFrame) -> np.ndarray:
    """Returns matrix view of the training data, where columns are scientists (sid) and
    rows are papers (pid)."""

    return df.pivot(index="sid", columns="pid", values="rating").values

def evaluate(valid_df: pd.DataFrame, pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """
    Inputs:
        valid_df: Validation data, returned from read_data_df for example.
        pred_fn: Function that takes in arrays of sid and pid and outputs their rating predictions.

    Outputs: Validation RMSE
    """
    
    preds = pred_fn(valid_df["sid"].values, valid_df["pid"].values)
    return root_mean_squared_error(valid_df["rating"].values, preds)

def matrix_pred_fn(train_recon: np.ndarray, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """
    Input:
        train_recon: (M, N) matrix with predicted values for every (sid, pid) pair.
        sids: (D,) vector with integer scientist IDs.
        pids: (D,) vector with integer paper IDs.
        
    Outputs: (D,) vector with predictions.
    """

    return train_recon[sids, pids]


def train(V, U, train_mat, mask, train_df, valid_df):
        
    for it in range(num_iterations):
        # Fix V, solve for U
        for i in range(N_USERS):
            V_i = V[mask[i]]
            train_mat_i = train_mat[i][mask[i]]
            A = V_i.T @ V_i + lambda_reg * np.eye(factor)
            b = V_i.T @ train_mat_i
            U[i] = np.linalg.solve(A, b)

        # Fix U, solve for V
        for j in range(N_PAPERS):
            U_j = U[mask[:, j]]
            train_mat_j = train_mat[:, j][mask[:, j]]
            A = U_j.T @ U_j + lambda_reg * np.eye(factor)
            b = U_j.T @ train_mat_j
            V[j] = np.linalg.solve(A, b)

        # Compute loss
        recon = U @ V.T
        squared_error = ((train_mat - recon)[mask])**2
        mse = np.sum(squared_error) / np.sum(mask)
        rmse = np.sqrt(mse)
        
        pred_fn = lambda sids, pids: matrix_pred_fn(recon, sids, pids)
        test_score = evaluate(train_df, pred_fn)
        val_score = evaluate(valid_df, pred_fn)

        print(f"Iteration {it + 1}/{num_iterations}, RMSE: {test_score:.4f}, Validation RMSE: {val_score:.4f}")
        
    return recon, test_score, val_score

def run(args):
    mode, arg = args
    
    df, tbr_df = read_data_df()
    
    if mode == 'cv':
        rmses = []
        kf = KFold(n_splits=arg, shuffle=True, random_state=SEED)

        for i, (train_idx, val_idx) in enumerate(kf.split(df), 1):
            train_df = df.iloc[train_idx]
            valid_df = df.iloc[val_idx]
            train_mat = read_data_matrix(train_df) # num_pid x num_sid

            U = np.random.normal(size=(N_USERS, factor))
            V = np.random.normal(size=(N_PAPERS, factor))
            
            mask = ~np.isnan(train_mat)
            
            _, _, val_score = train(V, U, train_mat, mask, train_df, valid_df)
            
            rmses.append(val_score)
            
        print("CV Mean Validation Score", np.mean(rmses))
        
    else:
        
        train_df, valid_df = split_df(df, split_size=arg)
        train_mat = read_data_matrix(train_df) # num_pid x num_sid


        U = np.random.normal(size=(N_USERS, factor))
        V = np.random.normal(size=(N_PAPERS, factor))
                
        mask = ~np.isnan(train_mat)
        
        recon, _, _ =train(V, U, train_mat, mask, train_df, valid_df)
        
        
        if mode == 'predict':
            pred_fn = lambda sids, pids: matrix_pred_fn(recon, sids, pids)
            
            df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

            sid_pid = df["sid_pid"].str.split("_", expand=True)
            sids = sid_pid[0]
            pids = sid_pid[1]
            sids = sids.astype(int).values
            pids = pids.astype(int).values
            
            df["rating"] = pred_fn(sids, pids)
            df.to_csv("submissions/submission_als.csv", index=False)
