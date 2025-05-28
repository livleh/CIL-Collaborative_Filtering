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
from config import general_params, embeddings_params as params

SEED = general_params['seed']
DATA_DIR = general_params['data_dir']
N_USERS = general_params['n_users']
N_PAPERS = general_params['n_papers']
NUM_EPOCHS = params['n_epochs']
DIM = params['dim']
LEARNING_RATE = params['lr']
BATCH_SIZE = params['batch_size']

torch.manual_seed(SEED)
np.random.seed(SEED)

def impute_values(mat: np.ndarray) -> np.ndarray:
    return np.nan_to_num(mat, nan=3.0)

def get_dataset(df: pd.DataFrame) -> torch.utils.data.Dataset:
    """Conversion from pandas data frame to torch dataset."""
    
    sids = torch.from_numpy(df["sid"].to_numpy())
    pids = torch.from_numpy(df["pid"].to_numpy())
    ratings = torch.from_numpy(df["rating"].to_numpy()).float()
    return torch.utils.data.TensorDataset(sids, pids, ratings)

def evaluate(valid_df: pd.DataFrame, pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """
    Inputs:
        valid_df: Validation data, returned from read_data_df for example.
        pred_fn: Function that takes in arrays of sid and pid and outputs their rating predictions.

    Outputs: Validation RMSE
    """
    
    preds = pred_fn(valid_df["sid"].values, valid_df["pid"].values)
    return root_mean_squared_error(valid_df["rating"].values, preds)

def make_submission(pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], filename: os.PathLike):
    """Makes a submission CSV file that can be submitted to kaggle.

    Inputs:
        pred_fn: Function that takes in arrays of sid and pid and outputs a score.
        filename: File to save the submission to.
    """
    
    df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    # Get sids and pids
    sid_pid = df["sid_pid"].str.split("_", expand=True)
    sids = sid_pid[0]
    pids = sid_pid[1]
    sids = sids.astype(int).values
    pids = pids.astype(int).values
    
    df["rating"] = pred_fn(sids, pids)
    df.to_csv(filename, index=False)

class EmbeddingDotProductModel(nn.Module):
    def __init__(self, num_scientists: int, num_papers: int, dim: int):
        super().__init__()

        # Assign to each scientist and paper an embedding
        self.scientist_emb = nn.Embedding(num_scientists, dim)
        self.paper_emb = nn.Embedding(num_papers, dim)
        
    def forward(self, sid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            sid: [B,], int
            pid: [B,], int
        
        Outputs: [B,], float
        """

        # Per-pair dot product
        return torch.sum(self.scientist_emb(sid) * self.paper_emb(pid), dim=-1)


def train(model, train_loader, device, optim, valid_loader):
    for epoch in range(NUM_EPOCHS):
        # Train model for an epoch
        total_loss = 0.0
        total_data = 0
        model.train()
        for sid, pid, ratings in tqdm(train_loader):
            # Move data to GPU
            sid = sid.to(device)
            pid = pid.to(device)
            ratings = ratings.to(device)

            # Make prediction and compute loss
            pred = model(sid, pid)
            loss = F.mse_loss(pred, ratings)

            # Compute gradients w.r.t. loss and take a step in that direction
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Keep track of running loss
            total_data += len(sid)
            total_loss += len(sid) * loss.item()

        # Evaluate model on validation data
        total_val_mse = 0.0
        total_val_data = 0
        model.eval()
        for sid, pid, ratings in tqdm(valid_loader):
            # Move data to GPU
            sid = sid.to(device)
            pid = pid.to(device)
            ratings = ratings.to(device)

            # Clamp predictions in [1,5], since all ground-truth ratings are
            pred = model(sid, pid).clamp(1, 5)
            mse = F.mse_loss(pred, ratings)

            # Keep track of running metrics
            total_val_data += len(sid)
            total_val_mse += len(sid) * mse.item()

        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train loss={total_loss / total_data:.3f}, Valid RMSE={(total_val_mse / total_val_data) ** 0.5:.3f}")

def run(args):
    mode, arg = args
    
    device = torch.device('mps' if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    
    df, tbr_df = read_data_df()
    
    # Define model (10k scientists, 1k papers, 32-dimensional embeddings) and optimizer
    model = EmbeddingDotProductModel(N_USERS, N_PAPERS, DIM).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if mode == 'cv':
        rmses = []
        kf = KFold(n_splits=arg, shuffle=True, random_state=SEED)

        for i, (train_idx, val_idx) in enumerate(kf.split(df), 1):
            train_df = df.iloc[train_idx]
            valid_df = df.iloc[val_idx]
            
            train_dataset = get_dataset(train_df)
            valid_dataset = get_dataset(valid_df)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            train(model, train_loader, device, optim, valid_loader)
            
            pred_fn = lambda sids, pids: model(torch.from_numpy(sids).to(device), torch.from_numpy(pids).to(device)).clamp(1, 5).cpu().numpy()

            # Evaluate on validation data
            with torch.no_grad():
                val_score = evaluate(valid_df, pred_fn)
            
            rmses.append(val_score)
            print(f"Validation RMSE: {val_score:.3f}")
        
        print("CV Mean Validation Score", np.mean(rmses))
    
    else:
        train_df, valid_df = split_df(df, split_size=arg)
        
        train_dataset = get_dataset(train_df)
        valid_dataset = get_dataset(valid_df)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        train(model, train_loader, device, optim, valid_loader)
        
        pred_fn = lambda sids, pids: model(torch.from_numpy(sids).to(device), torch.from_numpy(pids).to(device)).clamp(1, 5).cpu().numpy()

        # Evaluate on validation data
        with torch.no_grad():
            val_score = evaluate(valid_df, pred_fn)

        print(f"Validation RMSE: {val_score:.3f}")
        
        if mode == 'predict':
            with torch.no_grad():
                make_submission(pred_fn, "submissions/learned_embedding_submission.csv")