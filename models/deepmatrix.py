from typing import Tuple, Callable

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error
import os
from tqdm.auto import tqdm
from data_handler.data_handler import *
from config import general_params, mdm_params as params

SEED = general_params['seed']
DATA_DIR = general_params['data_dir']
SUBMISSION_DIR = general_params['submissions_dir']
N_USERS = general_params['n_users']
N_PAPERS = general_params['n_papers']
NUM_EPOCHS = params['n_epochs']
BATCH_SIZE = params['batch_size']
DIM = params['dim']
LEARNING_RATE = params['lr']
HIDDEN_DIM = params['hidden_dim']
MODEL_NAME = "modified_deep_matrix"
OUTPUT_DIR = 'saved_models'



torch.manual_seed(SEED)
np.random.seed(SEED)

def impute_values(mat: np.ndarray) -> np.ndarray:
    return np.nan_to_num(mat, nan=0.0)

def get_dataset(df: pd.DataFrame) -> torch.utils.data.Dataset:
    """Conversion from pandas data frame to torch dataset."""
    
    sids = torch.from_numpy(df["sid"].to_numpy())
    pids = torch.from_numpy(df["pid"].to_numpy())
    ratings = torch.from_numpy(df["rating"].to_numpy()).float()
    return torch.utils.data.TensorDataset(sids, pids, ratings)

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

def make_submission(filename: os.PathLike, model, device):
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
    
    df["rating"] = batched_pred_fn(model, sids, pids, device)
    df.to_csv(filename, index=False)
    
class DeepMatrixFactorizationModel(nn.Module):
    def __init__(self, num_scientists: int, num_papers: int, dim: int, hidden_dim: int, Y: np.ndarray, rating: np.ndarray):
        super().__init__()

        self.register_buffer("Y", torch.from_numpy(Y).float())
        self.register_buffer("rating", torch.from_numpy(rating).float())


        self.scientist_nn = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_papers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.ReLU()
        )

        self.paper_nn = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_scientists, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.ReLU()
        )

        self.srating_nn = nn.Sequential(
            nn.Linear(num_papers, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.ReLU()
        )

        self.prating_nn = nn.Sequential(
            nn.Linear(num_scientists, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.ReLU()
        )

        self.final_nn = nn.Sequential(
            nn.Linear(4*dim, 2*dim),
            nn.ReLU(),
            nn.Linear(2*dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )


    def forward(self, sid: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            sid: [B,], int
            pid: [B,], int

        Outputs: [B,], float
        """
        scientist_row = self.Y[sid, :]
        paper_row = self.Y[:, pid].T

        # to not leak info
        #scientist_row[pid] = 0
        #paper_row[sid] = 0

        srating_row = self.rating[sid, :]
        prating_row = self.rating[:, pid].T



        p = self.scientist_nn(scientist_row)
        q = self.paper_nn(paper_row)

        srating = self.srating_nn(srating_row)
        prating = self.prating_nn(prating_row)

        #r = p * srating
        #sr = q * prating


        # Per-pair dot product
        return self.final_nn(torch.cat([p,q,srating,prating], dim=1)).squeeze(1)
        #return self.final_nn(torch.cat([p,q], dim=1)).squeeze(1)
        
def train(model, train_loader, valid_loader, device, optim):
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    history = {
        'epoch': [],
        'train_loss': [],
        'val_rmse': [],
    }

    best_val_rmse = float('inf')
    best_model_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_best_val_rmse.pth")

    epochs = tqdm(range(NUM_EPOCHS), desc="Epochs")
    for epoch in epochs:
        # Train model for an epoch
        total_loss = 0.0
        total_data = 0
        model.train()
        train_batch = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for sid, pid, ratings in train_batch:
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
        valid_batch = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Valid]", leave=False)
        for sid, pid, ratings in valid_batch:
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

        curr_rmse = (total_val_mse / total_val_data) ** 0.5
        curr_loss = total_loss / total_data
        history['epoch'].append(epoch+1)
        history['train_loss'].append(curr_loss)
        history['val_rmse'].append(curr_rmse)

        saved_this_epoch = False
        if curr_rmse < best_val_rmse:
            best_val_rmse = curr_rmse
            torch.save(model.state_dict(), best_model_path)
            saved_this_epoch = True

        postfix_str = f"Train Loss={curr_loss:.3f}, Valid RMSE={curr_rmse:.3f}, Best RMSE={best_val_rmse:.3f}"
        if saved_this_epoch:
            postfix_str += " (Saved Best Model)"
        epochs.set_postfix_str(postfix_str)
        
    return history

def batched_pred_fn(model, sids, pids, device, batch_size=BATCH_SIZE):
    results = []
    num_samples = len(sids)
    best_model_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_best_val_rmse.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    for i in range(0, num_samples, batch_size):
        batch_sids = sids[i:i+batch_size]
        batch_pids = pids[i:i+batch_size]

        batch_sids_tensor = torch.from_numpy(batch_sids).to(device)
        batch_pids_tensor = torch.from_numpy(batch_pids).to(device)

        batch_preds = model(batch_sids_tensor, batch_pids_tensor).clamp(1, 5).cpu().numpy()
        results.append(batch_preds)

        del batch_sids_tensor, batch_pids_tensor
        torch.cuda.empty_cache()

    return np.concatenate(results)

def run(args):
    mode, arg = args
    
    device = torch.device('mps' if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    df, wishlist_df = read_data_df()
    
    
    
    if mode == 'cv':
        rmses = []
        kf = KFold(n_splits=arg, shuffle=True, random_state=SEED)

        for i, (train_idx, val_idx) in enumerate(kf.split(df), 1):
            train_df = df.iloc[train_idx]
            valid_df = df.iloc[val_idx]
            
            Y = read_data_matrix(train_df)
            Y = impute_values(Y)
            
            wishlist_df["rating"] = 1

            missing_sids = []
            for i in range(N_USERS):
                if wishlist_df[wishlist_df["sid"] == i].shape[0] == 0:
                    missing_sids.append(i)

            for i in range(len(missing_sids)):
                wishlist_df = pd.concat([wishlist_df, pd.DataFrame({"sid": [missing_sids[i]], "pid": [0], "rating": [0]})], ignore_index=True)

            wishlist = read_data_matrix(wishlist_df)
            wishlist = impute_values(wishlist)
            
            model = DeepMatrixFactorizationModel(N_USERS, N_PAPERS, DIM, HIDDEN_DIM, Y, wishlist).to(device)
            optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            train_dataset = get_dataset(train_df)
            valid_dataset = get_dataset(valid_df)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
            
            history = train(model, train_loader, valid_loader, device, optim)
            epoch = np.argmax(history['val_rmse'])
            rmses.append(history["val_rmse"][epoch])
        
        print("CV Mean Validation Score", np.mean(rmses))
        
    else:
        
        train_df, valid_df = split_df(df, split_size=arg)
        
        Y = read_data_matrix(train_df)
        Y = impute_values(Y)
        
        wishlist_df["rating"] = 1

        missing_sids = []
        for i in range(N_USERS):
            if wishlist_df[wishlist_df["sid"] == i].shape[0] == 0:
                missing_sids.append(i)

        for i in range(len(missing_sids)):
            wishlist_df = pd.concat([wishlist_df, pd.DataFrame({"sid": [missing_sids[i]], "pid": [0], "rating": [0]})], ignore_index=True)

        wishlist = read_data_matrix(wishlist_df)
        wishlist = impute_values(wishlist)
        
        model = DeepMatrixFactorizationModel(N_USERS, N_PAPERS, DIM, HIDDEN_DIM, Y, wishlist).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_dataset = get_dataset(train_df)
        valid_dataset = get_dataset(valid_df)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        
        history = train(model, train_loader, valid_loader, device, optim)
        
        epoch = np.argmax(history['val_rmse'])
        print("Best epoch: ", epoch)
        print("Metrics: ", history["val_rmse"][epoch])
        
        if mode == 'predict':
            with torch.no_grad():
                make_submission("submissions/deep_matrix_submission.csv", model, device)
            
