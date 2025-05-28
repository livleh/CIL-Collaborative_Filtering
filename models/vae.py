from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
from data_handler.data_handler import *
from config import general_params, vae_params as params
import os

SEED = general_params['seed']
DATA_DIR = general_params['data_dir']
N_USERS = general_params['n_users']
N_PAPERS = general_params['n_papers']
DS_NAME = params['ds_name']
BATCH_SIZE = params['batch_size']
LEARNING_RATE = params['lr']


class RatingDataset(Dataset):
    def __init__(self, rating_matrix, impute_func):
        self.original_data = rating_matrix
        self.data = impute_func(rating_matrix)
        self.mask = ~torch.isnan(torch.tensor(rating_matrix)).to(torch.bool) 

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]


class VAE_CF(nn.Module):
    def __init__(self, num_items, hidden_dim=20, latent_dim=10, dropout=0.5):
        super(VAE_CF, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_items)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, mask):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x[mask], x[mask])
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss.item(), kl_loss
    
def impute_int(matrix):
    matrix_filled = matrix.copy()
    return torch.tensor(np.nan_to_num(matrix_filled, nan=3), dtype=torch.float32)

def impute_mean(matrix):    
    matrix_filled = matrix.copy()
    for i in range(matrix.shape[0]):
        row_means = (matrix_filled[i][~np.isnan(matrix[i])]).mean()
        matrix_filled[i][np.isnan(matrix[i])] = row_means
    return torch.tensor(matrix_filled, dtype=torch.float32)

def impute_global_mean(matrix):
    global_mean = matrix[~np.isnan(matrix)].mean()
    matrix_filled = matrix.copy()
    matrix_filled[np.isnan(matrix)] = global_mean
    return torch.tensor(matrix_filled, dtype=torch.float32)

def train(model, train_loader, optimizer, train_mat, valid_mat, impute_func):
    for epoch in range(10):
        model.train()
        total_loss = 0
        total_data = 0
        for batch, batch_mask in train_loader:
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl = model.loss_function(recon, batch, mu, logvar, batch_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += recon_loss * len(batch)
            total_data += len(batch)
    
        model.eval()
        with torch.no_grad():
            # Use imputed train matrix to get representations
            train_tensor = impute_func(train_mat)
            recon, _, _ = model(train_tensor)
    
            # Evaluate only on valid entries in validation matrix
            valid_mask = ~np.isnan(valid_mat)
            valid_true = torch.tensor(valid_mat[valid_mask], dtype=torch.float32)
            valid_pred = recon[valid_mask]
    
            val_rmse = torch.sqrt(torch.mean((valid_true - valid_pred) ** 2)).item()
            print(f"Epoch {epoch+1:02d} | Train RMSE: {(total_loss / total_data) ** 0.5:.4f}, Validation RMSE: {val_rmse:.4f}")
    return val_rmse


def run(args):
    mode, arg = args
    
    device = torch.device('mps' if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    # Load data
    df, tbr_df = read_data_df()
    
    # Define imputation strategies to test
    imputation_strategies = {
        "const": impute_int,
        "row": impute_mean,
        "global": impute_global_mean,
    }
    
    impute_func = imputation_strategies[DS_NAME]
    
    if mode == 'cv':
        rmses = []
        kf = KFold(n_splits=arg, shuffle=True, random_state=SEED)

        for i, (train_idx, val_idx) in enumerate(kf.split(df), 1):
            train_df = df.iloc[train_idx]
            valid_df = df.iloc[val_idx]
            train_mat = read_data_matrix(train_df)
            valid_mat = read_data_matrix(valid_df)

            train_dataset = RatingDataset(train_mat, impute_func)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            model = VAE_CF(num_items=train_mat.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            
            val_rmse = train(model, train_loader, optimizer, train_mat, valid_mat, impute_func)
            
            rmses.append(val_rmse)
        
        print("CV Mean Validation Score", np.mean(rmses))
        
    else:
        train_df, valid_df = split_df(df, arg)
        train_mat = read_data_matrix(train_df)
        valid_mat = read_data_matrix(valid_df)

        train_dataset = RatingDataset(train_mat, impute_func)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = VAE_CF(num_items=train_mat.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        train(model, train_loader, optimizer, train_mat, valid_mat, impute_func)
        
        if mode == 'predict':
            
            df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

            sid_pid = df["sid_pid"].str.split("_", expand=True)
            sids = sid_pid[0]
            pids = sid_pid[1]
            sids = sids.astype(int).values
            pids = pids.astype(int).values
            
            model.eval()
            with torch.no_grad():
                recon_all = model(impute_func(train_mat))[0]  # [n_users, n_items]

            # now vectorized indexing:
            rows = torch.tensor(sids, dtype=torch.long)
            cols = torch.tensor(pids, dtype=torch.long)
            batch_preds = recon_all[rows, cols].cpu().numpy()
            
            df["rating"] = batch_preds
            df.to_csv("submissions/submission_vae.csv", index=False)