import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gc
from data_handler.data_handler import *
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm.auto import tqdm, trange


from config import attention_params as params, general_params
SEED = general_params['seed']
DATA_DIR = general_params['data_dir']
NUM_SCIENTISTS = general_params['n_users']
NUM_PAPERS = general_params['n_papers']
SID_WISHLIST_SIZE = params['sid_wishlist_size']
SID_CONTEXT_SIZE = params['sid_context_size'] 
PID_WISHLIST_SIZE = params['pid_wishlist_size']
PID_CONTEXT_SIZE = params['pid_context_size'] 
EMBEDDING_DIM = params['embedding_dim']
NUM_HEADS = params['num_heads'] 
DROPOUT_RATE = params['dropout_rate'] 
L2_REG = params['l2_reg']
LEARNING_RATE = params['lr'] 
BATCH_SIZE = params['batch_size'] 
EPOCHS = params['n_epochs'] 

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def get_dataset_sid(df: pd.DataFrame, wishlist_df: pd.DataFrame, context_size: int, wishlist_size: int, save_path: str) -> torch.utils.data.Dataset:
    """
    Constructs a PyTorch Dataset for each (sid, pid, rating) entry, enriched with:
    - A fixed-size context of (pid, rating) tuples rated by the same scientist (sid).
    - A fixed-size wishlist of paper IDs on the scientist's wishlist.

    The dataset is cached at `save_path` if it exists.
    """

    if os.path.exists(save_path):
        print(f"Loading dataset from {save_path}")
        return torch.load(save_path, weights_only=False)

    # Extract data as tensors
    sids = torch.from_numpy(df["sid"].to_numpy())
    pids = torch.from_numpy(df["pid"].to_numpy())
    ratings = torch.from_numpy(df["rating"].to_numpy()).float()

    # Build mappings for context and wishlist
    sid_to_context = df.groupby("sid")[["pid", "rating"]].apply(
        lambda x: list(zip(x["pid"], x["rating"]))
    ).to_dict()

    sid_to_wishlist = wishlist_df.groupby("sid")["pid"].apply(list).to_dict()

    sid_context = []
    sid_wishlist = []

    # Construct dataset entries
    for sid_val, pid_val in zip(sids, pids):
        sid = sid_val.item()
        pid = pid_val.item()

        # Context: (pid, rating) tuples rated by this sid
        context = [(p, r) for p, r in sid_to_context[sid] if p != pid]

        # Pad or sample context to fixed size
        if len(context) >= context_size:
            context = random.sample(context, context_size)
        else:
            context += [(-1, 0.0)] * (context_size - len(context))

        sid_context.append(torch.tensor(context, dtype=torch.int))

        # Wishlist: papers on sid's wishlist
        wishlist = sid_to_wishlist.get(sid, [])
        if pid in wishlist:
            wishlist.remove(pid)

        if len(wishlist) >= wishlist_size:
            wishlist = random.sample(wishlist, wishlist_size)
        else:
            wishlist += [-1] * (wishlist_size - len(wishlist))

        sid_wishlist.append(torch.tensor(wishlist, dtype=torch.int))

    # Stack all examples into tensors
    sid_context = torch.stack(sid_context)
    sid_wishlist = torch.stack(sid_wishlist)

    # Create dataset and save
    dataset = torch.utils.data.TensorDataset(sids, pids, ratings, sid_context, sid_wishlist)
    torch.save(dataset, save_path)

    return dataset


def get_dataset_pid(df: pd.DataFrame, wishlist_df: pd.DataFrame, context_size: int, wishlist_size: int, save_path: str) -> torch.utils.data.Dataset:
    """
    Constructs a PyTorch Dataset for each (sid, pid, rating) entry, enriched with:
    - A fixed-size context of (sid, rating) tuples representing scientists who rated the same paper (pid).
    - A fixed-size wishlist of sids who have the paper (pid) on their wishlist.

    The dataset is cached at `save_path` if it exists.
    """

    if os.path.exists(save_path):
        print(f"Loading dataset from {save_path}")
        return torch.load(save_path, weights_only=False)

    # Extract data as tensors
    sids = torch.from_numpy(df["sid"].to_numpy())
    pids = torch.from_numpy(df["pid"].to_numpy())
    ratings = torch.from_numpy(df["rating"].to_numpy()).float()

    # Build mappings for context and wishlist
    pid_to_context = df.groupby("pid")[["sid", "rating"]].apply(
        lambda x: list(zip(x["sid"], x["rating"]))
    ).to_dict()

    pid_to_wishlist = wishlist_df.groupby("pid")["sid"].apply(list).to_dict()

    pid_context = []
    pid_wishlist = []

    # Construct dataset entries
    for sid_val, pid_val in zip(sids, pids):
        sid = sid_val.item()
        pid = pid_val.item()

        # Context: (sid, rating) tuples from other sids who rated the same pid
        context = [(s, r) for s, r in pid_to_context[pid] if s != sid]

        if len(context) >= context_size:
            context = random.sample(context, context_size)
        else:
            context += [(-1, 0.0)] * (context_size - len(context))

        pid_context.append(torch.tensor(context, dtype=torch.int))

        # Wishlist: sids who wishlisted this pid
        wishlist = pid_to_wishlist.get(pid, [])
        if sid in wishlist:
            wishlist.remove(sid)

        if len(wishlist) >= wishlist_size:
            wishlist = random.sample(wishlist, wishlist_size)
        else:
            wishlist += [-1] * (wishlist_size - len(wishlist))

        pid_wishlist.append(torch.tensor(wishlist, dtype=torch.int))

    # Stack all examples into tensors
    pid_context = torch.stack(pid_context)
    pid_wishlist = torch.stack(pid_wishlist)

    # Create dataset and save
    dataset = torch.utils.data.TensorDataset(sids, pids, ratings, pid_context, pid_wishlist)
    torch.save(dataset, save_path)

    return dataset



def save_predictions_csv(model, train_loader, valid_loader, device, train_output_file: str, valid_output_file: str):
    """
    Generates predictions from a trained model on both training and validation sets,
    clamps and rounds the outputs to the nearest integer between 1 and 5,
    and saves them as CSV files.
    """
    model.eval()  # Set model to evaluation mode

    def collect_predictions(loader):
        """Run inference and collect predictions for a given DataLoader."""
        results = []

        for sid, pid, rating, context, wishlist in tqdm(loader):
            # Move inputs to the target device
            sid, pid, rating = sid.to(device), pid.to(device), rating.to(device)
            context, wishlist = context.to(device), wishlist.to(device)

            with torch.no_grad():
                # Run model prediction
                predictions = model(sid, pid, context, wishlist)

            # Collect results as list of dictionaries
            for s, p, pred in zip(sid.cpu(), pid.cpu(), predictions.cpu()):
                results.append({
                    "sid": s.item(),
                    "pid": p.item(),
                    "predicted": pred.item()
                })

        return results

    # Generate and save predictions for training data
    train_results = collect_predictions(train_loader)
    pd.DataFrame(train_results).to_csv(train_output_file, index=False)

    # Generate and save predictions for validation data
    if valid_loader:
        print(f"Train predictions saved to {train_output_file}")
        valid_results = collect_predictions(valid_loader)
        pd.DataFrame(valid_results).to_csv(valid_output_file, index=False)
        print(f"Validation predictions saved to {valid_output_file}")
    else:
        print(f"Submission predictions saved to {train_output_file}")
    
    
def get_dataset_combined(rating_file: str, pred_sid_file: str, pred_pid_file: str, save_path: str) -> torch.utils.data.Dataset:
    """
    Combines predictions from SID- and PID-based models with ground truth ratings
    into a PyTorch TensorDataset and caches the result.
    """
    full_save_path = os.path.join(DATA_DIR, save_path)

    if os.path.exists(full_save_path):
        print(f"Loading dataset from {full_save_path}")
        return torch.load(full_save_path, weights_only=False)

    # Load CSVs
    rating_df = pd.read_csv(os.path.join(DATA_DIR, rating_file))
    pred_sid_df = pd.read_csv(os.path.join(DATA_DIR, pred_sid_file))
    pred_pid_df = pd.read_csv(os.path.join(DATA_DIR, pred_pid_file))

    # Extract 'sid' and 'pid' from 'sid_pid' column in rating_df
    rating_df[['sid', 'pid']] = rating_df['sid_pid'].str.split("_", expand=True)
    rating_df = rating_df.drop(columns=["sid_pid"])
    rating_df["sid"] = rating_df["sid"].astype(int)
    rating_df["pid"] = rating_df["pid"].astype(int)

    # Merge prediction files on (sid, pid)
    merged_df = pd.merge(pred_sid_df, pred_pid_df, on=["sid", "pid"], suffixes=('_sid', '_pid'))

    # Merge with actual ratings
    merged_df = pd.merge(merged_df, rating_df, on=["sid", "pid"])

    # Convert to tensors
    sids = torch.tensor(merged_df["sid"].values, dtype=torch.int)
    pids = torch.tensor(merged_df["pid"].values, dtype=torch.int)
    ratings = torch.tensor(merged_df["rating"].values, dtype=torch.float)
    preds_sid = torch.tensor(merged_df["predicted_sid"].values, dtype=torch.float)
    preds_pid = torch.tensor(merged_df["predicted_pid"].values, dtype=torch.float)

    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(sids, pids, ratings, preds_sid, preds_pid)

    # Save for reuse
    torch.save(dataset, full_save_path)
    print(f"Combined dataset saved to {full_save_path}")

    return dataset

class AttentionRecommenderSID(nn.Module):
    """
    Attention-based recommender that focuses on scientist behavior (SID-centric).
    """

    def __init__(self, num_sids: int, num_pids: int, emb_dim: int, dropout_rate: float, num_heads: int):
        super().__init__()

        # Embedding layers
        self.sid_embedding = nn.Embedding(num_sids, emb_dim)
        self.pid_embedding = nn.Embedding(num_pids + 1, emb_dim, padding_idx=-1)
        self.rating_embedding = nn.Embedding(6, emb_dim, padding_idx=0)

        # Layer norms for stabilization
        self.norm_sid = nn.LayerNorm(emb_dim)
        self.norm_pid = nn.LayerNorm(emb_dim)
        self.norm_context_pid = nn.LayerNorm(emb_dim)
        self.norm_rating = nn.LayerNorm(emb_dim)
        self.output_norm = nn.LayerNorm(emb_dim)

        # Attention layers
        self.context_attention = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.wishlist_attention = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout_rate, batch_first=True)

        # Projection layers
        self.proj_attn_1 = nn.Linear(emb_dim, emb_dim)
        self.proj_attn_2 = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout_rate)

        # Gating and final prediction layers
        self.gate_layer = nn.Linear(2 * emb_dim, emb_dim)
        self.fc1 = nn.Linear(emb_dim + 1, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 1)

    def forward(self, sid, pid, context_pid_ratings, wishlist):
        """
        Args:
            sid: (B,) 
            pid: (B,) 
            context_pid_ratings: (B, K, 2) where each entry is (pid, rating)
            wishlist: (B, W) 
        Returns:
            Predicted rating: (B,)
        """
        context_pids = context_pid_ratings[:, :, 0].long()      # (B, K)
        context_ratings = context_pid_ratings[:, :, 1].long()   # (B, K)

        # Embedding lookups + normalization
        sid_embed = self.norm_sid(self.sid_embedding(sid))      # (B, D)
        pid_embed = self.norm_pid(self.pid_embedding(pid))      # (B, D)
        wishlist_embed = self.norm_pid(self.pid_embedding(wishlist))    # (B, W, D)
        context_pid_embed = self.norm_context_pid(self.pid_embedding(context_pids))         # (B, K, D)
        context_rating_embed = self.norm_rating(self.rating_embedding(context_ratings))     # (B, K, D)

        # Multi-head attention over context (with queries: pid + wishlist)
        query = torch.cat([pid_embed.unsqueeze(1), wishlist_embed], dim=1)      # (B, W+1, D)
        context_attn, _ = self.context_attention(query, context_pid_embed, context_rating_embed)    # (B, W+1, D)
        context_attn = self.output_norm(self.proj_attn_1(context_attn))         # (B, W+1, D)

        rating_pid = context_attn[:, 0, :]          # (B, D)
        ratings_wishlist = context_attn[:, 1:, :]   # (B, W, D)

        # Attention over wishlist
        rating_wishlist, _ = self.wishlist_attention(pid_embed.unsqueeze(1), wishlist_embed, ratings_wishlist)  # (B, 1, D)
        rating_wishlist = self.proj_attn_2(rating_wishlist.squeeze(1))      # (B, D)

        # Fusion via gating
        gate = torch.sigmoid(self.gate_layer(torch.cat([sid_embed, pid_embed], dim=1)))
        fused = gate * rating_pid + (1 - gate) * rating_wishlist        # (B, D)

        bias = torch.sum(sid_embed * pid_embed, dim=-1, keepdim=True)   # (B,)

        x = torch.cat([bias, fused], dim=1)         # (B, D+1)
        x = F.relu(self.fc1(self.dropout(x)))       # (B, D)
        x = self.fc2(self.dropout(x)).squeeze(1)    # (B,)

        return x


class AttentionRecommenderPID(nn.Module):
    """
    Attention-based recommender that focuses on product behavior (PID-centric).
    """

    def __init__(self, num_sids: int, num_pids: int, emb_dim: int, dropout_rate: float, num_heads: int):
        super().__init__()

        # Embedding layers
        self.sid_embedding = nn.Embedding(num_sids + 1, emb_dim, padding_idx=-1)
        self.pid_embedding = nn.Embedding(num_pids, emb_dim)
        self.rating_embedding = nn.Embedding(6, emb_dim, padding_idx=0)

        # Layer norms for stabilization
        self.norm_sid = nn.LayerNorm(emb_dim)
        self.norm_pid = nn.LayerNorm(emb_dim)
        self.norm_context_sid = nn.LayerNorm(emb_dim)
        self.norm_rating = nn.LayerNorm(emb_dim)
        self.output_norm = nn.LayerNorm(emb_dim)

        # Attention layers
        self.context_attention = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.wishlist_attention = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout_rate, batch_first=True)

        # Projection layers
        self.proj_attn_1 = nn.Linear(emb_dim, emb_dim)
        self.proj_attn_2 = nn.Linear(emb_dim, emb_dim)
        
        self.dropout = nn.Dropout(dropout_rate)

        # Gating and final prediction layers
        self.gate_layer = nn.Linear(2 * emb_dim, emb_dim)
        self.fc1 = nn.Linear(emb_dim + 1, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 1)

    def forward(self, sid, pid, context_pid_ratings, wishlist):
        """
        Args:
            sid: (B,)
            pid: (B,)
            context_pid_ratings: (B, K, 2) where each entry is (sid, rating)
            wishlist: (B, W) 
        Returns:
            Predicted rating: (B,)
        """
        context_sids = context_pid_ratings[:, :, 0].long()      # (B, K)
        context_ratings = context_pid_ratings[:, :, 1].long()   # (B, K)

        # Embedding lookups + normalization
        sid_embed = self.norm_sid(self.sid_embedding(sid))      # (B, D)
        pid_embed = self.norm_pid(self.pid_embedding(pid))      # (B, D)
        wishlist_embed = self.norm_sid(self.sid_embedding(wishlist))    # (B, W, D)
        context_sid_embed = self.norm_context_sid(self.sid_embedding(context_sids))         # (B, K, D)
        context_rating_embed = self.norm_rating(self.rating_embedding(context_ratings))     # (B, K, D)

        # Multi-head attention over context (with queries: pid + wishlist)
        query = torch.cat([sid_embed.unsqueeze(1), wishlist_embed], dim=1)      # (B, W+1, D)
        context_attn, _ = self.context_attention(query, context_sid_embed, context_rating_embed)    # (B, W+1, D)
        context_attn = self.output_norm(self.proj_attn_1(context_attn))         # (B, W+1, D)

        rating_sid = context_attn[:, 0, :]          # (B, D)
        ratings_wishlist = context_attn[:, 1:, :]   # (B, W, D)

        # Attention over wishlist
        rating_wishlist, _ = self.wishlist_attention(sid_embed.unsqueeze(1), wishlist_embed, ratings_wishlist)  # (B, 1, D)
        rating_wishlist = self.proj_attn_2(rating_wishlist.squeeze(1))      # (B, D)

        # Fusion via gating
        gate = torch.sigmoid(self.gate_layer(torch.cat([sid_embed, pid_embed], dim=1)))
        fused = gate * rating_sid + (1 - gate) * rating_wishlist    # (B, D)

        bias = torch.sum(sid_embed * pid_embed, dim=-1, keepdim=True)   # (B,)
        
        x = torch.cat([bias, fused], dim=1)         # (B, D+1)
        x = F.relu(self.fc1(self.dropout(x)))       # (B, D)
        x = self.fc2(self.dropout(x)).squeeze(1)    # (B,)

        return x
    
    
def train_model(model, optim, device, epochs, train_loader, valid_loader):
    """
    Trains the model using MSE loss and evaluates it on validation data
    after each epoch. Outputs training loss and validation RMSE.
    """

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    for epoch in trange(epochs):
        model.train()
        total_loss = 0.0
        total_data = 0

        # Training loop
        for sid, pid, rating, context, wishlist in train_loader:
            # Move data to the target device
            sid, pid, rating = sid.to(device), pid.to(device), rating.to(device)
            context, wishlist = context.to(device), wishlist.to(device)

            # Forward pass and compute loss
            predictions = model(sid, pid, context, wishlist)
            loss = F.mse_loss(predictions, rating)

            # Backpropagation
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Accumulate loss
            total_loss += loss.item() * sid.size(0)
            total_data += sid.size(0)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        total_val_data = 0

        with torch.no_grad():
            for sid, pid, rating, context, wishlist in valid_loader:
                sid, pid, rating = sid.to(device), pid.to(device), rating.to(device)
                context, wishlist = context.to(device), wishlist.to(device)

                predictions = model(sid, pid, context, wishlist)
                mse = F.mse_loss(predictions, rating)

                total_val_loss += mse.item() * sid.size(0)
                total_val_data += sid.size(0)

        # Calculate training and validation RMSE
        train_rmse = (total_loss / total_data) ** 0.5
        val_rmse = (total_val_loss / total_val_data) ** 0.5

        print(f"[Epoch {epoch + 1}/{epochs}] Train RMSE={train_rmse:.3f}, Valid RMSE={val_rmse:.3f}")
        
    return train_rmse, val_rmse

class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Trainable parameters for the cross layer
        self.weight = nn.Parameter(torch.randn(input_dim))  # (D,)
        self.bias = nn.Parameter(torch.randn(input_dim))    # (D,)

    def forward(self, x0, x):
        """
        Cross interaction layer:
        x_{l+1} = x0 * (w^T x) + b + x
        """
        xw = torch.sum(x * self.weight, dim=1, keepdim=True)  # (B, 1)
        out = x0 * xw + self.bias + x    # (B, D)
        return out


class RecommenderFinal(nn.Module):
    def __init__(self, num_sids, num_pids, emb_dim, hidden_dim, num_cross_layers, dropout_rate):
        super().__init__()
        
        # Embedding layers for SID and PID
        self.sid_embedding = nn.Embedding(num_sids, emb_dim)
        self.pid_embedding = nn.Embedding(num_pids, emb_dim)

        # Input features: sid_emb + pid_emb + pred_sid + pred_pid
        input_dim = 2 * emb_dim + 2

        # Cross layers to model explicit feature interactions
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_cross_layers)
        ])

        # DNN
        self.deep = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Final regression output
        )

    def forward(self, sid, pid, pred_sid, pred_pid):
        """
        Forward pass through the final hybrid recommender.
        """
        sid_emb = self.sid_embedding(sid)  # (B, D)
        pid_emb = self.pid_embedding(pid)  # (B, D)

        # Concatenate raw features and intermediate predictions
        features = torch.cat([sid_emb, pid_emb, pred_sid.unsqueeze(1), pred_pid.unsqueeze(1)], dim=1)  # (B, 2D+2)

        # Pass through Cross Network
        x = features
        for layer in self.cross_layers:
            x = layer(features, x)

        # Pass through DNN
        out = self.deep(x).squeeze(-1)

        # Final output clamped to rating range
        out = torch.clamp(out, 1, 5)

        return out
    
def train(train_df, valid_df, tbr_df, device):
    model_sid = AttentionRecommenderSID(num_sids=NUM_SCIENTISTS, num_pids=NUM_PAPERS, emb_dim=EMBEDDING_DIM, dropout_rate=DROPOUT_RATE, num_heads=NUM_HEADS).to(device)
    optim_sid = torch.optim.Adam(model_sid.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

    model_pid = AttentionRecommenderPID(num_sids=NUM_SCIENTISTS, num_pids=NUM_PAPERS, emb_dim=EMBEDDING_DIM, dropout_rate=DROPOUT_RATE, num_heads=NUM_HEADS).to(device)
    optim_pid = torch.optim.Adam(model_pid.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
    
    train_dataset_sid = get_dataset_sid(df=train_df, wishlist_df=tbr_df, context_size=SID_CONTEXT_SIZE, wishlist_size=SID_WISHLIST_SIZE, save_path="data/sid_train_dataset")
    valid_dataset_sid = get_dataset_sid(df=valid_df, wishlist_df=tbr_df, context_size=SID_CONTEXT_SIZE, wishlist_size=SID_WISHLIST_SIZE, save_path="data/sid_valid_dataset")

    train_loader_sid = torch.utils.data.DataLoader(train_dataset_sid, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader_sid = torch.utils.data.DataLoader(valid_dataset_sid, batch_size=BATCH_SIZE, shuffle=False)

    train_dataset_pid = get_dataset_pid(df=train_df, wishlist_df=tbr_df, context_size=PID_CONTEXT_SIZE, wishlist_size=PID_WISHLIST_SIZE, save_path="data/pid_train_dataset")
    valid_dataset_pid = get_dataset_pid(df=valid_df, wishlist_df=tbr_df, context_size=PID_CONTEXT_SIZE, wishlist_size=PID_WISHLIST_SIZE, save_path="data/pid_valid_dataset")

    train_loader_pid = torch.utils.data.DataLoader(train_dataset_pid, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader_pid = torch.utils.data.DataLoader(valid_dataset_pid, batch_size=BATCH_SIZE, shuffle=False)
        
    train_rmse_sid, val_rmse_sid = train_model(model_sid, optim_sid, device, EPOCHS, train_loader_sid, valid_loader_sid)
    torch.save(model_sid.state_dict(), 'attention_recommender_sid.pt')
    
    save_predictions_csv(model_sid, train_loader_sid, valid_loader_sid, device, train_output_file="data/sid_train_predictions.csv", valid_output_file="data/sid_valid_predictions.csv")

    train_rmse_pid, val_rmse_pid = train_model(model_pid, optim_pid, device, EPOCHS, train_loader_pid, valid_loader_pid)
    torch.save(model_pid.state_dict(), 'attention_recommender_pid.pt')

    save_predictions_csv(model_pid, train_loader_pid, valid_loader_pid, device, train_output_file="data/pid_train_predictions.csv", valid_output_file="data/pid_valid_predictions.csv")

    train_dataset_combined = get_dataset_combined(rating_file="train_ratings.csv", pred_sid_file="sid_train_predictions.csv", pred_pid_file="pid_train_predictions.csv", save_path="combined_train_dataset")
    valid_dataset_combined = get_dataset_combined(rating_file="train_ratings.csv", pred_sid_file="sid_valid_predictions.csv", pred_pid_file="pid_valid_predictions.csv", save_path="combined_valid_dataset")

    train_loader_combined = torch.utils.data.DataLoader(train_dataset_combined, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader_combined = torch.utils.data.DataLoader(valid_dataset_combined, batch_size=BATCH_SIZE, shuffle=False)

    model_combined = RecommenderFinal(num_sids=NUM_SCIENTISTS, num_pids=NUM_PAPERS, emb_dim=64, hidden_dim=128, num_cross_layers=2, dropout_rate=0.2).to(device)
    optim_combined = torch.optim.Adam(model_combined.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)

    train_rmse, val_rmse = train_model(model_combined, optim_combined, device, EPOCHS, train_loader_combined, valid_loader_combined)
    torch.save(model_combined.state_dict(), 'attention_recommender_combined.pt')

    return model_combined, train_rmse_sid, train_rmse_pid, train_rmse, val_rmse_sid, val_rmse_pid, val_rmse

def run(args):
    mode, arg = args
    df, tbr_df = read_data_df()
    device = torch.device('mps' if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    train_df, valid_df = split_df(df, split_size= arg)
    model_combined, train_rmse_sid, train_rmse_pid, train_rmse, val_rmse_sid, val_rmse_pid, val_rmse = train(train_df, valid_df, tbr_df, device)
    print("Metrics: ", model_combined, train_rmse_sid, train_rmse_pid, train_rmse, val_rmse_sid, val_rmse_pid, val_rmse)
    
    if mode == 'predict':
        sub_df = read_sample_df()
        sub_loader = torch.utils.data.DataLoader(sub_df, batch_size=BATCH_SIZE, shuffle=False)
        save_predictions_csv(model_combined, sub_loader, None, device, train_output_file="submissions/submissions_attention_combined.csv", valid_output_file="")

