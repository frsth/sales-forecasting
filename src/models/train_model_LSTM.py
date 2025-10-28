import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from src.features.build_features import build_features
from torch.utils.data import Dataset, DataLoader
import torch

# -------- CONFIG --------
MLFLOW_TRACKING_URI = "mlruns"   
EXPERIMENT_NAME = "daily_forecasting_optuna"
N_TRIALS = 50
N_SPLITS = 5
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

train_data_path = "data/train_data.csv"

# -------- LOAD & PREPARE DATA --------
def load_data(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    daily_df = build_features(df)
    # get log 1p to help lstm
    daily_df['total_sales'] = np.log1p(daily_df['total_sales'])
    # Start at day 7 to avoid NaNs in lag7
    daily_df = daily_df.iloc[7:]
    return daily_df


# --- Define model class --- #
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, output_size=1, dropout = 0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Class for dataloader
class SalesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

# Function to create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length,0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)





def train_model(data_scaled, params, n_splits=3):
    """
    data_scaled : np.array of shape (num_samples, num_features)
    params : dict with keys
        seq_length, num_epochs, hidden_size, num_layers, dropout, lr, batch_size
    """
    # Create sequences
    X, y = create_sequences(data_scaled, seq_length=params['seq_length'])
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_losses = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_dataset = SalesDataset(X_train, y_train)
        val_dataset   = SalesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Initialize fresh model per fold
        input_size = X_train.shape[2]
        model = LSTMModel(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        
        # Training loop
        for epoch in range(params['num_epochs']):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch).squeeze()
                weights = torch.where(y_batch > 0, 5.0, 1.0)
                weights = weights / weights.mean()
                loss = (weights * (y_pred.squeeze() - y_batch) ** 2).mean()
                loss.backward()
                optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch).squeeze()
                weights = torch.where(y_batch > 0, 5.0, 1.0)
                weights = weights / weights.mean()
                loss = (weights * (y_pred.squeeze() - y_batch) ** 2).mean()
                val_loss += loss.mean().item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
    
    # Return mean validation loss across folds
    mean_val_loss = np.mean(val_losses)
    return mean_val_loss