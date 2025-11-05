import os
import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
sys.path.append(os.getcwd())
from src.features.build_features import build_features
from torch.utils.data import Dataset, DataLoader
import torch

# -------- CONFIG --------
torch.manual_seed(42)
np.random.seed(42)

MLFLOW_TRACKING_URI = "mlruns"   
EXPERIMENT_NAME = "daily_forecasting_optuna"
N_TRIALS = 100
N_SPLITS = 5
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

#mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

train_data_path = "data/train_data.csv"

# -------- LOAD & PREPARE DATA --------
def load_data(path):

    # Load csv
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Build features
    daily_df = build_features(df)

    # get log 1p to help lstm
    daily_df['total_sales'] = np.log1p(daily_df['total_sales'])

    # Start at day 7 to avoid NaNs in lag7
    daily_df = daily_df.iloc[7:]

    # Fill NaN avg discounts
    daily_df['avg_discount'] = daily_df['avg_discount'].fillna(0)
    
    # Normalize data
    features = [f for f in daily_df.columns if f != 'datetime']
    data = daily_df[features].values

    return data


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





def train_model(data, params, n_splits=3):
    """
    daily_df : df of daily total sales
    params : dict with keys
        seq_length, num_epochs, hidden_size, num_layers, dropout, lr, batch_size
    """

    # Create sequences
    X, y = create_sequences(data, seq_length=params['seq_length'])
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_losses = []
    
    for train_idx, val_idx in tscv.split(X):
        # Fit scaler only on training part
        scaler = MinMaxScaler()
        scaler.fit(X[train_idx].reshape(-1, X.shape[2]))
        
        X_train = scaler.transform(X[train_idx].reshape(-1, X.shape[2]))
        X_val   = scaler.transform(X[val_idx].reshape(-1, X.shape[2]))

        # Reshape back
        X_train = X_train.reshape(len(train_idx), params["seq_length"], -1)
        X_val   = X_val.reshape(len(val_idx), params["seq_length"], -1)

        # Scale target separately (just to be clean)
        y_scaler = MinMaxScaler()
        y_train = y_scaler.fit_transform(y[train_idx].reshape(-1, 1)).flatten()
        y_val   = y_scaler.transform(y[val_idx].reshape(-1, 1)).flatten()

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
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
    
    # Return mean validation loss across folds
    mean_val_loss = np.mean(val_losses)
    return mean_val_loss


def train_final_model(data_scaled, best_params):

    # Create sequences with the best seq_length
    X, y = create_sequences(data_scaled, seq_length=best_params["seq_length"])

    train_dataset = SalesDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)

    # Initialize fresh model
    input_size = X.shape[2]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    # Train
    for epoch in range(best_params["num_epochs"]):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            weights = torch.where(y_batch > 0, 5.0, 1.0)
            weights = weights / weights.mean()
            loss = (weights * (y_pred - y_batch) ** 2).mean()
            loss.backward()
            optimizer.step()

    return model

def objective(trial):
    params = {
        "seq_length": trial.suggest_int("seq_length", 30, 90),
        "num_epochs": trial.suggest_int("num_epochs", 30, 75),
        "hidden_size": trial.suggest_int("hidden_size", 32, 128),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 10e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64])
    }
    
    val_loss = train_model(data_scaled, params, n_splits=3)

    with mlflow.start_run(nested =True):
        mlflow.log_params(params)
        score = val_loss
        mlflow.log_metric('val_loss',score)
        return score


if __name__ == "__main__":

    mlruns_path = os.path.abspath("./mlruns")  # in your project root
    mlflow.set_tracking_uri("file:///" + mlruns_path.replace("\\", "/"))
    # Load and preprocess data
    data_scaled = load_data(train_data_path)

    # Start a top-level MLflow run
    with mlflow.start_run(run_name="LSTM_runs"):
        # Create Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials= N_TRIALS)
    best_params = study.best_params

    print("Best hyperparameters:", best_params)


    # -------- Train final model --------

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_scaled)
    
    best_model = train_final_model(data_scaled, best_params)

    # Save model
    model_path = os.path.join(MODEL_DIR, "best_LSTM_model.pkl")
    joblib.dump(best_model, model_path)
    example_input = np.random.randn(1, best_params["seq_length"], data_scaled.shape[1]).astype(np.float32)

    # Log final run to MLflow
    with mlflow.start_run(run_name="best_model_LSTM"):
        mlflow.log_params(best_params)
        mlflow.pytorch.log_model(best_model, name="model", input_example = example_input)
        mlflow.log_artifact(model_path)

