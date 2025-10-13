import pandas as pd
import numpy as np
import optuna
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib
import os
from src.features.build_features import build_features

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
    return daily_df

# -------- TRAIN FUNCTION --------
def train_model(df):
    df = df.copy()
    df.dropna(inplace = True)
    features = [c for c in df.columns if c not in ["datetime", "total_sales"]]
    X = df[features]
    y = df["total_sales"]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)


    with mlflow.start_run(nested=True):
        mlflow.set_tag("model_type", "linreg")
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        mae = -scores.mean()
        mae_std = scores.std()
        mlflow.log_metric("cv_mae", mae )
        mlflow.log_metric("cv_std", mae_std )
        model.fit(X, y)
        model_path = os.path.join(MODEL_DIR, "best_linreg_model.pkl")
        joblib.dump(model, model_path)

        return model

        


if __name__ == "__main__":
    df = load_data(train_data_path)
    model = train_model(df)