import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.xgboost
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
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
    features = [c for c in df.columns if c not in ["datetime", "total_sales"]]
    X = df[features]
    y = df["total_sales"]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric":"mae",
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "random_state": 42,
        }

        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_type", "xgboost")
            mlflow.log_params(params)
            model = XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
            mae = -scores.mean()
            mae_std = scores.std()
            mlflow.log_metric("cv_mae", mae )
            mlflow.log_metric("cv_std", mae_std )
            return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    print("Best params:", best_params)

    # -------- Train final model --------
    best_model = XGBRegressor(**best_params)
    best_model.fit(X, y)

    # Save model
    model_path = os.path.join(MODEL_DIR, "best_xgboost_model.pkl")
    joblib.dump(best_model, model_path)

    # Log final run to MLflow
    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(best_params)
        mlflow.lightgbm.log_model(best_model, artifact_path="model")
        mlflow.log_artifact(model_path)
        mlflow.log_metric("final_mae", study.best_value)

    return best_model, best_params

if __name__ == "__main__":
    df = load_data(train_data_path)
    model, params = train_model(df)