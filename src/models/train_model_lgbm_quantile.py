#%%
import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import joblib
import os
from src.features.build_features import build_features
from sklearn.metrics import make_scorer

#%%
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


#%%
# -------- TRAIN FUNCTION --------
def train_model(df):
    df = df.copy()
    features = [c for c in df.columns if c not in ["datetime", "total_sales"]]
    X = df[features]
    y = df["total_sales"]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def quantile_loss(y_true, y_pred, alpha):
        diff = y_true - y_pred
        return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))
    quantile_scorer = make_scorer(lambda y_true, y_pred: -quantile_loss(y_true, y_pred, alpha=0.5))

    def objective(trial):
        params = {
            "objective": "quantile",
            "metric": "quantile",
            "alpha": 0.5,
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "random_state": 42,
        }

        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_type", "lightGBM_q50")
            mlflow.log_params(params)
            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(model, X, y, cv=tscv, scoring=quantile_scorer)
            mae = -scores.mean()
            mlflow.log_metric("cv_mae", mae)
            return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    print("Best params:", best_params)

    # -------- Train final model --------
    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(X, y)

    # Save model
    model_path = os.path.join(MODEL_DIR, "best_lgbm_q50_model.pkl")
    joblib.dump(best_model, model_path)

    # Log final run to MLflow
    with mlflow.start_run(run_name="best_model_lgbm_q50"):
        mlflow.log_params(best_params)
        mlflow.lightgbm.log_model(best_model, artifact_path="model")
        mlflow.log_artifact(model_path)
        mlflow.log_metric("final_mae", study.best_value)


    # Train alpha = 0.1 model with best params

    params_q10 = best_params.copy()
    params_q10.update({"objective": "quantile", "metric": "quantile"})
    params_q10['alpha'] = 0.1
    model_q10 = lgb.LGBMRegressor(**params_q10)
    model_q10.fit(X, y)
    # Save model
    model_path = os.path.join(MODEL_DIR, "best_lgbm_q10_model.pkl")
    joblib.dump(model_q10, model_path)
    # Log  to MLflow
    with mlflow.start_run(run_name="best_model_q10"):
        mlflow.log_params(params_q10)
        mlflow.lightgbm.log_model(model_q10, artifact_path="model")
        mlflow.log_artifact(model_path)

    # Train alpha = 0.9 model with best params

    params_q90 = best_params.copy()
    params_q90.update({"objective": "quantile", "metric": "quantile"})
    params_q90['alpha'] = 0.9
    model_q90 = lgb.LGBMRegressor(**params_q90)
    model_q90.fit(X, y)
    # Save model
    model_path = os.path.join(MODEL_DIR, "best_lgbm_q90_model.pkl")
    joblib.dump(model_q90, model_path)
    # Log  to MLflow
    with mlflow.start_run(run_name="best_model_q90"):
        mlflow.log_params(params_q90)
        mlflow.lightgbm.log_model(model_q90, artifact_path="model")
        mlflow.log_artifact(model_path)




    return best_model, best_params

if __name__ == "__main__":
    df = load_data(train_data_path)
    model, params = train_model(df)