#%%
import pandas as pd
import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mlflow.pytorch
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.build_features import build_features


SEQ_LENGTH = 40
test_data_path = "data/test_data.csv"
train_data_path = "data/train_data.csv"

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length,0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# -------- LOAD & PREPARE DATA --------
train_df = pd.read_csv(train_data_path)
train_df['datetime'] = pd.to_datetime(train_df['datetime'])
test_df = pd.read_csv(test_data_path)
test_df['datetime'] = pd.to_datetime(test_df['datetime'])
test_beginning = test_df['datetime'].min()
combined_df = pd.concat([train_df, test_df])

processed_df = build_features(combined_df)
processed_df['total_sales'] = np.log1p(processed_df['total_sales'])

test_df = processed_df[processed_df['datetime'].dt.date >= test_beginning.date()]
train_df = processed_df[processed_df['datetime'].dt.date < test_beginning.date()]
train_df = train_df.iloc[7:]
features = [f for f in train_df.columns if f != 'datetime']

train_data = train_df[features].values
test_data = test_df[features].values

# Normalize
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

data_combined = np.vstack([train_data_scaled[-SEQ_LENGTH:], test_data_scaled])


X_test, y_test = create_sequences(data_combined, SEQ_LENGTH)



predictions = []
true_values = []

# Load model
mlruns_path = r"C:\Users\Franc\OneDrive\Bureau\Projets\Forecasting\mlruns"
mlflow.set_tracking_uri("file:///" + mlruns_path.replace("\\", "/"))
mlflow.set_experiment("daily_forecasting_optuna")
model = mlflow.pytorch.load_model("runs:/1ef5919aa6134ee09b2293e0a1abd294/model")

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred = model(X_test_tensor).squeeze().numpy()


# re-scale

y_test_actual = np.expm1(y_test*scaler.data_range_[0] + scaler.data_min_[0]) 
y_pred_actual = np.expm1(y_pred*scaler.data_range_[0] + scaler.data_min_[0])


# Performance
from sklearn.metrics import mean_absolute_error


mae = mean_absolute_error(y_test_actual, y_pred_actual)
smape = 100 * np.mean(2 * np.abs(y_test_actual - y_pred_actual) / (np.abs(y_test_actual) + np.abs(y_pred_actual) + 1e-8))

print(f"MAE: {mae:.2f}")
print(f"MAPE: {smape:.2f}%")



# plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

ax = plt.plot(y_test_actual, label='Actual')
plt.plot(y_pred_actual, label='Predicted')
plt.legend()

plt.ylabel('Total Sales ($)')
plt.xlabel('Day')
plt.title('Daily Sales Forecast with LSTM')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
textstr = f"MAE: {mae:.2f}"
plt.text(0.95, 5500, textstr, fontsize=14,
        verticalalignment='top', bbox=props)

plt.show()


