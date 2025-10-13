#%%
import pandas as pd
import os
import pickle
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.build_features import build_features, build_features_test_data



test_data_path = "../../data/test_data.csv"
train_data_path = "../../data/train_data.csv"


# -------- LOAD & PREPARE DATA --------
df = pd.read_csv(train_data_path)
df['datetime'] = pd.to_datetime(df['datetime'])
train_df = build_features(df)

df = pd.read_csv(test_data_path)
df['datetime'] = pd.to_datetime(df['datetime'])
test_df = build_features_test_data(df)


# Import models
model_path = '../../models/best_lgbm_q50_model.pkl'
model_q50 = pickle.load(open(model_path, "rb"))
model_path = '../../models/best_lgbm_q10_model.pkl'
model_q10 = pickle.load(open(model_path, "rb"))
model_path = '../../models/best_lgbm_q90_model.pkl'
model_q90 = pickle.load(open(model_path, "rb"))

# lag features ffor first test day

test_df['lag1'] = np.nan
test_df.loc[0,'lag1'] = train_df.loc[train_df.index[-1],'total_sales']
test_df['lag7'] = np.nan
test_df.loc[0,'lag7'] = train_df.loc[train_df.index[-7],'total_sales']

#%% Sequential predictions

test_df['forecast'] = np.nan
test_df['forecast_q10'] = np.nan
test_df['forecast_q90'] = np.nan

features = [c for c in test_df.columns if c not in ["datetime", "total_sales", 'forecast', 'forecast_q10','forecast_q90']]

for i in test_df.index:

    if i-7 in test_df.index:
        test_df.loc[i, 'lag7'] = test_df.loc[i-7, 'forecast']
    else:
        test_df.loc[i, 'lag7'] = train_df.loc[train_df.index[-7 + i],'total_sales']

    X = test_df.loc[[i],features]
    y_pred = model_q50.predict(X)
    y_q10 = model_q10.predict(X)
    y_q90 = model_q90.predict(X)

    test_df.loc[i, 'forecast'] = max(y_pred,0)
    test_df.loc[i, 'forecast_q10'] = max(y_q10,0)
    test_df.loc[i, 'forecast_q90'] = max(y_q90,0)


    if i+1 in test_df.index:
        test_df.loc[i+1, 'lag1'] = max(y_pred,0)


#%% Plot
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
sns.set_theme()

# Shaded quantile interval
plt.fill_between(test_df["datetime"],test_df['forecast_q10'], test_df['forecast_q90'], color="skyblue", alpha=0.3,  label="10–90% prediction interval")

# Median prediction
plt.plot(test_df["datetime"], test_df['forecast'], color="blue", label="Predicted median (q=0.5)", linewidth=1)

# Actuals
plt.plot(test_df["datetime"], test_df['total_sales'], color="black", label="Actual sales", linewidth=1, alpha=0.8)

plt.xlabel("Date")
plt.ylabel("Daily Sales")
plt.title("Quantile Regression Forecast with 10–90% Interval")
plt.xlim([datetime.datetime(2017,8,1), datetime.datetime(2017,11,1) ])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%

from sklearn.metrics import mean_absolute_error
import numpy as np

y = test_df['total_sales']
pred_q50 = test_df['forecast']

mae = mean_absolute_error(y, pred_q50)
smape = 100 * np.mean(2 * np.abs(y - pred_q50) / (np.abs(y) + np.abs(pred_q50) + 1e-8))

print(f"MAE: {mae:.2f}")
print(f"MAPE: {smape:.2f}%")

