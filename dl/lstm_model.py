import numpy as np
import pandas as pd
import json, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

OUTPUT_PATH = "../results/dl"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("="*60)
print(" LSTM MODEL (FINAL)")
print("="*60)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("../data/metro_traffic_big.csv")
df["date_time"] = pd.to_datetime(df["datetime"])

# SINGLE LOCATION (IMPORTANT)
df = df[df["location_id"] == df["location_id"].iloc[0]]
df = df.sort_values("date_time")

# ==============================
# FEATURES
# ==============================
df["hour"] = df["date_time"].dt.hour
df["day"]  = df["date_time"].dt.dayofweek

df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

features = ["traffic_volume","hour_sin","hour_cos","temp_celsius","rain_1h"]

for col in features:
    if col not in df.columns:
        df[col] = 0

data = df[features].values

# SCALE
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# ==============================
# CREATE SEQUENCE
# ==============================
SEQ_LEN = 24

X, y = [], []
for i in range(len(data)-SEQ_LEN):
    X.append(data[i:i+SEQ_LEN])
    y.append(data[i+SEQ_LEN][0])

X, y = np.array(X), np.array(y)

split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==============================
# MODEL
# ==============================
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(SEQ_LEN, X.shape[2])),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# ==============================
# PREDICT
# ==============================
y_pred = model.predict(X_test)

# inverse scale
y_test_inv = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), len(features)-1))], axis=1)
)[:,0]

y_pred_inv = scaler.inverse_transform(
    np.concatenate([y_pred, np.zeros((len(y_pred), len(features)-1))], axis=1)
)[:,0]

# ==============================
# METRICS (FIXED)
# ==============================
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae  = mean_absolute_error(y_test_inv, y_pred_inv)
r2   = r2_score(y_test_inv, y_pred_inv)

print(f"LSTM → RMSE:{rmse:.2f}  MAE:{mae:.2f}  R²:{r2:.4f}")

# SAVE
json.dump({
    "metrics": {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }
}, open(f"{OUTPUT_PATH}/lstm_results.json","w"), indent=2)

print("Saved → lstm_results.json")
