import numpy as np
import pandas as pd
import json, os
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

OUTPUT_PATH = "../results/dl"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("="*60)
print(" AUTOENCODER (ANOMALY DETECTION)")
print("="*60)

df = pd.read_csv("../data/metro_traffic_big.csv")
df["date_time"] = pd.to_datetime(df["datetime"])

df = df[df["location_id"] == df["location_id"].iloc[0]]

features = ["traffic_volume","temp_celsius","rain_1h"]

for col in features:
    if col not in df.columns:
        df[col] = 0

data = df[features].values

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# ==============================
# MODEL
# ==============================
input_dim = data.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(data, data, epochs=10, batch_size=128, verbose=1)

# ==============================
# ANOMALY DETECTION
# ==============================
recon = autoencoder.predict(data)
error = np.mean((data - recon)**2, axis=1)

threshold = np.percentile(error, 95)
anomalies = np.sum(error > threshold)

print(f"Anomalies detected: {anomalies}")

json.dump({
    "anomalies": int(anomalies)
}, open(f"{OUTPUT_PATH}/autoencoder_results.json","w"), indent=2)

print("Saved → autoencoder_results.json")
