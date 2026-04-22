# Traffic Analysis Prediction

End-to-end Big Data + Deep Learning project for metro traffic forecasting and anomaly detection.

This project combines:
- Hadoop MapReduce for distributed batch jobs
- Apache Spark for fast analytics and ML (Random Forest)
- Deep Learning models (LSTM, Autoencoder, DNN comparison)
- Flask REST API
- Interactive dashboard

---

## Project Goals

- Analyze hourly traffic behavior by time, day, and weather.
- Predict traffic volume using classical ML and deep learning.
- Detect unusual traffic events (anomalies) without labels.
- Serve results through API endpoints and visualize them in a dashboard.

---

## Project Structure

```text
Traffic-Analysis-Prediction/
├── api/
│   └── app.py
├── dashboard/
│   └── index.html
├── data/
│   ├── metro_traffic.csv
│   └── generate_dataset.py
├── dl/
│   ├── lstm_model.py
│   ├── autoencoder.py
│   └── compare_models.py
├── hadoop/
│   ├── run_hadoop.sh
│   ├── mapreduce/
│   │   ├── mapper_hourly.py
│   │   ├── reducer_hourly.py
│   │   ├── mapper_peak.py
│   │   └── reducer_peak.py
│   ├── mapper_hourly.py
│   ├── reducer_hourly.py
│   ├── mapper_peak.py
│   └── reducer_peak.py
├── spark/
│   └── traffic_analysis.py
└── results/
    ├── spark/
    ├── hadoop/
    └── dl/
```

---

## Tech Stack

- Python 3.x
- Apache Hadoop 3.x (HDFS + Streaming MapReduce)
- Apache Spark 3.x (PySpark + MLlib)
- Flask + Flask-CORS
- TensorFlow/Keras (optional but recommended for real DL training)
- scikit-learn, pandas, numpy, matplotlib

---

## Setup

From project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install flask flask-cors pyspark pandas numpy scikit-learn matplotlib tensorflow scipy
```

If TensorFlow is not installed, DL scripts still run in simulated mode and generate fallback outputs for dashboard/API use.

---

## Data

### Use existing dataset

`data/metro_traffic.csv` is already included.

### Regenerate dataset (optional)

```bash
cd data
python3 generate_dataset.py
```

This generates 17,520 hourly records with weather and traffic patterns.

---

## Run Modules

### 1) Spark analytics + MLlib Random Forest

```bash
cd spark
python3 traffic_analysis.py
```

Outputs (under `results/spark/`):
- `hourly_traffic.json`
- `peak_hours.json`
- `weather_impact.json`
- `heatmap.json`
- `model_metrics.json`

### 2) Hadoop MapReduce jobs

```bash
cd hadoop
chmod +x run_hadoop.sh
./run_hadoop.sh
```

Outputs (under `results/hadoop/`):
- `hourly_traffic.tsv`
- `peak_hours.tsv`

### 3) Deep Learning models

Run from `dl/` directory:

```bash
python3 lstm_model.py
python3 autoencoder.py
python3 compare_models.py
```

Outputs (under `results/dl/`):
- `lstm_results.json`
- `autoencoder_results.json`
- `model_comparison.json`
- `lstm_model.keras` (if TensorFlow available)
- `autoencoder_model.keras` (if TensorFlow available)

### 4) Start API + Dashboard

```bash
cd api
python3 app.py
```

Open: `http://localhost:5000`

---

## API Endpoints

### Core analytics

- `GET /api/hourly` - hourly traffic summary
- `GET /api/peaks` - peak hour per day
- `GET /api/weather` - weather impact on traffic
- `GET /api/heatmap` - day-hour congestion grid
- `GET /api/metrics` - Spark ML model metrics
- `GET /api/summary` - dashboard summary stats
- `GET /api/predict?hour=8&dow=2&weather=Clear` - RF-style prediction

### Deep learning

- `GET /api/dl/lstm` - LSTM training results
- `GET /api/dl/autoencoder` - autoencoder anomaly results
- `GET /api/dl/comparison` - RF vs DNN vs LSTM comparison
- `GET /api/dl/predict?hour=8&dow=2&weather=Clear` - LSTM-style prediction
- `GET /api/dl/anomalies` - top anomalies
- `GET /api/dl/status` - DL output/model file availability

---

## Recommended Execution Order

1. Run Spark pipeline (`spark/traffic_analysis.py`)
2. Run DL models (`dl/lstm_model.py`, `dl/autoencoder.py`, `dl/compare_models.py`)
3. Start Flask API (`api/app.py`)
4. Open dashboard at `http://localhost:5000`
5. (Optional) Run Hadoop jobs for MapReduce outputs

---

## Notes

- The API includes fallback data so dashboard endpoints still respond even if result files are missing.
- Real model quality improves when TensorFlow training runs fully (instead of simulation mode).
- Paths in scripts are relative; run each script from its own folder as shown above.