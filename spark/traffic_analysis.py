"""
PySpark Traffic Analysis — BIG Dataset
========================================
Dataset: metro_traffic_big.csv
Records: 175,320 rows · 5 locations · 4 years · 37 features

Run:
  spark-submit --master local[*] --driver-memory 4g traffic_analysis_big.py
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
import json, os

spark = SparkSession.builder \
    .appName("TrafficBDA_Big") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

DATA_PATH   = "../data/metro_traffic_big.csv"
OUTPUT_PATH = "../results/spark"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 65)
print("  TRAFFIC BDA — BIG DATASET — PySpark")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────────
print("\n[1/7] Loading dataset...")
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
df.cache()
total = df.count()
locs  = df.select("location_id").distinct().count()
print(f"    Rows      : {total:,}")
print(f"    Locations : {locs}")
print(f"    Columns   : {len(df.columns)}")
df.printSchema()

# ── Job 1: Overall Hourly Average ────────────────────────────
print("\n[2/7] Job 1: Hourly Traffic Average...")
hourly = df.groupBy("hour").agg(
    F.round(F.avg("traffic_volume"),1).alias("avg_traffic"),
    F.round(F.avg("avg_speed_kmh"),1).alias("avg_speed"),
    F.round(F.avg("occupancy_pct"),1).alias("avg_occ"),
    F.max("traffic_volume").alias("max_traffic"),
    F.count("*").alias("records")
).orderBy("hour")

hourly_list = [r.asDict() for r in hourly.collect()]
with open(f"{OUTPUT_PATH}/hourly_traffic.json","w") as f:
    json.dump(hourly_list, f, indent=2)
print(f"    Saved hourly_traffic.json ({len(hourly_list)} rows)")

# ── Job 2: Peak Hours per Weekday ────────────────────────────
print("\n[3/7] Job 2: Peak Hour per Weekday...")
DAY_MAP={0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
peak_df = df.groupBy("day_of_week","hour").agg(
    F.round(F.avg("traffic_volume"),1).alias("avg_traffic")
)
w = Window.partitionBy("day_of_week").orderBy(F.desc("avg_traffic"))
peak_hours = peak_df.withColumn("rn",F.row_number().over(w)) \
    .filter(F.col("rn")==1).drop("rn").orderBy("day_of_week")

peak_list = []
for r in peak_hours.collect():
    d = r.asDict()
    d["day_name"] = DAY_MAP.get(d["day_of_week"],"?")
    peak_list.append(d)
with open(f"{OUTPUT_PATH}/peak_hours.json","w") as f:
    json.dump(peak_list, f, indent=2)
print(f"    Saved peak_hours.json")
for p in peak_list:
    print(f"      {p['day_name']:12s}: {p['hour']:02d}:00  avg {p['avg_traffic']:.0f}")

# ── Job 3: Per-Location Analysis ─────────────────────────────
print("\n[4/7] Job 3: Per-Location Stats...")
loc_stats = df.groupBy("location_id","location_name").agg(
    F.round(F.avg("traffic_volume"),1).alias("avg_traffic"),
    F.max("traffic_volume").alias("max_traffic"),
    F.round(F.avg("avg_speed_kmh"),1).alias("avg_speed"),
    F.round(F.avg("occupancy_pct"),1).alias("avg_occupancy"),
    F.sum("incident_flag").alias("total_incidents"),
    F.count("*").alias("records")
).orderBy(F.desc("avg_traffic"))

loc_list = [r.asDict() for r in loc_stats.collect()]
with open(f"{OUTPUT_PATH}/location_stats.json","w") as f:
    json.dump(loc_list, f, indent=2)
print(f"    Saved location_stats.json")
for l in loc_list:
    print(f"      {l['location_name']}: avg {l['avg_traffic']:.0f}  incidents {l['total_incidents']}")

# ── Job 4: Weather Impact ────────────────────────────────────
print("\n[5/7] Job 4: Weather Impact...")
weather = df.groupBy("weather_main").agg(
    F.round(F.avg("traffic_volume"),1).alias("avg_traffic"),
    F.round(F.avg("avg_speed_kmh"),1).alias("avg_speed"),
    F.count("*").alias("occurrences")
).orderBy(F.desc("avg_traffic"))

weather_list = [r.asDict() for r in weather.collect()]
with open(f"{OUTPUT_PATH}/weather_impact.json","w") as f:
    json.dump(weather_list, f, indent=2)
print(f"    Saved weather_impact.json ({len(weather_list)} conditions)")

# ── Job 5: Heatmap ───────────────────────────────────────────
print("\n[6/7] Job 5: Congestion Heatmap...")
heatmap = df.groupBy("day_of_week","hour").agg(
    F.round(F.avg("traffic_volume"),0).alias("avg_traffic"),
    F.round(F.avg("occupancy_pct"),1).alias("avg_occ")
).orderBy("day_of_week","hour")

hm_list = []
for r in heatmap.collect():
    d = r.asDict()
    d["day_name"] = DAY_MAP.get(d["day_of_week"],"?")
    v = d["avg_traffic"]
    d["congestion"] = "High" if v>4500 else ("Medium" if v>2000 else "Low")
    hm_list.append(d)
with open(f"{OUTPUT_PATH}/heatmap.json","w") as f:
    json.dump(hm_list, f, indent=2)
print(f"    Saved heatmap.json ({len(hm_list)} cells)")

# ── Job 6: Random Forest ML ──────────────────────────────────
print("\n[7/7] Job 6: Random Forest ML...")

loc_idx     = StringIndexer(inputCol="location_id",  outputCol="loc_idx",     handleInvalid="keep")
weather_idx = StringIndexer(inputCol="weather_main", outputCol="weather_idx", handleInvalid="keep")
season_idx  = StringIndexer(inputCol="season",       outputCol="season_idx",  handleInvalid="keep")

features = ["hour","day_of_week","month","is_weekend","is_peak_hour","is_night",
            "temp_celsius","rain_1h","snow_1h","clouds_all","humidity","wind_speed",
            "lanes","weather_idx","loc_idx","season_idx"]

assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="keep")
rf = RandomForestRegressor(featuresCol="features", labelCol="traffic_volume",
                           numTrees=50, maxDepth=8, seed=42)
pipeline = Pipeline(stages=[loc_idx, weather_idx, season_idx, assembler, rf])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"    Train: {train_df.count():,}  Test: {test_df.count():,}")

model       = pipeline.fit(train_df)
predictions = model.transform(test_df)

rmse = RegressionEvaluator(labelCol="traffic_volume",predictionCol="prediction",metricName="rmse").evaluate(predictions)
r2   = RegressionEvaluator(labelCol="traffic_volume",predictionCol="prediction",metricName="r2").evaluate(predictions)
mae  = RegressionEvaluator(labelCol="traffic_volume",predictionCol="prediction",metricName="mae").evaluate(predictions)

print(f"    RMSE: {rmse:.2f}   MAE: {mae:.2f}   R²: {r2:.4f}")

rf_model = model.stages[-1]
fi = list(zip(features, rf_model.featureImportances.toArray()))
fi.sort(key=lambda x: x[1], reverse=True)

metrics = {
    "rmse": round(rmse,2), "mae": round(mae,2), "r2": round(r2,4),
    "model": "RandomForestRegressor", "num_trees":50,
    "total_rows": total, "locations": locs,
    "feature_importances": [{"feature":f,"importance":round(float(i),4)} for f,i in fi]
}
with open(f"{OUTPUT_PATH}/model_metrics.json","w") as f:
    json.dump(metrics, f, indent=2)
print(f"    Saved model_metrics.json")

# Summary stats
summary = {
    "total_records": total,
    "locations": locs,
    "years": 4,
    "features": len(df.columns),
    "size_justification": "175,320 records across 5 highway locations and 4 years — sufficient for distributed BDA with Hadoop HDFS + Spark"
}
with open(f"{OUTPUT_PATH}/dataset_summary.json","w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 65)
print("  SPARK ANALYSIS COMPLETE")
print(f"  Total records processed: {total:,}")
print("=" * 65)
spark.stop()
