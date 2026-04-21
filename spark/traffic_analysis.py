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
    .appName("TrafficAnalysis") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

DATA_PATH   = "../data/metro_traffic.csv"
OUTPUT_PATH = "../results/spark"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 60)
print("  TRAFFIC BIG DATA ANALYSIS — PySpark")
print("=" * 60)

# Load data
print("\n[1/6] Loading dataset...")
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

df = df \
    .withColumn("hour",        F.hour("date_time")) \
    .withColumn("day_of_week", F.dayofweek("date_time")) \
    .withColumn("month",       F.month("date_time")) \
    .withColumn("is_weekend",  (F.dayofweek("date_time").isin([1,7])).cast(IntegerType())) \
    .withColumn("is_peak",     (
        ((F.hour("date_time") >= 6)  & (F.hour("date_time") <= 9)) |
        ((F.hour("date_time") >= 15) & (F.hour("date_time") <= 18))
    ).cast(IntegerType())) \
    .withColumn("temp_celsius", F.round(F.col("temp") - 273.15, 2))

df.cache()
print(f"    Rows loaded: {df.count():,}")

DAY_MAP = {1:"Sunday",2:"Monday",3:"Tuesday",4:"Wednesday",
           5:"Thursday",6:"Friday",7:"Saturday"}

# Job 1: Hourly traffic
print("\n[2/6] Hourly Traffic Summary...")
hourly = df.groupBy("hour").agg(
    F.round(F.avg("traffic_volume"), 1).alias("avg_traffic"),
    F.max("traffic_volume").alias("max_traffic"),
    F.min("traffic_volume").alias("min_traffic"),
    F.count("*").alias("records")
).orderBy("hour")

hourly_list = [row.asDict() for row in hourly.collect()]
with open(f"{OUTPUT_PATH}/hourly_traffic.json", "w") as f:
    json.dump(hourly_list, f, indent=2)
print(f"    Saved hourly_traffic.json")

# Job 2: Peak hours per weekday
print("\n[3/6] Peak Hour per Weekday...")
peak_df = df.groupBy("day_of_week","hour").agg(
    F.round(F.avg("traffic_volume"), 1).alias("avg_traffic")
)
w = Window.partitionBy("day_of_week").orderBy(F.desc("avg_traffic"))
peak_hours = peak_df.withColumn("rn", F.row_number().over(w)) \
    .filter(F.col("rn") == 1).drop("rn").orderBy("day_of_week")

peak_list = []
for row in peak_hours.collect():
    d = row.asDict()
    d["day_name"] = DAY_MAP.get(d["day_of_week"], "?")
    peak_list.append(d)

with open(f"{OUTPUT_PATH}/peak_hours.json", "w") as f:
    json.dump(peak_list, f, indent=2)
print(f"    Saved peak_hours.json")
for p in peak_list:
    print(f"      {p['day_name']:12s}: Peak at {p['hour']:02d}:00  avg {p['avg_traffic']:.0f}")

# Job 3: Weather impact
print("\n[4/6] Weather Impact...")
weather_impact = df.groupBy("weather_main").agg(
    F.round(F.avg("traffic_volume"), 1).alias("avg_traffic"),
    F.count("*").alias("occurrences")
).orderBy(F.desc("avg_traffic"))

weather_list = [row.asDict() for row in weather_impact.collect()]
with open(f"{OUTPUT_PATH}/weather_impact.json", "w") as f:
    json.dump(weather_list, f, indent=2)
print(f"    Saved weather_impact.json")

# Job 4: Heatmap
print("\n[5/6] Congestion Heatmap...")
heatmap = df.groupBy("day_of_week","hour").agg(
    F.round(F.avg("traffic_volume"), 0).alias("avg_traffic")
).orderBy("day_of_week","hour")

heatmap_list = []
for row in heatmap.collect():
    d = row.asDict()
    d["day_name"] = DAY_MAP.get(d["day_of_week"], "?")
    v = d["avg_traffic"]
    d["congestion"] = "High" if v > 4500 else ("Medium" if v > 2000 else "Low")
    heatmap_list.append(d)

with open(f"{OUTPUT_PATH}/heatmap.json", "w") as f:
    json.dump(heatmap_list, f, indent=2)
print(f"    Saved heatmap.json")

# Job 5: ML Prediction
print("\n[6/6] Random Forest Prediction...")
indexer   = StringIndexer(inputCol="weather_main", outputCol="weather_idx", handleInvalid="keep")
features  = ["hour","day_of_week","month","is_weekend","is_peak",
             "temp_celsius","rain_1h","snow_1h","clouds_all","weather_idx"]
assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="keep")
rf        = RandomForestRegressor(featuresCol="features", labelCol="traffic_volume",
                                  numTrees=50, maxDepth=8, seed=42)
pipeline  = Pipeline(stages=[indexer, assembler, rf])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"    Training: {train_df.count():,} | Test: {test_df.count():,}")

model       = pipeline.fit(train_df)
predictions = model.transform(test_df)

rmse = RegressionEvaluator(labelCol="traffic_volume", predictionCol="prediction",
                            metricName="rmse").evaluate(predictions)
r2   = RegressionEvaluator(labelCol="traffic_volume", predictionCol="prediction",
                            metricName="r2").evaluate(predictions)

print(f"    RMSE : {rmse:.2f}")
print(f"    R²   : {r2:.4f}")

rf_model    = model.stages[-1]
importances = list(zip(features, rf_model.featureImportances.toArray()))
importances.sort(key=lambda x: x[1], reverse=True)

metrics = {
    "rmse": round(rmse, 2),
    "r2":   round(r2, 4),
    "model": "RandomForestRegressor",
    "num_trees": 50,
    "total_rows": df.count(),
    "feature_importances": [
        {"feature": f, "importance": round(float(i), 4)}
        for f, i in importances
    ]
}
with open(f"{OUTPUT_PATH}/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"    Saved model_metrics.json")

print("\n" + "=" * 60)
print("  SPARK ANALYSIS COMPLETE")
print("=" * 60)
spark.stop()
