#!/bin/bash

HADOOP_HOME=${HADOOP_HOME:-~/hadoop}
HDFS_INPUT="/user/$USER/traffic/input"
HDFS_OUTPUT_HOURLY="/user/$USER/traffic/output_hourly"
HDFS_OUTPUT_PEAK="/user/$USER/traffic/output_peak"
LOCAL_DATA="../data/metro_traffic.csv"

echo "=== Traffic Analysis - Hadoop MapReduce Pipeline ==="

sudo service ssh start
start-dfs.sh
sleep 5

echo "[1] Setting up HDFS directories..."
hdfs dfs -mkdir -p $HDFS_INPUT
hdfs dfs -rm -r -f $HDFS_OUTPUT_HOURLY $HDFS_OUTPUT_PEAK

echo "[2] Uploading dataset to HDFS..."
hdfs dfs -put -f $LOCAL_DATA $HDFS_INPUT/metro_traffic.csv

echo "[3] Running Job 1: Hourly Traffic Average..."
mapred streaming \
    -input   $HDFS_INPUT/metro_traffic.csv \
    -output  $HDFS_OUTPUT_HOURLY \
    -mapper  "python3 mapper_hourly.py" \
    -reducer "python3 reducer_hourly.py" \
    -file    mapreduce/mapper_hourly.py \
    -file    mapreduce/reducer_hourly.py

echo "[4] Running Job 2: Peak Hour per Weekday..."
mapred streaming \
    -input   $HDFS_INPUT/metro_traffic.csv \
    -output  $HDFS_OUTPUT_PEAK \
    -mapper  "python3 mapper_peak.py" \
    -reducer "python3 reducer_peak.py" \
    -file    mapreduce/mapper_peak.py \
    -file    mapreduce/reducer_peak.py

echo "[5] Fetching results..."
mkdir -p ../results/hadoop
hdfs dfs -getmerge $HDFS_OUTPUT_HOURLY ../results/hadoop/hourly_traffic.tsv
hdfs dfs -getmerge $HDFS_OUTPUT_PEAK   ../results/hadoop/peak_hours.tsv

echo ""
echo "=== MapReduce Complete ==="
echo "Results in: results/hadoop/"
