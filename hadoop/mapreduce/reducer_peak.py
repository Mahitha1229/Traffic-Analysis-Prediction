#!/usr/bin/env python3
import sys
from collections import defaultdict

data = defaultdict(lambda: [0, 0])

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    key, volume = line.split("\t")
    data[key][0] += int(volume)
    data[key][1] += 1

weekday_peaks = defaultdict(list)
for key, (total, count) in data.items():
    day, hour = key.rsplit("_", 1)
    avg = round(total / count, 2)
    weekday_peaks[day].append((int(hour), avg))

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
print("Weekday\tPeak_Hour\tAvg_Traffic")
for day in DAYS:
    if day in weekday_peaks:
        peak = max(weekday_peaks[day], key=lambda x: x[1])
        print(f"{day}\t{peak[0]:02d}:00\t{peak[1]}")
