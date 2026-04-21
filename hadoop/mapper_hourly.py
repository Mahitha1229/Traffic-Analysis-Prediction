#!/usr/bin/env python3
import sys
import csv

reader = csv.reader(sys.stdin)
next(reader, None)

for row in reader:
    try:
        date_time   = row[0]
        traffic_vol = int(row[8])
        hour        = date_time.split(" ")[1].split(":")[0]
        print(f"{hour}\t{traffic_vol}")
    except (IndexError, ValueError):
        continue
