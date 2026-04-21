#!/usr/bin/env python3
import sys
import csv
from datetime import datetime

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

reader = csv.reader(sys.stdin)
next(reader, None)

for row in reader:
    try:
        dt       = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        day_name = DAYS[dt.weekday()]
        hour     = str(dt.hour).zfill(2)
        key      = f"{day_name}_{hour}"
        print(f"{key}\t{int(row[8])}")
    except (IndexError, ValueError):
        continue
