#!/usr/bin/env python3
import sys

current_hour = None
total = 0
count = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    hour, volume = line.split("\t")
    volume = int(volume)

    if current_hour == hour:
        total += volume
        count += 1
    else:
        if current_hour is not None:
            avg = round(total / count, 2)
            print(f"{current_hour}\t{avg}\t{count}")
        current_hour = hour
        total = volume
        count = 1

if current_hour is not None:
    avg = round(total / count, 2)
    print(f"{current_hour}\t{avg}\t{count}")
