import csv
import random
from datetime import datetime, timedelta

random.seed(42)

holidays = {
    "2023-01-01": "New Years Day",
    "2023-07-04": "Independence Day",
    "2023-11-23": "Thanksgiving Day",
    "2023-12-25": "Christmas Day",
    "2023-05-29": "Memorial Day",
    "2023-09-04": "Labor Day",
}

weather_conditions = [
    ("Clear",   "sky is clear",              0,    0,   10),
    ("Clouds",  "overcast clouds",           0,    0,   85),
    ("Rain",    "light rain",                2.5,  0,   70),
    ("Rain",    "moderate rain",             10,   0,   90),
    ("Snow",    "light snow",                0,    1.5, 80),
    ("Mist",    "mist",                      0,    0,   60),
    ("Fog",     "fog",                       0,    0,   40),
    ("Drizzle", "light intensity drizzle",   0.8,  0,   75),
]

def base_traffic(hour, is_weekend, is_holiday):
    if is_holiday:
        return random.randint(300, 800)
    if is_weekend:
        if 9 <= hour <= 18:
            return random.randint(1500, 3500)
        return random.randint(300, 1200)
    if 6 <= hour <= 9:
        return random.randint(4000, 6800)
    if 15 <= hour <= 18:
        return random.randint(4500, 7200)
    if 10 <= hour <= 14:
        return random.randint(2000, 4000)
    if 19 <= hour <= 22:
        return random.randint(1000, 2500)
    return random.randint(100, 800)

start = datetime(2022, 1, 1)
rows = []
current = start

for _ in range(17520):
    date_str = current.strftime("%Y-%m-%d")
    hour = current.hour
    is_weekend = current.weekday() >= 5
    is_holiday = date_str in holidays
    holiday_label = holidays.get(date_str, "None")

    season = current.month
    if season in [12, 1, 2]:
        temp = round(random.uniform(255, 268), 2)
    elif season in [3, 4, 5]:
        temp = round(random.uniform(268, 285), 2)
    elif season in [6, 7, 8]:
        temp = round(random.uniform(285, 305), 2)
    else:
        temp = round(random.uniform(270, 290), 2)

    weather = random.choice(weather_conditions)
    rain  = round(weather[2] * random.uniform(0.5, 2.0), 2)
    snow  = round(weather[3] * random.uniform(0.5, 2.0), 2)
    clouds = min(100, weather[4] + random.randint(-10, 10))

    traffic = base_traffic(hour, is_weekend, is_holiday)
    if weather[0] in ["Snow", "Fog"]:
        traffic = int(traffic * random.uniform(0.6, 0.85))
    elif weather[0] == "Rain":
        traffic = int(traffic * random.uniform(0.8, 0.95))

    rows.append([
        current.strftime("%Y-%m-%d %H:%M:%S"),
        holiday_label, temp, rain, snow, clouds,
        weather[0], weather[1], traffic
    ])
    current += timedelta(hours=1)

with open("metro_traffic.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date_time","holiday","temp","rain_1h","snow_1h",
                     "clouds_all","weather_main","weather_description","traffic_volume"])
    writer.writerows(rows)

print(f"Generated {len(rows)} rows -> metro_traffic.csv")
