import csv, random, math
from datetime import datetime, timedelta

random.seed(42)

# 5 highway locations
LOCATIONS = [
    {"id":"I-94_MN",  "name":"I-94 Minnesota",    "lat":44.97, "lon":-93.26, "lanes":4, "speed_limit":65},
    {"id":"I-35W_MN", "name":"I-35W Minneapolis",  "lat":44.88, "lon":-93.22, "lanes":3, "speed_limit":60},
    {"id":"I-494_MN", "name":"I-494 Bloomington",  "lat":44.85, "lon":-93.38, "lanes":4, "speed_limit":65},
    {"id":"US-52_MN", "name":"US-52 Saint Paul",   "lat":44.94, "lon":-93.07, "lanes":2, "speed_limit":55},
    {"id":"MN-36_MN", "name":"MN-36 Oakdale",      "lat":44.99, "lon":-92.96, "lanes":2, "speed_limit":55},
]

WEATHER_CONDITIONS = [
    ("Clear",   "sky is clear",              0,    0,   10,  1.00),
    ("Clear",   "few clouds",                0,    0,   20,  0.98),
    ("Clouds",  "scattered clouds",          0,    0,   40,  0.97),
    ("Clouds",  "overcast clouds",           0,    0,   85,  0.95),
    ("Rain",    "light rain",                2.5,  0,   70,  0.88),
    ("Rain",    "moderate rain",             10,   0,   90,  0.82),
    ("Rain",    "heavy rain",                20,   0,   95,  0.75),
    ("Drizzle", "light intensity drizzle",   0.8,  0,   75,  0.91),
    ("Snow",    "light snow",                0,    1.5, 80,  0.72),
    ("Snow",    "moderate snow",             0,    5,   90,  0.60),
    ("Snow",    "heavy snow",                0,    12,  95,  0.48),
    ("Mist",    "mist",                      0,    0,   60,  0.93),
    ("Fog",     "fog",                       0,    0,   40,  0.65),
    ("Fog",     "dense fog",                 0,    0,   20,  0.50),
    ("Thunderstorm","thunderstorm",          15,   0,   95,  0.70),
    ("Haze",    "haze",                      0,    0,   50,  0.88),
]

HOLIDAYS = {
    "01-01":("New Years Day",    0.30),
    "07-04":("Independence Day", 0.45),
    "11-11":("Veterans Day",     0.70),
    "12-25":("Christmas Day",    0.25),
    "12-24":("Christmas Eve",    0.55),
    "11-27":("Thanksgiving",     0.35),
    "11-28":("Thanksgiving",     0.35),
    "11-29":("Black Friday",     0.85),
    "05-27":("Memorial Day",     0.40),
    "09-02":("Labor Day",        0.40),
    "01-15":("MLK Day",          0.60),
    "02-19":("Presidents Day",   0.65),
}

def base_traffic(hour, weekday, loc_lanes, is_holiday, holiday_factor):
    cap = loc_lanes * 1800
    if is_holiday:
        return int(cap * holiday_factor * random.uniform(0.6, 0.9))
    if weekday >= 5:  # weekend
        if 10 <= hour <= 18:
            return int(cap * random.uniform(0.35, 0.55))
        elif 8 <= hour <= 21:
            return int(cap * random.uniform(0.20, 0.40))
        else:
            return int(cap * random.uniform(0.04, 0.10))
    # Weekday
    if 6 <= hour <= 9:    return int(cap * random.uniform(0.70, 0.95))
    if 15 <= hour <= 18:  return int(cap * random.uniform(0.75, 1.00))
    if 10 <= hour <= 14:  return int(cap * random.uniform(0.45, 0.65))
    if 19 <= hour <= 22:  return int(cap * random.uniform(0.25, 0.45))
    if 5 <= hour <= 6:    return int(cap * random.uniform(0.15, 0.30))
    return int(cap * random.uniform(0.03, 0.10))

def avg_speed(traffic, capacity, speed_limit, weather_factor):
    ratio = min(1.0, traffic / capacity)
    # Speed decreases as traffic increases (BPR function)
    speed = speed_limit * (1.0 - 0.8 * (ratio ** 2)) * weather_factor
    noise = random.uniform(-3, 3)
    return max(5, round(speed + noise, 1))

def congestion_index(traffic, capacity):
    r = traffic / capacity
    if r > 0.90: return "Severe"
    if r > 0.75: return "High"
    if r > 0.50: return "Medium"
    if r > 0.25: return "Low"
    return "Free Flow"

def incident_flag(hour, weather_code, ratio):
    # Higher chance at peak hours, bad weather, high volume
    base_prob = 0.008
    if hour in range(7,10) or hour in range(16,19): base_prob *= 2
    if weather_code in ["Snow","Fog","Thunderstorm"]: base_prob *= 3
    if ratio > 0.85: base_prob *= 2
    return 1 if random.random() < base_prob else 0

print("Generating BIG dataset — 5 locations × 4 years × 8760 hr = 175,200 rows per location")
print("Total target: ~875,000+ records")
print("Please wait...")

rows = []
header = [
    "datetime","location_id","location_name","latitude","longitude",
    "traffic_volume","capacity","occupancy_pct","avg_speed_kmh",
    "congestion_level","congestion_index",
    "holiday","holiday_name",
    "hour","day_of_week","day_name","month","month_name","year","week_of_year",
    "is_weekend","is_peak_hour","is_night","season",
    "weather_main","weather_desc","temp_kelvin","temp_celsius",
    "rain_1h","snow_1h","clouds_all","humidity","wind_speed",
    "weather_factor","incident_flag","lanes","speed_limit"
]

DAY_NAMES   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTH_NAMES = ["","January","February","March","April","May","June",
               "July","August","September","October","November","December"]
SEASONS     = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
               6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}

start = datetime(2020, 1, 1)
end   = datetime(2023, 12, 31, 23, 0, 0)
total = 0

for loc in LOCATIONS:
    current = start
    loc_rows = 0
    capacity = loc["lanes"] * 1800

    while current <= end:
        mm_dd = current.strftime("%m-%d")
        is_holiday = mm_dd in HOLIDAYS
        holiday_name, holiday_factor = HOLIDAYS.get(mm_dd, ("None", 1.0))

        hour    = current.hour
        weekday = current.weekday()
        month   = current.month
        season  = SEASONS[month]

        # Pick weather weighted by season
        if season == "Winter":
            weights = [5,5,8,8,6,6,4,4,12,10,6,8,6,4,2,6]
        elif season == "Summer":
            weights = [15,12,10,8,8,5,3,6,1,0,0,5,2,1,4,10]
        elif season == "Spring":
            weights = [10,8,10,10,12,10,6,8,3,1,0,8,4,2,5,8]
        else:  # Fall
            weights = [8,8,10,10,10,8,5,8,5,3,1,10,7,5,3,8]

        wtotal = sum(weights)
        r = random.random() * wtotal
        cumw = 0; wi = 0
        for k,w in enumerate(weights):
            cumw += w
            if r <= cumw: wi = k; break
        wdata = WEATHER_CONDITIONS[wi]
        wfactor = wdata[5]

        # Temperature by season
        if season == "Winter": temp_c = random.uniform(-18, 5)
        elif season == "Spring": temp_c = random.uniform(5, 22)
        elif season == "Summer": temp_c = random.uniform(18, 35)
        else: temp_c = random.uniform(3, 20)
        temp_k = round(temp_c + 273.15, 2)
        temp_c = round(temp_c, 2)

        rain  = round(wdata[2] * random.uniform(0.5, 2.0), 2)
        snow  = round(wdata[3] * random.uniform(0.5, 2.0), 2)
        clouds = min(100, wdata[4] + random.randint(-10, 10))
        humidity  = random.randint(30, 98)
        wind_spd  = round(random.uniform(0, 40), 1)

        vol = base_traffic(hour, weekday, loc["lanes"], is_holiday, holiday_factor)
        vol = int(vol * wfactor * random.uniform(0.92, 1.08))
        vol = max(0, min(capacity, vol))

        occ   = round(min(100, vol/capacity*100), 1)
        spd   = avg_speed(vol, capacity, loc["speed_limit"], wfactor)
        ci    = round(vol/capacity, 4)
        cong  = congestion_index(vol, capacity)
        inc   = incident_flag(hour, wdata[0], vol/capacity)
        is_peak = 1 if (6<=hour<=9 or 15<=hour<=18) else 0
        is_night = 1 if (hour>=22 or hour<=5) else 0

        rows.append([
            current.strftime("%Y-%m-%d %H:%M:%S"),
            loc["id"], loc["name"],
            loc["lat"], loc["lon"],
            vol, capacity, occ, spd,
            cong, ci,
            1 if is_holiday else 0, holiday_name,
            hour, weekday, DAY_NAMES[weekday],
            month, MONTH_NAMES[month], current.year,
            current.isocalendar()[1],
            1 if weekday>=5 else 0, is_peak, is_night, season,
            wdata[0], wdata[1],
            temp_k, temp_c, rain, snow, clouds, humidity, wind_spd,
            wfactor, inc,
            loc["lanes"], loc["speed_limit"]
        ])
        loc_rows += 1
        current += timedelta(hours=1)

    total += loc_rows
    print(f"  {loc['name']}: {loc_rows:,} rows")

print(f"\nTotal rows: {total:,}")
print("Writing CSV...")

with open("metro_traffic_big.csv","w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

size_mb = round(__import__('os').path.getsize('metro_traffic_big.csv')/1024/1024, 1)
print(f"Done! metro_traffic_big.csv → {total:,} rows · {size_mb} MB")
