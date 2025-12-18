import sys, os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from sqlalchemy import text
from db import engine

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

ZONE_ID = "ALL"
LAT = 39.9526
LON = -75.1652

def fetch_openmeteo_archive(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
        ]),
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        "zone_id": ZONE_ID,
        "temp_c": hourly.get("temperature_2m"),
        "humidity": hourly.get("relative_humidity_2m"),
        "wind_speed": hourly.get("wind_speed_10m"),
        "precipitation": hourly.get("precipitation"),
    })
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")

def upsert_weather(df: pd.DataFrame) -> int:
    sql = text("""
        INSERT INTO weather (timestamp, zone_id, temp_c, humidity, wind_speed, precipitation)
        VALUES (:timestamp, :zone_id, :temp_c, :humidity, :wind_speed, :precipitation)
        ON CONFLICT (timestamp, zone_id) DO UPDATE SET
            temp_c = EXCLUDED.temp_c,
            humidity = EXCLUDED.humidity,
            wind_speed = EXCLUDED.wind_speed,
            precipitation = EXCLUDED.precipitation
    """)
    rows = df.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)

def validate_date(s: str) -> str:
    # raises if invalid
    datetime.strptime(s, "%Y-%m-%d")
    return s

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/ingest_weather_historical.py YYYY-MM-DD YYYY-MM-DD")
        sys.exit(1)

    start_date = validate_date(sys.argv[1])
    end_date = validate_date(sys.argv[2])

    df = fetch_openmeteo_archive(start_date, end_date)
    attempted = upsert_weather(df)
    print(f"[weather_historical] attempted upsert rows={attempted}, window={start_date}..{end_date}, zone_id={ZONE_ID}")
