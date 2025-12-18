import sys, os
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text
from ingest_weather_historical import fetch_openmeteo_archive  # reuse your function
from db import engine

ZONE_ID = "ALL"

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

def iso(d: date) -> str:
    return d.isoformat()

if __name__ == "__main__":
    # Usage:
    # python scripts/ingest_weather_backfill.py 2024-01-01 2025-12-12
    if len(sys.argv) != 3:
        print("Usage: python scripts/ingest_weather_backfill.py YYYY-MM-DD YYYY-MM-DD")
        sys.exit(1)

    start = date.fromisoformat(sys.argv[1])
    end = date.fromisoformat(sys.argv[2])

    if end < start:
        raise ValueError("end date must be >= start date")

    cur = start
    total_attempted = 0

    while cur <= end:
        month_end = (cur + relativedelta(months=1) - timedelta(days=1))
        chunk_end = min(month_end, end)

        df = fetch_openmeteo_archive(iso(cur), iso(chunk_end))
        attempted = upsert_weather(df)
        total_attempted += attempted

        print(f"[weather_backfill] {iso(cur)}..{iso(chunk_end)} attempted={attempted}")

        cur = chunk_end + timedelta(days=1)

    print(f"[weather_backfill] DONE total_attempted={total_attempted} window={start}..{end}")
