import sys, os
from datetime import date, timedelta, datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text
import gridstatus
from db import engine

ZONE_ID = "ALL"
SOURCE = "NYISO"
MARKET = "DAY_AHEAD_HOURLY"

def fetch_day_prices(day: str) -> pd.DataFrame:
    ny = gridstatus.NYISO()
    df = ny.get_lmp(date=day, market=MARKET)  # returns all zones

    # Normalize to UTC
    df["timestamp"] = pd.to_datetime(df["Time"]).dt.tz_convert("UTC")

    # Keep only Zones
    df = df[df["Location Type"].str.lower() == "zone"].copy()

    # System price = average LMP across zones per hour
    agg = (
        df.groupby("timestamp", as_index=False)["LMP"]
        .mean()
        .rename(columns={"LMP": "price_per_mwh"})
    )

    agg["zone_id"] = ZONE_ID
    agg["source"] = SOURCE

    return agg[["timestamp", "zone_id", "price_per_mwh", "source"]].sort_values("timestamp")

def upsert_prices(df: pd.DataFrame) -> int:
    sql = text("""
        INSERT INTO grid_load (timestamp, zone_id, price_per_mwh, source)
        VALUES (:timestamp, :zone_id, :price_per_mwh, :source)
        ON CONFLICT (timestamp, zone_id) DO UPDATE SET
            price_per_mwh = EXCLUDED.price_per_mwh,
            source = EXCLUDED.source
    """)
    rows = df.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)

if __name__ == "__main__":
    # Usage:
    # python scripts/ingest_nyiso_price_backfill.py 2024-01-01 2025-12-12
    if len(sys.argv) != 3:
        print("Usage: python scripts/ingest_nyiso_price_backfill.py YYYY-MM-DD YYYY-MM-DD")
        sys.exit(1)

    start = date.fromisoformat(sys.argv[1])
    end = date.fromisoformat(sys.argv[2])

    cur = start
    total = 0

    while cur <= end:
        day = cur.isoformat()
        df = fetch_day_prices(day)
        attempted = upsert_prices(df)
        total += attempted
        print(f"[nyiso_price_backfill] {day} attempted={attempted}")
        cur += timedelta(days=1)

    print(f"[nyiso_price_backfill] DONE total_attempted={total} window={start}..{end}")
