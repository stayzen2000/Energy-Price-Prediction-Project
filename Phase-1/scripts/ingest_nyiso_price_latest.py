import sys, os
from datetime import date, timedelta

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
    df = ny.get_lmp(date=day, market=MARKET)

    df["timestamp"] = pd.to_datetime(df["Time"]).dt.tz_convert("UTC")
    df = df[df["Location Type"].str.lower() == "zone"].copy()

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
    today = date.today()
    days = [(today - timedelta(days=1)).isoformat(), today.isoformat()]

    total = 0
    for d in days:
        df = fetch_day_prices(d)
        attempted = upsert_prices(df)
        total += attempted
        print(f"[nyiso_price_latest] {d} attempted={attempted}")

    print(f"[nyiso_price_latest] DONE total_attempted={total}")
