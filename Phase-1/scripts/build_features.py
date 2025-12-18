import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")  # keep using your .env pattern

if not DB_URL:
    raise ValueError("DB_URL missing from .env")

engine = create_engine(DB_URL)

def main():
    df = pd.read_sql("SELECT * FROM v_hourly_base ORDER BY timestamp", engine)

    # make sure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")


    # group by zone (even if you only have ALL right now)
    df = df.sort_values(["zone_id", "timestamp"])

    # ---- basic time features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # ---- lag features for demand
    for lag in [1, 2, 24, 48, 168]:  # 1h, 2h, 1d, 2d, 1w
        df[f"demand_lag_{lag}"] = df.groupby("zone_id")["demand_mw"].shift(lag)

    # ---- rolling means
    for win in [3, 6, 24]:
        df[f"demand_rollmean_{win}"] = (
            df.groupby("zone_id")["demand_mw"]
              .shift(1)
              .rolling(win)
              .mean()
              .reset_index(level=0, drop=True)
        )

    # drop rows that don’t have enough history yet
    df = df.dropna(subset=["demand_lag_24", "demand_rollmean_24"])

    # write to DB as a table Power BI + model can use
    df.to_sql("features_hourly", engine, if_exists="replace", index=False)
    print(f"Wrote {len(df)} rows to features_hourly")

if __name__ == "__main__":
    main()
