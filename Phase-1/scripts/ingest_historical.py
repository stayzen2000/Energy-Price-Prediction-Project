# scripts/ingest_historical.py

"""
Phase 1 – Step 3: Ingest historical grid + weather data into Postgres.

This script will:
1. Connect to the Postgres database.
2. Pull historical grid load + price data (placeholder API).
3. Pull historical weather data (placeholder API).
4. Write both into the grid_load and weather tables.

You will later refine the API URLs & parsing once you pick exact endpoints.
"""

import sys
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import MetaData, Table
from sqlalchemy import text 
from sqlalchemy import func


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import REGION_ID, EIA_RESPONDENT

# -------------- Config & DB setup -------------- #

load_dotenv()  # load variables from .env

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

EIA_API_KEY = os.getenv("EIA_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# SQLAlchemy connection string
CONN_STR = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
engine = create_engine(CONN_STR)


def get_default_date_range(months_back: int = 6):
    """
    Returns start and end datetimes for the last `months_back` months.
    This uses your Python datetime knowledge.
    """
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30 * months_back)
    return start, end


# -------------- Fetch grid data -------------- #

def fetch_grid_data(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch PJM hourly demand (D) + demand forecast (DF) from EIA v2.
    Returns normalized rows with BOTH demand_mw and demand_forecast_mw.
    """
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY missing from .env")

    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

    # EIA expects time strings like 2025-12-14T22 (no minutes)
    start_str = start.strftime("%Y-%m-%dT%H")
    end_str = end.strftime("%Y-%m-%dT%H")

    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][0]": EIA_RESPONDENT,
        "facets[type][0]": "D",
        "facets[type][1]": "DF",
        "start": start_str,
        "end": end_str,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000
    }

    all_rows = []
    while True:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()["response"]
        data = payload.get("data", [])
        if not data:
            break

        all_rows.extend(data)

        total = int(payload.get("total", 0) or 0)
        offset = int(params.get("offset", 0))
        length = int(params.get("length", 0))

        if offset + length >= total:
            break

        params["offset"] = offset + length

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # Normalize timestamp
    df["timestamp"] = pd.to_datetime(df["period"], utc=True)

    # Pivot types into columns at each timestamp
    pivot = (
        df.pivot_table(index="timestamp", columns="type", values="value", aggfunc="first")
        .reset_index()
    )

    # Map to our DB schema
    pivot["region"] = REGION_ID
    pivot["zone_id"] = "NYISO"
    pivot["demand_mw"] = pivot.get("D")
    pivot["demand_forecast_mw"] = pivot.get("DF")
    pivot["price_per_mwh"] = None
    pivot["source"] = "EIA"

    out = pivot[["timestamp", "region", "zone_id", "demand_mw", "demand_forecast_mw", "price_per_mwh", "source"]]
    return out


# -------------- Write DataFrames to Postgres -------------- #

def write_grid_to_db(df):
    df = df.copy()

    # ensure numeric types (EIA values often come as strings)
    for col in ["demand_mw", "demand_forecast_mw", "price_per_mwh"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    metadata = MetaData()
    grid_table = Table("grid_load", metadata, autoload_with=engine)

    records = df.to_dict(orient="records")
    if not records:
        print("No grid data to write.")
        return

    stmt = insert(grid_table).values(records)

    # UPDATE the existing row if primary key already exists
    update_cols = {
    "region": stmt.excluded.region,
    "demand_mw": stmt.excluded.demand_mw,
    "demand_forecast_mw": stmt.excluded.demand_forecast_mw,

    # ✅ keep existing price if the incoming one is NULL
    "price_per_mwh": func.coalesce(stmt.excluded.price_per_mwh, grid_table.c.price_per_mwh),

    "source": stmt.excluded.source,
}

    stmt = stmt.on_conflict_do_update(
        index_elements=["timestamp", "zone_id"],
        set_=update_cols
    )

    with engine.begin() as conn:
        conn.execute(stmt)

    print(f"Upserted {len(records)} grid rows (inserted or updated).")



def write_weather_to_db(df):
    df = df.copy()

    # make sure numeric columns are numeric
    for col in ["temp_c", "humidity", "wind_speed", "precipitation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    metadata = MetaData()
    weather_table = Table("weather", metadata, autoload_with=engine)

    records = df.to_dict(orient="records")
    if not records:
        print("No weather data to write.")
        return

    stmt = insert(weather_table).values(records)

    # UPDATE the existing row if primary key already exists
    stmt = stmt.on_conflict_do_update(
        index_elements=["timestamp", "zone_id"],
        set_={
            "temp_c": stmt.excluded.temp_c,
            "humidity": stmt.excluded.humidity,
            "wind_speed": stmt.excluded.wind_speed,
            "precipitation": stmt.excluded.precipitation,
        }
    )

    with engine.begin() as conn:
        conn.execute(stmt)

    print(f"Upserted {len(records)} weather rows (inserted or updated).")


# -------------- Main runner -------------- #

def main():
    # Usage:
    #   python ingest_historical.py 2024-01-01 2025-12-17
    # If no args, defaults to last 6 months

    if len(sys.argv) >= 3:
        start_str = sys.argv[1]
        end_str = sys.argv[2]
        start = pd.to_datetime(start_str, utc=True)
        end = pd.to_datetime(end_str, utc=True)
    else:
        start, end = get_default_date_range(months_back=6)

    print(f"Fetching data from {start} to {end}")

    grid_df = fetch_grid_data(start, end)
    print(f"Fetched {len(grid_df)} grid rows")

    write_grid_to_db(grid_df)

    print("Historical ingestion complete.")


if __name__ == "__main__":
    main()
