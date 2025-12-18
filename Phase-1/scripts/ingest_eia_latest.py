from datetime import datetime, timedelta, timezone

from ingest_historical import fetch_grid_data, write_grid_to_db


def utc_hour_floor(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def main(lookback_hours: int = 48):
    end = utc_hour_floor(datetime.now(timezone.utc))
    start = end - timedelta(hours=lookback_hours)

    print(f"[EIA LATEST] Fetching grid data from {start} to {end}")

    grid_df = fetch_grid_data(start, end)

    print(f"[EIA LATEST] Fetched {len(grid_df)} rows")

    # This should use your new conflict-handling insert
    write_grid_to_db(grid_df)

    print("[EIA LATEST] Done.")


if __name__ == "__main__":
    main()
