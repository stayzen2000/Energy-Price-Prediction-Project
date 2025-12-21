from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure Phase-2 root is importable (helps if you later import shared utilities)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

IN_PATH = Path("data/processed/training_frame_24h.parquet")
OUT_PATH = Path("data/processed/features_calendar_24h.parquet")

TARGET_COL = "target_demand_mw_t_plus_24h"
LOCAL_TZ = "America/New_York"


def main() -> None:
    # Load
    df = pd.read_parquet(IN_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Convert to local timezone for calendar features (NYISO aligns to US/Eastern)
    ts_local = df["timestamp"].dt.tz_convert(LOCAL_TZ)

    # Raw calendar components
    df["hour"] = ts_local.dt.hour
    df["dayofweek"] = ts_local.dt.dayofweek  # Mon=0..Sun=6
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = ts_local.dt.month

    # Cyclical encodings to avoid fake ordinal distances (e.g., hour 23 close to hour 0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Minimal columns for modeling (keep timestamp for splitting/backtests; keep demand_mw for later lags)
    keep_cols = [
        "timestamp",
        TARGET_COL,
        "demand_mw",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "month_sin",
        "month_cos",
    ]
    feat_df = df[keep_cols].copy()

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(OUT_PATH, index=False)

    # Print validation-style summary
    print(f"Saved calendar feature dataset: {OUT_PATH}")
    print(f"Rows: {len(feat_df):,}")
    print("Columns:")
    print(feat_df.columns.tolist())
    print("Null counts (calendar features should be 0):")
    print(feat_df.isna().sum().to_string())


if __name__ == "__main__":
    main()
