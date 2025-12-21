from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

IN_PATH = Path("data/processed/features_calendar_24h.parquet")
OUT_PATH = Path("data/processed/features_lagroll_24h.parquet")

TARGET_COL = "target_demand_mw_t_plus_24h"


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # === Lag features (using only past/present demand_mw) ===
    df["demand_lag_1"] = df["demand_mw"].shift(1)
    df["demand_lag_24"] = df["demand_mw"].shift(24)
    df["demand_lag_48"] = df["demand_mw"].shift(48)
    df["demand_lag_168"] = df["demand_mw"].shift(168)

    # === Rolling features (computed using history up to t) ===
    # Important: rolling() at time t uses values up to t (not future), so it's leakage-safe.
    df["demand_roll_mean_24"] = df["demand_mw"].rolling(window=24, min_periods=24).mean()
    df["demand_roll_std_24"] = df["demand_mw"].rolling(window=24, min_periods=24).std()

    df["demand_roll_mean_168"] = df["demand_mw"].rolling(window=168, min_periods=168).mean()
    df["demand_roll_std_168"] = df["demand_mw"].rolling(window=168, min_periods=168).std()

    # Drop rows where target or required lag/rolling features are missing
    feature_cols = [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "is_weekend",
        "month_sin", "month_cos",
        "demand_lag_1", "demand_lag_24", "demand_lag_48", "demand_lag_168",
        "demand_roll_mean_24", "demand_roll_std_24",
        "demand_roll_mean_168", "demand_roll_std_168",
    ]

    before = len(df)
    df = df.dropna(subset=[TARGET_COL] + feature_cols).reset_index(drop=True)
    after = len(df)

    keep_cols = ["timestamp", TARGET_COL] + feature_cols
    feat_df = df[keep_cols].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(OUT_PATH, index=False)

    print(f"Saved lag+rolling feature dataset: {OUT_PATH}")
    print(f"Rows before: {before:,} | Rows after: {after:,} | Dropped: {before-after:,}")
    print("Null counts (should be 0 for kept columns):")
    print(feat_df.isna().sum().to_string())


if __name__ == "__main__":
    main()
