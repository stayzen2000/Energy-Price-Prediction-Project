from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# === Paths ===
RAW_CSV = Path("data/raw/Phase2_Training_Dataset.csv")
OUT_PARQUET = Path("data/processed/training_frame_24h.parquet")
OUT_CSV = Path("data/processed/training_frame_24h.csv")

HORIZON_HOURS = 24


def main() -> None:
    # Load
    df = pd.read_csv(RAW_CSV)

    # Parse + sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Treat zero demand as missing
    df.loc[df["demand_mw"] == 0, "demand_mw"] = np.nan

    # Horizon-aligned target: y(t) = demand(t + 24)
    df["target_demand_mw_t_plus_24h"] = df["demand_mw"].shift(-HORIZON_HOURS)

    # Drop rows where target is missing
    before = len(df)
    df = df.dropna(subset=["target_demand_mw_t_plus_24h"]).reset_index(drop=True)
    after = len(df)

    # Ensure output directory exists
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_PARQUET}")
    print(f"Saved: {OUT_CSV}")
    print(f"Rows before: {before:,}")
    print(f"Rows after:  {after:,}")
    print(f"Dropped:     {before - after:,}")
    print(
        f"Final time range: {df['timestamp'].min()} → {df['timestamp'].max()}"
    )


if __name__ == "__main__":
    main()
