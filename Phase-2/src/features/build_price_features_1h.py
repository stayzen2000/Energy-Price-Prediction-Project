"""
Phase 2B — Step 4: Build price-focused features for t+1 forecasting

Input:
- data/processed/training_frame_price_1h.parquet

Output:
- data/processed/model_frame_price_1h.parquet

Features added (causal):
- price_lag_1,2,3,6,24
- price_roll_mean_3,6,24
- price_roll_std_6,24

Notes:
- No modeling here.
- Drops rows with insufficient history (due to lag/rolling).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "training_frame_price_1h.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_frame_price_1h.parquet"

TS_COL = "timestamp"
PRICE_COL = "price_per_mwh"
TARGET_COL = "target_price_per_mwh_t_plus_1"


def main() -> int:
    try:
        if not IN_PATH.exists():
            raise FileNotFoundError(f"Missing input parquet: {IN_PATH}")

        df = pd.read_parquet(IN_PATH).sort_values(TS_COL).reset_index(drop=True)

        # --- Lag features (price history) ---
        lags = [1, 2, 3, 6, 24]
        for k in lags:
            df[f"price_lag_{k}"] = df[PRICE_COL].shift(k)

        # --- Rolling statistics (use past/current only) ---
        # rolling mean windows
        for w in [3, 6, 24]:
            df[f"price_roll_mean_{w}"] = df[PRICE_COL].rolling(window=w, min_periods=w).mean()

        # rolling std windows (volatility proxy)
        for w in [6, 24]:
            df[f"price_roll_std_{w}"] = df[PRICE_COL].rolling(window=w, min_periods=w).std()

        # Identify new feature cols
        new_feature_cols = (
            [f"price_lag_{k}" for k in lags]
            + [f"price_roll_mean_{w}" for w in [3, 6, 24]]
            + [f"price_roll_std_{w}" for w in [6, 24]]
        )

        # Drop rows with NaNs in target or any new features
        # (Target should already be clean, but keep it explicit.)
        before = len(df)
        df = df.dropna(subset=[TARGET_COL] + new_feature_cols).reset_index(drop=True)
        after = len(df)
        dropped = before - after

        # Save
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PATH, index=False)

        print(f"[OK] Wrote: {OUT_PATH}")
        print(f"Rows before feature drop: {before:,}")
        print(f"Rows after feature drop:  {after:,}")
        print(f"Dropped due to history:   {dropped:,}")
        print(f"Total columns now:        {len(df.columns):,}")
        print(f"TS range:                 {df[TS_COL].min()} → {df[TS_COL].max()}")
        print("\nNew feature columns:")
        for c in new_feature_cols:
            print(f" - {c}")
        return 0

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
