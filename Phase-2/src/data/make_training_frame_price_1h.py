"""
Phase 2B — Step 2: Build training frame for next-hour price forecasting

Input (processed, Phase 2A artifact):
- data/processed/training_frame_24h.parquet

Output (new Phase 2B artifact):
- data/processed/training_frame_price_1h.parquet

Notes:
- No modeling here.
- No feature engineering beyond constructing the t+1 target.
- Strictly leakage-safe: target = price(t+1), features at time t.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "training_frame_24h.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "training_frame_price_1h.parquet"

TS_COL = "timestamp"
PRICE_COL = "price_per_mwh"
TARGET_COL = "target_price_per_mwh_t_plus_1"


def main() -> int:
    try:
        if not IN_PATH.exists():
            raise FileNotFoundError(f"Missing input parquet: {IN_PATH}")

        df = pd.read_parquet(IN_PATH)

        # Defensive sort to ensure shift is correct
        df = df.sort_values(TS_COL).reset_index(drop=True)

        # Construct t+1 target (price one hour ahead)
        df[TARGET_COL] = df[PRICE_COL].shift(-1)

        # Drop rows where label is missing (last row + any discontinuities/missing price)
        before = len(df)
        df = df.dropna(subset=[PRICE_COL, TARGET_COL]).reset_index(drop=True)
        after = len(df)

        dropped = before - after

        # Optional: enforce monotonic timestamps post-drop
        if not df[TS_COL].is_monotonic_increasing:
            raise ValueError("Timestamp not monotonic after processing — unexpected.")

        # Save
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT_PATH, index=False)

        print(f"[OK] Wrote: {OUT_PATH}")
        print(f"Rows before: {before:,}")
        print(f"Rows after:  {after:,}")
        print(f"Dropped:     {dropped:,}")
        print(f"Columns:     {len(df.columns):,}")
        print(f"TS range:    {df[TS_COL].min()} → {df[TS_COL].max()}")
        return 0

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
