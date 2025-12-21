from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Ensure Phase-2 root is on sys.path so imports work if needed later
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


DATA_PATH = Path("data/processed/training_frame_24h.parquet")
TARGET_COL = "target_demand_mw_t_plus_24h"
PRED_COL = "demand_mw"


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@dataclass
class BacktestWindow:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")


def add_month(ts: pd.Timestamp, n: int = 1) -> pd.Timestamp:
    # Safe month increment without extra dependencies
    year = ts.year + (ts.month - 1 + n) // 12
    month = (ts.month - 1 + n) % 12 + 1
    return pd.Timestamp(year=year, month=month, day=1, tz="UTC")


def build_monthly_windows(
    df: pd.DataFrame,
    first_train_end: str,
    last_test_month_start: str,
) -> list[BacktestWindow]:
    """
    Expanding window backtest:
      - Train: from beginning -> train_end
      - Test: next calendar month (test_start -> test_end)
      - Step forward by month
    """
    first_train_end_ts = pd.to_datetime(first_train_end, utc=True)
    last_test_month_start_ts = pd.to_datetime(last_test_month_start, utc=True)

    windows: list[BacktestWindow] = []

    cur_train_end = first_train_end_ts
    while True:
        test_start = add_month(month_start(cur_train_end), 1)  # month after train_end month
        if test_start > last_test_month_start_ts:
            break

        test_end = add_month(test_start, 1) - pd.Timedelta(hours=1)  # inclusive end
        windows.append(BacktestWindow(train_end=cur_train_end, test_start=test_start, test_end=test_end))

        # advance train_end to end of that test month (so next window trains through it)
        cur_train_end = test_end

    return windows


def eval_naive_on_window(df: pd.DataFrame, w: BacktestWindow) -> dict:
    train_df = df[df["timestamp"] <= w.train_end].copy()
    test_df = df[(df["timestamp"] >= w.test_start) & (df["timestamp"] <= w.test_end)].copy()

    # Drop NaNs only for evaluation
    test_eval = test_df.dropna(subset=[TARGET_COL, PRED_COL]).copy()

    y_true = test_eval[TARGET_COL].to_numpy()
    y_pred = test_eval[PRED_COL].to_numpy()

    return {
        "train_end": w.train_end,
        "test_start": w.test_start,
        "test_end": w.test_end,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "eval_rows": len(test_eval),
        "dropped_eval_rows": len(test_df) - len(test_eval),
        "mae": mae(y_true, y_pred) if len(test_eval) else np.nan,
        "rmse": rmse(y_true, y_pred) if len(test_eval) else np.nan,
    }


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Choose backtest bounds:
    # Start training with at least ~6 months history (you can change later).
    # first_train_end means: first window trains through this date, then tests on next month.
    first_train_end = "2024-06-30 23:00:00+00:00"

    # Last month we want to test (start of month). We stop before we run off the end of dataset.
    # Your data ends 2025-12-16, so last test month start should be 2025-11-01 (tests Nov).
    last_test_month_start = "2025-11-01 00:00:00+00:00"

    windows = build_monthly_windows(df, first_train_end=first_train_end, last_test_month_start=last_test_month_start)

    print("=== Rolling Monthly Backtest: Naive Seasonal Baseline (24h ahead) ===")
    print(f"Data range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"First train_end: {first_train_end}")
    print(f"Last test month start: {last_test_month_start}")
    print(f"Num windows: {len(windows)}")

    results = [eval_naive_on_window(df, w) for w in windows]
    res_df = pd.DataFrame(results)

    # Print table
    display_cols = ["train_end", "test_start", "test_end", "eval_rows", "mae", "rmse"]
    print("\nPer-window results:")
    print(res_df[display_cols].to_string(index=False))

    # Summary
    print("\nSummary:")
    print(f"MAE  mean: {res_df['mae'].mean():.3f} | median: {res_df['mae'].median():.3f} | max: {res_df['mae'].max():.3f}")
    print(f"RMSE mean: {res_df['rmse'].mean():.3f} | median: {res_df['rmse'].median():.3f} | max: {res_df['rmse'].max():.3f}")

    # Save report artifact
    out_path = Path("reports/rolling_backtest_naive_24h.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"\n📝 Saved: {out_path}")


if __name__ == "__main__":
    main()
