from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure Phase-2 root is on sys.path (consistent with your other scripts)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = Path("data/processed/features_calendar_24h.parquet")

TARGET_COL = "target_demand_mw_t_plus_24h"
FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "is_weekend",
    "month_sin", "month_cos",
]

LOCAL_TZ = "America/New_York"


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
    year = ts.year + (ts.month - 1 + n) // 12
    month = (ts.month - 1 + n) % 12 + 1
    return pd.Timestamp(year=year, month=month, day=1, tz="UTC")


def build_monthly_windows(first_train_end: str, last_test_month_start: str) -> list[BacktestWindow]:
    first_train_end_ts = pd.to_datetime(first_train_end, utc=True)
    last_test_month_start_ts = pd.to_datetime(last_test_month_start, utc=True)

    windows: list[BacktestWindow] = []
    cur_train_end = first_train_end_ts

    while True:
        test_start = add_month(month_start(cur_train_end), 1)
        if test_start > last_test_month_start_ts:
            break

        test_end = add_month(test_start, 1) - pd.Timedelta(hours=1)
        windows.append(BacktestWindow(train_end=cur_train_end, test_start=test_start, test_end=test_end))

        cur_train_end = test_end

    return windows


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Pipeline:
    """
    Ridge works best when features are standardized.
    StandardScaler + Ridge is a clean baseline pipeline.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Drop rows with missing y or missing features (calendar features shouldn't be missing)
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).reset_index(drop=True)

    # Backtest bounds: same as your naive rolling backtest
    first_train_end = "2024-06-30 23:00:00+00:00"
    last_test_month_start = "2025-11-01 00:00:00+00:00"

    windows = build_monthly_windows(first_train_end=first_train_end, last_test_month_start=last_test_month_start)

    print("=== Rolling Monthly Backtest: Ridge (calendar-only) 24h ahead ===")
    print(f"Data range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Num windows: {len(windows)}")
    print(f"Features: {FEATURE_COLS}")

    results = []

    for w in windows:
        train_df = df[df["timestamp"] <= w.train_end]
        test_df = df[(df["timestamp"] >= w.test_start) & (df["timestamp"] <= w.test_end)]

        X_train = train_df[FEATURE_COLS].to_numpy()
        y_train = train_df[TARGET_COL].to_numpy()

        X_test = test_df[FEATURE_COLS].to_numpy()
        y_test = test_df[TARGET_COL].to_numpy()

        model = fit_ridge(X_train, y_train, alpha=1.0)
        y_pred = model.predict(X_test)

        results.append(
            {
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "mae": mae(y_test, y_pred),
                "rmse": rmse(y_test, y_pred),
            }
        )

    res_df = pd.DataFrame(results)

    print("\nPer-window results:")
    print(res_df[["train_end", "test_start", "test_end", "test_rows", "mae", "rmse"]].to_string(index=False))

    print("\nSummary:")
    print(f"MAE  mean: {res_df['mae'].mean():.3f} | median: {res_df['mae'].median():.3f} | max: {res_df['mae'].max():.3f}")
    print(f"RMSE mean: {res_df['rmse'].mean():.3f} | median: {res_df['rmse'].median():.3f} | max: {res_df['rmse'].max():.3f}")

    out_path = Path("reports/rolling_backtest_ridge_calendar_24h.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"\n📝 Saved: {out_path}")


if __name__ == "__main__":
    main()
