from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = Path("data/processed/features_lagroll_24h.parquet")
TARGET_COL = "target_demand_mw_t_plus_24h"

FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "is_weekend",
    "month_sin", "month_cos",
    "demand_lag_1",
    "demand_lag_24",
    "demand_lag_48",
    "demand_lag_168",
    "demand_roll_mean_24",
    "demand_roll_std_24",
    "demand_roll_mean_168",
    "demand_roll_std_168",
]

TRAIN_END = "2025-06-30 23:00:00+00:00"
VAL_END = "2025-09-30 23:00:00+00:00"


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).reset_index(drop=True)

    train_end_ts = pd.to_datetime(TRAIN_END, utc=True)
    val_end_ts = pd.to_datetime(VAL_END, utc=True)

    train = df[df["timestamp"] <= train_end_ts]
    val = df[(df["timestamp"] > train_end_ts) & (df["timestamp"] <= val_end_ts)]
    test = df[df["timestamp"] > val_end_ts]

    X_train, y_train = train[FEATURE_COLS].to_numpy(), train[TARGET_COL].to_numpy()
    X_val, y_val = val[FEATURE_COLS].to_numpy(), val[TARGET_COL].to_numpy()
    X_test, y_test = test[FEATURE_COLS].to_numpy(), test[TARGET_COL].to_numpy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print("=== Ridge Train/Val/Test (24h ahead) ===")
    print(f"Train end: {TRAIN_END}")
    print(f"Val end:   {VAL_END}")
    print(f"Rows: train={len(train):,} | val={len(val):,} | test={len(test):,}")

    print("\n--- Validation ---")
    print(f"MAE:  {mae(y_val, val_pred):.3f}")
    print(f"RMSE: {rmse(y_val, val_pred):.3f}")

    print("\n--- Test ---")
    print(f"MAE:  {mae(y_test, test_pred):.3f}")
    print(f"RMSE: {rmse(y_test, test_pred):.3f}")


if __name__ == "__main__":
    main()
