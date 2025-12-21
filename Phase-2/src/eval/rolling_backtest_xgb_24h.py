from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = Path("data/processed/features_lagroll_24h.parquet")
OUT_PATH = Path("reports/rolling_backtest_xgb_24h.csv")

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

# Backtest settings (match your earlier backtest)
FIRST_TRAIN_END = "2024-06-30 23:00:00+00:00"  # first window trains on Jan-Jun 2024
TEST_MONTH_FREQ = "MS"  # month starts
EARLY_STOPPING_ROUNDS = 200
NUM_BOOST_ROUND = 5000
SEED = 42


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    # Given a month start (MS), return the last timestamp in that month at 23:00
    # Works for hourly UTC series.
    next_month = (ts + pd.offsets.MonthBegin(1))
    return next_month - pd.Timedelta(hours=1)


def main() -> None:
    import xgboost as xgb

    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Ensure no NaNs in features/target
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).reset_index(drop=True)

    data_start = df["timestamp"].min()
    data_end = df["timestamp"].max()

    first_train_end = pd.to_datetime(FIRST_TRAIN_END, utc=True)

    # The first test month starts right after the initial training end month
    first_test_start = (first_train_end + pd.offsets.MonthBegin(1)).normalize()

    # last test start is the last month we can fully evaluate
    last_test_start = (
    df["timestamp"]
    .max()
    .to_period("M")
    .to_timestamp(how="start")
    .tz_localize("UTC")
)
    
    # But ensure the month is fully present
    last_full_month_start = (
    (df["timestamp"].max() - pd.offsets.MonthBegin(1))
    .to_period("M")
    .to_timestamp(how="start")
    .tz_localize("UTC")
)

    # Use last_full_month_start to avoid partial last month
    last_test_start = last_full_month_start

    test_starts = pd.date_range(start=first_test_start, end=last_test_start, freq=TEST_MONTH_FREQ, tz="UTC")

    print("=== Rolling Monthly Backtest: XGBoost (lag+rolling) 24h ahead ===")
    print(f"Data range: {data_start} → {data_end}")
    print(f"First train_end: {first_train_end}")
    print(f"Last test month start: {test_starts[-1] if len(test_starts)>0 else None}")
    print(f"Num windows: {len(test_starts)}")
    print(f"xgboost version: {xgb.__version__}")

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": SEED,
    }

    rows = []
    for test_start in test_starts:
        test_end = month_end(test_start)

        train_df = df[df["timestamp"] <= (test_start - pd.Timedelta(hours=1))]
        test_df = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)]

        # Safety: skip empty windows
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = train_df[FEATURE_COLS].to_numpy()
        y_train = train_df[TARGET_COL].to_numpy()

        X_test = test_df[FEATURE_COLS].to_numpy()
        y_test = test_df[TARGET_COL].to_numpy()

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLS)

        # Early stopping needs a validation set.
        # For rolling backtests, use the last 30 days of training as "internal val".
        # This keeps it leakage-safe because it's still prior to test_start.
        cutoff_val_start = (test_start - pd.Timedelta(days=30))
        val_df = train_df[train_df["timestamp"] >= cutoff_val_start]
        trn_df = train_df[train_df["timestamp"] < cutoff_val_start]

        # If val slice is too small (e.g., early windows), fall back to no early stopping
        use_es = len(val_df) >= 24 * 7 and len(trn_df) >= 24 * 30  # at least 7 days val, 30 days train

        if use_es:
            dtrn = xgb.DMatrix(trn_df[FEATURE_COLS].to_numpy(), label=trn_df[TARGET_COL].to_numpy(), feature_names=FEATURE_COLS)
            dval = xgb.DMatrix(val_df[FEATURE_COLS].to_numpy(), label=val_df[TARGET_COL].to_numpy(), feature_names=FEATURE_COLS)

            booster = xgb.train(
                params=params,
                dtrain=dtrn,
                num_boost_round=NUM_BOOST_ROUND,
                evals=[(dval, "val")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )
            y_pred = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
            best_it = booster.best_iteration
        else:
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=500,  # conservative if no early stop
                evals=[(dtrain, "train")],
                verbose_eval=False,
            )
            y_pred = booster.predict(dtest)
            best_it = None

        window_mae = mae(y_test, y_pred)
        window_rmse = rmse(y_test, y_pred)

        rows.append(
            {
                "train_end": (test_start - pd.Timedelta(hours=1)),
                "test_start": test_start,
                "test_end": test_end,
                "test_rows": len(test_df),
                "mae": window_mae,
                "rmse": window_rmse,
                "best_iteration": best_it,
                "used_early_stopping": bool(use_es),
            }
        )

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("\nPer-window results (head):")
    print(out.head(10).to_string(index=False))

    print("\nSummary:")
    if len(out) > 0:
        print(f"MAE  mean: {out['mae'].mean():.3f} | median: {out['mae'].median():.3f} | max: {out['mae'].max():.3f}")
        print(f"RMSE mean: {out['rmse'].mean():.3f} | median: {out['rmse'].median():.3f} | max: {out['rmse'].max():.3f}")

    print(f"\n📝 Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
