from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = Path("data/processed/features_lagroll_24h.parquet")
MODEL_OUT = Path("models/xgb_24h.json")
REPORT_OUT = Path("reports/xgb_train_val_test_metrics.txt")

TARGET_COL = "target_demand_mw_t_plus_24h"
FEATURE_COLS = [
    # calendar
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "is_weekend",
    "month_sin", "month_cos",
    # lags
    "demand_lag_1",
    "demand_lag_24",
    "demand_lag_48",
    "demand_lag_168",
    # rollings
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


def time_split(df: pd.DataFrame):
    train_end_ts = pd.to_datetime(TRAIN_END, utc=True)
    val_end_ts = pd.to_datetime(VAL_END, utc=True)

    train = df[df["timestamp"] <= train_end_ts].copy()
    val = df[(df["timestamp"] > train_end_ts) & (df["timestamp"] <= val_end_ts)].copy()
    test = df[df["timestamp"] > val_end_ts].copy()
    return train, val, test


def main() -> None:
    import xgboost as xgb

    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Safety: remove any NaNs (should already be none)
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS).reset_index(drop=True)

    train_df, val_df, test_df = time_split(df)

    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()

    X_val = val_df[FEATURE_COLS].to_numpy()
    y_val = val_df[TARGET_COL].to_numpy()

    X_test = test_df[FEATURE_COLS].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    print("=== XGBoost Train/Val/Test (24h ahead) ===")
    print(f"Train end: {TRAIN_END}")
    print(f"Val end:   {VAL_END}")
    print(f"Rows: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"xgboost version: {xgb.__version__}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLS)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": 42,
    }

    num_boost_round = 5000
    early_stopping_rounds = 200

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    # Predict using the best iteration found by early stopping
    val_pred = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    test_pred = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))

    val_mae = mae(y_val, val_pred)
    val_rmse = rmse(y_val, val_pred)

    test_mae = mae(y_test, test_pred)
    test_rmse = rmse(y_test, test_pred)

    print("\n--- Validation ---")
    print(f"MAE:  {val_mae:.3f}")
    print(f"RMSE: {val_rmse:.3f}")

    print("\n--- Test ---")
    print(f"MAE:  {test_mae:.3f}")
    print(f"RMSE: {test_rmse:.3f}")

    print("\n--- Training metadata ---")
    print(f"best_iteration: {booster.best_iteration}")
    print(f"best_score:     {booster.best_score}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_OUT))
    print(f"\n✅ Saved model: {MODEL_OUT}")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(
        "\n".join(
            [
                "XGBoost Train/Val/Test (24h ahead) — xgb.train API",
                f"xgboost_version={xgb.__version__}",
                f"TRAIN_END={TRAIN_END}",
                f"VAL_END={VAL_END}",
                f"train_rows={len(train_df)} val_rows={len(val_df)} test_rows={len(test_df)}",
                f"features={FEATURE_COLS}",
                "",
                f"VAL_MAE={val_mae:.6f}",
                f"VAL_RMSE={val_rmse:.6f}",
                f"TEST_MAE={test_mae:.6f}",
                f"TEST_RMSE={test_rmse:.6f}",
                "",
                f"best_iteration={booster.best_iteration}",
                f"best_score={booster.best_score}",
                f"model_path={MODEL_OUT.as_posix()}",
            ]
        )
    )
    print(f"📝 Saved report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
