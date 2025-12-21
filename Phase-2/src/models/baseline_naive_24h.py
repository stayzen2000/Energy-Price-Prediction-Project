from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


DATA_PATH = Path("data/processed/training_frame_24h.parquet")  # update to your final name

TARGET_COL = "target_demand_mw_t_plus_24h"
PRED_COL = "demand_mw"


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Count NaNs before evaluation
    n_true_nan = df[TARGET_COL].isna().sum()
    n_pred_nan = df[PRED_COL].isna().sum()

    # Drop rows where we can't compute the baseline
    eval_df = df.dropna(subset=[TARGET_COL, PRED_COL]).copy()

    print("=== Naive Seasonal Baseline (24h ahead) ===")
    print(f"Rows total:      {len(df):,}")
    print(f"Rows eval:       {len(eval_df):,}")
    print(f"NaNs in y_true:  {n_true_nan:,}")
    print(f"NaNs in y_pred:  {n_pred_nan:,}")
    print(f"Rows dropped for eval: {len(df) - len(eval_df):,}")

    y_true = eval_df[TARGET_COL].to_numpy()
    y_pred = eval_df[PRED_COL].to_numpy()

    print(f"MAE:  {mae(y_true, y_pred):.3f}")
    print(f"RMSE: {rmse(y_true, y_pred):.3f}")

    sample = eval_df[["timestamp", "demand_mw", TARGET_COL]].head(5)
    print("\nSample rows:")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
