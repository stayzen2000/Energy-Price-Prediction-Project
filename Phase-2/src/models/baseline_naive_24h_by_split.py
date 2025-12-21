from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.eval.splits import make_time_splits

DATA_PATH = Path("data/processed/training_frame_24h.parquet")
TARGET_COL = "target_demand_mw_t_plus_24h"
PRED_COL = "demand_mw"

TRAIN_END = "2025-06-30 23:00:00+00:00"
VAL_END = "2025-09-30 23:00:00+00:00"


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def eval_split(name: str, df: pd.DataFrame) -> None:
    eval_df = df.dropna(subset=[TARGET_COL, PRED_COL]).copy()
    y_true = eval_df[TARGET_COL].to_numpy()
    y_pred = eval_df[PRED_COL].to_numpy()

    print(f"\n--- {name} ---")
    print(f"Rows: {len(df):,} | Eval rows: {len(eval_df):,} | Dropped: {len(df)-len(eval_df):,}")
    print(f"MAE:  {mae(y_true, y_pred):.3f}")
    print(f"RMSE: {rmse(y_true, y_pred):.3f}")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)

    train, val, test = make_time_splits(df, train_end=TRAIN_END, val_end=VAL_END)

    print("=== Naive Seasonal Baseline (24h ahead) by Split ===")
    print(f"Train end: {TRAIN_END}")
    print(f"Val end:   {VAL_END}")

    eval_split("TRAIN", train)
    eval_split("VAL", val)
    eval_split("TEST", test)


if __name__ == "__main__":
    main()
