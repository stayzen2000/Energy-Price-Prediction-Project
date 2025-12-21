"""
Phase 2B — Step 5: Train XGBoost for next-hour price forecasting (native API, version-safe)

Input:
- data/processed/model_frame_price_1h.parquet

Outputs:
- models/phase2b_xgb_price_1h.json
- reports/phase2b_xgb_price_1h.md
- data/processed/preds_xgb_price_1h.parquet

Key constraints:
- Time-aware splits only (no shuffle)
- Strict leakage prevention
- Early stopping via native xgb.train (version-safe)
"""

from __future__ import annotations

from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd
import xgboost as xgb


# --------------------------------------------------
# Paths / config
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IN_PATH = PROJECT_ROOT / "data" / "processed" / "model_frame_price_1h.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDS_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = MODELS_DIR / "phase2b_xgb_price_1h.json"
REPORT_PATH = REPORTS_DIR / "phase2b_xgb_price_1h.md"
PREDS_PATH = PREDS_DIR / "preds_xgb_price_1h.parquet"

TS_COL = "timestamp"
TARGET_COL = "target_price_per_mwh_t_plus_1"
ID_COLS = ["zone_id"]

TEST_FRACTION = 0.20
VAL_FRACTION_OF_TRAIN = 0.10

BASELINE_PERSIST_MAE = 5.5138
BASELINE_PERSIST_RMSE = 9.0417


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


# --------------------------------------------------
# Feature selection
# --------------------------------------------------
def select_features(df: pd.DataFrame) -> list[str]:
    exclude = set([TS_COL, TARGET_COL] + ID_COLS)
    features = []

    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("target_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(c)

    if not features:
        raise ValueError("No numeric feature columns selected.")

    return features


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> int:
    try:
        print("RUNNING FILE:", __file__)
        print("XGBOOST VERSION:", xgb.__version__)

        if not IN_PATH.exists():
            raise FileNotFoundError(f"Missing input parquet: {IN_PATH}")

        df = pd.read_parquet(IN_PATH).sort_values(TS_COL).reset_index(drop=True)

        if TARGET_COL not in df.columns:
            raise KeyError(f"Missing target column: {TARGET_COL}")

        if df[TARGET_COL].isna().any():
            raise ValueError("Target contains NaNs.")

        feature_cols = select_features(df)

        # -------------------------
        # Time-based splits
        # -------------------------
        n = len(df)
        test_n = int(round(n * TEST_FRACTION))
        split_test = n - test_n

        df_train_full = df.iloc[:split_test]
        df_test = df.iloc[split_test:]

        train_n = len(df_train_full)
        val_n = int(round(train_n * VAL_FRACTION_OF_TRAIN))
        split_val = train_n - val_n

        df_train = df_train_full.iloc[:split_val]
        df_val = df_train_full.iloc[split_val:]

        X_train, y_train = df_train[feature_cols], df_train[TARGET_COL].values
        X_val, y_val = df_val[feature_cols], df_val[TARGET_COL].values
        X_test, y_test = df_test[feature_cols], df_test[TARGET_COL].values

        # -------------------------
        # Native XGBoost DMatrix
        # -------------------------
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

        params = {
            "objective": "reg:squarederror",
            "eta": 0.03,
            "max_depth": 4,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
            "alpha": 0.0,
            "seed": 42,
        }

        # -------------------------
        # Train with early stopping
        # -------------------------
        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=3000,
            evals=[(dval, "val")],
            early_stopping_rounds=75,
            verbose_eval=False,
        )

        # -------------------------
        # Predict
        # -------------------------
        pred_test = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

        test_mae = mae(y_test, pred_test)
        test_rmse = rmse(y_test, pred_test)

        mae_improve = (BASELINE_PERSIST_MAE - test_mae) / BASELINE_PERSIST_MAE * 100.0
        rmse_improve = (BASELINE_PERSIST_RMSE - test_rmse) / BASELINE_PERSIST_RMSE * 100.0

        # -------------------------
        # Save artifacts
        # -------------------------
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        PREDS_DIR.mkdir(parents=True, exist_ok=True)

        bst.save_model(str(MODEL_PATH))

        pd.DataFrame({
            TS_COL: df_test[TS_COL].values,
            "y_true": y_test,
            "y_pred": pred_test,
        }).to_parquet(PREDS_PATH, index=False)

        report = f"""# Phase 2B — XGBoost Price t+1 Results (Native API)

**XGBoost version:** {xgb.__version__}

## Test Performance
- **MAE:** {test_mae:.4f}
- **RMSE:** {test_rmse:.4f}

## Baseline (Persistence)
- MAE: {BASELINE_PERSIST_MAE:.4f}
- RMSE: {BASELINE_PERSIST_RMSE:.4f}

## Improvement vs Baseline
- **MAE improvement:** {mae_improve:.2f}%
- **RMSE improvement:** {rmse_improve:.2f}%

## Notes
- Time-based splits only
- Native early stopping
- Fully leakage-safe
"""

        REPORT_PATH.write_text(report, encoding="utf-8")

        print("\n=== XGBoost Test Metrics ===")
        print(f"MAE  = {test_mae:.4f}  (baseline {BASELINE_PERSIST_MAE:.4f}, improvement {mae_improve:.2f}%)")
        print(f"RMSE = {test_rmse:.4f} (baseline {BASELINE_PERSIST_RMSE:.4f}, improvement {rmse_improve:.2f}%)")

        print("\n[OK] Saved model:", MODEL_PATH)
        print("[OK] Saved report:", REPORT_PATH)
        print("[OK] Saved preds:", PREDS_PATH)

        return 0

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
