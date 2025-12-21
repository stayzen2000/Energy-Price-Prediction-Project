"""
Phase 2B — Step 3: Baseline models for next-hour price forecasting (Option A: single time split)

Baselines:
1) Persistence:   pred(t+1) = price(t)
2) Rolling mean:  pred(t+1) = mean(price[t-23..t])  (24h trailing mean, causal)

Input:
- data/processed/training_frame_price_1h.parquet

Outputs:
- reports/phase2b_baselines_price_1h.md
"""

from __future__ import annotations

from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "training_frame_price_1h.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUT_REPORT = REPORTS_DIR / "phase2b_baselines_price_1h.md"

TS_COL = "timestamp"
PRICE_COL = "price_per_mwh"
TARGET_COL = "target_price_per_mwh_t_plus_1"

# Option A: single time split
TEST_FRACTION = 0.20  # last 20% of rows = test (time-ordered)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> int:
    try:
        if not IN_PATH.exists():
            raise FileNotFoundError(f"Missing input parquet: {IN_PATH}")

        df = pd.read_parquet(IN_PATH).sort_values(TS_COL).reset_index(drop=True)

        # Basic sanity checks
        required = [TS_COL, PRICE_COL, TARGET_COL]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        if df[[PRICE_COL, TARGET_COL]].isna().any().any():
            raise ValueError("Found NaNs in price/target in training_frame_price_1h. This should have been dropped.")

        n = len(df)
        test_n = int(round(n * TEST_FRACTION))
        split_idx = n - test_n

        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()

        # -------------------------
        # Baseline 1: Persistence
        # pred(t+1) = price(t)
        # On the frame at time t, we already have price(t) and target_price(t+1).
        # -------------------------
        test["pred_persistence"] = test[PRICE_COL]

        # -------------------------
        # Baseline 2: 24h trailing mean (causal)
        # pred(t+1) = mean(price[t-23..t])
        #
        # IMPORTANT: rolling must be computed on the full series, then sliced,
        # so the test rows' rolling mean uses only past values.
        # -------------------------
        df["pred_rollmean_24h"] = df[PRICE_COL].rolling(window=24, min_periods=24).mean()
        test = test.merge(
            df[[TS_COL, "pred_rollmean_24h"]],
            on=TS_COL,
            how="left",
            validate="one_to_one",
        )

        # Rolling mean will be NaN for the first 23 rows of the whole dataset,
        # which might land inside train; but if the split is late, test should mostly be fine.
        # Still, we drop NaNs for fair evaluation.
        def eval_baseline(pred_col: str) -> dict:
            sub = test[[TS_COL, TARGET_COL, pred_col]].dropna()
            y_true = sub[TARGET_COL].to_numpy(dtype=float)
            y_pred = sub[pred_col].to_numpy(dtype=float)
            return {
                "rows_evaluated": int(len(sub)),
                "mae": mae(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "test_start": str(sub[TS_COL].min()),
                "test_end": str(sub[TS_COL].max()),
            }

        res_persist = eval_baseline("pred_persistence")
        res_roll24 = eval_baseline("pred_rollmean_24h")

        # -------------------------
        # Write report
        # -------------------------
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        md = f"""# Phase 2B — Baselines (Price t+1, Option A: single time split)

**Input:** `{IN_PATH.as_posix()}`  
**Split:** last {int(TEST_FRACTION*100)}% of rows used as test (time-ordered)  
- Train rows: **{len(train):,}**
- Test rows:  **{len(test):,}**

Target: `{TARGET_COL}` (next-hour price)

---

## Baseline 1 — Persistence
Definition: **pred(t+1) = price(t)**

- Rows evaluated: **{res_persist['rows_evaluated']:,}**
- Test range: **{res_persist['test_start']} → {res_persist['test_end']}**
- **MAE:** {res_persist['mae']:.4f}
- **RMSE:** {res_persist['rmse']:.4f}

---

## Baseline 2 — 24h Rolling Mean (causal)
Definition: **pred(t+1) = mean(price[t-23..t])** (requires 24 hours history)

- Rows evaluated: **{res_roll24['rows_evaluated']:,}**
- Test range: **{res_roll24['test_start']} → {res_roll24['test_end']}**
- **MAE:** {res_roll24['mae']:.4f}
- **RMSE:** {res_roll24['rmse']:.4f}

---

## Notes / Interpretation Guide
- Persistence is often a strong baseline for t+1 price.
- Rolling mean typically smooths noise but will underpredict spikes.
- If an ML model cannot beat persistence on both MAE and RMSE, it is not production-worthy.
"""

        OUT_REPORT.write_text(md, encoding="utf-8")
        print(f"[OK] Wrote report: {OUT_REPORT}")
        print("\n=== Baseline Results (Test) ===")
        print(f"Persistence  MAE={res_persist['mae']:.4f}  RMSE={res_persist['rmse']:.4f}  rows={res_persist['rows_evaluated']:,}")
        print(f"RollMean24h  MAE={res_roll24['mae']:.4f}  RMSE={res_roll24['rmse']:.4f}  rows={res_roll24['rows_evaluated']:,}")
        return 0

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
