"""
Phase 2B — Step 6: Diagnostics for next-hour price forecasting

Input:
- data/processed/preds_xgb_price_1h.parquet

Output:
- reports/phase2b_diagnostics_price_1h.md

Purpose:
- Understand where the model wins/loses
- Separate normal vs spike regimes to interpret MAE vs RMSE behavior
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "preds_xgb_price_1h.parquet"
OUT_PATH = PROJECT_ROOT / "reports" / "phase2b_diagnostics_price_1h.md"
TS_COL = "timestamp"


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing preds file: {IN_PATH}")

    df = pd.read_parquet(IN_PATH).sort_values(TS_COL).reset_index(drop=True)

    # Basic residuals
    df["err"] = df["y_pred"] - df["y_true"]
    df["abs_err"] = df["err"].abs()

    overall = {
        "rows": len(df),
        "mae": mae(df["y_true"], df["y_pred"]),
        "rmse": rmse(df["y_true"], df["y_pred"]),
    }

    # Regime thresholds based on true price distribution in the test set
    p95 = float(df["y_true"].quantile(0.95))
    p99 = float(df["y_true"].quantile(0.99))

    def metrics(sub):
        return {
            "rows": int(len(sub)),
            "mae": mae(sub["y_true"], sub["y_pred"]),
            "rmse": rmse(sub["y_true"], sub["y_pred"]),
            "median_abs_err": float(sub["abs_err"].median()),
        }

    normal_0_95 = df[df["y_true"] <= p95]
    high_95_99 = df[(df["y_true"] > p95) & (df["y_true"] <= p99)]
    spike_top_1 = df[df["y_true"] > p99]

    m_normal = metrics(normal_0_95)
    m_high = metrics(high_95_99)
    m_spike = metrics(spike_top_1)

    # Time-of-day breakdown (optional but very useful)
    # timestamp is tz-aware; .dt.hour works fine
    df["hour_utc"] = pd.to_datetime(df[TS_COL]).dt.hour
    by_hour = (
    df.groupby("hour_utc", group_keys=False)
      .apply(lambda g: pd.Series({
          "rows": len(g),
          "mae": mae(g["y_true"], g["y_pred"]),
          "rmse": rmse(g["y_true"], g["y_pred"]),
      }))
      .reset_index()
)

    # Write report
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    hour_table = "\n".join(
        [f"| {int(r.hour_utc):02d} | {int(r.rows)} | {r.mae:.4f} | {r.rmse:.4f} |"
         for r in by_hour.itertuples(index=False)]
    )

    md = f"""# Phase 2B — Diagnostics (Price t+1)

**Input:** `{IN_PATH.as_posix()}`  
Test range: **{df[TS_COL].min()} → {df[TS_COL].max()}**

---

## 1) Overall Performance
- Rows: **{overall['rows']:,}**
- **MAE:** {overall['mae']:.4f}
- **RMSE:** {overall['rmse']:.4f}

---

## 2) Regime Breakdown (by true price quantiles)
Thresholds computed on test set:
- p95 (high-price threshold): **{p95:.4f}**
- p99 (spike threshold): **{p99:.4f}**

### Normal (<= p95)
- Rows: **{m_normal['rows']:,}**
- MAE: {m_normal['mae']:.4f}
- RMSE: {m_normal['rmse']:.4f}
- Median |error|: {m_normal['median_abs_err']:.4f}

### High (p95–p99)
- Rows: **{m_high['rows']:,}**
- MAE: {m_high['mae']:.4f}
- RMSE: {m_high['rmse']:.4f}
- Median |error|: {m_high['median_abs_err']:.4f}

### Spike (top 1% > p99)
- Rows: **{m_spike['rows']:,}**
- MAE: {m_spike['mae']:.4f}
- RMSE: {m_spike['rmse']:.4f}
- Median |error|: {m_spike['median_abs_err']:.4f}

Interpretation guide:
- Expect errors to blow up in the spike regime.
- The goal is not “perfect spikes” but improved normal-regime accuracy **without** collapsing in spikes.

---

## 3) Performance by Hour (UTC)
| Hour | Rows | MAE | RMSE |
|---:|---:|---:|---:|
{hour_table}
"""

    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"[OK] Wrote diagnostics report: {OUT_PATH}")


if __name__ == "__main__":
    main()
