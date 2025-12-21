"""
Phase 2B — Step 1: Price validation + target feasibility (t+1)

Inputs (processed, Phase 2A artifacts):
- data/processed/training_frame_24h.parquet

Outputs:
- reports/phase2b_price_validation.md

This script does NOT train a model.
It validates price quality and confirms we can build a leakage-safe t+1 target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/phase2b -> project root
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "training_frame_24h.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "phase2b_price_validation.md"

PRICE_COL = "price_per_mwh"
TS_COL = "timestamp"

# Thresholds for "spike" counts (you can adjust later, but start with something sensible)
SPIKE_THRESHOLDS = [100, 200, 500, 1000]


@dataclass
class ValidationResult:
    n_rows: int
    n_cols: int
    ts_monotonic_increasing: bool
    ts_unique: bool
    ts_min: str
    ts_max: str

    price_missing_pct: float
    price_min: float
    price_max: float
    price_mean: float
    price_std: float
    quantiles: dict

    largest_gap_hours: float
    num_gaps_gt_1h: int

    spike_counts_abs: dict
    target_rows_dropped: int
    target_missing_pct: float


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")


def _compute_time_gaps_hours(ts: pd.Series) -> tuple[float, int]:
    """
    Returns:
      (largest_gap_hours, num_gaps_gt_1h)
    Assumes ts is sorted.
    """
    # Ensure tz-aware timestamps remain intact
    diffs = ts.diff().dropna()
    # diffs is Timedelta; convert to hours
    gaps_hours = diffs.dt.total_seconds() / 3600.0
    largest = float(gaps_hours.max()) if len(gaps_hours) else 0.0
    num_gt_1h = int((gaps_hours > 1.01).sum())  # tolerate tiny drift
    return largest, num_gt_1h


def validate_price_frame(df: pd.DataFrame) -> ValidationResult:
    # Basic shape
    n_rows, n_cols = df.shape

    # Required columns
    missing_cols = [c for c in [TS_COL, PRICE_COL] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}. Found columns: {list(df.columns)}")

    # Timestamp checks
    ts = df[TS_COL]
    ts_monotonic = bool(ts.is_monotonic_increasing)
    ts_unique = bool(ts.is_unique)
    ts_min = str(ts.min())
    ts_max = str(ts.max())

    # Price stats
    price = df[PRICE_COL]
    price_missing_pct = float(price.isna().mean() * 100.0)

    # Dropna for distribution stats (but keep missingness separately)
    price_nonnull = price.dropna()
    if len(price_nonnull) == 0:
        raise ValueError("Price column is entirely null; cannot proceed.")

    price_min = float(price_nonnull.min())
    price_max = float(price_nonnull.max())
    price_mean = float(price_nonnull.mean())
    price_std = float(price_nonnull.std(ddof=1))

    q_levels = [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    q = price_nonnull.quantile(q_levels).to_dict()
    quantiles = {f"p{int(k*100):02d}": float(v) for k, v in q.items()}

    # Time gaps (sort defensively)
    df_sorted = df.sort_values(TS_COL).reset_index(drop=True)
    largest_gap_hours, num_gaps_gt_1h = _compute_time_gaps_hours(df_sorted[TS_COL])

    # Spike counts (absolute)
    spike_counts_abs = {}
    abs_price = price_nonnull.abs()
    for thr in SPIKE_THRESHOLDS:
        spike_counts_abs[f"abs_gt_{thr}"] = int((abs_price > thr).sum())

    # Target feasibility: t+1 hour label = shift(-1)
    target = df_sorted[PRICE_COL].shift(-1)
    # Rows dropped: last row + any rows where shifted target becomes NaN
    target_rows_dropped = int(target.isna().sum())
    target_missing_pct = float(target.isna().mean() * 100.0)

    return ValidationResult(
        n_rows=n_rows,
        n_cols=n_cols,
        ts_monotonic_increasing=ts_monotonic,
        ts_unique=ts_unique,
        ts_min=ts_min,
        ts_max=ts_max,
        price_missing_pct=price_missing_pct,
        price_min=price_min,
        price_max=price_max,
        price_mean=price_mean,
        price_std=price_std,
        quantiles=quantiles,
        largest_gap_hours=largest_gap_hours,
        num_gaps_gt_1h=num_gaps_gt_1h,
        spike_counts_abs=spike_counts_abs,
        target_rows_dropped=target_rows_dropped,
        target_missing_pct=target_missing_pct,
    )


def write_report(res: ValidationResult) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Simple “interpretation helpers” (not gospel, just flags)
    flags = []
    if not res.ts_monotonic_increasing:
        flags.append("Timestamps are not monotonic increasing (sorting required everywhere).")
    if not res.ts_unique:
        flags.append("Duplicate timestamps detected (must deduplicate or aggregate before modeling).")
    if res.price_missing_pct > 1.0:
        flags.append(f"Price missingness is >1% ({res.price_missing_pct:.2f}%). Investigate gaps.")
    if res.num_gaps_gt_1h > 0:
        flags.append(f"Detected {res.num_gaps_gt_1h} timestamp gaps >1 hour (largest gap: {res.largest_gap_hours:.2f}h).")
    if res.quantiles.get("p99", 0.0) - res.quantiles.get("p50", 0.0) > 200:
        flags.append("Heavy tail appears strong (p99 far from median). Expect spike-driven errors.")

    flag_block = "\n".join([f"- ⚠️ {f}" for f in flags]) if flags else "- ✅ No obvious red flags triggered by simple heuristics."

    q_lines = "\n".join([f"- **{k}**: {v:,.4f}" for k, v in res.quantiles.items()])

    spike_lines = "\n".join([f"- **{k}**: {v:,}" for k, v in res.spike_counts_abs.items()])

    md = f"""# Phase 2B — Price Validation Report (t+1 target feasibility)

**Input:** `{DATA_PATH.as_posix()}`  
**Generated:** {pd.Timestamp.utcnow().isoformat()}Z

---

## 1) Frame Overview
- Rows: **{res.n_rows:,}**
- Columns: **{res.n_cols:,}**
- Timestamp range: **{res.ts_min} → {res.ts_max}**
- Timestamp monotonic increasing: **{res.ts_monotonic_increasing}**
- Timestamp unique: **{res.ts_unique}**

---

## 2) Price Column Health (`{PRICE_COL}`)
- Missingness: **{res.price_missing_pct:.4f}%**
- Min: **{res.price_min:,.4f}**
- Max: **{res.price_max:,.4f}**
- Mean: **{res.price_mean:,.4f}**
- Std: **{res.price_std:,.4f}**

### Quantiles
{q_lines}

### Spike counts (absolute price)
{spike_lines}

---

## 3) Timestamp Continuity
- Largest gap (hours): **{res.largest_gap_hours:.4f}**
- Number of gaps > 1h: **{res.num_gaps_gt_1h:,}**

---

## 4) Target Feasibility (t+1)
Target definition (conceptual):
- `target_price_per_mwh_t_plus_1 = price_per_mwh.shift(-1)`

- Rows dropped due to target NaNs: **{res.target_rows_dropped:,}**
- Target missingness: **{res.target_missing_pct:.4f}%**

Interpretation:
- Expect at least **1 dropped row** (last timestamp) in a continuous series.
- More than 1 suggests either missing price values or discontinuities.

---

## 5) Quick Flags / Next Actions
{flag_block}

### Recommended next step
Implement baselines (persistence + 24h rolling mean) using a time-aware split + monthly rolling backtest.

"""

    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"[OK] Wrote report: {REPORT_PATH}")


def main() -> int:
    try:
        _ensure_exists(DATA_PATH)
        df = pd.read_parquet(DATA_PATH)
        res = validate_price_frame(df)
        write_report(res)

        # Print a compact console summary too
        print("\n=== Phase 2B Price Validation Summary ===")
        print(f"Rows: {res.n_rows:,} | Cols: {res.n_cols:,}")
        print(f"TS monotonic: {res.ts_monotonic_increasing} | TS unique: {res.ts_unique}")
        print(f"Price missing %: {res.price_missing_pct:.4f}%")
        print(f"Price p50: {res.quantiles.get('p50', float('nan')):,.4f} | p99: {res.quantiles.get('p99', float('nan')):,.4f}")
        print(f"Gaps >1h: {res.num_gaps_gt_1h:,} | Largest gap hours: {res.largest_gap_hours:.4f}")
        print(f"Target dropped rows: {res.target_rows_dropped:,} | Target missing %: {res.target_missing_pct:.4f}%")
        return 0

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
