from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class InsightThresholds:
    # Demand thresholds (MW)
    demand_p90: float
    demand_p95: float

    # Price thresholds ($/MWh)
    price_p95: float
    price_p99: float

    # Volatility flag multiplier
    vol_k: float = 2.0


def compute_thresholds(
    demand_history_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    *,
    demand_col: str = "demand_mw",
    price_col: str = "price_per_mwh",
    vol_k: float = 2.0,
    # If your history frames do not contain raw demand/price columns,
    # you can fall back to the target columns or lag columns by passing demand_col/price_col accordingly.
) -> InsightThresholds:
    """
    Compute deterministic thresholds from historical *realized* values.

    Best practice:
    - Demand peak threshold uses demand distribution percentiles (p90/p95).
    - Price regime thresholds use price distribution percentiles (p95/p99).
    """

    # Ensure numeric
    d = pd.to_numeric(demand_history_df[demand_col], errors="coerce").dropna()
    p = pd.to_numeric(price_history_df[price_col], errors="coerce").dropna()

    if len(d) < 1000:
        raise ValueError(f"Demand history too small for stable percentiles: n={len(d)}")
    if len(p) < 1000:
        raise ValueError(f"Price history too small for stable percentiles: n={len(p)}")

    return InsightThresholds(
        demand_p90=float(d.quantile(0.90)),
        demand_p95=float(d.quantile(0.95)),
        price_p95=float(p.quantile(0.95)),
        price_p99=float(p.quantile(0.99)),
        vol_k=float(vol_k),
    )


def derive_demand_insights(
    demand_forecast_df: pd.DataFrame,
    thresholds: InsightThresholds,
    *,
    ts_col: str = "forecast_ts",
    yhat_col: str = "yhat_demand_mw",
    peak_level: str = "p95",
    top_n: int = 3,
) -> Dict[str, Any]:
    """
    Identify peak demand windows in the next 24 forecast hours.
    Returns a dict with:
      - peak_hours: list of timestamps
      - top_peaks: top N hours with highest demand
      - explanation: narrative-ready text
    """
    df = demand_forecast_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df[yhat_col] = pd.to_numeric(df[yhat_col], errors="coerce")

    if peak_level not in {"p90", "p95"}:
        raise ValueError("peak_level must be 'p90' or 'p95'")

    threshold_value = thresholds.demand_p90 if peak_level == "p90" else thresholds.demand_p95

    peaks = df[df[yhat_col] >= threshold_value].sort_values(ts_col)

    top_peaks = df.sort_values(yhat_col, ascending=False).head(top_n)

    explanation = (
        f"Peak demand hours are forecast hours where predicted demand >= historical {peak_level} "
        f"threshold ({threshold_value:,.0f} MW). This is a deterministic percentile rule "
        f"to highlight unusually high-load periods."
    )

    return {
        "threshold_mw": threshold_value,
        "peak_level": peak_level,
        "peak_hours": [t.isoformat() for t in peaks[ts_col].tolist()],
        "top_peaks": [
            {"ts": row[ts_col].isoformat(), "yhat_demand_mw": float(row[yhat_col])}
            for _, row in top_peaks.iterrows()
        ],
        "explanation": explanation,
    }


def derive_price_insights(
    price_forecast_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    thresholds: InsightThresholds,
    *,
    forecast_ts_col: str = "forecast_ts",
    forecast_yhat_col: str = "yhat_price_per_mwh",
    history_ts_col: str = "timestamp",
    history_price_col: str = "price_per_mwh",
    history_roll_std_col: str = "price_roll_std_24",
) -> Dict[str, Any]:
    """
    Classify next-hour price forecast into regime (normal/high/spike),
    and compute a volatility flag based on predicted jump vs typical rolling std.
    """
    fc = price_forecast_df.copy()
    fc[forecast_ts_col] = pd.to_datetime(fc[forecast_ts_col], utc=True)
    fc[forecast_yhat_col] = pd.to_numeric(fc[forecast_yhat_col], errors="coerce")

    if len(fc) != 1:
        raise ValueError("Expected exactly 1-row price forecast (next hour).")

    yhat = float(fc[forecast_yhat_col].iloc[0])
    fts = fc[forecast_ts_col].iloc[0]

    # Regime classification using p95 / p99 thresholds
    if yhat > thresholds.price_p99:
        regime = "spike"
        regime_reason = f"Predicted price {yhat:.2f} > p99 threshold ({thresholds.price_p99:.2f})."
    elif yhat > thresholds.price_p95:
        regime = "high"
        regime_reason = f"Predicted price {yhat:.2f} > p95 threshold ({thresholds.price_p95:.2f})."
    else:
        regime = "normal"
        regime_reason = f"Predicted price {yhat:.2f} <= p95 threshold ({thresholds.price_p95:.2f})."

    # Volatility flag: compare predicted price vs last observed actual
    hist = price_history_df.copy()
    hist[history_ts_col] = pd.to_datetime(hist[history_ts_col], utc=True)
    hist = hist.sort_values(history_ts_col).reset_index(drop=True)

    last_row = hist.tail(1)
    last_ts = last_row[history_ts_col].iloc[0]
    last_price = float(pd.to_numeric(last_row[history_price_col].iloc[0], errors="coerce"))

    # rolling std may already exist; if not, compute from history_price_col
    if history_roll_std_col in hist.columns and pd.notna(last_row[history_roll_std_col].iloc[0]):
        roll_std_24 = float(pd.to_numeric(last_row[history_roll_std_col].iloc[0], errors="coerce"))
    else:
        # fallback computation: last 24 prices std
        tail24 = pd.to_numeric(hist[history_price_col].tail(24), errors="coerce").dropna()
        roll_std_24 = float(tail24.std()) if len(tail24) >= 12 else float("nan")

    delta = abs(yhat - last_price)
    vol_flag = False
    vol_reason = "Volatility check unavailable (missing or unstable rolling std)."

    if pd.notna(roll_std_24) and roll_std_24 > 0:
        vol_flag = delta > thresholds.vol_k * roll_std_24
        vol_reason = (
            f"|pred - last| = {delta:.2f}. Threshold = {thresholds.vol_k:.1f} * std24 "
            f"({roll_std_24:.2f}) = {(thresholds.vol_k * roll_std_24):.2f}. "
            f"Flag={vol_flag}."
        )

    explanation = (
        "Price regime uses deterministic historical percentiles (p95=high, p99=spike). "
        "Volatility flag checks whether the predicted jump is unusually large vs typical 24h variability."
    )

    return {
        "forecast_ts": fts.isoformat(),
        "predicted_price": yhat,
        "regime": regime,
        "regime_reason": regime_reason,
        "volatility_flag": bool(vol_flag),
        "volatility_reason": vol_reason,
        "last_observed_ts": last_ts.isoformat(),
        "last_observed_price": last_price,
        "explanation": explanation,
    }
