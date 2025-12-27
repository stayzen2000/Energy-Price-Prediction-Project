from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from config import CFG
from io_utils import read_parquet, load_model, save_json, ensure_dir
from inference import forecast_next_24h_demand, forecast_next_1h_price
from insights import compute_thresholds, derive_demand_insights, derive_price_insights
from recommendations import generate_recommendations


def get_feature_cols_from_booster(booster):
    feature_names = getattr(booster, "feature_names", None)
    if feature_names is None:
        raise ValueError(
            "This XGBoost Booster has no stored feature_names.\n"
            "Fix: re-save model with feature_names or load a saved feature list."
        )
    return list(feature_names)


def main() -> None:
    print("RUNNER VERSION: v3_with_recs_2025-12-27")
    print("=== Phase 3 (Forecast + Insights + Recommendations Runner) ===")

    # --- Load feature frames (Phase 2, read-only) ---
    print(f"\n[1/6] Loading demand feature frame: {CFG.DEMAND_FEATURE_FRAME}")
    demand_df = read_parquet(CFG.DEMAND_FEATURE_FRAME)

    print(f"[1/6] Loading price feature frame:  {CFG.PRICE_FEATURE_FRAME}")
    price_df = read_parquet(CFG.PRICE_FEATURE_FRAME)

    # Parse timestamps
    demand_df[CFG.TS_COL] = pd.to_datetime(demand_df[CFG.TS_COL], utc=True)
    price_df[CFG.TS_COL] = pd.to_datetime(price_df[CFG.TS_COL], utc=True)

    demand_asof_ts = demand_df[CFG.TS_COL].max()
    price_asof_ts = price_df[CFG.TS_COL].max()

    print(f"Demand frame rows={len(demand_df):,}  as_of(ts_max)={demand_asof_ts}")
    print(f"Price  frame rows={len(price_df):,}   as_of(ts_max)={price_asof_ts}")

    # --- Load models (Phase 2, read-only) ---
    print(f"\n[2/6] Loading demand model: {CFG.DEMAND_MODEL_PATH}")
    demand_model = load_model(CFG.DEMAND_MODEL_PATH)

    print(f"[2/6] Loading price model:  {CFG.PRICE_MODEL_PATH}")
    price_model = load_model(CFG.PRICE_MODEL_PATH)

    # --- Feature columns from model ---
    demand_feature_cols = get_feature_cols_from_booster(demand_model)
    price_feature_cols = get_feature_cols_from_booster(price_model)

    print(f"\n[3/6] Demand feature cols (from model): {len(demand_feature_cols)}")
    print(f"[3/6] Price  feature cols (from model): {len(price_feature_cols)}")

    # --- Forecasts ---
    print("\n[4/6] Running forecasts...")
    demand_forecast = forecast_next_24h_demand(
        feature_df=demand_df,
        model=demand_model,
        feature_cols=demand_feature_cols,
        ts_col=CFG.TS_COL,
        horizon_hours=CFG.DEMAND_HORIZON_HOURS,
    )

    price_forecast = forecast_next_1h_price(
        feature_df=price_df,
        model=price_model,
        feature_cols=price_feature_cols,
        ts_col=CFG.TS_COL,
        horizon_hours=CFG.PRICE_HORIZON_HOURS,
    )

    # Forecast windows (for clean labeling)
    demand_window_start = demand_forecast["forecast_ts"].min()
    demand_window_end = demand_forecast["forecast_ts"].max()
    price_forecast_ts = price_forecast["forecast_ts"].iloc[0]

    print("\n=== Demand Forecast (24 rows) ===")
    print(f"As-of (feature timestamp): {demand_asof_ts}")
    print(f"Forecast window: {demand_window_start} → {demand_window_end}")
    print(demand_forecast.head(3).to_string(index=False))
    print("...")
    print(demand_forecast.tail(3).to_string(index=False))

    print("\n=== Price Forecast (1 row) ===")
    print(f"As-of (feature timestamp): {price_asof_ts}")
    print(f"Forecast timestamp: {price_forecast_ts}")

    print(f"price_forecast shape: {price_forecast.shape}")
    print("price_forecast columns:", list(price_forecast.columns))

    # Bulletproof printing
    print("price_forecast row dict:", price_forecast.iloc[0].to_dict())
    print("\nprice_forecast table:")
    print(price_forecast.to_string(index=False))


    # --- Thresholds + Insights (deterministic, no ML) ---
    print("\n[5/6] Computing thresholds and deriving insights...")

    thresholds = compute_thresholds(
        demand_history_df=demand_df,
        price_history_df=price_df,
        demand_col="demand_lag_1",      # realized demand proxy at time t
        price_col="price_per_mwh",      # realized price
        vol_k=2.0,
    )

    demand_insights = derive_demand_insights(
        demand_forecast_df=demand_forecast,
        thresholds=thresholds,
        ts_col="forecast_ts",
        yhat_col="yhat_demand_mw",
        peak_level="p95",   # change to "p90" if you want a watchlist
        top_n=3,
    )

    price_insights = derive_price_insights(
        price_forecast_df=price_forecast,
        price_history_df=price_df,
        thresholds=thresholds,
        forecast_ts_col="forecast_ts",
        forecast_yhat_col="yhat_price_per_mwh",
        history_ts_col="timestamp",
        history_price_col="price_per_mwh",
        history_roll_std_col="price_roll_std_24",
    )

    # --- Recommendations (generic, conservative) ---
    print("\n[6/6] Generating recommendations...")
    recommendations = generate_recommendations(
        demand_insights=demand_insights,
        price_insights=price_insights,
        include_normal_notes=True,
    )

    # Print insight summary
    print("\n=== INSIGHTS SUMMARY ===")
    print(f"Demand peak threshold ({demand_insights['peak_level']}): {demand_insights['threshold_mw']:.0f} MW")
    print(f"Peak hours flagged: {len(demand_insights['peak_hours'])}")
    print("Top demand peaks:")
    for item in demand_insights["top_peaks"]:
        print(f"  - {item['ts']}  →  {item['yhat_demand_mw']:.0f} MW")

    print("\nPrice regime:")
    print(f"  Forecast: {price_insights['predicted_price']:.2f} $/MWh at {price_insights['forecast_ts']}")
    print(f"  Regime:   {price_insights['regime']}  ({price_insights['regime_reason']})")
    print(f"  Vol flag: {price_insights['volatility_flag']}  ({price_insights['volatility_reason']})")

    # Print recommendations
    print("\n=== RECOMMENDATIONS ===")
    for r in recommendations:
        print(f"- [{r['severity'].upper()}] {r['type']}: {r['message']}")
        print(f"  Why: {r['why']}")
        # keep evidence short in terminal
        ev = r.get("evidence", {})
        if "forecast_ts_utc" in ev:
            print(f"  Evidence: ts={ev.get('forecast_ts_utc')} price={ev.get('predicted_price_per_mwh')}")
        print()

    # --- Save final bundle ---
    run_ts = datetime.now(timezone.utc).isoformat()

    bundle = {
        "generated_at_utc": run_ts,
        "as_of": {
            "demand_feature_ts_max_utc": str(demand_asof_ts),
            "price_feature_ts_max_utc": str(price_asof_ts),
            "demand_forecast_window_utc": [str(demand_window_start), str(demand_window_end)],
            "price_forecast_ts_utc": str(price_forecast_ts),
        },
        "limitations": {
            "frozen_snapshot": True,
            "statement": (
                "This project uses a frozen, immutable dataset snapshot for reproducibility and auditability. "
                "Forecasts are generated relative to the latest available timestamps in the Phase-2 feature frames "
                "(as_of values above), not the current wall-clock time. In production, these same trained models "
                "would consume live, continuously materialized features from real-time ingestion; that serving layer "
                "is intentionally out of scope for this portfolio build."
            ),
            "note_on_alignment": (
                "Demand and price feature frames may end on different timestamps, so their 'as_of' times and forecast "
                "times can differ. This is surfaced explicitly to avoid confusion."
            ),
        },
        "thresholds": {
            "demand_p90": thresholds.demand_p90,
            "demand_p95": thresholds.demand_p95,
            "price_p95": thresholds.price_p95,
            "price_p99": thresholds.price_p99,
            "vol_k": thresholds.vol_k,
        },
        "forecasts": {
            "demand_24h": demand_forecast.to_dict(orient="records"),
            "price_1h": price_forecast.to_dict(orient="records"),
        },
        "insights": {
            "demand": demand_insights,
            "price": price_insights,
        },
        "recommendations": recommendations,
    }

    ensure_dir(CFG.OUTPUT_DIR)
    out_path = CFG.OUTPUT_DIR / f"phase3_bundle_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    save_json(bundle, out_path)

    print(f"\nSaved Phase-3 bundle → {out_path}")
    print_presentation_summary(
    demand_forecast=demand_forecast,
    price_forecast=price_forecast,
    demand_insights=demand_insights,
    price_insights=price_insights,
    recommendations=recommendations,
    demand_asof=demand_asof_ts,
    price_asof=price_asof_ts,
)
    print("Done.")


def print_presentation_summary(
    *,
    demand_forecast,
    price_forecast,
    demand_insights,
    price_insights,
    recommendations,
    demand_asof,
    price_asof,
):
    print("\n" + "=" * 60)
    print("PHASE 3 — FORECAST, INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)

    print("\nAs-of timestamps:")
    print(f"• Demand data: {demand_asof}")
    print(f"• Price data:  {price_asof}")

    # ---------------- Demand ----------------
    print("\n" + "-" * 50)
    print("DEMAND FORECAST (Next 24 Hours)")
    print(f"Forecast window: {demand_forecast['forecast_ts'].min()} → {demand_forecast['forecast_ts'].max()}")
    print("-" * 50)

    df = demand_forecast.copy()
    df["hour"] = df["forecast_ts"].dt.strftime("%H:%M")
    df["mw"] = df["yhat_demand_mw"].round(0).astype(int)

    print("Hour (UTC)        Forecast Demand (MW)")
    print("-" * 38)

    for _, row in df.head(3).iterrows():
        print(f"{row['hour']:<16} {row['mw']:>8,}")

    print("...")

    for _, row in df.tail(3).iterrows():
        print(f"{row['hour']:<16} {row['mw']:>8,}")

    # ---------------- Price ----------------
    print("\n" + "-" * 50)
    print("PRICE FORECAST (Next Hour)")
    print("-" * 50)

    price_row = price_forecast.iloc[0]

    print(f"Forecast hour:    {price_row['forecast_ts']}")
    print(f"Predicted price:  ${price_row['yhat_price_per_mwh']:.2f} / MWh")
    print(f"Price regime:     {price_insights['regime'].upper()}")
    print(f"Volatility flag:  {price_insights['volatility_flag']}")

    # ---------------- Insights ----------------
    print("\nINSIGHTS SUMMARY")
    print(f"• Demand peak threshold (p95): {int(demand_insights['threshold_mw']):,} MW")
    print(f"• Peak demand hours flagged: {len(demand_insights['peak_hours'])}")

    top_peaks = demand_insights.get("top_peaks", [])
    if top_peaks:
        print("• Top demand hours:")

        first = top_peaks[0]

        # Case A: list of dicts: {"ts": ..., "yhat_demand_mw": ...}
        if isinstance(first, dict):
            for item in top_peaks:
                ts = item.get("ts")
                mw = item.get("yhat_demand_mw")
                if mw is None:
                    continue
                print(f"  - {ts} → {int(round(float(mw))):,} MW")

        # Case B: list of tuples: (ts, mw)
        else:
            for ts, mw in top_peaks:
                print(f"  - {ts} → {int(round(float(mw))):,} MW")


    # ---------------- Recommendations ----------------
    print("\nRECOMMENDATIONS")
    for r in recommendations:
        sev = r["severity"].upper()
        print(f"[{sev}] {r['type']}: {r['message']}")
        print(f"  Why: {r['why']}")
        if "predicted_price_per_mwh" in r.get("evidence", {}):
            print(f"  Evidence: price=${r['evidence']['predicted_price_per_mwh']:.2f}")
        print()

    print("-" * 50)


if __name__ == "__main__":
    main()
