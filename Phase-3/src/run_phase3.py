from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from config import CFG
from io_utils import read_parquet, load_model, save_json, ensure_dir
from inference import forecast_next_24h_demand, forecast_next_1h_price


def get_feature_cols_from_booster(booster):
    """
    Best practice: use the model's recorded feature_names to prevent schema drift and
    accidental inclusion of non-numeric columns (e.g., 'NYISO').
    """
    feature_names = getattr(booster, "feature_names", None)
    if feature_names is None:
        raise ValueError(
            "This XGBoost Booster has no stored feature_names.\n\n"
            "Fix options:\n"
            "1) BEST: In Phase-2 training, pass feature_names into xgb.DMatrix(...) and re-save the model.\n"
            "2) GOOD: Save a feature_list.json during Phase-2 and load it here.\n"
            "3) QUICK: Fall back to numeric-only columns (less strict, but works).\n"
        )
    return list(feature_names)


def main() -> None:
    print("=== Phase 3 (Forecast-Only Runner) ===")

    # --- Load feature frames ---
    print(f"\n[1/4] Loading demand feature frame: {CFG.DEMAND_FEATURE_FRAME}")
    demand_df = read_parquet(CFG.DEMAND_FEATURE_FRAME)

    print(f"[1/4] Loading price feature frame:  {CFG.PRICE_FEATURE_FRAME}")
    price_df = read_parquet(CFG.PRICE_FEATURE_FRAME)

    # Ensure timestamps parse
    demand_df[CFG.TS_COL] = pd.to_datetime(demand_df[CFG.TS_COL], utc=True)
    price_df[CFG.TS_COL] = pd.to_datetime(price_df[CFG.TS_COL], utc=True)

    print(f"Demand frame rows={len(demand_df):,}  ts_max={demand_df[CFG.TS_COL].max()}")
    print(f"Price  frame rows={len(price_df):,}   ts_max={price_df[CFG.TS_COL].max()}")

    # --- Load models ---
    print(f"\n[2/4] Loading demand model: {CFG.DEMAND_MODEL_PATH}")
    demand_model = load_model(CFG.DEMAND_MODEL_PATH)

    print(f"[2/4] Loading price model:  {CFG.PRICE_MODEL_PATH}")
    price_model = load_model(CFG.PRICE_MODEL_PATH)

    # --- Feature columns: use model feature names (prevents string cols like 'NYISO') ---
    demand_feature_cols = get_feature_cols_from_booster(demand_model)
    price_feature_cols = get_feature_cols_from_booster(price_model)

    print(f"\n[3/4] Demand feature cols (from model): {len(demand_feature_cols)}")
    print(f"[3/4] Price  feature cols (from model): {len(price_feature_cols)}")

    # --- Forecast ---
    print("\n[4/4] Running forecasts...")
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

    # --- Print to terminal ---
    print("\n=== Demand Forecast (next 24 hourly points) ===")
    print(demand_forecast.head(5).to_string(index=False))
    print("...")
    print(demand_forecast.tail(5).to_string(index=False))

    print("\n=== Price Forecast (next hour) ===")
    print(price_forecast.to_string(index=False))

    # --- Save output bundle ---
    run_ts = datetime.now(timezone.utc).isoformat()
    out = {
        "generated_at_utc": run_ts,
        "demand_forecast": demand_forecast.to_dict(orient="records"),
        "price_forecast": price_forecast.to_dict(orient="records"),
    }

    ensure_dir(CFG.OUTPUT_DIR)
    out_path = CFG.OUTPUT_DIR / f"phase3_forecasts_only_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    save_json(out, out_path)

    print(f"\nSaved forecast bundle → {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
