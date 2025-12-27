from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd


def _get_booster_feature_names(model: Any) -> Optional[List[str]]:
    # XGBoost Booster stores feature names here (sometimes None if trained without them)
    if model.__class__.__name__ == "Booster":
        return list(model.feature_names) if model.feature_names else None
    return None

def _predict_any_model(model: Any, X: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    X_use = X[feature_cols].copy()

    # Force numeric conversion (anything non-numeric becomes NaN)
    X_use = X_use.apply(pd.to_numeric, errors="coerce")

    # If NaNs appear, that means your feature_cols still included non-numeric columns
    if X_use.isna().any().any():
        bad_cols = X_use.columns[X_use.isna().any()].tolist()
        raise ValueError(
            "Non-numeric or invalid values detected in features after coercion.\n"
            f"Bad columns: {bad_cols}\n"
            "Fix: ensure your feature_cols contain only numeric model features."
        )

    # Native XGBoost Booster path
    if model.__class__.__name__ == "Booster":
        import xgboost as xgb

        dmat = xgb.DMatrix(X_use, feature_names=feature_cols)
        preds = model.predict(dmat)
        return pd.Series(preds, index=X_use.index)

    # sklearn-like path
    if hasattr(model, "predict"):
        preds = model.predict(X_use)
        return pd.Series(preds, index=X_use.index)

    raise TypeError(f"Unsupported model type: {type(model)}")


def forecast_next_24h_demand(
    feature_df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    ts_col: str = "timestamp",
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """
    Demand model is trained as: row at time t -> predicts demand at t + 24.
    To forecast the *next 24 forecast hours*, take the last 24 rows, predict each,
    then shift each row's timestamp forward by +24h to label forecast timestamps.
    """
    df = feature_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col).reset_index(drop=True)

    last_rows = df.tail(horizon_hours).copy()

    preds = _predict_any_model(model, last_rows, feature_cols)

    out = pd.DataFrame(
        {
            "feature_ts": last_rows[ts_col].values,
            "forecast_ts": (last_rows[ts_col] + pd.Timedelta(hours=horizon_hours)).values,
            "yhat_demand_mw": preds.values,
        }
    )
    return out.sort_values("forecast_ts").reset_index(drop=True)


def forecast_next_1h_price(
    feature_df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    ts_col: str = "timestamp",
    horizon_hours: int = 1,
) -> pd.DataFrame:
    """
    Price model is trained as: row at time t -> predicts price at t + 1.
    Use only the last row.
    """
    df = feature_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col).reset_index(drop=True)

    last_row = df.tail(1).copy()

    pred = _predict_any_model(model, last_row, feature_cols).iloc[0]

    out = pd.DataFrame(
        {
            "feature_ts": [last_row[ts_col].iloc[0]],
            "forecast_ts": [last_row[ts_col].iloc[0] + pd.Timedelta(hours=horizon_hours)],
            "yhat_price_per_mwh": [float(pred)],
        }
    )
    return out
