from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pd.read_parquet(path)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_model(path: Path) -> Any:
    """
    Supports:
    - XGBoost Booster saved via booster.save_model("model.json")
    - joblib pickle (.pkl)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            f"Fix: export your trained model from Phase-2 and update config.py paths."
        )

    suffix = path.suffix.lower()

    # XGBoost Booster
    if suffix in [".json", ".ubj", ".bin", ".model"]:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster

    # joblib pickle
    if suffix in [".pkl", ".joblib"]:
        import joblib

        return joblib.load(path)

    raise ValueError(f"Unsupported model format: {path} (suffix={suffix})")
