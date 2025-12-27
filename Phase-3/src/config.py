from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Phase3Config:
    # ---- Project roots ----
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]  # Phase-3/

    # ---- Phase 2 processed feature frames (local repo relative paths) ----
    PHASE2_ROOT: Path = Path(__file__).resolve().parents[2] / "Phase-2"
    DEMAND_FEATURE_FRAME: Path = PHASE2_ROOT / "data" / "processed" / "features_lagroll_24h.parquet"
    PRICE_FEATURE_FRAME: Path = PHASE2_ROOT / "data" / "processed" / "model_frame_price_1h.parquet"

    # ---- Trained model artifacts (ABSOLUTE Windows paths) ----
    # Update the filenames below to match what you actually have in that folder.
    DEMAND_MODEL_PATH: Path = Path(
    r"C:\Users\aleaf\OneDrive\Desktop\Projects\Energy-Price-Prediction-Project\Phase-2\models\xgb_24h.json"
    )

    PRICE_MODEL_PATH: Path = Path(
    r"C:\Users\aleaf\OneDrive\Desktop\Projects\Energy-Price-Prediction-Project\Phase-2\models\phase2b_xgb_price_1h.json"
    )
    # ---- Output ----
    OUTPUT_DIR: Path = Path(__file__).resolve().parents[1] / "outputs"

    # ---- Column names ----
    TS_COL: str = "timestamp"
    DEMAND_TARGET_COL: str = "target_demand_mw_t_plus_24h"
    PRICE_TARGET_COL: str = "target_price_per_mwh_t_plus_1"

    # ---- Forecast horizons ----
    DEMAND_HORIZON_HOURS: int = 24
    PRICE_HORIZON_HOURS: int = 1


CFG = Phase3Config()
