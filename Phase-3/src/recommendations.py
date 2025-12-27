from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Recommendation:
    type: str              # e.g. "price_spike", "price_high", "demand_peak_watch"
    severity: str          # e.g. "info", "watch", "high"
    message: str           # human-readable action
    why: str               # deterministic trigger explanation
    evidence: Dict[str, Any]  # values that triggered the recommendation


def _rec(
    *,
    rec_type: str,
    severity: str,
    message: str,
    why: str,
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """Return JSON-serializable dict for output bundle."""
    r = Recommendation(
        type=rec_type,
        severity=severity,
        message=message,
        why=why,
        evidence=evidence,
    )
    return {
        "type": r.type,
        "severity": r.severity,
        "message": r.message,
        "why": r.why,
        "evidence": r.evidence,
    }


def generate_recommendations(
    demand_insights: Dict[str, Any],
    price_insights: Dict[str, Any],
    *,
    include_normal_notes: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert deterministic insight signals into conservative, business-friendly recommendations.

    Principles:
    - Every recommendation must map to a specific signal.
    - No claims of guaranteed savings.
    - Keep language generic (works for SMBs).
    """
    recs: List[Dict[str, Any]] = []

    # -------------------------
    # Demand-based recommendations
    # -------------------------
    peak_hours = demand_insights.get("peak_hours", [])
    peak_level = demand_insights.get("peak_level", "p95")
    threshold_mw = demand_insights.get("threshold_mw")

    if peak_hours:
        recs.append(
            _rec(
                rec_type="demand_peak",
                severity="watch" if peak_level == "p90" else "high",
                message=(
                    "Higher-than-usual demand is forecast during the flagged hours. "
                    "If you have flexible operations, consider shifting non-critical energy use "
                    "to lower-demand hours."
                ),
                why=(
                    f"One or more forecast hours exceeded the historical {peak_level} demand threshold."
                ),
                evidence={
                    "peak_level": peak_level,
                    "threshold_mw": threshold_mw,
                    "peak_hours_utc": peak_hours,
                    "top_peaks": demand_insights.get("top_peaks", []),
                },
            )
        )
    elif include_normal_notes:
        recs.append(
            _rec(
                rec_type="demand_normal",
                severity="info",
                message=(
                    "No unusual demand peaks were flagged in the 24-hour forecast window. "
                    "Normal operating plans are likely sufficient."
                ),
                why="No forecast hour exceeded the selected historical demand threshold.",
                evidence={
                    "peak_level": peak_level,
                    "threshold_mw": threshold_mw,
                    "top_peaks": demand_insights.get("top_peaks", []),
                },
            )
        )

    # -------------------------
    # Price-based recommendations
    # -------------------------
    regime = price_insights.get("regime")
    forecast_ts = price_insights.get("forecast_ts")
    predicted_price = price_insights.get("predicted_price")
    vol_flag = price_insights.get("volatility_flag", False)

    # We include p95/p99 evidence if available
    # (these live in thresholds in the bundle, but price_insights has the reason text)
    if regime == "spike":
        recs.append(
            _rec(
                rec_type="price_spike",
                severity="high",
                message=(
                    "A price spike is forecast for the next hour. If feasible, defer discretionary "
                    "energy-intensive tasks (e.g., batch jobs, heavy equipment use) until prices normalize."
                ),
                why="Next-hour price forecast classified as 'spike' using p99 regime threshold.",
                evidence={
                    "forecast_ts_utc": forecast_ts,
                    "predicted_price_per_mwh": predicted_price,
                    "regime_reason": price_insights.get("regime_reason"),
                    "volatility_flag": bool(vol_flag),
                    "volatility_reason": price_insights.get("volatility_reason"),
                },
            )
        )
    elif regime == "high":
        recs.append(
            _rec(
                rec_type="price_high",
                severity="watch",
                message=(
                    "Higher-than-usual prices are forecast for the next hour. Consider monitoring usage "
                    "and shifting non-urgent consumption where possible."
                ),
                why="Next-hour price forecast classified as 'high' using p95 regime threshold.",
                evidence={
                    "forecast_ts_utc": forecast_ts,
                    "predicted_price_per_mwh": predicted_price,
                    "regime_reason": price_insights.get("regime_reason"),
                    "volatility_flag": bool(vol_flag),
                    "volatility_reason": price_insights.get("volatility_reason"),
                },
            )
        )
    elif include_normal_notes:
        recs.append(
            _rec(
                rec_type="price_normal",
                severity="info",
                message=(
                    "No elevated price risk was flagged for the next hour. Standard operating plans are likely sufficient."
                ),
                why="Next-hour price forecast classified as 'normal' (<= p95 threshold).",
                evidence={
                    "forecast_ts_utc": forecast_ts,
                    "predicted_price_per_mwh": predicted_price,
                    "regime_reason": price_insights.get("regime_reason"),
                    "volatility_flag": bool(vol_flag),
                    "volatility_reason": price_insights.get("volatility_reason"),
                },
            )
        )

    # -------------------------
    # Volatility advisory (optional add-on)
    # -------------------------
    # If volatility flag triggers, add a conservative monitoring note.
    if vol_flag:
        recs.append(
            _rec(
                rec_type="price_volatility",
                severity="watch",
                message=(
                    "Price movement is unusually large compared to recent variability. "
                    "Consider monitoring closely and avoiding major discretionary consumption changes until conditions stabilize."
                ),
                why="Volatility flag triggered (predicted jump exceeds k * rolling std).",
                evidence={
                    "forecast_ts_utc": forecast_ts,
                    "predicted_price_per_mwh": predicted_price,
                    "volatility_reason": price_insights.get("volatility_reason"),
                    "last_observed_ts_utc": price_insights.get("last_observed_ts"),
                    "last_observed_price": price_insights.get("last_observed_price"),
                },
            )
        )

    return recs
