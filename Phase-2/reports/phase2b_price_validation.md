# Phase 2B — Price Validation Report (t+1 target feasibility)

**Input:** `C:/Users/aleaf/OneDrive/Desktop/Projects/Energy-Price-Prediction-Project/Phase-2/data/processed/training_frame_24h.parquet`  
**Generated:** 2025-12-20T22:33:21.153155+00:00Z

---

## 1) Frame Overview
- Rows: **17,179**
- Columns: **10**
- Timestamp range: **2024-01-01 00:00:00+00:00 → 2025-12-16 23:00:00+00:00**
- Timestamp monotonic increasing: **True**
- Timestamp unique: **True**

---

## 2) Price Column Health (`price_per_mwh`)
- Missingness: **0.3085%**
- Min: **13.6147**
- Max: **363.6460**
- Mean: **47.0758**
- Std: **33.0107**

### Quantiles
- **p00**: 13.6147
- **p01**: 17.4053
- **p05**: 20.4282
- **p25**: 27.3885
- **p50**: 36.5620
- **p75**: 52.3900
- **p95**: 115.8932
- **p99**: 178.0022
- **p100**: 363.6460

### Spike counts (absolute price)
- **abs_gt_100**: 1,281
- **abs_gt_200**: 116
- **abs_gt_500**: 0
- **abs_gt_1000**: 0

---

## 3) Timestamp Continuity
- Largest gap (hours): **3.0000**
- Number of gaps > 1h: **3**

---

## 4) Target Feasibility (t+1)
Target definition (conceptual):
- `target_price_per_mwh_t_plus_1 = price_per_mwh.shift(-1)`

- Rows dropped due to target NaNs: **53**
- Target missingness: **0.3085%**

Interpretation:
- Expect at least **1 dropped row** (last timestamp) in a continuous series.
- More than 1 suggests either missing price values or discontinuities.

---

## 5) Quick Flags / Next Actions
- ⚠️ Detected 3 timestamp gaps >1 hour (largest gap: 3.00h).

### Recommended next step
Implement baselines (persistence + 24h rolling mean) using a time-aware split + monthly rolling backtest.

