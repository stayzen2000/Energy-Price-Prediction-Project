# Phase 2B — Baselines (Price t+1, Option A: single time split)

**Input:** `C:/Users/aleaf/OneDrive/Desktop/Projects/Energy-Price-Prediction-Project/Phase-2/data/processed/training_frame_price_1h.parquet`  
**Split:** last 20% of rows used as test (time-ordered)  
- Train rows: **13,700**
- Test rows:  **3,425**

Target: `target_price_per_mwh_t_plus_1` (next-hour price)

---

## Baseline 1 — Persistence
Definition: **pred(t+1) = price(t)**

- Rows evaluated: **3,425**
- Test range: **2025-07-25 06:00:00+00:00 → 2025-12-14 22:00:00+00:00**
- **MAE:** 5.5138
- **RMSE:** 9.0417

---

## Baseline 2 — 24h Rolling Mean (causal)
Definition: **pred(t+1) = mean(price[t-23..t])** (requires 24 hours history)

- Rows evaluated: **3,425**
- Test range: **2025-07-25 06:00:00+00:00 → 2025-12-14 22:00:00+00:00**
- **MAE:** 12.7339
- **RMSE:** 20.8680

---

## Notes / Interpretation Guide
- Persistence is often a strong baseline for t+1 price.
- Rolling mean typically smooths noise but will underpredict spikes.
- If an ML model cannot beat persistence on both MAE and RMSE, it is not production-worthy.
