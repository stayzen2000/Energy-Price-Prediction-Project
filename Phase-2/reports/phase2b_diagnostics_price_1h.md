# Phase 2B — Diagnostics (Price t+1)

**Input:** `C:/Users/aleaf/OneDrive/Desktop/Projects/Energy-Price-Prediction-Project/Phase-2/data/processed/preds_xgb_price_1h.parquet`  
Test range: **2025-07-25 11:00:00 → 2025-12-14 22:00:00**

---

## 1) Overall Performance
- Rows: **3,420**
- **MAE:** 4.0436
- **RMSE:** 7.9477

---

## 2) Regime Breakdown (by true price quantiles)
Thresholds computed on test set:
- p95 (high-price threshold): **127.9639**
- p99 (spike threshold): **185.9035**

### Normal (<= p95)
- Rows: **3,249**
- MAE: 3.3271
- RMSE: 5.2440
- Median |error|: 2.0232

### High (p95–p99)
- Rows: **136**
- MAE: 12.4503
- RMSE: 17.2924
- Median |error|: 8.6217

### Spike (top 1% > p99)
- Rows: **35**
- MAE: 37.8916
- RMSE: 49.5737
- Median |error|: 28.4333

Interpretation guide:
- Expect errors to blow up in the spike regime.
- The goal is not “perfect spikes” but improved normal-regime accuracy **without** collapsing in spikes.

---

## 3) Performance by Hour (UTC)
| Hour | Rows | MAE | RMSE |
|---:|---:|---:|---:|
| 00 | 142 | 4.5882 | 10.2948 |
| 01 | 142 | 2.6389 | 4.5650 |
| 02 | 142 | 1.4441 | 2.0303 |
| 03 | 142 | 2.5492 | 3.3169 |
| 04 | 142 | 2.5071 | 4.8649 |
| 05 | 142 | 1.9860 | 3.3257 |
| 06 | 142 | 1.9016 | 3.1388 |
| 07 | 142 | 1.4525 | 2.1332 |
| 08 | 142 | 1.8149 | 2.3129 |
| 09 | 142 | 4.0272 | 5.4975 |
| 10 | 142 | 5.9313 | 8.7664 |
| 11 | 143 | 5.8045 | 7.6705 |
| 12 | 143 | 5.2778 | 7.1511 |
| 13 | 143 | 2.8464 | 4.0969 |
| 14 | 143 | 2.3882 | 3.6251 |
| 15 | 143 | 2.1196 | 3.0512 |
| 16 | 143 | 2.4157 | 3.7770 |
| 17 | 143 | 2.5206 | 3.7098 |
| 18 | 143 | 3.5888 | 8.5871 |
| 19 | 143 | 5.3982 | 10.2989 |
| 20 | 143 | 11.1788 | 17.6456 |
| 21 | 143 | 10.9883 | 18.3157 |
| 22 | 143 | 6.0904 | 11.4187 |
| 23 | 142 | 5.5033 | 8.6933 |
