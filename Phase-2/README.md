# Phase 2 — Modeling & Forecasting

**Demand (24h Ahead) & Price (Next-Hour)**

Phase 2 of the Energy Intelligence System converts validated, immutable data from Phase 1 into production-grade forecasting models for electricity demand and pricing. This phase prioritizes correctness, interpretability, and honest performance assessment over headline metrics.

---

## 🎯 Objectives

Phase 2 answers a critical question: **Can we generate reliable, leakage-safe short-horizon forecasts for energy demand and prices using realistic modeling assumptions?**

### Key Principles
- **Strict leakage prevention** — No future information in features
- **Time-aware evaluation** — No random shuffling of time series data
- **Strong baseline justification** — Naive models establish credible benchmarks
- **Conservative model selection** — Prioritize robustness over complexity
- **Diagnostic transparency** — Honest reporting of model limitations

---

## 📊 Data Foundation

All modeling uses a **frozen CSV snapshot** from Phase 1.

| Property | Value |
|----------|-------|
| **Source** | Phase 1 validated snapshot |
| **Rows** | ~17,200 hourly observations |
| **Time Range** | 2024-01-01 → 2025-12-17 |
| **Zone** | NYISO (single zone) |
| **Frequency** | Hourly |
| **Timezone** | UTC |
| **Mutability** | Immutable (read-only) |

### Core Columns
- `timestamp` — Hourly timestamp (UTC)
- `zone_id` — Geographic zone identifier
- `demand_mw` — Realized electricity load
- `demand_forecast_mw` — True ex-ante forecast
- `price_per_mwh` — Electricity price
- **Weather**: `temp_c`, `humidity`, `wind_speed`, `precipitation`

No live ingestion, database reads, or schema changes occur in Phase 2.

---

## 🔋 Phase 2A — Demand Forecasting (24-Hour Ahead)

### Objective
Forecast electricity demand 24 hours ahead: **`target_demand_mw_t_plus_24h`**

### Why 24 Hours?
- Strong daily and weekly seasonality in demand
- Operationally meaningful forecast horizon
- Stable modeling baseline before price forecasting

### Data Validation & Cleaning
Validation performed via dedicated scripts in `src/data/`:
- Timestamp continuity and ordering
- Duplicate detection
- Missingness analysis
- Numeric sanity checks

**Key findings:**
- ~5 rows where `demand_mw == 0` → treated as ingestion artifacts
- `price_per_mwh` missingness (~0.45%) ignored during Phase 2A
- Validation summaries written to `reports/`

### Feature Engineering (Leakage-Safe)

All features constructed using **strictly causal logic**.

#### Calendar Features
Cyclical encodings preserve periodic structure:
- Hour of day (sin/cos)
- Day of week (sin/cos)
- Month (sin/cos)
- Weekend indicator

**Output:** `data/processed/features_calendar_24h.parquet`

#### Lag Features
- `demand_lag_1` — Previous hour
- `demand_lag_24` — Same hour yesterday
- `demand_lag_48` — Same hour two days ago
- `demand_lag_168` — Same hour last week

#### Rolling Statistics
- 24h rolling mean & std
- 168h rolling mean & std

**Output:** `data/processed/features_lagroll_24h.parquet`

Rows dropped only where lags were undefined (beginning of dataset).

### Models Evaluated

#### 1. Naive Seasonal Baseline
```
ŷ(t + 24) = demand(t)
```
Strong benchmark due to demand seasonality.

#### 2. Ridge Regression
- Linear model with L2 regularization
- Highly interpretable
- Structured baseline using engineered features

#### 3. XGBoost (Final Model)
- Captures nonlinear interactions
- Conservative hyperparameters
- Early stopping on time-based validation set

### Evaluation Strategy

#### Time-Aware Split
- **Train:** up to 2025-06-30
- **Validation:** 2025-07-01 → 2025-09-30
- **Test:** 2025-10-01 → 2025-11-30

**No random shuffling. No leakage.**

#### Rolling Monthly Backtests
- Expanding training window
- Monthly forward evaluation
- 17–18 rolling windows
- **Output:** `reports/rolling_backtest_*.csv`

### Results Summary

| Model | Median MAE (MW) | Notes |
|-------|----------------|-------|
| Naive (t-24) | ~829 | Strong seasonal benchmark |
| Ridge | ~877 | Stable, interpretable |
| **XGBoost** | **~820** | **Best overall** |

**Observations:**
- XGBoost consistently outperformed linear models
- Summer volatility dominated error behavior
- Ridge occasionally matched XGBoost during stable periods

---

## ⚡ Phase 2B — Price Forecasting (Next-Hour)

### Objective
Forecast electricity price one hour ahead: **`target_price_per_mwh_t_plus_1`**

### Why Next-Hour?
- Electricity prices are noisy and spike-prone
- Short horizons retain predictive signal
- Operationally realistic
- Avoids over-promising long-horizon accuracy

### Price-Specific Validation
- **Missingness:** ~0.31% → dropped
- **Distribution:** Heavy-tailed
  - p95 ≈ $128/MWh
  - p99 ≈ $186/MWh
- **No clipping or transformation applied** (spikes preserved intentionally)

### Baselines

#### 1. Persistence (Primary Baseline)
```
ŷ(t + 1) = price(t)
```
- MAE ≈ 5.51
- RMSE ≈ 9.04

#### 2. Rolling Mean (24h)
Performed significantly worse — demonstrates smoothing alone is insufficient.

### Feature Engineering

#### Added for Price
- **Lagged price:** 1, 2, 3, 6, 24 hours
- **Rolling price statistics:** mean & std (3h, 6h, 24h)

#### Reused from Demand
- Demand & ex-ante demand forecast
- Weather features
- Calendar features

**Output:** `data/processed/model_frame_price_1h.parquet`

### Final Model — XGBoost

- Native XGBoost API (`xgb.train`)
- Conservative parameters
- Early stopping on validation slice
- Time-aware 80/20 split

### Results vs Persistence

| Metric | Persistence | XGBoost | Improvement |
|--------|-------------|---------|-------------|
| **MAE** | 5.51 | **4.04** | **-26.7%** |
| **RMSE** | 9.04 | **7.95** | **-12.1%** |

### Diagnostics & Regime Analysis

#### Normal Regime (≤ p95)
- MAE ≈ 3.33
- Strong, stable improvement

#### High Regime (p95–p99)
- Controlled degradation

#### Spike Regime (> p99)
- Large errors expected
- Driven by grid constraints not present in dataset
- **No evidence of leakage**

#### Time-of-Day Insight
Highest error during evening peak hours (UTC 20–22)

---

## 📓 Notebooks

Notebooks are **diagnostic-only** and contain no production logic.

```
notebooks/
├── phase2_demand_diagnostics.ipynb
└── phase2_price_diagnostics.ipynb
```

**Provide:**
- Error distributions
- Rolling performance visualization
- Regime analysis
- Narrative interpretation

All production logic lives in `src/`.

---

## 📁 Project Structure

### Processed Data
```
data/processed/
├── features_calendar_24h.parquet
├── features_lagroll_24h.parquet
└── model_frame_price_1h.parquet
```

### Reports
```
reports/
├── rolling_backtest_*.csv
├── phase2b_*
└── validation summaries & diagnostics
```

All outputs are versioned and reproducible.

---

## ✅ Phase 2 Status

| Component | Status |
|-----------|--------|
| **Phase 2A — Demand** | ✅ Locked & complete |
| **Phase 2B — Price** | ✅ Locked & complete |

### Criteria Met
- ✅ Leakage-safe modeling
- ✅ Strong baselines
- ✅ Honest diagnostics
- ✅ Reproducible artifacts
- ✅ Defensible claims

---

## 🔮 Next Steps: Phase 3

Phase 2 establishes forecasting primitives.

**Phase 3 will focus on:**
- Decision support systems
- Alerts & thresholds
- Operational insights
- Translating forecasts into actionable recommendations

---

## 🛠️ Usage

### Prerequisites
- Python 3.8+
- Dependencies in `requirements.txt`
- Phase 1 validated data snapshot

### Running Models
```bash
# Feature engineering
python src/data/build_features.py

# Train demand model
python src/models/train_demand_24h.py

# Train price model
python src/models/train_price_1h.py

# Generate diagnostics
jupyter notebook notebooks/phase2_demand_diagnostics.ipynb
```

### Reproducibility
All random seeds, data hashes, and parameters are logged for full reproducibility.

