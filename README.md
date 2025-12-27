# ⚡ Energy Intelligence — Demand & Price Forecasting System

## 🔍 Overview

Energy Intelligence is an end-to-end, production-style energy forecasting and decision-support system designed to predict short-horizon electricity demand and prices using real grid and weather data.

The project demonstrates how industry-grade data pipelines, leakage-safe time-series modeling, and disciplined ML evaluation practices are extended into operational decision-making systems — from ingestion to forecasting to insights and recommendations.

## 🎯 Purpose of This Project

Electricity demand and pricing are increasingly volatile due to:

- Weather-driven load swings
- Grid congestion and scarcity events
- Growing demand from power-intensive operations

Most organizations lack tools that allow them to anticipate risk before it materializes.

This project focuses on:

- 🔮 Forecasting short-horizon demand and prices
- 🧠 Translating forecasts into actionable decisions
- 🏗️ Demonstrating production-grade data & ML system design

## 👥 Who This Project Helps

### 💨 HVAC & Energy Services Companies
- Optimize heating and cooling schedules
- Reduce peak-load exposure
- Minimize equipment stress

### 🏢 Building & Facility Managers
- Plan daily energy usage more effectively
- Anticipate high-demand or high-price windows
- Improve cost predictability

### 🏭 Industrial & Manufacturing Sites
- Shift energy-intensive operations to lower-risk periods
- Reduce downtime and overload risk
- Control operational energy spend

### 🖥️ Data Centers
- Manage power-intensive workloads
- Anticipate price volatility
- Improve reliability and cost efficiency

### 🏠 Energy-Aware Consumers
- Understand when electricity is most expensive
- Adjust usage to reduce bills
- Make informed, data-driven decisions

## 💡 How This System Creates Value

Instead of reacting to energy costs after the fact, the system enables users to:

- 📊 Forecast upcoming demand and price movements
- ⚠️ Identify high-risk periods (peaks, volatility regimes)
- 🔄 Adjust operations proactively
- 🧠 Consume insights through interpretable metrics and recommendations

This leads to better planning, lower costs, and reduced operational risk.

## ✨ Key Capabilities (Current)

- Live and historical NYISO grid data ingestion
- Weather-enriched time-series datasets
- Leakage-safe feature engineering (lags, rolling statistics, calendars)
- Demand forecasting (24-hour horizon)
- Price forecasting (next-hour horizon)
- Time-aware evaluation and rolling backtests
- Deterministic insight derivation and recommendations
- Reproducible, structured decision-support outputs

## 🧭 Project Phases

### ✅ Phase 1 — Data Foundation (Completed)

- Integrated live and historical NYISO & EIA data
- Built a normalized PostgreSQL schema for time-series storage
- Dockerized ingestion workflows for reproducibility
- Orchestrated ingestion with n8n

📂 See: `Phase-1/README.md`

### ✅ Phase 2 — Modeling & Forecasting (Completed)

Phase 2 converts validated energy data into defensible forecasting intelligence.

#### Phase 2A — Demand Forecasting (24h Ahead)
- Leakage-safe feature engineering
- Naive seasonal baseline
- Ridge regression baseline
- XGBoost (final model)
- Time-based splits and rolling monthly backtests

#### Phase 2B — Price Forecasting (Next-Hour)
- Persistence baseline (price(t+1) = price(t))
- Price-specific lag & rolling features
- XGBoost with early stopping
- Regime-aware diagnostics (normal / high / spike)
- Time-of-day error analysis

**Key outcome:** XGBoost outperforms strong baselines while degrading honestly under spike conditions.

📂 See: `Phase-2/README.md`  
📓 Diagnostics notebooks included for both demand and price.

### ✅ Phase 3 — Insight & Decision Layer (Completed)

Phase 3 operationalizes the trained forecasting models into business-ready decision intelligence.

Rather than introducing new ML, this phase focuses on how forecasts are consumed in real systems.

#### What Phase 3 Does

- Loads frozen Phase-2 feature frames and trained models
- Runs read-only inference (no retraining, no tuning)
- Generates:
  - 24-hour demand forecasts
  - Next-hour price forecasts
- Derives deterministic insights:
  - Peak demand windows (percentile-based thresholds)
  - Price regimes (normal / high / spike)
  - Volatility flags
- Produces conservative, explainable recommendations
- Packages everything into a single structured output contract (JSON)

This output is designed to be consumed downstream by dashboards, APIs, or conversational interfaces.

📂 See: `Phase-3/README.md`

#### Important Limitation (Intentional)

Phase 3 runs on a frozen dataset snapshot from Phase 2 to preserve:

- Reproducibility
- Auditability
- Leakage safety

As a result:

- Forecasts are generated relative to the latest available feature timestamps
- Live, wall-clock forecasts are intentionally not shown
- Demand and price forecasts may reference different "as-of" times

This is expected behavior and is explicitly logged in outputs. Live ingestion and real-time serving are deferred to Phase 5 (Production Deployment).

### 🔜 Phase 4 — Visualization & Consumption (Planned)

Phase 4 will build user-facing interfaces on top of the Phase-3 output contract:

- Visual dashboards (forecast curves, risk indicators)
- Recommendation panels
- Conversational UI (LLM-powered Q&A grounded in the same contract)

Phase 4 is a presentation layer, not a decision engine.

### 🔜 Phase 5 — Deployment & Monitoring (Planned)

- Scheduled live data ingestion
- Feature materialization for serving
- API-based inference
- Containerized deployment (AWS)
- Logging, monitoring, and data freshness checks

## 🛠️ Tech Stack (Actual, Not Aspirational)

| Component | Technology |
|-----------|------------|
| Language | Python |
| Database | PostgreSQL |
| ML Models | Ridge Regression, XGBoost |
| Time-Series Handling | Pandas, NumPy |
| Workflow Orchestration | n8n |
| Containerization | Docker |
| Diagnostics & Reporting | Jupyter, Markdown |
| Cloud (Planned) | AWS |

Deep learning frameworks (TensorFlow / PyTorch) are intentionally deferred until justified by data and use case.

## 🚦 Why This Project Matters

This project is deliberately built to reflect how real ML systems are developed, not how demo notebooks are written.

It demonstrates:

- 🧱 Production-style data engineering
- ⏱️ Correct time-series modeling discipline
- 📉 Honest baseline comparisons
- 🔍 Diagnostic transparency
- 🧠 Business-aware ML decision-making

## 📌 Current Status

- ✅ Phase 1 — Complete
- ✅ Phase 2 — Complete
- ✅ Phase 3 — Complete
- 🔜 Phase 4 — Planned

## 📂 Project Structure

```
Energy-Intelligence/
├── Phase-1/              # Data ingestion & orchestration
│   └── README.md
├── Phase-2/              # Modeling & forecasting (COMPLETE)
│   ├── src/
│   ├── notebooks/
│   ├── data/processed/
│   ├── reports/
│   └── README.md
├── Phase-3/              # Insight & decision layer (COMPLETE)
│   ├── src/
│   ├── outputs/
│   └── README.md
├── Phase-4/              # Visualization & dashboards (planned)
├── Phase-5/              # Deployment & monitoring (planned)
└── README.md
```

## 🔒 Final Note

- Phase 2 models are locked and reproducible.
- Phase 3 decisions are deterministic and explainable.

All future phases build on this foundation — not the other way around.

## 🚀 Getting Started

*(Add setup instructions here)*

## ✅ Next Recommended Step

Add screenshots or a short GIF of:

- Phase 3 terminal output
- Phase 4 dashboard (once started)

This will significantly increase recruiter engagement.
