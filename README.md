# ⚡ Energy Intelligence — Demand & Price Forecasting System

## 🔍 Overview

Energy Intelligence is an end-to-end, production-style energy forecasting and decision-support system designed to predict short-horizon electricity demand and prices using real grid and weather data.

The project demonstrates how industry-grade data pipelines, leakage-safe time-series modeling, and disciplined ML evaluation practices are built and connected in real-world environments — from ingestion to forecasting to actionable insights.

---

## 🎯 Purpose of This Project

Electricity demand and pricing are increasingly volatile due to:

- Weather-driven load swings
- Grid congestion and scarcity events
- Growing demand from power-intensive operations

Most organizations lack tools that allow them to anticipate risk before it materializes.

**This project focuses on:**

- 🔮 Forecasting short-horizon demand and prices
- 🧠 Enabling proactive operational decision-making
- 🏗️ Demonstrating production-grade data & ML engineering practices

---

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

---

## 💡 How This System Creates Value

Instead of reacting to energy costs after the fact, the system enables users to:

- 📊 Forecast upcoming demand and price movements
- ⚠️ Identify high-risk periods (peaks, volatility regimes)
- 🔄 Adjust operations proactively
- 🧠 Consume insights through interpretable metrics and diagnostics

This leads to better planning, lower costs, and reduced operational risk.

---

## ✨ Key Capabilities (Current)

- Live and historical NYISO grid data ingestion
- Weather-enriched time-series datasets
- Leakage-safe feature engineering (lags, rolling statistics, calendars)
- Demand forecasting (24-hour horizon)
- Price forecasting (next-hour horizon)
- Time-aware evaluation and rolling backtests
- Diagnostic notebooks and reproducible reports

> ⚠️ **Note:** Prediction serving, dashboards, and AI summaries are intentionally deferred to later phases.

---

## 🧭 Project Phases

### ✅ Phase 1 — Data Foundation (Completed)

- Integrated live and historical NYISO & EIA data
- Built a normalized PostgreSQL schema for time-series storage
- Dockerized ingestion workflows for reproducibility
- Orchestrated ingestion with n8n

📂 **See:** `Phase-1/README.md`

---

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

📂 **See:** `Phase-2/README.md`  
📓 Diagnostics notebooks included for both demand and price.

---

### 🔜 Phase 3 — Insight & Decision Layer (Planned)

Phase 3 will transform forecasts into operational intelligence, including:

- Risk flags and thresholds
- Actionable alerts (e.g., upcoming high-price windows)
- Forecast confidence and regime awareness
- Human-readable summaries for operators

This phase focuses on decision support, not model complexity.

---

### 🔜 Phase 4 — Visualization & Consumption (Planned)

Interactive dashboards for:

- Historical trends
- Forecast windows
- Risk indicators

Operator-facing views designed for action, not exploration.

---

### 🔜 Phase 5 — Deployment & Monitoring (Planned)

- API-based forecast serving
- Containerized deployment
- Logging, monitoring, and alerting
- Scalability considerations

---

## 🛠️ Tech Stack (Actual, Not Aspirational)

| Component | Technology |
|-----------|------------|
| **Language** | Python |
| **Database** | PostgreSQL |
| **ML Models** | Ridge Regression, XGBoost |
| **Time-Series Handling** | Pandas, NumPy |
| **Workflow Orchestration** | n8n |
| **Containerization** | Docker |
| **Diagnostics & Reporting** | Jupyter, Markdown |
| **Cloud (Planned)** | AWS |

> Deep learning frameworks (TensorFlow / PyTorch) are intentionally deferred until justified by data and use case.

---

## 🚦 Why This Project Matters

This project is deliberately built to reflect how real ML systems are developed, not how demo notebooks are written.

**It demonstrates:**

- 🧱 Production-style data engineering
- ⏱️ Correct time-series modeling discipline
- 📉 Honest baseline comparisons
- 🔍 Diagnostic transparency
- 🧠 Business-aware ML decision-making

---

## 📌 Current Status

- ✅ **Phase 1** — Complete
- ✅ **Phase 2** — Complete
- 🔜 **Phase 3** — Planned

---

## 📂 Project Structure

```
Energy-Intelligence/
├── Phase-1/              # Data ingestion & orchestration
│   └── README.md
├── Phase-2/              # Modeling & forecasting (COMPLETE)
│   ├── src/              # Production scripts
│   ├── notebooks/        # Diagnostics & interpretation
│   ├── data/processed/   # Feature & model frames
│   ├── reports/          # Evaluation outputs
│   └── README.md
├── Phase-3/              # Insight & decision layer (planned)
├── Phase-4/              # Visualization & dashboards (planned)
├── Phase-5/              # Deployment & monitoring (planned)
└── README.md             # This file
```

---

## 🔒 Final Note

**Phase 2 is locked, reproducible, and defensible.**

All subsequent phases build on this forecasting foundation — not the other way around.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Docker & Docker Compose
- n8n (for orchestration)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/energy-intelligence.git
cd energy-intelligence

# Set up Phase 1 (data ingestion)
cd Phase-1
docker-compose up -d

# Set up Phase 2 (modeling)
cd ../Phase-2
pip install -r requirements.txt
python src/train_demand_model.py
python src/train_price_model.py
```

For detailed setup instructions, see the README in each phase directory.
