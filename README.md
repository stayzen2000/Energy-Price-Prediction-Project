# ⚡ Real-Time Energy Demand & Price Forecasting System

## 🔍 Overview

This project is an end-to-end, production-style energy forecasting and decision-support system that uses live grid data to predict next-day energy demand and electricity pricing. The system combines real-time data ingestion, time-series machine learning models, and an AI-powered insights layer to help users make informed, cost-saving energy decisions.

The goal of this project is to demonstrate how industry-grade data pipelines and forecasting systems are built, deployed, and consumed in real-world environments.

---

## 🎯 Purpose of This Project

Energy consumption and electricity pricing are increasingly volatile due to weather patterns, grid congestion, and growing demand from power-intensive operations. Most businesses and individuals lack accessible tools that allow them to anticipate high-demand or high-cost periods before they happen.

This project focuses on:

- 🔮 **Forecasting short-term energy demand and pricing** using real grid data
- 🧠 **Enabling proactive decision-making** instead of reactive cost management
- 🏗️ **Showcasing production-ready data engineering and ML practices**

---

## 👥 Who This Project Helps

### 💨 HVAC Companies
- Optimize heating and cooling schedules
- Reduce peak-load charges
- Minimize equipment strain

### 🏢 Building & Facility Managers
- Plan daily energy usage more effectively
- Identify high-risk demand periods
- Improve operational efficiency

### 🏭 Factories & Industrial Sites
- Shift production to lower-cost energy windows
- Prevent overloads and downtime
- Control operational energy spend

### 🖥️ Data Centers
- Manage power-intensive server workloads
- Anticipate peak pricing windows
- Improve reliability and cost efficiency

### 🏠 Everyday Consumers
- Understand when electricity will be most expensive
- Adjust energy usage to reduce monthly bills
- Make informed, data-driven decisions

---

## 💡 How This System Helps Businesses

The system converts raw grid and weather data into forward-looking intelligence. Instead of reacting to energy costs after the fact, users can:

- 📊 Forecast the next 24 hours of energy demand and pricing
- ⚠️ Identify upcoming peak-load or peak-price periods
- 🔄 Adjust operations proactively
- 📝 Receive AI-generated summaries rather than manually interpreting charts

This leads to **lower costs, better planning, and reduced operational risk**.

---

## ✨ Key Features

- Live ingestion of NYISO and EIA grid data
- Time-series forecasting using TensorFlow (LSTM / GRU / TCN)
- Industry-standard relational storage with PostgreSQL
- Containerized ingestion and services using Docker
- Workflow orchestration with n8n
- REST API for real-time predictions
- Interactive dashboard for historical and forecasted views
- Optional AI-generated insights and alerts

---

## 🧭 Project Phases

### ✅ Phase 1: Data Foundation (Completed)

- Integrated live and historical energy demand and pricing data from the NYISO and EIA APIs
- Built a normalized PostgreSQL schema for time-series storage
- Dockerized the ingestion environment for reproducibility and reliability
- Orchestrated and scheduled ingestion workflows using n8n

### 🔮 Phase 2: Modeling

- Engineer time-series features (lags, rolling windows, weather joins)
- Train and evaluate forecasting models (LSTM / GRU / TCN)
- Track experiments using MLflow or Weights & Biases

### 🚀 Phase 3: Prediction Serving

- Serve forecasts through a FastAPI-based prediction service
- Enable scheduled and near-real-time updates
- Containerize services for deployment

### 📊 Phase 4: Dashboard & AI Insights

Build an interactive dashboard showing:
- Past 24-hour usage
- Next 24-hour demand and price forecasts

Add AI-generated summaries and recommendations:
- Highlight anomalies and peak-risk windows

### ☁️ Phase 5: Deployment & Monitoring

- Deploy services to AWS
- Enable alerting, logging, and monitoring
- Prepare the system for real-world scalability

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python |
| **Database** | PostgreSQL |
| **ML Frameworks** | TensorFlow (primary), PyTorch (optional) |
| **Workflow Orchestration** | n8n |
| **APIs** | FastAPI |
| **Containerization** | Docker |
| **LLM Integration** | OpenAI API (optional), LangChain (optional) |
| **Visualization** | Streamlit or React |
| **Cloud** | AWS |

---

## 🚦 Why This Project Matters

This project is intentionally designed to move beyond toy datasets and isolated notebooks. It reflects how real-world energy analytics and forecasting systems are built—combining data engineering, machine learning, orchestration, and decision support.

It demonstrates:

- 🧱 **Production-style data pipelines**
- 📈 **Applied time-series forecasting**
- 🔌 **API-based system design**
- 🧠 **Business-focused ML applications**

---

## 📌 Status

- ✅ **Phase 1 completed**
- 🚧 **Phase 2 in progress**

---

## 📂 Project Structure

```
Energy-Intelligence/
├── Phase-1/              # Data ingestion and orchestration
│   ├── scripts/          # Ingestion scripts
│   ├── db/               # Database initialization
│   ├── docker-compose.yml
│   ├── Dockerfile.runner
│   ├── requirements.txt
│   ├── .env
│   ├── runner_api.py
│   ├── NYISO n8n Scheduled Ingestor.json # n8n scheduled hourly trigger
│   └── README.md
├── Phase-2/              # Feature engineering and modeling (upcoming)
├── Phase-3/              # Prediction API service (planned)
├── Phase-4/              # Dashboard and insights (planned)
├── Phase-5/              # Deployment and monitoring (planned)
└── README.md             # This file
```

