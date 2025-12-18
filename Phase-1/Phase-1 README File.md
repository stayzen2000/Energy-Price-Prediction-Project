# 📊 Energy Intelligence — Phase 1

**Hourly Data Ingestion & Orchestration (NYISO)**

## Overview

Phase 1 of the Energy Intelligence project establishes a production-style data foundation for ingesting, storing, and orchestrating live energy data on an hourly basis.

The primary goal of this phase was to build a **reliable, reproducible, and idempotent** ingestion pipeline that pulls data from multiple external APIs, handles real-world data timing inconsistencies, and stores everything safely in PostgreSQL — all orchestrated via n8n.

**This phase intentionally focuses only on data ingestion and orchestration.**  
No forecasting models, dashboards, or AI agents are included yet.

---

## 🎯 Phase 1 Objectives

- Ingest live + recent historical energy-related data hourly
- Normalize and store data in PostgreSQL
- Ensure idempotent upserts (no duplicates)
- Handle timezone correctness
- Orchestrate ingestion using n8n
- Log ingestion runs for observability
- Fully Dockerized for reproducibility

---

## 📦 Data Sources Used

Due to authentication and complexity constraints with PJM (see Challenges), this phase uses NYISO data sources instead.

### 1️⃣ Grid Load & Demand (EIA API)

- Hourly system-wide demand
- Demand forecast where available
- **Source:** U.S. Energy Information Administration (EIA)

### 2️⃣ Electricity Pricing (NYISO)

- Hourly day-ahead market prices
- **Source:** NYISO via `gridstatus`

### 3️⃣ Weather Data (Open-Meteo API)

- Hourly weather features:
  - Temperature
  - Humidity
  - Wind speed
  - Precipitation
- Used as exogenous variables for future modeling

---

## 🗄️ Database Design

PostgreSQL is used as the primary datastore.

### Core Tables

- `weather`
- `grid_load`
- `ingestion_runs` (execution logging)

### Key Design Decisions

- All timestamps stored as `TIMESTAMPTZ` (UTC)
- Composite primary keys: `(timestamp, zone_id)`
- Upserts (`ON CONFLICT DO UPDATE`) used everywhere to:
  - Prevent duplicates
  - Allow partial data arrival
  - Handle real-world API lag

### Important Note on NULL Values

It is **expected behavior** that:

- Recent hours may contain price data but missing demand
- Or demand data but missing price

This happens because different APIs update on different schedules.  
The pipeline is intentionally designed **not to overwrite existing non-null data**.

---

## 🐳 Dockerized Architecture

Everything in Phase 1 runs locally via Docker.

### Containers

- **PostgreSQL** – persistent storage
- **Python Runner** – ingestion scripts
- **Runner API (FastAPI)** – execution interface for n8n
- **n8n** – orchestration engine

### Why Docker?

- Ensures reproducibility
- Eliminates environment drift
- Mirrors real production workflows
- Allows clean "from-scratch" rebuilds

Schema initialization is automatic on a fresh database volume.

---

## 🔁 Orchestration with n8n

Hourly ingestion is orchestrated using n8n (local instance).

### Workflow Sequence

```
Schedule Trigger (Hourly)
   ↓
Weather Ingestion
   ↓
Grid Load Ingestion
   ↓
Price Ingestion
   ↓
Execution Logging
```

### Why an HTTP Runner API?

n8n does not reliably support shell execution across environments.  
Instead, a lightweight FastAPI runner service exposes endpoints like:

- `/run/weather`
- `/run/grid`
- `/run/price`

This approach:

- Avoids shell access issues
- Works identically locally and in future cloud deployments
- Is production-aligned

---

## 📋 Execution Logging

Every ingestion step writes to the `ingestion_runs` table:

| Column        | Purpose                       |
|---------------|-------------------------------|
| `run_ts`      | Timestamp of execution        |
| `workflow_name` | Workflow identifier         |
| `step_name`   | weather / grid / price        |
| `status`      | success / (failure in future phases) |
| `message`     | Raw execution output          |

This provides:

- Auditability
- Debug visibility
- Proof of orchestration correctness

---

## 🧠 Challenges & How They Were Solved

### 1️⃣ PJM Authentication Complexity

**Challenge:**  
PJM requires more advanced authentication and authorization flows than expected.

**Decision:**  
To keep Phase 1 focused and unblocked, PJM was deferred and NYISO was used instead.

**Outcome:**  
Scripts were refactored to:
- Use NYISO-compatible APIs
- Maintain the same architectural structure
- Preserve future extensibility

### 2️⃣ Partial Data & NULL Confusion

**Challenge:**  
Seeing NULL values when joining grid and price data initially appeared to be a bug.

**Root Cause:**  
Different APIs update at different times.

**Solution:**
- Preserve partial rows
- Use idempotent upserts
- Never overwrite existing non-null values

This mirrors real-world energy data pipelines.

### 3️⃣ n8n Execution Limitations

**Challenge:**  
Direct command execution nodes were unavailable or unreliable.

**Solution:**
- Introduced a FastAPI-based runner
- Triggered ingestion via HTTP requests
- Simplified orchestration and future cloud migration

### 4️⃣ PostgreSQL Connectivity Confusion

**Challenge:**  
Initial uncertainty about whether n8n, Docker, and PostgreSQL were pointing to the same database.

**Resolution:**
- Verified Docker network connectivity
- Confirmed database identity via SQL introspection
- Validated writes directly from psql

---

## ✅ Phase 1 Completion Criteria (Met)

- ✔ Hourly ingestion runs successfully
- ✔ No duplicate rows
- ✔ Correct timezone handling
- ✔ Execution logs captured
- ✔ Fully Dockerized
- ✔ Reproducible from a clean slate

---

## 🚧 What's Next (Phase 2 Preview)

Phase 2 will focus on:

- Feature engineering
- Aggregations (hourly → daily)
- Forecast-ready tables
- Anomaly detection & modeling

Cloud deployment and production hosting are intentionally deferred to a later phase.

---

## 📌 How to Run Locally

```bash
docker compose up -d --build
```

Verify ingestion:

```sql
SELECT MAX(timestamp) FROM weather;
SELECT MAX(timestamp) FROM grid_load;
SELECT * FROM ingestion_runs ORDER BY run_ts DESC;
```

---

## 📂 Project Structure

```
Energy-Intelligence/
└── Phase-1/
    ├── scripts/           # Ingestion scripts
    ├── db/                # Database initialization
    ├── docker-compose.yml # Container orchestration
    ├── Dockerfile.runner  # Python runner container
    ├── requirements.txt   # Python dependencies
    └── README.md          # This file
```