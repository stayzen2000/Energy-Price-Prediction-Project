# ⚙️ Phase 3 — Insight & Decision Layer

## Overview

Phase 3 transforms raw machine-learning forecasts into actionable, explainable, business-oriented intelligence.

While Phase 2 focused on modeling (training, validation, and evaluation), Phase 3 focuses on operationalization—specifically:

- Running inference using frozen, trained models
- Translating forecasts into interpretable insights
- Generating conservative, deterministic recommendations
- Packaging everything into a single, structured output contract for downstream consumption

This phase intentionally avoids dashboards, cloud deployment, and retraining. Its goal is to demonstrate how real-world ML systems bridge the gap between models and decisions.

## What Phase 3 Is (and Is Not)

### ✅ Phase 3 Is

- A read-only inference layer
- A rules-based insight engine
- A decision-support system
- A clean contract producer for UI and deployment layers

### ❌ Phase 3 Is Not

- A modeling or retraining phase
- A real-time ingestion system
- A dashboard or visualization layer
- A cloud / AWS deployment
- An autonomous AI agent

Those concerns are intentionally deferred to later phases.

## Architecture Overview

```
Phase 2 (LOCKED)
│
├── Trained Demand Model (XGBoost, t+24h)
├── Trained Price Model (XGBoost, t+1h)
├── Frozen Feature Frames (immutable snapshot)
│
▼
Phase 3 (THIS PHASE)
│
├── Inference (read-only)
├── Deterministic Insight Derivation
├── Rule-Based Recommendations
├── Structured Output Bundle (JSON contract)
│
▼
Phase 4 (NEXT)
│
├── Visual Dashboard (UI over contract)
├── Conversational UI (LLM over contract)
```

**Phase 3 never modifies Phase 2 artifacts. It treats them as immutable inputs.**

## Inputs (From Phase 2)

Phase 3 consumes the following frozen artifacts from Phase 2:

### 1. Feature Frames (Immutable Snapshots)

**Demand features:**
- `features_lagroll_24h.parquet` (ends at 2025-12-16 23:00 UTC)

**Price features:**
- `model_frame_price_1h.parquet` (ends at 2025-12-14 22:00 UTC)

These frames include:
- Lagged demand/price values
- Rolling statistics
- Calendar encodings
- Weather and contextual features
- Leakage-safe engineered features

### 2. Trained Models

- Demand forecasting model (24-hour horizon)
- Price forecasting model (1-hour horizon)

Both models are:
- Fully trained
- Evaluated and documented in Phase 2
- Loaded in Phase 3 without retraining

## Step 1 — Forecast Inference (Read-Only)

Phase 3 runs inference using the trained models and the frozen feature frames.

### Forecasts Produced

- **Demand:** 24 hourly forecasts (t+24 relative to feature timestamps)
- **Price:** 1 hourly forecast (t+1 relative to feature timestamps)

### Important Detail: "As-Of" Timestamps

Because Phase 3 uses frozen feature frames:

- Demand forecasts are generated relative to 2025-12-16
- Price forecasts are generated relative to 2025-12-14

This is expected and intentional.

Phase 3 explicitly logs:
- Feature frame "as_of" timestamps
- Forecast windows
- Timestamp misalignment warnings

This prevents confusion and reinforces auditability.

## Step 2 — Insight Derivation (No ML)

Forecasts alone are not actionable. Phase 3 derives insights using deterministic, explainable rules.

### Demand Insights

- Identifies peak demand windows using historical percentiles (p90 / p95)
- Highlights top forecasted demand hours
- Flags only statistically meaningful peaks (avoids alert fatigue)

### Price Insights

Classifies next-hour price into regimes:
- `normal`
- `high` (above p95)
- `spike` (above p99)

Adds a volatility flag based on rolling standard deviation:
- Separates price level risk from price movement risk

**No machine learning is used at this stage—only transparent rules.**

## Step 3 — Recommendations (Decision Support)

Insights are translated into conservative, business-friendly recommendations.

### Design Principles

- Every recommendation maps directly to a signal
- No claims of guaranteed savings
- No autonomous actions
- Clear "why" and "evidence" for each recommendation

### Example Outputs

- "No unusual demand peaks detected; standard operations are likely sufficient."
- "Next-hour price is in spike regime (above p99); consider deferring discretionary energy use if feasible."

This mirrors how real decision-support systems operate in production.

## Output Contract (Phase 3 Bundle)

Phase 3 produces a single structured JSON artifact:

```
Phase-3/outputs/phase3_bundle_YYYYMMDDTHHMMSSZ.json
```

### Bundle Contents

- Metadata (generation time, as_of timestamps)
- Forecasts (demand + price)
- Insight summaries
- Recommendations
- Thresholds used
- Explicit limitations and disclaimers

**This bundle is the sole input for Phase 4 and Phase 5.**

## Known Limitations (Intentional)

### Frozen Snapshot (No Live Ingestion)

Phase 3 operates on a frozen Phase-2 dataset snapshot.

As a result:
- Forecasts are relative to the latest available feature timestamps
- Live "current-time" forecasts are not shown
- Demand and price forecasts may reference different dates

**This is intentional and documented.**

### Why This Design Was Chosen

- Preserves reproducibility
- Avoids hidden data leakage
- Demonstrates correct separation between:
  - Training
  - Inference
  - Serving

In production (Phase 5), the same models would consume continuously updated features from a live ingestion pipeline. That serving layer is intentionally out of scope for Phase 3.

## Why Phase 3 Matters

Phase 3 demonstrates:

- How trained ML models are actually used
- How forecasts become decisions
- How to design trustworthy, explainable systems
- How to create clean contracts between system layers

**This phase reflects real-world ML engineering, not just modeling.**

## What Comes Next — Phase 4

Phase 4 will build user interfaces on top of the Phase 3 contract.

### Phase 4 Goals

- Visual dashboard (charts, cards, recommendations)
- Conversational UI (LLM-powered Q&A)
- Both UIs consume the same immutable Phase-3 bundle

### Phase 4 Will Include

- Demand forecast visualization
- Price regime indicators
- Recommendation panels
- Chat-based explanations grounded in bundle data

### Phase 4 Will NOT Include

- Retraining
- Live ingestion
- Cloud deployment
- Autonomous AI behavior

## Phase 3 Status

- ✅ Complete
- ✅ Forecasts generated
- ✅ Insights derived
- ✅ Recommendations produced
- ✅ Output contract finalized
- ➡️ Ready for Phase 4