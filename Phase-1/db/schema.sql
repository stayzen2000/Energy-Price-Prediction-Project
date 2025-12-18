CREATE TABLE IF NOT EXISTS grid_load (
  timestamp TIMESTAMPTZ NOT NULL,
  region TEXT,
  zone_id TEXT NOT NULL,
  demand_mw DOUBLE PRECISION,
  demand_forecast_mw DOUBLE PRECISION,
  price_per_mwh DOUBLE PRECISION,
  source TEXT,
  PRIMARY KEY (timestamp, zone_id)
);

CREATE TABLE IF NOT EXISTS weather (
  timestamp TIMESTAMPTZ NOT NULL,
  zone_id TEXT NOT NULL,
  temp_c DOUBLE PRECISION,
  humidity DOUBLE PRECISION,
  wind_speed DOUBLE PRECISION,
  precipitation DOUBLE PRECISION,
  PRIMARY KEY (timestamp, zone_id)
);

-- Optional: logging table (nice to have)
CREATE TABLE IF NOT EXISTS ingestion_runs (
  id BIGSERIAL PRIMARY KEY,
  run_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  workflow_name TEXT NOT NULL,
  step_name TEXT NOT NULL,
  status TEXT NOT NULL,
  message TEXT,
  error_text TEXT
);
