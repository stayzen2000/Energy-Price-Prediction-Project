# config.py

# --- Core project scope ---

# Main system / ISO
REGION_ID = "NYISO"

# Forecast horizon (like a weather app: next 24 hours)
FORECAST_HORIZON_HOURS = 24

# Time resolution in minutes (hourly data)
TIME_RESOLUTION_MINUTES = 60

# Targets we will forecast (per zone)
TARGET_COLUMNS = [
    "demand_mw",        # load (MW)
    "price_per_mwh"     # electricity price
]

# Later extension: list of zones inside NYISO
NYISO_ZONES = [
    # placeholder; you will fill these IDs once you pick them from API docs
    # e.g. "AECO", "PENELEC", "PPL", ...
]

# --- EIA API ---
EIA_API_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data"

# PJM settings
EIA_RESPONDENT = "NYIS"

# We want both actual demand and EIA day-ahead forecast
EIA_DEMAND_TYPES = ["D", "DF"]
