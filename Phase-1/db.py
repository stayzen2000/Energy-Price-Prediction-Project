import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Always load .env from the project root (same folder as db.py)
ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError(
        f"Missing DB env vars. Loaded .env from: {ENV_PATH}\n"
        f"DB_USER={DB_USER}, DB_PASSWORD={'SET' if DB_PASSWORD else None}, DB_NAME={DB_NAME}"
    )

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL, future=True)
