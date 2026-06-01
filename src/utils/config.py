import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
SQL_DIR = BASE_DIR / "sql"
DB_PATH = DATA_DIR / "credit_data.db"

# Model constants
TARGET_COL = "TARGET"
MODEL_PATH = MODEL_DIR / "rf_pipeline.joblib"