import pandas as pd
import sqlite3
from src.utils.config import DATA_DIR, DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_csv(filename, nrows=None):
    filepath = DATA_DIR / filename
    logger.info(f"Loading data from {filepath}")
    return pd.read_csv(filepath, nrows=nrows)

def init_sqlite_db(df):
    """Creates a local SQLite DB"""
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('loan_applications', conn, if_exists='replace', index=False)
    conn.close()
    logger.info("SQLite database initialized for Talk-to-Data.")