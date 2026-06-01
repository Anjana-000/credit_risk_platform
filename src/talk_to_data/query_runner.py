import sqlite3
import pandas as pd
from src.utils.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

def execute_sql(query):
    """Executes the generated SQL safely against the local DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.to_dict(orient='records')
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return str(e)