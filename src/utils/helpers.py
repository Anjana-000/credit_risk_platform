import os
from dotenv import load_dotenv

def load_env_vars():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in .env file.")
    return api_key