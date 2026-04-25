import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent

load_dotenv()


class Setting(BaseModel):
    POSTGRES_USER: str = os.getenv('POSTGRES_USER', '')
    POSTGRES_PASSWORD: str = os.getenv('POSTGRES_PASSWORD', '')
    POSTGRES_DB: str = os.getenv('POSTGRES_DB', '')
    DATABASE_URL: str = os.getenv('DATABASE_URL', '')
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')
    API_KEY_gemini: str = os.getenv('API_KEY_gemini', '')
    LOG_PATH: str | Path = ROOT_DIR / 'knn.log'


setting = Setting()
