import os
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

# GEMINI API KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# MySQL DB 설정
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}