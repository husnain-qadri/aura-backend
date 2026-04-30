import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL = 'openai/gpt-oss-120b'
FLASK_PORT = int(os.getenv('PORT', os.getenv('FLASK_PORT', '8080')))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
MAX_UPLOAD_SIZE_MB = 50
MAX_PDF_DOWNLOAD_SIZE_MB = 50
RESOLVE_WORKERS = 8
S2_MAX_RPS = 5
SPECTER_MODEL_NAME = 'allenai/specter2_base'
