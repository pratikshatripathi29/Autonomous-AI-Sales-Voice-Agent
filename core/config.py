"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve paths relative to project root so imports work from any working directory.
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
KNOWLEDGE_DIR: Path = DATA_DIR / "knowledge"
DB_DIR: Path = DATA_DIR / "db"
FAISS_INDEX_DIR: Path = Path(os.getenv("FAISS_INDEX_DIR", str(DATA_DIR / "faiss_index")))

# Load .env explicitly from project root so secrets never rely on shell CWD.
load_dotenv(dotenv_path=BASE_DIR / ".env")

DB_PATH: Path = Path(os.getenv("DB_PATH", str(DB_DIR / "sales_agent.db")))

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip()
GROQ_WHISPER_MODEL: str = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo").strip()
CRM_WEBHOOK_URL: str = os.getenv("CRM_WEBHOOK_URL", "").strip()
CRM_MODE: str = os.getenv("CRM_MODE", "local").strip().lower()
CRM_EXPORT_PATH: Path = Path(os.getenv("CRM_EXPORT_PATH", str(DATA_DIR / "crm_export.csv")))
TIMEZONE_NAME: str = os.getenv("TIMEZONE_NAME", "Asia/Kolkata").strip()

COMPANY_NAME: str = os.getenv("COMPANY_NAME", "NovaCloud AI").strip()
COMPANY_PRODUCTS: str = os.getenv(
    "COMPANY_PRODUCTS",
    "AI Voice Receptionist, Automated Lead Qualifier, Smart Demo Scheduler",
).strip()
MEMORY_WINDOW_SIZE: int = int(os.getenv("MEMORY_WINDOW_SIZE") or "10")

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper().strip()

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "pa": "Punjabi",
    "es": "Spanish",
    "fr": "French",
}


def configure_logging() -> None:
    """Configure root logging format and level once for the application."""
    root_logger = logging.getLogger()
    # Guard against duplicate handlers when Streamlit re-runs the script on interaction.
    if not root_logger.handlers:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


def get_config_status() -> dict[str, bool]:
    """Return boolean readiness flags for required API keys without exposing secrets."""
    return {
        "gemini_configured": bool(GEMINI_API_KEY),
        "groq_configured": bool(GROQ_API_KEY),
        "crm_webhook_configured": bool(CRM_WEBHOOK_URL),
    }

