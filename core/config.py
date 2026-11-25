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


def _get_env_or_secret(key: str, default: str = "") -> str:
    """Read config from OS/.env first, falling back to Streamlit Cloud st.secrets."""
    env_val = os.getenv(key)
    if env_val is not None and env_val.strip():
        return env_val.strip()

    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key]).strip()
        # Also support users who group secrets under a TOML section like [secrets] or [api].
        for section_val in st.secrets.values():
            if isinstance(section_val, dict) and key in section_val:
                return str(section_val[key]).strip()
    except Exception:
        # Running outside Streamlit or without a secrets.toml file is expected locally.
        pass

    return default


DB_PATH: Path = Path(_get_env_or_secret("DB_PATH", str(DB_DIR / "sales_agent.db")))

GEMINI_API_KEY: str = _get_env_or_secret("GEMINI_API_KEY")
GROQ_API_KEY: str = _get_env_or_secret("GROQ_API_KEY")

GEMINI_MODEL: str = _get_env_or_secret("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL: str = _get_env_or_secret("EMBEDDING_MODEL", "gemini-embedding-001")
GROQ_WHISPER_MODEL: str = _get_env_or_secret("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
CRM_WEBHOOK_URL: str = _get_env_or_secret("CRM_WEBHOOK_URL")
CRM_MODE: str = _get_env_or_secret("CRM_MODE", "local").lower()
CRM_EXPORT_PATH: Path = Path(_get_env_or_secret("CRM_EXPORT_PATH", str(DATA_DIR / "crm_export.csv")))
TIMEZONE_NAME: str = _get_env_or_secret("TIMEZONE_NAME", "Asia/Kolkata")

COMPANY_NAME: str = _get_env_or_secret("COMPANY_NAME", "NovaCloud AI")
COMPANY_PRODUCTS: str = _get_env_or_secret(
    "COMPANY_PRODUCTS",
    "AI Voice Receptionist, Automated Lead Qualifier, Smart Demo Scheduler",
)
MEMORY_WINDOW_SIZE: int = int(_get_env_or_secret("MEMORY_WINDOW_SIZE", "10") or "10")

LOG_LEVEL: str = _get_env_or_secret("LOG_LEVEL", "INFO").upper()

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

