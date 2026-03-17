"""
Configuration module for the AI Fact-Checker backend.

Loads environment variables from .env file and provides
centralized access to all configuration settings.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def _getenv_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    if val == "":
        return default
    return val


def _warn_if_missing(name: str, value: Optional[str]) -> None:
    if not value:
        logger.warning("Missing environment variable for API key; related sources will be skipped. key=%s", name)


class Settings:
    """
    Application settings loaded from environment variables.

    Attributes:
        NEWS_API_KEY: API key for NewsAPI (https://newsapi.org).
        OPENAI_API_KEY: API key for OpenAI / LLM provider.
        EMBEDDING_MODEL_NAME: Sentence-Transformer model for embeddings.
        LLM_MODEL_NAME: Language model to use for fact-check analysis.
        BACKEND_HOST: Host address for the FastAPI server.
        BACKEND_PORT: Port number for the FastAPI server.
        CORS_ORIGINS: Comma-separated list of allowed CORS origins.
    """

    # --- API Keys ---
    # Note: keep backwards-compatible aliases (NEWS_API_KEY, GOOGLE_FACT_CHECK_API_KEY, etc.)
    # while standardizing on the env var names required by the production audit.
    OPENROUTER_API_KEY: Optional[str] = _getenv_str("OPENROUTER_API_KEY")
    NEWSAPI_KEY: Optional[str] = _getenv_str("NEWSAPI_KEY") or _getenv_str("NEWS_API_KEY")
    GNEWS_API_KEY: Optional[str] = _getenv_str("GNEWS_API_KEY")
    NEWSDATA_API_KEY: Optional[str] = _getenv_str("NEWSDATA_API_KEY")
    TAVILY_API_KEY: Optional[str] = _getenv_str("TAVILY_API_KEY")
    GOOGLE_FACTCHECK_API_KEY: Optional[str] = _getenv_str("GOOGLE_FACTCHECK_API_KEY") or _getenv_str("GOOGLE_FACT_CHECK_API_KEY")
    GOOGLE_KG_API_KEY: Optional[str] = _getenv_str("GOOGLE_KG_API_KEY") or _getenv_str("GOOGLE_KNOWLEDGE_GRAPH_API_KEY")

    # Other / legacy keys used elsewhere in the codebase
    OPENAI_API_KEY: Optional[str] = _getenv_str("OPENAI_API_KEY")
    OPENAI_BASE_URL: str = _getenv_str("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    OPTIIC_API_KEY: Optional[str] = _getenv_str("OPTIIC_API_KEY")
    NEWS_API_KEY: Optional[str] = NEWSAPI_KEY  # backward compatible alias
    GOOGLE_FACT_CHECK_API_KEY: Optional[str] = GOOGLE_FACTCHECK_API_KEY  # backward compatible alias
    GOOGLE_KNOWLEDGE_GRAPH_API_KEY: Optional[str] = GOOGLE_KG_API_KEY  # backward compatible alias
    
    # --- Additional Google keys (not part of audit list, but may be referenced) ---
    GOOGLE_CLOUD_VISION_API_KEY: Optional[str] = _getenv_str("GOOGLE_CLOUD_VISION_API_KEY")
    GOOGLE_CUSTOM_SEARCH_API_KEY: Optional[str] = _getenv_str("GOOGLE_CUSTOM_SEARCH_API_KEY")

    # --- Model Configuration ---
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    # Multi-agent / pipeline model configuration (env override, OpenRouter-friendly defaults)
    MODEL_CLAIM_EXTRACTOR: str = _getenv_str("MODEL_CLAIM_EXTRACTOR", "meta-llama/llama-3.2-3b-instruct") or "meta-llama/llama-3.2-3b-instruct"
    MODEL_QUERY_EXPANDER: str = _getenv_str("MODEL_QUERY_EXPANDER", "meta-llama/llama-3.2-3b-instruct") or "meta-llama/llama-3.2-3b-instruct"
    MODEL_EVIDENCE_ANALYST: str = _getenv_str("MODEL_EVIDENCE_ANALYST", "qwen/qwen3-next-80b-a3b-instruct") or "qwen/qwen3-next-80b-a3b-instruct"
    MODEL_FACTCHECK_JUDGE: str = _getenv_str("MODEL_FACTCHECK_JUDGE", "openai/gpt-oss-120b") or "openai/gpt-oss-120b"
    MODEL_VALIDATION_AGENT: str = _getenv_str("MODEL_VALIDATION_AGENT", "mistral/mistral-small-3.1-24b") or "mistral/mistral-small-3.1-24b"
    MODEL_NEWS_SUMMARIZER: str = _getenv_str("MODEL_NEWS_SUMMARIZER", "meta-llama/llama-3.2-3b-instruct") or "meta-llama/llama-3.2-3b-instruct"

    OCR_MODEL_NAME: str = os.getenv("OCR_MODEL_NAME", "openrouter/hunter-alpha")

    # --- Server Configuration ---
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    CORS_ORIGINS: list[str] = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")

    # --- Vector DB Configuration ---
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./vector_db")
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "384"))  # MiniLM-L6 dim


# Singleton settings instance
settings = Settings()

# Log missing keys once at startup; downstream services should still check per-call.
_warn_if_missing("OPENROUTER_API_KEY", settings.OPENROUTER_API_KEY)
_warn_if_missing("NEWSAPI_KEY", settings.NEWSAPI_KEY)
_warn_if_missing("GNEWS_API_KEY", settings.GNEWS_API_KEY)
_warn_if_missing("NEWSDATA_API_KEY", settings.NEWSDATA_API_KEY)
_warn_if_missing("TAVILY_API_KEY", settings.TAVILY_API_KEY)
_warn_if_missing("GOOGLE_FACTCHECK_API_KEY", settings.GOOGLE_FACTCHECK_API_KEY)
_warn_if_missing("GOOGLE_KG_API_KEY", settings.GOOGLE_KG_API_KEY)
