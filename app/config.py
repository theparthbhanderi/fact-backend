"""
Configuration module for the AI Fact-Checker backend.

Loads environment variables from .env file and provides
centralized access to all configuration settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # --- Model Configuration ---
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

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
