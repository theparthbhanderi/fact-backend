import logging
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# Primary OpenRouter Client
_client = None

def get_openrouter_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.OPENROUTER_API_KEY or settings.OPENROUTER_API_KEY.startswith("your_"):
            # Don't hard-crash the API process; upstream pipeline should handle this gracefully.
            logger.warning("Missing API key; OpenRouter calls will be skipped. key=%s", "OPENROUTER_API_KEY")
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        
        _client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client
