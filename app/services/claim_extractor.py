"""
Claim Extractor Service.

Uses an LLM to extract the primary factual claim from a messy block of text (like OCR output).
Ignores opinions, greetings, and irrelevant conversational text.
"""

import logging
from app.config import settings
from app.services.llm_analyzer import get_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise text-extraction AI.
Your job is to read raw text (which may contain errors, opinions, or conversational filler) and extract ONLY the primary, testable factual claim.

Rules:
1. Ignore opinions, greetings, questions, or irrelevant noise.
2. Return ONLY the extracted claim as a single string.
3. If multiple claims exist, extract the main one.
4. If no factual claim exists, return the exact string: "NO_CLAIM_FOUND".
5. Do NOT include quotes, explanations, or conversational text in your response.
"""

def extract_primary_claim(text: str) -> str:
    """
    Extract the primary factual claim from a block of text using an LLM.

    Args:
        text: The raw text (e.g., from OCR).

    Returns:
        The extracted claim as a string, or "NO_CLAIM_FOUND".
    """
    logger.info(f"🧠 Extracting primary claim from text ({len(text)} chars)")

    if not text or not text.strip():
        return "NO_CLAIM_FOUND"

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your_"):
        logger.error("OPENAI_API_KEY is not configured.")
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text:\n{text}"},
            ],
            temperature=0.1,  # Low temperature for precise extraction
            max_tokens=200,
        )

        extracted = response.choices[0].message.content.strip()
        logger.info(f"✅ Extracted claim: '{extracted}'")
        return extracted

    except Exception as e:
        logger.error(f"❌ Claim extraction failed: {e}")
        # Fallback to returning the raw text if LLM fails,
        # so the pipeline doesn't completely break, but log the error.
        return text.strip()
