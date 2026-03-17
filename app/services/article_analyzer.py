"""
Article Analyzer Service.

Uses an LLM to read a long news article and extract the main testable/factual claims.
Limits the output to a specified number of claims (e.g., 5) to keep processing times reasonable.
"""

import json
import logging
from app.config import settings
from app.services.llm_analyzer import get_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise fact-extraction AI.
Your job is to read a news article and extract the main, testable factual claims.

Rules:
1. Ignore opinions, commentary, or general background information.
2. Focus on the core claims being made (e.g., statistical claims, health claims, actions taken by public figures).
3. Do NOT invent or deduce claims; only extract what is explicitly stated.
4. Return a maximum of {max_claims} key claims.
5. You MUST respond with valid JSON in this exact format:
{
  "claims": [
    "Claim 1",
    "Claim 2"
  ]
}
If no factual claims are found, return:
{
  "claims": []
}"""

def extract_article_claims(text: str, max_claims: int = 5) -> list[str]:
    """
    Extract up to `max_claims` factual claims from article text using an LLM.

    Args:
        text: The full text of the article.
        max_claims: The maximum number of claims to extract.

    Returns:
        A list of string claims.
    """
    logger.info(f"🧠 Extracting up to {max_claims} claims from article ({len(text)} chars)")

    if not text or len(text.strip()) < 50:
        logger.warning("Article text is too short for claim extraction.")
        return []

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your_"):
        logger.error("OPENAI_API_KEY is not configured.")
        raise RuntimeError("OPENAI_API_KEY is not set.")

    # Format the prompt with the dynamic max_claims value
    prompt = SYSTEM_PROMPT.replace("{max_claims}", str(max_claims))
    
    # Truncate text if it's absurdly long to save tokens (first ~8000 chars is usually enough for key claims)
    truncated_text = text[:8000]

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Article Text:\n{truncated_text}"},
            ],
            temperature=0.1,  # Low temperature for precise extraction
            max_tokens=500,
            response_format={ "type": "json_object" } # Force JSON Object output
        )

        content = response.choices[0].message.content.strip()
        
        try:
            result = json.loads(content)
            claims = result.get("claims", [])
            
            # Ensure it's actually a list of strings
            valid_claims = [str(c) for c in claims if c and isinstance(c, str)]
            
            logger.info(f"✅ Extracted {len(valid_claims)} claims.")
            return valid_claims[:max_claims] # Enforce max claims limit strictly
            
        except json.JSONDecodeError:
            logger.error("❌ Failed to parse LLM JSON response for article claims.")
            return []

    except Exception as e:
        logger.error(f"❌ Claim extraction failed: {e}")
        raise RuntimeError(f"Claim extraction failed: {e}")
