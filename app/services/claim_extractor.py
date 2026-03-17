"""
Claim Extractor Agent.

Uses Llama-3.2-3B-Instruct (via OpenRouter) to extract distinct, testable factual claims 
from a noisy user input, outputting them as a list of strings.
"""

import json
import logging
from app.config import settings
from app.services.openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise text-extraction AI.
Your ONLY job is to read raw text (which may contain errors, opinions, or conversational filler) 
and extract the primary, verifiable factual claims.

Rules:
1. Ignore opinions, greetings, questions, or irrelevant noise.
2. If the user presents multiple distinct facts, split them into separate claims.
3. If no factual claim exists, return an empty array [].
4. Do NOT include explanations, introductions, or conversational text.
5. You MUST respond with a valid JSON array of strings, and nothing else.

Example output format:
["The Eiffel Tower is 1000 feet tall.", "France was founded in 1792."]
"""

def extract_primary_claims(text: str) -> list[str]:
    """
    Extract factual claims from raw text as a list of strings using LLM.
    """
    logger.info(f"🧠 Extracting claims from text ({len(text)} chars)")

    if not text or not text.strip():
        return []

    try:
        client = get_openrouter_client()
        response = client.chat.completions.create(
            model=settings.MODEL_CLAIM_EXTRACTOR,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text:\n{text}"},
            ],
            temperature=0.1,  # Precise extraction
            max_tokens=300,
        )

        content = response.choices[0].message.content.strip()
        
        # Parse JSON array response
        try:
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            claims = json.loads(content)
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                logger.info(f"✅ Extracted {len(claims)} claims.")
                return claims
            else:
                raise ValueError("LLM did not return a list of strings")
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON array from Claim Extractor LLM. Returning raw string as a single claim.")
            return [content]

    except Exception as e:
        logger.error(f"❌ Claim extraction failed: {e}")
        # Fallback to returning the raw text as a single claim
        return [text.strip()]
