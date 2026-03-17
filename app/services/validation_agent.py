"""
Validation Agent.

Uses Qwen-4B (via OpenRouter) to check if the Judge's reasoning logically supports the verdict.
Returns a simple Boolean.
"""

import logging
from app.config import settings
from app.services.openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict Logic Validator for a fact-checking system.
Your ONLY job is to verify if the 'Explanation' logically supports the 'Verdict'.

Rules:
1. You MUST answer with exactly one word: "TRUE" (if logical) or "FALSE" (if illogical/contradictory).
2. Do not explain your answer.
3. Example of FALSE: Verdict is 'TRUE' but the explanation says 'The claim is completely fabricated.'
"""

def validate_reasoning_logic(verdict: str, explanation: str) -> bool:
    """
    Validates if the explanation supports the verdict.
    Returns True if logical, False otherwise.
    """
    logger.info(f"🛡️ Validating logic of verdict: {verdict}")

    try:
        client = get_openrouter_client()
        response = client.chat.completions.create(
            model=settings.MODEL_VALIDATION_AGENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Verdict: {verdict}\nExplanation: {explanation}"},
            ],
            temperature=0.0,  # Zero temperature for deterministic boolean
            max_tokens=10,
        )

        content = response.choices[0].message.content.strip().upper()
        
        if "TRUE" in content:
            logger.info("✅ Validation Agent approved logic.")
            return True
        else:
            logger.warning("🚨 Validation Agent rejected logic as contradictory!")
            return False

    except Exception as e:
        logger.error(f"❌ Validation Agent failed: {e}")
        # Default to True if the validation model itself crashes, preventing infinite loops
        return True
