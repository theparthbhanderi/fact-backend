import logging
from textblob import TextBlob
from app.config import settings
from app.services.llm_analyzer import get_client

logger = logging.getLogger(__name__)

def correct_claim_text(claim: str) -> str:
    """
    Detect spelling mistakes and grammatical errors in the claim and correct them automatically.
    
    Steps:
    1 Detect spelling errors
    2 Fix grammar mistakes
    3 Normalize capitalization
    4 Remove unnecessary punctuation
    
    Returns the clean corrected claim.
    """
    if not claim or not claim.strip():
        return claim
    
    claim = claim.strip()
    
    # 1. Use LLM for context-aware correction
    try:
        client = get_client()
        
        system_prompt = (
            "You are a language correction system.\n"
            "Your job is to correct spelling and grammar errors in user input.\n"
            "Do not change the meaning of the sentence.\n"
            "Return only the corrected sentence."
        )
        
        response = client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Input:\n\"{claim}\""}
            ],
            temperature=0.0,
            max_tokens=150,
        )
        
        corrected = response.choices[0].message.content.strip()
        
        # Clean up possible prefix output formats
        if corrected.lower().startswith("output:"):
            corrected = corrected[7:].strip()
            
        # Strip quotes if the LLM wrapped it
        if corrected.startswith('"') and corrected.endswith('"'):
            corrected = corrected[1:-1].strip()
            
        if corrected:
            logger.info(f"✨ LLM Spell Correction: '{claim}' -> '{corrected}'")
            return corrected
            
    except Exception as e:
        logger.warning(f"⚠️ LLM spell correction failed: {e}. Falling back to TextBlob.")
        
    # 2. Fallback local dictionary check using TextBlob
    try:
        blob = TextBlob(claim)
        corrected_fallback = str(blob.correct())
        logger.info(f"🔧 TextBlob Fallback Spell Correction: '{claim}' -> '{corrected_fallback}'")
        return corrected_fallback
    except Exception as e:
        logger.error(f"❌ TextBlob fallback failed: {e}. Returning original claim.")
        
    return claim
