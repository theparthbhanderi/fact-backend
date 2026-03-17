"""
Translation Service.

Translates structured fact-check result objects into target languages
(like Hindi or Gujarati) using an LLM. Designed to keep structural 
keys intact while translating the meaningful text fields:
- verdict
- explanation
- claim / original_claim / corrected_claim
- evidence text/snippets

Does NOT translate URLs, internal keys, or proper noun Source Names.
"""

import json
import logging
from app.config import settings
from app.services.llm_analyzer import get_client, _parse_llm_response

logger = logging.getLogger(__name__)

TRANSLATION_PROMPT = """You are an expert translator specializing in preserving factual accuracy and tone.

Your goal is to translate a structured fact-check JSON object from English into the requested target language.

Follow these strict rules:
1. Translate ONLY the text values of these specific keys:
   - claim
   - original_claim
   - corrected_claim
   - normalized_claim
   - verdict (e.g., TRUE -> सच / True, FALSE -> झूठ / Khotu)
   - explanation
   - title (inside evidence objects)
   - text / snippet / content (inside evidence objects)
2. DO NOT translate keys themselves (i.e. 'verdict' must stay 'verdict').
3. DO NOT translate URLs, website links, or numeric confidence scores.
4. Try to keep publisher/source organization names (e.g., "Reuters", "World Health Organization") in their original English or a highly recognized transliteration if appropriate, but do not aggressively translate them.
5. The output MUST be valid, well-formed JSON matching the exact original structure.

Target Language: {target_language}

Original JSON to translate:
{json_payload}
"""

def translate_fact_check_result(result_data: dict, target_language: str) -> dict:
    """
    Translates a fact-check result dictionary into the target language using an LLM.

    Args:
        result_data: The original fact-check result dictionary.
        target_language: The target language code or name (e.g., 'hi', 'gu', 'Hindi', 'Gujarati').

    Returns:
        A new dictionary with translated string fields, preserving original structure.
    """
    logger.info(f"🌐 Translating fact-check result to {target_language}...")

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your_"):
        logger.error("OPENAI_API_KEY is not configured for translation.")
        raise RuntimeError("OPENAI_API_KEY is not set. Translation unavailable.")

    client = get_client()

    # Map language codes to expressive names for the prompt
    lang_map = {
        "hi": "Hindi (हिंदी)",
        "gu": "Gujarati (ગુજરાતી)",
        "en": "English",
        "es": "Spanish (Español)",
    }
    
    expressive_target = lang_map.get(target_language.lower(), target_language)

    # Convert the payload to a string
    json_payload_str = json.dumps(result_data, ensure_ascii=False, indent=2)

    prompt = TRANSLATION_PROMPT.format(
        target_language=expressive_target,
        json_payload=json_payload_str
    )

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL_NAME, # Using the configured model (e.g. gpt-4o-mini)
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.1, # Low temperature for accurate, deterministic translation
        )

        content = response.choices[0].message.content.strip()
        logger.info(f"📝 LLM translation response received ({len(content)} chars)")
        
        translated_result = _parse_llm_response(content)
        
        # Merge back any missing structural keys just in case the LLM trimmed them
        for key in ["confidence", "confidence_breakdown"]:
            if key in result_data and key not in translated_result:
                translated_result[key] = result_data[key]

        return translated_result

    except Exception as e:
        logger.error(f"❌ Translation failed: {e}")
        # Soft fallback: return original if translation fails
        return result_data
