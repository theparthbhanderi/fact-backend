from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging
from app.services.translation_service import translate_fact_check_result

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["Translation"]
)

class TranslationRequest(BaseModel):
    target_language: str
    result_data: dict

@router.post("/translate")
async def translate_result(request: TranslationRequest):
    """
    Translates a structured fact-check JSON payload into the target language.
    
    Valid target languages: 'en', 'hi', 'gu', 'es', etc.
    """
    logger.info(f"Received request to translate into {request.target_language}")
    
    # If English, just return early (no translation needed)
    if request.target_language.lower() == "en":
        return request.result_data

    try:
        translated_result = translate_fact_check_result(
            result_data=request.result_data,
            target_language=request.target_language
        )
        return translated_result
    except Exception as e:
        logger.error(f"Error during translation route: {e}")
        raise HTTPException(status_code=500, detail="Failed to translate fact-check result.")
