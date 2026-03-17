from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict, Any
import logging
import os
import shutil
import tempfile
from app.services.ocr_service import extract_text_from_image
from app.services.claim_extractor import extract_primary_claims
from app.services.factcheck_engine import run_fact_check_pipeline

# Initialize Router
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/fact-check-image")
async def fact_check_image(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    1) Receives uploaded screenshot
    2) Extracts text via OCR
    3) Isolates the factual claim via LLM
    4) Runs standard fact-checking pipeline on extracted claim
    5) Returns verdict
    """
    logger.info(f"📨 Received image for OCR fact-check: {image.filename}")

    # Validate file type
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload PNG, JPG, JPEG, or WEBP."
        )

    temp_file_path = None
    try:
        # 1. Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as temp_file:
            shutil.copyfileobj(image.file, temp_file)
            temp_file_path = temp_file.name

        # 2. Extract text with OCR
        raw_ocr_text = extract_text_from_image(temp_file_path)
        
        # Check if text was found
        if not raw_ocr_text or len(raw_ocr_text.strip()) < 5:
            logger.warning("⚠️ OCR found no readable text or text is too short.")
            raise HTTPException(
                status_code=400,
                detail="No readable text detected in the image. Please upload a clear screenshot of the claim."
            )

        logger.info(f"📝 Raw OCR Text: '{raw_ocr_text}'")

        # 3. Use LLM to cleanly extract the factual claim from OCR mess
        extracted_claims = extract_primary_claims(raw_ocr_text)

        if not extracted_claims:
            logger.warning("⚠️ LLM could not parse a factual claim from OCR text.")
            raise HTTPException(
                status_code=400,
                detail="No factual claims detected in this image. The image may just contain opinions or conversational text."
            )
            
        extracted_claim = extracted_claims[0]

        logger.info(f"🎯 Isolated Claim for Pipeline: '{extracted_claim}'")

        # 4. Send isolated claim to existing fact-checking pipeline
        logger.info("🚀 Routing isolated claim to fact-check engine...")
        result = run_fact_check_pipeline(extracted_claim)
        
        return result

    except HTTPException as he:
        # Re-raise HTTP exceptions to pass them to client
        raise he
    except Exception as e:
        logger.error(f"❌ Error during image fact-checking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        # 5. Cleanup: Delete temp image
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"🧹 Cleaned up temporary image: {temp_file_path}")
