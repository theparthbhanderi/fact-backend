from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict, Any
import logging
import os
import shutil
import tempfile
from app.services.ocr_service import extract_text_from_image
from app.services.factcheck_engine import run_fact_check_pipeline

# Initialize Router
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/fact-check-image")
async def fact_check_image(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    1) Receives uploaded screenshot
    2) Extracts text via OCR
    3) Runs standard fact-checking pipeline on extracted text
    4) Returns verdict
    """
    logger.info(f"📨 Received image for OCR fact-check: {image.filename}")

    # Validate file type
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload PNG, JPG, or JPEG."
        )

    temp_file_path = None
    try:
        # 1. Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as temp_file:
            shutil.copyfileobj(image.file, temp_file)
            temp_file_path = temp_file.name

        # 2. Extract text with OCR
        extracted_claim = extract_text_from_image(temp_file_path)
        
        # Check if text was found
        if not extracted_claim or len(extracted_claim.strip()) < 5:
            # If text is too short or empty, reject it
            logger.warning("⚠️ OCR found no readable text or text is too short.")
            raise HTTPException(
                status_code=400,
                detail="No readable text detected in the image. Please upload a clear screenshot of the claim."
            )

        logger.info(f"📝 OCR Extracted Claim: '{extracted_claim}'")

        # 3. Send extracted text to existing fact-checking pipeline
        logger.info("🚀 Routing extracted claim to fact-check engine...")
        result = run_fact_check_pipeline(extracted_claim)
        
        return result

    except HTTPException as he:
        # Re-raise HTTP exceptions to pass them to client
        raise he
    except Exception as e:
        logger.error(f"❌ Error during image fact-checking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        # 4. Cleanup: Delete temp image
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"🧹 Cleaned up temporary image: {temp_file_path}")
