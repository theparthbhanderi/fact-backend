import logging
import re
import requests
import base64

logger = logging.getLogger(__name__)

def clean_ocr_text(text: str) -> str:
    """Clean the text extracted by OCR."""
    if not text:
        return ""
    # Remove multiple newlines and spaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an uploaded image using the free OCR.Space API.
    Replaces Tesseract to avoid complex system requirements.
    """
    try:
        logger.info(f"🖼️ Running OCR on image via Cloud API: {image_path}")
        
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_img = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Determine image format
        ext = image_path.lower().split('.')[-1]
        base64_prefix = f"data:image/{ext};base64,"
        full_base64 = base64_prefix + base64_img

        # Send to OCR.Space API (free public key)
        payload = {
            "apikey": "helloworld",
            "base64Image": full_base64,
            "language": "eng",
            "OCREngine": 2, # Engine 2 is better for complex/screenshot text
        }
        
        response = requests.post(
            "https://api.ocr.space/parse/image",
            data=payload,
            timeout=30 # 30s timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"OCR API returned status {response.status_code}")
            
        result = response.json()
        
        if result.get("IsErroredOnProcessing"):
            error_msg = result.get('ErrorMessage', ['Unknown error'])[0]
            raise RuntimeError(f"OCR API error: {error_msg}")
            
        # Extract parsed text
        parsed_text = ""
        for parsed_result in result.get("ParsedResults", []):
            if parsed_result.get("ParsedText"):
                parsed_text += parsed_result["ParsedText"] + " "
                
        cleaned_text = clean_ocr_text(parsed_text)
        
        logger.info(f"✅ OCR Extracted Text ({len(cleaned_text)} chars)")
        return cleaned_text
        
    except Exception as e:
        logger.error(f"❌ OCR extraction failed: {str(e)}")
        raise e
