import logging
import re
import requests
import base64
from app.config import settings

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
    Extract text from an uploaded image using the OpenRouter Vision API (e.g., Gemma 3).
    Replaces OCR.Space for a premium and fast experience.
    """
    try:
        logger.info(f"🖼️ Running OCR on image via OpenRouter Vision API: {image_path}")
        
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_img = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Determine image format
        ext = image_path.lower().split('.')[-1]
        base64_prefix = f"data:image/{ext};base64,"
        full_base64 = base64_prefix + base64_img

        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "AI Fact-Checker",
            "Content-Type": "application/json"
        }

        payload = {
            "model": settings.OCR_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all readable text from this image accurately. Return ONLY the extracted text. Do not provide markdown formatting around the output, no commentary, and no extra details."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": full_base64
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60 # Vision models can take a bit longer
        )
        
        if response.status_code != 200:
            logger.error(f"OpenRouter API Error: {response.text}")
            raise RuntimeError(f"OpenRouter API returned status {response.status_code}")
            
        result = response.json()
        
        if "choices" not in result or len(result["choices"]) == 0:
            logger.error(f"Invalid OpenRouter response structure: {result}")
            raise RuntimeError("OpenRouter API returned an invalid structure.")
            
        extracted_text = result["choices"][0]["message"]["content"]
        cleaned_text = clean_ocr_text(extracted_text)
        
        logger.info(f"✅ OCR Extracted Text ({len(cleaned_text)} chars)")
        return cleaned_text
        
    except Exception as e:
        logger.error(f"❌ OCR extraction failed: {str(e)}")
        raise e
