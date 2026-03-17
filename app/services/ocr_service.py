import logging
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

_tesseract_available: Optional[bool] = None


def _easyocr_extract_in_subprocess(image_path: str) -> str:
    """
    Run EasyOCR inside a subprocess.

    Torch/OpenMP can segfault the interpreter on some macOS builds. Doing OCR in
    a subprocess ensures a crash does NOT take down the FastAPI server process.
    """
    import os

    # Reduce chance of OpenMP-related crashes / oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(image_path)
    extracted_texts = [res[1] for res in results]
    return " ".join(extracted_texts)


def _try_easyocr(image_path: str, timeout_s: int = 60) -> Optional[str]:
    """
    Attempt EasyOCR without importing torch in the main process.
    Returns extracted text or None if unavailable/crashes/timeouts.
    """
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
            fut = ex.submit(_easyocr_extract_in_subprocess, image_path)
            return fut.result(timeout=timeout_s)
    except Exception as e:
        logger.warning(f"⚠️ EasyOCR failed (will fallback): {str(e)}")
        return None


def _check_tesseract_available() -> bool:
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available
    try:
        import pytesseract

        _ = pytesseract.get_tesseract_version()
        _tesseract_available = True
        return True
    except Exception:
        _tesseract_available = False
        return False

def clean_ocr_text(text: str) -> str:
    """Clean the text extracted by OCR."""
    if not text:
        return ""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an uploaded image.

    Strategy:
    - Prefer EasyOCR if installed (no external system dependency).
    - Fallback to Tesseract (pytesseract) if available.
    """
    try:
        logger.info(f"🖼️ Running OCR on: {image_path}")

        # Prefer EasyOCR (subprocess) when available
        text = _try_easyocr(image_path)
        if text:
            cleaned = clean_ocr_text(text)
            logger.info(f"✅ EasyOCR Extracted: {len(cleaned)} chars")
            return cleaned

        if _check_tesseract_available():
            import pytesseract
            from PIL import Image

            logger.info(f"🖼️ Running OCR (Tesseract) on: {image_path}")
            text = pytesseract.image_to_string(Image.open(image_path))
            cleaned = clean_ocr_text(text)
            logger.info(f"✅ Tesseract Extracted: {len(cleaned)} chars")
            return cleaned

        raise RuntimeError(
            "OCR is not available. Install `easyocr` (recommended) or install Tesseract "
            "and ensure it is on PATH."
        )

    except Exception as e:
        logger.error(f"❌ Local OCR extraction failed: {str(e)}")
        raise e
