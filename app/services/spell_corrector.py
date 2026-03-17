import logging

logger = logging.getLogger(__name__)

def correct_claim_text(claim: str) -> str:
    """
    Bypassed text correction.
    The new Claim Extractor Agent (Llama 3.2 3B) automatically corrects 
    grammar and fixes messy OCR text during the extraction phase, 
    making local aggressive spell checking (which harms proper nouns) obsolete.
    """
    return claim.strip() if claim else claim
