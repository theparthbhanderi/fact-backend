import re

class ClaimNormalizer:
    """Normalize user claims for better search and LLM processing."""
    
    # Common sensational words to strip
    OPINION_WORDS = [
        r"\bbreaking\b", r"\bshocking\b", r"\bmust watch\b", r"\bviral\b", 
        r"\bunbelievable\b", r"\bmind blowing\b", r"\bexposed\b"
    ]
    
    @staticmethod
    def normalize(claim: str) -> str:
        """Clean and normalize a raw claim string."""
        normalized = claim
        
        # 1. Remove Emojis (basic range)
        normalized = re.sub(r'[^\w\s.,!?-]', '', normalized)
        
        # 2. Lowercase for opinion word replacement
        # We'll do case-insensitive regex instead to preserve original casing where needed
        for word_pattern in ClaimNormalizer.OPINION_WORDS:
            normalized = re.sub(word_pattern, "", normalized, flags=re.IGNORECASE)
            
        # 3. Remove repeated punctuation
        normalized = re.sub(r'!{2,}', '!', normalized)
        normalized = re.sub(r'\?{2,}', '?', normalized)
        normalized = re.sub(r'\.{2,}', '.', normalized)
        
        # 4. Clean up extra whitespace that might result from removals
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # 5. Trim leading/trailing punctuation if it's left hanging awkwardly
        normalized = re.sub(r'^[\W_]+|[\W_]+$', '', normalized)
        
        return normalized.strip()

# Singleton usage
claim_normalizer = ClaimNormalizer()

def normalize_claim(claim: str) -> str:
    """Helper function to normalize a claim."""
    return claim_normalizer.normalize(claim)
