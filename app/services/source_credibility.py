import logging

logger = logging.getLogger(__name__)

# Expanded credibility database (IMPROVEMENT 4)
SOURCE_SCORES = {
    # Tier 1: Highly Trusted Fact-Checkers & Agencies
    "Reuters": 0.98,
    "Associated Press": 0.98,
    "AP News": 0.98,
    "BBC": 0.95,
    "BBC News": 0.95,
    "PBS": 0.93,
    "Nature": 0.95,
    "Science": 0.95,
    "WHO": 0.97,
    "World Health Organization": 0.97,
    "NASA": 0.97,
    "CDC": 0.96,
    "Snopes": 0.92,
    "PolitiFact": 0.92,
    "FactCheck.org": 0.92,
    "AFP Fact Check": 0.93,

    # Tier 2: Mainstream News (High Credibility)
    "The New York Times": 0.90,
    "The Washington Post": 0.90,
    "The Wall Street Journal": 0.88,
    "NPR": 0.92,
    "Financial Times": 0.88,
    "The Guardian": 0.88,
    "Al Jazeera": 0.85,

    # Tier 3: Mainstream News (Moderate-High Credibility)
    "CNN": 0.80,
    "NBC News": 0.85,
    "ABC News": 0.85,
    "CBS News": 0.85,
    "Bloomberg": 0.88,
    "Time": 0.82,
    "USA Today": 0.80,

    # Tier 4: Mixed/Polarized Credibility
    "Fox News": 0.65,
    "MSNBC": 0.70,
    "New York Post": 0.60,
    "The Daily Mail": 0.50,
    
    # Indian context (from previous step)
    "The Hindu": 0.88,
    "The Indian Express": 0.85,
    "NDTV": 0.80,
    "India Today": 0.75,
    "Times of India": 0.70,
    "Hindustan Times": 0.70,
}

def get_source_credibility(source_name: str, url: str = None) -> float:
    """
    Get the credibility score of a source.
    Returns a score between 0.0 and 1.0.
    """
    if not source_name:
        return _get_fallback_score(url)

    # Clean the source name for matching
    clean_name = source_name.strip()
    
    # Exact match
    if clean_name in SOURCE_SCORES:
        return SOURCE_SCORES[clean_name]
        
    # Case-insensitive match or partial match
    for key, score in SOURCE_SCORES.items():
        if key.lower() in clean_name.lower() or clean_name.lower() in key.lower():
            return score
            
    # Unknown source fallback
    return _get_fallback_score(url)

def _get_fallback_score(url: str) -> float:
    """Determine credibility fallback based on domain segments (e.g. .gov, .edu)."""
    base_unknown_score = 0.50
    
    if url:
        # Government domains are highly credible
        if ".gov" in url:
            return 0.95
        # Educational institutions
        if ".edu" in url:
            return 0.85
        # Organizational domains generally get a slight bump over standard .com
        if ".org" in url:
            return 0.60
            
    return base_unknown_score
