import hashlib
import logging
import re
from typing import Any, Dict, List

from app.services.claim_normalizer import normalize_claim

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]+")

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "them",
    "his",
    "her",
    "their",
    "our",
    "your",
    "my",
}


def detect_language(text: str) -> str:
    """
    Lightweight language detection without extra dependencies.
    - returns 'hi' if Devanagari present
    - returns 'gu' if Gujarati present
    - else 'en'
    """
    s = text or ""
    for ch in s:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            return "hi"
        if 0x0A80 <= code <= 0x0AFF:
            return "gu"
    return "en"


def extract_entities(text: str) -> List[str]:
    """
    Heuristic NER:
    - sequences of TitleCase tokens (e.g., "New York Times")
    - ALLCAPS acronyms (e.g., "NASA", "WHO")
    """
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]+|[A-Z]{2,}", text)
    entities: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if buf:
            ent = " ".join(buf).strip()
            if len(ent) >= 2 and ent not in entities:
                entities.append(ent)
            buf = []

    for t in tokens:
        if t.isupper() and len(t) >= 2:
            flush()
            if t not in entities:
                entities.append(t)
            continue

        if t[:1].isupper() and t[1:].islower():
            buf.append(t)
            continue

        flush()

    flush()
    return entities[:12]


def extract_topics(text: str) -> List[str]:
    """
    Keyword topics from normalized text:
    - lowercase
    - drop stopwords / short tokens
    - keep top unique terms by frequency
    """
    if not text:
        return []
    words = [w.lower() for w in _WORD_RE.findall(text)]
    words = [w for w in words if len(w) >= 3 and w not in _STOPWORDS]
    if not words:
        return []
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    topics = sorted(freq.keys(), key=lambda k: (freq[k], len(k)), reverse=True)
    return topics[:10]


def claim_hash(normalized_claim: str) -> str:
    s = (normalized_claim or "").strip().lower().encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def process_raw_claims(raw_input: str, claims: List[str]) -> List[Dict[str, Any]]:
    """
    Convert raw claim strings into structured claim objects, with hash-based dedupe.
    """
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for c in claims:
        claim = (c or "").strip()
        if not claim:
            continue
        lang = detect_language(claim)
        normalized = normalize_claim(claim)
        h = claim_hash(normalized)
        if h in seen:
            continue
        seen.add(h)
        out.append(
            {
                "claim": claim,
                "normalized_claim": normalized,
                "entities": extract_entities(claim),
                "topics": extract_topics(normalized),
                "language": lang,
                "hash": h,
                "raw_input": raw_input,
            }
        )

    logger.info("ClaimProcessor raw_len=%s claims_in=%s claims_out=%s", len(raw_input or ""), len(claims or []), len(out))
    return out

