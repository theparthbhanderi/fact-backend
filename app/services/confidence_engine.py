"""
Confidence Engine.

Calculates a final confidence score by combining multiple signals:

    1. LLM reasoning confidence (50%)
    2. Evidence semantic similarity (30%)
    3. Source credibility (20%)
    4. Multi-source agreement bonus (+0.05)

This replaces the raw LLM confidence with a more reliable,
multi-factor score inspired by professional fact-checking platforms.

Usage:
    from app.services.confidence_engine import calculate_confidence
    result = calculate_confidence(llm_confidence=0.85, evidence_list=[...])
"""

import logging
import urllib.parse
from app.services.source_credibility import get_source_credibility

logger = logging.getLogger(__name__)

# ── Weights (Production Spec) ───────────────────────────────────────────
# Phase 9 target:
# - Source credibility: 40%
# - Evidence agreement: 30%
# - LLM certainty: 20%
# - Semantic similarity: 10%
W_SOURCE = 0.40
W_AGREEMENT = 0.30
W_LLM = 0.20
W_SIMILARITY = 0.10
# Keep knowledge_score input for backward compatibility, but do not weight it by default.
W_KNOWLEDGE = 0.0

# ── Limits ─────────────────────────────────────────────────────────────
MAX_CONFIDENCE = 0.98   # Hard cap
MIN_CONFIDENCE = 0.05


def calculate_confidence(
    llm_confidence: float,
    evidence_list: list[dict],
    agreement_score: float = 0.0,
    knowledge_score: float = 0.0,
) -> dict:
    """
    Calculate a multi-signal confidence score.

    Combines LLM confidence, average evidence similarity, and
    average source credibility into a weighted final score.
    Adds a bonus if multiple credible sources agree.

    Args:
        llm_confidence: The LLM's self-reported confidence (0–1).
        evidence_list: List of evidence dicts.
        agreement_score: Decimal representing consensus (0.0 - 1.0).

    Returns:
        A dict containing:
            - final_confidence (float): Weighted score (0–0.98).
            - llm_confidence (float): Input LLM confidence.
            - avg_similarity (float): Mean similarity score.
            - avg_source_score (float): Mean source credibility.
            - agreement_score (float): Consensus score.
    """
    # ── Sanitize LLM confidence ────────────────────────────────────
    llm_conf = max(0.0, min(1.0, float(llm_confidence)))

    # ── Compute average similarity (retrieved evidence scores) ─────
    similarity_scores: list[float] = []
    for e in evidence_list:
        raw = (
            e.get("similarity_score")
            if e.get("similarity_score") is not None
            else e.get("score")
            if e.get("score") is not None
            else e.get("relevance_score")
            if e.get("relevance_score") is not None
            else 0.0
        )
        try:
            similarity_scores.append(float(raw))
        except Exception:
            similarity_scores.append(0.0)
    avg_similarity = (
        sum(similarity_scores) / len(similarity_scores)
        if similarity_scores
        else 0.0
    )

    # ── Compute average source credibility ─────────────────────────
    source_scores = [
        get_source_credibility(e.get("source", ""), e.get("url", ""))
        for e in evidence_list
    ]
    avg_source_score = (
        sum(source_scores) / len(source_scores)
        if source_scores
        else DEFAULT_SOURCE
    )

    # ── Weighted combination ───────────────────────────────────────
    final = (
        W_LLM * llm_conf
        + W_SOURCE * avg_source_score
        + W_AGREEMENT * agreement_score
        + W_SIMILARITY * avg_similarity
        + W_KNOWLEDGE * max(0.0, min(1.0, float(knowledge_score)))
    )

    # ── Source diversity penalty (unique domains) ──────────────────
    domains = set()
    for e in evidence_list:
        url = (e.get("url") or "").strip()
        if not url:
            continue
        try:
            netloc = urllib.parse.urlparse(url).netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            if netloc:
                domains.add(netloc)
        except Exception:
            continue
    unique_domains = len(domains)
    if unique_domains < 2 and evidence_list:
        final *= 0.7

    # ── Clamp & round ──────────────────────────────────────────────
    final = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, final))
    final = round(final, 2)

    logger.info(
        f"📊 Confidence Engine:\n"
        f"   LLM confidence:   {llm_conf:.2f} (×{W_LLM})\n"
        f"   Avg source score: {avg_source_score:.2f} (×{W_SOURCE})\n"
        f"   Agreement score:  {agreement_score:.2f} (×{W_AGREEMENT})\n"
        f"   Avg similarity:   {avg_similarity:.2f} (×{W_SIMILARITY})\n"
        f"   Knowledge score:  {float(knowledge_score):.2f} (×{W_KNOWLEDGE})\n"
        f"   Unique domains:   {unique_domains} (penalty={'0.7x' if unique_domains < 2 and evidence_list else 'none'})\n"
        f"   Final confidence: {final}"
    )

    return {
        "final_confidence": final,
        "llm_confidence": round(llm_conf, 2),
        "avg_similarity": round(avg_similarity, 2),
        "avg_source_score": round(avg_source_score, 2),
        "agreement_score": round(agreement_score, 2),
        "knowledge_score": round(float(knowledge_score), 2),
    }


# Fallback if no sources provided
DEFAULT_SOURCE = 0.60
