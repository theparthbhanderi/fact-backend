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
from app.services.source_credibility import get_source_credibility

logger = logging.getLogger(__name__)

# ── Weights (IMPROVEMENT 10) ───────────────────────────────────────────
W_LLM = 0.40           # LLM reasoning confidence
W_SOURCE = 0.30        # Source credibility
W_AGREEMENT = 0.20     # Evidence agreement (consensus bonus)
W_SIMILARITY = 0.10    # Evidence semantic similarity

# ── Limits ─────────────────────────────────────────────────────────────
MAX_CONFIDENCE = 0.98   # Hard cap


def calculate_confidence(
    llm_confidence: float,
    evidence_list: list[dict],
    agreement_score: float = 0.0,
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

    # ── Compute average similarity ─────────────────────────────────
    similarity_scores = [
        float(e.get("score", 0.0))
        for e in evidence_list
        if e.get("score") is not None
    ]
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
        + W_SIMILARITY * avg_similarity
        + W_SOURCE * avg_source_score
        + W_AGREEMENT * agreement_score
    )

    # ── Clamp & round ──────────────────────────────────────────────
    final = round(min(final, MAX_CONFIDENCE), 2)

    logger.info(
        f"📊 Confidence Engine:\n"
        f"   LLM confidence:   {llm_conf:.2f} (×{W_LLM})\n"
        f"   Avg similarity:   {avg_similarity:.2f} (×{W_SIMILARITY})\n"
        f"   Avg source score: {avg_source_score:.2f} (×{W_SOURCE})\n"
        f"   Agreement score:  {agreement_score:.2f} (×{W_AGREEMENT})\n"
        f"   Final confidence: {final}"
    )

    return {
        "final_confidence": final,
        "llm_confidence": round(llm_conf, 2),
        "avg_similarity": round(avg_similarity, 2),
        "avg_source_score": round(avg_source_score, 2),
        "agreement_score": round(agreement_score, 2),
    }


# Fallback if no sources provided
DEFAULT_SOURCE = 0.60
