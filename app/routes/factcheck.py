"""
Fact-Check API route.

Defines the POST /fact-check endpoint that runs the complete
AI fact-checking pipeline:

    Claim → News Search → Article Extraction → Embeddings
    → FAISS Vector Search → LLM Analysis (RAG)
    → Multi-Signal Confidence Scoring → Verdict
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models.request_models import FactCheckRequest
from app.models.response_models import (
    FactCheckResponse,
    ConfidenceBreakdown,
    EvidenceItem,
    ClaimResult,
)
from app.services.factcheck_engine import run_fact_check_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Fact-Check"])

def _build_snippet(e: dict) -> str:
    # UI should prefer the extracted evidence sentence
    snippet = (
        e.get("text")
        or e.get("sentence")
        or e.get("snippet")
        or e.get("content")
        or e.get("description")
        or ""
    )
    snippet = snippet.strip()
    if len(snippet) > 300:
        return snippet[:300].rstrip() + "…"
    return snippet

def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@router.post("/fact-check", response_model=FactCheckResponse)
async def fact_check(request: FactCheckRequest):
    """
    Perform a full AI-powered fact-check on the given claim.

    Complete Pipeline:
        1. Search for relevant news articles via NewsAPI.
        2. Extract full article text using newspaper3k.
        3. Generate embeddings with Sentence-Transformers.
        4. Store in FAISS and perform semantic similarity search.
        5. Send top evidence + claim to LLM for RAG analysis.
        6. Calculate multi-signal confidence score.
        7. Return structured verdict with breakdown.

    Args:
        request: FactCheckRequest containing the claim string.

    Returns:
        FactCheckResponse with verdict, confidence, breakdown, and evidence.
    """
    try:
        claim = request.claim
        logger.info(f"📨 Received fact-check request: '{claim}'")

        # ── Run the full pipeline ──────────────────────────────────
        result = run_fact_check_pipeline(claim)

        # ── Build response ─────────────────────────────────────────
        breakdown = result.get("confidence_breakdown", {})
        claims_results = result.get("claims", []) or []

        def _to_claim_result(r: dict) -> ClaimResult:
            br = r.get("confidence_breakdown", {}) or {}
            return ClaimResult(
                claim_id=r.get("claim_id", ""),
                original_claim=r.get("original_claim", ""),
                corrected_claim=r.get("corrected_claim", ""),
                verdict=r.get("verdict", "UNVERIFIED"),
                confidence=float(r.get("confidence", 0.0) or 0.0),
                confidence_breakdown=ConfidenceBreakdown(
                    llm_confidence=br.get("llm_confidence", 0.0),
                    avg_similarity=br.get("avg_similarity", 0.0),
                    avg_source_score=br.get("avg_source_score", 0.0),
                ),
                explanation=r.get("explanation", ""),
                evidence=[
                    EvidenceItem(
                        title=e.get("title", ""),
                        url=e.get("url", ""),
                        source=e.get("source", ""),
                        snippet=_build_snippet(e),
                        credibility_score=_clamp01(float(e.get("source_credibility", e.get("credibility_score", 0.0)) or 0.0)),
                        source_credibility=_clamp01(float(e.get("source_credibility", e.get("credibility_score", 0.0)) or 0.0)),
                        similarity_score=_clamp01(
                            float(e.get("similarity_score", e.get("score", 0.0)) or 0.0)
                        ),
                        evidence_rank=int(e.get("evidence_rank", 0) or 0),
                    )
                    for e in (r.get("evidence", []) or [])
                ],
            )

        response = FactCheckResponse(
            original_claim=result["original_claim"],
            corrected_claim=result["corrected_claim"],
            verdict=result["verdict"],
            confidence=result["confidence"],
            confidence_breakdown=ConfidenceBreakdown(
                llm_confidence=breakdown.get("llm_confidence", 0.0),
                avg_similarity=breakdown.get("avg_similarity", 0.0),
                avg_source_score=breakdown.get("avg_source_score", 0.0),
            ),
            explanation=result["explanation"],
            evidence=[
                EvidenceItem(
                    title=e.get("title", ""),
                    url=e.get("url", ""),
                    source=e.get("source", ""),
                    snippet=_build_snippet(e),
                    credibility_score=_clamp01(float(e.get("source_credibility", e.get("credibility_score", 0.0)) or 0.0)),
                    source_credibility=_clamp01(float(e.get("source_credibility", e.get("credibility_score", 0.0)) or 0.0)),
                    similarity_score=_clamp01(
                        float(e.get("similarity_score", e.get("score", 0.0)) or 0.0)
                    ),
                    evidence_rank=int(e.get("evidence_rank", 0) or 0),
                )
                for e in result.get("evidence", [])
            ],
            claims=[_to_claim_result(r) for r in claims_results],
        )

        logger.info(
            f"✅ Returning verdict: {response.verdict} "
            f"({response.confidence:.0%}) for: '{claim}'"
        )

        return response

    except RuntimeError as e:
        logger.error(f"❌ Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fact-check pipeline error: {str(e)}",
        )
