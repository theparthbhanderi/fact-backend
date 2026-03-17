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
)
from app.services.factcheck_engine import run_fact_check_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Fact-Check"])


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
                    snippet=e.get("content", ""),
                )
                for e in result.get("evidence", [])
            ],
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
