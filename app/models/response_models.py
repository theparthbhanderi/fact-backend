"""
Pydantic response models for the AI Fact-Checker API.

These models define the shape of outgoing API responses,
ensuring consistent, well-documented JSON structures.
"""

from enum import Enum
from pydantic import BaseModel, Field


class Verdict(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    MISLEADING = "MISLEADING"
    UNVERIFIED = "UNVERIFIED"
    DISPUTED = "DISPUTED"


class EvidenceItem(BaseModel):
    """
    A single piece of evidence retrieved during fact-checking.

    Attributes:
        title: Title or headline of the evidence article.
        url: URL linking to the original source article.
        source: Name of the publishing source (e.g., Reuters, BBC).
        snippet: Short extracted snippet of the relevant text.
    """

    title: str = Field(..., description="Headline of the evidence article.")
    url: str = Field(..., description="URL to the original source.")
    source: str = Field(default="", description="Publishing source name.")
    snippet: str = Field(default="", description="Short snippet of relevant text.")
    credibility_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Source credibility score (0–1).",
    )
    source_credibility: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Source credibility score (0–1).",
    )
    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score to the claim (0–1).",
    )
    evidence_rank: int = Field(
        default=0,
        ge=0,
        description="Rank of this evidence sentence in the final global selection (1 = best).",
    )


class ConfidenceBreakdown(BaseModel):
    """
    Transparency breakdown of the confidence score.

    Shows how each signal contributes to the final confidence.
    """

    llm_confidence: float = Field(
        ..., description="LLM's self-reported confidence (0–1)."
    )
    avg_similarity: float = Field(
        ..., description="Average semantic similarity score (0–1)."
    )
    avg_source_score: float = Field(
        ..., description="Average source credibility score (0–1)."
    )

    agreement_score: float = Field(
        default=0.0, description="Cross-source agreement score (0–1)."
    )
    knowledge_score: float = Field(
        default=0.0, description="Knowledge graph support score (0–1)."
    )
    memory_hit: bool = Field(
        default=False, description="Whether FAISS memory fast-path was used."
    )
    cached: bool = Field(
        default=False, description="Whether DB fast-path cache was used."
    )


class ClaimResult(BaseModel):
    claim_id: str = Field(..., description="Stable identifier for this claim within the request.")
    original_claim: str = Field(..., description="Original raw input submitted by the user.")
    corrected_claim: str = Field(..., description="Corrected claim used for retrieval/judging.")
    verdict: Verdict = Field(..., description="Verdict for this claim.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0–1.0).")
    confidence_breakdown: ConfidenceBreakdown = Field(..., description="Breakdown of confidence scoring signals.")
    explanation: str = Field(..., description="Explanation of the verdict.")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Evidence items used for this claim.")


class FactCheckResponse(BaseModel):
    """
    Full response from the fact-check pipeline.

    Attributes:
        claim: The original claim that was fact-checked.
        verdict: The fact-check result — one of TRUE, FALSE, or UNVERIFIED.
        confidence: Multi-signal confidence score between 0.0 and 1.0.
        confidence_breakdown: Breakdown of individual scoring signals.
        explanation: Human-readable explanation of the verdict.
        evidence: List of evidence sources used in the analysis.
    """

    original_claim: str = Field(..., description="The original messy claim entered by the user.")
    corrected_claim: str = Field(..., description="The automatically corrected claim.")
    verdict: Verdict = Field(
        ...,
        description="Verdict: TRUE, FALSE, MISLEADING, UNVERIFIED, or DISPUTED.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Multi-signal confidence score (0.0 – 1.0).",
    )
    confidence_breakdown: ConfidenceBreakdown = Field(
        ...,
        description="Breakdown of confidence scoring signals.",
    )
    explanation: str = Field(
        ...,
        description="Explanation of the fact-check verdict.",
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence sources used in the analysis.",
    )

    # Multi-claim support (backward compatible: top-level fields mirror the first claim)
    claims: list[ClaimResult] = Field(
        default_factory=list,
        description="Per-claim fact-check results when multiple claims are extracted.",
    )
