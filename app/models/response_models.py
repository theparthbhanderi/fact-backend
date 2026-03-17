"""
Pydantic response models for the AI Fact-Checker API.

These models define the shape of outgoing API responses,
ensuring consistent, well-documented JSON structures.
"""

from pydantic import BaseModel, Field


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
    verdict: str = Field(
        ...,
        description="Verdict: TRUE, FALSE, or UNVERIFIED.",
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
