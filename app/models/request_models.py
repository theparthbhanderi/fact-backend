"""
Pydantic request models for the AI Fact-Checker API.

These models validate and document the shape of incoming
request payloads for all API endpoints.
"""

from pydantic import BaseModel, Field


class FactCheckRequest(BaseModel):
    """
    Request body for the POST /api/fact-check endpoint.

    Attributes:
        claim: The news claim or statement to be fact-checked.
               Example: "NASA discovered alien life on Mars."
    """

    claim: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The news claim or statement to fact-check.",
        json_schema_extra={"example": "NASA discovered alien life on Mars"},
    )
