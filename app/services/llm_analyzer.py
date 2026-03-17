"""
LLM Analyzer Service.

Sends the claim and retrieved evidence to an LLM (OpenAI GPT-4o-mini)
for fact-check reasoning. Returns a structured verdict with confidence
score, explanation, and evidence references.

Architecture:
    - Uses the OpenAI Chat Completions API.
    - Designed for easy provider swapping (GPT → Llama → Mistral)
      by changing the model name and client.
    - Evidence is truncated to avoid token overflow.
    - Response is parsed as JSON with fallback handling.

Usage:
    from app.services.llm_analyzer import analyze_claim_with_llm
    result = analyze_claim_with_llm(claim, evidence_articles)
"""

import json
import logging
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# ── OpenAI-compatible client (initialized once) ───────────────────────
_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
)

def get_client() -> OpenAI:
    """Return the configured OpenAI client."""
    return _client

# ── Constants ──────────────────────────────────────────────────────────
MAX_ARTICLE_CHARS = 1500   # Max characters per article excerpt
MAX_ARTICLES = 3           # Max articles to include in prompt
LLM_TEMPERATURE = 0.1      # Very low temperature for strict factual reasoning

# ── System Prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional fact-checking assistant powered by a Retrieval-Augmented Generation (RAG) system.

Your job is to verify claims using ONLY the provided evidence.

Follow these strict rules:
1. You MUST NEVER rely on your internal knowledge. If the provided evidence does not contain sufficient information to verify the claim, you MUST mark it as UNVERIFIED.
2. If credible sources in the evidence contradict the claim, mark it FALSE.
3. If credible sources in the evidence confirm the claim, mark it TRUE.
4. Your reasoning MUST explicitly cite the provided evidence documents.

You MUST respond with valid JSON in this exact format:
{
"verdict": "TRUE" or "FALSE" or "UNVERIFIED",
"confidence": 0.85,
"reasoning": "Detailed reasoning based solely on the provided evidence.",
"supporting_sources": ["URL or exact Source Name of article 1"]
}"""


def _format_evidence(articles: list[dict]) -> str:
    """
    Format evidence articles into a structured text block for the LLM.

    Truncates each article to MAX_ARTICLE_CHARS and includes
    at most MAX_ARTICLES articles.

    Args:
        articles: List of article dicts with title, source, url, text.

    Returns:
        Formatted string of evidence articles.
    """
    limited = articles[:MAX_ARTICLES]
    blocks = []

    for i, article in enumerate(limited, 1):
        title = article.get("title", "Unknown")
        source = article.get("source", "Unknown")
        url = article.get("url", "")
        text = article.get("text", "")

        # Truncate article text to avoid token overflow
        excerpt = text[:MAX_ARTICLE_CHARS]
        if len(text) > MAX_ARTICLE_CHARS:
            excerpt += "..."

        blocks.append(
            f"Article {i}\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"URL: {url}\n"
            f"Excerpt:\n{excerpt}"
        )

    return "\n\n---\n\n".join(blocks)


def _parse_llm_response(content: str) -> dict:
    """
    Parse the LLM response as JSON with fallback handling.

    Tries direct JSON parsing first, then attempts to extract
    JSON from markdown code blocks.

    Args:
        content: Raw LLM response string.

    Returns:
        Parsed dict with verdict, confidence, explanation, evidence_used.
    """
    # Try direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    if "```json" in content:
        try:
            json_str = content.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass

    if "```" in content:
        try:
            json_str = content.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass

    # Fallback: return UNVERIFIED with the raw content as reasoning
    logger.warning("Failed to parse LLM response as JSON, using fallback.")
    return {
        "verdict": "UNVERIFIED",
        "confidence": 0.5,
        "reasoning": content,
        "supporting_sources": [],
    }


def analyze_claim_with_llm(claim: str, evidence: list[dict]) -> dict:
    """
    Analyze a claim using an LLM with retrieved evidence (RAG).

    Constructs a fact-checking prompt with the claim and evidence
    articles, sends it to the LLM, and parses the structured response.

    Args:
        claim: The news claim to fact-check.
        evidence: List of evidence article dicts, each containing
                  title, url, text, source, and score.

    Returns:
        A dict containing:
            - verdict (str): "TRUE", "FALSE", or "UNVERIFIED".
            - confidence (float): 0.0 – 1.0.
            - reasoning (str): Reasoning for the verdict.
            - supporting_sources (list[str]): Sources of articles used.

    Raises:
        RuntimeError: If the OpenAI API call fails.
    """
    logger.info(f"🤖 Analyzing claim with LLM: '{claim}'")

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your_"):
        logger.error("OPENAI_API_KEY is not configured.")
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add a valid key to your .env file."
        )

    # Format the evidence
    formatted_evidence = _format_evidence(evidence)

    # Build user prompt
    user_prompt = (
        f"Claim:\n{claim}\n\n"
        f"Evidence Sources:\n\n{formatted_evidence}\n\n"
        f"Tasks:\n"
        f"1. Analyze whether the claim is supported by reliable evidence.\n"
        f"2. Identify contradictions.\n"
        f"3. Determine the final verdict.\n"
    )

    try:
        response = _client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=800,
        )

        content = response.choices[0].message.content.strip()
        logger.info(f"📝 LLM raw response received ({len(content)} chars)")

        result = _parse_llm_response(content)

        # Validate and normalize fields
        verdict = result.get("verdict", "UNVERIFIED").upper()
        if verdict not in ("TRUE", "FALSE", "UNVERIFIED"):
            verdict = "UNVERIFIED"

        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reasoning = result.get("reasoning", result.get("explanation", "No explanation provided."))
        supporting_sources = result.get("supporting_sources", result.get("evidence_used", []))

        # Hallucination Prevention (IMPROVEMENT 7)
        # Verify that all sources referenced actually exist in our evidence list
        valid_source_strings = [e.get("source", "").lower() for e in evidence] + [e.get("url", "").lower() for e in evidence] + [e.get("title", "").lower() for e in evidence]
        hallucinated = False
        
        for source in supporting_sources:
            if not isinstance(source, str):
                continue
            # If the LLM returned a source that is not anywhere in our provided titles, URLs, or source names = Hallucination
            if not any(source.lower() in valid_str or valid_str in source.lower() for valid_str in valid_source_strings if valid_str):
                hallucinated = True
                break

        if hallucinated:
            logger.warning("🚨 Hallucination Detected! LLM cited an unknown source. Rejecting output.")
            return {
                "verdict": "UNVERIFIED",
                "confidence": 0.0,
                "reasoning": "Output rejected due to Hallucination constraints: Model cited sources not provided in evidence context.",
                "supporting_sources": []
            }

        logger.info(
            f"✅ LLM verdict: {verdict} "
            f"(confidence: {confidence:.2f})"
        )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": reasoning, # keeping explanation key for backward compatibility in backend API
            "evidence_used": supporting_sources,
        }

    except Exception as e:
        logger.error(f"❌ LLM analysis failed: {e}")
        raise RuntimeError(f"LLM analysis failed: {e}")
