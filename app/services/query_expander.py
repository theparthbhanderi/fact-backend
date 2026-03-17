"""
Query Expander Agent.

Uses Mistral Small (via OpenRouter) to generate highly optimized, 
platform-agnostic search queries from a raw claim.
"""

import json
import logging
import re
from app.config import settings
from app.services.openrouter_client import get_openrouter_client
from app.services.claim_memory_engine import search_similar_claim_memory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert search query generator for a fact-checking and journalism platform.
Your job is to take a raw claim and convert it into high-yield search queries.

Rules:
1. Generate exactly 5 optimized search queries.
2. Query 1 (literal): Direct, literal search of key entities + event.
3. Query 2 (keywords): Simplified keyword search (drop prepositions/minor words).
4. Query 3 (fact-check): Must include phrase like "fact check" or "debunk" or "verified".
5. Query 4 (rumor/hoax): Must include terms like "rumor", "hoax", "fake", "viral".
6. Query 5 (official statement): Must include terms like "official statement", "government", "agency", "press release".
7. Return ONLY a valid JSON array of exactly 5 strings. Do not include any conversational text.

Example input: "Pakistan hospital strike killed 400 people"
Example output:
[
  "Pakistan hospital strike killed 400 people",
  "Pakistan hospital strike 400 killed casualties",
  "Fact check Pakistan hospital strike 400 killed",
  "Pakistan hospital strike rumor hoax viral",
  "Official statement Pakistan hospital strike press release"
]
"""

def generate_search_queries(claim: str) -> list[str]:
    """
    Generate optimized search queries for a given claim using Mistral Small.
    """
    logger.info(f"🔍 Generating search queries for: '{claim}'")

    if not claim or not claim.strip():
        return []

    mem_q: list[str] = []
    # Memory-assisted query expansion: reuse best past queries for similar claims
    try:
        mem = search_similar_claim_memory(claim)
        if mem.get("memory_match") and mem.get("search_queries"):
            mem_q = [q for q in mem.get("search_queries") if isinstance(q, str) and q.strip()]
            if mem_q:
                logger.info("QueryExpanderMemoryReuse similarity=%.2f queries=%s", float(mem.get("similarity_score", 0.0) or 0.0), mem_q)
    except Exception:
        mem_q = []

    try:
        client = get_openrouter_client()
        response = client.chat.completions.create(
            model=settings.MODEL_QUERY_EXPANDER,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Claim:\n{claim}"},
            ],
            temperature=0.3,
            max_tokens=150,
        )

        content = response.choices[0].message.content.strip()
        
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            queries = json.loads(content)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                merged = []
                for q in mem_q:
                    if q not in merged:
                        merged.append(q)
                for q in queries:
                    if q not in merged:
                        merged.append(q)
                # Enforce exactly 5, but preserve memory-derived queries first.
                merged = [q.strip() for q in merged if isinstance(q, str) and q.strip()]
                merged = merged[:5] if len(merged) >= 5 else merged
                logger.info(f"✅ Generated queries: {merged}")
                return merged
            else:
                raise ValueError("LLM did not return a list of strings")
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON array from Query Expander.")
            # Fallback
            base = [q for q in mem_q if isinstance(q, str) and q.strip()]
            if base:
                return base[:5]
            return [
                claim,
                _keywords_fallback(claim),
                f"Fact check {claim}",
                f"{claim} rumor hoax",
                f"Official statement {claim}",
            ]

    except Exception as e:
        logger.error(f"❌ Query expansion failed: {e}")
        # Fallback to the raw claim
        base = [q for q in mem_q if isinstance(q, str) and q.strip()]
        if base:
            return base[:5]
        return [
            claim,
            _keywords_fallback(claim),
            f"Fact check {claim}",
            f"{claim} rumor hoax",
            f"Official statement {claim}",
        ]


def _keywords_fallback(claim: str) -> str:
    # cheap keywords-only query (keeps entities, drops filler)
    words = [w for w in re.split(r"\s+", (claim or "").strip()) if w]
    stop = {"the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "for", "with", "at", "by", "from", "as", "is", "are", "was", "were"}
    kept = [w for w in words if w.lower() not in stop]
    return " ".join(kept[:12]) if kept else claim
