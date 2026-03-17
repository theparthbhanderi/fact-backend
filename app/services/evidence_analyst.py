"""
Evidence Analyst Agent.

Uses Llama-3.3-70B-Instruct (via OpenRouter) to read article excerpts/markdown 
and extract a bulleted list of sterile, objective facts relevant to the claim.
"""

import logging
import json
from typing import Any, Dict, List, Optional
from app.config import settings
from app.services.openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a meticulous investigative researcher.
Your ONLY job is to read provided evidence sentences and extract objective, verifiable facts related to the user's claim.

Rules:
1. You MUST NOT draw conclusions or state whether the claim is true or false.
2. You MUST NOT include your own opinion. Do not evaluate the reliability of the source.
3. Use ONLY the provided evidence sentences. Do NOT use outside knowledge.
4. Ignore irrelevant sentences.
5. Do NOT invent facts.
6. Every fact MUST cite the evidence sentence rank it came from.
7. The quote MUST be an exact substring (or near-exact) from that evidence sentence.
8. stance must be one of: "SUPPORTS", "REFUTES", "NEUTRAL".
9. Return ONLY a valid JSON object matching the format below.

Output format:
{
  "facts": [
    {
      "source": "Reuters",
      "fact": "Afghan officials reported fewer than 50 casualties.",
      "stance": "SUPPORTS",
      "evidence_rank": 3,
      "quote": "reported fewer than 50 casualties"
    }
  ]
}
"""

def _parse_and_validate(payload: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(payload)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    facts = obj.get("facts")
    if not isinstance(facts, list):
        return None
    cleaned: List[Dict[str, Any]] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        source = str(f.get("source", "") or "").strip()
        fact = str(f.get("fact", "") or "").strip()
        stance = str(f.get("stance", "") or "").strip().upper()
        quote = str(f.get("quote", "") or "").strip()
        try:
            rank = int(f.get("evidence_rank"))
        except Exception:
            rank = 0
        if not source or not fact or not quote or rank <= 0:
            continue
        if stance not in {"SUPPORTS", "REFUTES", "NEUTRAL"}:
            stance = "NEUTRAL"
        cleaned.append(
            {
                "source": source,
                "fact": fact,
                "stance": stance,
                "evidence_rank": rank,
                "quote": quote,
            }
        )
    if not cleaned:
        return None
    return {"facts": cleaned}


def extract_facts_from_evidence(claim: str, articles: list[dict]) -> str:
    """
    Reads retrieved articles and outputs a sterile fact dossier.
    """
    logger.info(f"📚 Extracting evidence dossier for claim using {len(articles)} articles.")

    if not articles:
        return ""

    evidence_lines = []
    for i, art in enumerate(articles, 1):
        source = art.get("source", "Unknown")
        sentence = (art.get("text") or art.get("sentence") or art.get("snippet") or "").strip()
        if not sentence:
            continue
        evidence_lines.append(f"{i}. [{source}] {sentence}")

    context_str = "\n".join(evidence_lines)

    try:
        client = get_openrouter_client()
        last = ""
        for attempt in range(1, 3):  # 1 initial + 1 retry if missing citations
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"CLAIM:\n{claim}\n\n"
                        f"EVIDENCE SENTENCES (ranked lines):\n{context_str}\n\n"
                        "TASK:\nExtract ONLY verifiable facts related to the claim.\n"
                        "Each fact MUST include evidence_rank and an exact quote from that ranked evidence line.\n"
                        "Return JSON only.\n"
                    ),
                },
            ]
            if attempt > 1:
                messages.append(
                    {
                        "role": "user",
                        "content": "RETRY: Your previous output had missing/invalid citations. Ensure EVERY fact has evidence_rank (integer) and quote (exact snippet).",
                    }
                )
            response = client.chat.completions.create(
                model=settings.MODEL_EVIDENCE_ANALYST,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=900,
            )
            content = response.choices[0].message.content.strip()
            last = content
            parsed = _parse_and_validate(content)
            if parsed is not None:
                dossier = json.dumps(parsed, ensure_ascii=False)
                logger.info("✅ Generated Fact Dossier facts=%s attempt=%s", len(parsed.get("facts", [])), attempt)
                return dossier

        logger.warning("Evidence Analyst produced invalid JSON after retry; returning empty dossier.")
        return ""

    except Exception as e:
        logger.error(f"❌ Evidence Analyst failed: {e}")
        return ""
