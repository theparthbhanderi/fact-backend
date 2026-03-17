import json
import logging
from typing import Dict, List, Any, Optional

from app.config import settings
from app.services.openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)


STANCE_SYSTEM_PROMPT = """You are a strict stance classifier for fact verification.
You will be given a CLAIM and a single EVIDENCE sentence.
Your job is to classify whether the evidence SUPPORTS, REFUTES, or is NEUTRAL toward the claim.

Rules:
- Use ONLY the evidence sentence.
- If the evidence does not directly speak to the claim, return NEUTRAL.
- Do not guess or infer missing context.
- Return ONLY valid JSON with the schema:
  {"stance": "SUPPORTS" | "REFUTES" | "NEUTRAL"}
"""


def classify_evidence_stance(claim: str, sentence: str) -> str:
    """
    Returns one of: SUPPORTS, REFUTES, NEUTRAL
    """
    sentence = (sentence or "").strip()
    if not sentence:
        return "NEUTRAL"

    try:
        client = get_openrouter_client()
    except Exception as e:
        logger.warning("Consensus stance classifier unavailable; defaulting to NEUTRAL. error=%s", e)
        return "NEUTRAL"

    prompt = f"""CLAIM:
{claim}

EVIDENCE:
{sentence}

TASK:
Determine whether the evidence supports, refutes, or is neutral toward the claim.
"""

    try:
        resp = client.chat.completions.create(
            model=settings.MODEL_EVIDENCE_ANALYST,
            messages=[
                {"role": "system", "content": STANCE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        stance = str(data.get("stance", "NEUTRAL")).upper().strip()
        if stance not in ("SUPPORTS", "REFUTES", "NEUTRAL"):
            return "NEUTRAL"
        return stance
    except Exception as e:
        logger.warning("Consensus stance classification failed; defaulting to NEUTRAL. error=%s", e)
        return "NEUTRAL"


def analyze_evidence_consensus(claim: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Input evidence_list items (sentence-level preferred):
      {"source": str, "text": str, "similarity_score": float, "credibility_score": float}

    Output:
    {
      "agreement_score": float,
      "supporting_sources": int,
      "contradicting_sources": int,
      "neutral_sources": int,
      "consensus_label": "supporting|contradicting|disputed|inconclusive"
    }
    """
    total = len(evidence_list or [])
    if total < 3:
        return {
            "agreement_score": 0.0,
            "supporting_sources": 0,
            "contradicting_sources": 0,
            "neutral_sources": 0,
            "consensus_label": "inconclusive",
        }

    stance_by_item: List[str] = []
    stance_sources: Dict[str, set] = {"SUPPORTS": set(), "REFUTES": set(), "NEUTRAL": set()}

    for ev in evidence_list:
        source = str(ev.get("source", "Unknown") or "Unknown")
        sentence = str(ev.get("text", "") or ev.get("snippet", "") or "")
        stance = classify_evidence_stance(claim, sentence)
        stance_by_item.append(stance)
        stance_sources.setdefault(stance, set()).add(source)

    supports = sum(1 for s in stance_by_item if s == "SUPPORTS")
    refutes = sum(1 for s in stance_by_item if s == "REFUTES")

    support_ratio = supports / total if total else 0.0
    refute_ratio = refutes / total if total else 0.0
    agreement_score = abs(support_ratio - refute_ratio)

    if support_ratio > 0.65:
        label = "supporting"
    elif refute_ratio > 0.65:
        label = "contradicting"
    elif 0.35 <= support_ratio <= 0.65 and 0.35 <= refute_ratio <= 0.65:
        label = "disputed"
    else:
        label = "inconclusive"

    result = {
        "agreement_score": float(max(0.0, min(1.0, agreement_score))),
        "supporting_sources": len(stance_sources.get("SUPPORTS", set())),
        "contradicting_sources": len(stance_sources.get("REFUTES", set())),
        "neutral_sources": len(stance_sources.get("NEUTRAL", set())),
        "consensus_label": label,
    }

    logger.info(
        'ConsensusEngine claim=%r supports=%s refutes=%s label=%s',
        claim[:120],
        supports,
        refutes,
        label,
    )
    logger.info(
        "consensus_support_count=%s consensus_refute_count=%s consensus_label=%s",
        supports,
        refutes,
        label,
    )
    return result

