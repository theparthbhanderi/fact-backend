import logging
import re
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import requests

from app.services.embedding_service import generate_embedding
from app.services.knowledge_graph import check_google_knowledge_graph

logger = logging.getLogger(__name__)

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# Reliability priors (as specified)
WIKIDATA_RELIABILITY = 0.85
GOOGLE_KG_RELIABILITY = 0.90
WIKIPEDIA_RELIABILITY = 0.85


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float((a @ b) / denom)


def _extract_search_term(claim: str) -> str:
    # Keep it simple: pick first few informative words to avoid broken SPARQL filters.
    stop = {
        "did",
        "does",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "in",
        "on",
        "of",
        "to",
        "and",
        "or",
        "with",
        "from",
        "for",
        "by",
        "every",
        "morning",
        "can",
        "will",
        "confirmed",
    }
    words = re.findall(r"[A-Za-z0-9\-']+", (claim or "").lower())
    picked = [w for w in words if len(w) > 3 and w not in stop]
    if not picked:
        return (claim or "").strip()[:40]
    return " ".join(picked[:4])


def _wikidata_sparql(term: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Queries Wikidata for entities whose label contains term and returns structured sources.
    Output items: {source,title,url,snippet}
    """
    if not term:
        return []

    query = f"""
    SELECT ?item ?itemLabel ?description WHERE {{
      ?item rdfs:label ?itemLabel.
      OPTIONAL {{ ?item schema:description ?description. FILTER(LANG(?description) = "en") }}
      FILTER(CONTAINS(LCASE(?itemLabel), LCASE("{term}")))
      FILTER(LANG(?itemLabel) = "en")
    }}
    LIMIT {int(limit)}
    """

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "AIFactChecker/1.0 (KnowledgeVerifier; contact: factchecker@example.com)",
    }

    # Fail-fast: knowledge layer must not stall the pipeline if Wikidata is slow/rate-limited.
    try:
        resp = requests.get(WIKIDATA_ENDPOINT, params={"query": query}, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        bindings = data.get("results", {}).get("bindings", []) or []
        out: List[Dict[str, Any]] = []
        for b in bindings:
            url = (b.get("item", {}) or {}).get("value", "") or ""
            title = (b.get("itemLabel", {}) or {}).get("value", "") or ""
            desc = (b.get("description", {}) or {}).get("value", "") or ""
            if not title and not desc:
                continue
            out.append(
                {
                    "source": "Wikidata",
                    "title": title,
                    "url": url,
                    "snippet": desc.strip(),
                }
            )
        return out
    except Exception as e:
        logger.warning("Wikidata SPARQL failed. term=%r error=%s", term, e)
        return []


def _google_kg_sources(claim: str) -> List[Dict[str, Any]]:
    """
    Uses existing knowledge_graph.check_google_knowledge_graph and returns structured sources.
    Output items: {source,title,url,snippet}
    """
    try:
        gkg = check_google_knowledge_graph(claim)
        if not gkg:
            return []
        return [
            {
                "source": "Google Knowledge Graph",
                "title": gkg.get("title", "") or "",
                "url": gkg.get("url", "") or "",
                "snippet": gkg.get("snippet", gkg.get("description", "")) or "",
            }
        ]
    except Exception as e:
        logger.warning("Google Knowledge Graph lookup failed. error=%s", e)
        return []


def _wikipedia_sources(claim: str, term: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Fallback knowledge source when Wikidata is slow/unavailable.
    Output items: {source,title,url,snippet}
    """
    try:
        import wikipedia  # type: ignore

        titles = []
        # Search both the raw claim and a distilled term for better entity recall.
        titles.extend(wikipedia.search(claim, results=limit) or [])
        if term and term.lower() not in (claim or "").lower():
            titles.extend(wikipedia.search(term, results=limit) or [])

        # Heuristic enrichments for common fact-check patterns
        lc = (claim or "").lower()
        if "india" in lc and "moon" in lc:
            titles.extend(["Chandrayaan programme", "Chandrayaan-3", "Moon landing"])

        # De-dupe while preserving order
        seen = set()
        deduped = []
        for t in titles:
            if not t or t in seen:
                continue
            seen.add(t)
            deduped.append(t)
        titles = deduped[: max(limit, 6)]

        out: List[Dict[str, Any]] = []
        for t in titles:
            try:
                page = wikipedia.page(t, auto_suggest=False)
                snippet = page.summary[:500] + "..." if len(page.summary) > 500 else page.summary
                out.append(
                    {
                        "source": "Wikipedia",
                        "title": page.title,
                        "url": page.url,
                        "snippet": snippet,
                    }
                )
            except Exception:
                continue
        return out
    except Exception:
        return []


def verify_claim_with_knowledge_graph(claim: str) -> Dict[str, Any]:
    """
    Knowledge graph verification fallback.

    Returns:
    {
      "knowledge_found": bool,
      "sources": [{source,title,url,snippet}],
      "knowledge_confidence": float,
      "knowledge_score": float,
      "knowledge_evidence": [evidence objects compatible with pipeline]
    }
    """
    logger.info('knowledge_verification_triggered claim=%r', (claim or "")[:120])

    term = _extract_search_term(claim)
    wikidata_sources = _wikidata_sparql(term, limit=5)
    google_sources = _google_kg_sources(claim)
    # If Wikidata times out / returns nothing, fall back to Wikipedia summaries for general knowledge claims.
    wikipedia_sources = _wikipedia_sources(claim, term=term, limit=3) if not wikidata_sources else []
    sources = wikidata_sources + wikipedia_sources + google_sources

    logger.info("knowledge_sources_found=%s", len(sources))

    if not sources:
        return {
            "knowledge_found": False,
            "sources": [],
            "knowledge_confidence": 0.0,
            "knowledge_score": 0.0,
            "knowledge_evidence": [],
        }

    claim_vec = generate_embedding(claim)
    best_score = 0.0
    knowledge_evidence: List[Dict[str, Any]] = []

    for src in sources:
        snippet = (src.get("snippet") or "").strip()
        if not snippet:
            continue
        sim = _cosine(generate_embedding(snippet), claim_vec)
        best_score = max(best_score, sim)

        if src.get("source") == "Wikidata":
            reliability = WIKIDATA_RELIABILITY
        elif src.get("source") == "Google Knowledge Graph":
            reliability = GOOGLE_KG_RELIABILITY
        else:
            reliability = WIKIPEDIA_RELIABILITY
        knowledge_score = (sim * 0.5) + (reliability * 0.5)

        knowledge_evidence.append(
            {
                "title": src.get("title", ""),
                "url": src.get("url", ""),
                "source": src.get("source", ""),
                "text": snippet[:300] + ("…" if len(snippet) > 300 else ""),
                "similarity_score": float(max(0.0, min(1.0, sim))),
                "score": float(max(0.0, min(1.0, sim))),
                "credibility_score": 0.9,
                "source_credibility": 0.9,
                "knowledge_score": float(max(0.0, min(1.0, knowledge_score))),
            }
        )

    # Aggregate knowledge confidence from best match
    # (kept separate from per-item knowledge_score used by confidence engine)
    knowledge_confidence = float(max(0.0, min(1.0, best_score)))
    avg_knowledge_score = (
        sum(e.get("knowledge_score", 0.0) for e in knowledge_evidence) / len(knowledge_evidence)
        if knowledge_evidence
        else 0.0
    )

    logger.info(
        'KnowledgeEngine claim=%r knowledge_sources=%s similarity_score=%.2f',
        (claim or "")[:120],
        len(sources),
        knowledge_confidence,
    )
    logger.info("knowledge_similarity_score=%.4f", knowledge_confidence)

    return {
        "knowledge_found": True,
        "sources": sources,
        "knowledge_confidence": knowledge_confidence,
        "knowledge_score": float(max(0.0, min(1.0, avg_knowledge_score))),
        "knowledge_evidence": knowledge_evidence,
    }

