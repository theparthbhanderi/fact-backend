import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np

from app.services.embedding_service import generate_embedding
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


MEMORY_SIMILARITY_THRESHOLD = 0.85


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float((a @ b) / denom)


def search_similar_claim_memory(claim: str) -> Dict[str, Any]:
    """
    Search the FAISS vector store for previously stored claim-level memory entries.

    Return:
    {
      "memory_match": bool,
      "similarity_score": float,
      "stored_claim": "...",
      "verdict": "...",
      "confidence": float,
      "explanation": "...",
      "evidence": [...],
      "sources": [...],
      "timestamp": "..."
    }
    """
    logger.info("memory_search_started claim=%r", (claim or "")[:120])

    store = VectorStore()
    results = store.search(claim, top_k=8)
    mem_results = [r for r in results if r.get("doc_type") == "claim_memory"]

    if not mem_results:
        return {
            "memory_match": False,
            "similarity_score": 0.0,
            "stored_claim": "",
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "explanation": "",
            "evidence": [],
            "sources": [],
            "timestamp": "",
        }

    best = mem_results[0]
    # Compute true cosine similarity for the memory threshold contract.
    # FAISS store uses L2 distance proxy; for claim-memory we need a stable 0–1 similarity.
    try:
        qv = generate_embedding(claim)
        sv = generate_embedding(best.get("text", "") or "")
        similarity = _cosine(qv, sv)
    except Exception:
        similarity = float(best.get("score", 0.0) or 0.0)
    payload = (best.get("extra") or {}).get("memory_payload", {}) or {}

    memory_match = similarity >= MEMORY_SIMILARITY_THRESHOLD
    if memory_match:
        logger.info("MemoryHit claim=%r similarity=%.2f", (claim or "")[:120], similarity)

    return {
        "memory_match": bool(memory_match),
        "similarity_score": similarity,
        "stored_claim": payload.get("claim", best.get("text", "")),
        "verdict": payload.get("verdict", "UNVERIFIED"),
        "confidence": float(payload.get("confidence", 0.0) or 0.0),
        "explanation": payload.get("explanation", ""),
        "evidence": payload.get("evidence", []) or [],
        "sources": payload.get("sources", []) or [],
        "timestamp": payload.get("timestamp", ""),
        "search_queries": payload.get("search_queries", []) or [],
    }


def store_claim_memory(
    *,
    claim: str,
    verdict: str,
    confidence: float,
    explanation: str,
    evidence: List[Dict[str, Any]],
    search_queries: Optional[List[str]] = None,
) -> bool:
    """
    Store a completed fact-check as a claim-memory entry in FAISS.
    """
    try:
        ts = datetime.utcnow().isoformat() + "Z"
        doc = {
            "doc_id": f"claim::{hash(claim)}::{ts}",
            "doc_type": "claim_memory",
            "title": "Claim Memory",
            "url": f"memory://claim/{hash(claim)}",
            "source": "MemoryEngine",
            "text": claim,
            "timestamp": ts,
            "memory_payload": {
                "claim": claim,
                "verdict": verdict,
                "confidence": float(confidence),
                "explanation": explanation,
                "evidence": evidence,
                "sources": list({(e.get('source') or '') for e in (evidence or []) if (e.get('source') or '').strip()}),
                "timestamp": ts,
                "search_queries": search_queries or [],
            },
        }
        store = VectorStore()
        added = store.add_documents([doc])
        logger.info(
            "memory_store_success claim=%r stored=%s",
            (claim or "")[:120],
            bool(added),
        )
        return bool(added)
    except Exception as e:
        logger.warning("memory_store_failed claim=%r error=%s", (claim or "")[:120], e)
        return False

