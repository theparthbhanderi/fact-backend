import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.models.factcheck_record import FactCheckRecord
from app.services.claim_memory_engine import store_claim_memory
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

def save_fact_check(
    db: Session,
    original_claim: str,
    claim: str,
    verdict: str,
    confidence: float,
    explanation: str,
    evidence: List[Dict[str, Any]],
    search_queries: Optional[List[str]] = None,
) -> FactCheckRecord:
    """Save a new fact-check result to the database."""
    try:
        new_record = FactCheckRecord(
            original_claim=original_claim,
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            evidence=evidence
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        logger.info(f"💾 Saved fact-check to database: {claim[:30]}... (ID: {new_record.id})")

        # Store in claim memory (FAISS) for fast future retrieval
        store_claim_memory(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            evidence=evidence or [],
            search_queries=search_queries or [],
        )

        # Evidence learning: store high-quality evidence sentences individually
        try:
            store = VectorStore()
            docs = []
            for i, e in enumerate(evidence or []):
                txt = (e.get("text") or e.get("snippet") or "").strip()
                if not txt:
                    continue
                url = e.get("url") or ""
                docs.append(
                    {
                        "doc_id": f"evidence::{new_record.id}::{i}",
                        "doc_type": "evidence_memory",
                        "title": e.get("title", ""),
                        "url": url if url else f"memory://evidence/{new_record.id}/{i}",
                        "source": e.get("source", ""),
                        "text": txt[:300] + ("…" if len(txt) > 300 else ""),
                        "timestamp": str(new_record.timestamp),
                        "extra_claim": claim,
                        "extra_verdict": verdict,
                    }
                )
            if docs:
                added = store.add_documents(docs)
                if added:
                    logger.info("memory_store_success evidence_sentences_added=%s", added)
        except Exception as e:
            logger.warning("Evidence memory store failed: %s", e)

        return new_record
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to save fact-check to DB: {str(e)}")
        # We don't want to crash the whole pipeline if DB save fails
        return None

def get_recent_fact_checks(db: Session, limit: int = 10) -> List[FactCheckRecord]:
    """Retrieve the most recent fact-checks."""
    return db.query(FactCheckRecord).order_by(FactCheckRecord.timestamp.desc()).limit(limit).all()

def search_fact_checks(db: Session, query: str, limit: int = 10) -> List[FactCheckRecord]:
    """Search previous fact-checks by claim text."""
    search_term = f"%{query}%"
    return db.query(FactCheckRecord)\
             .filter(FactCheckRecord.claim.ilike(search_term))\
             .order_by(FactCheckRecord.timestamp.desc())\
             .limit(limit)\
             .all()
