import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models.factcheck_record import FactCheckRecord

logger = logging.getLogger(__name__)

def save_fact_check(db: Session, original_claim: str, claim: str, verdict: str, confidence: float, explanation: str, evidence: List[Dict[str, Any]]) -> FactCheckRecord:
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
