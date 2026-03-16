from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.database import get_db
from app.services.history_service import get_recent_fact_checks, search_fact_checks

router = APIRouter()

@router.get("/history", response_model=List[Dict[str, Any]])
def read_recent_history(limit: int = 10, db: Session = Depends(get_db)):
    """Fetch the most recent fact-check results."""
    records = get_recent_fact_checks(db, limit=limit)
    # Convert SQLAlchemy models to dicts
    return [
        {
            "id": r.id,
            "claim": r.claim,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "explanation": r.explanation,
            "evidence": r.evidence,
            "timestamp": r.timestamp.isoformat()
        } for r in records
    ]

@router.get("/history/search", response_model=List[Dict[str, Any]])
def search_history(q: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    """Search past claims by keyword."""
    records = search_fact_checks(db, query=q)
    return [
        {
            "id": r.id,
            "claim": r.claim,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "explanation": r.explanation,
            "evidence": r.evidence,
            "timestamp": r.timestamp.isoformat()
        } for r in records
    ]
