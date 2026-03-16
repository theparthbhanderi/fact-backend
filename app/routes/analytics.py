from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.database import get_db
from app.services.trend_analyzer import (
    get_most_checked_claims,
    get_most_false_claims,
    get_daily_factcheck_stats
)

router = APIRouter(tags=["Analytics"])

@router.get("/analytics/trending", response_model=List[Dict[str, Any]])
def read_trending_claims(limit: int = 10, db: Session = Depends(get_db)):
    """Fetch the most frequently checked claims."""
    return get_most_checked_claims(db, limit=limit)

@router.get("/analytics/false-claims", response_model=List[Dict[str, Any]])
def read_false_claims(limit: int = 10, db: Session = Depends(get_db)):
    """Fetch the most common false claims."""
    return get_most_false_claims(db, limit=limit)

@router.get("/analytics/activity", response_model=List[Dict[str, Any]])
def read_activity_stats(db: Session = Depends(get_db)):
    """Fetch daily fact-check counts."""
    return get_daily_factcheck_stats(db)
