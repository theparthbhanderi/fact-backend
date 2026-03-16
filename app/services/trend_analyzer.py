import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.models.factcheck_record import FactCheckRecord

logger = logging.getLogger(__name__)

def get_most_checked_claims(db: Session, limit: int = 10) -> List[Dict[str, Any]]:
    """Return claims ordered by the total number of times they were checked."""
    results = (
        db.query(
            FactCheckRecord.claim, 
            func.count(FactCheckRecord.id).label('count')
        )
        .group_by(FactCheckRecord.claim)
        .order_by(desc('count'))
        .limit(limit)
        .all()
    )
    
    return [{"claim": r.claim, "count": r.count} for r in results]

def get_most_false_claims(db: Session, limit: int = 10) -> List[Dict[str, Any]]:
    """Return claims where verdict is FALSE, ordered by frequency."""
    results = (
        db.query(
            FactCheckRecord.claim, 
            func.count(FactCheckRecord.id).label('count')
        )
        .filter(FactCheckRecord.verdict == "FALSE")
        .group_by(FactCheckRecord.claim)
        .order_by(desc('count'))
        .limit(limit)
        .all()
    )
    
    return [{"claim": r.claim, "count": r.count} for r in results]

def get_daily_factcheck_stats(db: Session) -> List[Dict[str, Any]]:
    """Return the number of fact checks performed per day."""
    # SQLite datetime strings format: YYYY-MM-DD HH:MM:SS
    # func.date extracts the YYYY-MM-DD part
    results = (
        db.query(
            func.date(FactCheckRecord.timestamp).label('date'),
            func.count(FactCheckRecord.id).label('count')
        )
        .group_by(func.date(FactCheckRecord.timestamp))
        .order_by(func.date(FactCheckRecord.timestamp))
        .all()
    )
    
    return [{"date": r.date, "count": r.count} for r in results]
