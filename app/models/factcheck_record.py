from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON
from datetime import datetime
from app.database import Base

class FactCheckRecord(Base):
    __tablename__ = "fact_checks"

    id = Column(Integer, primary_key=True, index=True)
    original_claim = Column(Text, index=True, nullable=True)
    claim = Column(Text, index=True, nullable=False)
    verdict = Column(String, index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    explanation = Column(Text, nullable=False)
    
    # Store the complex list of evidence dictionaries as JSON
    evidence = Column(JSON, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
