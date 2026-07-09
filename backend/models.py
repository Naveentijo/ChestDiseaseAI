import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from backend.database import Base

class PredictionRecord(Base):
    """
    SQLAlchemy model storing clinician prediction history and details.
    """
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True, nullable=True)
    image_name = Column(String, nullable=False)
    
    # Store predictions dictionary as JSON string to support expanding label sets
    predictions = Column(Text, nullable=False)
    
    confidence_score = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
