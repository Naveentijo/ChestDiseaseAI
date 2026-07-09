from sqlalchemy.orm import Session
from typing import List, Optional
import json
from backend.models import PredictionRecord

class DBService:
    """
    DBService wraps SQLAlchemy CRUD operations for storing prediction records and fetching history.
    """
    def __init__(self, db: Session):
        self.db = db
        
    def save_prediction(
        self,
        patient_id: Optional[str],
        image_name: str,
        predictions: dict,
        confidence_score: float
    ) -> PredictionRecord:
        """Saves a diagnostic record to the database."""
        # Convert dictionary to JSON string to save in Text column
        pred_json = json.dumps(predictions)
        
        record = PredictionRecord(
            patient_id=patient_id,
            image_name=image_name,
            predictions=pred_json,
            confidence_score=confidence_score
        )
        
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record
        
    def get_history(self, limit: int = 100, patient_id: Optional[str] = None) -> List[PredictionRecord]:
        """Queries clinician diagnosis history, optionally filtering by patient ID."""
        query = self.db.query(PredictionRecord)
        
        if patient_id:
            query = query.filter(PredictionRecord.patient_id == patient_id)
            
        # Order by latest first
        return query.order_by(PredictionRecord.timestamp.desc()).limit(limit).all()
