from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import json

from backend.schemas import PredictionResponse
from backend.dependencies import get_db
from backend.services.db_service import DBService

router = APIRouter()

@router.get("/history", response_model=List[PredictionResponse], summary="Retrieve prediction history records.")
def get_prediction_history(
    patient_id: Optional[str] = Query(None, description="Filter history by Patient ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max history records to return"),
    db: Session = Depends(get_db)
):
    """
    Clinician logs endpoint:
    Fetches prediction histories from SQLite, ordering from latest to oldest.
    """
    db_service = DBService(db)
    records = db_service.get_history(limit=limit, patient_id=patient_id)
    
    response_list = []
    for r in records:
        # Load predictions JSON string to dictionary
        preds = json.loads(r.predictions)
        
        # Derive detected list (prob >= 0.5)
        detected = [name for name, val in preds.items() if val >= 0.5]
        
        response_list.append(PredictionResponse(
            id=r.id,
            patient_id=r.patient_id,
            image_name=r.image_name,
            predictions=preds,
            detected_diseases=detected,
            confidence_score=r.confidence_score,
            timestamp=r.timestamp
        ))
        
    return response_list
