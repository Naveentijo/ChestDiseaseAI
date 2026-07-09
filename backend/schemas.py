from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import datetime

class PredictionResponse(BaseModel):
    id: Optional[int] = None
    patient_id: Optional[str] = None
    image_name: str
    predictions: Dict[str, float] = Field(..., description="Dict mapping disease labels to probabilities")
    detected_diseases: List[str] = Field(..., description="List of diseases matching threshold criteria")
    confidence_score: float = Field(..., description="Highest confidence level detected")
    timestamp: datetime.datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    database_connected: bool
