from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.schemas import HealthResponse
from backend.dependencies import get_db, get_ml_service
from backend.services.ml_service import MLService

router = APIRouter()

@router.get("/health", response_model=HealthResponse, summary="Retrieve backend and model health statuses.")
def check_health(
    db: Session = Depends(get_db),
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Diagnostic endpoint verifying:
    1. Database connectivity.
    2. Cached model memory loading.
    3. Target inference execution device.
    """
    model_loaded = ml_service.model is not None
    device_name = ml_service.device
    
    # Check database connectivity
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        pass
        
    status = "healthy" if (model_loaded and db_connected) else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        device=device_name,
        database_connected=db_connected
    )
