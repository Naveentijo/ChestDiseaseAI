import io
import json
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional

from backend.schemas import PredictionResponse
from backend.dependencies import get_db, get_ml_service
from backend.services.ml_service import MLService
from backend.services.db_service import DBService

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse, summary="Predict disease on uploaded chest X-ray image.")
async def predict_chest_xray(
    file: UploadFile = File(..., description="Chest X-ray image file (PNG/JPG)"),
    patient_id: Optional[str] = Form(None, description="Optional alphanumeric Patient ID for history tracking"),
    db: Session = Depends(get_db),
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Diagnostic classification endpoint:
    1. Reads and decodes binary image upload.
    2. Runs model inference.
    3. Persists diagnostic summary to SQLite history.
    4. Returns class probabilities and detected conditions.
    """
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PNG, JPG, or JPEG image.")
        
    try:
        contents = await file.read()
        
        # 1. Run predictions
        predictions, max_confidence, detected_diseases = ml_service.predict(contents)
        
        # 2. Persist to database
        db_service = DBService(db)
        record = db_service.save_prediction(
            patient_id=patient_id,
            image_name=file.filename,
            predictions=predictions,
            confidence_score=max_confidence
        )
        
        return PredictionResponse(
            id=record.id,
            patient_id=record.patient_id,
            image_name=record.image_name,
            predictions=predictions,
            detected_diseases=detected_diseases,
            confidence_score=record.confidence_score,
            timestamp=record.timestamp
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/gradcam", summary="Generate Grad-CAM explainability overlay on uploaded chest X-ray.")
async def generate_gradcam_heatmap(
    file: UploadFile = File(..., description="Chest X-ray image file"),
    target_class: str = Form(..., description="Target class disease name to generate overlay for"),
    ml_service: MLService = Depends(get_ml_service)
):
    """
    Explainability endpoint:
    1. Runs model forward/backward hook sequences for target class.
    2. Overlays heatmap on image.
    3. Streams overlaid PNG image directly to clinician viewer.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
        
    try:
        contents = await file.read()
        
        # Generate blended PNG image bytes
        blended_bytes = ml_service.generate_gradcam_overlay(contents, target_class)
        
        return StreamingResponse(
            io.BytesIO(blended_bytes),
            media_type="image/png"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")

# Make sure 'os' is imported
import os
