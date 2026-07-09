import io
import os
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.dependencies import get_ml_service
from backend.services.ml_service import MLService

from backend.database import Base, engine
# Create tables for testing environment
Base.metadata.create_all(bind=engine)

# Initialize TestClient
client = TestClient(app)

@pytest.fixture(scope="module")
def mock_image_bytes():
    """Generates a dummy chest X-ray image in memory for API testing."""
    # Create random noise image of shape (224, 224, 3)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return encoded.tobytes()


def test_health_endpoint():
    """Verifies GET /health returns successful status and connectivity checks."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data
    assert "database_connected" in data
    assert data["status"] in ["healthy", "degraded"]


def test_predict_and_history_workflow(mock_image_bytes):
    """Verifies POST /predict and GET /history workflow integration."""
    # 1. Predict diagnostic output for dummy upload
    file_payload = {"file": ("test_xray.jpg", io.BytesIO(mock_image_bytes), "image/jpeg")}
    form_payload = {"patient_id": "test-patient-001"}
    
    response = client.post(
        "/api/v1/predict",
        files=file_payload,
        data=form_payload
    )
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["image_name"] == "test_xray.jpg"
    assert data["patient_id"] == "test-patient-001"
    assert "predictions" in data
    assert "detected_diseases" in data
    assert "confidence_score" in data
    assert "timestamp" in data
    
    # Check predictions probabilities shape (must have 5 competition diseases)
    assert len(data["predictions"]) == 5
    for name, prob in data["predictions"].items():
        assert 0.0 <= prob <= 1.0

    # 2. Query history logs and check if the record was persisted
    history_response = client.get("/api/v1/history?patient_id=test-patient-001")
    assert history_response.status_code == 200
    
    history_data = history_response.json()
    assert len(history_data) > 0
    assert history_data[0]["patient_id"] == "test-patient-001"
    assert history_data[0]["image_name"] == "test_xray.jpg"


def test_gradcam_endpoint(mock_image_bytes):
    """Verifies POST /gradcam streams PNG overlay images back."""
    file_payload = {"file": ("test_xray.jpg", io.BytesIO(mock_image_bytes), "image/jpeg")}
    # Use one of the competition disease labels
    ml_service = get_ml_service()
    target_class = ml_service.class_names[0]
    
    form_payload = {"target_class": target_class}
    
    response = client.post(
        "/api/v1/gradcam",
        files=file_payload,
        data=form_payload
    )
    
    # Check response headers and binary stream
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    
    # Try reading the output image stream to check if it's a valid image
    img_bytes = response.content
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    assert img_decoded is not None
    assert img_decoded.shape == (224, 224, 3)


def test_predict_invalid_file(mock_image_bytes):
    """Verifies predicting on text files raises 400 validation error."""
    file_payload = {"file": ("test_xray.txt", io.BytesIO(b"Dummy text contents"), "text/plain")}
    response = client.post("/api/v1/predict", files=file_payload)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
