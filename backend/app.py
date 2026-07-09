import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.database import Base, engine
from backend.dependencies import get_ml_service
from backend.routers import health, predict, history
from ml.chest_ai.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager handling database setup and ML weight loading at startup."""
    # 1. Startup Logic
    logger.info("Starting up FastAPI application...")
    
    # Create SQLite database tables if they do not exist
    logger.info("Initializing SQLite database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Load PyTorch model weights to RAM
    try:
        ml_service = get_ml_service()
        ml_service.load_model()
    except Exception as e:
        logger.error(f"Failed to load PyTorch model weights at startup: {e}. API will run in degraded mode.")
        
    yield
    
    # 2. Shutdown Logic
    logger.info("Shutting down FastAPI application...")
    # Clear hooks to prevent memory leaks
    try:
        ml_service = get_ml_service()
        if ml_service.gradcam:
            ml_service.gradcam.remove_hooks()
    except Exception:
        pass

# Initialize FastAPI App
app = FastAPI(
    title="ChestDiseaseAI REST APIs",
    description="Backend API platform for Chest X-ray Disease Detection & Explainability (Grad-CAM).",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
# Allows React/Next.js frontend (e.g. running on localhost:3000) to communicate with API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(history.router, prefix="/api/v1", tags=["History"])

@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirects base path to Swagger UI docs."""
    return RedirectResponse(url="/docs")
