from typing import Generator
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.services.ml_service import MLService

# Singleton instance of MLService, cached in memory
_ml_service_instance = MLService()

def get_db() -> Generator[Session, None, None]:
    """Dependency injection helper to yield database session local instances."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_ml_service() -> MLService:
    """Dependency injection helper to yield in-memory cached MLService singleton."""
    return _ml_service_instance
