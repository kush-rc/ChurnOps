"""Health check and readiness endpoints."""

from datetime import datetime

from fastapi import APIRouter

from src.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        model_loaded=True,  # TODO: check actual model status
        version="1.0.0",
        timestamp=datetime.now(),
    )


@router.get("/ready")
async def readiness_check():
    """Check if the API is ready to serve predictions."""
    # TODO: check model, database, etc.
    return {"ready": True, "timestamp": datetime.now().isoformat()}
