"""Monitoring endpoints for drift and performance data."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/monitoring/drift")
async def get_drift_status():
    """Get current drift detection status."""
    # TODO: pull from drift detector
    return {
        "drift_detected": False,
        "last_check": None,
        "psi_score": 0.0,
        "drifted_features": [],
    }


@router.get("/monitoring/performance")
async def get_performance():
    """Get current model performance metrics."""
    # TODO: pull from performance tracker
    return {
        "status": "ok",
        "accuracy": 0.80,
        "f1": 0.72,
        "auc": 0.85,
        "total_predictions": 0,
    }


@router.get("/monitoring/alerts")
async def get_alerts():
    """Get recent monitoring alerts."""
    # TODO: pull from alert manager
    return {"alerts": [], "total": 0}
