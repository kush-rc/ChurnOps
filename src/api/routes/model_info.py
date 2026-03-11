"""Model information endpoint."""

from fastapi import APIRouter

from src.api.schemas import ModelInfoResponse

router = APIRouter()


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the currently loaded model."""
    # TODO: pull from MLflow model registry
    return ModelInfoResponse(
        model_name="xgboost",
        model_version="1",
        stage="Production",
        metrics={
            "roc_auc": 0.85,
            "f1": 0.72,
            "accuracy": 0.80,
        },
        features_count=50,
        trained_at=None,
    )
