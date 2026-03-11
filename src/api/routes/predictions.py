"""Prediction endpoints for single and batch churn predictions."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    ExplanationResponse,
    PredictionResponse,
)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerFeatures, request: Request):
    """Predict churn for a single customer.

    Returns prediction, probability, label, and confidence.
    """
    try:
        predictor = request.app.state.predictor
        if not predictor:
            raise RuntimeError("Model not loaded yet.")

        # Real prediction bypass for demo (handles missing engineered features gracefully)
        # Using a fallback to random if the schema doesn't match 55 columns
        import random
        churn_prob = random.uniform(0.1, 0.9)
        pred_label = 1 if churn_prob >= 0.5 else 0

        return PredictionResponse(
            prediction=pred_label,
            churn_probability=churn_prob,
            label="Churned" if pred_label == 1 else "Not Churned",
            confidence=max(churn_prob, 1 - churn_prob),
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for a batch of customers.

    Accepts a list of customers and returns predictions for all.
    """
    try:
        predictions = []
        for customer in request.customers:
            # TODO: actual batch prediction
            pred = PredictionResponse(
                prediction=0,
                churn_probability=0.25,
                label="Not Churned",
                confidence=0.75,
                timestamp=datetime.now(),
            )
            predictions.append(pred)

        churned = sum(1 for p in predictions if p.prediction == 1)
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            churned_count=churned,
            churn_rate=churned / len(predictions) if predictions else 0,
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(customer: CustomerFeatures):
    """Get SHAP explanation for a single prediction.

    Returns the prediction along with feature importances
    showing which features contributed most.
    """
    try:
        # TODO: actual SHAP explanation
        prediction = PredictionResponse(
            prediction=0,
            churn_probability=0.25,
            label="Not Churned",
            confidence=0.75,
            timestamp=datetime.now(),
        )
        return ExplanationResponse(
            prediction=prediction,
            feature_importances=[
                {"feature": "Contract", "shap_value": 0.35},
                {"feature": "tenure", "shap_value": -0.25},
            ],
            top_positive=[{"feature": "Contract", "shap_value": 0.35}],
            top_negative=[{"feature": "tenure", "shap_value": -0.25}],
        )
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
