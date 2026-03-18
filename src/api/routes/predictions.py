"""Prediction endpoints for single and batch churn predictions."""

import io
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from loguru import logger

from src.api.schemas import (
    BatchCustomerResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchUploadResponse,
    ConfidenceBreakdown,
    CustomerFeatures,
    ExplanationResponse,
    HistogramBin,
    PredictionResponse,
)
from src.models.predict import ChurnPredictor

router = APIRouter()

# Global predictor cache (domain -> ChurnPredictor)
predictors: dict[str, ChurnPredictor] = {}

def get_predictor(domain: str) -> ChurnPredictor:
    """Get or initialize the predictor for a domain."""
    if domain not in predictors:
        logger.info(f"Initializing ChurnPredictor for domain: {domain}")
        predictors[domain] = ChurnPredictor(domain=domain)
    return predictors[domain]


def preload_all_predictors() -> None:
    """Pre-load only the active domain predictor to save memory on Render (512MB limit)."""
    import gc
    from src.utils.config import get_config
    
    try:
        config = get_config()
        active_domain = config.get("active_dataset", "telco")
        
        logger.info(f"🔥 Pre-loading active predictor for domain: {active_domain}...")
        get_predictor(active_domain)
        
        # Reclaim memory from temporary loading objects
        gc.collect()
        logger.info("✅ Active predictor pre-loaded. Other domains will lazy-load on demand to stay under 512MB.")
    except Exception as e:
        logger.error(f"Failed to pre-load active domain: {e}")


@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@router.post("/predict", response_model=PredictionResponse)
async def predict_single(
    customer: CustomerFeatures,
    request: Request,
    domain: str = Query("telco", description="Industry domain for prediction"),
):
    """Predict churn for a single customer.

    Returns prediction, probability, label, and confidence.
    Uses the trained MLflow model and its associated preprocessing pipeline.
    """
    try:
        features = customer.model_dump(exclude_none=True)
        predictor = get_predictor(domain)
        result = predictor.predict_single(features)

        return PredictionResponse(
            prediction=result["prediction"],
            churn_probability=result["churn_probability"],
            label=result["label"],
            confidence=result["confidence"],
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    domain: str = Query("telco", description="Industry domain for prediction"),
):
    """Predict churn for a batch of customers.

    Accepts a list of customers and returns predictions for all.
    """
    try:
        predictor = get_predictor(domain)

        # We could optimize this by batching the dataframe, but for now
        # we'll keep the loop for simplicity given the schema validation.
        predictions = []
        for customer in request.customers:
            features = customer.model_dump(exclude_none=True)
            result = predictor.predict_single(features)
            pred = PredictionResponse(
                prediction=result["prediction"],
                churn_probability=result["churn_probability"],
                label=result["label"],
                confidence=result["confidence"],
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
        logger.exception(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    customer: CustomerFeatures,
    domain: str = Query("telco", description="Industry domain for prediction"),
):
    """Get feature importance explanation for a single prediction.

    Returns the prediction along with SHAP-like feature importances.
    Currently returns dummy importances until SHAP is fully integrated
    with the new MLflow pipeline.
    """
    try:
        features = customer.model_dump(exclude_none=True)
        predictor = get_predictor(domain)
        result = predictor.predict_single(features)

        # Compute exact SHAP values using the trained MLflow model
        importances = predictor.explain_single(features)

        prediction = PredictionResponse(
            prediction=result["prediction"],
            churn_probability=result["churn_probability"],
            label=result["label"],
            confidence=result["confidence"],
            timestamp=datetime.now(),
        )

        # Split into positive and negative contributions
        top_positive = [fi for fi in importances if float(fi["shap_value"]) > 0][:5]
        top_negative = [fi for fi in importances if float(fi["shap_value"]) < 0][:5]

        return ExplanationResponse(
            prediction=prediction,
            feature_importances=importances,
            top_positive=top_positive,
            top_negative=top_negative,
        )
    except Exception as e:
        logger.exception(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/predict/upload", response_model=BatchUploadResponse)
async def predict_upload(
    file: UploadFile = File(...),  # noqa: B008
    domain: str = Query("telco", description="Industry domain for prediction"),
):
    """Upload a CSV file and run batch churn predictions.

    Returns per-row predictions plus aggregate statistics:
    probability distribution, risk segmentation, and top-risk customers.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        logger.info(f"Batch upload: {len(df)} rows from {file.filename}")

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        predictor = get_predictor(domain)
        expected_cols = predictor._extract_model_feature_names() or []
        
        # 1. Map columns case-insensitively and check coverage
        df_cols = [c.strip().lower() for c in df.columns]
        expected_lower = [c.lower() for c in expected_cols]
        
        found_cols = [c for c in expected_lower if c in df_cols]
        missing_cols = [c for c in expected_cols if c.lower() not in df_cols]
        
        coverage = len(found_cols) / len(expected_cols) if expected_cols else 1.0
        
        # 🧪 Safety Threshold: If < 50% features match, it's likely the wrong file
        if coverage < 0.5:
            error_msg = (
                f"Invalid CSV format for domain: {domain}. "
                f"Found only {len(found_cols)} out of {len(expected_cols)} required features ({coverage:.1%}). "
                f"Missing critical fields: {', '.join(missing_cols[:5])}..."
            )
            logger.warning(f"Batch upload rejected: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

        # 2. Rename columns to match model exactly (case-insensitive fix)
        mapping = {}
        for col in df.columns:
            for exp in expected_cols:
                if col.strip().lower() == exp.lower():
                    mapping[col] = exp
                    break
        
        df = df.rename(columns=mapping)

        # Run predictions for each row
        all_results = []
        for idx, row in df.iterrows():
            features = row.dropna().to_dict()
            try:
                result = predictor.predict_single(features)
                all_results.append(BatchCustomerResult(
                    row_index=int(idx) + 1,
                    prediction=result["prediction"],
                    churn_probability=result["churn_probability"],
                    label=result["label"],
                    confidence=result["confidence"],
                ))
            except Exception as row_err:
                logger.warning(f"Row {idx} failed: {row_err}")
                all_results.append(BatchCustomerResult(
                    row_index=int(idx) + 1,
                    prediction=0,
                    churn_probability=0.0,
                    label="Error",
                    confidence=0.0,
                ))

        # Aggregate statistics
        total = len(all_results)
        churned = sum(1 for r in all_results if r.prediction == 1)
        probs = [r.churn_probability for r in all_results]
        avg_prob = sum(probs) / total if total else 0.0

        # Probability distribution histogram (10 bins)
        bins = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
        bin_counts = [0] * 10
        for p in probs:
            idx = min(int(p * 10), 9)
            bin_counts[idx] += 1
        histogram = [HistogramBin(bin=bins[i], count=bin_counts[i]) for i in range(10)]

        # Risk segmentation
        low = sum(1 for p in probs if p <= 0.4)
        medium = sum(1 for p in probs if 0.4 < p <= 0.7)
        high = sum(1 for p in probs if p > 0.7)

        # Top 10 highest risk customers
        sorted_results = sorted(all_results, key=lambda r: r.churn_probability, reverse=True)
        top_risk = sorted_results[:10]

        return BatchUploadResponse(
            total=total,
            churned_count=churned,
            churn_rate=churned / total if total else 0.0,
            avg_probability=round(avg_prob, 4),
            probability_distribution=histogram,
            confidence_breakdown=ConfidenceBreakdown(low=low, medium=medium, high=high),
            top_risk_customers=top_risk,
            predictions=all_results,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Batch upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
