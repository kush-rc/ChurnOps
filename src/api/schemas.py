"""
Pydantic Schemas for API Request/Response Models
=================================================
"""

from datetime import datetime

from pydantic import BaseModel, Field

# ---- Request Models ----

class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction."""
    gender: str | None = Field(None, example="Female")
    SeniorCitizen: int | None = Field(None, ge=0, le=1, example=0)
    Partner: str | None = Field(None, example="Yes")
    Dependents: str | None = Field(None, example="No")
    tenure: float | None = Field(None, ge=0, example=12)
    PhoneService: str | None = Field(None, example="Yes")
    MultipleLines: str | None = Field(None, example="No")
    InternetService: str | None = Field(None, example="Fiber optic")
    OnlineSecurity: str | None = Field(None, example="No")
    OnlineBackup: str | None = Field(None, example="Yes")
    DeviceProtection: str | None = Field(None, example="No")
    TechSupport: str | None = Field(None, example="No")
    StreamingTV: str | None = Field(None, example="Yes")
    StreamingMovies: str | None = Field(None, example="Yes")
    Contract: str | None = Field(None, example="Month-to-month")
    PaperlessBilling: str | None = Field(None, example="Yes")
    PaymentMethod: str | None = Field(None, example="Electronic check")
    MonthlyCharges: float | None = Field(None, ge=0, example=70.35)
    TotalCharges: float | None = Field(None, ge=0, example=1397.5)

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 1397.5,
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    customers: list[CustomerFeatures]


# ---- Response Models ----

class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    prediction: int = Field(..., description="0 = Not Churned, 1 = Churned")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    label: str = Field(..., description="Human-readable label")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level")
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[PredictionResponse]
    total: int
    churned_count: int
    churn_rate: float


class BatchCustomerResult(BaseModel):
    """Single row result in a batch upload response."""
    row_index: int
    prediction: int
    churn_probability: float
    label: str
    confidence: float


class HistogramBin(BaseModel):
    """A single histogram bin."""
    bin: str
    count: int


class ConfidenceBreakdown(BaseModel):
    """Risk confidence breakdown."""
    low: int = 0
    medium: int = 0
    high: int = 0


class BatchUploadResponse(BaseModel):
    """Response for CSV batch upload predictions."""
    total: int
    churned_count: int
    churn_rate: float
    avg_probability: float
    probability_distribution: list[HistogramBin]
    confidence_breakdown: ConfidenceBreakdown
    top_risk_customers: list[BatchCustomerResult]
    predictions: list[BatchCustomerResult]


class ExplanationResponse(BaseModel):
    """Response for model explanation."""
    prediction: PredictionResponse
    feature_importances: list[dict]
    top_positive: list[dict] = Field(description="Top features pushing toward churn")
    top_negative: list[dict] = Field(description="Top features pushing away from churn")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_version: str | None
    stage: str | None
    metrics: dict
    features_count: int
    trained_at: str | None
