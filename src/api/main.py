"""
FastAPI Application
===================
Main entry point for the churn prediction API.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import make_asgi_app

from src.api.routes import health, model_info, monitoring, predictions
from src.utils.config import get_config
from src.models.predict import ChurnPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger.info("🚀 Starting Churn Prediction API...")
    try:
        app.state.predictor = ChurnPredictor(model_uri="models:/churn-prediction-model/latest")
        logger.info("✅ Model loaded successfully from MLflow.")
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        app.state.predictor = None
        
    yield
    # Shutdown: cleanup
    logger.info("🛑 Shutting down API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()

    app = FastAPI(
        title="Churn Prediction API",
        description=(
            "Production-grade API for predicting customer churn. "
            "Supports real-time and batch predictions with model explainability."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    cors_origins = config.get("api", {}).get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(model_info.router, prefix="/api/v1", tags=["Model Info"])
    app.include_router(monitoring.router, prefix="/api/v1", tags=["Monitoring"])

    return app


# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    config = get_config()
    api_config = config.get("api", {})
    uvicorn.run(
        "src.api.main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("debug", False),
    )
