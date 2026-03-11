# ============================================================
# MLOps Pipeline for Customer Churn Prediction - Makefile
# ============================================================
# Usage: make <target>
# Run `make help` to see all available targets.
# ============================================================

.PHONY: help setup install data train serve test lint format clean mlflow up down demo

PYTHON = venv/Scripts/python
PIP = venv/Scripts/pip
PREFECT = venv/Scripts/prefect
UVICORN = venv/Scripts/uvicorn
STREAMLIT = venv/Scripts/streamlit
MLFLOW = venv/Scripts/mlflow
PYTEST = venv/Scripts/pytest

# Colors for terminal output
BLUE = \033[34m
GREEN = \033[32m
YELLOW = \033[33m
RED = \033[31m
RESET = \033[0m

## ---- Setup & Install ----

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

setup: ## Create virtual environment and install all dependencies
	python -m venv venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)Setup complete! Activate venv: venv\\Scripts\\activate$(RESET)"

install: ## Install dependencies only (venv must exist)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

## ---- Data Pipeline ----

download-data: ## Download datasets from Kaggle
	$(PYTHON) scripts/download_data.py

data: ## Run full data pipeline (ingest → validate → clean → features)
	$(PYTHON) -m pipelines.data_pipeline

## ---- Model Training ----

train: ## Train all models and log to MLflow
	$(PYTHON) -m pipelines.training_pipeline

tune: ## Run hyperparameter tuning with Optuna
	$(PYTHON) -m src.models.hyperparameter_tuning

## ---- Model Serving ----

serve: ## Start FastAPI server locally (port 8000)
	$(UVICORN) src.api.main:app --host 0.0.0.0 --port 8000 --reload

## ---- MLflow ----

mlflow: ## Start MLflow tracking server (port 5000)
	$(MLFLOW) server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000

## ---- Monitoring ----

prefect: ## Start Prefect server (port 4200)
	$(PREFECT) server start

## ---- Docker ----

up: ## Start full stack with Docker Compose
	docker-compose -f deployment/docker-compose.yml up -d --build

down: ## Stop all Docker Compose services
	docker-compose -f deployment/docker-compose.yml down

## ---- Demo ----

demo: ## Launch Streamlit demo dashboard (port 8501)
	$(STREAMLIT) run streamlit_app/app.py --server.port 8501

## ---- Testing ----

test: ## Run all tests with coverage
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v -m integration

test-data: ## Run data quality tests
	$(PYTEST) tests/data/ -v -m data

## ---- Code Quality ----

lint: ## Run ruff linter
	$(PYTHON) -m ruff check src/ tests/ pipelines/

format: ## Format code with ruff
	$(PYTHON) -m ruff format src/ tests/ pipelines/
	$(PYTHON) -m ruff check --fix src/ tests/ pipelines/

typecheck: ## Run mypy type checker
	$(PYTHON) -m mypy src/

quality: lint typecheck test ## Run all quality checks (lint + typecheck + test)

## ---- Cleanup ----

clean: ## Remove build artifacts and cache files
	@echo "Cleaning up..."
	@if exist __pycache__ rd /s /q __pycache__
	@if exist .pytest_cache rd /s /q .pytest_cache
	@if exist .mypy_cache rd /s /q .mypy_cache
	@if exist .ruff_cache rd /s /q .ruff_cache
	@if exist htmlcov rd /s /q htmlcov
	@if exist reports rd /s /q reports
	@if exist *.egg-info rd /s /q *.egg-info
	@echo "$(GREEN)Cleaned!$(RESET)"

clean-data: ## Remove all processed data (keeps raw)
	@echo "Cleaning processed data..."
	@if exist data\processed rd /s /q data\processed
	@if exist data\features rd /s /q data\features
	@mkdir data\processed
	@mkdir data\features
	@echo "$(GREEN)Data cleaned!$(RESET)"
