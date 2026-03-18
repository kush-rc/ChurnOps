# ChurnOps — MLOps Customer Churn Prediction Platform

> A production-grade MLOps pipeline that predicts customer churn across **8 industry verticals** using ensemble machine learning, real-time inference via FastAPI, and a React analytics dashboard.

---

## Why This Project Exists

Customer churn (when a user stops using a product) is one of the most expensive problems in business. This project demonstrates how to build a **complete, end-to-end MLOps system** that can:

1. **Generate** realistic churn datasets for 8 industries (Telecom, Banking, E-commerce, SaaS, Healthcare, Gaming, OTT, Hospitality)
2. **Train & tune** 6 ML algorithms with Optuna hyperparameter optimization, logged via MLflow
3. **Serve** real-time and batch predictions through a FastAPI REST API
4. **Explain** predictions using SHAP (SHapley Additive exPlanations) for model transparency
5. **Visualize** everything in a premium React dashboard with dark mode

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML/Data** | XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression, scikit-learn |
| **Experiment Tracking** | MLflow (model registry, metrics, artifacts) |
| **Hyperparameter Tuning** | Optuna (Bayesian optimization) |
| **API** | FastAPI + Uvicorn |
| **Explainability** | SHAP (PermutationExplainer with caching) |
| **Frontend** | React 18 + Vite + Recharts + Framer Motion + Lucide Icons |
| **Containerization** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions (lint → test → build → deploy) |
| **Code Quality** | Ruff (linter/formatter) + MyPy (type checker) + pre-commit hooks |
| **Data Validation** | Great Expectations patterns via custom validators |
| **Monitoring** | Prometheus metrics + Evidently AI drift detection |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### 1. Clone & Setup Backend
```bash
git clone https://github.com/kushchhunchha/churn-prediction-mlops.git
cd churn-prediction-mlops

python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS/Linux

pip install -r requirements.txt
pip install -e .
```

### 2. Generate Data & Train Models
```bash
# Generate synthetic datasets for all 8 domains
python scripts/generate_8_domains.py

# Train models (logs to MLflow automatically)
python scripts/run_training.py

# Optional: hyperparameter tuning
python scripts/run_tuning.py
```

### 3. Start Backend API
```bash
$env:PYTHONPATH="."; uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

### 5. Docker (Alternative)
```bash
docker-compose up --build
# API → http://localhost:8000
# Frontend → http://localhost:3000
```

---

## Project Structure

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a detailed file-by-file breakdown.

```
churn-prediction-mlops/
├── configs/              # YAML configs for data, models, training, monitoring
├── data/                 # Raw CSVs, processed parquets, feature stores
├── docs/                 # Architecture, API reference, Docker guide
├── frontend/             # React + Vite dashboard (5 pages)
├── models/               # Saved preprocessors (joblib files)
├── reports/              # Model comparison JSONs, confusion matrices
├── scripts/              # Data generation, training, tuning, testing
├── src/                  # Core Python source code
│   ├── api/              # FastAPI app, routes, schemas
│   ├── data/             # Ingest, preprocess, validate, feature engineering
│   ├── models/           # Train, predict, evaluate, explain, tune, registry
│   ├── monitoring/       # Drift detection, metrics, alerts
│   └── utils/            # Config loader, helpers, logging
├── tests/                # Unit + integration tests
├── Dockerfile            # Backend container
├── docker-compose.yml    # Full stack orchestration
├── Makefile              # CLI shortcuts (make serve, make test, etc.)
├── pyproject.toml        # Project metadata + tool config
└── requirements.txt      # Python dependencies
```

---

## Documentation

| Document | Description |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system architecture, every file explained |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | All REST endpoints with request/response examples |
| [DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md) | Container setup, why Docker, environment config |
| [DATA_GENERATION_EXPLAINED.md](docs/DATA_GENERATION_EXPLAINED.md) | How synthetic data is created for 8 industries |
| [model_card.md](docs/model_card.md) | Model performance, training details, limitations |

---

## Features

### Dashboard
- Live training population stats, model ensemble info, inference health
- Infrastructure maturity indicators and intelligence feed

### Prediction
- Single-customer prediction with domain-specific input forms
- Real-time churn probability with confidence scores

### Batch Analysis
- CSV upload with drag-and-drop
- Probability distribution histogram, risk segmentation donut chart
- Top-risk customer table, full paginated results with CSV export

### Model Comparison
- Side-by-side comparison of all trained algorithms
- Accuracy, Precision, Recall, F1, AUC metrics

### Explainability
- Real SHAP values computed from the trained XGBoost model
- Top positive/negative feature contributions per prediction

### Dark Mode
- Toggle between light and dark themes (persistent across sessions)

---

## Make Commands

```bash
make setup          # Create venv + install everything
make serve          # Start FastAPI server (port 8000)
make mlflow         # Start MLflow UI (port 5000)
make test           # Run all tests with coverage
make lint           # Run Ruff linter
make clean          # Remove caches and build artifacts
make up             # Start Docker Compose stack
make down           # Stop Docker Compose stack
```

---

## License

MIT License — see [pyproject.toml](pyproject.toml) for details.

Built by **Kush Chhunchha**.
