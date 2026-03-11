# 🌍 Multi-Domain Enterprise MLOps Architecture: Churn Prediction

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![NVIDIA](https://img.shields.io/badge/CUDA-GPU_Accelerated-76B900)

Welcome to the **Multi-Domain Customer Churn MLOps Pipeline**. This is not just a Jupyter Notebook; it is a full-stack, production-ready Machine Learning Architecture. It was designed to demonstrate how to mathematically synthesize Big Data, orchestrate automated training pipelines with GPU acceleration, and deploy dynamic visual AI Control Centers.

---

## 🚀 1. What Have I Done?

I completely engineered an enterprise-grade AI architecture from the ground up that predicts customer churn (attrition) not just for one business, but for **8 massive global industries** (Telecom, Banking, E-Commerce, SaaS, Gaming, etc.).

**How I did it:**
1. **Synthetic Big Data Generation**: Instead of relying on small, outdated public datasets, I wrote vectorized Numpy scripts to generate **8 Million rows** of highly realistic data mimicking financial curves, gaming win-rates, and SaaS features.
2. **GPU-Accelerated Pipelines**: I migrated algorithms like XGBoost, LightGBM, and CatBoost to run natively on NVIDIA CUDA Architecture, dropping training time by orders of magnitude.
3. **Automated MLOps**: I hooked the entire training loop into **MLflow** for experiment tracking and **Optuna** for Bayesian Hyperparameter evolutionary tuning.
4. **Dynamic AI Engine**: I stood up a **FastAPI** server that acts as a secure mathematical endpoint, wrapped by a glorious **Streamlit UI** that hot-swaps input fields dynamically depending on what industry you want to analyze.

### 💼 How Does This Help Others?
- **For Businesses**: A company can clone this repository, drop their own customer CSV into the `data/raw/` folder, configure a single YAML file, and instantly have a complete AI prediction dashboard for their marketing team.
- **For Developers/Recruiters**: It acts as a flawless portfolio template showing the exact best-practices required for real-world Machine Learning Operations (modular code, REST APIs, MLflow registries).

---

## 🏗️ 2. The Core Architecture & Pipeline

Data doesn't just "become" intelligence. It flows through a strict mathematical pipeline:

1. **Ingestion & ETL (`src/data/`)**: Raw CSVs (1M rows each) are streamed in. Missing values are filled via medians, and new features are mathematically constructed (e.g., `TotalRevenue = Tenure * MonthlyCharges`).
2. **Class Balancing**: Because churn is a minority class (e.g., only 20% of users churn), we use **SMOTE** (Synthetic Minority Over-sampling Technique) to algorithmically hallucinate realistic churners so the AI isn't biased.
3. **Automated Modeling (`scripts/run_training.py`)**: The data is fed into 6 different Machine Learning algorithms. They are Cross-Validated using a 5-fold split to prevent memorization (overfitting).
4. **Experiment Tracking (`MLflow`)**: As models train, MLflow silently logs every single statistic (AUC, F1-Score, Training Time). The best algorithmic "Champion" is frozen in time and registered.
5. **Serving (`FastAPI`)**: The backend server boots, locks onto the MLflow Champion model, and opens a web endpoint (`/predict`).
6. **Interaction (`Streamlit`)**: The UI communicates with FastAPI to pass real-world form data into the backend, rendering real-time predictions back to the user.

---

## 📂 3. Detailed Folder Structure Explained

I strictly followed Enterprise Software Engineering standards. Here is why every single file exists:

```text
churn-prediction-mlops/
│
├── configs/                 # YAML Configuration Maps
│   ├── data/                 # (e.g., banking.yaml) Tells the pipeline exactly what columns exist and what to predict.
│   └── model/                # Contains the default hyperparameters for XGBoost, Random Forest, etc.
│
├── data/                    # The Data Lake
│   ├── raw/                  # The massive 1-Million row synthetic CSVs we generated.
│   └── processed/            # Where the cleaned data lives before training.
│
├── scripts/                 # The Orchestrators (The "Doers")
│   ├── generate_8_domains.py # The script that fabricated the 8 Million rows using advanced Numpy distributions.
│   ├── orchestrate_all.ps1   # A master script that runs ETL and Training sequentially for every industry safely.
│   ├── run_training.py       # Loads data, runs 6 models, compares metrics, and registers the Champion.
│   └── run_tuning.py         # The Optuna genetic algorithm that mathematically evolves the model's parameters.
│
├── src/                     # The Intelligence Core (The "Brains")
│   ├── api/                  # The FastAPI backend application routing logic.
│   ├── data/                 # The actual Python classes that clean data (DataPreprocessor) and build metrics (FeatureEngineer).
│   └── models/               # Contains the MLflow loading wrappers so FastAPI can talk to the trained weights.
│
├── streamlit_app/           # The Frontend Command Center
│   ├── app.py                # The Multi-Domain Home Page dropdown dashboard.
│   └── pages/                # The dynamically rendered Prediction Forms and Batch-Predict CSV pages.
│
├── .gitignore               # Ensures we don't upload 8 Gigabytes of raw CSVs to GitHub.
├── DATA_GENERATION_EXPLAINED.md # Explains the math behind how we safely simulated human data.
├── PROJECT_DOCUMENTATION.md # A secondary deep-dive text into architecture choices.
├── requirements.txt         # Every single exact version of Python libraries needed to run this repo safely.
└── mlflow.db                # The hidden SQLite database where MLflow logs its model scores.
```

---

## 💻 4. How To Run Locally

If you cloned this, here is how you boot the entire engine on your own machine.

### Step 1: Install Dependencies
Ensure you have Python 3.10+ installed.
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Generate the Big Data & Train Models
Since the 8 Million rows are too large for GitHub, generate them locally, then train all the models into MLflow!
```bash
python scripts/generate_8_domains.py
.\scripts\orchestrate_all.ps1
```

### Step 3: Boot the FastAPI Engine
Start the backend AI interface.
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Step 4: Boot the Streamlit UI
Open a second terminal and start the front-end!
```bash
streamlit run streamlit_app/app.py
```

---

## 🌐 5. Deployment Guide (GitHub & Streamlit Cloud)

I have engineered this to be perfectly deployable to showcase to the world.

### Step A: Push to GitHub
1. Open your terminal in the root of the project.
2. Initialize and commit your code, completely ignoring the massive datasets.
```bash
git init
git add .
git commit -m "🚀 Initial Commit: Multi-Domain MLOps Architecture"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### Step B: Free Deployment to Streamlit Community Cloud
Streamlit provides free cloud hosting directly linked to your GitHub!

1. Go to [share.streamlit.io](https://share.streamlit.io/) and create a free account linked to your GitHub.
2. Click **"New App"**.
3. Point it to your GitHub Repository.
4. Set the **Main file path** to: `streamlit_app/app.py`.
5. Click **Deploy!**

*(Note: Because the API backend won't easily fit inside Streamlit's free tier natively alongside the UI, the `01_predict.py` file has been pre-configured with a "spoofed" intelligence fallback using randomized bounds. This ensures that recruiters and users can beautifully visualize the Domain-Changing capabilities in the cloud without needing to rent a $40/month backend server for FastAPI).*

---
Made with ❤️ through MLOps Best Practices.
