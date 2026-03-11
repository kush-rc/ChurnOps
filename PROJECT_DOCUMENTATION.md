# 🎓 Complete Project Documentation: Churn Prediction MLOps Pipeline

Welcome to the comprehensive guide for our Customer Churn Prediction project! This document explains exactly **what we built, why we chose specific technologies, how the folders are structured, how the data was handled, and how anyone else can run this project from scratch.**

---

## 🎯 1. Project Overview & Architecture

**Goal:** Build a production-ready Machine Learning system that predicts whether a telecom customer will cancel their subscription ("churn").

### 💼 Who Is This For & Why Should They Run It?

This project is built to be a **turn-key MLOps portfolio piece and template**. Here is exactly who benefits from cloning and running this repository:

1. **Hiring Managers / Recruiters**: They can clone this repo to immediately verify your capability to write production-grade Python, build APIs, and construct end-to-end architectures (not just Jupyter Notebooks). It proves you understand the full lifecycle of an ML model.
2. **Other Junior Data Scientists / Students**: They can use this as a learning template. Most tutorials stop at `model.fit()`. This project shows them how to properly serialize models, track experiments with MLflow, and serve them via FastAPI.
3. **Startup Founders / Small Businesses**: A telecom, SaaS, or subscription-based business can literally take this codebase, swap `telco_churn_massive.csv` with their own company's customer data, tweak the `telco.yaml` config, and immediately deploy a live churn prediction dashboard to save their business money.

### 🏗️ Why this Architecture?
Instead of just training a model in a Jupyter Notebook (which is what beginners do), we built a scalable **MLOps Pipeline**. This means the model isn't just a file on a computer; it's a living service that can take in new data, make predictions via a REST API, and be visually monitored by a dashboard.

- **Data Processing (ETL)**: We use custom Python scripts (`DataPreprocessor`, `FeatureEngineer`) to clean and generate new columns. This is much faster and cleaner than doing it in a messy notebook.
- **Model Training**: We use `scikit-learn`, `xgboost`, and `lightgbm` because *tree-based models dominate tabular data*. Neural Networks require too much data formatting to match XGBoost's raw performance on CSV files.
- **Experiment Tracking**: We use **MLflow**. Why? Because when you train 6 different models with 50 different parameters, you will logically forget which one was the best. MLflow acts as a searchable database logger for our machine learning experiments.
- **Model Serving (API)**: We use **FastAPI** instead of Flask. Why? Because FastAPI is natively asynchronous, handles large batches of prediction requests simultaneously, and automatically generates beautiful API documentation (Swagger UI).
- **Dashboard UI**: We use **Streamlit**. It allows us to build a stunning, interactive web application entirely in Python in a fraction of the time it would take to write a React.js frontend.

---

## 📂 2. Folder Structure Explained

Here is what every folder does and why it exists:

```text
churn-prediction-mlops/
│
├── configs/             # ⚙️ Rules & Settings
│   ├── data/            # Contains telco.yaml. Tells the code EXACTLY where the CSV is and what columns to drop.
│   └── model/           # Contains YAML files for XGBoost, LightGBM, etc., with their default hyperparameter values.
│
├── data/                # 💾 The Actual Data
│   ├── features/        # Where the cleaned & engineered 'parquet' files are saved.
│   └── raw/             # Where the original messy CSV files live (e.g., our massive 5-Lakh dataset).
│
├── mlruns/              # 📈 MLflow Database
│   └── ...              # Automatically generated folder by MLflow. Stores the physical pickled model files and training logs.
│
├── scripts/             # 🛠️ Pipeline Executors (The "Doers")
│   ├── run_training.py  # The main script! Loads data, applies SMOTE, trains 6 models, and logs them to MLflow.
│   ├── run_tuning.py    # Runs Optuna to find the mathematically "best" hyperparameter numbers for our top models.
│   └── test_pipeline.py # A quick sanity-check script to ensure columns drop correctly without full training.
│
├── src/                 # 🧠 Core Intelligence Code (The "Brains")
│   ├── api/             # FastAPI Backend. Contains the code to launch the server and serve the ML model to the web.
│   ├── data/            # Contains classes for data cleaning (`preprocess.py`) and feature creation (`features.py`).
│   ├── models/          # Contains the prediction classes that load the MLflow model and score new customers.
│   └── utils/           # Helper functions for loading configs, timers, and logging.
│
├── streamlit_app/       # 🎨 Frontend Web Application
│   ├── app.py           # The main home page dashboard.
│   └── pages/           # Individual dashboard pages (Predict, Model Comparison, Drift Monitor, SHAP explainability).
│
├── venv/                # 🐍 Python Virtual Environment (Keeps our dependencies isolated from your main computer).
│
├── requirements.txt     # 📄 The master list of all Python libraries needed to run this project.
└── README.md            # The high-level summary for GitHub.
```

---

## 💾 3. Data Flow: How We Processed It

### The Dataset
We originally downloaded three datasets (Telco, Banking, E-commerce) to make the code resilient, but ultimately decided to focus on the **IBM Telco Customer Churn** dataset for our UI.

**The "Big Data" Problem:**
The original dataset only had 7,043 rows. In the real world, models are trained on millions of rows. To simulate this without illegally scraping private company data, we wrote a script to synthesize a **500,000 row (5-Lakh)** dataset (`telco_churn_massive.csv`). We injected statistical noise into the billing columns so the data represents realistic variances.

### The ETL Pipeline
1. **Cleaning**: The `DataPreprocessor` class reads the massive CSV. It converts all total charges to actual numbers, dropping impossible empty spaces, and filling missing values with the median.
2. **Feature Engineering**: The `FeatureEngineer` class looks at raw columns and creates *new* math-based insights:
   - `TotalRevenue` = `tenure` * `MonthlyCharges`.
   - `num_services` = Counting how many add-ons (like TechSupport or StreamingTV) a customer bought.
3. **SMOTE**: Because only ~26% of customers churn, the AI becomes biased toward saying "No Churn" all the time. We applied **SMOTE** (Synthetic Minority Over-sampling Technique) to mathematically synthesize realistic "Churned" customers during training, bringing the data into 50/50 balance.

---

## 🤖 4. Model Training & Pipeline

When you run `scripts/run_training.py`, the following happens:
1. It loops through **6 Algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, and a Neural Network (MLP).
2. It trains them on the 500k balanced rows.
3. It validates them using **Stratified 5-Fold Cross Validation** (splitting the data 5 different ways to ensure the model isn't just heavily memorizing one specific chunk of data).
4. After comparing all 6, the system finds the one with the highest **AUC-ROC Score** (Area Under the Receiver Operating Characteristic Curve — a fancy mathematical grade for how well the model separates the 'churners' from the 'stayers').
5. It **Registers** the winning model into MLflow as the new "Production Model".

---

## 🚀 5. How Someone Else Can Run It

If you upload this to GitHub and another developer downloads it, they just follow these exact steps to launch the entire system:

**1. Install Dependencies**
They create a virtual environment and install the exact libraries we used:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**2. Train the Models**
Because you already included the configuration YAML files, they literally only type ONE command to replicate your entire training pipeline on 500,000 rows.
```bash
python scripts/run_training.py
```
*(This automatically logs the best model to their local MLflow).*

**3. Automatically Tune the Best Models (Optuna)**
If they want to mathematically squeeze every drop of accuracy out of the models, they can run the Auto-Tuning script. This triggers **Bayesian Hyperparameter Optimization** using `optuna`. It systematically loops through 5 different models (Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost), testing hundreds of parameter combinations and saving the winning matrix directly to MLflow!
```bash
python scripts/run_tuning.py
```

**4. Launch the Backend API**
Open a new terminal and start the FastAPI engine. This automatically detects the best model MLflow just trained and loads it into memory natively.
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**5. Launch the Frontend Dashboard**
Open a second terminal and start the Streamlit UI, which connects to the API visually.
```bash
streamlit run streamlit_app/app.py
```

---

## 📈 6. Scaling to 100 Million+ Rows (Enterprise Big Data)
What happens if a massive Telecom company wants to use this pipeline on **100 Million** daily rows? 

This exact codebase architecture is ready to scale, but replacing a single local laptop requires distributed cloud computing. Here is what an architecture shift looks like:

1. **Storage Shift**: Standard `.csv` files break down at 10M+ rows. You would migrate the raw data to a **Parquet Data Lake** (AWS S3) or a **Cloud Data Warehouse** (Snowflake / Google BigQuery).
2. **Compute Shift**: `pandas` cannot hold 100M rows in RAM. You swap out `pandas` inside `src/data/preprocess.py` for **Apache Spark** (PySpark) or **Dask**. This splits the data across a cluster of 50 different machines to calculate the feature engineering simultaneously.
3. **Training Shift**: Standard Scikit-Learn maxes out on CPU cores. You would switch the XGBoost and LightGBM models over to **Ray Train** or **Databricks**, which natively command hundreds of GPUs across the cloud to train on billions of rows in minutes. 
4. **API Serving Shift**: A single FastAPI Uvicorn instance maxes out at processing maybe a few thousand predictions per second. To hit massive scale, you put FastAPI inside a **Docker Image** and deploy it to **Kubernetes** behind an AWS Application Load Balancer. As traffic goes up, Kubernetes automatically spawns 20 identical copies of the FastAPI server to handle the load concurrently.

This repository serves as the foundational microservice code that gets containerized and shipped to those scaled environments!

---

## 🌟 Summary of "Why"
* Why **Streamlit**? Because a native frontend developer requires a second repository structure, NodeJS, and specialized deployment. Streamlit allows Data Scientists to build fully working React-level dashboards using only Python.
* Why **Altair over Streamlit Bar Charts**? Because Streamlit deprecated `horizontal=True` in recent versions, causing type-errors. Rewriting the charts directly in Altair guaranteed stability across all Python versions.
## 🌍 7. Multi-Domain Support (Cross-Industry Churn)

While the default UI focuses on the **Telecom** dataset, this repository is engineered to support *Omni-Industry Data Generation and Prediction*. The pipeline includes a synthetic Big Data generator (`scripts/generate_8_domains.py`) capable of generating millions of rows for 8 specialized industries:

1. 📱 **Telecom** (Airtel, Jio, AT&T): Predicts network switching based on call drops, data limits, and plan types.
2. 🏦 **Banking & Finance** (Chase, HDFC): Predicts account closure based on dropping balances, credit scores, and transaction infrequency. 
3. 🛒 **E-Commerce / Retail** (Amazon, Flipkart): Predicts 90+ day inactivity based on order frequency, return rates, and discount usage.
4. 🎬 **OTT / Streaming** (Netflix, Spotify): Predicts subscription cancellation using watch hours, payment failures, and login frequency.
5. 🏥 **Healthcare / Insurance** (UnitedHealth): Predicts policy lapses based on claims history, premium payment delays, and health scores.
6. 🎮 **Gaming** (PUBG, Clash of Clans): Predicts player uninstalls via win rates, daily active time, and in-app purchase drops.
7. 🏋️ **SaaS / Software** (Salesforce, Slack): Predicts non-renewals tracking feature usage, support tickets, and team sizes.
8. 🏨 **Hospitality / Travel** (Uber, OYO): Predicts platform switching via loyalty points, ratings, and booking infrequency.

Each domain has its own dedicated YAML configuration file (`configs/data/*.yaml`) and a rapidly-trained XGBoost model registered in MLflow (`runs:/.../{domain}-churn-model`). This demonstrates that the MLOps architecture is completely domain-agnostic and infinitely scalable to any subscription-based business model!

### 🌐 The Multi-Domain AI Control Center (Streamlit UI)
The provided Streamlit Dashboard (`streamlit_app/app.py`) has been completely overhauled to serve as an enterprise-scale **Multi-Domain AI Control Center**. 

**Features Included:**
1. **Dynamic Hot-Swapping**: A beautiful dropdown menu on the home page allows users to instantly switch between any of the 8 global industries (Telecom, Banking, E-commerce, Gaming, etc.).
2. **Real-Time MLflow Connection**: When an industry is selected, the stream automatically reaches into the MLflow registry and securely latches onto the mathematical champion model for that exact domain (e.g., hooking up `gaming-churn-model` instead of `telco-churn-model`).
3. **GPU-Accelerated Engineering**: All 8 Million rows were specifically processed, trained, and optimized using **NVIDIA RTX 4060 CUDA Acceleration**, bypassing standard CPU bottlenecks to deliver massive prediction arrays in seconds.
