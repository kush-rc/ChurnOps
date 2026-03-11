"""
Phase 2: Model Training & Experiment Tracking
==============================================
Trains 6 ML models on the Telco churn dataset, logs everything
to MLflow, compares results, and registers the best model.
"""

import sys
import os
import warnings
import json
import time
from pathlib import Path

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

from src.utils.config import get_config, get_dataset_config, get_path, get_training_config, get_model_config


def load_features(dataset_name: str = "telco"):
    """Load engineered features."""
    config = get_dataset_config(dataset_name)
    features_path = get_path("data_features") / f"{config['name']}_features.parquet"
    df = pd.read_parquet(features_path)

    target = config["target"]
    X = df.drop(columns=[target])
    y = df[target]

    print(f"  Loaded features: X={X.shape}, y={y.shape}")
    print(f"  Target distribution: {dict(y.value_counts())}")
    print(f"  Churn rate: {y.mean():.1%}")
    return X, y


def setup_mlflow():
    """Initialize MLflow tracking."""
    config = get_config()
    mlflow_config = config.get("mlflow", {})

    tracking_uri = f"sqlite:///{project_root / 'mlflow.db'}"
    experiment_name = mlflow_config.get("experiment_name", "churn-prediction")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"  MLflow tracking URI: {tracking_uri}")
    print(f"  MLflow experiment: {experiment_name}")
    return tracking_uri


def get_models():
    """Create all model instances with configs."""
    models = {}

    # 1. Logistic Regression (baseline)
    lr_config = get_model_config("logistic_regression")
    models["logistic_regression"] = {
        "model": LogisticRegression(**lr_config["params"]),
        "config": lr_config,
        "log_fn": mlflow.sklearn.log_model,
    }

    # 2. Random Forest (GPU Accelerated via XGBRFClassifier)
    rf_config = get_model_config("random_forest")
    rf_params = rf_config["params"].copy()
    rf_params.pop("use_label_encoder", None) # Remove if exists
    models["random_forest"] = {
        "model": XGBRFClassifier(**rf_params),
        "config": rf_config,
        "log_fn": mlflow.xgboost.log_model,
    }

    # 3. XGBoost
    xgb_config = get_model_config("xgboost")
    xgb_params = xgb_config["params"].copy()
    xgb_params.pop("use_label_encoder", None)  # deprecated param
    models["xgboost"] = {
        "model": XGBClassifier(**xgb_params),
        "config": xgb_config,
        "log_fn": mlflow.xgboost.log_model,
    }

    # 4. LightGBM
    lgbm_config = get_model_config("lightgbm")
    models["lightgbm"] = {
        "model": LGBMClassifier(**lgbm_config["params"]),
        "config": lgbm_config,
        "log_fn": mlflow.sklearn.log_model,
    }

    # 5. CatBoost
    cb_config = get_model_config("catboost")
    models["catboost"] = {
        "model": CatBoostClassifier(**cb_config["params"]),
        "config": cb_config,
        "log_fn": mlflow.sklearn.log_model,
    }

    # 6. Neural Network (MLP) - Temporarily disabled due to CPU bottleneck on 1M+ rows.
    # Tabular data performs much better on GPU-accelerated XGBoost/LightGBM natively anyway.
    # nn_config = get_model_config("neural_net")
    # nn_params = nn_config["params"].copy()
    # if isinstance(nn_params.get("hidden_layer_sizes"), list):
    #     nn_params["hidden_layer_sizes"] = tuple(nn_params["hidden_layer_sizes"])
    # models["neural_net"] = {
    #     "model": MLPClassifier(**nn_params),
    #     "config": nn_config,
    #     "log_fn": mlflow.sklearn.log_model,
    # }

    return models


def evaluate_model(model, X_test, y_test):
    """Compute comprehensive evaluation metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # Probability-based metrics
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        metrics["avg_precision"] = average_precision_score(y_test, y_proba)
        metrics["log_loss"] = log_loss(y_test, y_proba)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def train_single_model(name, model_info, X_train, y_train, X_test, y_test):
    """Train a single model and log to MLflow."""
    model = model_info["model"]
    config = model_info["config"]
    log_fn = model_info["log_fn"]

    start_time = time.time()

    with mlflow.start_run(run_name=name) as run:
        # Log parameters
        params_to_log = {k: str(v) for k, v in config.get("params", {}).items()}
        params_to_log["model_type"] = name
        params_to_log["imbalance_strategy"] = "smote"
        mlflow.log_params(params_to_log)

        # Train
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        metrics["training_time_seconds"] = round(train_time, 2)

        # Cross-validation score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("\n      ▶️ Running 5-Fold Cross Validation (Sequential to protect GPU VRAM)...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1, verbose=3)
        metrics["cv_roc_auc_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_roc_auc_std"] = round(cv_scores.std(), 4)

        # Log metrics
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        # Log model artifact
        try:
            log_fn(model, "model")
        except Exception:
            mlflow.sklearn.log_model(model, "model")

        # Log confusion matrix as artifact
        cm_data = {
            "confusion_matrix": [[int(metrics["true_negatives"]), int(metrics["false_positives"])],
                                  [int(metrics["false_negatives"]), int(metrics["true_positives"])]],
            "labels": ["Not Churned", "Churned"],
        }
        cm_path = project_root / "reports" / f"confusion_matrix_{name}.json"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cm_path, "w") as f:
            json.dump(cm_data, f, indent=2)
        mlflow.log_artifact(str(cm_path))

        return {
            "model_name": name,
            "run_id": run.info.run_id,
            **metrics,
        }


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="telco", help="Dataset to train on (e.g. telco, ecommerce, saas)")
    args = parser.parse_args()
    
    dataset_name = args.dataset

    print("=" * 65)
    print(f"  🧠 PHASE 2: MODEL TRAINING (Domain: {dataset_name.upper()})")
    print("=" * 65)

    # Step 1: Load features
    print(f"\n📂 [1/5] Loading feature data for {dataset_name}...")
    X, y = load_features(dataset_name)

    # Step 2: Setup MLflow
    print("\n📊 [2/5] Setting up MLflow...")
    setup_mlflow()

    # Step 3: Split data
    print("\n✂️  [3/5] Splitting data...")
    training_config = get_training_config()
    test_size = training_config.get("test_size", 0.2)
    seed = training_config.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")

    # Step 3b: Apply SMOTE
    print("\n⚖️  Applying SMOTE for class imbalance...")
    smote = SMOTE(sampling_strategy=0.8, random_state=seed)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"  Before SMOTE: {X_train.shape[0]:,} samples (churn rate: {y_train.mean():.1%})")
    print(f"  After SMOTE:  {X_train_balanced.shape[0]:,} samples (churn rate: {y_train_balanced.mean():.1%})")

    # Step 4: Train all models
    print("\n🚀 [4/5] Training 6 models...")
    print("-" * 65)

    models = get_models()
    all_results = []

    for i, (name, model_info) in enumerate(models.items(), 1):
        print(f"\n  [{i}/6] Training: {name}...")
        try:
            result = train_single_model(
                name, model_info,
                X_train_balanced, y_train_balanced,
                X_test, y_test
            )
            all_results.append(result)
            print(f"    ✅ AUC: {result.get('roc_auc', 0):.4f} | "
                  f"F1: {result.get('f1', 0):.4f} | "
                  f"Accuracy: {result.get('accuracy', 0):.4f} | "
                  f"Time: {result.get('training_time_seconds', 0):.1f}s")
        except Exception as e:
            print(f"    ❌ Failed: {e}")

    # Step 5: Compare and summarize
    print("\n\n" + "=" * 65)
    print("  📊 MODEL COMPARISON RESULTS")
    print("=" * 65)

    # Sort by AUC
    all_results.sort(key=lambda r: r.get("roc_auc", 0), reverse=True)

    print(f"\n  {'Model':<25} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8}")
    print("  " + "-" * 63)
    for i, r in enumerate(all_results):
        marker = " 🏆" if i == 0 else ""
        print(f"  {r['model_name']:<25} "
              f"{r.get('roc_auc', 0):>8.4f} "
              f"{r.get('f1', 0):>8.4f} "
              f"{r.get('precision', 0):>8.4f} "
              f"{r.get('recall', 0):>8.4f} "
              f"{r.get('accuracy', 0):>8.4f}{marker}")

    # Best model details
    best = all_results[0]
    print(f"\n  🏆 BEST MODEL: {best['model_name']}")
    print(f"     AUC-ROC:    {best.get('roc_auc', 0):.4f}")
    print(f"     F1 Score:   {best.get('f1', 0):.4f}")
    print(f"     Precision:  {best.get('precision', 0):.4f}")
    print(f"     Recall:     {best.get('recall', 0):.4f}")
    print(f"     Accuracy:   {best.get('accuracy', 0):.4f}")
    print(f"     CV AUC:     {best.get('cv_roc_auc_mean', 0):.4f} ± {best.get('cv_roc_auc_std', 0):.4f}")
    print(f"     Run ID:     {best.get('run_id', 'N/A')}")

    # Save results
    results_path = project_root / "reports" / f"model_comparison_{dataset_name}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  📄 Results saved to: {results_path}")

    # Register best model
    print(f"\n📦 [5/5] Registering best model in MLflow...")
    try:
        model_uri = f"runs:/{best['run_id']}/model"
        model_name = f"{dataset_name}-churn-model"
        result = mlflow.register_model(model_uri, model_name)
        print(f"  ✅ Registered: {model_name} v{result.version}")
    except Exception as e:
        print(f"  ⚠️  Registration note: {e}")

    print("\n" + "=" * 65)
    print("  ✅ PHASE 2 COMPLETE!")
    print(f"  View experiments: mlflow server --port 5000")
    print("=" * 65)


if __name__ == "__main__":
    main()
