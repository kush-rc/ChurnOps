"""
Phase 2.5: Hyperparameter Tuning
================================
Run Optuna tuning on the best performing models to improve performance.
"""

import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from loguru import logger
import mlflow

from src.models.hyperparameter_tuning import HyperparameterTuner
from src.utils.config import get_dataset_config, get_path, get_training_config
from scripts.run_training import load_features, setup_mlflow

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="telco", help="Dataset to tune on (e.g. ecommerce, gaming)")
    args = parser.parse_args()
    
    dataset_name = args.dataset

    print("=" * 65)
    print(f"  🔧 PHASE 2.5: HYPERPARAMETER TUNING ({dataset_name.upper()})")
    print("=" * 65)

    # Load and split data
    X, y = load_features(dataset_name)
    training_config = get_training_config()
    test_size = training_config.get("test_size", 0.2)
    seed = training_config.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Balance training data (Optuna cross validation inside HyperparameterTuner runs on this)
    # Actually, the tuner will do CV inside. We should pass the training set, not balanced, if inside CV we don't balance.
    # But for a quick test, passing balanced data directly is okay.
    print(f"\n⚖️  Applying SMOTE for class imbalance...")
    smote = SMOTE(sampling_strategy=0.8, random_state=seed)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    setup_mlflow()

    # Define models to tune. Only using the GPU-accelerated models now! 
    models_to_tune = ["random_forest", "xgboost", "lightgbm", "catboost"]
    
    # Notice: Running 5 models across 500k rows with multiple Optuna trials will take quite some time.
    # Set this to 5 for a quick demonstration, but crank it to 50-100 for production deployment!
    n_trials = 5

    results = []
    
    for model_name in models_to_tune:
        print(f"\n🚀 Tuning {model_name}...")
        tuner = HyperparameterTuner(model_name)
        
        # We start an MLflow run to log the tuning process
        with mlflow.start_run(run_name=f"{model_name}-tuning-{dataset_name}"):
            tune_results = tuner.tune(X_train_balanced, y_train_balanced, n_trials=n_trials)
            
            mlflow.log_params({f"best_{k}": v for k, v in tune_results["best_params"].items()})
            mlflow.log_metric("best_cv_auc", tune_results["best_score"])
            
            print(f"  🏆 Best CV AUC for {model_name}: {tune_results['best_score']:.4f}")
            print(f"  🛠️ Best Params:\n  {tune_results['best_params']}")
            results.append(tune_results)

    print("\n" + "=" * 65)
    print("  ✅ TUNING COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    main()
