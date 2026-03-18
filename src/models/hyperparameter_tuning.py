"""
Hyperparameter Tuning Module
=============================
Uses Optuna for Bayesian hyperparameter optimization.
"""

from typing import Any

import numpy as np
import optuna
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils.config import get_model_config, get_training_config
from src.utils.helpers import timer


class HyperparameterTuner:
    """Bayesian hyperparameter optimization with Optuna."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        self.training_config = get_training_config()
        self.best_params: dict | None = None
        self.study: optuna.Study | None = None

    def _create_objective(
        self, X: np.ndarray, y: np.ndarray
    ):
        """Create an Optuna objective function for the model."""
        tuning_config = self.model_config.get("tuning", {})
        search_space = tuning_config.get("search_space", {})
        cv_folds = self.training_config.get("cv_folds", 5)
        scoring = self.training_config.get("cv_scoring", "roc_auc")
        seed = self.training_config.get("random_state", 42)

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for param_name, bounds in search_space.items():
                if isinstance(bounds, list) and len(bounds) == 2:
                    if isinstance(bounds[0], float):
                        params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1])
                    elif isinstance(bounds[0], int):
                        params[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1])
                    elif isinstance(bounds[0], str):
                        params[param_name] = trial.suggest_categorical(param_name, bounds)
                elif isinstance(bounds, list):
                    params[param_name] = trial.suggest_categorical(param_name, bounds)

            # Create model with suggested params
            model = self._create_model(params)

            # Cross-validate (n_jobs=1 to protect GPU VRAM, verbose=3 to show detailed progress)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            print("\n      ▶️ Evaluator: Running 5-Fold Cross Validation fold-by-fold...", flush=True)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1, verbose=3)

            return scores.mean()

        return objective

    def _create_model(self, params: dict) -> Any:
        """Create a model instance with given parameters."""
        base_params = self.model_config.get("params", {}).copy()
        base_params.update(params)

        if self.model_name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**base_params)
        elif self.model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**base_params)
        elif self.model_name == "catboost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**base_params)
        elif self.model_name == "random_forest":
            from xgboost import XGBRFClassifier
            return XGBRFClassifier(**base_params)
        elif self.model_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**base_params)
        elif self.model_name == "neural_net":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(**base_params)

        raise ValueError(f"Unknown model: {self.model_name}")

    @timer
    def tune(
        self, X: np.ndarray, y: np.ndarray, n_trials: int | None = None
    ) -> dict:
        """Run hyperparameter optimization.

        Args:
            X: Feature matrix.
            y: Target variable.
            n_trials: Number of Optuna trials (overrides config).

        Returns:
            Dictionary with best parameters and score.
        """
        tuning_config = self.model_config.get("tuning", {})
        n_trials = n_trials or tuning_config.get("n_trials", 50)

        logger.info(f"🔧 Tuning {self.model_name} with {n_trials} trials...")

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(direction="maximize")
        objective = self._create_objective(X, y)
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = self.study.best_params
        best_score = self.study.best_value

        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")

        return {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_score": best_score,
            "n_trials": n_trials,
        }


if __name__ == "__main__":
    logger.info("Run hyperparameter tuning via: python -m src.models.hyperparameter_tuning")
