"""
Data Validation Module
======================
Validates data quality using custom checks and Great Expectations.
Ensures data integrity at every stage of the pipeline.
"""

from typing import Any

import pandas as pd
from loguru import logger

from src.utils.config import get_dataset_config
from src.utils.helpers import timer


class DataValidator:
    """Validates data quality with configurable checks."""

    def __init__(self, dataset_name: str | None = None):
        self.config = get_dataset_config(dataset_name)
        self.validation_results: list[dict[str, Any]] = []

    def _add_result(self, check: str, passed: bool, details: str = "") -> None:
        """Record a validation check result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.validation_results.append({
            "check": check,
            "passed": passed,
            "details": details,
        })
        logger.info(f"{status} | {check}: {details}")

    @timer
    def validate(self, df: pd.DataFrame) -> bool:
        """Run all validation checks on a DataFrame.

        Args:
            df: DataFrame to validate.

        Returns:
            True if all critical checks pass.
        """
        self.validation_results = []
        logger.info(f"🔍 Running validation for '{self.config['name']}' dataset...")

        self._check_not_empty(df)
        self._check_required_columns(df)
        self._check_target_column(df)
        self._check_missing_values(df)
        self._check_duplicates(df)
        self._check_data_types(df)
        self._check_numerical_ranges(df)
        self._check_categorical_values(df)

        passed = all(r["passed"] for r in self.validation_results)
        total = len(self.validation_results)
        n_passed = sum(1 for r in self.validation_results if r["passed"])

        logger.info(f"\n{'='*50}")
        logger.info(f"Validation: {n_passed}/{total} checks passed")
        logger.info(f"Overall: {'✅ PASSED' if passed else '❌ FAILED'}")

        return passed

    def _check_not_empty(self, df: pd.DataFrame) -> None:
        self._add_result(
            "DataFrame not empty",
            len(df) > 0,
            f"{len(df)} rows"
        )

    def _check_required_columns(self, df: pd.DataFrame) -> None:
        required = set(
            self.config.get("categorical_features", [])
            + self.config.get("numerical_features", [])
            + [self.config["target"]]
        )
        missing = required - set(df.columns)
        self._add_result(
            "Required columns present",
            len(missing) == 0,
            f"Missing: {missing}" if missing else "All present"
        )

    def _check_target_column(self, df: pd.DataFrame) -> None:
        target = self.config["target"]
        if target in df.columns:
            n_classes = df[target].nunique()
            self._add_result(
                "Target is binary",
                n_classes == 2,
                f"{n_classes} unique values in '{target}'"
            )
        else:
            self._add_result("Target column exists", False, f"'{target}' not found")

    def _check_missing_values(self, df: pd.DataFrame) -> None:
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 30]
        self._add_result(
            "No columns >30% missing",
            len(high_missing) == 0,
            f"High missing: {dict(high_missing)}" if len(high_missing) > 0
            else f"Max missing: {missing_pct.max():.1f}%"
        )

    def _check_duplicates(self, df: pd.DataFrame) -> None:
        n_dupes = df.duplicated().sum()
        self._add_result(
            "Duplicate rows <5%",
            n_dupes / len(df) < 0.05,
            f"{n_dupes} duplicates ({n_dupes/len(df)*100:.1f}%)"
        )

    def _check_data_types(self, df: pd.DataFrame) -> None:
        numerical = self.config.get("numerical_features", [])
        issues = []
        for col in numerical:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(col)
        self._add_result(
            "Numerical columns are numeric",
            len(issues) == 0,
            f"Non-numeric: {issues}" if issues else "All correct"
        )

    def _check_numerical_ranges(self, df: pd.DataFrame) -> None:
        numerical = self.config.get("numerical_features", [])
        issues = []
        for col in numerical:
            if (
                col in df.columns
                and pd.api.types.is_numeric_dtype(df[col])
                and (df[col].min() < -1e6 or df[col].max() > 1e9)
            ):
                issues.append(f"{col}: [{df[col].min()}, {df[col].max()}]")
        self._add_result(
            "Numerical values in reasonable range",
            len(issues) == 0,
            f"Outliers: {issues}" if issues else "All in range"
        )

    def _check_categorical_values(self, df: pd.DataFrame) -> None:
        categorical = self.config.get("categorical_features", [])
        issues = []
        for col in categorical:
            if col in df.columns and df[col].nunique() > 50:
                issues.append(f"{col}: {df[col].nunique()} unique values")
        self._add_result(
            "Categorical cardinality <50",
            len(issues) == 0,
            f"High cardinality: {issues}" if issues else "All acceptable"
        )

    def get_report(self) -> dict:
        """Get the full validation report."""
        return {
            "dataset": self.config["name"],
            "total_checks": len(self.validation_results),
            "passed": sum(1 for r in self.validation_results if r["passed"]),
            "failed": sum(1 for r in self.validation_results if not r["passed"]),
            "results": self.validation_results,
        }


if __name__ == "__main__":
    from src.data.ingest import ingest_dataset

    df = ingest_dataset()
    validator = DataValidator()
    validator.validate(df)
