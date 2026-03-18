"""
Drift Detection Module
======================
Uses Evidently AI to detect data and model drift.
Generates HTML reports and Prometheus-compatible metrics.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report
from loguru import logger

from src.utils.config import get_dataset_config, get_monitoring_config
from src.utils.helpers import timer


class DriftDetector:
    """Detects data and concept drift using Evidently AI."""

    def __init__(self, dataset_name: str | None = None):
        self.monitoring_config = get_monitoring_config()
        self.dataset_config = get_dataset_config(dataset_name)
        self.drift_config = self.monitoring_config.get("drift", {})

    def _get_column_mapping(self) -> ColumnMapping:
        """Create Evidently column mapping from config."""
        return ColumnMapping(
            target=self.dataset_config["target"],
            numerical_features=self.dataset_config.get("numerical_features", []),
            categorical_features=self.dataset_config.get("categorical_features", []),
        )

    @timer
    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run drift detection between reference and current data.

        Args:
            reference_data: Baseline/training data.
            current_data: New/production data.

        Returns:
            Drift detection results.
        """
        logger.info("🔍 Running drift detection...")

        column_mapping = self._get_column_mapping()

        # Data drift report
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )

        # Extract results
        results = data_drift_report.as_dict()
        drift_detected = self._check_drift_thresholds(results)

        logger.info(f"Drift detected: {drift_detected}")
        return {
            "drift_detected": drift_detected,
            "report": results,
        }

    def _check_drift_thresholds(self, report: dict) -> bool:
        """Check if drift exceeds configured thresholds."""
        try:
            metrics = report.get("metrics", [])
            for metric in metrics:
                result = metric.get("result", {})
                if result.get("dataset_drift", False):
                    return True
        except Exception as e:
            logger.error(f"Error checking drift thresholds: {e}")
        return False

    @timer
    def generate_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate an HTML drift report.

        Args:
            reference_data: Baseline data.
            current_data: New data.
            output_path: Where to save the HTML report.

        Returns:
            Path to the generated report.
        """
        column_mapping = self._get_column_mapping()

        report = Report(
            metrics=[
                DataDriftPreset(),
                TargetDriftPreset(),
            ]
        )
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )

        if output_path is None:
            reports_dir = Path("monitoring/evidently/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            output_path = reports_dir / "drift_report.html"

        output_path = Path(output_path)
        report.save_html(str(output_path))
        logger.info(f"📊 Drift report saved: {output_path}")

        return output_path
