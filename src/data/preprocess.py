"""
Data Preprocessing Module
=========================
Handles data cleaning, type conversion, missing value imputation,
encoding, and scaling. Outputs clean Parquet files.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from src.utils.config import get_dataset_config, get_path
from src.utils.helpers import save_dataframe, timer

if TYPE_CHECKING:
    from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    """Cleans and preprocesses raw data for feature engineering."""

    def __init__(self, dataset_name: str | None = None):
        self.config = get_dataset_config(dataset_name)
        self.preprocessing_config = self.config.get("preprocessing", {})
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.imputation_values: dict[str, Any] = {}
        self.scaler = None
        self._is_fit = False

    @timer
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full preprocessing pipeline.

        Args:
            df: Raw DataFrame from ingestion.

        Returns:
            Cleaned and preprocessed DataFrame.
        """
        logger.info(f"🧹 Preprocessing '{self.config['name']}' dataset...")
        logger.info(f"Input shape: {df.shape}")

        df = df.copy()

        # Input imputation for sparse inference payloads
        if not self._is_fit:
            target = self.config.get("target")
            # Add missing numerical features with 0
            for col in self.config.get("numerical_features", []):
                if col != target and col not in df.columns:
                    df[col] = 0.0
            # Add missing categorical features with "Unknown"
            for col in self.config.get("categorical_features", []):
                if col != target and col not in df.columns:
                    df[col] = "Unknown"

        # Step 1: Drop unnecessary columns
        df = self._drop_columns(df)

        self._is_fit = True

        # Step 2: Handle target column
        df = self._encode_target(df)

        # Step 3: Fix data types
        df = self._fix_data_types(df)

        # Step 4: Handle missing values
        df = self._handle_missing_values(df)

        # Step 5: Remove duplicates
        df = self._remove_duplicates(df)

        # Step 6: Drop ID column (keep for reference but not training)
        id_col = self.config.get("id_column")
        if id_col and id_col in df.columns:
            df = df.drop(columns=[id_col])
            logger.info(f"Dropped ID column: {id_col}")

        logger.info(f"Output shape: {df.shape}")
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns specified in config."""
        drop_cols = self.config.get("drop_columns", [])
        existing = [c for c in drop_cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            logger.info(f"Dropped columns: {existing}")
        return df

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode target column to binary (0/1)."""
        target = self.config["target"]
        positive_label = self.config.get("target_positive_label")

        if target in df.columns and positive_label is not None and df[target].dtype == object:
            df[target] = (df[target] == positive_label).astype(int)
            logger.info(f"Encoded target '{target}': '{positive_label}' → 1, rest → 0")

        return df

    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for numerical columns."""
        numerical = self.config.get("numerical_features", [])
        for col in numerical:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on config strategy."""
        strategy = self.preprocessing_config.get("handle_missing", "median")

        missing_before = df.isnull().sum().sum()
        if missing_before == 0:
            logger.info("No missing values found")
            return df

        logger.info(f"Missing values before: {missing_before}")

        numerical = self.config.get("numerical_features", [])
        categorical = self.config.get("categorical_features", [])

        # Numerical: fill with median/mean
        for col in numerical:
            if col in df.columns and (df[col].isnull().any() or not self._is_fit):
                if self._is_fit:
                    if strategy == "median":
                        self.imputation_values[col] = df[col].median()
                    elif strategy == "mean":
                        self.imputation_values[col] = df[col].mean()
                    elif strategy == "drop":
                        pass # Handled differently, but let's default to median
                        self.imputation_values[col] = df[col].median()

                if col in self.imputation_values and not pd.isna(self.imputation_values[col]):
                    df[col] = df[col].fillna(self.imputation_values[col])
                elif strategy == "drop" and self._is_fit:
                    df = df.dropna(subset=[col])

        # Categorical: fill with mode
        for col in categorical:
            if col in df.columns and (df[col].isnull().any() or not self._is_fit):
                if self._is_fit:
                    mode_vals = df[col].mode()
                    if len(mode_vals) > 0:
                        self.imputation_values[col] = mode_vals[0]

                if col in self.imputation_values and not pd.isna(self.imputation_values[col]):
                    df[col] = df[col].fillna(self.imputation_values[col])

        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values after: {missing_after}")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(f"Removed {n_removed} duplicate rows")
        return df

    def save_processed(self, df: pd.DataFrame) -> Path:
        """Save preprocessed data as Parquet.

        Returns:
            Path to the saved file.
        """
        output_dir = get_path("data_processed")
        filename = f"{self.config['name']}_clean.parquet"
        output_path = output_dir / filename
        save_dataframe(df, output_path)
        return output_path

    def save_state(self) -> Path:
        """Save the fitted preprocessor state using joblib."""
        import joblib
        output_dir = get_path("models") / "preprocessors"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.config['name']}_preprocessor.joblib"
        joblib.dump(self, path)
        logger.info(f"💾 Saved preprocessor state to {path}")
        return path

    @classmethod
    def load_state(cls, dataset_name: str):
        """Load a fitted preprocessor state."""
        import joblib
        path = get_path("models") / "preprocessors" / f"{dataset_name}_preprocessor.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor state not found at {path}")
        preprocessor = joblib.load(path)
        preprocessor._is_fit = False  # Set to false for inference so it doesn't relearn
        return preprocessor


if __name__ == "__main__":
    from src.data.ingest import ingest_dataset

    df = ingest_dataset()
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.preprocess(df)
    path = preprocessor.save_processed(df_clean)
    print(f"\nSaved to: {path}")
    print(f"Clean data shape: {df_clean.shape}")
    print(f"\nData types:\n{df_clean.dtypes}")
