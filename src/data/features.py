"""
Feature Engineering Module
==========================
Creates derived features from cleaned data to improve model performance.
Generates 50+ features including interactions, aggregations, and domain-specific features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.utils.config import get_dataset_config, get_path
from src.utils.helpers import load_dataframe, save_dataframe, timer


class FeatureEngineer:
    """Creates engineered features for churn prediction models."""

    def __init__(self, dataset_name: str | None = None):
        self.config = get_dataset_config(dataset_name)
        self.preprocessing_config = self.config.get("preprocessing", {})
        self.encoders: dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []

    @timer
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering pipeline.

        Args:
            df: Cleaned DataFrame from preprocessing.

        Returns:
            DataFrame with engineered features.
        """
        logger.info(f"⚙️  Engineering features for '{self.config['name']}'...")
        logger.info(f"Input shape: {df.shape}")

        df = df.copy()

        # Step 1: Create domain-specific features
        df = self._create_domain_features(df)

        # Step 2: Create interaction features
        df = self._create_interaction_features(df)

        # Step 3: Create binned features
        df = self._create_binned_features(df)

        # Step 4: Encode categorical features
        df = self._encode_categoricals(df)

        # Step 5: Scale numerical features
        df = self._scale_numericals(df)

        self.feature_names = [c for c in df.columns if c != self.config["target"]]
        logger.info(f"Output shape: {df.shape}")
        logger.info(f"Total features: {len(self.feature_names)}")

        return df

    def _create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features based on dataset type."""
        dataset_name = self.config["name"]

        if dataset_name == "telco":
            df = self._telco_features(df)
        elif dataset_name == "bank":
            df = self._bank_features(df)
        elif dataset_name == "ecommerce":
            df = self._ecommerce_features(df)

        return df

    def _telco_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Telco-specific feature engineering."""
        if "tenure" in df.columns and "MonthlyCharges" in df.columns:
            # Revenue features
            df["TotalRevenue"] = df["tenure"] * df["MonthlyCharges"]
            df["AvgMonthlySpend"] = np.where(
                df["tenure"] > 0,
                df.get("TotalCharges", df["TotalRevenue"]) / df["tenure"],
                df["MonthlyCharges"]
            )

            # Tenure-based features
            df["tenure_years"] = df["tenure"] / 12
            df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
            df["is_loyal_customer"] = (df["tenure"] >= 36).astype(int)

            # Charge features
            df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

        # Service count
        service_cols = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        existing_service_cols = [c for c in service_cols if c in df.columns]
        if existing_service_cols:
            df["num_services"] = df[existing_service_cols].apply(
                lambda row: sum(1 for v in row if v not in ["No", "No internet service", 0]),
                axis=1
            )

        # Security features
        security_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
        existing_security = [c for c in security_cols if c in df.columns]
        if existing_security:
            df["has_security_features"] = df[existing_security].apply(
                lambda row: sum(1 for v in row if v == "Yes"),
                axis=1
            )

        # Streaming features
        stream_cols = ["StreamingTV", "StreamingMovies"]
        existing_stream = [c for c in stream_cols if c in df.columns]
        if existing_stream:
            df["num_streaming"] = df[existing_stream].apply(
                lambda row: sum(1 for v in row if v == "Yes"),
                axis=1
            )

        logger.info(f"Created {len(df.columns) - len(self.config.get('categorical_features', []))} Telco features")
        return df

    def _bank_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bank-specific feature engineering."""
        if "Balance" in df.columns and "EstimatedSalary" in df.columns:
            df["balance_salary_ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
            df["has_zero_balance"] = (df["Balance"] == 0).astype(int)

        if "Age" in df.columns:
            df["age_group"] = pd.cut(
                df["Age"], bins=[0, 30, 40, 50, 60, 100],
                labels=["young", "adult", "middle", "senior", "elderly"]
            ).astype(str)

        if "NumOfProducts" in df.columns:
            df["has_multiple_products"] = (df["NumOfProducts"] > 1).astype(int)

        if "CreditScore" in df.columns:
            df["credit_score_bin"] = pd.cut(
                df["CreditScore"], bins=[0, 580, 670, 740, 800, 900],
                labels=["poor", "fair", "good", "very_good", "excellent"]
            ).astype(str)

        logger.info("Created Bank-specific features")
        return df

    def _ecommerce_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """E-commerce specific feature engineering."""
        if "OrderCount" in df.columns and "Tenure" in df.columns:
            df["orders_per_tenure"] = df["OrderCount"] / (df["Tenure"] + 1)

        if "CashbackAmount" in df.columns and "OrderCount" in df.columns:
            df["cashback_per_order"] = df["CashbackAmount"] / (df["OrderCount"] + 1)

        if "Complain" in df.columns:
            df["has_complained"] = df["Complain"].astype(int)

        if "SatisfactionScore" in df.columns:
            df["is_dissatisfied"] = (df["SatisfactionScore"] <= 2).astype(int)

        logger.info("Created E-commerce specific features")
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numerical columns."""
        numerical = [c for c in self.config.get("numerical_features", []) if c in df.columns]

        if len(numerical) >= 2:
            # Create top interaction pairs (limit to avoid explosion)
            pairs_created = 0
            for i, col1 in enumerate(numerical[:5]):
                for col2 in numerical[i+1:6]:
                    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                        pairs_created += 1

            logger.info(f"Created {pairs_created} interaction features")

        return df

    def _create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binned versions of numerical features."""
        numerical = [c for c in self.config.get("numerical_features", []) if c in df.columns]

        for col in numerical:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
                try:
                    df[f"{col}_bin"] = pd.qcut(df[col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
                    df[f"{col}_bin"] = df[f"{col}_bin"].astype(str)
                except Exception:
                    pass  # Skip if binning fails

        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features based on config strategy."""
        strategy = self.preprocessing_config.get("encode_strategy", "onehot")
        categorical = [c for c in df.select_dtypes(include=["object", "category"]).columns
                       if c != self.config["target"]]

        if not categorical:
            return df

        if strategy == "label":
            for col in categorical:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            logger.info(f"Label encoded {len(categorical)} columns")

        elif strategy == "onehot":
            df = pd.get_dummies(df, columns=categorical, drop_first=True, dtype=int)
            logger.info(f"One-hot encoded {len(categorical)} columns → {len(df.columns)} total")

        return df

    def _scale_numericals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        strategy = self.preprocessing_config.get("scale_strategy", "standard")
        target = self.config["target"]
        numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical = [c for c in numerical if c != target]

        if not numerical:
            return df

        if strategy == "standard":
            self.scaler = StandardScaler()
        elif strategy == "minmax":
            self.scaler = MinMaxScaler()
        elif strategy == "robust":
            self.scaler = RobustScaler()

        if self.scaler:
            df[numerical] = self.scaler.fit_transform(df[numerical])
            logger.info(f"Scaled {len(numerical)} numerical features ({strategy})")

        return df

    def save_features(self, df: pd.DataFrame) -> Path:
        """Save engineered features as Parquet.

        Returns:
            Path to the saved file.
        """
        output_dir = get_path("data_features")
        filename = f"{self.config['name']}_features.parquet"
        output_path = output_dir / filename
        save_dataframe(df, output_path)
        return output_path


if __name__ == "__main__":
    from src.data.ingest import ingest_dataset
    from src.data.preprocess import DataPreprocessor

    df = ingest_dataset()
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.preprocess(df)

    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df_clean)
    path = engineer.save_features(df_features)

    print(f"\nSaved to: {path}")
    print(f"Feature matrix shape: {df_features.shape}")
    print(f"Feature names ({len(engineer.feature_names)}):")
    for f in engineer.feature_names[:20]:
        print(f"  - {f}")
    if len(engineer.feature_names) > 20:
        print(f"  ... and {len(engineer.feature_names) - 20} more")
