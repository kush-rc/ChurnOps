"""Unit tests for data preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import DataPreprocessor


class TestPreprocessor:
    """Tests for data preprocessing."""

    def test_remove_duplicates(self, sample_telco_data):
        """Test duplicate removal."""
        # Add duplicates
        df = pd.concat([sample_telco_data, sample_telco_data.iloc[:5]], ignore_index=True)
        preprocessor = DataPreprocessor("telco")
        result = preprocessor._remove_duplicates(df)
        assert len(result) <= len(df)

    def test_encode_target_binary(self, sample_telco_data):
        """Test target encoding to 0/1."""
        preprocessor = DataPreprocessor("telco")
        result = preprocessor._encode_target(sample_telco_data.copy())
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_handle_missing_values(self, sample_telco_data):
        """Test missing value handling."""
        df = sample_telco_data.copy()
        df.loc[0, "MonthlyCharges"] = np.nan
        preprocessor = DataPreprocessor("telco")
        result = preprocessor._handle_missing_values(df)
        assert result["MonthlyCharges"].isnull().sum() == 0
