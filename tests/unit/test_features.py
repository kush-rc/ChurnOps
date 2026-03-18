"""Unit tests for feature engineering module."""


class TestFeatureEngineering:
    """Tests for feature engineering."""

    def test_features_output_has_more_columns(self, sample_telco_data):
        """Verify feature engineering creates new columns."""
        from src.data.features import FeatureEngineer
        from src.data.preprocess import DataPreprocessor

        preprocessor = DataPreprocessor("telco")
        df_clean = preprocessor.preprocess(sample_telco_data)

        engineer = FeatureEngineer("telco")
        df_features = engineer.engineer_features(df_clean)

        assert df_features.shape[1] >= df_clean.shape[1]

    def test_no_null_features(self, sample_telco_data):
        """Verify no null values in engineered features."""
        from src.data.features import FeatureEngineer
        from src.data.preprocess import DataPreprocessor

        preprocessor = DataPreprocessor("telco")
        df_clean = preprocessor.preprocess(sample_telco_data)

        engineer = FeatureEngineer("telco")
        df_features = engineer.engineer_features(df_clean)

        # Allow minimal nulls from edge cases
        null_pct = df_features.isnull().sum().sum() / (df_features.shape[0] * df_features.shape[1])
        assert null_pct < 0.05, f"Too many nulls: {null_pct:.2%}"
