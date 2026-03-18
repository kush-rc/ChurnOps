"""Unit tests for data ingestion module."""



class TestDataIngest:
    """Tests for data ingestion."""

    def test_sample_data_has_expected_columns(self, sample_telco_data):
        """Verify sample data has the expected columns."""
        expected_cols = {"customerID", "gender", "tenure", "MonthlyCharges", "Churn"}
        assert expected_cols.issubset(set(sample_telco_data.columns))

    def test_sample_data_not_empty(self, sample_telco_data):
        """Verify sample data is not empty."""
        assert len(sample_telco_data) > 0

    def test_sample_data_has_target(self, sample_telco_data):
        """Verify target column exists and is binary."""
        assert "Churn" in sample_telco_data.columns
        assert sample_telco_data["Churn"].nunique() == 2
