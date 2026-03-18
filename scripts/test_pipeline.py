"""Quick test script to verify the data pipeline works end-to-end."""

import sys
import warnings
import os

warnings.filterwarnings("ignore")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Disable loguru to keep output clean
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="telco", help="Dataset to test (e.g. telco, ecommerce, saas)")
    args = parser.parse_args()
    
    dataset_name = args.dataset

    print("=" * 60)
    print(f"  DATA PIPELINE TEST - {dataset_name.upper()} Customer Churn")
    print("=" * 60)

    # Step 1: Ingest
    print("\n[1/4] INGESTING DATA...")
    from src.data.ingest import ingest_dataset
    df = ingest_dataset(dataset_name)
    print(f"  ✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Target distribution: {dict(df['Churn'].value_counts())}")

    # Step 2: Validate
    print("\n[2/4] VALIDATING DATA...")
    from src.data.validate import DataValidator
    validator = DataValidator(dataset_name)
    is_valid = validator.validate(df)
    report = validator.get_report()
    print(f"  ✅ Validation: {report['passed']}/{report['total_checks']} checks passed")

    # Step 3: Preprocess
    print("\n[3/4] PREPROCESSING...")
    from src.data.preprocess import DataPreprocessor
    preprocessor = DataPreprocessor(dataset_name)
    df_clean = preprocessor.preprocess(df)
    path_clean = preprocessor.save_processed(df_clean)
    path_prep_state = preprocessor.save_state()
    print(f"  ✅ Cleaned: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    print(f"  Saved to: {path_clean}")
    print(f"  Saved state to: {path_prep_state}")

    # Step 4: Feature Engineering
    print("\n[4/4] ENGINEERING FEATURES...")
    from src.data.features import FeatureEngineer
    engineer = FeatureEngineer(dataset_name)
    df_features = engineer.engineer_features(df_clean)
    path_features = engineer.save_features(df_features)
    path_eng_state = engineer.save_state()
    print(f"  ✅ Features: {df_features.shape[0]:,} rows × {df_features.shape[1]} columns")
    print(f"  Total feature columns: {len(engineer.feature_names)}")
    print(f"  Saved to: {path_features}")
    print(f"  Saved state to: {path_eng_state}")

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ ALL 4 PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n  Raw data:      {df.shape}")
    print(f"  After clean:   {df_clean.shape}")
    print(f"  After features: {df_features.shape}")
    print(f"  Features created: {df_features.shape[1] - df.shape[1]} new columns")


if __name__ == "__main__":
    main()
