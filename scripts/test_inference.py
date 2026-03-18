"""
End-to-End Inference Test - Diagnose column mismatch
"""
import sys
import traceback
import pandas as pd
import numpy as np

def test_inference():
    print("=" * 60)
    print("END-TO-END INFERENCE TEST")
    print("=" * 60)

    # Step 1: Load saved pipeline states
    print("\n[1/5] Loading saved pipeline states...")
    try:
        from src.data.preprocess import DataPreprocessor
        from src.data.features import FeatureEngineer

        preprocessor = DataPreprocessor.load_state("telco")
        print(f"  OK DataPreprocessor loaded (imputation_values: {len(preprocessor.imputation_values)} cols)")

        engineer = FeatureEngineer.load_state("telco")
        print(f"  OK FeatureEngineer loaded (feature_names: {len(engineer.feature_names)} cols)")
        print(f"     _is_fit={engineer._is_fit}, ohe={type(engineer.ohe).__name__}")
        if engineer.ohe is not None:
            print(f"     OHE feature_names_in_: {list(engineer.ohe.feature_names_in_)}")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    # Step 2: Raw input
    print("\n[2/5] Creating raw input DataFrame...")
    payload = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.6, "TotalCharges": 780,
    }
    df = pd.DataFrame([payload])
    print(f"  OK Input shape: {df.shape}")

    # Step 3: Preprocessing
    print("\n[3/5] Preprocessing...")
    try:
        df_clean = preprocessor.preprocess(df)
        print(f"  OK Clean shape: {df_clean.shape}")
    except Exception as e:
        print(f"  FAILED at preprocessing: {e}")
        traceback.print_exc()
        return False

    # Step 4: Feature engineering
    print("\n[4/5] Engineering features...")
    try:
        df_features = engineer.engineer_features(df_clean)
        print(f"  OK Features shape: {df_features.shape}")

        expected = set(engineer.feature_names)
        actual = set(df_features.columns)
        missing = expected - actual
        extra = actual - expected

        if missing:
            print(f"  WARNING Missing from inference ({len(missing)}): {sorted(missing)[:10]}")
        if extra:
            print(f"  WARNING Extra in inference ({len(extra)}): {sorted(extra)[:10]}")
        if not missing and not extra:
            print(f"  OK Feature columns match perfectly ({len(expected)} features)")

        # Add missing columns with 0 and order correctly
        for col in missing:
            df_features[col] = 0
        X = df_features[engineer.feature_names]
        print(f"  OK Final X shape for model: {X.shape}")

    except Exception as e:
        print(f"  FAILED at feature engineering: {e}")
        traceback.print_exc()
        return False

    # Step 5: Load MLflow model and predict
    print("\n[5/5] Loading MLflow model and predicting...")
    try:
        import mlflow
        from src.utils.config import get_config
        config = get_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

        client = mlflow.tracking.MlflowClient()
        model_name = "telco-churn-model"

        try:
            latest = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            if latest:
                latest_version = sorted(latest, key=lambda v: int(v.version))[-1]
                model_uri = f"models:/{model_name}/{latest_version.version}"
                print(f"  Found model: {model_uri} (stage={latest_version.current_stage})")
                model = mlflow.pyfunc.load_model(model_uri)
            else:
                print(f"  No registered model '{model_name}'. Trying run artifacts...")
                experiments = [e for e in client.search_experiments() if "telco" in e.name.lower()]
                if experiments:
                    runs = client.search_runs(experiment_ids=[experiments[0].experiment_id], max_results=1, order_by=["start_time DESC"])
                    if runs:
                        run_id = runs[0].info.run_id
                        model_uri = f"runs:/{run_id}/model"
                        print(f"  Using run artifact: {model_uri}")
                        model = mlflow.pyfunc.load_model(model_uri)
                    else:
                        print(f"  FAILED: No runs found")
                        return False
                else:
                    print(f"  FAILED: No telco experiments found")
                    return False
        except Exception as e:
            print(f"  Could not find registered model, searching experiments: {e}")
            experiments = [e for e in client.search_experiments() if "telco" in e.name.lower()]
            if experiments:
                runs = client.search_runs(experiment_ids=[experiments[0].experiment_id], max_results=1, order_by=["start_time DESC"])
                if runs:
                    run_id = runs[0].info.run_id
                    model_uri = f"runs:/{run_id}/model"
                    print(f"  Using run artifact: {model_uri}")
                    model = mlflow.pyfunc.load_model(model_uri)
                else:
                    print(f"  FAILED: No runs found in telco experiments")
                    return False
            else:
                print(f"  FAILED: No telco experiments found")
                return False

        # Check model signature
        try:
            sig = model.metadata.get_input_schema()
            if sig:
                model_cols = [col.name for col in sig.inputs]
                our_cols = list(X.columns)
                model_set = set(model_cols)
                our_set = set(our_cols)
                missing_for_model = model_set - our_set
                extra_for_model = our_set - model_set
                
                print(f"  Model expects {len(model_cols)} features, we have {len(our_cols)}")
                if missing_for_model:
                    print(f"  MISMATCH - Model needs but we dont have ({len(missing_for_model)}): {sorted(missing_for_model)[:15]}")
                if extra_for_model:
                    print(f"  MISMATCH - We have but model doesnt need ({len(extra_for_model)}): {sorted(extra_for_model)[:15]}")
                if not missing_for_model and not extra_for_model:
                    print(f"  OK Columns match model signature perfectly!")
        except Exception as e:
            print(f"  WARNING: Could not check model signature: {e}")

        # Try prediction
        try:
            predictions = model.predict(X)
            print(f"  OK Predictions: {predictions}")
            print(f"     Type: {type(predictions)}, shape: {getattr(predictions, 'shape', 'N/A')}")
        except Exception as e:
            print(f"  FAILED: Model prediction error: {e}")
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("END-TO-END INFERENCE TEST PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)
