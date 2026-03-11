import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Model Comparison")
st.markdown("Compare the performance of all models trained in Phase 2.")

# Load the JSON results
report_path = Path(__file__).parent.parent.parent / "reports" / "model_comparison.json"

if report_path.exists():
    with open(report_path, "r") as f:
        results = json.load(f)
    
    # Process the data
    df = pd.DataFrame(results)
    
    # Format the df for display
    display_cols = ["model_name", "roc_auc", "f1", "accuracy", "precision", "recall", "cv_roc_auc_mean", "training_time_seconds"]
    df_display = df[display_cols].copy()
    
    # Rename columns nicely
    df_display.columns = ["Model", "AUC-ROC (Test)", "F1 Score", "Accuracy", "Precision", "Recall", "CV AUC-ROC", "Time (sec)"]
    
    # Clean model names
    df_display["Model"] = df_display["Model"].str.replace("_", " ").str.title()
    
    st.markdown("### Model Performance Metrics")
    
    # Highlight the best model
    st.dataframe(
        df_display.style.highlight_max(subset=["AUC-ROC (Test)", "F1 Score", "CV AUC-ROC"], color='#1E3D59')
                 .format({"Time (sec)": "{:.2f}s"}),
        use_container_width=True,
        hide_index=True
    )
    
    # Visualizations
    st.markdown("### Metric Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.bar_chart(df_display.set_index("Model")[["AUC-ROC (Test)", "CV AUC-ROC"]])
    
    with col2:
        st.bar_chart(df_display.set_index("Model")[["F1 Score", "Accuracy"]])
        
    st.info("💡 **Logistic Regression** achieved the best test AUC-ROC, while tree-based algorithms like **LightGBM** and **CatBoost** showed stronger CV stability.")
else:
    st.warning("No model comparison report found. Please run the training pipeline first.")
