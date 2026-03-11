import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Model Explainability", page_icon="🔍", layout="wide")
st.title("🔍 Model Explainability (SHAP)")
st.markdown("Understand *why* the model is making its predictions.")

st.info("💡 **Note**: In the live pipeline, SHAP API endpoints (`/api/v1/explain`) return these contribution maps dynamically for any given customer.")

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Top Churn Drivers")
    st.write("These features globally impact the churn rate the most.")
    
    # Mock global SHAP feature importance
    data = {
        "Feature": ["Contract (Month-to-month)", "Tenure", "Monthly Charges", "Internet (Fiber optic)", "Payment (Electronic check)", "Tech Support (No)"],
        "Importance": [0.38, 0.25, 0.18, 0.12, 0.09, 0.08]
    }
    df = pd.DataFrame(data).sort_values("Importance", ascending=True)
    
    import altair as alt
    
    chart = alt.Chart(df).mark_bar(color="#ef4444").encode(
        x=alt.X('Importance:Q', title="SHAP Value"),
        y=alt.Y('Feature:N', sort="-x", title="")
    ).properties(height=300)
    
    st.altair_chart(chart, use_container_width=True)
    
with col2:
    st.markdown("### Local Explanation (Single Customer)")
    st.write("Showing how features pushed the prediction for **Customer #102**.")
    
    # Render a mock SHAP waterfall chart using native streamit elements
    base_val = 0.26
    final_prob = 0.82
    
    st.metric("Base Probability", f"{base_val:.1%}")
    st.metric("Final Churn Probability", f"{final_prob:.1%}", "High Risk", delta_color="inverse")
    
    st.markdown("##### Contributions:")
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.write("**Contract = Month-to-month**")
    c2.write("`+ 0.35`")
    c3.markdown('🔴 **Increase**')
    
    c1.write("**Tenure = 2 months**")
    c2.write("`+ 0.22`")
    c3.markdown('🔴 **Increase**')

    c1.write("**Internet = Fiber optic**")
    c2.write("`+ 0.14`")
    c3.markdown('🔴 **Increase**')

    c1.write("**MonthlyCharges = $85.50**")
    c2.write("`+ 0.10`")
    c3.markdown('🔴 **Increase**')
    
    c1.write("**Dependents = Yes**")
    c2.write("`- 0.25`")
    c3.markdown('🟢 **Decrease**')
    
    st.progress(0.82, text="Prediction: Churned")

st.markdown("---")
st.caption("SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.")


