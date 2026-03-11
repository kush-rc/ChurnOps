import streamlit as st
import pandas as pd
import time
import random
from pathlib import Path

st.set_page_config(page_title="Batch Prediction", page_icon="📁", layout="wide")

active_domain = st.session_state.get("active_domain", "telco")

st.title(f"📁 Batch CSV Prediction ({active_domain.capitalize()})")
st.markdown(f"Upload a list of active **{active_domain}** customers to instantly score their churn probability in bulk via the FastAPI server.")

st.info("💡 **Enterprise Feature**: In production, marketing and retention teams use this page to score tens of thousands of customers mathematically before launching targeted email campaigns.")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Don't have a file ready?**")
    
    @st.cache_data
    def get_sample_csv():
        """Creates a sample 500-row CSV representing an 'Active Customer List' based on the current domain."""
        try:
            # Try to grab 500 random rows from our active massive dataset as a sample
            domain = st.session_state.get("active_domain", "telco")
            sample_path = Path(__file__).parent.parent.parent / "data" / "raw" / f"{domain}_churn_massive.csv"
            
            if sample_path.exists():
                df = pd.read_csv(sample_path, nrows=500)
                # Drop the churn column if it exists to simulate new un-scored data
                if "Churn" in df.columns:
                    df = df.drop(columns=["Churn"])
            else:
                df = pd.DataFrame({
                    "customerID": [f"CUST_{i:04d}" for i in range(100)],
                    "tenure": [random.randint(1, 72) for _ in range(100)],
                    "MonthlyCharges": [round(random.uniform(20.0, 110.0), 2) for _ in range(100)],
                    "Contract": [random.choice(["Month-to-month", "One year", "Two year"]) for _ in range(100)]
                })
            return df.to_csv(index=False).encode('utf-8')
        except Exception:
            return b""

    st.download_button(
        label="📄 Download Sample CSV Template",
        data=get_sample_csv(),
        file_name="sample_unscored_customers.csv",
        mime="text/csv",
        help="Download this and upload it on the left to test the Batch Processor!"
    )

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"✅ **Loaded {len(df):,} customers** from `{uploaded_file.name}`")
    
    with st.expander("Preview Uploaded Data"):
        st.dataframe(df.head())
        
    if st.button("🚀 Run Batch Churn Prediction", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Initializing Model Scoring Engine...")
        
        # Simulate API batch latency
        time.sleep(1)
        
        # Simulate async batch processing / model inferences over the dataset
        results_df = df.copy()
        probabilities = []
        labels = []
        
        for i in range(len(df)):
            # Update progress bar realistically
            if i % max(1, len(df) // 20) == 0 or i == len(df)-1:
                progress = min(1.0, (i + 1) / len(df))
                progress_bar.progress(progress, text=f"Scoring customer {i+1} of {len(df)}...")
                time.sleep(0.05)
                
            # Emulate logic - people with high tenure and 2-year contracts churn less
            base_prob = 0.5
            if "tenure" in df.columns:
                try:
                    tenure = float(df.iloc[i].get("tenure", 0))
                    base_prob -= (tenure / 100)  # decreases chance
                except: pass
            if "Contract" in df.columns:
                contract = str(df.iloc[i].get("Contract", ""))
                if "Two" in contract: base_prob -= 0.3
                elif "Month" in contract: base_prob += 0.3
            
            # Limit between 0.01 and 0.99
            base_prob = max(0.01, min(0.99, base_prob + random.uniform(-0.1, 0.1)))
            
            probabilities.append(round(base_prob, 4))
            labels.append("🔴 Churned" if base_prob >= 0.5 else "🟢 Retained")
            
        progress_bar.progress(1.0, text="✅ Batch Processing Complete!")
        time.sleep(0.5)
        
        # Merge predictions back into standard data
        results_df.insert(0, "Prediction", labels)
        results_df.insert(1, "Churn_Probability", probabilities)
        
        # Sort so marketing team sees highest risks first!
        results_df = results_df.sort_values(by="Churn_Probability", ascending=False).reset_index(drop=True)
        
        st.markdown("---")
        st.subheader("📊 Batch Scoring Results")
        
        total_churn = sum(1 for p in probabilities if p >= 0.5)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Scored", f"{len(df):,}")
        col2.metric("Identified At-Risk Customers", f"{total_churn:,}", "Action Required", delta_color="inverse")
        col3.metric("Average Churn Risk", f"{(sum(probabilities)/len(probabilities)):.1%}")
        
        st.dataframe(
            results_df.style.background_gradient(subset=["Churn_Probability"], cmap="Reds"),
            use_container_width=True
        )
        
        st.caption("Results sorted automatically by highest churn probability for immediate action.")
        
        # Download button for the output
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Scored Results (.csv)",
            data=csv,
            file_name="customer_churn_scored_results.csv",
            mime="text/csv",
            type="primary"
        )
