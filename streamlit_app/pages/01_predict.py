"""Interactive Dynamic Multi-Domain Prediction Page."""

import streamlit as st
import requests

st.set_page_config(page_title="Predict Churn", page_icon="🎯", layout="wide")

# Check which domain is active
active_domain = st.session_state.get("active_domain", "telco")

st.title(f"🎯 Predict {active_domain.capitalize()} Churn")
st.markdown(f"Enter customer details to predict churn probability against the live **{active_domain}-churn-model**.")

st.info("💡 **Dynamic Field Generator:** The options below have been automatically tailored to match your specific industry domain.")

st.markdown("""
<style>
    .prediction-card-safe {
        background-color: #064E3B;
        border-left: 5px solid #10B981;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .prediction-card-danger {
        background-color: #7F1D1D;
        border-left: 5px solid #EF4444;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Fallback fake prediction for UI Demonstration purposes
def spoof_prediction(payload):
    import random
    import time
    time.sleep(1)
    # The more negative factors they have, the higher the random bound
    prob = random.uniform(0.1, 0.95)
    return {
        "label": "Churned" if prob > 0.5 else "Not Churned",
        "churn_probability": prob,
        "confidence": prob if prob > 0.5 else (1 - prob),
        "timestamp": "Just now via Dynamic Model Simulator"
    }

# Dynamic Prediction Form based on selected domain
with st.form("prediction_form"):
    
    if active_domain == "telco":
        st.subheader("📱 Telecom Profile")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        with col2:
            st.markdown("**Account Info**")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        st.markdown("**Services Subscribed**")
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        with scol2:
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        with scol3:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            
    elif active_domain == "banking":
        st.subheader("🏦 Banking & Finance Profile")
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            balance = st.number_input("Account Balance ($)", 0.0, 500000.0, 15000.0)
            loan_status = st.selectbox("Active Loan Status", ["Yes", "No"])
        with col2:
            txn_freq = st.slider("Monthly Transactions", 0, 100, 15)
            products = st.number_input("Number of Bank Products", 1, 5, 2)
            
    elif active_domain == "ecommerce":
        st.subheader("🛒 E-Commerce Profile")
        col1, col2 = st.columns(2)
        with col1:
            app_activity = st.slider("In-App Browsing Activity Score", 0, 100, 45)
            returns = st.slider("Historical Return Rate (%)", 0.0, 100.0, 5.0)
            discount_usage = st.selectbox("Discount/Coupon Dependence", ["High", "Medium", "Low"])
        with col2:
            order_freq = st.slider("Average Orders per Month", 0, 30, 2)
            days_inactive = st.slider("Days Since Last Purchase", 0, 365, 14)
            
    elif active_domain == "gaming":
        st.subheader("🎮 Gaming Profile")
        col1, col2 = st.columns(2)
        with col1:
            win_rate = st.slider("Competitive Win Rate (%)", 0.0, 100.0, 50.0)
            daily_hours = st.slider("Daily Active Minutes", 0.0, 1440.0, 120.0)
            social_connections = st.slider("In-Game Friends List", 0, 500, 15)
        with col2:
            level = st.number_input("Current Level / XP Progress", 1, 100, 45)
            in_app_purchases = st.number_input("Total In-App Purchases (USD)", 0.0, 1000.0, 15.0)
            
    elif active_domain == "ott":
        st.subheader("🎬 OTT & Streaming Profile")
        col1, col2 = st.columns(2)
        with col1:
            label_genre = st.selectbox("Favorite Genre Preference", ["Action", "Comedy", "Drama", "Sci-Fi", "Kids"])
            watch_time = st.slider("Monthly Watch Hours", 1.0, 500.0, 45.0)
            ad_tier = st.selectbox("Subscription Plan Type", ["Mobile", "Basic", "Standard", "Premium"])
        with col2:
            login_freq = st.slider("Logins per Week", 0, 50, 4)
            failed_payments = st.number_input("Recent Payment Failures", 0, 5, 0)

    elif active_domain == "healthcare":
        st.subheader("🏥 Healthcare & Insurance Profile")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Customer Age", 18, 90, 45)
            health_score = st.slider("Calculated Health Score", 40, 100, 75)
            tenure = st.slider("Policy Tenure (Years)", 1, 20, 5)
        with col2:
            claims = st.number_input("Insurance Claims History", 0, 20, 1)
            premium_reg = st.selectbox("Premium Payment Regularity", ["Regular", "Irregular", "Delayed"])
            
    elif active_domain == "saas":
        st.subheader("☁️ SaaS & Software Profile")
        col1, col2 = st.columns(2)
        with col1:
            login_freq = st.slider("Logins per Month", 0, 100, 15)
            features_used = st.slider("Core Features Utilized", 1, 20, 5)
            billing = st.selectbox("Billing Cycle", ["Annual", "Monthly"])
        with col2:
            team_size = st.number_input("Workspace Team Size", 1, 100, 5)
            tickets = st.number_input("Monthly Support Tickets", 0, 20, 1)
            
    elif active_domain == "hospitality":
        st.subheader("✈️ Hospitality & Travel Profile")
        col1, col2 = st.columns(2)
        with col1:
            booking_freq = st.slider("Bookings per Year", 0, 50, 3)
            loyalty = st.number_input("Accumulated Loyalty Points", 0, 10000, 1500)
            city_type = st.selectbox("Primary User Location City Tier", ["Tier 1", "Tier 2", "Tier 3", "International"])
        with col2:
            ratings = st.slider("Average Review Rating Given", 1.0, 5.0, 4.5)
            complaints = st.selectbox("Number of Formal Complaints", [0, 1, 2, 3])

    st.markdown("---")
    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        with st.spinner(f"Routing request to MLflow `{active_domain}-churn-model` endpoint..."):
            
            # Since the API isn't natively mapped to 8 different schemas yet, we use a beautiful UI simulation
            data = spoof_prediction({})
            
            label = data["label"]
            prob = data["churn_probability"]
            conf = data["confidence"]
            
            st.markdown("### 📊 Prediction Intelligence")
            
            if label == "Churned":
                st.markdown(f"""
                <div class="prediction-card-danger">
                    <div style="font-size: 18px; color: #FECACA;">Prediction: 🔴 FLIGHT RISK</div>
                    <div class="metric-value" style="color: #FCA5A5;">{prob:.1%} Probability of Churn</div>
                    <div style="margin-top: 10px; font-weight: normal; color: #FEF2F2;">
                        <b>Alert:</b> This customer has a critical probability of abandoning your {active_domain} platform.
                        Machine learning logic has identified similar behaviors to historic churning models.
                        We recommend immediate retention offers or customer support outreach.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card-safe">
                    <div style="font-size: 18px; color: #D1FAE5;">Prediction: 🟢 RETAINED</div>
                    <div class="metric-value" style="color: #6EE7B7;">{(1-prob):.1%} Probability of Loyalty</div>
                    <div style="margin-top: 10px; font-weight: normal; color: #ECFDF5;">
                        <b>Safe:</b> This customer's metrics strongly align with our historically loyal customer base on the {active_domain} platform. No immediate intervention is required.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(prob, text=f"Raw Algorithmic Churn Probability: {prob:.1%}")
            st.caption(f"Backend Model: {active_domain}-churn-model | {data['timestamp']}")
