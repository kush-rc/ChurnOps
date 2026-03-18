"""
Massive Domain Synthetic Data Generator
=======================================
Generates 1,000,000 rows of highly realistic data for 8 major industry domains.
1. Telecom
2. Banking & Finance
3. E-commerce / Retail
4. OTT / Streaming
5. Healthcare / Insurance
6. Gaming
7. SaaS / Software
8. Hospitality / Travel
"""

import os
import time

import numpy as np
import pandas as pd


def generate_banking(target_rows, out_path):
    print(f"🏦 Generating Banking Data ({target_rows:,} rows)...")
    np.random.seed(42)
    # Features: Account balance, transaction frequency, loan status, credit score, product count
    id_col = [f"BNK_{i:07d}" for i in range(target_rows)]
    balance = np.random.lognormal(mean=10, sigma=1.5, size=target_rows) # highly skewed
    tx_freq = np.random.poisson(lam=15, size=target_rows)
    loan = np.random.choice(["Yes", "No"], size=target_rows, p=[0.4, 0.6])
    credit_score = np.random.normal(loc=650, scale=100, size=target_rows).clip(300, 850).astype(int)
    product_count = np.random.randint(1, 5, size=target_rows)

    # Churn Logic: Low balance, low tx frequency, low credit score
    churn_prob = 0.1
    churn_prob += np.where(balance < 1000, 0.2, 0)
    churn_prob += np.where(tx_freq < 5, 0.2, 0)
    churn_prob += np.where(credit_score < 500, 0.1, 0)
    churn_prob -= (product_count * 0.05)
    churn_prob = np.clip(churn_prob, 0, 1)

    churn = (np.random.rand(target_rows) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": id_col, "AccountBalance": np.round(balance, 2),
        "TransactionFrequency": tx_freq, "LoanStatus": loan,
        "CreditScore": credit_score, "ProductCount": product_count,
        "Churn": np.where(churn==1, "Yes", "No")
    })
    df.to_csv(out_path, index=False)

def generate_ecommerce(target_rows, out_path):
    print(f"🛒 Generating E-commerce Data ({target_rows:,} rows)...")
    np.random.seed(42)
    id_col = [f"ECO_{i:07d}" for i in range(target_rows)]
    order_freq = np.random.poisson(lam=2, size=target_rows)
    days_since_last = np.random.randint(1, 365, size=target_rows)
    returns = np.random.randint(0, 10, size=target_rows)
    discount_usage = np.random.choice(["High", "Medium", "Low"], size=target_rows, p=[0.2, 0.5, 0.3])
    app_activity = np.random.randint(0, 100, size=target_rows)

    churn_prob = (days_since_last / 365.0) * 0.5 + (returns * 0.05)
    churn_prob -= (order_freq * 0.05) + (app_activity * 0.005)
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = (np.random.rand(target_rows) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": id_col, "OrderFrequency": order_freq, "DaysSinceLastPurchase": days_since_last,
        "Returns": returns, "DiscountUsage": discount_usage, "AppActivityScore": app_activity,
        "Churn": np.where(churn==1, "Yes", "No")
    })
    df.to_csv(out_path, index=False)

def generate_ott(target_rows, out_path):
    print(f"🎬 Generating OTT/Streaming Data ({target_rows:,} rows)...")
    np.random.seed(42)
    id_col = [f"OTT_{i:07d}" for i in range(target_rows)]
    watch_hours = np.random.exponential(scale=20, size=target_rows)
    genre = np.random.choice(["Action", "Comedy", "Drama", "Sci-Fi", "Kids"], size=target_rows)
    login_freq = np.random.poisson(lam=10, size=target_rows)
    payment_fails = np.random.randint(0, 3, size=target_rows)
    plan = np.random.choice(["Mobile", "Basic", "Standard", "Premium"], size=target_rows)

    churn_prob = 0.2
    churn_prob += (payment_fails * 0.2)
    churn_prob -= (watch_hours * 0.01)
    churn_prob -= (login_freq * 0.02)
    churn_prob += np.where(plan == "Mobile", 0.1, 0)
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = (np.random.rand(target_rows) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": id_col, "WatchHours": np.round(watch_hours, 1), "GenrePreference": genre,
        "LoginFrequency": login_freq, "PaymentFailures": payment_fails, "PlanType": plan,
        "Churn": np.where(churn==1, "Yes", "No")
    })
    df.to_csv(out_path, index=False)

def generate_healthcare(target_rows, out_path):
    print(f"🏥 Generating Healthcare Data ({target_rows:,} rows)...")
    np.random.seed(42)
    id_col = [f"HLT_{i:07d}" for i in range(target_rows)]
    claims = np.random.poisson(lam=1, size=target_rows)
    premium_reg = np.random.choice(["Regular", "Irregular", "Delayed"], size=target_rows, p=[0.7, 0.2, 0.1])
    age = np.random.randint(18, 90, size=target_rows)
    health_score = np.random.randint(40, 100, size=target_rows)
    tenure = np.random.randint(1, 20, size=target_rows)

    churn_prob = 0.15
    churn_prob += np.where(premium_reg == "Delayed", 0.3, 0)
    churn_prob += np.where(claims > 3, 0.1, 0)
    churn_prob -= (tenure * 0.02)
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = (np.random.rand(target_rows) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": id_col, "ClaimsHistory": claims, "PremiumRegularity": premium_reg,
        "Age": age, "HealthScore": health_score, "TenureYears": tenure,
        "Churn": np.where(churn==1, "Yes", "No")
    })
    df.to_csv(out_path, index=False)

def generate_gaming(target_rows, out_path):
    print(f"🎮 Generating Gaming Data ({target_rows:,} rows)...")
    np.random.seed(42)
    id_col = [f"GAM_{i:07d}" for i in range(target_rows)]
    daily_time = np.random.exponential(scale=45, size=target_rows) # minutes
    iap_amount = np.random.lognormal(mean=2, sigma=1, size=target_rows)
    win_rate = np.random.uniform(0.1, 0.9, size=target_rows)
    social = np.random.randint(0, 50, size=target_rows)
    level = np.random.randint(1, 100, size=target_rows)

    churn_prob = 0.4
    churn_prob -= (daily_time * 0.005)
    churn_prob -= (iap_amount * 0.001)
    churn_prob -= (social * 0.01)
    churn_prob += np.where(win_rate < 0.3, 0.2, 0) # frustration churn
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = (np.random.rand(target_rows) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": id_col, "DailyActiveMinutes": np.round(daily_time, 1),
        "InAppPurchasesUSD": np.round(iap_amount, 2), "WinRate": np.round(win_rate, 2),
        "SocialConnections": social, "LevelProgress": level,
        "Churn": np.where(churn==1, "Yes", "No")
    })
    df.to_csv(out_path, index=False)

def generate_saas(target_rows, out_path):
    print(f"🏋️ Generating SaaS Data ({target_rows:,} rows)...")
    np.random.seed(42)
    id_col = [f"SAS_{i:07d}" for i in range(target_rows)]
    login_freq = np.random.poisson(lam=15, size=target_rows)
    features_used = np.random.randint(1, 20, size=target_rows)
    tickets = np.random.poisson(lam=0.5, size=target_rows)
    team_size = np.random.randint(1, 100, size=target_rows)
    billing = np.random.choice(["Annual", "Monthly"], size=target_rows)

    churn_prob = 0.2
    churn_prob -= (login_freq * 0.01)
    churn_prob -= (features_used * 0.02)
    churn_prob += (tickets * 0.05)
    churn_prob += np.where(billing == "Monthly", 0.1, -0.1)
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = (np.random.rand(target_rows) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": id_col, "LoginFrequency": login_freq, "FeaturesUsed": features_used,
        "SupportTickets": tickets, "TeamSize": team_size, "BillingCycle": billing,
        "Churn": np.where(churn==1, "Yes", "No")
    })
    df.to_csv(out_path, index=False)

def generate_hospitality(target_rows, out_path):
    print(f"🏨 Generating Hospitality/Travel Data ({target_rows:,} rows)...")
    np.random.seed(42)
    id_col = [f"HOS_{i:07d}" for i in range(target_rows)]
    booking_freq = np.random.poisson(lam=3, size=target_rows)
    ratings = np.random.uniform(1.0, 5.0, size=target_rows)
    complaints = np.random.choice([0, 1, 2, 3], size=target_rows, p=[0.8, 0.1, 0.05, 0.05])
    loyalty = np.random.randint(0, 10000, size=target_rows)
    city = np.random.choice(["Tier 1", "Tier 2", "Tier 3", "International"], size=target_rows)

    churn_prob = 0.3
    churn_prob -= (booking_freq * 0.05)
    churn_prob += np.where(ratings < 3.0, 0.2, -0.1)
    churn_prob += (complaints * 0.2)
    churn_prob -= (loyalty * 0.00005)
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = (np.random.rand(target_rows) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": id_col, "BookingFrequency": booking_freq, "AverageRating": np.round(ratings, 1),
        "Complaints": complaints, "LoyaltyPoints": loyalty, "CityType": city,
        "Churn": np.where(churn==1, "Yes", "No")
    })
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    t0 = time.time()
    base_dir = r"c:\Users\Kush Chhunchha\Desktop\anti_projects\churn-prediction-mlops\data\raw"
    os.makedirs(base_dir, exist_ok=True)

    ROWS = 1000000 # 1M rows per industry

    # We already have Telco (telco_churn_massive.csv) so we skip it to save time, but generate the other 7
    generate_banking(ROWS, os.path.join(base_dir, "banking_churn_massive.csv"))
    generate_ecommerce(ROWS, os.path.join(base_dir, "ecommerce_churn_massive.csv"))
    generate_ott(ROWS, os.path.join(base_dir, "ott_churn_massive.csv"))
    generate_healthcare(ROWS, os.path.join(base_dir, "healthcare_churn_massive.csv"))
    generate_gaming(ROWS, os.path.join(base_dir, "gaming_churn_massive.csv"))
    generate_saas(ROWS, os.path.join(base_dir, "saas_churn_massive.csv"))
    generate_hospitality(ROWS, os.path.join(base_dir, "hospitality_churn_massive.csv"))

    print(f"\n🎉 Successfully generated 7 Million rows of highly realistic synthetic enterprise data in {time.time()-t0:.1f}s!")
