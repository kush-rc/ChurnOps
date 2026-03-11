# 🧮 Synthetic Big Data Generation Explained
This document explains the mathematical logic and programming techniques used to synthetically construct the **8 Million rows of MLOps data** (1 million rows per industry across 8 domains) found in `scripts/generate_8_domains.py`.

Creating highly realistic synthetic customer churn data is critical when building an Enterprise Architecture for a portfolio, because real-world company data (like Chase Bank or Netflix) at a million-row scale is strictly highly confidential.

## 🛠️ How We Synthesized Data (The Math)
Instead of just randomizing numbers blindly (which machine learning algorithms can easily spot and overfit against), we used **probabilistic distributions** from the `numpy` library to weave natural, human-like behavior into the data.

### 1. The Normal Distribution (`np.random.normal`)
We used this for features that naturally center around an average:
- **Example**: `CreditScore = np.random.normal(loc=650, scale=100)`. This ensures most people hover around a 650 credit score, with very few at 300 and very few at 850.

### 2. The Log-Normal Distribution (`np.random.lognormal`)
We used this for "highly skewed" financial features.
- **Example**: `Balance = np.random.lognormal(mean=10, sigma=1.5)`. In real life, most people have lower account balances ($1,000-$5,000), while a tiny 1% of the population act as "Whales" with $500,000+ balances. The Log-Normal distribution mathematically replicates wealth disparity perfectly.

### 3. The Poisson Distribution (`np.random.poisson`)
Used for counting "events in a timeframe".
- **Example**: `LoginFrequency = np.random.poisson(lam=15)`. This accurately simulates that a user normally logs in 15 times a month, rather than generating an impossible perfectly flat curve.

---

## 🗂️ The 8 Domain Schemas & Churn Logic

Our generator creates a "Base Probability" of churn per user, and then *mathematically adjusts that probability based on their specific features*. For instance, if a user has many Complaints, their Base Probability of churn goes up dramatically. We then flip a weighted coin using `np.random.rand()` against that final probability to definitively decide if they Churn or Not.

Here is exactly what we created for each domain:

### 1. 🏦 Banking & Finance (`banking`)
* **Core Logic**: People leave banks when they have low money, low engagement, and poor credit.
* **Features**:
  - `AccountBalance`: Log-Normal distribution (skewed right).
  - `TransactionFrequency`: Poisson distribution (avg 15/month).
  - `CreditScore`: Normal distribution bounded between 300 and 850.
* **Churn Trigger**: If Balance < $1000 (+20% churn risk). If Transacting < 5 times/month (+20% risk).

### 2. 🛒 E-Commerce & Retail (`ecommerce`)
* **Core Logic**: People stop shopping if they haven't bought recently, or if they keep returning items.
* **Features**:
  - `DaysSinceLastPurchase`: A strict integer count.
  - `Returns`: Highly penalized action.
  - `OrderFrequency`: Average buys per month.
* **Churn Trigger**: The longer `DaysSinceLastPurchase` stretches to 365, the more the Churn probability scales toward 50%. A high `Return` rate heavily accelerates churn.

### 3. 🎬 OTT & Streaming (`ott`)
* **Core Logic**: People cancel streaming services if they aren't watching, or if their Credit Card fails.
* **Features**:
  - `WatchHours`: Simulated via an `Exponential` distribution (most people watch average amounts; very few binge 24/7).
  - `PaymentFailures`: Triggered by strict integer bounds.
* **Churn Trigger**: Every Payment Failure adds a massive 20% to the churn risk. High watch-hours heavily deduct from churn risk.

### 4. 🏥 Healthcare & Insurance (`healthcare`)
* **Core Logic**: People lapse insurance policies if they don't pay premiums regularly, or if they file excessive claims proving friction.
* **Features**:
  - `PremiumRegularity`: Categorical ("Regular", "Irregular", "Delayed").
  - `ClaimsHistory`: Poisson count of medical/accident claims.
* **Churn Trigger**: A "Delayed" premium status instantly adds 30% to the churn probability. Long `Tenure` (years as a customer) mathematically protects against churn.

### 5. 🎮 Gaming (`gaming`)
* **Core Logic**: Players rage-quit if they lose too much or stop playing daily.
* **Features**:
  - `WinRate`: Uniform percentage between 10% and 90%.
  - `DailyActiveMinutes`: Exponential curve favoring 45-minute average sessions.
  - `SocialConnections`: In-game friends list.
* **Churn Trigger**: A Win Rate below 30% instantly triggers a 20% "Frustration Churn" spike. High Social Connections (friends) deeply protect the user from uninstalling.

### 6. ☁️ SaaS & Software (`saas`)
* **Core Logic**: B2B teams cancel contracts if they don't use the core features or file tons of support tickets.
* **Features**:
  - `SupportTickets`: Heavily penalized.
  - `FeaturesUsed`: Deep engagement metric.
  - `BillingCycle`: Monthly vs Annual.
* **Churn Trigger**: Every support ticket increases churn risk by 5%. Users on "Monthly" billing are inherently 10% more likely to churn than those locked into "Annual" contracts.

### 7. ✈️ Hospitality & Travel (`hospitality`)
* **Core Logic**: Brand loyalty drops if ratings are bad or frequency dips.
* **Features**:
  - `AverageRating`: 1.0 to 5.0 scale.
  - `LoyaltyPoints`: Thousands of accumulated points.
  - `Complaints`: Customer service friction.
* **Churn Trigger**: Leaving a rating under 3.0 spikes churn risk by 20%. Any formal complaint spikes risk by 20%. Having high `LoyaltyPoints` acts as an anchor to keep them retained.

### 8. 📱 Telecom (`telco`)
*(Generated previously using noise-injection over an existing IBM Base Dataset).*
* **Core Logic**: Based on Tenure and Monthly Contracts. High monthly charges with low tenure accelerates churn massively.

---
## ✨ Why This Proves Enterprise Competence

By reading the `generate_8_domains.py` script closely alongside this file, you can prove to any technical recruiter that you understand:
1. **Feature Engineering**: You understand what inputs natively correlate to business outcomes (e.g., Win Rates -> Gaming Churn).
2. **Computational Scale**: You dynamically allocate arrays across 1,000,000 elements instantly utilizing vectorized Numpy operations, heavily bypassing slow Python `For-Loops`.
3. **Data Quality Simulation**: You didn't just type `random.randint(...)`; you used correct Log-Normal distributions to accurately reflect financial wealth curves.
