import numpy as np
import pandas as pd

# Define columns based on telco domain schema
columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Generate 20 realistic records
data = []
for i in range(20):
    tenure = np.random.randint(1, 72)
    monthly = round(np.random.uniform(20, 120), 2)
    data.append([
        np.random.choice(['Male', 'Female']),
        np.random.choice([0, 1]),
        np.random.choice(['Yes', 'No']),
        np.random.choice(['Yes', 'No']),
        tenure,
        np.random.choice(['Yes', 'No']),
        np.random.choice(['Yes', 'No', 'No phone service']),
        np.random.choice(['Fiber optic', 'DSL', 'No']),
        np.random.choice(['Yes', 'No', 'No internet service']),
        np.random.choice(['Yes', 'No', 'No internet service']),
        np.random.choice(['Yes', 'No', 'No internet service']),
        np.random.choice(['Yes', 'No', 'No internet service']),
        np.random.choice(['Yes', 'No', 'No internet service']),
        np.random.choice(['Yes', 'No', 'No internet service']),
        np.random.choice(['Month-to-month', 'One year', 'Two year']),
        np.random.choice(['Yes', 'No']),
        np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
        monthly,
        round(tenure * monthly, 2)
    ])

df = pd.DataFrame(data, columns=columns)
df.to_csv('c:/Users/Kush Chhunchha/Desktop/anti_projects/churn-prediction-mlops/data/reference/test_batch_telco.csv', index=False)
print("CSV generated successfully at data/reference/test_batch_telco.csv")
