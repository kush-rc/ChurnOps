const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const DOMAINS = {
  telco: { label: 'Telecom', key: 'telco' },
  banking: { label: 'Banking & Finance', key: 'banking' },
  ecommerce: { label: 'E-commerce / Retail', key: 'ecommerce' },
  ott: { label: 'OTT / Streaming', key: 'ott' },
  healthcare: { label: 'Healthcare / Insurance', key: 'healthcare' },
  gaming: { label: 'Gaming', key: 'gaming' },
  saas: { label: 'SaaS / Software', key: 'saas' },
  hospitality: { label: 'Hospitality / Travel', key: 'hospitality' },
};

export async function predictChurn(domain, features) {
  const res = await fetch(`${API_BASE}/api/v1/predict?domain=${domain}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(features),
  });
  if (!res.ok) throw new Error(`Prediction failed: ${res.statusText}`);
  return res.json();
}

export async function explainPrediction(domain, features) {
  const res = await fetch(`${API_BASE}/api/v1/explain?domain=${domain}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(features),
  });
  if (!res.ok) throw new Error(`Explanation failed: ${res.statusText}`);
  return res.json();
}

export async function getHealthStatus() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) return { status: 'offline', model_loaded: false };
    return res.json();
  } catch {
    return { status: 'offline', model_loaded: false };
  }
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/api/v1/model/info`);
  if (!res.ok) throw new Error('Failed to fetch model info');
  return res.json();
}

// Domain-specific form field definitions
export const DOMAIN_FIELDS = {
  telco: {
    title: 'Telecom Profile',
    fields: [
      { name: 'gender', label: 'Gender', type: 'select', options: ['Male', 'Female'], col: 1 },
      { name: 'SeniorCitizen', label: 'Senior Citizen', type: 'select', options: [0, 1], col: 1 },
      { name: 'Partner', label: 'Partner', type: 'select', options: ['Yes', 'No'], col: 1 },
      { name: 'Dependents', label: 'Dependents', type: 'select', options: ['Yes', 'No'], col: 1 },
      { name: 'tenure', label: 'Tenure (months)', type: 'slider', min: 0, max: 72, default: 12, col: 2 },
      { name: 'MonthlyCharges', label: 'Monthly Charges ($)', type: 'number', min: 0, max: 200, default: 70, step: 0.01, col: 2 },
      { name: 'Contract', label: 'Contract', type: 'select', options: ['Month-to-month', 'One year', 'Two year'], col: 2 },
      { name: 'PaymentMethod', label: 'Payment Method', type: 'select', options: ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], col: 2 },
      { name: 'PaperlessBilling', label: 'Paperless Billing', type: 'select', options: ['Yes', 'No'], col: 2 },
      { name: 'PhoneService', label: 'Phone Service', type: 'select', options: ['Yes', 'No'], col: 3 },
      { name: 'MultipleLines', label: 'Multiple Lines', type: 'select', options: ['Yes', 'No', 'No phone service'], col: 3 },
      { name: 'InternetService', label: 'Internet Service', type: 'select', options: ['Fiber optic', 'DSL', 'No'], col: 3 },
      { name: 'OnlineSecurity', label: 'Online Security', type: 'select', options: ['Yes', 'No', 'No internet service'], col: 3 },
      { name: 'OnlineBackup', label: 'Online Backup', type: 'select', options: ['Yes', 'No', 'No internet service'], col: 3 },
      { name: 'DeviceProtection', label: 'Device Protection', type: 'select', options: ['Yes', 'No', 'No internet service'], col: 3 },
      { name: 'TechSupport', label: 'Tech Support', type: 'select', options: ['Yes', 'No', 'No internet service'], col: 3 },
      { name: 'StreamingTV', label: 'Streaming TV', type: 'select', options: ['Yes', 'No', 'No internet service'], col: 3 },
      { name: 'StreamingMovies', label: 'Streaming Movies', type: 'select', options: ['Yes', 'No', 'No internet service'], col: 3 },
    ],
  },
  banking: {
    title: 'Banking & Finance Profile',
    fields: [
      { name: 'CreditScore', label: 'Credit Score', type: 'slider', min: 300, max: 850, default: 650, col: 1 },
      { name: 'AccountBalance', label: 'Account Balance ($)', type: 'number', min: 0, max: 500000, default: 15000, col: 1 },
      { name: 'LoanStatus', label: 'Active Loan Status', type: 'select', options: ['Yes', 'No'], col: 1 },
      { name: 'TransactionFrequency', label: 'Monthly Transactions', type: 'slider', min: 0, max: 100, default: 15, col: 2 },
      { name: 'ProductCount', label: 'Number of Products', type: 'number', min: 1, max: 5, default: 2, col: 2 },
    ],
  },
  ecommerce: {
    title: 'E-Commerce Profile',
    fields: [
      { name: 'AppActivityScore', label: 'App Activity Score', type: 'slider', min: 0, max: 100, default: 45, col: 1 },
      { name: 'Returns', label: 'Return Rate (%)', type: 'slider', min: 0, max: 100, default: 5, step: 0.1, col: 1 },
      { name: 'DiscountUsage', label: 'Discount Dependence', type: 'select', options: ['High', 'Medium', 'Low'], col: 1 },
      { name: 'OrderFrequency', label: 'Orders / Month', type: 'slider', min: 0, max: 30, default: 2, col: 2 },
      { name: 'DaysSinceLastPurchase', label: 'Days Since Last Purchase', type: 'slider', min: 0, max: 365, default: 14, col: 2 },
    ],
  },
  gaming: {
    title: 'Gaming Profile',
    fields: [
      { name: 'WinRate', label: 'Win Rate (%)', type: 'slider', min: 0, max: 100, default: 50, step: 0.1, col: 1 },
      { name: 'DailyActiveMinutes', label: 'Daily Active Minutes', type: 'slider', min: 0, max: 1440, default: 120, col: 1 },
      { name: 'SocialConnections', label: 'In-Game Friends', type: 'slider', min: 0, max: 500, default: 15, col: 1 },
      { name: 'LevelProgress', label: 'Level / XP', type: 'number', min: 1, max: 100, default: 45, col: 2 },
      { name: 'InAppPurchasesUSD', label: 'In-App Purchases ($)', type: 'number', min: 0, max: 1000, default: 15, step: 0.01, col: 2 },
    ],
  },
  ott: {
    title: 'OTT & Streaming Profile',
    fields: [
      { name: 'GenrePreference', label: 'Favorite Genre', type: 'select', options: ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Kids'], col: 1 },
      { name: 'WatchHours', label: 'Monthly Watch Hours', type: 'slider', min: 1, max: 500, default: 45, step: 0.1, col: 1 },
      { name: 'PlanType', label: 'Subscription Plan', type: 'select', options: ['Mobile', 'Basic', 'Standard', 'Premium'], col: 1 },
      { name: 'LoginFrequency', label: 'Logins / Week', type: 'slider', min: 0, max: 50, default: 4, col: 2 },
      { name: 'PaymentFailures', label: 'Payment Failures', type: 'number', min: 0, max: 5, default: 0, col: 2 },
    ],
  },
  healthcare: {
    title: 'Healthcare & Insurance Profile',
    fields: [
      { name: 'Age', label: 'Customer Age', type: 'slider', min: 18, max: 90, default: 45, col: 1 },
      { name: 'HealthScore', label: 'Health Score', type: 'slider', min: 40, max: 100, default: 75, col: 1 },
      { name: 'TenureYears', label: 'Policy Tenure (Years)', type: 'slider', min: 1, max: 20, default: 5, col: 1 },
      { name: 'ClaimsHistory', label: 'Claims History', type: 'number', min: 0, max: 20, default: 1, col: 2 },
      { name: 'PremiumRegularity', label: 'Premium Regularity', type: 'select', options: ['Regular', 'Irregular', 'Delayed'], col: 2 },
    ],
  },
  saas: {
    title: 'SaaS & Software Profile',
    fields: [
      { name: 'LoginFrequency', label: 'Logins / Month', type: 'slider', min: 0, max: 100, default: 15, col: 1 },
      { name: 'FeaturesUsed', label: 'Features Utilized', type: 'slider', min: 1, max: 20, default: 5, col: 1 },
      { name: 'BillingCycle', label: 'Billing Cycle', type: 'select', options: ['Annual', 'Monthly'], col: 1 },
      { name: 'TeamSize', label: 'Team Size', type: 'number', min: 1, max: 100, default: 5, col: 2 },
      { name: 'SupportTickets', label: 'Support Tickets / Month', type: 'number', min: 0, max: 20, default: 1, col: 2 },
    ],
  },
  hospitality: {
    title: 'Hospitality & Travel Profile',
    fields: [
      { name: 'BookingFrequency', label: 'Bookings / Year', type: 'slider', min: 0, max: 50, default: 3, col: 1 },
      { name: 'LoyaltyPoints', label: 'Loyalty Points', type: 'number', min: 0, max: 10000, default: 1500, col: 1 },
      { name: 'CityType', label: 'Location Tier', type: 'select', options: ['Tier 1', 'Tier 2', 'Tier 3', 'International'], col: 1 },
      { name: 'AverageRating', label: 'Avg Review Rating', type: 'slider', min: 1, max: 5, default: 4.5, step: 0.1, col: 2 },
      { name: 'Complaints', label: 'Formal Complaints', type: 'select', options: [0, 1, 2, 3], col: 2 },
    ],
  },
};

export async function uploadBatchCSV(domain, file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/v1/predict/upload?domain=${domain}`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Upload failed: ${res.statusText}`);
  }
  return res.json();
}
