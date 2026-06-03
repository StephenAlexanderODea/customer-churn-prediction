# Customer Churn Prediction Model

**Student:** Stephen Odea  
**Course:** IN503 – Introduction to Machine Learning  
**Unit:** Unit 10: Customer Churn Prediction Assignment  
**Model:** Logistic Regression with Scikit-learn  

---

## Project Description

This project implements a **logistic regression model** to predict customer churn for TelcoMax, a telecommunications company. The model identifies high-risk customers likely to leave within 6 months, enabling proactive retention strategies.

### Problem Statement
TelcoMax faces a **26.6% customer churn rate** among its 7,043 customer base. Understanding which customers are at risk allows the company to implement targeted retention programs before they leave.

### Dataset Overview
- **Source:** Telco Customer Churn Dataset (Microsoft Research)
- **Total Records:** 7,032 customers (after cleaning)
- **Features:** 20 (19 predictors + 1 target)
- **Target Variable:** Churn (Binary: Yes/No)
  - No Churn: 73.4% (5,163 customers)
  - Churn: 26.6% (1,869 customers)

### Key Characteristics
- **Customer Demographics:** Gender, senior citizen status, partner/dependent info
- **Services Information:** Phone service, internet type, online security, tech support, streaming services
- **Account Information:** Tenure, contract type, billing method, monthly/total charges

---

## Installation Instructions

### Prerequisites
- Python 3.7 or higher
- Google Colab account (recommended) or local Python environment
- ~2GB of available storage

### Step 1: Clone or Download the Project

```bash
# If using Google Colab, upload the notebook file
# File → Upload notebook → Select Odea_Stephen_IN503_Unit10_Assignment.ipynb
```

### Step 2: Install Required Libraries

```bash
# Install kagglehub for dataset access
pip install kagglehub

# Install core dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 3: Run in Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload the notebook: `Odea_Stephen_IN503_Unit10_Assignment.ipynb`
3. Run all cells in order: **Runtime → Run all**
4. The dataset will auto-download on first run (~800MB)

### Step 4: Or Run Locally

```bash
# From terminal in your project directory
jupyter notebook Odea_Stephen_IN503_Unit10_Assignment.ipynb
```

---

## Usage Examples

### Loading and Using the Trained Model

```python
import joblib
import pandas as pd

# Load the pre-trained model pipeline
model_pipeline = joblib.load('customer_churn_model.pkl')

# Create a new customer profile for prediction
new_customer = {
    'gender': ['Male'],
    'SeniorCitizen': [0],
    'Partner': ['No'],
    'Dependents': ['No'],
    'tenure': [6],  # 6 months
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'Contract': ['Month-to-month'],
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [75.00],
    'TotalCharges': [450.00]
}

# Convert to DataFrame
customer_df = pd.DataFrame(new_customer)

# Make prediction
prediction = model_pipeline.predict(customer_df)
churn_prediction = 'Yes' if prediction[0] == 1 else 'No'

print(f"Churn Prediction: {churn_prediction}")
```

### Making Batch Predictions

```python
# Load multiple customers from CSV
customers_df = pd.read_csv('customers.csv')

# Get predictions for all customers
predictions = model_pipeline.predict(customers_df)

# Add predictions to dataframe
customers_df['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]

# Save results
customers_df.to_csv('churn_predictions.csv', index=False)
```

### Identifying High-Risk Customers

```python
# Get prediction probabilities
probabilities = model_pipeline.predict_proba(customers_df)

# Extract churn probability (class 1)
customers_df['Churn_Probability'] = probabilities[:, 1]

# Filter high-risk customers (>60% churn probability)
high_risk = customers_df[customers_df['Churn_Probability'] > 0.60]

print(f"High-risk customers: {len(high_risk)} out of {len(customers_df)}")
```

---

## Key Findings Summary

### Model Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **80.38%** |
| **Precision (Churn)** | 0.65 |
| **Recall (Churn)** | 0.57 |
| **F1-Score (Churn)** | 0.61 |
| **Test Set Size** | 1,407 customers |

**Confusion Matrix:**
```
              Predicted No Churn    Predicted Churn
Actually No         917                  116
Actually Yes        160                  214
```

### Top 10 Most Impactful Features

| Rank | Feature | Coefficient | Impact |
|------|---------|-------------|--------|
| 1 | Tenure | -1.349 | ⬇️ Strong negative (longer tenure = less churn) |
| 2 | Two-Year Contract | -0.786 | ⬇️ Strong negative (long contracts = less churn) |
| 3 | Total Charges | +0.641 | ⬆️ Positive (higher charges correlate with churn) |
| 4 | DSL Internet | -0.615 | ⬇️ Negative (DSL users churn less) |
| 5 | Month-to-Month Contract | +0.601 | ⬆️ Positive (high churn risk) |
| 6 | Fiber Optic Internet | +0.559 | ⬆️ Positive (high churn risk) |

### Critical Business Insights

#### 1. **Contract Type is Critical**
- **Month-to-month contracts:** 42% churn rate (highest risk)
- **Two-year contracts:** 3% churn rate (lowest risk)
- **Recommendation:** Incentivize longer-term contracts for new customers

#### 2. **Tenure is the Strongest Predictor**
- Customers with <6 months tenure: 47% churn rate
- Customers with >12 months tenure: 15% churn rate
- **Recommendation:** Implement strong onboarding and support in first 6 months

#### 3. **Internet Service Type Matters**
- **Fiber optic:** 42% churn (service quality issues?)
- **Cable:** 20% churn
- **DSL:** 19% churn
- **Recommendation:** Investigate fiber optic service quality and pricing

#### 4. **Monthly Charges Correlate with Churn**
- Customers paying >$100/month: 40% churn
- Customers paying <$50/month: 15% churn
- **Recommendation:** Review pricing strategy; consider tiered pricing

#### 5. **Data Quality Issues Identified**
- 11 customers had corrupted/missing TotalCharges values
- Successfully handled with null value removal
- No accuracy impact after cleaning

---

## Business Recommendations

### Immediate Actions (0-3 months)
1. **Deploy Model for Scoring**
   - Score all customers for churn risk
   - Flag 1,000+ high-risk customers

2. **Launch Retention Program**
   - Contact high-risk customers with personalized offers
   - Focus on month-to-month contract holders
   - Target customers with fiber optic service

3. **Optimize Onboarding**
   - Implement intensive support in first 6 months
   - Quick-start guides and check-in calls
   - Early issue resolution

### Medium-term Actions (3-12 months)
1. **Contract Strategy**
   - Offer discounts for switching to 1-2 year contracts
   - Include commitment incentives at sign-up

2. **Service Quality**
   - Audit fiber optic service delivery
   - Compare with competitors' offerings
   - Implement service improvements

3. **Pricing Review**
   - Analyze price sensitivity by segment
   - Create value bundles to justify higher charges
   - Test promotional pricing for retention

---

## Model Architecture

### Data Preprocessing Pipeline
```
Raw Data (7,043 rows × 21 features)
    ↓
[Handling Missing Values]
    ↓
[Remove Duplicates & Invalid Data]
    ↓
[Remove ID Columns]
    ↓
Cleaned Data (7,032 rows × 20 features)
    ↓
[Train-Test Split: 80-20]
    ↓
[Standardize Numerical Features]
[One-Hot Encode Categorical Features]
    ↓
[Logistic Regression Model]
    ↓
Predictions & Evaluation
```

### Model Details
- **Algorithm:** Logistic Regression
- **Solver:** liblinear
- **Regularization:** L2 (default)
- **Train/Test Split:** 80/20 with stratification
- **Random State:** 42 (reproducibility)

### Feature Engineering
- **Numerical Features (3):** tenure, MonthlyCharges, TotalCharges
  - Standardized using StandardScaler
  
- **Categorical Features (17):** gender, SeniorCitizen, Partner, etc.
  - One-hot encoded using OneHotEncoder
  - Handles unknown categories gracefully

---

## Dependencies List

### Core ML Libraries
```
scikit-learn >= 0.22      # Machine learning & preprocessing
pandas >= 1.0             # Data manipulation
numpy >= 1.18             # Numerical computing
```

### Data Access
```
kagglehub >= 0.2          # Download datasets from Kaggle
```

### Visualization
```
matplotlib >= 3.0         # Plotting and visualization
seaborn >= 0.10           # Statistical visualization
```

### Model Persistence
```
joblib >= 1.0             # Save/load trained models
```

### Additional Tools
```
tensorflow >= 2.0         # (Optional) for future deep learning work
```

### Installation

```bash
# Install all dependencies at once
pip install scikit-learn pandas numpy kagglehub matplotlib seaborn joblib

# Or use requirements.txt
pip install -r requirements.txt
```

**Requirements.txt Contents:**
```
scikit-learn>=0.22
pandas>=1.0
numpy>=1.18
kagglehub>=0.2
matplotlib>=3.0
seaborn>=0.10
joblib>=1.0
```