"""
Exploratory Data Analysis Script for Insurance Claim Prediction
Loads the dataset and performs basic EDA
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# Load the dataset
data_path = os.path.join('v2-advance_model', 'data', 'raw', 'car_insurance_claim.csv')
df = pd.read_csv(data_path)

# 1. Print dataset shape
print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"\nDataset Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# 2. Print column names
print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)
print(df.columns.tolist())

# 3. Print data types
print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)
print(df.dtypes)

# 4. Check missing values
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
missing_values = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

# Convert CLM_AMT to numeric (remove $ and , symbols)
df['CLM_AMT'] = df['CLM_AMT'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
df['CLM_AMT'] = pd.to_numeric(df['CLM_AMT'], errors='coerce')

# 5. Analyze target column: CLM_AMT
print("\n" + "=" * 50)
print("TARGET COLUMN ANALYSIS: CLM_AMT")
print("=" * 50)
print(f"\nBasic Statistics:")
print(df['CLM_AMT'].describe())

# 6. Print skewness of CLM_AMT
skewness = stats.skew(df['CLM_AMT'].dropna())
print(f"\nSkewness of CLM_AMT: {skewness:.4f}")

# 7. Plot histogram of CLM_AMT
print("\n" + "=" * 50)
print("GENERATING PLOTS")
print("=" * 50)

import numpy as np

plt.figure(figsize=(12, 5))

# Histogram of CLM_AMT
plt.subplot(1, 2, 1)
plt.hist(df['CLM_AMT'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.title('Histogram of CLM_AMT')
plt.xlabel('Claim Amount')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.5)

# Histogram of log(CLM_AMT + 1)
plt.subplot(1, 2, 2)
log_clm_amt = np.log(df['CLM_AMT'] + 1)
plt.hist(log_clm_amt, bins=50, edgecolor='black', alpha=0.7, color='coral')
plt.title('Histogram of log(CLM_AMT + 1)')
plt.xlabel('log(Claim Amount + 1)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.5)

plt.tight_layout()
plt.savefig('v2-advance_model/outputs/claim_amount_histograms.png', dpi=100)
print("Plots saved to: outputs/claim_amount_histograms.png")
plt.close()

print("\nEDA Complete!")