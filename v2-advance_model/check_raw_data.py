"""
Check raw data distribution to understand the zero-inflated target problem.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load raw data
df = pd.read_csv('v2-advance_model/data/raw/car_insurance_claim.csv')

# Clean the CLM_AMT column (it has $ and , in it)
df['CLM_AMT_CLEAN'] = df['CLM_AMT'].str.replace('$','', regex=False).str.replace(',','', regex=False).astype(float)

print("=" * 70)
print("RAW DATA ANALYSIS")
print("=" * 70)
print(f"Total rows: {len(df)}")
print(f"Zero claims: {(df['CLM_AMT_CLEAN']==0).sum()} ({(df['CLM_AMT_CLEAN']==0).sum()/len(df)*100:.1f}%)")
print(f"Non-zero claims: {(df['CLM_AMT_CLEAN']>0).sum()} ({(df['CLM_AMT_CLEAN']>0).sum()/len(df)*100:.1f}%)")
print()
print(f"Min (non-zero): ${df[df['CLM_AMT_CLEAN']>0]['CLM_AMT_CLEAN'].min():.2f}")
print(f"Max: ${df['CLM_AMT_CLEAN'].max():.2f}")
print(f"Mean (all): ${df['CLM_AMT_CLEAN'].mean():.2f}")
print(f"Mean (non-zero): ${df[df['CLM_AMT_CLEAN']>0]['CLM_AMT_CLEAN'].mean():.2f}")
print(f"Median (all): ${df['CLM_AMT_CLEAN'].median():.2f}")
print(f"Median (non-zero): ${df[df['CLM_AMT_CLEAN']>0]['CLM_AMT_CLEAN'].median():.2f}")
print()

# Check log1p of non-zero values
print("Log1p transformation analysis:")
non_zero = df[df['CLM_AMT_CLEAN'] > 0]['CLM_AMT_CLEAN']
log_values = np.log1p(non_zero)
print(f"log1p(min non-zero): {log_values.min():.4f}")
print(f"log1p(max): {log_values.max():.4f}")
print(f"log1p(mean non-zero): {log_values.mean():.4f}")
print()

# Check what happens with zeros
print("What happens with zeros:")
print(f"log1p(0) = {np.log1p(0):.4f}")
print(f"expm1(0) = {np.expm1(0):.4f}")
print(f"expm1(log1p(0)) = {np.expm1(np.log1p(0)):.4f}")
print()

# Distribution of non-zero claims
print("Distribution of non-zero claim amounts:")
claims = df[df['CLM_AMT_CLEAN'] > 0]['CLM_AMT_CLEAN']
bins = [0, 100, 500, 1000, 5000, 10000, 25000, 50000, float('inf')]
bin_labels = ['$0-100', '$100-500', '$500-1K', '$1K-5K', '$5K-10K', '$10K-25K', '$25K-50K', '>$50K']
for i in range(len(bins)-1):
    count = ((claims >= bins[i]) & (claims < bins[i+1])).sum()
    print(f"  {bin_labels[i]:<15}: {count:>5} ({count/len(claims)*100:>5.1f}%)")

print()
print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print("")
print("PROBLEM: The target has many zeros (~50%+), which after log1p become 0.")
print("This creates a model that tends to predict near-zero values.")
print("")
print("SOLUTIONS:")
print("1. Use a two-stage model:")
print("   - Stage 1: Predict if there will be a claim (classification)")
print("   - Stage 2: Predict claim amount (only for claims > 0)")
print("2. Or use Tweedie regression (handles zero-inflated continuous data)")
print("3. Or filter to only non-zero claims for training the amount predictor")
