"""Quick data analysis to understand the target distribution"""
import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv('v2-advance_model/data/raw/car_insurance_claim.csv')

# Clean the CLM_AMT column (it has $ and , in it)
df['CLM_AMT_CLEAN'] = df['CLM_AMT'].str.replace('$','',regex=False).str.replace(',','',regex=False).astype(float)

print("=" * 60)
print("TARGET VARIABLE ANALYSIS")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Zero claims: {(df['CLM_AMT_CLEAN']==0).sum()}")
print(f"Non-zero claims: {(df['CLM_AMT_CLEAN']>0).sum()}")
print(f"Percentage of zeros: {(df['CLM_AMT_CLEAN']==0).sum() / len(df) * 100:.2f}%")
print()
print(f"Min (non-zero): {df[df['CLM_AMT_CLEAN']>0]['CLM_AMT_CLEAN'].min():.2f}")
print(f"Max: {df['CLM_AMT_CLEAN'].max():.2f}")
print(f"Mean (all): {df['CLM_AMT_CLEAN'].mean():.2f}")
print(f"Mean (non-zero): {df[df['CLM_AMT_CLEAN']>0]['CLM_AMT_CLEAN'].mean():.2f}")
print(f"Median (all): {df['CLM_AMT_CLEAN'].median():.2f}")
print(f"Median (non-zero): {df[df['CLM_AMT_CLEAN']>0]['CLM_AMT_CLEAN'].median():.2f}")
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

# Check processed data
print("=" * 60)
print("PROCESSED DATA ANALYSIS")
print("=" * 60)
train_target = pd.read_csv('v2-advance_model/data/processed/train_target.csv')
print(f"Train target shape: {train_target.shape}")
print(f"Train target min: {train_target['CLM_AMT'].min():.4f}")
print(f"Train target max: {train_target['CLM_AMT'].max():.4f}")
print(f"Train target mean: {train_target['CLM_AMT'].mean():.4f}")
print(f"Train target zeros: {(train_target['CLM_AMT']==0).sum()}")
print(f"Train target non-zeros: {(train_target['CLM_AMT']>0).sum()}")
print()

# Check if log1p was applied correctly
print("If log1p was applied correctly to original data:")
print(f"  log1p(0) = {np.log1p(0):.4f}")
print(f"  log1p(100) = {np.log1p(100):.4f}")
print(f"  log1p(1000) = {np.log1p(1000):.4f}")
print(f"  log1p(5000) = {np.log1p(5000):.4f}")
print(f"  log1p(10000) = {np.log1p(10000):.4f}")
