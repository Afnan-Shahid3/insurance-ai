import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.preprocessing import (
    load_data,
    clean_data,
    encode_features
)

DATA_PATH = os.path.join('v2-advance_model', 'data', 'raw', 'car_insurance_claim.csv')
OUTPUT_DIR = os.path.join('v2-advance_model', 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_diagnostics(claim_flag, claim_amount, dataset_name):
    total = len(claim_flag)
    zero_count = (claim_flag == 0).sum()
    positive_count = (claim_flag == 1).sum()
    positive_amounts = claim_amount[claim_flag == 1]
    
    print(f"\n{dataset_name} Diagnostics:")
    print(f"  Total rows: {total}")
    print(f"  Zero claims: {zero_count} ({zero_count/total*100:.1f}%)")
    print(f"  Positive claims: {positive_count} ({positive_count/total*100:.1f}%)")
    
    if len(positive_amounts) > 0:
        print(f"  Claim amount range: ${positive_amounts.min():.0f} - ${positive_amounts.max():.0f}")
        print(f"  Mean claim amount: ${positive_amounts.mean():.0f}")
    else:
        print("  No positive claims found")

print("=" * 50)
print("STEP 1: Loading Data")
print("=" * 50)
df = load_data(DATA_PATH)
print(f"Data loaded successfully!")
print(f"Shape: {df.shape}")

print("\n" + "=" * 50)
print("STEP 2: Cleaning Data")
print("=" * 50)
df = clean_data(df)
print(f"Data cleaned successfully!")
print(f"Shape after cleaning: {df.shape}")

print("\n" + "=" * 50)
print("CONVERTING CLAIM AMOUNT TO NUMERIC")
print("=" * 50)
df['CLM_AMT'] = (
    df['CLM_AMT']
    .astype(str)
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
)
df['CLM_AMT'] = pd.to_numeric(df['CLM_AMT'], errors='coerce')
df['CLM_AMT'] = df['CLM_AMT'].fillna(0)
print(f"Converted CLM_AMT to numeric")
print(f"Min: ${df['CLM_AMT'].min():.2f}, Max: ${df['CLM_AMT'].max():.2f}, Mean: ${df['CLM_AMT'].mean():.2f}")

print("\n" + "=" * 50)
print("STEP 3: Creating Two-Stage Targets")
print("=" * 50)
original_target = df['CLM_AMT'].copy()
claim_flag = np.where(original_target > 0, 1, 0)
claim_flag = pd.Series(claim_flag, name='claim_flag')
claim_amount = original_target.copy()
claim_amount[claim_flag == 0] = np.nan

print(f"Two-stage targets created successfully!")
print(f"Features shape: {df.shape}")
print(f"Claim flag shape: {claim_flag.shape}")
print(f"Claim amount shape: {claim_amount.shape}")
print_diagnostics(claim_flag, claim_amount, "FULL DATASET")

print("\n" + "=" * 50)
print("STEP 4: Encoding Features")
print("=" * 50)
X = encode_features(df)
print(f"Features encoded successfully!")
print(f"Shape after encoding: {X.shape}")

print("\n" + "=" * 50)
print("STEP 5: Train-Test Split")
print("=" * 50)
temp_df = pd.DataFrame(index=range(len(X)))
train_idx, test_idx = train_test_split(
    temp_df.index, 
    test_size=0.2, 
    random_state=42, 
    stratify=claim_flag
)
X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
flag_train = claim_flag.iloc[train_idx]
flag_test = claim_flag.iloc[test_idx]
amount_train = claim_amount.iloc[train_idx]
amount_test = claim_amount.iloc[test_idx]
original_train = original_target.iloc[train_idx]
original_test = original_target.iloc[test_idx]

print(f"Train features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Train claim_flag shape: {flag_train.shape}")
print(f"Test claim_flag shape: {flag_test.shape}")
print(f"Train claim_amount shape: {amount_train.shape}")
print(f"Test claim_amount shape: {amount_test.shape}")
print_diagnostics(flag_train, amount_train, "TRAIN SET")
print_diagnostics(flag_test, amount_test, "TEST SET")

print("\n" + "=" * 50)
print("STEP 6: Processing Claim Amount")
print("=" * 50)
# Filter to ONLY positive claims before processing
amount_train_positive = amount_train[flag_train == 1].reset_index(drop=True)
amount_test_positive = amount_test[flag_test == 1].reset_index(drop=True)

# Apply log transformation
amount_train_log = np.log1p(amount_train_positive)
amount_test_log = np.log1p(amount_test_positive)

print("\n" + "=" * 50)
print("STEP 7: Saving Output Files")
print("=" * 50)
X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test.csv'), index=False)
pd.DataFrame(flag_train).to_csv(os.path.join(OUTPUT_DIR, 'train_claim_flag.csv'), index=False)
pd.DataFrame(flag_test).to_csv(os.path.join(OUTPUT_DIR, 'test_claim_flag.csv'), index=False)
amount_train_log.reset_index(drop=True).to_frame(name='claim_amount').to_csv(
    os.path.join(OUTPUT_DIR, 'train_claim_amount.csv'), index=False)
amount_test_log.reset_index(drop=True).to_frame(name='claim_amount').to_csv(
    os.path.join(OUTPUT_DIR, 'test_claim_amount.csv'), index=False)
original_train.reset_index(drop=True).to_frame(name='original_claim').to_csv(
    os.path.join(OUTPUT_DIR, 'train_target_original.csv'), index=False)
original_test.reset_index(drop=True).to_frame(name='original_claim').to_csv(
    os.path.join(OUTPUT_DIR, 'test_target_original.csv'), index=False)

print("\nSaved all required files:")
print(f"  X_train.csv, X_test.csv")
print(f"  train_claim_flag.csv, test_claim_flag.csv")
print(f"  train_claim_amount.csv, test_claim_amount.csv")
print(f"  train_target_original.csv, test_target_original.csv")

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE!")
print("=" * 50)