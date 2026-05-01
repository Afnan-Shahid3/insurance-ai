"""
Data Preprocessing Script for Insurance Claim Prediction
Loads data, applies preprocessing steps, and saves train/test files
"""

import os
import sys
import pandas as pd

# Add project root to path to import src modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.preprocessing import (
    load_data,
    clean_data,
    encode_features,
    split_features_target,
    apply_log_transform,
    train_test_split_data
)

# Define paths
DATA_PATH = os.path.join('v2-advance_model', 'data', 'raw', 'car_insurance_claim.csv')
OUTPUT_DIR = os.path.join('v2-advance_model', 'data', 'processed')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load data
print("=" * 50)
print("STEP 1: Loading Data")
print("=" * 50)
df = load_data(DATA_PATH)
print(f"Data loaded successfully!")
print(f"Shape: {df.shape}")

# Step 2: Clean data
print("\n" + "=" * 50)
print("STEP 2: Cleaning Data")
print("=" * 50)
df = clean_data(df)
print(f"Data cleaned successfully!")
print(f"Shape after cleaning: {df.shape}")

# Step 3: Encode features
print("\n" + "=" * 50)
print("STEP 3: Encoding Features")
print("=" * 50)
df = encode_features(df)
print(f"Features encoded successfully!")
print(f"Shape after encoding: {df.shape}")

# Step 4: Split features and target
print("\n" + "=" * 50)
print("STEP 4: Splitting Features and Target")
print("=" * 50)
X, y = split_features_target(df, target="CLM_AMT")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Step 5: Apply log transform to target
print("\n" + "=" * 50)
print("STEP 5: Applying Log Transform to Target")
print("=" * 50)
y_log = apply_log_transform(y)
print(f"Log transform applied successfully!")
print(f"Original target - min: {y.min()}, max: {y.max()}")
print(f"Log target - min: {y_log.min():.4f}, max: {y_log.max():.4f}")

# Step 6: Train-test split
print("\n" + "=" * 50)
print("STEP 6: Train-Test Split")
print("=" * 50)
X_train, X_test, y_train, y_test = train_test_split_data(X, y_log, test_size=0.2, random_state=42)
print(f"Train features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Train target shape: {y_train.shape}")
print(f"Test target shape: {y_test.shape}")

# Step 7: Save output files
print("\n" + "=" * 50)
print("STEP 7: Saving Output Files")
print("=" * 50)

# Save train features
train_features_path = os.path.join(OUTPUT_DIR, 'train_features.csv')
X_train.to_csv(train_features_path, index=False)
print(f"Saved: {train_features_path}")

# Save test features
test_features_path = os.path.join(OUTPUT_DIR, 'test_features.csv')
X_test.to_csv(test_features_path, index=False)
print(f"Saved: {test_features_path}")

# Save train target
train_target_path = os.path.join(OUTPUT_DIR, 'train_target.csv')
pd.DataFrame(y_train).to_csv(train_target_path, index=False)
print(f"Saved: {train_target_path}")

# Save test target
test_target_path = os.path.join(OUTPUT_DIR, 'test_target.csv')
pd.DataFrame(y_test).to_csv(test_target_path, index=False)
print(f"Saved: {test_target_path}")

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE!")
print("=" * 50)
print(f"\nAll files saved to: {OUTPUT_DIR}")
