"""
Preprocessing script for insurance claims data.
Run with: python scripts/02_preprocess_data.py
"""

import os
import pandas as pd

# Add project root to path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.preprocessing import preprocess_pipeline


if __name__ == "__main__":
    # Configuration
    RAW_DATA_PATH = "data/raw/insurance_claims.csv"
    TARGET_COL = "total_claim_amount"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Output paths
    OUTPUT_DIR = "data/processed"
    TRAIN_FEATURES_PATH = os.path.join(OUTPUT_DIR, "train_features.csv")
    TEST_FEATURES_PATH = os.path.join(OUTPUT_DIR, "test_features.csv")
    TRAIN_TARGET_PATH = os.path.join(OUTPUT_DIR, "train_target.csv")
    TEST_TARGET_PATH = os.path.join(OUTPUT_DIR, "test_target.csv")
    
    print("=" * 60)
    print("INSURANCE CLAIM PREPROCESSING")
    print("=" * 60)
    print(f"Raw data: {RAW_DATA_PATH}")
    print(f"Target column: {TARGET_COL}")
    print(f"Test size: {TEST_SIZE * 100}%")
    
    # Check if raw data exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"\nERROR: Raw data file not found at {RAW_DATA_PATH}")
        sys.exit(1)
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocess_pipeline(
        data_path=RAW_DATA_PATH,
        target_col=TARGET_COL,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory created: {OUTPUT_DIR}")
    
    # Save processed data
    print("\nSaving processed data...")
    
    # Save features
    X_train.to_csv(TRAIN_FEATURES_PATH, index=False)
    print(f"  Saved training features: {TRAIN_FEATURES_PATH}")
    
    X_test.to_csv(TEST_FEATURES_PATH, index=False)
    print(f"  Saved testing features: {TEST_FEATURES_PATH}")
    
    # Save targets
    y_train.to_csv(TRAIN_TARGET_PATH, index=False)
    print(f"  Saved training target: {TRAIN_TARGET_PATH}")
    
    y_test.to_csv(TEST_TARGET_PATH, index=False)
    print(f"  Saved testing target: {TEST_TARGET_PATH}")
    
    print("\n" + "=" * 60)
    print("ALL FILES SAVED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    print(f"\nFeatures count: {X_train.shape[1]}")