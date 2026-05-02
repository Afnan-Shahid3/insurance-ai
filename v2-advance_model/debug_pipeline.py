"""
Debug script to verify the training pipeline works correctly.
Run this after retraining to check if predictions are in the correct range.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 70)
print("PIPELINE DEBUG SCRIPT")
print("=" * 70)

# --------------------------------------------------------------------------
# Step 1: Check processed data
# --------------------------------------------------------------------------
print("\n[1] CHECKING PROCESSED DATA...")
print("-" * 70)

processed_dir = 'v2-advance_model/data/processed'

# Load targets
try:
    train_target = pd.read_csv(os.path.join(processed_dir, 'train_target.csv'))
    test_target = pd.read_csv(os.path.join(processed_dir, 'test_target.csv'))
    train_target_orig = pd.read_csv(os.path.join(processed_dir, 'train_target_original.csv'))
    test_target_orig = pd.read_csv(os.path.join(processed_dir, 'test_target_original.csv'))
    
    print("✓ All target files loaded successfully")
    
    train_log = train_target.iloc[:, 0]
    test_log = test_target.iloc[:, 0]
    train_orig = train_target_orig.iloc[:, 0]
    test_orig = test_target_orig.iloc[:, 0]
    
    print(f"  Train target (log)      - min: {train_log.min():.4f}, max: {train_log.max():.4f}, mean: {train_log.mean():.4f}")
    print(f"  Test target (log)       - min: {test_log.min():.4f}, max: {test_log.max():.4f}, mean: {test_log.mean():.4f}")
    print(f"  Train target (original) - min: {train_orig.min():.2f}, max: {train_orig.max():.2f}, mean: {train_orig.mean():.2f}")
    print(f"  Test target (original)  - min: {test_orig.min():.2f}, max: {test_orig.max():.2f}, mean: {test_orig.mean():.2f}")
    
    # Check if log transform was applied correctly
    log_min = train_log.min()
    if log_min < 0:
        print("  ⚠️  WARNING: Log target has negative values. Check transformation!")
    else:
        print("  ✓ Log target values look reasonable (non-negative)")
        
except FileNotFoundError as e:
    print(f"  ✗ Error: {e}")
    print("  Make sure to run 02_preprocess_data.py first!")

# --------------------------------------------------------------------------
# Step 2: Load and check the model
# --------------------------------------------------------------------------
print("\n[2] CHECKING MODEL...")
print("-" * 70)

model_path = 'v2-advance_model/models/saved_models/best_model.pkl'

if os.path.exists(model_path):
    try:
        saved_data = joblib.load(model_path)
        
        if isinstance(saved_data, dict) and 'model' in saved_data:
            model = saved_data['model']
            metadata = saved_data.get('metadata', {})
            print("✓ Model loaded with metadata")
            print(f"  Model type: {type(model).__name__}")
            if metadata:
                print(f"  Model name: {metadata.get('model_name', 'Unknown')}")
                print(f"  R² Score: {metadata.get('metrics', {}).get('R2', 'N/A')}")
                print(f"  Training target range (log): [{metadata.get('training_target_min', 'N/A')}, {metadata.get('training_target_max', 'N/A')}]")
        else:
            model = saved_data
            print("✓ Model loaded (legacy format, no metadata)")
            print(f"  Model type: {type(model).__name__}")
            
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        model = None
else:
    print(f"  ✗ Model file not found at: {model_path}")
    print("  Make sure to run 03_train_model.py first!")
    model = None

# --------------------------------------------------------------------------
# Step 3: Make sample predictions
# --------------------------------------------------------------------------
print("\n[3] TESTING PREDICTIONS...")
print("-" * 70)

if model is not None:
    try:
        # Load training features to get column names
        train_features = pd.read_csv(os.path.join(processed_dir, 'train_features.csv'))
        
        # Create a sample input (average values)
        sample_input = train_features.mean().to_frame().T
        
        # Predict
        pred_log = model.predict(sample_input)[0]
        pred_original = np.expm1(pred_log)
        
        print(f"  Sample prediction (log scale):    {pred_log:.4f}")
        print(f"  Sample prediction (original):     ${pred_original:,.2f}")
        
        if pred_original < 100:
            print("  ✗ CRITICAL: Prediction is near-zero! Model is collapsed!")
            print("    This indicates the training pipeline is still broken.")
        elif pred_original < 1000:
            print("  ⚠️  WARNING: Prediction seems low. Check model training.")
        else:
            print("  ✓ Prediction is in a reasonable range!")
        
        # Predict on a few test samples
        print("\n  Predicting on first 5 test samples...")
        test_features = pd.read_csv(os.path.join(processed_dir, 'test_features.csv'))
        test_target_log = pd.read_csv(os.path.join(processed_dir, 'test_target.csv')).squeeze()
        test_target_orig = pd.read_csv(os.path.join(processed_dir, 'test_target_original.csv')).squeeze()
        
        preds_log = model.predict(test_features.iloc[:5])
        preds_original = np.expm1(preds_log)
        
        for i in range(5):
            actual_original = test_target_orig.iloc[i]
            print(f"    Sample {i+1}: Predicted=${preds_original[i]:,.2f}, Actual=${actual_original:,.2f}")
            
    except Exception as e:
        print(f"  ✗ Error during prediction: {e}")
        traceback.print_exc()
else:
    print("  Skipped (model not loaded)")

# --------------------------------------------------------------------------
# Step 4: Check feature alignment
# --------------------------------------------------------------------------
print("\n[4] CHECKING FEATURE ALIGNMENT...")
print("-" * 70)

try:
    train_features = pd.read_csv(os.path.join(processed_dir, 'train_features.csv'))
    test_features = pd.read_csv(os.path.join(processed_dir, 'test_features.csv'))
    
    print(f"  Train features: {train_features.shape}")
    print(f"  Test features:  {test_features.shape}")
    
    # Check for column mismatch
    train_cols = set(train_features.columns)
    test_cols = set(test_features.columns)
    
    missing_in_test = train_cols - test_cols
    extra_in_test = test_cols - train_cols
    
    if missing_in_test:
        print(f"  ⚠️  Columns in train but not in test: {missing_in_test}")
    if extra_in_test:
        print(f"  ⚠️  Columns in test but not in train: {extra_in_test}")
    if not missing_in_test and not extra_in_test:
        print("  ✓ Train and test have the same columns")
        
except Exception as e:
    print(f"  ✗ Error: {e}")

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("")
print("If you see any ✗ or ⚠️  warnings above, the pipeline needs fixing.")
print("")
print("Steps to fix:")
print("  1. Run: python v2-advance_model/scripts/02_preprocess_data.py")
print("  2. Run: python v2-advance_model/scripts/03_train_model.py")
print("  3. Run this debug script again to verify: python v2-advance_model/debug_pipeline.py")
print("  4. Run Streamlit: streamlit run v2-advance_model/app/streamlit_app.py")
print("")
print("=" * 70)
