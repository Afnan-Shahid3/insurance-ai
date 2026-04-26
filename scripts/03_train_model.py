"""
Model training script for Insurance Claim Prediction.
Run with: python scripts/03_train_model.py
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.train import train_pipeline


if __name__ == "__main__":
    # Configuration
    TRAIN_FEATURES = "data/processed/train_features.csv"
    TRAIN_TARGET = "data/processed/train_target.csv"
    TEST_FEATURES = "data/processed/test_features.csv"
    TEST_TARGET = "data/processed/test_target.csv"
    
    # Model save path
    MODEL_PATH = "models/saved_models/best_model.pkl"
    
    print("=" * 60)
    print("INSURANCE CLAIM MODEL TRAINING")
    print("=" * 60)
    print(f"Training features: {TRAIN_FEATURES}")
    print(f"Training target: {TRAIN_TARGET}")
    print(f"Test features: {TEST_FEATURES}")
    print(f"Test target: {TEST_TARGET}")
    print(f"Model will be saved to: {MODEL_PATH}")
    
    # Check if processed data exists
    if not os.path.exists(TRAIN_FEATURES):
        print(f"\nERROR: Training features not found at {TRAIN_FEATURES}")
        print("Please run preprocessing first: python scripts/02_preprocess_data.py")
        sys.exit(1)
    
    if not os.path.exists(TRAIN_TARGET):
        print(f"\nERROR: Training target not found at {TRAIN_TARGET}")
        sys.exit(1)
    
    # Run training pipeline
    best_model_info = train_pipeline(
        train_features_path=TRAIN_FEATURES,
        train_target_path=TRAIN_TARGET,
        test_features_path=TEST_FEATURES,
        test_target_path=TEST_TARGET,
        save_model_path=MODEL_PATH
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest Model: {best_model_info['model_name']}")
    print(f"Test MAE: {best_model_info['mae']:.2f}")
    print(f"Test RMSE: {best_model_info['rmse']:.2f}")
    print(f"Test R2: {best_model_info['r2']:.4f}")
    print(f"\nModel saved at: {MODEL_PATH}")