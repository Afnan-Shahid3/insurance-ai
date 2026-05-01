"""
Model Training Script for Insurance Claim Prediction
Loads processed data, trains models, and saves best model
"""
# Ensure input data is 2D with shape (1, n_features) before passing to model.predict

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path to import src modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.train import (
    load_processed_data,
    train_models,
    evaluate_model,
    compare_models,
    save_model
)

# Define paths
DATA_DIR = 'v2-advance_model/data/processed'
MODEL_DIR = 'v2-advance_model/models/saved_models'

# Ensure output directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Step 1: Load processed data
print("=" * 60)
print("STEP 1: Loading Processed Data")
print("=" * 60)
X_train, X_test, y_train, y_test = load_processed_data(DATA_DIR)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
print(f"Train features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Train target shape: {y_train.shape}")
print(f"Test target shape: {y_test.shape}")

# Step 2: Train models
print("\n" + "=" * 60)
print("STEP 2: Training Models")
print("=" * 60)
models = train_models(X_train, y_train, random_state=42)

# Step 3: Evaluate models
print("\n" + "=" * 60)
print("STEP 3: Evaluating Models")
print("=" * 60)

models_results = {}
for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    metrics = evaluate_model(model, X_test, y_test)
    models_results[model_name] = metrics
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  R2:   {metrics['R2']:.4f}")

# Step 4: Compare models
print("\n" + "=" * 60)
print("STEP 4: Model Comparison")
print("=" * 60)

# Print comparison table
print("\n{:<20} {:>12} {:>12} {:>12}".format("Model", "MAE", "RMSE", "R2"))
print("-" * 60)
for model_name, metrics in models_results.items():
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f}".format(
        model_name, metrics['MAE'], metrics['RMSE'], metrics['R2']
    ))

# Find best model
best_model_name = compare_models(models_results)
best_model = models[best_model_name]

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_model_name}")
print("=" * 60)

# Step 5: Save best model
print("\n" + "=" * 60)
print("STEP 5: Saving Best Model")
print("=" * 60)

model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
save_model(best_model, model_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nBest Model: {best_model_name}")
print(f"R2 Score: {models_results[best_model_name]['R2']:.4f}")
print(f"Model saved to: {model_path}")
