"""
Model evaluation and explanation script.
Run with: python scripts/04_evaluate_model.py
"""

import os
import sys
import pandas as pd
import pickle
import numpy as np
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.explainer import (
    load_model,
    get_feature_importance,
    print_feature_importance,
    explain_prediction,
    full_explanation,
    get_risk_level
)


def load_test_data():
    """Load test features and target for evaluation."""
    X_test = pd.read_csv("data/processed/test_features.csv")
    y_test = pd.read_csv("data/processed/test_target.csv")
    
    # Flatten if needed
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    
    return X_test, y_test


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/saved_models/best_model.pkl"
    
    print("=" * 60)
    print("MODEL EVALUATION & EXPLANATION")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Please run training first: python scripts/03_train_model.py")
        sys.exit(1)
    
    # Step 1: Load model
    print("\n[Step 1] Loading trained model...")
    model = load_model(MODEL_PATH)
    print(f"Model type: {type(model).__name__}")
    
    # Step 2: Load test data
    print("\n[Step 2] Loading test data...")
    X_test, y_test = load_test_data()
    print(f"Test data shape: {X_test.shape}")
    
    # Get feature names
    feature_names = X_test.columns.tolist()
    print(f"Number of features: {len(feature_names)}")
    
    # Step 3: Show feature importance
    print("\n[Step 3] Extracting feature importance...")
    importance_df = get_feature_importance(model, feature_names, top_n=10)
    print_feature_importance(importance_df)
    
    # Step 4: Make prediction on first test sample
    print("\n[Step 4] Sample Prediction with Explanation...")
    
    # Get first test sample
    sample = X_test.iloc[[0]]
    actual_value = y_test.iloc[0]
    
    # Generate full explanation
    full_explanation(model, sample, feature_names)
    
    # Compare with actual
    print(f"\n[Comparison]")
    print(f"Actual Claim Cost: ${actual_value:,.2f}")
    print(f"Predicted Claim Cost: ${model.predict(sample)[0]:,.2f}")
    print(f"Difference: ${abs(actual_value - model.predict(sample)[0]):,.2f}")
    
    # Step 5: Evaluate on full test set
    print("\n" + "=" * 60)
    print("FULL TEST SET EVALUATION")
    print("=" * 60)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Set Metrics:")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²:   {r2:.4f}")
    
    # Risk level distribution
    print("\n[Risk Level Distribution on Test Set]")
    predictions = model.predict(X_test)
    risk_counts = {
        "LOW": sum(1 for p in predictions if p < 10000),
        "MEDIUM": sum(1 for p in predictions if 10000 <= p < 30000),
        "HIGH": sum(1 for p in predictions if p >= 30000)
    }
    
    total = len(predictions)
    print(f"  LOW ( < $10,000):    {risk_counts['LOW']:3d} ({risk_counts['LOW']/total*100:.1f}%)")
    print(f"  MEDIUM ($10k-$30k):  {risk_counts['MEDIUM']:3d} ({risk_counts['MEDIUM']/total*100:.1f}%)")
    print(f"  HIGH ( > $30,000):   {risk_counts['HIGH']:3d} ({risk_counts['HIGH']/total*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)