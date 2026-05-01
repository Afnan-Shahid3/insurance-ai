"""
Model training module for Insurance Claim Prediction
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from xgboost import XGBRegressor


def load_processed_data(data_dir='v2-advance_model/data/processed'):
    """
    Load processed train and test data.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing processed data files
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    # Load train features
    train_features_path = os.path.join(data_dir, 'train_features.csv')
    X_train = pd.read_csv(train_features_path)
    
    # Load test features
    test_features_path = os.path.join(data_dir, 'test_features.csv')
    X_test = pd.read_csv(test_features_path)
    
    # Load train target
    train_target_path = os.path.join(data_dir, 'train_target.csv')
    y_train = pd.read_csv(train_target_path).squeeze()
    
    # Load test target
    test_target_path = os.path.join(data_dir, 'test_target.csv')
    y_test = pd.read_csv(test_target_path).squeeze()
    
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train, random_state=42):
    """
    Train Random Forest and XGBoost models.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series or np.array
        Training target
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing trained models
    """
    models = {}
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    print("Random Forest training complete!")
    
    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    print("XGBoost training complete!")
    
    return models


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : trained model
        Model to evaluate
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series or np.array
        Test target
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    return metrics


def compare_models(models_results):
    """
    Compare models based on evaluation metrics.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary with model names as keys and metrics as values
        
    Returns:
    --------
    str
        Name of the best model based on R2 score
    """
    # Find best model based on R2 score
    best_model_name = None
    best_r2 = float('-inf')
    
    for model_name, metrics in models_results.items():
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_model_name = model_name
    
    return best_model_name


def save_model(model, path):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : trained model
        Model to save
    path : str
        Path where model will be saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model using joblib
    joblib.dump(model, path)
    print(f"Model saved to: {path}")



