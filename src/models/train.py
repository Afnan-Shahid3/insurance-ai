import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(features_path: str, target_path: str) -> tuple:
    """
    Load features and target data from CSV files.
    
    Args:
        features_path: Path to features CSV
        target_path: Path to target CSV
        
    Returns:
        tuple: (X, y) - features and target DataFrames
    """
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path)
    
    # Flatten y if it's a DataFrame (single column)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    print(f"Loaded features shape: {X.shape}")
    print(f"Loaded target shape: {y.shape}")
    
    return X, y


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """
    Evaluate model using MAE, RMSE, and R2 score.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model for display
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # R2 Score
    r2 = r2_score(y_true, y_pred)
    
    # Print results
    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"{'='*40}")
    print(f"MAE  (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"R2   (R-squared Score): {r2:.4f}")
    
    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def train_random_forest(X_train, y_train, **kwargs) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor model.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for RandomForestRegressor
        
    Returns:
        RandomForestRegressor: Trained model
    """
    print("\n[Training] Random Forest Regressor...")
    
    # Default parameters (can be customized)
    params = {
        "n_estimators": kwargs.get("n_estimators", 100),
        "max_depth": kwargs.get("max_depth", 10),
        "random_state": kwargs.get("random_state", 42),
        "n_jobs": -1  # Use all CPU cores
    }
    
    print(f"Parameters: {params}")
    
    # Create and train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    print("Random Forest training complete!")
    
    return model


def train_xgboost(X_train, y_train, **kwargs):
    """
    Train an XGBoost Regressor model.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for XGBRegressor
        
    Returns:
        XGBRegressor: Trained model
    """
    try:
        from xgboost import XGBRegressor
        
        print("\n[Training] XGBoost Regressor...")
        
        # Default parameters
        params = {
            "n_estimators": kwargs.get("n_estimators", 100),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.1),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": -1
        }
        
        print(f"Parameters: {params}")
        
        # Create and train model
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        print("XGBoost training complete!")
        
        return model
        
    except ImportError:
        print("\nXGBoost not installed. Skipping XGBoost training.")
        print("To install: pip install xgboost")
        return None


def compare_models(results: list) -> dict:
    """
    Compare all model results and find the best one.
    
    Args:
        results: List of dictionaries containing model results
        
    Returns:
        dict: Best model information
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    # Create comparison table
    print(f"\n{'Model':<25} {'MAE':<12} {'RMSE':<12} {'R2':<10}")
    print("-" * 60)
    
    best_model = None
    best_r2 = -float("inf")
    
    for result in results:
        print(f"{result['model_name']:<25} {result['mae']:<12.2f} {result['rmse']:<12.2f} {result['r2']:<10.4f}")
        
        # Track best model (highest R2)
        if result["r2"] > best_r2:
            best_r2 = result["r2"]
            best_model = result
    
    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_model['model_name']}")
    print(f"Best R2 Score: {best_model['r2']:.4f}")
    print("=" * 60)
    
    return best_model


def save_model(model, filepath: str) -> None:
    """
    Save trained model to a pickle file.
    
    Args:
        model: Trained model
        filepath: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model using pickle
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {filepath}")


def train_pipeline(
    train_features_path: str,
    train_target_path: str,
    test_features_path: str,
    test_target_path: str,
    save_model_path: str = "models/saved_models/best_model.pkl"
) -> dict:
    """
    Complete training pipeline.
    
    Args:
        train_features_path: Path to training features
        train_target_path: Path to training target
        test_features_path: Path to test features
        test_target_path: Path to test target
        save_model_path: Path to save the best model
        
    Returns:
        dict: Best model information
    """
    print("=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    X_train, y_train = load_data(train_features_path, train_target_path)
    X_test, y_test = load_data(test_features_path, test_target_path)
    
    # Store results for comparison
    results = []
    trained_models = {}
    
    # Step 2: Train Random Forest
    print("\n[Step 2] Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate on test set
    rf_pred = rf_model.predict(X_test)
    rf_result = evaluate_model(y_test, rf_pred, "Random Forest")
    results.append(rf_result)
    trained_models["Random Forest"] = rf_model
    
    # Step 3: Train XGBoost (if available)
    print("\n[Step 3] Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    
    if xgb_model is not None:
        # Evaluate on test set
        xgb_pred = xgb_model.predict(X_test)
        xgb_result = evaluate_model(y_test, xgb_pred, "XGBoost")
        results.append(xgb_result)
        trained_models["XGBoost"] = xgb_model
    
    # Step 4: Compare models
    print("\n[Step 4] Comparing models...")
    best = compare_models(results)
    
    # Step 5: Save best model
    print("\n[Step 5] Saving best model...")
    best_model = trained_models[best["model_name"]]
    save_model(best_model, save_model_path)
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    
    return best