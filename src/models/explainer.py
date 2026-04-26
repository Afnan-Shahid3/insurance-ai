import pandas as pd
import pickle
import numpy as np


def load_model(model_path: str):
    """
    Load trained model from pickle file.
    
    Args:
        model_path: Path to the saved model (.pkl file)
        
    Returns:
        Trained model object
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from: {model_path}")
    return model


def get_feature_importance(model, feature_names: list, top_n: int = 10) -> pd.DataFrame:
    """
    Extract feature importance from the model.
    
    Args:
        model: Trained model (must have feature_importances_ attribute)
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        pd.DataFrame: Top N features with their importance scores
    """
    # Get feature importances from the model
    importances = model.feature_importances_
    
    # Create a DataFrame with feature names and importance scores
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    
    # Sort by importance (highest first)
    importance_df = importance_df.sort_values("importance", ascending=False)
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    return top_features


def print_feature_importance(importance_df: pd.DataFrame) -> None:
    """
    Print feature importance in a readable format.
    
    Args:
        importance_df: DataFrame with feature importance
    """
    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)
    
    print(f"\n{'Rank':<6} {'Feature':<40} {'Importance':<12}")
    print("-" * 60)
    
    for idx, row in importance_df.iterrows():
        rank = importance_df.index.get_loc(idx) + 1
        print(f"{rank:<6} {row['feature']:<40} {row['importance']:.4f}")
    
    print("=" * 60)


def explain_prediction(model, input_data: pd.DataFrame, feature_names: list) -> str:
    """
    Generate a human-readable explanation for a prediction.
    
    Args:
        model: Trained model
        input_data: Input features for prediction (single row as DataFrame)
        feature_names: List of feature names
        
    Returns:
        str: Human-readable explanation
    """
    # Get prediction
    prediction = model.predict(input_data)[0]
    
    # Get feature importance
    importance_df = get_feature_importance(model, feature_names, top_n=10)
    
    # Get top 5 features for explanation
    top_5_features = importance_df.head(5)['feature'].tolist()
    
    # Clean up feature names (remove prefix like 'policy_state_')
    cleaned_features = []
    for feat in top_5_features:
        # Try to make it more readable
        if '_' in feat:
            # Take the last part after underscore (e.g., 'policy_state_CA' -> 'CA')
            parts = feat.rsplit('_', 1)
            if len(parts) > 1 and len(parts[1]) < 15:
                cleaned_features.append(parts[-1])
            else:
                cleaned_features.append(feat.replace('_', ' '))
        else:
            cleaned_features.append(feat)
    
    # Build explanation
    explanation = f"""
================================================================================
PREDICTION EXPLANATION
================================================================================

Predicted Claim Cost: ${prediction:,.2f}

KEY FACTORS INFLUENCING THIS PREDICTION:
"""
    
    for i, feat in enumerate(cleaned_features, 1):
        importance = importance_df.iloc[i-1]['importance']
        explanation += f"  {i}. {feat} (importance: {importance:.2%})\n"
    
    explanation += """
================================================================================
INTERPRETATION:
- The model uses these top features to predict the claim cost
- Higher importance means the feature has more impact on the prediction
- This helps understand WHY the model makes certain predictions
================================================================================
"""
    
    return explanation


def get_risk_level(predicted_cost: float) -> str:
    """
    Classify risk level based on predicted claim cost.
    
    Args:
        predicted_cost: Predicted claim amount
        
    Returns:
        str: Risk level (Low, Medium, or High)
    """
    if predicted_cost < 10000:
        return "LOW"
    elif predicted_cost < 30000:
        return "MEDIUM"
    else:
        return "HIGH"


def full_explanation(model, input_data: pd.DataFrame, feature_names: list) -> None:
    """
    Generate complete explanation with prediction and risk level.
    
    Args:
        model: Trained model
        input_data: Input features
        feature_names: List of feature names
    """
    # Get prediction
    prediction = model.predict(input_data)[0]
    
    # Get risk level
    risk_level = get_risk_level(prediction)
    
    # Print prediction
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Predicted Claim Cost: ${prediction:,.2f}")
    print(f"Risk Level: {risk_level}")
    print("=" * 60)
    
    # Get and print feature importance
    importance_df = get_feature_importance(model, feature_names, top_n=10)
    print_feature_importance(importance_df)
    
    # Generate detailed explanation
    explanation = explain_prediction(model, input_data, feature_names)
    print(explanation)