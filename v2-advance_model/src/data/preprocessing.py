"""
Preprocessing functions for Insurance Claim Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Load dataset from CSV file.
    
    Parameters:
    -----------
    path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """
    Clean the dataframe by handling missing values and dropping irrelevant columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df = df.copy()
    
    # Drop ID column (irrelevant for modeling)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Fill missing values in numeric columns with median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill missing values in categorical columns with mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value[0])
    
    return df


def encode_features(df):
    """
    Encode categorical variables using One-Hot Encoding.
    Only keep and encode the features used in the Streamlit app.
    """
    df = df.copy()

    # Only keep the features used in the app
    keep_cols = ['AGE', 'INCOME', 'CAR_TYPE', 'CAR_AGE', 'MVR_PTS', 'CLM_FREQ', 'RED_CAR', 'TIF', 'YOJ']
    
    # Drop columns not in keep_cols
    df = df[keep_cols]

    # Encode CAR_TYPE
    categorical_cols = ['CAR_TYPE']
    if categorical_cols:
        df = pd.get_dummies(
            df,
            columns=categorical_cols,
            drop_first=True,
            dtype=int
        )

    return df


def create_two_stage_targets(df, claim_amount_col="CLM_AMT"):
    """
    Create two-stage targets for insurance claim prediction:
    1. claim_flag: Binary (0 = no claim, 1 = claim > 0)
    2. claim_amount: Amount only for rows where claim_flag = 1
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    claim_amount_col : str
        Name of the claim amount column (default: "CLM_AMT")
        
    Returns:
    --------
    pd.DataFrame, pd.Series, pd.Series
        df_clean: Dataframe without claim amount column
        claim_flag: Binary series (0/1)
        claim_amount: Series with claim amounts (only positive, NaN for zero claims)
    """
    df = df.copy()
    
    if claim_amount_col not in df.columns:
        raise KeyError(f"Claim amount column '{claim_amount_col}' not found in dataframe")
    
    # Convert claim amount to numeric if needed
    if df[claim_amount_col].dtype == 'object':
        df[claim_amount_col] = df[claim_amount_col].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df[claim_amount_col] = pd.to_numeric(df[claim_amount_col], errors='coerce')
    
    # Create claim_flag (binary target)
    claim_flag = (df[claim_amount_col] > 0).astype(int)
    
    # Create claim_amount (only for positive claims, NaN for zero claims)
    claim_amount = df[claim_amount_col].copy()
    claim_amount = claim_amount.where(claim_amount > 0)
    
    # Remove claim amount column from features
    df_clean = df.drop(columns=[claim_amount_col])
    
    return df_clean, claim_flag, claim_amount


def print_target_diagnostics(claim_flag, claim_amount, dataset_name="Dataset"):
    """
    Print diagnostics about the target distribution.
    
    Parameters:
    -----------
    claim_flag : pd.Series
        Binary claim flag (0/1)
    claim_amount : pd.Series
        Claim amounts (with NaN for zero claims)
    dataset_name : str
        Name of the dataset for printing
    """
    print("\n" + "=" * 60)
    print(f"TARGET DIAGNOSTICS: {dataset_name}")
    print("=" * 60)
    
    total = len(claim_flag)
    zero_claims = (claim_flag == 0).sum()
    positive_claims = (claim_flag == 1).sum()
    
    print(f"Total samples: {total}")
    print(f"Zero claims (flag=0): {zero_claims} ({zero_claims/total*100:.1f}%)")
    print(f"Positive claims (flag=1): {positive_claims} ({positive_claims/total*100:.1f}%)")
    
    # Distribution of positive claims
    positive_amounts = claim_amount.dropna()
    if len(positive_amounts) > 0:
        print(f"\nDistribution of POSITIVE claim amounts:")
        print(f"  Count: {len(positive_amounts)}")
        print(f"  Min: {positive_amounts.min():.2f}")
        print(f"  Max: {positive_amounts.max():.2f}")
        print(f"  Mean: {positive_amounts.mean():.2f}")
        print(f"  Median: {positive_amounts.median():.2f}")
        print(f"  Std: {positive_amounts.std():.2f}")
        print(f"  25th percentile: {positive_amounts.quantile(0.25):.2f}")
        print(f"  75th percentile: {positive_amounts.quantile(0.75):.2f}")
        print(f"  95th percentile: {positive_amounts.quantile(0.95):.2f}")
        print(f"  99th percentile: {positive_amounts.quantile(0.99):.2f}")
    else:
        print("\nNo positive claims found!")


def split_features_target(df, target="CLM_AMT"):
    """
    Split dataframe into features and target.
    Converts target to numeric if needed.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe columns: {list(df.columns)}")

    # Convert target to numeric if stored as string
    if df[target].dtype == 'object':
        df[target] = df[target].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df[target] = pd.to_numeric(df[target], errors='coerce')

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def apply_log_transform(y):
    """
    Apply log transformation to the target variable.
    
    Parameters:
    -----------
    y : pd.Series or np.array
        Target variable (should only contain positive values)
        
    Returns:
    --------
    np.array
        Log-transformed target variable
    """
    # Handle any negative or zero values by adding 1 before log
    y_transformed = np.log1p(y)
    return y_transformed


def inverse_log_transform(y_log):
    """
    Inverse transform log-transformed target back to original scale.
    
    Parameters:
    -----------
    y_log : pd.Series or np.array
        Log-transformed target variable
        
    Returns:
    --------
    np.array
        Original-scale target variable
    """
    return np.expm1(y_log)


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into training and testing sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series or np.array
        Target variable
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def train_test_split_data_stratified(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Split features and target into training and testing sets with optional stratification.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series or np.array
        Target variable
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    stratify : array-like, optional
        Stratification labels
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    return X_train, X_test, y_train, y_test
