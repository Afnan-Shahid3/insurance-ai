import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def remove_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that cause data leakage.
    
    These columns are components of total_claim_amount:
    - injury_claim
    - property_claim
    - vehicle_claim
    
    If we keep them, the model just learns: total = injury + property + vehicle
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with leaky features removed
    """
    # Columns to remove (components of target)
    leaky_cols = ['injury_claim', 'property_claim', 'vehicle_claim']
    
    # Check which columns exist
    existing_leaky = [col for col in leaky_cols if col in df.columns]
    
    if existing_leaky:
        print(f"Removing leaky columns: {existing_leaky}")
        df = df.drop(columns=existing_leaky)
        print(f"DataFrame shape after removal: {df.shape}")
    else:
        print("No leaky columns found to remove.")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    For numeric columns: fill with median
    For categorical columns: fill with mode (most frequent value)
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    df = df.copy()
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Fill missing values in numeric columns with median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled {col} with median: {median_val}")
    
    # Fill missing values in categorical columns with mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  Filled {col} with mode: {mode_val}")
    
    print(f"Missing values after handling: {df.isnull().sum().sum()}")
    return df


def encode_categorical(df: pd.DataFrame, target_col: str = "total_claim_amount") -> pd.DataFrame:
    """
    Encode categorical variables using pandas get_dummies (One-Hot Encoding).
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column to exclude from encoding
        
    Returns:
        pd.DataFrame: DataFrame with categorical columns encoded
    """
    df = df.copy()
    
    # Identify categorical columns (object type)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    print(f"\nEncoding {len(categorical_cols)} categorical columns...")
    
    # Use get_dummies to one-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"DataFrame shape after encoding: {df_encoded.shape}")
    print(f"New columns added: {df_encoded.shape[1] - df.shape[1]}")
    
    return df_encoded


def split_data(
    df: pd.DataFrame,
    target_col: str = "total_claim_amount",
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        test_size: Proportion of data for testing (0.2 = 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separate features (X) and target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"\nTraining set size: {len(X_train)} samples ({ (1-test_size)*100 }%)")
    print(f"Testing set size: {len(X_test)} samples ({ test_size*100 }%)")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(
    data_path: str,
    target_col: str = "total_claim_amount",
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Complete preprocessing pipeline.
    
    Args:
        data_path: Path to raw CSV file
        target_col: Name of target column
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")
    
    # Step 2: Remove leaky features
    print("\n[Step 2] Removing leaky features...")
    df = remove_leaky_features(df)
    
    # Step 3: Handle missing values
    print("\n[Step 3] Handling missing values...")
    df = handle_missing_values(df)
    
    # Step 4: Encode categorical variables
    print("\n[Step 4] Encoding categorical variables...")
    df = encode_categorical(df, target_col=target_col)
    
    # Step 5: Split data
    print("\n[Step 5] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        df, target_col=target_col, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test