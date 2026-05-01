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
    Encode categorical variables using Label Encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded categorical variables
    """
    df = df.copy()
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Apply Label Encoding to each categorical column
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df


def split_features_target(df, target="CLM_AMT"):
    """
    Split dataframe into features and target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target : str
        Name of the target column
        
    Returns:
    --------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series
        Target series
    """
    # Convert target column to numeric if it's string (e.g., with $ or ,)
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
        Target variable
        
    Returns:
    --------
    np.array
        Log-transformed target variable
    """
    # Handle any negative or zero values by adding 1 before log
    y_transformed = np.log1p(y)
    return y_transformed


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
