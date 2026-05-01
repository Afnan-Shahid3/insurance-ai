import pandas as pd

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    df = pd.read_csv(filepath)
    return df