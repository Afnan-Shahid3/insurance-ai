"""
Entry point for Exploratory Data Analysis (EDA).
Run with: python scripts/01_explore_data.py
"""

import sys
import os

# Add project root to path so we can import from src/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.load_data import load_csv
from src.data.explore_data import run_eda


if __name__ == "__main__":
    # Path to your data file
    DATA_PATH = "data/raw/insurance_claims.csv"
    
    print("=" * 60)
    print("STARTING EDA PIPELINE")
    print("=" * 60)
    print(f"Data path: {DATA_PATH}")
    print(f"Project root: {project_root}")
    
    # Check if file exists
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Data file not found at {DATA_PATH}")
        print("Please ensure the dataset exists at the expected location.")
        sys.exit(1)
    
    # Run EDA
    df = run_eda(DATA_PATH)
    
    print(f"\nDataset shape: {df.shape}")
    print("Columns: " + ", ".join(df.columns))
    print("\nEDA complete! Check the reports/figures/ directory for visualizations.")