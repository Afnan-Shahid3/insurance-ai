import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.data.load_data import load_csv


def basic_info(df: pd.DataFrame) -> None:
    """Print basic information about the dataset."""
    print("=" * 60)
    print("BASIC DATA INFO")
    print("=" * 60)
    
    print("\n--- Head (first 5 rows) ---")
    print(df.head())
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- Statistical Summary ---")
    print(df.describe())


def missing_values(df: pd.DataFrame) -> None:
    """Analyze missing values in the dataset."""
    print("\n" + "=" * 60)
    print("MISSING VALUES ANALYSIS")
    print("=" * 60)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Percentage": missing_pct
    })
    
    subset = missing_df[missing_df["Missing Count"] > 0]
    if subset.empty:
        print("No missing values found!")
    else:
        print(subset)


def target_analysis(df: pd.DataFrame, target: str = "total_claim_amount") -> None:
    """Analyze the target variable distribution."""
    print("\n" + "=" * 60)
    print(f"TARGET VARIABLE ANALYSIS: '{target}'")
    print("=" * 60)
    
    target_data = df[target]
    
    print(f"\n--- Basic Stats for '{target}' ---")
    print(f"Mean: {target_data.mean():.2f}")
    print(f"Median: {target_data.median():.2f}")
    print(f"Std Dev: {target_data.std():.2f}")
    print(f"Min: {target_data.min():.2f}")
    print(f"Max: {target_data.max():.2f}")
    
    # Create output directory
    os.makedirs("reports/figures", exist_ok=True)
    
    # Plot histogram
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(target_data, kde=True, bins=30)
    plt.title(f"Distribution of {target}")
    plt.xlabel(target)
    plt.ylabel("Frequency")
    
    # Plot boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=target_data)
    plt.title(f"Boxplot of {target}")
    plt.xlabel(target)
    
    plt.tight_layout()
    plt.savefig("reports/figures/target_analysis.png")
    print("\nSaved plot to reports/figures/target_analysis.png")
    plt.close()


def simple_plots(df: pd.DataFrame) -> None:
    """Generate simple exploratory plots for numeric features."""
    print("\n" + "=" * 60)
    print("GENERATING SIMPLE PLOTS")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    # Remove target from list
    if "total_claim_amount" in numeric_cols:
        numeric_cols.remove("total_claim_amount")
    
    # Limit to first 4 features
    plot_cols = numeric_cols[:4]
    
    if not plot_cols:
        print("No numeric columns found for plotting.")
        return
    
    for col in plot_cols:
        plt.figure(figsize=(12, 4))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"Distribution of {col}")
        plt.ylabel("Frequency")
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        
        plt.tight_layout()
        filename = f"reports/figures/explore_{col}.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close()


def run_eda(data_path: str, target: str = "total_claim_amount") -> pd.DataFrame:
    """Run the complete EDA pipeline."""
    # Load data
    print("Loading data...")
    df = load_csv(data_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # Create output directory
    os.makedirs("reports/figures", exist_ok=True)
    
    # Run analyses
    basic_info(df)
    missing_values(df)
    target_analysis(df, target=target)
    simple_plots(df)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    
    return df