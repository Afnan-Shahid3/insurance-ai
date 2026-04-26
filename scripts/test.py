# scripts/01_explore_data_simple.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Starting EDA...")

# Check if file exists
DATA_PATH = "data/raw/insurance_claims.csv"
print(f"Looking for: {DATA_PATH}")
print(f"File exists: {os.path.exists(DATA_PATH)}")

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Data loaded! Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Basic info
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Target stats (total_claim_amount) ---")
print(df['total_claim_amount'].describe())

# Create plots folder
os.makedirs("reports/figures", exist_ok=True)

# Plot target
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['total_claim_amount'], kde=True, bins=30)
plt.title("Claim Amount Distribution")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['total_claim_amount'])
plt.title("Claim Amount Boxplot")

plt.tight_layout()
plt.savefig("reports/figures/target_analysis.png")
print("\nPlot saved!")
plt.close()

print("\nEDA Complete!")