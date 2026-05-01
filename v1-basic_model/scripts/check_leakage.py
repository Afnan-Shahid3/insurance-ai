import pandas as pd

# Load the raw data
df = pd.read_csv("data/raw/insurance_claims.csv")

print("Checking for data leakage...")
print(f"Dataset shape: {df.shape}")

# Check if total = sum of others
calculated = df['injury_claim'] + df['property_claim'] + df['vehicle_claim']
matches = (calculated == df['total_claim_amount']).sum()

print(f"\nRows where (injury + property + vehicle) = total_claim_amount: {matches}/{len(df)}")

if matches == len(df):
    print("\n⚠️ DATA LEAKAGE CONFIRMED!")
    print("The model is just learning: total = injury + property + vehicle")
else:
    print("\n✓ No exact leakage - values don't match perfectly")