"""
Model Evaluation and Decision Engine Demo Script
Transforms ML predictions into insurance business decisions
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add project root to path to import src modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def inverse_log_transform(x):
    return np.expm1(x)

from src.models.train import load_processed_data
from src.models.decision_engine import calculate_final_payout

# Define paths
MODEL_PATH = 'v2-advance_model/models/saved_models/best_model.pkl'
DATA_DIR = 'v2-advance_model/data/processed'

# Step 1: Load trained model
print("=" * 70)
print("STEP 1: Loading Trained Model")
print("=" * 70)
model = joblib.load(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")

# Step 2: Load test data
print("\n" + "=" * 70)
print("STEP 2: Loading Test Data")
print("=" * 70)
X_train, X_test, y_train, y_test = load_processed_data(DATA_DIR)
y_test = np.array(y_test)
print(f"Test features shape: {X_test.shape}")
print(f"Test target shape: {y_test.shape}")

# Step 3: Make predictions (already in log scale, convert to real)
print("\n" + "=" * 70)
print("STEP 3: Making Predictions")
print("=" * 70)
y_pred_log = model.predict(X_test)
y_pred_real = inverse_log_transform(y_pred_log)
print(f"Predictions made for {len(y_pred_real)} test samples")
print(f"Prediction range: ${y_pred_real.min():,.2f} - ${y_pred_real.max():,.2f}")

# Step 4: Create sample test cases with decision engine
print("\n" + "=" * 70)
print("STEP 4: Running Decision Engine on Sample Cases")
print("=" * 70)

# Create sample scenarios for demonstration
sample_scenarios = [
    {
        'name': 'Scenario A: Clean Record, Gold Customer',
        'predicted_cost': y_pred_real[0],
        'input_data': {
            'DUI': False,
            'valid_license': True,
            'fraud_indicator': False,
            'policy_expired': False,
            'authorized_driver': True,
            'illegal_activity': False,
            'commercial_use': False,
            'street_racing': False,
            'roadworthy': True,
            'geographic_exclusion': False,
            'fault_percentage': 0,
            'speeding_penalty': 0,
            'distracted_driving': False,
            'dashcam': True,
            'failure_to_mitigate': False,
            'preexisting_damage_pct': 0,
            'depreciation_pct': 0,
            'salvage_value': 0,
            'oem_parts': True,
            'policy_tier': 'Gold',
            'customer_tenure': 7,
            'previous_claims': 1,
            'accident_forgiveness': False
        }
    },
    {
        'name': 'Scenario B: At-Fault, No Dashcam',
        'predicted_cost': y_pred_real[1],
        'input_data': {
            'DUI': False,
            'valid_license': True,
            'fraud_indicator': False,
            'policy_expired': False,
            'authorized_driver': True,
            'illegal_activity': False,
            'commercial_use': False,
            'street_racing': False,
            'roadworthy': True,
            'geographic_exclusion': False,
            'fault_percentage': 30,
            'speeding_penalty': 20,
            'distracted_driving': True,
            'dashcam': False,
            'failure_to_mitigate': False,
            'preexisting_damage_pct': 0,
            'depreciation_pct': 10,
            'salvage_value': 0,
            'oem_parts': True,
            'policy_tier': 'Basic',
            'customer_tenure': 2,
            'previous_claims': 2,
            'accident_forgiveness': False
        }
    },
    {
        'name': 'Scenario C: Platinum Customer, Minor Claim',
        'predicted_cost': y_pred_real[2],
        'input_data': {
            'DUI': False,
            'valid_license': True,
            'fraud_indicator': False,
            'policy_expired': False,
            'authorized_driver': True,
            'illegal_activity': False,
            'commercial_use': False,
            'street_racing': False,
            'roadworthy': True,
            'geographic_exclusion': False,
            'fault_percentage': 10,
            'speeding_penalty': 0,
            'distracted_driving': False,
            'dashcam': True,
            'failure_to_mitigate': False,
            'preexisting_damage_pct': 0,
            'depreciation_pct': 5,
            'salvage_value': 0,
            'oem_parts': True,
            'policy_tier': 'Platinum',
            'customer_tenure': 12,
            'previous_claims': 0,
            'accident_forgiveness': True
        }
    },
    {
        'name': 'Scenario D: DENIED - DUI + No License',
        'predicted_cost': y_pred_real[3],
        'input_data': {
            'DUI': True,
            'valid_license': False,
            'fraud_indicator': False,
            'policy_expired': False,
            'authorized_driver': True,
            'illegal_activity': False,
            'commercial_use': False,
            'street_racing': False,
            'roadworthy': True,
            'geographic_exclusion': False,
            'fault_percentage': 0,
            'speeding_penalty': 0,
            'distracted_driving': False,
            'dashcam': True,
            'failure_to_mitigate': False,
            'preexisting_damage_pct': 0,
            'depreciation_pct': 0,
            'salvage_value': 0,
            'oem_parts': True,
            'policy_tier': 'Gold',
            'customer_tenure': 5,
            'previous_claims': 0,
            'accident_forgiveness': False
        }
    }
]

# Process each scenario
for scenario in sample_scenarios:
    print("\n" + "-" * 70)
    print(scenario['name'])
    print("-" * 70)
    
    result = calculate_final_payout(
        scenario['predicted_cost'],
        scenario['input_data']
    )
    
    print(f"\nML PREDICTED COST: ${result['base_amount']:,.2f}")
    
    if result['is_denied']:
        print(f"\n*** CLAIM DENIED ***")
        print(f"Reason: {result['denial_reason']}")
        print(f"FINAL PAYOUT: $0.00")
    else:
        print(f"\n--- REDUCTIONS APPLIED ---")
        if result['reductions']:
            for reason, amount in result['reductions'].items():
                print(f"  - {reason}: -${amount:,.2f}")
            print(f"  Subtotal after reductions: ${result['reduced_amount']:,.2f}")
        else:
            print("  No reductions applied")
        
        print(f"\n--- LOYALTY ADJUSTMENTS ---")
        if result['loyalty_benefits']:
            for benefit in result['loyalty_benefits']:
                print(f"  + {benefit}")
        else:
            print("  No loyalty benefits applied")
        
        print(f"\n========================================")
        print(f"FINAL APPROVED PAYOUT: ${result['final_payout']:,.2f}")
        print(f"========================================")
        print(f"\nExplanation: {result['explanation']}")

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY: ML PREDICTION vs FINAL INSURANCE DECISION")
print("=" * 70)
print(f"\n{'Scenario':<35} {'ML Prediction':>15} {'Final Payout':>15}")
print("-" * 70)
for scenario in sample_scenarios:
    result = calculate_final_payout(
        scenario['predicted_cost'],
        scenario['input_data']
    )
    name = scenario['name'][:33]
    ml_pred = f"${result['base_amount']:,.2f}"
    final = f"${result['final_payout']:,.2f}" if not result['is_denied'] else "$0.00 (DENIED)"
    print(f"{name:<35} {ml_pred:>15} {final:>15}")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
print("""
How ML Prediction is Converted to Business Decision:
-------------------------------------------------------
1. ML model predicts claim cost (in log scale, then converted to real dollars)
2. Decision engine applies business rules:
   - Hard denial rules (DUI, no license, fraud, etc.) → immediate rejection
   - Reduction rules (fault %, speeding, no evidence) → percentage penalties
   - Loyalty adjustments (tier, tenure, claim history) → bonus percentages
3. Final payout = ML prediction - reductions + loyalty bonuses

Key Business Logic:
------------------
- Clean records with high loyalty = higher payouts
- At-fault accidents with no evidence = significant reductions
- Major violations (DUI, fraud) = automatic denial
- Customer retention through tier benefits and tenure rewards
""")
