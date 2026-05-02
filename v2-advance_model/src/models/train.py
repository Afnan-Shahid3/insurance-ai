import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Define paths
DATA_DIR = os.path.join('v2-advance_model', 'data', 'processed')
MODEL_DIR = os.path.join('v2-advance_model', 'models')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load training data
print("Loading training data...")
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
flag_train = pd.read_csv(os.path.join(DATA_DIR, 'train_claim_flag.csv'))
amount_train = pd.read_csv(os.path.join(DATA_DIR, 'train_claim_amount.csv'))

# Extract targets
y_flag = flag_train['claim_flag']
y_amount = amount_train['claim_amount']

# Stage 1: Classifier - Predict claim_flag
print("\n" + "=" * 60)
print("TRAINING CLASSIFICATION MODEL (claim_flag)")
print("=" * 60)
print(f"Training samples: {len(X_train)}")
print(f"Class distribution: {y_flag.value_counts().to_dict()}")
print(f"Positive class ratio: {y_flag.mean():.2%}")

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
classifier.fit(X_train, y_flag)

# Save classifier
classifier_path = os.path.join(MODEL_DIR, 'classifier.pkl')
joblib.dump(classifier, classifier_path)
print(f"Saved classifier to: {classifier_path}")

# Stage 2: Regressor - Predict claim_amount (only positive claims)
print("\n" + "=" * 60)
print("TRAINING REGRESSION MODEL (claim_amount - positive only)")
print("=" * 60)

# train_claim_amount.csv has only positive rows, so align by positive mask.
positive_mask = y_flag == 1
X_train_pos = X_train.loc[positive_mask].reset_index(drop=True)
y_amount_pos = y_amount.reset_index(drop=True)

if len(X_train_pos) != len(y_amount_pos):
    raise ValueError(
        f"Mismatch in positive training samples: features={len(X_train_pos)}, "
        f"targets={len(y_amount_pos)}"
    )

print(f"Positive claims for regression: {len(X_train_pos)}")
print(f"Target min: {y_amount_pos.min():.4f}, max: {y_amount_pos.max():.4f}, mean: {y_amount_pos.mean():.4f}")

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
regressor.fit(X_train_pos, y_amount_pos)

# Save regressor
regressor_path = os.path.join(MODEL_DIR, 'regressor.pkl')
joblib.dump(regressor, regressor_path)
print(f"Saved regressor to: {regressor_path}")

# Save feature names
feature_names_path = os.path.join(MODEL_DIR, 'feature_names.txt')
with open(feature_names_path, 'w') as f:
    for name in X_train.columns:
        f.write(name + '\n')
print(f"Saved feature names to: {feature_names_path}")

print("\nTraining complete!")