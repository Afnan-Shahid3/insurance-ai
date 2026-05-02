import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, precision_score, recall_score

# Ensure imports for wrapper class from src.models
SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
SYS_PATH_ROOT = os.path.dirname(SCRIPT_ROOT)
sys.path.insert(0, SYS_PATH_ROOT)
from src.models.model_wrappers import ProbabilityThresholdClassifier

# Define paths
DATA_DIR = os.path.join('v2-advance_model', 'data', 'processed')
MODEL_DIR = os.path.join('v2-advance_model', 'models')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load training data
print("Loading training data...")
X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
claim_flag_train = pd.read_csv(os.path.join(DATA_DIR, 'train_claim_flag.csv'))
claim_amount_train = pd.read_csv(os.path.join(DATA_DIR, 'train_claim_amount.csv'))

# Extract targets
y_flag = claim_flag_train['claim_flag'].astype(int)
y_amount = pd.to_numeric(claim_amount_train['claim_amount'], errors='coerce')

# Stage 1: Classifier - Predict claim_flag
print("\n" + "=" * 60)
print("TRAINING claim_flag CLASSIFIER")
print("=" * 60)
print(f"Total training rows: {len(X_train)}")
print("Class distribution:")
flag_counts = y_flag.value_counts().sort_index()
for target_class, count in flag_counts.items():
    print(f"  {target_class}: {count} ({count / len(X_train):.2%})")
print(f"Positive class ratio: {y_flag.mean():.2%}")

base_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
base_classifier.fit(X_train, y_flag)

# Evaluate raw predicted probabilities on training data
proba = base_classifier.predict_proba(X_train)[:, 1]
avg_prob = float(np.mean(proba))
default_pct = float(np.mean(proba >= 0.5) * 100)
print(f"Average predicted probability for class 1 (training data): {avg_prob:.4f}")
print(f"% of training rows above default threshold 0.50: {default_pct:.2f}%")

# Histogram of predicted probabilities
hist, bin_edges = np.histogram(proba, bins=np.linspace(0.0, 1.0, 11))
print("Predicted probability histogram for class 1:")
for low, high, count in zip(bin_edges[:-1], bin_edges[1:], hist):
    print(f"  {low:.2f}-{high:.2f}: {count}")

best_threshold = 0.3
best_f1 = -1.0
best_metrics = None
for threshold in np.arange(0.2, 0.41, 0.05):
    preds = (proba >= threshold).astype(int)
    precision = precision_score(y_flag, preds, zero_division=0)
    recall = recall_score(y_flag, preds, zero_division=0)
    f1 = f1_score(y_flag, preds, zero_division=0)
    count_above = int(preds.sum())
    pct_above = float(count_above / len(proba) * 100)
    print(
        f"Threshold {threshold:.2f}: precision={precision:.3f}, recall={recall:.3f}, "
        f"f1={f1:.3f}, positives={count_above} ({pct_above:.2f}%)"
    )
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(threshold)
        best_metrics = (precision, recall, f1, pct_above)

print(f"Selected tuned threshold: {best_threshold:.2f} (best F1={best_f1:.3f})")
if best_metrics is not None:
    print(f"% predictions above tuned threshold: {best_metrics[3]:.2f}%")

classifier = ProbabilityThresholdClassifier(
    base_classifier=base_classifier,
    threshold=best_threshold
)

# Save classifier
classifier_path = os.path.join(MODEL_DIR, 'classifier.pkl')
with open(classifier_path, 'wb') as f:
    pickle.dump(classifier, f)

# Stage 2: Regressor - Predict claim_amount
print("Training claim amount regressor...")
# train_claim_amount.csv contains only positive claim rows,
# so align positive feature rows with positive target rows explicitly.
positive_mask = y_flag == 1
X_train_pos = X_train.loc[positive_mask].reset_index(drop=True)
y_amount_pos = y_amount.reset_index(drop=True)

if X_train_pos.empty or y_amount_pos.empty:
    raise ValueError('No positive claim rows available for regression training.')

if len(X_train_pos) != len(y_amount_pos):
    raise ValueError(
        f"Positive feature rows ({len(X_train_pos)}) and positive target rows "
        f"({len(y_amount_pos)}) do not match."
    )

print(f"Positive training rows for regression: {len(X_train_pos)}")
print(f"Claim amount target sample range: {y_amount_pos.min():.4f} - {y_amount_pos.max():.4f}")

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
regressor.fit(X_train_pos, y_amount_pos)

# Save regressor
regressor_path = os.path.join(MODEL_DIR, 'regressor.pkl')
with open(regressor_path, 'wb') as f:
    pickle.dump(regressor, f)

# Save feature names
feature_names_path = os.path.join(MODEL_DIR, 'feature_names.txt')
with open(feature_names_path, 'w') as f:
    for name in X_train.columns:
        f.write(name + '\n')

print("Training complete!")
print(f"Saved classifier to: {classifier_path}")
print(f"Saved regressor to: {regressor_path}")
print(f"Saved feature names to: {feature_names_path}")