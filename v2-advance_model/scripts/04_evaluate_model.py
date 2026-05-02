import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_DIR = os.path.join('v2-advance_model', 'models')
DATA_DIR = os.path.join('v2-advance_model', 'data', 'processed')

print("Loading models...")
classifier = joblib.load(os.path.join(MODEL_DIR, 'classifier.pkl'))
regressor = joblib.load(os.path.join(MODEL_DIR, 'regressor.pkl'))

print("Loading test data...")
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
flag_test = pd.read_csv(os.path.join(DATA_DIR, 'test_claim_flag.csv'))
amount_test = pd.read_csv(os.path.join(DATA_DIR, 'test_claim_amount.csv'))
original_test = pd.read_csv(os.path.join(DATA_DIR, 'test_target_original.csv'))

y_flag_test = flag_test['claim_flag']
y_amount_test = amount_test['claim_amount']
y_original_test = original_test['original_claim']

print("Running prediction pipeline...")
flag_pred = classifier.predict(X_test)

mask_pred_positive = flag_pred == 1
X_test_positive = X_test[mask_pred_positive]

amount_pred = np.zeros(len(X_test))
if len(X_test_positive) > 0:
    amount_pred_log = regressor.predict(X_test_positive)
    amount_pred_positive = np.expm1(amount_pred_log)
    amount_pred[mask_pred_positive] = amount_pred_positive

print("=" * 50)
print("CLASSIFIER RESULTS")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_flag_test, flag_pred):.4f}")
print(f"Precision: {precision_score(y_flag_test, flag_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_flag_test, flag_pred, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_flag_test, flag_pred, zero_division=0):.4f}")

print("\n" + "=" * 50)
print("REGRESSION RESULTS")
print("=" * 50)
mask_actual_positive = y_flag_test == 1
y_true_positive = y_original_test[mask_actual_positive]
y_pred_positive = amount_pred[mask_actual_positive]

if len(y_true_positive) > 0:
    print(f"MAE: ${mean_absolute_error(y_true_positive, y_pred_positive):.2f}")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_true_positive, y_pred_positive)):.2f}")
    print(f"R² Score: {r2_score(y_true_positive, y_pred_positive):.4f}")
else:
    print("No actual positive claims for regression metrics")

print("\n" + "=" * 50)
print("PREDICTION DEBUG")
print("=" * 50)
print(f"Min predicted claim amount: ${amount_pred[amount_pred > 0].min():.2f}" if any(amount_pred > 0) else "Min predicted: $0.00")
print(f"Max predicted claim amount: ${amount_pred.max():.2f}")
print(f"Mean predicted claim amount: ${amount_pred.mean():.2f}")
print(f"Percentage of zero predictions: {(flag_pred == 0).mean() * 100:.2f}%")
print(f"Number of predicted claims: {mask_pred_positive.sum()}")
print(f"Number of actual claims: {mask_actual_positive.sum()}")

print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
if np.all(flag_pred == 0):
    print("WARNING: All predictions are zero claims!")
elif np.var(amount_pred) < 1e-5:
    print("WARNING: Model collapsed - near-zero variance in predictions!")
else:
    print("Sanity checks passed")

if len(y_true_positive) > 0:
    positive_errors = np.abs(y_true_positive - y_pred_positive)
    print(f"Average error on positive claims: ${positive_errors.mean():.2f}")