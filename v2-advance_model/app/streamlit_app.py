"""Streamlit Web App for Insurance Claim Prediction"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# ----------------------------------------------------------------------
# Project paths & imports
# ----------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.decision_engine import calculate_final_payout

MODEL_PATH = "v2-advance_model/models/saved_models/best_model.pkl"
TRAIN_FEATURES_PATH = "v2-advance_model/data/processed/train_features.csv"

# ----------------------------------------------------------------------
# Caching helpers
# ----------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    """Load the trained model from disk."""
    return joblib.load(model_path)


@st.cache_data
def load_training_columns():
    """Return the ordered list of feature column names."""
    train_features = pd.read_csv(TRAIN_FEATURES_PATH)
    return train_features.columns.tolist()


@st.cache_data
def load_training_dataframe():
    """Load the entire training dataframe – needed for categorical mapping."""
    return pd.read_csv(TRAIN_FEATURES_PATH)


def encode_categorical(df: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any column that was *object* in the original training set
    to the integer codes used during model training.
    Unseen categories are mapped to ``0``.
    """
    # Columns that were categorical when the model was trained
    categorical_cols = training_df.select_dtypes(include=["object"]).columns

    for col in df.columns:
        if col in categorical_cols:
            # Preserve the exact ordering of categories from training
            categories = pd.Categorical(training_df[col]).categories
            # Align current values to those categories; unseen -> -1 => 0
            coded = pd.Categorical(df[col].astype(str), categories=categories).codes
            coded = np.where(coded == -1, 0, coded)
            df[col] = coded.astype(int)
    return df


# ----------------------------------------------------------------------
# Manual encoding mappings (matching training data encoding)
# ----------------------------------------------------------------------
CAR_TYPE_MAP = {"Sedan": 0, "SUV": 1, "Truck": 2, "Sports Car": 3, "Compact": 4, "Luxury": 5}
POLICY_TIER_MAP = {"Basic": 0, "Gold": 1, "Platinum": 2}
SEVERITY_MAP = {"Minor": 0, "Moderate": 1, "Major": 2, "Total Loss": 3}


def encode_user_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical string values to integers based on the mapping
    used during model training.
    """
    # Convert all columns to object first to allow mixed assignments
    df = df.astype(object)
    
    for col in df.columns:
        col_l = col.lower()
        val = df.loc[0, col]
        
        # Skip if already numeric
        if isinstance(val, (int, float)):
            continue
            
        # Map car_type
        if "car_type" in col_l or "vehicle_type" in col_l:
            if isinstance(val, str) and val in CAR_TYPE_MAP:
                df.loc[0, col] = CAR_TYPE_MAP[val]
        
        # Map policy_tier
        elif "tier" in col_l:
            if isinstance(val, str) and val in POLICY_TIER_MAP:
                df.loc[0, col] = POLICY_TIER_MAP[val]
        
        # Map accident_severity
        elif "severity" in col_l:
            if isinstance(val, str) and val in SEVERITY_MAP:
                df.loc[0, col] = SEVERITY_MAP[val]
    
    # Convert all columns to numeric (coerce errors to 0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df


def create_input_dataframe(user_inputs: dict, training_columns: list) -> pd.DataFrame:
    """
    Build a one‑row DataFrame that matches the training schema.
    Uses training data median values as defaults for unmapped fields.
    """
    # Default values based on training data median (approximate)
    # These will be overridden by user inputs where applicable
    default_values = {
        'KIDSDRIV': 0,
        'BIRTH': 3000,      # Birth date proxy
        'AGE': 45,
        'HOMEKIDS': 0,
        'YOJ': 5,           # Years on job - similar to tenure
        'INCOME': 3500,
        'PARENT1': 0,
        'HOME_VAL': 2000,
        'MSTATUS': 0,
        'GENDER': 1,
        'EDUCATION': 2,
        'OCCUPATION': 4,
        'TRAVTIME': 30,
        'CAR_USE': 1,
        'BLUEBOOK': 1300,
        'TIF': 5,           # Time in force - similar to customer tenure
        'CAR_TYPE': 0,
        'RED_CAR': 0,
        'OLDCLAIM': 0,
        'CLM_FREQ': 0,
        'REVOKED': 0,
        'MVR_PTS': 1,
        'CAR_AGE': 8,
        'CLAIM_FLAG': 0,
        'URBANICITY': 0,
    }
    
    # Start with default values
    data = {col: default_values.get(col, 0) for col in training_columns}

    # UI field → exact column names in the training data
    feature_mapping = {
        "age": ["AGE"],
        "income": ["INCOME"],
        "car_type": ["CAR_TYPE"],
        "car_age": ["CAR_AGE"],
        "overspeeding": ["MVR_PTS"],
        "distracted_driving": ["CLM_FREQ"],
        "has_dashcam": ["RED_CAR"],
        "policy_tier": [],
        "customer_tenure": ["TIF", "YOJ"],
        "fault_percentage": ["CLM_FREQ"],
    }

    for ui_key, ui_val in user_inputs.items():
        if ui_key not in feature_mapping:
            continue
        
        # Apply scaling to match training data ranges
        # Training INCOME max ~8150, user enters 50000 -> scale by 0.16
        scale_factors = {
            'income': 0.16,      # Scale down: 50000 * 0.16 = 8000
            'age': 1.0,
            'car_age': 1.0,
            'customer_tenure': 1.0,
        }
        scale = scale_factors.get(ui_key, 1.0)
        scaled_val = int(ui_val * scale) if scale != 1.0 else ui_val
        
        for col in feature_mapping[ui_key]:
            if col in training_columns:
                data[col] = scaled_val
    
    # Create DataFrame from dict
    df = pd.DataFrame([data])
    return df


# ----------------------------------------------------------------------
# Streamlit UI layout
# ----------------------------------------------------------------------
st.set_page_config(page_title="Insurance Claim Calculator",
                   page_icon="🛡️",
                   layout="centered")
st.title("🛡️ Insurance Claim Calculator")
st.markdown("---")

# ---- Sidebar: model status ------------------------------------------------
with st.sidebar:
    st.header("Model Info")
    st.info("Using the trained ML model")
    try:
        _ = load_model(MODEL_PATH)
        st.success("Model Loaded")
    except Exception as exc:
        st.error(f"Problem loading model: {exc}")

# ---- Main form -----------------------------------------------------------
st.subheader("Claim Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual Income ($)", min_value=0,
                             value=50000, step=1000)
    car_type = st.selectbox(
        "Car Type",
        ["Sedan", "SUV", "Truck", "Sports Car", "Compact", "Luxury"]
    )
    car_age = st.number_input("Car Age (years)", min_value=0,
                              max_value=30, value=5)
    accident_severity = st.selectbox(
        "Accident Severity",
        ["Minor", "Moderate", "Major", "Total Loss"]
    )

with col2:
    overspeeding = st.radio("Overspeeding?", ["No", "Yes"])
    distracted_driving = st.radio("Distracted Driving?", ["No", "Yes"])
    has_dashcam = st.radio("Has Dashcam?", ["Yes", "No"])
    policy_tier = st.selectbox("Policy Tier", ["Basic", "Gold", "Platinum"])
    customer_tenure = st.number_input(
        "Customer Tenure (years)", min_value=0, max_value=50, value=3
    )
    fault_percentage = st.slider("Fault Percentage", 0, 100, 0)

st.markdown("---")

# ----------------------------------------------------------------------
# Prediction & decision engine
# ----------------------------------------------------------------------
if st.button("Calculate Claim", type="primary"):
    try:
        # Load model + training metadata
        model = load_model(MODEL_PATH)
        training_columns = load_training_columns()
        training_df = load_training_dataframe()

        # --------------------------------------------------------------
        # 1️⃣ Gather UI inputs
        # --------------------------------------------------------------
        user_inputs = {
            "age": age,
            "income": income,
            "car_type": car_type,
            "car_age": car_age,
            "accident_severity": accident_severity,
            "overspeeding": 1 if overspeeding == "Yes" else 0,
            "distracted_driving": 1 if distracted_driving == "Yes" else 0,
            "has_dashcam": 1 if has_dashcam == "Yes" else 0,
            "policy_tier": policy_tier,
            "customer_tenure": customer_tenure,
            "fault_percentage": fault_percentage,
        }

        # --------------------------------------------------------------
        # 2️⃣ Build a dataframe that matches the training schema
        # --------------------------------------------------------------
        input_df = create_input_dataframe(user_inputs, training_columns)

        # --------------------------------------------------------------
        # 3️⃣ Encode categorical columns (the step that previously broke)
        # --------------------------------------------------------------
        input_df = encode_categorical(input_df, training_df)
        input_df = encode_user_inputs(input_df)

        # --------------------------------------------------------------
        # 4️⃣ Predict claim cost (model already outputs real dollars)
        # --------------------------------------------------------------
        predicted_cost = model.predict(input_df)[0]
        predicted_cost = np.expm1(predicted_cost)
        predicted_cost = max(0, predicted_cost)

        st.subheader("Prediction Results")
        st.success(f"Predicted Claim Cost: ${predicted_cost:,.2f}")

        # --------------------------------------------------------------
        # 5️⃣ Feed prediction into the decision engine
        # --------------------------------------------------------------
        decision_input = {
            "DUI": False,
            "valid_license": True,
            "fraud_indicator": False,
            "policy_expired": False,
            "authorized_driver": True,
            "illegal_activity": False,
            "commercial_use": False,
            "commercial_coverage": True,
            "street_racing": overspeeding == "Yes",
            "roadworthy": True,
            "geographic_exclusion": False,
            "fault_percentage": fault_percentage,
            "speeding_penalty": 20 if overspeeding == "Yes" else 0,
            "distracted_driving": distracted_driving == "Yes",
            "dashcam": has_dashcam == "Yes",
            "failure_to_mitigate": False,
            "preexisting_damage_pct": 0,
            "depreciation_pct": min(car_age * 2, 40),
            "salvage_value": 0,
            "oem_parts": True,
            "policy_tier": policy_tier,
            "customer_tenure": customer_tenure,
            "previous_claims": 0,
            "accident_forgiveness": False,
        }

        result = calculate_final_payout(predicted_cost, decision_input)

        # --------------------------------------------------------------
        # 6️⃣ Display the final decision
        # --------------------------------------------------------------
        st.markdown("---")
        st.subheader("Final Decision")

        if result["is_denied"]:
            st.error("CLAIM DENIED")
            st.write(f"Reason: {result['denial_reason']}")
        else:
            if result["reductions"]:
                st.warning("Reductions Applied:")
                for name, amount in result["reductions"].items():
                    st.write(f"- {name}: -${amount:,.2f}")
                st.write(
                    f"Amount after reductions: ${result['reduced_amount']:,.2f}"
                )

            if result["loyalty_benefits"]:
                st.info("Loyalty Benefits:")
                for benefit in result["loyalty_benefits"]:
                    st.write(f"- {benefit}")

            st.success(f"Final Approved Payout: ${result['final_payout']:,.2f}")

    except Exception as exc:
        st.error(f"Error during calculation: {exc}")

st.markdown("---")
st.caption("Insurance AI V2 – Claim Prediction System")
