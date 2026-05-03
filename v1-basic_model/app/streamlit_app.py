"""
Streamlit Web App for Insurance Claim Prediction
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.policy_engine.insurance_rules import (
    DenialFlags,
    ReductionFactors,
    evaluate_claim,
)

# Page configuration
st.set_page_config(
    page_title="Insurance Claim Predictor",
    layout="centered"
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def load_model(model_path: str):
    """Load the trained model from pickle file."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_training_columns():
    """Load the feature column names from training data."""
    train_features_path = BASE_DIR / "data" / "processed" / "train_features.csv"
    train_features = pd.read_csv(train_features_path)
    return train_features.columns.tolist()


def get_risk_level(predicted_cost: float) -> tuple:
    """
    Classify risk level based on predicted claim cost.
    
    Returns:
        tuple: (risk_level, color)
    """
    if predicted_cost < 10000:
        return "LOW", "green"
    elif predicted_cost < 30000:
        return "MEDIUM", "orange"
    else:
        return "HIGH", "red"


def create_input_dataframe(user_inputs: dict, training_columns: list) -> pd.DataFrame:
    """
    Convert user inputs to model-ready DataFrame.
    
    Args:
        user_inputs: Dictionary of user input values
        training_columns: List of column names from training
        
    Returns:
        pd.DataFrame: Properly formatted input for the model
    """
    # Start with a template from training data (all zeros)
    df = pd.DataFrame(columns=training_columns)
    df.loc[0] = 0  # Initialize with zeros
    
    # Fill in numeric values
    numeric_fields = [
        'months_as_customer', 'age', 'policy_number', 'policy_bind_date',
        'policy_annual_premium', 'policy_deductable', 'umbrella_limit',
        'insured_zip', 'capital-gains', 'capital-loss',
        'incident_hour_of_the_day', 'number_of_vehicles_involved',
        'property_damage', 'bodily_injuries', 'witnesses',
        'auto_year', 'total_claim_amount'
    ]
    
    for field in numeric_fields:
        if field in user_inputs and field in df.columns:
            df[field] = user_inputs[field]
    
    # Fill in categorical values (one-hot encoded)
    categorical_mappings = {
        'policy_state': ['policy_state_OH', 'policy_state_ND', 'policy_state_HV',
                        'policy_state_NM', 'policy_state_WV', 'policy_state_IA'],
        'policy_csl': ['policy_csl_250/500', 'policy_csl_500/1000', 'policy_csl_1000/3000'],
        'insured_sex': ['insured_sex_MALE', 'insured_sex_FEMALE'],
        'insured_education_level': ['insured_education_level_College',
                                    'insured_education_level_High School',
                                    'insured_education_level_Masters',
                                    'insured_education_level_PhD',
                                    'insured_education_level_JD',
                                    'insured_education_level_MD'],
        'insured_occupation': ['insured_occupation_tech-support', 'insured_occupation_farming-fishing',
                               'insured_occupation_prof-specialty', 'insured_occupation_other-service',
                               'insured_occupation_exec-managerial', 'insured_occupation_craft-repair',
                               'insured_occupation_transport-moving', 'insured_occupation_handlers-cleaners',
                               'insured_occupation_machine-op-inspct', 'insured_occupation_adm-clerical',
                               'insured_occupation_sales', 'insured_occupation_protective-serv'],
        'insured_hobbies': ['insured_hobbies_basketball', 'insured_hobbies_board-games',
                           'insured_hobbies_bungee-jumping', 'insured_hobbies_camping',
                           'insured_hobbies_chess', 'insured_hobbies_cross-fit',
                           'insured_hobbies_dancing', 'insured_hobbies_golf',
                           'insured_hobbies_hunting', 'insured_hobbies_kayaking',
                           'insured_hobbies_martial-arts', 'insured_hobbies_paintball',
                           'insured_hobbies_polo', 'insured_hobbies_reading',
                           'insured_hobbies_skydiving', 'insured_hobbies_snowboard',
                           'insured_hobbies_yachting'],
        'insured_relationship': ['_husband', '_not-in-family', '_own-child', '_unmarried', '_wife'],
        'incident_type': ['incident_type_Parked Car', 'incident_type_Vehicle Theft',
                         'incident_type_Multi-vehicle Collision',
                         'incident_type_Single Vehicle Collision'],
        'collision_type': ['collision_type_Front Collision', 'collision_type_Rear Collision',
                          'collision_type_Side Collision'],
        'incident_severity': ['incident_severity_Trivial Damage', 'incident_severity_Minor Damage',
                             'incident_severity_Major Damage', 'incident_severity_Total Loss'],
        'authorities_contacted': ['authorities_contacted_Ambulance', 'authorities_contacted_Fire',
                                   'authorities_contacted_None', 'authorities_contacted_Other',
                                   'authorities_contacted_Police'],
        'incident_state': ['incident_state_OH', 'incident_state_NY', 'incident_state_PA',
                          'incident_state_WA', 'incident_state_IN', 'incident_state_MI',
                          'incident_state_IL'],
        'auto_make': ['auto_make_Dodge', 'auto_make_Saab', 'auto_make_Suburu',
                     'auto_make_Nissan', 'auto_make_Mercury', 'auto_make_Audi',
                     'auto_make_Toyota', 'auto_make_Ford', 'auto_make_BMW',
                     'auto_make_Mercedes', 'auto_make_Honda', 'auto_make_Jeep',
                     'auto_make_Chevrolet', 'auto_make_Volkswagen']
    }
    
    # Set categorical columns to 1 where they match
    for cat_field, one_hot_cols in categorical_mappings.items():
        if cat_field in user_inputs:
            selected_value = user_inputs[cat_field]
            # Find the matching one-hot column
            for col in one_hot_cols:
                if col in df.columns:
                    if selected_value in col:
                        df[col] = 1
    
    # Remove target column if present
    if 'total_claim_amount' in df.columns:
        df = df.drop(columns=['total_claim_amount'])
    
    return df


def process_claim(prediction: float, inputs: dict) -> dict:
    """
    Run the full claim pipeline against the ML prediction.

    Pipeline:
        1. base_payout = ML prediction
        2. Denial check — any True flag → final_payout = 0, pipeline stops
        3. Reduction engine — multiplicative penalties applied to base_payout
        4. Return structured result

    Args:
        prediction: Raw ML model output in dollars — used directly as base payout.
        inputs:     Flat dict of denial flags, reduction factors, and their values.

    Returns:
        {
            "final_payout": float,
            "base_payout":  float,
            "denied":       bool,
            "reason":       str | None,
            "adjustments":  list[dict],
        }
    """
    denial_flags = DenialFlags(
        dui_dwi=inputs.get("dui_dwi", False),
        no_valid_license=inputs.get("no_valid_license", False),
        fraud_or_staged_accident=inputs.get("fraud_or_staged_accident", False),
        commercial_use_not_covered=inputs.get("commercial_use_not_covered", False),
        racing_or_illegal_driving=inputs.get("racing_or_illegal_driving", False),
        geographic_exclusion=inputs.get("geographic_exclusion", False),
        lapse_in_coverage=inputs.get("lapse_in_coverage", False),
        intentional_damage=inputs.get("intentional_damage", False),
    )

    reduction_factors = ReductionFactors(
        comparative_negligence_pct=float(inputs.get("comparative_negligence", 0)),
        overspeeding=inputs.get("overspeeding", False),
        distracted_driving=inputs.get("distracted_driving", False),
        traffic_violations=inputs.get("traffic_violations", False),
        no_dashcam=inputs.get("no_dashcam_evidence", False),
        failure_to_mitigate=inputs.get("failure_to_mitigate", False),
        poor_maintenance=inputs.get("poor_maintenance", False),
        unauthorized_repair=inputs.get("unauthorized_repair", False),
        salvage_value=float(inputs.get("salvage_value", 0)),
    )

    result = evaluate_claim(
        ml_damage_estimate=float(prediction),
        denial_flags=denial_flags,
        reduction_factors=reduction_factors,
    )

    return {
        "final_payout": result["final_payout"],
        "base_payout":  result["ml_damage_estimate"],
        "denied":       result["denied"],
        "reason":       result["denial_reason"],
        "adjustments":  result["adjustments"],
    }


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Title and description
    st.title("Insurance Claim Cost Predictor")
    st.markdown("""
    **AI-Powered Insurance Claim Prediction System**
    
    This machine learning model predicts the expected cost of an insurance claim based on 
    various factors including driver information, vehicle details, and incident characteristics.
    """)
    
    st.divider()
    
    # Check if model exists
    model_path = BASE_DIR / "models" / "saved_models" / "best_model.pkl"
    
    if not model_path.exists():
        st.error(f"Model not found at {model_path}. Please run training first!")
        st.info("Run: python scripts/03_train_model.py")
        return
    
    # Load model and training columns
    try:
        model = load_model(str(model_path))
        training_columns = load_training_columns()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    # =============================================================================
    # INPUT FORM
    # =============================================================================
    
    st.subheader("Enter Claim Details")
    
    # Customer Information
    st.markdown("### Customer Information")
    
    col1, col2 = st.columns(2)
    with col1:
        months_as_customer = st.number_input(
            "Months as Customer",
            min_value=0,
            max_value=600,
            value=100,
            help="How long the customer has been with the company"
        )
    with col2:
        age = st.number_input(
            "Driver Age",
            min_value=18,
            max_value=100,
            value=35
        )
    
    insured_sex = st.selectbox(
        "Driver Gender",
        ["MALE", "FEMALE"]
    )
    
    insured_education = st.selectbox(
        "Education Level",
        ["High School", "College", "Masters", "PhD", "JD", "MD"]
    )
    
    insured_occupation = st.selectbox(
        "Occupation",
        ["tech-support", "farming-fishing", "prof-specialty", "other-service",
         "exec-managerial", "craft-repair", "transport-moving", "handlers-cleaners",
         "machine-op-inspct", "adm-clerical", "sales", "protective-serv"]
    )
    
    # Policy Information
    st.markdown("### Policy Information")
    
    col1, col2 = st.columns(2)
    with col1:
        policy_annual_premium = st.number_input(
            "Annual Premium ($)",
            min_value=0,
            value=1500,
            step=100
        )
    with col2:
        policy_deductable = st.number_input(
            "Deductible ($)",
            min_value=0,
            value=1000,
            step=100
        )
    
    policy_state = st.selectbox(
        "Policy State",
        ["OH", "ND", "HV", "NM", "WV", "IA"]
    )
    
    # Vehicle Information
    st.markdown("### Vehicle Information")
    
    col1, col2 = st.columns(2)
    with col1:
        auto_make = st.selectbox(
            "Vehicle Make",
            ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Audi", 
             "Nissan", "Subaru", "Jeep", "Chevrolet", "Volkswagen", "Dodge", "Saab", "Mercury"]
        )
    with col2:
        auto_year = st.number_input(
            "Vehicle Year",
            min_value=1990,
            max_value=2024,
            value=2018
        )
    
    # Incident Information
    st.markdown("### Incident Information")
    
    incident_type = st.selectbox(
        "Incident Type",
        ["Vehicle Theft", "Parked Car", "Multi-vehicle Collision", "Single Vehicle Collision"]
    )
    
    incident_severity = st.selectbox(
        "Incident Severity",
        ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        bodily_injuries = st.slider(
            "Bodily Injuries",
            min_value=0,
            max_value=5,
            value=0
        )
    with col2:
        witnesses = st.slider(
            "Witnesses",
            min_value=0,
            max_value=3,
            value=1
        )

    st.markdown("### Policy Engine Inputs")

    st.markdown("#### Denial Conditions")
    col1, col2 = st.columns(2)
    with col1:
        dui_dwi                    = st.checkbox("DUI / DWI", value=False)
        no_valid_license           = st.checkbox("No license / expired license", value=False)
        fraud_or_staged_accident   = st.checkbox("Fraud / staged accident", value=False)
        lapse_in_coverage          = st.checkbox("Lapse in coverage", value=False)
    with col2:
        intentional_damage         = st.checkbox("Intentional damage", value=False)
        commercial_use_not_covered = st.checkbox("Commercial use not covered", value=False)
        racing_or_illegal_driving  = st.checkbox("Racing / illegal driving", value=False)
        geographic_exclusion       = st.checkbox("Geographic exclusion", value=False)

    st.markdown("#### Reduction Factors")
    comparative_negligence = st.slider(
        "Comparative negligence (% fault)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Percent fault assigned to the claimant."
    )
    col1, col2 = st.columns(2)
    with col1:
        overspeeding        = st.checkbox("Overspeeding", value=False)
        distracted_driving  = st.checkbox("Distracted driving", value=False)
        traffic_violations  = st.checkbox("Traffic violations", value=False)
        no_dashcam_evidence = st.checkbox("No dashcam evidence", value=False)
    with col2:
        failure_to_mitigate = st.checkbox("Failure to mitigate loss", value=False)
        poor_maintenance    = st.checkbox("Poor vehicle maintenance", value=False)
        unauthorized_repair = st.checkbox("Unauthorized repair", value=False)
    salvage_value = st.slider(
        "Salvage value deduction ($)",
        min_value=0,
        max_value=10000,
        value=0,
        step=100,
        help="Flat salvage deduction from the payout."
    )

    # =============================================================================
    # PREDICTION
    # =============================================================================
    
    st.divider()
    
    # Predict button
    if st.button("Predict Claim Cost", type="primary"):
        # Collect all inputs into a dictionary
        user_inputs = {
            'months_as_customer': months_as_customer,
            'age': age,
            'insured_sex': insured_sex,
            'insured_education_level': insured_education,
            'insured_occupation': insured_occupation,
            'policy_state': policy_state,
            'policy_annual_premium': policy_annual_premium,
            'policy_deductable': policy_deductable,
            'auto_make': auto_make,
            'auto_year': auto_year,
            'incident_type': incident_type,
            'incident_severity': incident_severity,
            'bodily_injuries': bodily_injuries,
            'witnesses': witnesses,
            'capital-gains': 0,
            'capital-loss': 0,
            'insured_zip': 12345,
            'policy_number': 1,
            'policy_bind_date': 1,
            'umbrella_limit': 0,
            'number_of_vehicles_involved': 1,
            'property_damage': 0,
            'incident_hour_of_the_day': 12,
            'incident_state': 'OH',
            'incident_city': 'city1',
            'incident_location': 'location1',
            'authorities_contacted': 'Police',
            'collision_type': 'Rear Collision',
            'insured_relationship': '_husband',
            'insured_hobbies': 'reading',
            'dui_dwi': dui_dwi,
            'no_valid_license': no_valid_license,
            'fraud_or_staged_accident': fraud_or_staged_accident,
            'commercial_use_not_covered': commercial_use_not_covered,
            'racing_or_illegal_driving': racing_or_illegal_driving,
            'geographic_exclusion': geographic_exclusion,
            'lapse_in_coverage': lapse_in_coverage,
            'intentional_damage': intentional_damage,
            'comparative_negligence': comparative_negligence,
            'overspeeding': overspeeding,
            'distracted_driving': distracted_driving,
            'traffic_violations': traffic_violations,
            'no_dashcam_evidence': no_dashcam_evidence,
            'failure_to_mitigate': failure_to_mitigate,
            'poor_maintenance': poor_maintenance,
            'unauthorized_repair': unauthorized_repair,
            'salvage_value': salvage_value,
        }
        
        # Create input DataFrame
        input_df = create_input_dataframe(user_inputs, training_columns)
        
        # Make prediction
        try:
            predicted_cost = model.predict(input_df)[0]
            policy_result  = process_claim(predicted_cost, user_inputs)
            final_payout   = policy_result['final_payout']
            denied         = policy_result['denied']
            denial_reason  = policy_result['reason']
            adjustments    = policy_result['adjustments']
            
            # Get risk level
            risk_level, risk_color = get_risk_level(predicted_cost)
            
            # Display results
            st.subheader("Prediction Results")
            
            # Predicted cost
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="margin: 0; color: #1f77b4;">Predicted Claim Cost</h2>
                <h1 style="margin: 10px 0; color: #2ecc71;">${predicted_cost:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Final payout
            st.markdown(f"""
            <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #dcdcdc;">
                <h2 style="margin: 0; color: #ff7f0e;">Final Payout</h2>
                <h1 style="margin: 10px 0; color: #d62728;">${final_payout:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            if denied:
                st.error(f"Denied: {denial_reason}")
            else:
                st.success("Claim approved subject to policy engine adjustments.")

            if adjustments:
                st.markdown("### Policy Adjustments Applied")
                for adjustment in adjustments:
                    st.write(f"• {adjustment['description']}")
            else:
                st.info("No policy adjustments were applied.")
            
            # Risk level
            st.markdown(f"""
            <div style="text-align: center; margin-top: 20px;">
                <h3>Risk Level: <span style="color: {risk_color};">{risk_level}</span></h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Explanation
            st.subheader("Explanation")
            st.markdown("""
            **Key factors that influence this prediction:**
            
            Based on the model's analysis, the following factors have the highest impact on the predicted claim cost:
            """)
            
            # Show top features
            importance_df = pd.DataFrame({
                'feature': training_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(5)
            
            for idx, row in importance_df.iterrows():
                st.write(f"• **{row['feature']}** (importance: {row['importance']*100:.2f}%)")
            
            st.info("The model uses machine learning to analyze patterns in historical insurance claims to predict the expected cost.")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write("Please check that the model was trained correctly.")
    
    # =============================================================================
    # FOOTER
    # =============================================================================
    
    st.divider()
    st.markdown("""
    ---
    **Project Details:**
    - Built with Python, Scikit-learn, and Streamlit
    - Model: Random Forest Regressor
    - Part of Intro to AI Course Project
    """)


if __name__ == "__main__":
    main()