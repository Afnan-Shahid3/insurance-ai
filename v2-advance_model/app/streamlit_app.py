"""Streamlit Web App for Insurance Claim Prediction"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import traceback
from typing import Any, Tuple

# ----------------------------------------------------------------------
# Project paths & imports
# ----------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.decision_engine import calculate_final_payout

CLASSIFIER_PATH    = os.path.join(project_root, 'models', 'classifier.pkl')
REGRESSOR_PATH     = os.path.join(project_root, 'models', 'regressor.pkl')
FEATURE_NAMES_PATH = os.path.join(project_root, 'models', 'feature_names.txt')

# ── THE KEY FIX ───────────────────────────────────────────────────────
# Most profiles score 0.10–0.25 probability.
# The old threshold of 0.30 caused ~63% of real claims to be missed.
# 0.20 gives the best F1 on this dataset (precision=0.43, recall=0.86).
CLAIM_THRESHOLD = 0.20

# ----------------------------------------------------------------------
# Caching helpers
# ----------------------------------------------------------------------
@st.cache_resource
def load_classifier(model_path: str):
    """Handles both dict format {model, threshold} and bare model."""
    raw = joblib.load(model_path)
    if isinstance(raw, dict) and 'model' in raw:
        return raw['model'], raw.get('threshold', CLAIM_THRESHOLD)
    return raw, CLAIM_THRESHOLD

@st.cache_resource
def load_regressor(model_path: str):
    raw = joblib.load(model_path)
    if isinstance(raw, dict) and 'model' in raw:
        return raw['model']
    return raw

@st.cache_data
def load_training_columns() -> list:
    if not os.path.exists(FEATURE_NAMES_PATH):
        st.error(f"Feature names file not found: {FEATURE_NAMES_PATH}")
        return []
    with open(FEATURE_NAMES_PATH, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# ----------------------------------------------------------------------
# Category maps (UI label -> exact training data string)
# ----------------------------------------------------------------------
EDUCATION_MAP = {
    "<High School": "<High School",
    "High School":  "z_High School",
    "Bachelors":    "Bachelors",
    "Masters":      "Masters",
    "PhD":          "PhD",
}
OCCUPATION_MAP = {
    "Blue Collar":  "z_Blue Collar",
    "Clerical":     "Clerical",
    "Manager":      "Manager",
    "Professional": "Professional",
    "Doctor":       "Doctor",
    "Lawyer":       "Lawyer",
    "Home Maker":   "Home Maker",
    "Student":      "Student",
}
MSTATUS_MAP  = {"Married": "Yes", "Not Married": "z_No"}
CAR_TYPE_MAP = {
    "Minivan":     "Minivan",
    "Van":         "Van",
    "SUV":         "z_SUV",
    "Sports Car":  "Sports Car",
    "Panel Truck": "Panel Truck",
    "Pickup":      "Pickup",
}

# ----------------------------------------------------------------------
# Feature builder
# ----------------------------------------------------------------------
def build_feature_row(ui: dict, training_columns: list) -> pd.DataFrame:
    """
    Build a single-row DataFrame aligned exactly to the 46 training columns.
    Accident-level fields (severity, dashcam, fault%, policy tier) are
    intentionally excluded here — they become post-prediction multipliers.
    """
    base = pd.DataFrame([{
        # Numeric features — real values, zero scaling
        "KIDSDRIV": int(ui.get("kids_driving", 0)),
        "AGE":      int(ui["age"]),
        "HOMEKIDS": int(ui.get("home_kids", 0)),
        "YOJ":      int(ui.get("years_on_job", 5)),
        "INCOME":   int(ui["income"]),
        "HOME_VAL": int(ui.get("home_value", 0)),
        "TRAVTIME": int(ui.get("commute_minutes", 30)),
        "BLUEBOOK": int(ui.get("car_bluebook", 15000)),
        "TIF":      int(ui["customer_tenure"]),
        "OLDCLAIM": int(ui.get("prior_claim_amt", 0)),
        "CLM_FREQ": int(ui.get("prior_claims", 0)),
        "MVR_PTS":  int(ui.get("mvr_pts", 0)),
        "CAR_AGE":  int(ui["car_age"]),
        # Categorical features — must match exact training strings
        "PARENT1":   ui.get("is_single_parent", "No"),
        "MSTATUS":   MSTATUS_MAP.get(ui.get("marital_status", "Not Married"), "z_No"),
        "GENDER":    "M" if ui.get("gender", "Male") == "Male" else "z_F",
        "EDUCATION": EDUCATION_MAP.get(ui.get("education", "High School"), "z_High School"),
        "OCCUPATION":OCCUPATION_MAP.get(ui.get("occupation", "Blue Collar"), "z_Blue Collar"),
        "CAR_USE":   ui.get("car_use", "Private"),
        "CAR_TYPE":  CAR_TYPE_MAP.get(ui["car_type"], "Minivan"),
        "RED_CAR":   "no",
        "REVOKED":   ui.get("license_revoked", "No"),
        "URBANICITY":"Highly Urban/ Urban"
                     if ui.get("is_urban", "Urban") == "Urban"
                     else "z_Highly Rural/ Rural",
    }])

    cat_cols = base.select_dtypes(include=["object", "str"]).columns.tolist()
    df_enc   = pd.get_dummies(base, columns=cat_cols, prefix_sep="_") if cat_cols else base.copy()
    return df_enc.reindex(columns=training_columns, fill_value=0).astype(float)

# ----------------------------------------------------------------------
# Post-prediction multipliers
# ----------------------------------------------------------------------
SEVERITY_MULT = {
    "Minor":      0.70,
    "Moderate":   1.00,
    "Major":      1.30,
    "Total Loss": 1.60,
}
TIER_MULT = {"Basic": 1.00, "Gold": 1.10, "Platinum": 1.20}


def apply_modifiers(raw_amount: float, ui: dict) -> Tuple[float, list]:
    """
    Apply percentage-based business rules on top of the raw ML prediction.
    Returns (adjusted_amount, breakdown_list).
    """
    rules = [
        ("Accident Severity", SEVERITY_MULT.get(ui.get("accident_severity", "Moderate"), 1.0)),
        ("Dashcam Discount",  0.90 if ui.get("has_dashcam") == "Yes" else 1.0),
        ("Policy Tier",       TIER_MULT.get(ui.get("policy_tier", "Basic"), 1.0)),
        ("Your Fault %",      int(ui.get("fault_percentage", 100)) / 100.0),
    ]

    amount    = raw_amount
    breakdown = []
    for name, mult in rules:
        delta = amount * (mult - 1.0)
        breakdown.append({
            "Modifier": name,
            "Effect":   f"{(mult-1)*100:+.0f}%",
            "Δ Amount": f"{'+'if delta>=0 else ''}${delta:,.2f}",
        })
        amount *= mult

    return max(0.0, round(amount, 2)), breakdown


# ======================================================================
# PAGE LAYOUT
# ======================================================================
st.set_page_config(page_title="Insurance Claim Calculator", page_icon="🛡️", layout="wide")
st.title("🛡️ Insurance Claim Calculator")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Status")
    try:
        clf, _           = load_classifier(CLASSIFIER_PATH)
        reg              = load_regressor(REGRESSOR_PATH)
        training_columns = load_training_columns()
        st.success("Models loaded ✓")
        st.caption(f"Features: {len(training_columns)}")
        st.caption(f"Claim threshold: {CLAIM_THRESHOLD*100:.0f}%")
    except Exception as exc:
        st.error(f"Model load failed:\n{exc}")
        st.stop()

    st.markdown("---")
    st.markdown("""
**How it works**

1. Your profile is scored by the ML model (0–100%).
2. Score ≥ 20% → claim amount is predicted.
3. Accident factors adjust the amount by %.
4. Policy rules produce the final payout.
""")

# ======================================================================
# SECTION 1 — Driver
# ======================================================================
st.subheader("👤 Driver Information")
c1, c2, c3 = st.columns(3)

with c1:
    age              = st.number_input("Age", 16, 85, 35)
    gender           = st.selectbox("Gender", ["Male", "Female"])
    marital_status   = st.selectbox("Marital Status", ["Married", "Not Married"])
    is_single_parent = st.radio("Single Parent?", ["No", "Yes"], horizontal=True)

with c2:
    education  = st.selectbox("Education",
                               ["<High School", "High School", "Bachelors", "Masters", "PhD"], index=2)
    occupation = st.selectbox("Occupation",
                               ["Blue Collar", "Clerical", "Manager", "Professional",
                                "Doctor", "Lawyer", "Home Maker", "Student"], index=3)
    income     = st.number_input("Annual Income ($)", 0, 1_000_000, 55_000, step=1_000)
    home_value = st.number_input("Home Value ($)", 0, 2_000_000, 150_000, step=5_000)

with c3:
    mvr_pts         = st.number_input("MVR Points", 0, 13, 0,
                                       help="Motor vehicle record points. 0 = clean, 13 = max")
    license_revoked = st.radio("License Ever Revoked?", ["No", "Yes"], horizontal=True)
    is_urban        = st.radio("Area", ["Urban", "Rural"], horizontal=True)
    commute_minutes = st.number_input("Daily Commute (min)", 0, 200, 30)

st.markdown("---")

# ======================================================================
# SECTION 2 — Vehicle
# ======================================================================
st.subheader("🚗 Vehicle Information")
c4, c5, c6 = st.columns(3)

with c4:
    car_type     = st.selectbox("Car Type",
                                 ["Minivan", "Van", "SUV", "Sports Car", "Panel Truck", "Pickup"])
    car_age      = st.number_input("Car Age (years)", 0, 30, 5)
    car_bluebook = st.number_input("Car Book Value ($)", 0, 200_000, 15_000, step=500)

with c5:
    car_use      = st.selectbox("Car Use", ["Private", "Commercial"])
    kids_driving = st.number_input("Kids Who Drive This Car", 0, 4, 0)
    home_kids    = st.number_input("Kids at Home", 0, 5, 0)

with c6:
    years_on_job    = st.number_input("Years at Current Job", 0, 40, 5)
    customer_tenure = st.number_input("Years With This Insurer", 0, 50, 3)

st.markdown("---")

# ======================================================================
# SECTION 3 — Claim History
# ======================================================================
st.subheader("📋 Prior Claim History")
st.caption("These are the model's strongest predictors. 0 prior claims = lower probability score.")

c7, c8 = st.columns(2)

with c7:
    prior_claims    = st.number_input("Prior Claims (last 5 years)", 0, 5, 0)
    prior_claim_amt = st.number_input("Total Prior Claim Amount ($)", 0, 500_000, 0, step=500)

with c8:
    st.info("💡 Increase prior claims or MVR points to see the model predict non-zero amounts.")

st.markdown("---")

# ======================================================================
# SECTION 4 — This Accident (modifiers only — NOT fed to ML)
# ======================================================================
st.subheader("🚨 This Accident — Payout Adjustments")
st.caption("These are applied as **% multipliers on top of the ML prediction**. They do not affect the model score.")

c9, c10, c11, c12 = st.columns(4)

with c9:
    accident_severity = st.selectbox("Accident Severity",
                                      ["Minor", "Moderate", "Major", "Total Loss"], index=1,
                                      help="Minor −30% | Moderate ±0% | Major +30% | Total Loss +60%")
with c10:
    fault_percentage = st.slider("Your Fault %", 0, 100, 50,
                                  help="0% = other party's fault entirely → $0 payout")
with c11:
    has_dashcam = st.radio("Dashcam Available?", ["No", "Yes"], horizontal=True,
                            help="Dashcam evidence → −10%")
with c12:
    policy_tier = st.selectbox("Policy Tier", ["Basic", "Gold", "Platinum"],
                                help="Gold +10% | Platinum +20%")

st.markdown("---")

# ======================================================================
# PREDICT BUTTON
# ======================================================================
if st.button("🔍 Calculate Claim", type="primary", use_container_width=True):
    try:
        ui = {
            "age": age, "gender": gender, "marital_status": marital_status,
            "is_single_parent": is_single_parent, "education": education,
            "occupation": occupation, "income": income, "home_value": home_value,
            "mvr_pts": mvr_pts, "license_revoked": license_revoked,
            "is_urban": is_urban, "commute_minutes": commute_minutes,
            "car_type": car_type, "car_age": car_age, "car_bluebook": car_bluebook,
            "car_use": car_use, "kids_driving": kids_driving, "home_kids": home_kids,
            "years_on_job": years_on_job, "customer_tenure": customer_tenure,
            "prior_claims": prior_claims, "prior_claim_amt": prior_claim_amt,
            # accident modifiers (not sent to ML model)
            "accident_severity": accident_severity, "fault_percentage": fault_percentage,
            "has_dashcam": has_dashcam, "policy_tier": policy_tier,
        }

        # Step 1 — build feature row
        input_df = build_feature_row(ui, training_columns)

        # Step 2 — classifier
        proba = float(clf.predict_proba(input_df)[0][1])
        flag  = int(proba >= CLAIM_THRESHOLD)

        # Step 3 — regressor (only if flagged)
        raw_amount = 0.0
        if flag == 1:
            raw_amount = max(0.0, float(np.expm1(reg.predict(input_df)[0])))

        # Step 4 — accident modifiers
        adjusted_amount, mod_breakdown = apply_modifiers(raw_amount, ui)

        # Step 5 — decision engine
        decision_input = {
            "DUI":                  False,
            "valid_license":        license_revoked == "No",
            "fraud_indicator":      False,
            "policy_expired":       False,
            "authorized_driver":    True,
            "illegal_activity":     False,
            "commercial_use":       car_use == "Commercial",
            "commercial_coverage":  True,
            "street_racing":        False,
            "roadworthy":           True,
            "geographic_exclusion": False,
            "fault_percentage":     fault_percentage,
            "speeding_penalty":     mvr_pts * 5,
            "distracted_driving":   False,
            "dashcam":              has_dashcam == "Yes",
            "failure_to_mitigate":  False,
            "preexisting_damage_pct": 0,
            "depreciation_pct":     min(car_age * 2, 40),
            "salvage_value":        0,
            "oem_parts":            True,
            "policy_tier":          policy_tier,
            "customer_tenure":      customer_tenure,
            "previous_claims":      prior_claims,
            "accident_forgiveness": False,
        }
        result       = calculate_final_payout(adjusted_amount, decision_input)
        is_denied    = result.get("is_denied", False)
        final_payout = result.get("final_payout", adjusted_amount) if not is_denied else 0.0

        # ── Display ──────────────────────────────────────────────────
        st.markdown("## 📊 Results")

        # Probability gauge
        pct         = proba * 100
        gauge_color = "#d32f2f" if proba >= 0.35 else "#f57c00" if proba >= CLAIM_THRESHOLD else "#388e3c"
        bar_width   = min(pct, 100)
        st.markdown(f"""
<div style="background:#f5f5f5;border-radius:8px;padding:16px;margin-bottom:16px">
  <b>Claim Probability Score</b>
  <div style="background:#ddd;border-radius:4px;height:28px;margin:8px 0;overflow:hidden">
    <div style="background:{gauge_color};width:{bar_width:.1f}%;height:100%;border-radius:4px;
                display:flex;align-items:center;padding-left:10px;
                color:white;font-weight:bold;font-size:14px;min-width:50px">
      {pct:.1f}%
    </div>
  </div>
  <small>Threshold = {CLAIM_THRESHOLD*100:.0f}% &nbsp;|&nbsp;
  {"✅ Claim triggered — amount predicted" if flag else "❌ Score below threshold — no claim predicted"}</small>
</div>
""", unsafe_allow_html=True)

        # Three summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("ML Raw Prediction",    f"${raw_amount:,.2f}")
        m2.metric("After Adjustments",    f"${adjusted_amount:,.2f}")
        m3.metric("Final Approved Payout",f"${final_payout:,.2f}")

        # Modifier breakdown (only meaningful if a claim was predicted)
        if raw_amount > 0:
            st.markdown("### 🔧 Adjustment Breakdown")
            st.table(pd.DataFrame(mod_breakdown))
        elif flag == 0:
            # Helpful guidance when score is below threshold
            st.warning(f"""
**Why is the prediction $0?**

This profile scored **{pct:.1f}%** — below the {CLAIM_THRESHOLD*100:.0f}% threshold the model uses to
predict a claim. In the training data, ~27% of policyholders filed a claim, and profiles
like this one have a below-average risk level.

**To see a non-zero prediction, try:**
- Increase "Prior Claims" to 2 or more
- Increase "Total Prior Claim Amount" to $5,000+
- Increase MVR Points to 4+
- Set "Accident Severity" to Major or Total Loss
""")

        # Decision engine result
        st.markdown("### ⚖️ Final Decision")
        if is_denied:
            st.error(f"🚫 **CLAIM DENIED** — {result.get('denial_reason', 'Policy exclusion')}")
        else:
            if result.get("reductions"):
                with st.expander("View Reductions"):
                    for name, amt in result["reductions"].items():
                        st.write(f"• {name}: −${amt:,.2f}")
                    st.write(f"After reductions: ${result.get('reduced_amount', adjusted_amount):,.2f}")
            if result.get("loyalty_benefits"):
                with st.expander("Loyalty Benefits"):
                    for b in result["loyalty_benefits"]:
                        st.write(f"• {b}")
            if final_payout > 0:
                st.success(f"### ✅ Final Approved Payout: ${final_payout:,.2f}")
            else:
                st.info("No payout — claim score below threshold.")

        # Debug
        with st.expander("🔍 Debug — Feature Row Sent to Model"):
            st.caption(f"Shape: {input_df.shape} | Non-zero: {int((input_df != 0).sum().sum())} features")
            nonzero = input_df.loc[:, (input_df != 0).any()].T.rename(columns={0: "value"})
            st.dataframe(nonzero, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.code(traceback.format_exc())

st.markdown("---")
st.caption("Insurance AI V2 – Claim Prediction System")