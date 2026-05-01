"""
Decision Engine for Insurance Claim Processing
Transforms ML predictions into business decisions
"""

import numpy as np


def check_denial_conditions(input_data):
    """
    Check if claim should be denied based on hard rules.
    
    Parameters:
    -----------
    input_data : dict
        Dictionary containing claim information
        
    Returns:
    --------
    is_denied : bool
        True if claim should be denied
    reason : str
        Reason for denial if denied, else empty string
    """
    denial_reasons = []
    
    # Check for DUI / intoxicated driving
    if input_data.get('DUI', False):
        denial_reasons.append("DUI / intoxicated driving")
    
    # Check for no valid license
    if not input_data.get('valid_license', True):
        denial_reasons.append("No valid driver's license")
    
    # Check for fraud indicators
    if input_data.get('fraud_indicator', False):
        denial_reasons.append("Suspected fraud / staged accident")
    
    # Check if policy expired
    if input_data.get('policy_expired', False):
        denial_reasons.append("Policy expired / lapsed")
    
    # Check for unauthorized driver
    if not input_data.get('authorized_driver', True):
        denial_reasons.append("Unauthorized driver")
    
    # Check for illegal activity
    if input_data.get('illegal_activity', False):
        denial_reasons.append("Illegal activity during accident")
    
    # Check for commercial use without coverage
    if input_data.get('commercial_use', False) and not input_data.get('commercial_coverage', True):
        denial_reasons.append("Commercial use without proper coverage")
    
    # Check for street racing
    if input_data.get('street_racing', False):
        denial_reasons.append("Street racing / stunt driving")
    
    # Check for non-roadworthy vehicle
    if not input_data.get('roadworthy', True):
        denial_reasons.append("Non-roadworthy vehicle")
    
    # Check geographic exclusion
    if input_data.get('geographic_exclusion', False):
        denial_reasons.append("Geographic exclusion breach")
    
    is_denied = len(denial_reasons) > 0
    reason = "; ".join(denial_reasons) if denial_reasons else ""
    
    return is_denied, reason


def apply_reductions(base_amount, input_data):
    """
    Apply percentage penalties/reductions to predicted claim amount.
    
    Parameters:
    -----------
    base_amount : float
        Original predicted claim amount
    input_data : dict
        Dictionary containing claim information
        
    Returns:
    --------
    adjusted_amount : float
        Amount after reductions
    breakdown : dict
        Dictionary showing each reduction applied
    """
    breakdown = {}
    adjusted = base_amount
    
    # Comparative negligence (fault percentage)
    fault_pct = input_data.get('fault_percentage', 0)
    if fault_pct > 0:
        reduction = base_amount * (fault_pct / 100)
        adjusted -= reduction
        breakdown['Comparative Negligence'] = reduction
    
    # Overspeeding penalty
    speeding_pct = input_data.get('speeding_penalty', 0)
    if speeding_pct > 0:
        reduction = adjusted * (speeding_pct / 100)
        adjusted -= reduction
        breakdown['Overspeeding Penalty'] = reduction
    
    # Distracted driving penalty
    if input_data.get('distracted_driving', False):
        reduction = adjusted * 0.15  # 15% penalty
        adjusted -= reduction
        breakdown['Distracted Driving'] = reduction
    
    # No dashcam penalty
    if not input_data.get('dashcam', False):
        reduction = adjusted * 0.05  # 5% penalty
        adjusted -= reduction
        breakdown['No Dashcam Evidence'] = reduction
    
    # Failure to mitigate damage
    if input_data.get('failure_to_mitigate', False):
        reduction = adjusted * 0.10
        adjusted -= reduction
        breakdown['Failure to Mitigate'] = reduction
    
    # Pre-existing damage
    preexisting_pct = input_data.get('preexisting_damage_pct', 0)
    if preexisting_pct > 0:
        reduction = base_amount * (preexisting_pct / 100)
        adjusted -= reduction
        breakdown['Pre-existing Damage'] = reduction
    
    # Depreciation (betterment rule)
    depreciation_pct = input_data.get('depreciation_pct', 0)
    if depreciation_pct > 0:
        reduction = base_amount * (depreciation_pct / 100)
        adjusted -= reduction
        breakdown['Depreciation (Betterment)'] = reduction
    
    # Salvage value deduction
    if input_data.get('salvage_value', 0) > 0:
        reduction = min(input_data.get('salvage_value', 0), adjusted * 0.20)
        adjusted -= reduction
        breakdown['Salvage Value'] = reduction
    
    # Non-OEM parts adjustment
    if not input_data.get('oem_parts', True):
        reduction = adjusted * 0.10
        adjusted -= reduction
        breakdown['Non-OEM Parts'] = reduction
    
    # Ensure adjusted amount is not negative
    adjusted = max(0, adjusted)
    
    return adjusted, breakdown


def apply_loyalty_adjustments(amount, input_data):
    """
    Apply loyalty/CRM adjustments to the claim amount.
    
    Parameters:
    -----------
    amount : float
        Current claim amount after reductions
    input_data : dict
        Dictionary containing customer information
        
    Returns:
    --------
    final_amount : float
        Amount after loyalty adjustments
    benefits : list
        List of loyalty benefits applied
    """
    benefits = []
    final_amount = amount
    
    # Policy tier adjustments (reduce penalties)
    tier = input_data.get('policy_tier', 'Basic')
    if tier == 'Gold':
        # Gold tier: 10% reduction in penalties (effectively increase payout)
        final_amount = final_amount * 1.10
        benefits.append("Gold Tier: 10% loyalty bonus")
    elif tier == 'Platinum':
        # Platinum tier: 20% reduction in penalties
        final_amount = final_amount * 1.20
        benefits.append("Platinum Tier: 20% loyalty bonus")
    
    # Customer tenure (years of membership)
    tenure = input_data.get('customer_tenure', 0)
    if tenure >= 10:
        final_amount = final_amount * 1.15
        benefits.append(f"Loyal Customer ({tenure} years): 15% bonus")
    elif tenure >= 5:
        final_amount = final_amount * 1.10
        benefits.append(f"Loyal Customer ({tenure} years): 10% bonus")
    elif tenure >= 3:
        final_amount = final_amount * 1.05
        benefits.append(f"Customer ({tenure} years): 5% bonus")
    
    # Claim history - first claim forgiveness
    claim_count = input_data.get('previous_claims', 0)
    if claim_count == 0:
        final_amount = final_amount * 1.05
        benefits.append("First Claim: 5% forgiveness bonus")
    
    # Accident forgiveness
    if input_data.get('accident_forgiveness', False):
        final_amount = final_amount * 1.10
        benefits.append("Accident Forgiveness: 10% bonus")
    
    return final_amount, benefits


def calculate_final_payout(predicted_cost, input_data):
    """
    Calculate final payout by running through the complete decision pipeline.
    
    Parameters:
    -----------
    predicted_cost : float
        ML model predicted claim cost
    input_data : dict
        Dictionary containing all claim and customer information
        
    Returns:
    --------
    dict
        Dictionary containing:
        - is_denied: bool
        - denial_reason: str
        - base_amount: float
        - reductions: dict
        - reduced_amount: float
        - loyalty_benefits: list
        - final_payout: float
        - explanation: str
    """
    result = {
        'is_denied': False,
        'denial_reason': '',
        'base_amount': predicted_cost,
        'reductions': {},
        'reduced_amount': 0,
        'loyalty_benefits': [],
        'final_payout': 0,
        'explanation': ''
    }
    
    # Step 1: Check denial conditions
    is_denied, denial_reason = check_denial_conditions(input_data)
    result['is_denied'] = is_denied
    result['denial_reason'] = denial_reason
    
    if is_denied:
        result['final_payout'] = 0
        result['explanation'] = f"CLAIM DENIED: {denial_reason}"
        return result
    
    # Step 2: Apply reductions
    reduced_amount, reductions = apply_reductions(predicted_cost, input_data)
    result['reductions'] = reductions
    result['reduced_amount'] = reduced_amount
    
    # Step 3: Apply loyalty adjustments
    final_payout, loyalty_benefits = apply_loyalty_adjustments(reduced_amount, input_data)
    result['loyalty_benefits'] = loyalty_benefits
    result['final_payout'] = final_payout
    
    # Step 4: Generate explanation
    explanation_parts = []
    
    if reductions:
        reduction_total = predicted_cost - reduced_amount
        explanation_parts.append(
            f"Claim reduced by ${reduction_total:,.2f} due to: "
            + ", ".join(reductions.keys())
        )
    
    if loyalty_benefits:
        explanation_parts.append(
            "Loyalty benefits applied: " + "; ".join(loyalty_benefits)
        )
    
    result['explanation'] = ". ".join(explanation_parts) if explanation_parts else "Claim approved with standard processing."
    
    return result
