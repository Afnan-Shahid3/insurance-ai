"""
Insurance Policy Rules Engine
Pure rule-based module — no ML, no dataset dependency.

Pipeline:
    predicted_claim (from ML model)
        → Rule B: Denial check  (any match → payout = 0, stop)
        → Rule A: Car price cap (cap at % of car value)
        → Rule C: Reductions    (multipliers + salvage deduction)
        → final_payout

Usage:
    from src.policy_engine.insurance_rules import (
        PolicyConfig, DenialFlags, ReductionFactors, evaluate_claim
    )

    result = evaluate_claim(
        predicted_claim=45000.0,
        car_price=30000.0,
        denial_flags=DenialFlags(lapse_in_coverage=True),
        reduction_factors=ReductionFactors(),
    )
    # result["denied"] → True
    # result["final_payout"] → 0.0
"""

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# A — CONFIGURATION  (all thresholds in one place, fully overridable)
# =============================================================================

@dataclass
class PolicyConfig:
    """
    Configurable thresholds and penalty rates for the rules engine.
    Override any field to customise behaviour without touching rule logic.
    """
    # Rule A
    car_price_cap_pct: float = 70.0             # max payout as % of car market value

    # Rule C — penalty percentages (0-100)
    overspeeding_penalty_pct: float = 10.0
    distracted_driving_penalty_pct: float = 15.0
    no_dashcam_penalty_pct: float = 5.0
    failure_to_mitigate_penalty_pct: float = 10.0
    non_oem_parts_discount_pct: float = 10.0


# =============================================================================
# B — DENIAL FLAGS  (one per exclusion condition)
# =============================================================================

@dataclass
class DenialFlags:
    """
    Boolean conditions — if ANY is True the claim is fully denied (payout = 0).
    All default to False so callers only set what applies.
    """
    dui_dwi: bool = False
    unlicensed_driver: bool = False
    expired_license: bool = False
    excluded_driver: bool = False
    lapse_in_coverage: bool = False
    intentional_damage_or_fraud: bool = False
    commercial_use_undeclared: bool = False
    racing_or_stunt_driving: bool = False
    illegal_activity: bool = False
    non_roadworthy_vehicle: bool = False
    geographic_exclusion: bool = False


# Human-readable reason for each denial flag (same key names as the dataclass fields)
_DENIAL_LABELS: dict[str, str] = {
    "dui_dwi":
        "DUI/DWI — claim denied under policy exclusion.",
    "unlicensed_driver":
        "Unlicensed driver — not covered under policy.",
    "expired_license":
        "Expired driver's licence — coverage void at time of incident.",
    "excluded_driver":
        "Driver is explicitly excluded on this policy.",
    "lapse_in_coverage":
        "Lapse in coverage at time of incident — policy was not active.",
    "intentional_damage_or_fraud":
        "Intentional damage or fraud detected — claim denied.",
    "commercial_use_undeclared":
        "Undeclared commercial use — policy exclusion applies.",
    "racing_or_stunt_driving":
        "Racing/stunt driving — excluded from standard coverage.",
    "illegal_activity":
        "Illegal activity during incident — claim denied.",
    "non_roadworthy_vehicle":
        "Vehicle was non-roadworthy at time of incident.",
    "geographic_exclusion":
        "Incident occurred in a geographically excluded area.",
}


# =============================================================================
# C — REDUCTION FACTORS
# =============================================================================

@dataclass
class ReductionFactors:
    """
    Inputs for the claim reduction engine.

    Percentage fields (comparative_negligence_pct, depreciation_pct) accept
    values in the range 0-100.  salvage_value is a dollar amount deducted last.
    """
    # Proportional fault — e.g., 30.0 means driver is 30% at fault
    comparative_negligence_pct: float = 0.0

    # Boolean penalty triggers (rates live in PolicyConfig)
    overspeeding: bool = False
    distracted_driving: bool = False
    no_dashcam: bool = False
    failure_to_mitigate: bool = False
    non_oem_parts: bool = False

    # Dollar/percentage adjustments
    depreciation_pct: float = 0.0    # betterment/wear-and-tear deduction (%)
    salvage_value: float = 0.0       # recovered salvage in $ — deducted from payout


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _pct_reduction(
    current: float,
    pct: float,
    rule: str,
    description: str,
    adjustments: list,
) -> float:
    """Reduce *current* by *pct* percent, append an audit entry, return new value."""
    reduction = current * (pct / 100.0)
    after = current - reduction
    adjustments.append({
        "rule": rule,
        "description": description,
        "payout_before": round(current, 2),
        "payout_after": round(after, 2),
    })
    return after


# =============================================================================
# RULE A — CAR PRICE CAP
# =============================================================================

def _apply_car_price_cap(
    claim: float,
    car_price: float,
    cap_pct: float,
    adjustments: list,
) -> float:
    """
    Cap the claim so that final payout cannot exceed cap_pct% of car_price.
    Skipped when car_price <= 0 (value unknown / not applicable).
    """
    if car_price <= 0:
        return claim

    cap_amount = car_price * (cap_pct / 100.0)
    if claim > cap_amount:
        adjustments.append({
            "rule": "car_price_cap",
            "description": (
                f"Claim ${claim:,.2f} exceeds {cap_pct:.0f}% of car value "
                f"(${car_price:,.2f} × {cap_pct:.0f}% = ${cap_amount:,.2f}). "
                f"Capped at ${cap_amount:,.2f}."
            ),
            "payout_before": round(claim, 2),
            "payout_after": round(cap_amount, 2),
        })
        return cap_amount

    return claim


# =============================================================================
# RULE B — DENIAL ENGINE
# =============================================================================

def _check_denial(denial_flags: DenialFlags) -> tuple[bool, Optional[str]]:
    """
    Evaluate every denial condition in declaration order.
    Returns (denied, reason_string).  Stops at the first match.
    """
    for flag_name, label in _DENIAL_LABELS.items():
        if getattr(denial_flags, flag_name, False):
            return True, label
    return False, None


# =============================================================================
# RULE C — REDUCTION ENGINE
# =============================================================================

def _apply_reductions(
    claim: float,
    factors: ReductionFactors,
    config: PolicyConfig,
) -> tuple[float, list]:
    """
    Apply all reduction multipliers and the salvage deduction in a fixed order.
    Returns (reduced_claim, adjustments_list).

    Order matters — each step operates on the running total from the previous:
        1. Comparative negligence  (largest, applied first)
        2. Overspeeding
        3. Distracted driving
        4. No dashcam
        5. Failure to mitigate
        6. Depreciation / betterment
        7. Non-OEM parts discount
        8. Salvage value deduction  (dollar amount, applied last)
    """
    adjustments: list = []

    # 1. Comparative negligence — driver's % fault reduces payout proportionally
    if factors.comparative_negligence_pct > 0:
        pct = min(factors.comparative_negligence_pct, 100.0)
        claim = _pct_reduction(
            claim, pct,
            "comparative_negligence",
            f"Driver is {pct:.1f}% at fault — payout reduced by {pct:.1f}%.",
            adjustments,
        )

    # 2. Overspeeding
    if factors.overspeeding:
        claim = _pct_reduction(
            claim, config.overspeeding_penalty_pct,
            "overspeeding_penalty",
            f"Overspeeding confirmed — {config.overspeeding_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 3. Distracted driving
    if factors.distracted_driving:
        claim = _pct_reduction(
            claim, config.distracted_driving_penalty_pct,
            "distracted_driving_penalty",
            f"Distracted driving — {config.distracted_driving_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 4. No dashcam footage
    if factors.no_dashcam:
        claim = _pct_reduction(
            claim, config.no_dashcam_penalty_pct,
            "no_dashcam_penalty",
            f"No dashcam footage — {config.no_dashcam_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 5. Failure to mitigate loss
    if factors.failure_to_mitigate:
        claim = _pct_reduction(
            claim, config.failure_to_mitigate_penalty_pct,
            "failure_to_mitigate_penalty",
            f"Failure to mitigate loss — {config.failure_to_mitigate_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 6. Depreciation / betterment adjustment
    if factors.depreciation_pct > 0:
        pct = min(factors.depreciation_pct, 100.0)
        claim = _pct_reduction(
            claim, pct,
            "depreciation_betterment",
            f"Depreciation/betterment adjustment — {pct:.1f}% reduction for wear and tear.",
            adjustments,
        )

    # 7. Non-OEM parts discount
    if factors.non_oem_parts:
        claim = _pct_reduction(
            claim, config.non_oem_parts_discount_pct,
            "non_oem_parts_discount",
            f"Non-OEM replacement parts — {config.non_oem_parts_discount_pct:.0f}% discount applied.",
            adjustments,
        )

    # 8. Salvage value deduction (flat dollar amount — applied after all multipliers)
    if factors.salvage_value > 0:
        before = claim
        claim = max(0.0, claim - factors.salvage_value)
        adjustments.append({
            "rule": "salvage_value_deduction",
            "description": (
                f"Salvage value ${factors.salvage_value:,.2f} deducted "
                f"(vehicle retained partial value after total loss)."
            ),
            "payout_before": round(before, 2),
            "payout_after": round(claim, 2),
        })

    return claim, adjustments


# =============================================================================
# PUBLIC API — single entry point
# =============================================================================

def evaluate_claim(
    predicted_claim: float,
    car_price: float,
    denial_flags: DenialFlags,
    reduction_factors: ReductionFactors,
    config: Optional[PolicyConfig] = None,
) -> dict:
    """
    Run the full insurance policy rules engine against a predicted claim amount.

    Execution order (each stage can short-circuit or modify the running total):
        1. Denial check    → if any flag is set: payout = 0, return immediately
        2. Car price cap   → payout cannot exceed config.car_price_cap_pct % of car_price
        3. Reductions      → sequential multipliers + salvage deduction

    Args:
        predicted_claim:   Raw ML model output in dollars (pre-policy adjustment).
        car_price:         Current market value of the insured vehicle in dollars.
                           Pass 0 or a negative value to skip the car price cap.
        denial_flags:      DenialFlags instance — any True field voids the claim.
        reduction_factors: ReductionFactors instance — penalty/deduction inputs.
        config:            Optional PolicyConfig for custom thresholds.
                           Defaults to PolicyConfig() when None.

    Returns:
        dict with keys:
            "final_payout"  (float)         — adjusted payout in dollars
            "denied"        (bool)           — True if claim was voided
            "denial_reason" (str | None)     — human-readable denial reason, or None
            "adjustments"   (list[dict])     — ordered audit trail of every applied rule
                Each adjustment dict contains:
                    "rule"           (str)   — machine-readable rule identifier
                    "description"    (str)   — human-readable explanation
                    "payout_before"  (float) — running total before this rule
                    "payout_after"   (float) — running total after this rule
    """
    if config is None:
        config = PolicyConfig()

    adjustments: list = []

    # ---- RULE B: Denial (highest priority — voids the claim entirely) --------
    denied, denial_reason = _check_denial(denial_flags)
    if denied:
        return {
            "final_payout": 0.0,
            "denied": True,
            "denial_reason": denial_reason,
            "adjustments": [
                {
                    "rule": "claim_denial",
                    "description": denial_reason,
                    "payout_before": round(float(predicted_claim), 2),
                    "payout_after": 0.0,
                }
            ],
        }

    current = float(predicted_claim)

    # ---- RULE A: Car price cap ------------------------------------------------
    current = _apply_car_price_cap(
        current, float(car_price), config.car_price_cap_pct, adjustments
    )

    # ---- RULE C: Reductions ---------------------------------------------------
    current, reduction_adjustments = _apply_reductions(current, reduction_factors, config)
    adjustments.extend(reduction_adjustments)

    return {
        "final_payout": round(max(0.0, current), 2),
        "denied": False,
        "denial_reason": None,
        "adjustments": adjustments,
    }
