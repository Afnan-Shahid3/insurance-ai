"""
Insurance Policy Rules Engine
Pure rule-based module — no ML, no dataset dependency.

Pipeline:
    ml_damage_estimate (from ML model — used directly as base payout)
        → Rule B: Denial check  (any flag → payout = 0, pipeline stops immediately)
        → Dashcam trigger        (no_dashcam=True forces overspeeding +
                                  distracted_driving + failure_to_mitigate on)
        → Rule C: Reductions     (multipliers + salvage deduction)
        → final_payout

Usage:
    from src.policy_engine.insurance_rules import (
        PolicyConfig, DenialFlags, ReductionFactors, evaluate_claim
    )

    result = evaluate_claim(
        ml_damage_estimate=45000.0,
        denial_flags=DenialFlags(lapse_in_coverage=True),
        reduction_factors=ReductionFactors(),
    )
    # result["denied"] → True
    # result["final_payout"] → 0.0
"""

import dataclasses
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
    # Rule C — penalty percentages (0-100); applied multiplicatively to running payout
    overspeeding_penalty_pct:       float = 20.0
    distracted_driving_penalty_pct: float = 15.0
    traffic_violations_penalty_pct: float = 15.0
    no_dashcam_penalty_pct:         float = 15.0
    failure_to_mitigate_penalty_pct: float = 5.0
    poor_maintenance_penalty_pct:   float = 15.0
    unauthorized_repair_penalty_pct: float = 5.0


# =============================================================================
# B — DENIAL FLAGS  (one per exclusion condition)
# =============================================================================

@dataclass
class DenialFlags:
    """
    Denial conditions — if ANY field is True the claim is fully denied (payout = 0).
    All default to False; callers set only the flags that apply.

    Fields map 1-to-1 with _DENIAL_LABELS so the denial engine needs no custom logic.
    """
    dui_dwi:                    bool = False   # driving under influence of alcohol/drugs
    no_valid_license:           bool = False   # unlicensed driver OR expired licence
    fraud_or_staged_accident:   bool = False   # fraudulent claim or deliberately staged event
    commercial_use_not_covered: bool = False   # vehicle used commercially without declared coverage
    racing_or_illegal_driving:  bool = False   # racing, stunt driving, or other illegal road use
    geographic_exclusion:       bool = False   # incident occurred outside covered territory
    lapse_in_coverage:          bool = False   # policy was not active at time of incident
    intentional_damage:         bool = False   # owner intentionally caused the damage


# Denial reason labels — keys must match DenialFlags field names exactly
_DENIAL_LABELS: dict[str, str] = {
    "dui_dwi":
        "DUI/DWI — claim denied under policy exclusion.",
    "no_valid_license":
        "No valid licence — driver was unlicensed or licence had expired.",
    "fraud_or_staged_accident":
        "Fraud or staged accident detected — claim denied.",
    "commercial_use_not_covered":
        "Commercial use without declared coverage — policy exclusion applies.",
    "racing_or_illegal_driving":
        "Racing, stunt, or illegal driving — excluded from standard coverage.",
    "geographic_exclusion":
        "Incident occurred in a geographically excluded area.",
    "lapse_in_coverage":
        "Lapse in coverage at time of incident — policy was not active.",
    "intentional_damage":
        "Intentional self-inflicted damage — claim denied.",
}


# =============================================================================
# C — REDUCTION FACTORS
# =============================================================================

@dataclass
class ReductionFactors:
    """
    Inputs for the claim reduction engine.

    comparative_negligence_pct accepts values 0-100.
    salvage_value is a flat dollar amount deducted last.
    """
    # Proportional fault — e.g., 30.0 means driver is 30% at fault
    comparative_negligence_pct: float = 0.0

    # Boolean penalty triggers (rates live in PolicyConfig)
    overspeeding:        bool = False
    distracted_driving:  bool = False
    traffic_violations:  bool = False
    no_dashcam:          bool = False
    failure_to_mitigate: bool = False
    poor_maintenance:    bool = False
    unauthorized_repair: bool = False

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
# DEPRECIATION
# =============================================================================

def apply_depreciation(amount: float, age_years: int) -> float:
    """
    Apply compound depreciation at 5% per year.

    Formula: final_amount = amount * (0.95 ** age_years)

    Each year reduces the remaining value by 5% of that year's value,
    NOT 5% of the original — making this compound, not linear.

    Args:
        amount:     The base amount in dollars to depreciate.
        age_years:  Vehicle age in years. Negative values are treated as 0.

    Returns:
        Depreciated amount rounded to 2 decimal places.

    Examples:
        apply_depreciation(10000, 0)  → 10000.00  (no depreciation)
        apply_depreciation(10000, 1)  →  9500.00  (5% off)
        apply_depreciation(10000, 2)  →  9025.00  (5% off 9500, not 5% off 10000 twice)
        apply_depreciation(10000, 10) →  5987.37
    """
    age_years = max(0, age_years)
    return round(amount * (0.95 ** age_years), 2)


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
# DASHCAM TRIGGER
# =============================================================================

def _apply_dashcam_trigger(
    claim: float,
    factors: ReductionFactors,
    adjustments: list,
) -> ReductionFactors:
    """
    Trigger system — NOT a reduction.

    When dashcam evidence is unavailable (factors.no_dashcam is True), force-activate:
        - overspeeding
        - distracted_driving
        - failure_to_mitigate

    These flags are then applied as normal percentage reductions by Rule C.
    The missing-evidence penalty (no_dashcam_penalty_pct) is also applied by Rule C
    via the existing no_dashcam flag — this function does not touch it.

    Only newly activated flags are listed in the audit entry.
    If all three were already set by the caller, the trigger is still recorded.
    Returns factors unchanged when no_dashcam is False.
    """
    if not factors.no_dashcam:
        return factors

    newly_activated = [
        label
        for flag, label in (
            (factors.overspeeding,        "overspeeding"),
            (factors.distracted_driving,  "distracted driving"),
            (factors.failure_to_mitigate, "failure to mitigate"),
        )
        if not flag
    ]

    description = (
        f"No dashcam evidence — activating: {', '.join(newly_activated)}."
        if newly_activated
        else "No dashcam evidence — all associated penalties already active."
    )

    adjustments.append({
        "rule":          "dashcam_trigger",
        "description":   description,
        "payout_before": round(claim, 2),
        "payout_after":  round(claim, 2),   # trigger only; payout unchanged here
    })

    return dataclasses.replace(
        factors,
        overspeeding=True,
        distracted_driving=True,
        failure_to_mitigate=True,
    )


# =============================================================================
# RULE C — REDUCTION ENGINE
# =============================================================================

def _apply_reductions(
    claim: float,
    factors: ReductionFactors,
    config: PolicyConfig,
) -> tuple[float, list]:
    """
    Apply all reductions multiplicatively to the running payout total.
    Each step operates on the value produced by the previous step.
    Returns (reduced_claim, adjustments_list).

    IMPORTANT: claim must be the post-base-payout value — never the ML estimate.

    Order:
        0. Comparative negligence   (proportional fault — applied first, largest impact)
        1. Overspeeding             20%
        2. Distracted driving       15%
        3. Traffic violations       15%
        4. Missing dashcam          15%
        5. Failure to mitigate       5%
        6. Poor maintenance         15%
        7. Unauthorized repair       5%
        8. Salvage deduction        (flat $ amount — applied last)
    """
    adjustments: list = []

    # 0. Comparative negligence — proportional fault, applied before flat-rate penalties
    if factors.comparative_negligence_pct > 0:
        pct = min(factors.comparative_negligence_pct, 100.0)
        claim = _pct_reduction(
            claim, pct,
            "comparative_negligence",
            f"Driver is {pct:.1f}% at fault — payout reduced by {pct:.1f}%.",
            adjustments,
        )

    # 1. Overspeeding — 20%
    if factors.overspeeding:
        claim = _pct_reduction(
            claim, config.overspeeding_penalty_pct,
            "overspeeding_penalty",
            f"Overspeeding — {config.overspeeding_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 2. Distracted driving — 15%
    if factors.distracted_driving:
        claim = _pct_reduction(
            claim, config.distracted_driving_penalty_pct,
            "distracted_driving_penalty",
            f"Distracted driving — {config.distracted_driving_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 3. Traffic violations — 15%
    if factors.traffic_violations:
        claim = _pct_reduction(
            claim, config.traffic_violations_penalty_pct,
            "traffic_violations_penalty",
            f"Traffic violations on record — {config.traffic_violations_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 4. Missing dashcam evidence — 15%
    if factors.no_dashcam:
        claim = _pct_reduction(
            claim, config.no_dashcam_penalty_pct,
            "no_dashcam_penalty",
            f"No dashcam evidence — {config.no_dashcam_penalty_pct:.0f}% missing-evidence penalty applied.",
            adjustments,
        )

    # 5. Failure to mitigate loss — 5%
    if factors.failure_to_mitigate:
        claim = _pct_reduction(
            claim, config.failure_to_mitigate_penalty_pct,
            "failure_to_mitigate_penalty",
            f"Failure to mitigate loss — {config.failure_to_mitigate_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 6. Poor vehicle maintenance — 15%
    if factors.poor_maintenance:
        claim = _pct_reduction(
            claim, config.poor_maintenance_penalty_pct,
            "poor_maintenance_penalty",
            f"Poor vehicle maintenance — {config.poor_maintenance_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 7. Unauthorized repair — 5%
    if factors.unauthorized_repair:
        claim = _pct_reduction(
            claim, config.unauthorized_repair_penalty_pct,
            "unauthorized_repair_penalty",
            f"Unauthorized repair work — {config.unauthorized_repair_penalty_pct:.0f}% penalty applied.",
            adjustments,
        )

    # 8. Salvage value deduction — flat $ amount, applied after all multipliers
    if factors.salvage_value > 0:
        before = claim
        claim = max(0.0, claim - factors.salvage_value)
        adjustments.append({
            "rule":          "salvage_value_deduction",
            "description":   f"Salvage value ${factors.salvage_value:,.2f} deducted.",
            "payout_before": round(before, 2),
            "payout_after":  round(claim, 2),
        })

    return claim, adjustments


# =============================================================================
# PUBLIC API — single entry point
# =============================================================================

def evaluate_claim(
    ml_damage_estimate: float,
    denial_flags: DenialFlags,
    reduction_factors: ReductionFactors,
    config: Optional[PolicyConfig] = None,
) -> dict:
    """
    Run the insurance policy rules engine.

    ml_damage_estimate is the ML model output and is used directly as the base
    payout. Denial rules are checked first; if the claim is not denied, reduction
    rules are applied sequentially to produce the final payout.

    Execution order:
        B. Denial check → ANY flag True → payout = 0, pipeline stops immediately
        C. Reductions   → sequential multipliers + salvage deduction

    Args:
        ml_damage_estimate: ML model output in dollars — used as base payout.
        denial_flags:       DenialFlags instance — any True field voids the claim.
        reduction_factors:  ReductionFactors instance — penalty/deduction inputs.
        config:             Optional PolicyConfig for custom thresholds.

    Returns:
        dict with keys:
            "ml_damage_estimate" (float)      — raw ML output / base payout
            "final_payout"       (float)      — adjusted payout in dollars
            "denied"             (bool)       — True if claim was voided
            "denial_reason"      (str | None) — human-readable denial reason
            "adjustments"        (list[dict]) — ordered audit trail
    """
    if config is None:
        config = PolicyConfig()

    # ---- RULE B: Denial — checked first, stops pipeline immediately ----------
    denied, denial_reason = _check_denial(denial_flags)
    if denied:
        return {
            "ml_damage_estimate": round(float(ml_damage_estimate), 2),
            "final_payout":       0.0,
            "denied":             True,
            "denial_reason":      denial_reason,
            "adjustments": [
                {
                    "rule":          "claim_denial",
                    "description":   denial_reason,
                    "payout_before": round(float(ml_damage_estimate), 2),
                    "payout_after":  0.0,
                }
            ],
        }

    adjustments: list = []
    current = float(ml_damage_estimate)

    # ---- DASHCAM TRIGGER: force-activate penalties before reductions ----------
    reduction_factors = _apply_dashcam_trigger(current, reduction_factors, adjustments)

    # ---- RULE C: Reductions --------------------------------------------------
    current, reduction_adjustments = _apply_reductions(current, reduction_factors, config)
    adjustments.extend(reduction_adjustments)

    return {
        "ml_damage_estimate": round(float(ml_damage_estimate), 2),
        "final_payout":       round(max(0.0, current), 2),
        "denied":             False,
        "denial_reason":      None,
        "adjustments":        adjustments,
    }